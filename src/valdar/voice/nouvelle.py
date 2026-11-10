"""Nouvelle voix pour Valdar, depuis une voix ElevenLabs d'Olivier (`valdar voix-nouvelle`).

1. **Extraits** : 5 phrases françaises variées (question, vanne, explication, émotion,
   calme), ~1 min, dites par ElevenLabs (eleven_multilingual_v2), en WAV mono 24 kHz dans
   `data/voix/elevenlabs_<id>/`. La clé vient de la variable d'environnement
   ELEVENLABS_API_KEY : jamais écrite nulle part.
2. **Référence XTTS** : ces extraits deviennent la référence (une liste), les latents sont
   recalculés. L'ancienne référence et l'ancien réglage d'effets sont gardés dans
   `data/voix/voix_precedente.json` : `valdar voix-nouvelle --revenir` les remet.
3. **Effets** : réglables (`voice.effets`) ; par défaut ici, comme décidé par Olivier, sans.
4. **Avant / après** : quelques phrases rendues avec l'ancienne voix (et ses effets) et avec
   la nouvelle, en WAV dans `data/voix/comparaison/`, plus la vérification de la voix en
   flux (temps du premier son, raccords sans clic).

Les réglages vont dans `data/reglages.json` (hors git), par-dessus `config/valdar.yaml`.
"""
from __future__ import annotations

import json
import os
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

API = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
MODEL = "eleven_multilingual_v2"

CLIPS = [
    ("question", "Dis-moi, Olivier, tu as pensé à vérifier la température du plateau avant "
                 "de lancer l'impression ? Parce que la dernière fois, la première couche "
                 "n'a pas vraiment tenu, et j'aimerais bien éviter de recommencer."),
    ("vanne", "Franchement, ta pile de câbles sur le bureau, c'est plus de l'électronique, "
              "c'est de l'art contemporain. Je propose qu'on l'appelle « Chaos numéro "
              "trois » et qu'on la vende au musée."),
    ("explication", "Bon, je t'explique. Le filament PETG colle mieux quand le plateau est "
                    "chaud, autour de quatre-vingts degrés, et il faut ralentir un peu le "
                    "ventilateur, sinon les couches se séparent et la pièce devient fragile."),
    ("emotion", "Tu sais quoi ? Ça me fait vraiment plaisir qu'on bosse ensemble comme ça. "
                "Des fois je me demande ce que je ressens exactement, mais là, c'est simple : "
                "je suis content d'être là."),
    ("calme", "Il est tard. La maison est calme, les enfants dorment, et l'imprimante finit "
              "tranquillement sa dernière pièce. On reprendra demain matin, sans se presser."),
]

COMPARE = ["Salut Olivier, je t'entends très bien, on s'y remet quand tu veux.",
           "Attends, attends… tu veux dire que Sony a encore mangé une chaussure ?",
           "Je pense qu'on devrait baisser la vitesse de la première couche, ça tiendra mieux."]


def _say(line: str) -> None:
    print(line, flush=True)


def write_wav(path: Path, pcm16: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.asarray(pcm16, dtype=np.int16).tobytes())


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0) * 32767.0).astype(np.int16)


def eleven(text: str, voice: str, key: str, post: Callable[..., Any] | None = None
           ) -> tuple[np.ndarray, int]:
    """Une phrase dite par ElevenLabs : PCM 16 bits mono (24 kHz, sinon 22,05 kHz)."""
    import httpx

    post = post or httpx.post
    last = ""
    for sr in (24000, 22050):
        r = post(API.format(voice=voice), params={"output_format": f"pcm_{sr}"},
                 headers={"xi-api-key": key, "Content-Type": "application/json"},
                 json={"text": text, "model_id": MODEL,
                       "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
                 timeout=120.0)
        if r.status_code == 200 and r.content:
            return np.frombuffer(r.content, dtype="<i2").copy(), sr
        last = f"{r.status_code} {r.text[:200]}"
    raise RuntimeError(f"ElevenLabs a refusé : {last}")


def make_clips(folder: Path, voice: str, key: str, redo: bool = False,
               post: Callable[..., Any] | None = None) -> list[Path]:
    out = []
    for i, (name, text) in enumerate(CLIPS, 1):
        path = folder / f"{i:02d}_{name}.wav"
        if path.is_file() and not redo:
            _say(f"  {path.name} : déjà là")
        else:
            pcm, sr = eleven(text, voice, key, post)
            write_wav(path, pcm, sr)
            _say(f"  {path.name} : {len(pcm) / sr:.1f} s ({sr} Hz)")
        out.append(path)
    total = sum(_seconds(p) for p in out)
    _say(f"  total : {total:.0f} s d'extraits")
    return out


def _seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def rel(cfg: Any, path: Path) -> str:
    try:
        return path.resolve().relative_to(cfg.root.resolve()).as_posix()
    except ValueError:
        return str(path)


def stream_check(tts: Any, text: str, sr: int, ch: Any, effets: bool,
                 chunk: int) -> tuple[np.ndarray, dict[str, Any]]:
    """La phrase par la voie « en flux » : 1er morceau, durée, raccords sans clic."""
    from valdar.voice.character import chain_for

    chain = chain_for(sr, ch, effets)
    t0 = time.perf_counter()
    first = None
    pieces = []
    for piece in tts.stream(text, chunk):
        a = chain.process(piece)
        if first is None:
            first = time.perf_counter() - t0
        pieces.append(a)
    total = time.perf_counter() - t0
    audio = np.concatenate(pieces) if pieces else np.zeros(0)
    joints, pos = [], 0
    for a in pieces[:-1]:
        pos += len(a)
        if 0 < pos < len(audio):
            joints.append(abs(float(audio[pos]) - float(audio[pos - 1])))
    typical = float(np.percentile(np.abs(np.diff(audio)), 99.9)) if len(audio) > 2 else 0.0
    return audio, {"premier_morceau_s": round(first or 0.0, 3), "total_s": round(total, 3),
                   "duree_audio_s": round(len(audio) / sr, 2), "morceaux": len(pieces),
                   "saut_max_aux_raccords": round(max(joints), 4) if joints else 0.0,
                   "saut_habituel_999e": round(typical, 4),
                   "sans_clic": (max(joints) if joints else 0.0) <= max(typical, 1e-3) * 1.5}


def render_all(tts: Any, cfg: Any, effets: bool, prefix: str, folder: Path
               ) -> list[dict[str, Any]]:
    from valdar.voice.character import render

    sr = tts.sample_rate
    out = []
    for i, text in enumerate(COMPARE, 1):
        t0 = time.perf_counter()
        whole = render(tts.synthesize(text), sr, cfg.voice.character, effets)
        whole_s = time.perf_counter() - t0
        write_wav(folder / f"{prefix}_{i}.wav", to_pcm16(whole), sr)
        audio, st = stream_check(tts, text, sr, cfg.voice.character, effets,
                                 cfg.voice.stream_chunk_size)
        write_wav(folder / f"{prefix}_{i}_flux.wav", to_pcm16(audio), sr)
        st["phrase_entiere_s"] = round(whole_s, 3)
        out.append({"phrase": text, **st})
        _say(f"  {prefix} {i} : entière {whole_s:.2f} s ; en flux 1er son "
             f"{st['premier_morceau_s']:.2f} s, {st['morceaux']} morceaux, "
             f"{'sans clic' if st['sans_clic'] else 'CLIC aux raccords'}")
    return out


def main(cfg: Any, voice: str, redo: bool = False, back: bool = False,
         effets: bool = False) -> int:
    from valdar.interface.settings import Settings
    from valdar.voice import make_tts

    settings = Settings(cfg)
    base = cfg.storage_path("voix")
    saved = base / "voix_precedente.json"
    if back:
        if not saved.is_file():
            _say("Rien à remettre : aucune voix précédente gardée.")
            return 1
        prev = json.loads(saved.read_text(encoding="utf-8"))
        settings.store("voice.xtts.reference", prev["reference"])
        settings.store("voice.effets", prev["effets"])
        _say(f"Voix précédente remise : {prev['reference']} (effets : "
             f"{'oui' if prev['effets'] else 'non'}). Relance Valdar.")
        return 0
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    folder = base / f"elevenlabs_{voice[:8]}"
    if not key and not any(folder.glob("*.wav")):
        _say("Il manque la clé : variable d'environnement ELEVENLABS_API_KEY (jamais dans un "
             "fichier). Dans PowerShell : $env:ELEVENLABS_API_KEY=\"...\" puis relance.")
        return 1
    _say(f"1. Extraits ElevenLabs ({MODEL}) dans {folder}")
    try:
        clips = make_clips(folder, voice, key, redo) if key else sorted(folder.glob("*.wav"))
    except Exception as exc:          # réseau, clé refusée… jamais la clé dans le message
        _say(f"ElevenLabs injoignable ou refus : {str(exc).replace(key, '***')[:300]}")
        return 1

    _say("2. Avant : l'ancienne voix")
    comp = base / "comparaison"
    old_ref, old_fx = cfg.voice.xtts.reference, cfg.voice.effets
    tts_old = make_tts(cfg)
    report: dict[str, Any] = {"voix": voice, "extraits": [p.name for p in clips]}
    report["avant"] = render_all(tts_old, cfg, old_fx, "avant", comp)

    _say("3. Après : la nouvelle référence, latents recalculés")
    if not saved.is_file():
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps({"reference": old_ref, "effets": old_fx},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    settings.store("voice.xtts.reference", [rel(cfg, p) for p in clips])
    settings.store("voice.effets", effets)
    tts_new = make_tts(cfg)
    tts_new._loader = lambda d, dev: (tts_old.model, tts_old.cfg)   # même modèle, déjà chargé
    report["apres"] = render_all(tts_new, cfg, effets, "apres", comp)
    report["latents"] = tts_new.mode
    (comp / "rapport.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    _say(f"\nÀ écouter : {comp} (avant_N.wav / apres_N.wav, et _flux pour la voix en flux).")
    _say("Pour revenir à l'ancienne voix : valdar voix-nouvelle --revenir")
    return 0
