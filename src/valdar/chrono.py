"""`valdar chrono` : chronomètre un tour complet, étape par étape, sur la vraie machine.

Mesure, sans rien écrire dans la vraie mémoire (copie des bases dans un dossier temporaire) :
- l'environnement : Ollama (version, modèles chargés), carte graphique ;
- **au clavier** : prompt (jetons), chargement du modèle, 1er jeton, 1re phrase, fin ;
- **la voix** : phrase entière contre flux (XTTS), 1er son ;
- **les oreilles** : whisper sur une vraie phrase (dite par XTTS), en temps réel, avec et sans
  transcription anticipée : fin de parole → texte prêt ;
- **le sens** : bge-m3 sur le processeur (serveur à part), temps d'une requête, et calibrage
  de `embeddings.min_similarity` ; vérifie que Gemma reste chargé ;
- **la mémoire à la demande** : Gemma appelle-t-il `fouiller_memoire` pour le passé, et pas
  pour un bonjour ?

Résultat : affiché, et écrit dans `data/chrono.json` (pour la suite des réglages).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# Phrases de calibrage : paires qui veulent dire la même chose, paires sans rapport.
SAME = [("la vitre du bed est sale", "le plateau en verre de l'imprimante est encrassé"),
        ("Sony a mangé ma chaussure", "le chien a mâchouillé une de mes baskets"),
        ("j'ai mal au crâne ce soir", "ce soir j'ai une migraine"),
        ("la buse est bouchée", "le filament ne sort plus de la tête d'impression"),
        ("on mange quoi ce midi ?", "qu'est-ce qu'on fait à manger pour le déjeuner ?"),
        ("Héloïse a eu une bonne note", "ma fille a bien réussi son contrôle")]
OTHER = [("la vitre du bed est sale", "Sony a mangé ma chaussure"),
         ("j'ai mal au crâne ce soir", "la buse est bouchée"),
         ("on mange quoi ce midi ?", "le firmware Klipper est à jour"),
         ("Héloïse a eu une bonne note", "il pleut sur Vieux-Thann"),
         ("le plateau en verre est encrassé", "on regarde un film ce soir ?"),
         ("ma fille a bien réussi son contrôle", "le filament PETG colle mieux à 80 degrés")]
TURNS = [("salut, ça va ?", "bonjour"),
         ("salut !", "bonjour (2e fois : cache)"),
         ("tu te souviens de ce qu'on s'est dit sur la buse ?", "passé"),
         ("c'est quoi déjà qu'on avait décidé pour ta voix ?", "passé")]
SPOKEN = "Valdar, tu te souviens de ce qu'on a dit hier sur la buse de l'imprimante ?"


def _p(line: str) -> None:
    print(line, flush=True)


class Spy:
    """Enveloppe du modèle : note pour chaque appel le 1er jeton et l'usage d'Ollama."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, system="", tools=None, params=None, **kw):
        t0 = time.perf_counter()
        first: list[float] = []
        cb = kw.get("on_token")
        if cb is not None:
            def on_token(piece: str) -> None:
                if not first:
                    first.append(time.perf_counter() - t0)
                cb(piece)
            kw["on_token"] = on_token
        res = self.inner.chat(messages, system=system, tools=tools, params=params, **kw)
        u = getattr(res, "usage", {}) or {}
        self.calls.append({
            "secondes": round(time.perf_counter() - t0, 3),
            "premier_jeton_s": round(first[0], 3) if first else None,
            "jetons_prompt": u.get("prompt_eval_count"),
            "jetons_sortie": u.get("eval_count"),
            "chargement_s": round(u.get("load_duration", 0) / 1e9, 3),
            "eval_prompt_s": round(u.get("prompt_eval_duration", 0) / 1e9, 3),
            "eval_sortie_s": round(u.get("eval_duration", 0) / 1e9, 3),
            "outils_demandes": [c.name for c in getattr(res, "tool_calls", [])],
        })
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _ps(url: str) -> list[dict[str, Any]]:
    try:
        r = httpx.get(url.rstrip("/") + "/api/ps", timeout=3.0)
        return [{"nom": m.get("name"), "vram_go": round((m.get("size_vram") or 0) / 1e9, 2),
                 "taille_go": round((m.get("size") or 0) / 1e9, 2)}
                for m in r.json().get("models", [])]
    except Exception as exc:
        return [{"erreur": str(exc)}]


def _gpu() -> dict[str, Any]:
    import subprocess

    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=5, creationflags=flags)
        used, total = (float(x) for x in r.stdout.split(",")[:2])
        return {"utilise_mo": used, "total_mo": total}
    except Exception as exc:
        return {"erreur": str(exc)}


def _scratch(cfg: Any) -> Path:
    """Copie des données (pour des prompts réalistes) dans un dossier jetable."""
    src = cfg.storage_path("x").parent
    dst = Path(tempfile.mkdtemp(prefix="valdar-chrono-"))
    for f in src.glob("*"):
        if f.is_file() and f.suffix in (".db", ".json", ".jsonl") and f.name != "chrono.json":
            shutil.copy2(f, dst / f.name)
    return dst


def run(cfg: Any, voice: bool = True, ears: bool = True, sense: bool = True) -> dict[str, Any]:
    from valdar.memory.sidecar import Sidecar
    from valdar.runtime import Runtime
    from valdar.voice.stream import SentenceStream

    out: dict[str, Any] = {"quand": time.strftime("%Y-%m-%d %H:%M:%S")}
    url = cfg.llm.url
    try:
        ver = httpx.get(url + "/api/version", timeout=3).json().get("version")
    except Exception as exc:
        _p(f"Ollama ne répond pas ({exc}) : lance-le d'abord.")
        return {"erreur": "ollama absent"}
    out["env"] = {"ollama": ver, "OLLAMA_MAX_LOADED_MODELS":
                  os.environ.get("OLLAMA_MAX_LOADED_MODELS", "(non défini ici)"),
                  "charges_avant": _ps(url), "carte_avant": _gpu()}
    _p(f"Ollama {ver} ; chargés : {out['env']['charges_avant']}")

    # ------------------------------------------------------------ le sens (avant Gemma)
    side = None
    if sense:
        side = Sidecar(cfg.embeddings.sidecar_port)
        ok = side.start(wait=30)
        out["sens"] = _sense(cfg, side.url if ok else None, side.error)

    # ------------------------------------------------------------ clavier
    work = _scratch(cfg)
    c2 = cfg.model_copy(deep=True)
    c2.storage.dir = str(work)
    c2._root = cfg.root
    rt = Runtime(c2, persist=False, printer=False)
    spy = Spy(rt.agent.llm)
    rt.agent.llm = spy
    turns = []
    for text, kind in TURNS:
        spy.calls.clear()
        marks: dict[str, float] = {}
        t0 = time.perf_counter()

        def said(sentence: str, marks: dict[str, float] = marks, t0: float = t0) -> None:
            marks.setdefault("premiere_phrase_s", time.perf_counter() - t0)

        stream = SentenceStream(said)
        reply = rt.handle(text, on_text=stream.feed)
        stream.close()
        total = time.perf_counter() - t0
        first_tok = next((c["premier_jeton_s"] for c in spy.calls
                          if c["premier_jeton_s"] is not None), None)
        offset = sum(c["secondes"] for c in spy.calls[:-1]) if len(spy.calls) > 1 else 0.0
        turns.append({"message": text, "genre": kind, "total_s": round(total, 3),
                      "premiere_phrase_s": round(marks.get("premiere_phrase_s", total), 3),
                      "premier_jeton_s": first_tok, "appels": list(spy.calls),
                      "outils": reply.tools_used, "reponse": reply.text[:160],
                      "attente_outils_s": round(offset, 3), "erreur": reply.error})
        c0 = spy.calls[0] if spy.calls else {}
        _p(f"[clavier] « {text} » : 1re phrase {turns[-1]['premiere_phrase_s']:.2f} s, "
           f"total {total:.2f} s, prompt {c0.get('jetons_prompt')} jetons "
           f"(éval {c0.get('eval_prompt_s')} s, chargement {c0.get('chargement_s')} s), "
           f"outils {reply.tools_used or '—'}")
    out["clavier"] = turns
    out["fouiller_memoire"] = {
        "bonjour_sans_outil": all("fouiller_memoire" not in t["outils"] for t in turns
                                  if t["genre"].startswith("bonjour")),
        "passe_avec_outil": [("fouiller_memoire" in t["outils"]) for t in turns
                             if t["genre"] == "passé"]}
    out["charges_apres_clavier"] = _ps(url)
    rt.stop()

    # ------------------------------------------------------------ voix
    tts = None
    if voice and cfg.voice.enabled:
        tts, out["voix"] = _voice(cfg)
    # ------------------------------------------------------------ oreilles
    if ears:
        out["oreilles"] = _ears(cfg, tts)
    if sense and side is not None:
        out["charges_fin"] = _ps(url)
        out["sidecar_charges"] = _ps(side.url)
    out["carte_fin"] = _gpu()
    path = cfg.storage_path("chrono.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _p(f"\nÉcrit : {path}")
    shutil.rmtree(work, ignore_errors=True)
    return out


def _sense(cfg: Any, url: str | None, error: str) -> dict[str, Any]:
    from valdar.memory.embedder import Embedder

    if url is None:
        _p(f"[sens] serveur des plongements absent : {error}")
        return {"erreur": error}
    emb = Embedder(cfg.embeddings, url)
    t0 = time.perf_counter()
    first = emb.embed_one("premier appel")
    cold = time.perf_counter() - t0
    if first is None:
        _p(f"[sens] bge-m3 indisponible : {emb.last_error}")
        return {"erreur": emb.last_error}
    warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        emb.embed_one("tu te souviens de la buse ?")
        warm.append(time.perf_counter() - t0)

    def cos(a: str, b: str) -> float:
        m = emb.embed_many([a, b])
        return float(m[0] @ m[1]) if m is not None else float("nan")

    same = [round(cos(a, b), 3) for a, b in SAME]
    other = [round(cos(a, b), 3) for a, b in OTHER]
    lo, hi = min(same), max(other)
    proposal = round((lo + hi) / 2, 2) if lo > hi else round(np.percentile(same, 25), 2)
    res = {"premier_appel_s": round(cold, 3), "requete_s": round(float(np.median(warm)), 3),
           "meme_sens": same, "sans_rapport": other, "seuil_propose": proposal,
           "separable": lo > hi}
    _p(f"[sens] requête {res['requete_s']*1000:.0f} ms (1er appel {cold:.1f} s) ; même sens "
       f"{min(same):.2f}–{max(same):.2f}, sans rapport {min(other):.2f}–{max(other):.2f} → "
       f"seuil proposé {proposal}")
    return res


def _voice(cfg: Any) -> tuple[Any, dict[str, Any]]:
    from valdar.voice import make_tts
    from valdar.voice.character import CharacterStream, apply

    tts = make_tts(cfg)
    if tts.check_files():
        return None, {"erreur": "fichiers de voix absents"}
    t0 = time.perf_counter()
    try:
        tts.load()
    except Exception as exc:
        _p(f"[voix] impossible : {exc}")
        return None, {"erreur": str(exc)}
    res: dict[str, Any] = {"chargement_s": round(time.perf_counter() - t0, 2),
                           "mode": getattr(tts, "mode", "")}
    sentence = "Salut Olivier, je t'entends très bien, on s'y remet quand tu veux."
    tts.synthesize("Échauffement.")            # premier appel : compilation, caches
    t0 = time.perf_counter()
    raw = tts.synthesize(sentence)
    whole = time.perf_counter() - t0
    apply(raw, tts.sample_rate, cfg.voice.character)
    res["phrase_entiere_s"] = round(whole, 3)
    res["duree_audio_s"] = round(len(raw) / tts.sample_rate, 2)
    for size in (10, 20, 40):
        chain = CharacterStream(tts.sample_rate, cfg.voice.character)
        t0 = time.perf_counter()
        first = None
        n = 0
        for piece in tts.stream(sentence, size):
            a = chain.process(piece)
            if first is None:
                first = time.perf_counter() - t0
            n += len(a)
        res[f"flux_{size}"] = {"premier_morceau_s": round(first or 0, 3),
                               "total_s": round(time.perf_counter() - t0, 3),
                               "temps_reel": round((time.perf_counter() - t0)
                                                   / max(n / tts.sample_rate, 1e-3), 2)}
    _p(f"[voix] phrase entière {whole:.2f} s ; flux 1er son : "
       + ", ".join(f"{k[5:]}→{v['premier_morceau_s']:.2f} s" for k, v in res.items()
                   if k.startswith("flux_")))
    try:
        import torch

        res["vram_voix_go"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    except Exception:
        pass
    return tts, res


def _ears(cfg: Any, tts: Any) -> dict[str, Any]:
    from valdar.ears.gate import SR, EnergyVAD, Gate
    from valdar.ears.models import OnePass, WhisperSTT

    res: dict[str, Any] = {}
    if tts is None:
        _p("[oreilles] pas de voix pour fabriquer une phrase de test : je passe")
        return {"erreur": "voix absente"}
    from scipy.signal import resample_poly

    wav = tts.synthesize(SPOKEN).astype(np.float32) / 32768.0
    speech = resample_poly(wav, SR, tts.sample_rate).astype(np.float32) * 0.5
    ec = cfg.ears
    try:
        stt = WhisperSTT(ec.stt_model, ec.stt_device, ec.stt_compute_type, beam_size=1,
                         hint=ec.names[0], threads=ec.stt_threads)
    except Exception as exc:
        return {"erreur": str(exc)}
    stt.transcribe(speech[: SR])             # échauffement
    t0 = time.perf_counter()
    tr = stt.transcribe(speech)
    res["whisper"] = {"modele": ec.stt_model, "duree_parole_s": round(len(speech) / SR, 2),
                      "transcription_s": round(time.perf_counter() - t0, 3), "texte": tr.text}
    _p(f"[oreilles] whisper {ec.stt_model} : {res['whisper']['transcription_s']:.2f} s pour "
       f"{len(speech) / SR:.1f} s de parole → « {tr.text} »")
    for early in (0, ec.early_ms):
        both = OnePass(stt, ec.names)
        got: list[Any] = []
        c = ec.model_copy(update={"early_ms": early})
        g = Gate(c, EnergyVAD(0.01), both, both, got.append)
        block = SR // 10                     # en temps réel, par blocs de 100 ms
        audio = np.concatenate([speech, np.zeros(int(1.2 * SR), np.float32)])
        for i in range(0, len(audio), block):
            g.feed(audio[i:i + block])
            time.sleep(0.1)
        key = "anticipee" if early else "sans_anticipation"
        res[key] = got[0].timing if got else {"erreur": "rien entendu"}
        _p(f"[oreilles] {key} : fin de parole → texte prêt "
           f"{res[key].get('fin_de_parole_au_texte_s')} s")
    return res


def main(cfg: Any, voice: bool = True, ears: bool = True, sense: bool = True) -> int:
    run(cfg, voice=voice, ears=ears, sense=sense)
    return 0
