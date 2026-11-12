"""Le téléphone d'Olivier comme poste de Valdar : sa voix, ses oreilles, ses yeux.

Trois chemins, chacun capable d'échouer sans rien casser :

- **écouter** : le téléphone envoie du WAV 16 kHz (fabriqué dans le navigateur, donc aucun
  décodeur à installer ici), whisper transcrit. Pas de whisper ? il le dit, et Olivier écrit.
- **répondre avec sa voix** : chaque phrase est synthétisée par la même voix que sur le PC
  (XTTS), rangée en mémoire, et le téléphone va la chercher. Voix pas chargée (la carte
  graphique est prise par le cortex) ? il le dit au lieu de faire semblant — le téléphone
  peut alors lire la réponse avec la voix du système, si Olivier l'a laissé faire.
- **voir** : une image de la caméra du téléphone, de temps en temps. Elle sert à reconnaître
  qui parle et à savoir où il est. Caméra refusée ou absente : rien ne se bloque.

Les images et les sons ne sont **pas** écrits sur le disque : ils vivent en mémoire le temps
d'être utilisés, puis disparaissent.
"""
from __future__ import annotations

import io
import threading
import time
import wave
from collections import OrderedDict
from typing import Any

import numpy as np

SR = 16000
MAX_AUDIO = 40          # réponses gardées en mémoire, pas plus
MAX_WAV_BYTES = 8_000_000


class TelError(RuntimeError):
    pass


def wav_to_mono16k(data: bytes) -> np.ndarray:
    """WAV (16 bits) → float32 mono à 16 kHz, pour whisper. Rien d'autre n'est accepté."""
    if len(data) > MAX_WAV_BYTES:
        raise TelError("enregistrement trop long")
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise TelError(f"son illisible ({exc})") from exc
    if width != 2:
        raise TelError("il faut du WAV 16 bits")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != SR:                       # ré-échantillonnage linéaire : suffisant pour la voix
        n = int(round(len(x) * SR / float(rate)))
        if n <= 1:
            raise TelError("enregistrement trop court")
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buf.getvalue()


class Phone:
    """Ce que le serveur appelle pour le téléphone. Tout est paresseux et sans fracas."""

    def __init__(self, session: Any, cfg: Any):
        self.s = session
        self.cfg = cfg                      # la section tel: de la configuration
        self._stt: Any = None
        self._stt_error = ""
        self._tts: Any = None
        self._tts_error = ""
        self._lock = threading.Lock()       # une seule synthèse à la fois (une seule carte)
        self.clips: OrderedDict[str, bytes] = OrderedDict()
        self.last_seen: dict[str, Any] = {}
        self.seen_at = 0.0
        self._watch: Any = None
        self._frame: Any = None

    # ------------------------------------------------------------ ses oreilles
    def _ears(self) -> Any:
        if self._stt is not None or self._stt_error:
            return self._stt
        ec = self.s.cfg.ears
        try:
            from valdar.ears.models import WhisperSTT

            self._stt = WhisperSTT(ec.stt_model, ec.stt_device, ec.stt_compute_type,
                                   beam_size=1, hint=ec.names[0] if ec.names else "Valdar",
                                   threads=ec.stt_threads)
        except Exception as exc:
            self._stt_error = str(exc)
        return self._stt

    def transcribe(self, data: bytes) -> str:
        audio = wav_to_mono16k(data)
        if float(np.abs(audio).max() or 0.0) < 0.004:
            raise TelError("je n'ai rien entendu")
        ears = self._ears()
        if ears is None:
            raise TelError(f"je n'ai pas d'oreilles ici ({self._stt_error}) — écris-moi")
        out = ears.transcribe(audio)
        text = (getattr(out, "text", "") or "").strip()
        if not text:
            raise TelError("je n'ai pas compris, redis-le")
        return text

    # ------------------------------------------------------------ sa voix
    def _voice(self) -> Any:
        """La voix du PC si elle est déjà chargée ; sinon on la charge pour le téléphone."""
        sp = getattr(self.s, "speaker", None)
        if sp is not None and getattr(sp, "tts", None) is not None:
            if sp.ready.is_set() and sp.load_error:
                self._tts_error = sp.load_error
                return None
            return sp.tts
        if self._tts is not None or self._tts_error:
            return self._tts
        if not self.cfg.charger_la_voix:
            self._tts_error = "sa voix n'est pas allumée sur le PC"
            return None
        try:
            from valdar.__main__ import _build_tts

            tts = _build_tts(self.s.cfg)
            tts.load()
            self._tts = tts
        except Exception as exc:
            self._tts_error = str(exc)
        return self._tts

    def voice_state(self) -> dict[str, Any]:
        sp = getattr(self.s, "speaker", None)
        prete = sp is not None and getattr(sp, "tts", None) is not None and not sp.load_error
        return {"voix": bool(prete or self._tts is not None),
                "pourquoi": "" if prete or self._tts is not None else
                            (self._tts_error or "sa voix n'est pas chargée"),
                "oreilles": bool(self._stt is not None or not self._stt_error),
                "secours_navigateur": bool(self.cfg.voix_du_telephone)}

    def say(self, text: str) -> str:
        """Synthétise une phrase et renvoie son identifiant (vide si pas de voix)."""
        from valdar.voice.character import speakable

        clean = speakable(text)
        if not clean.strip():
            return ""
        tts = self._voice()
        if tts is None:
            return ""
        try:
            with self._lock:
                audio = tts.synthesize(clean)
                rate = int(getattr(tts, "sample_rate", SR))
                try:                     # le grain de sa voix, comme sur le PC
                    from valdar.voice.character import render

                    audio = render(audio, rate, self.s.cfg.voice.character,
                                   self.s.cfg.voice.effects)
                except Exception:
                    pass
                wav = to_wav(audio, rate)
        except Exception as exc:
            self._tts_error = str(exc)
            return ""
        cid = f"{int(time.time() * 1000):x}{len(self.clips):02x}"
        self.clips[cid] = wav
        while len(self.clips) > MAX_AUDIO:
            self.clips.popitem(last=False)
        return cid

    def clip(self, cid: str) -> bytes | None:
        return self.clips.get(cid)

    # ------------------------------------------------------------ parler
    def answer(self, text: str, who: Any, voice: bool) -> None:
        """Un tour de conversation venu du téléphone. La voix part vers le téléphone."""
        def speak(phrase: str) -> None:
            cid = self.say(phrase) if voice else ""
            if cid:
                self.s.bus.publish("tel_audio", {"id": cid, "texte": phrase})
        self.s.answer(text, who=who, source="téléphone", speak=speak if voice else None)

    # ------------------------------------------------------------ ses yeux
    def _eyes(self) -> Any:
        """Le même regard que sur le PC, mais nourri par la caméra du téléphone."""
        if self._watch is not None:
            return self._watch
        from valdar.vision.watch import Watch

        maison = getattr(self.s, "watch", None)
        self._watch = Watch(self.s.cfg.vision, grab=lambda: self._frame,
                            detector=getattr(maison, "detector", None),
                            faces=getattr(self.s, "face_id", None))
        if maison is not None:         # il voit une seule maison, pas deux
            self._watch.seen = maison.seen
        return self._watch

    def look(self, jpeg: bytes) -> dict[str, Any]:
        """Une image de la caméra du téléphone : qui c'est, et ce qu'il y a autour."""
        if len(jpeg) > 4_000_000:
            return {"erreur": "image trop grosse"}
        frame = self._decode(jpeg)
        if frame is None:
            return {"erreur": "je ne sais pas lire cette image ici (OpenCV absent)"}
        self._frame = frame
        try:
            scene = self._eyes().step()
        except Exception as exc:
            return {"erreur": f"je n'arrive pas a regarder : {exc}"}
        finally:
            self._frame = None             # rien n'est garde
        out = scene.to_dict() if scene is not None else {}
        connu = next((f for f in (out.get("faces") or []) if f.get("person")), None)
        out["qui"] = None if connu is None else {"id": connu["person"],
                                                 "score": connu.get("similarity")}
        self.last_seen, self.seen_at = out, time.time()
        return out

    @staticmethod
    def _decode(jpeg: bytes) -> Any:
        try:
            import cv2
        except ImportError:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return None if img is None else img
