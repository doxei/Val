"""Le portier (avenant 3 §1) : Valdar écoute la pièce sans la transcrire.

Chaîne : micro → tampon circulaire **en mémoire vive** → détection de parole (VAD) → segment
de parole → mot d'éveil ? → seulement alors transcription et réponse. Pendant une conversation
engagée, l'éveil reste ouvert quelques secondes (comme RAUB).

Invariants (testés) :
1. rien n'est jamais écrit sur disque : ni audio, ni texte entendu par hasard ;
2. sans éveil et hors conversation engagée, aucun segment n'atteint la transcription
   « complète » ; le mot d'éveil peut être cherché par un petit modèle (voir `wake.py`), dont
   le texte est jeté aussitôt ;
3. quand quelqu'un parle pendant que Valdar parle, il se tait (coupure).
"""
from __future__ import annotations

import collections
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from valdar.config.loader import EarsConfig

SR = 16000


class VAD(Protocol):
    def is_speech(self, frame: np.ndarray) -> float: ...   # probabilité de parole 0..1


class Wake(Protocol):
    def heard(self, segment: np.ndarray) -> bool: ...


class STT(Protocol):
    def transcribe(self, segment: np.ndarray) -> Transcript: ...


@dataclass
class Transcript:
    text: str
    logprob: float = 0.0
    no_speech: float = 0.0


@dataclass
class Heard:
    """Ce que le portier laisse passer : une phrase adressée à Valdar."""
    text: str
    t: float
    seconds: float
    tone: dict[str, float]


_JUNK = ("sous-titres", "amara", "merci d'avoir regardé", "abonnez-vous",
         "n'oubliez pas de vous abonner")


def is_junk(text: str) -> bool:
    """Phrases que whisper invente sur du bruit (relevées dans RAUB)."""
    t = text.lower().strip().rstrip(".")
    return not t or any(j in t for j in _JUNK) or len(t) < 2


def tone(segment: np.ndarray) -> dict[str, float]:
    """Tonalité de la voix, pour la voie basse : énergie et débit (grossiers, sans modèle)."""
    x = segment.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    frames = x[: len(x) // 320 * 320].reshape(-1, 320) if x.size >= 320 else x.reshape(1, -1)
    e = np.sqrt(np.mean(frames * frames, axis=1))
    voiced = e > max(0.01, 0.3 * float(e.max()) if e.size else 0.0)
    bursts = int(np.sum(np.diff(voiced.astype(int)) == 1)) if voiced.size > 1 else 0
    dur = len(x) / SR
    return {"energie": rms, "debit": bursts / dur if dur > 0 else 0.0, "duree": dur}


class EnergyVAD:
    """VAD de secours (énergie) : pour les tests et quand Silero manque."""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if frame.size else 0.0
        return min(1.0, rms / (2 * self.threshold))


class Gate:
    def __init__(self, cfg: EarsConfig, vad: VAD, wake: Wake, stt: STT,
                 on_heard: Callable[[Heard], None],
                 on_barge_in: Callable[[], None] | None = None,
                 speaking: Callable[[], bool] | None = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.vad, self.wake, self.stt = vad, wake, stt
        self.on_heard = on_heard
        self.on_barge_in = on_barge_in
        self.speaking = speaking or (lambda: False)
        self.clock = clock
        frame = int(SR * cfg.frame_ms / 1000)
        self.frame = frame
        self.preroll: collections.deque = collections.deque(
            maxlen=max(1, int(cfg.preroll_seconds * 1000 / cfg.frame_ms)))
        self._segment: list[np.ndarray] = []
        self._silence = 0
        self._speech = 0
        self.engaged_until = 0.0
        self.stats = {"segments": 0, "woken": 0, "transcribed": 0, "ignored": 0, "junk": 0}

    # ------------------------------------------------------------------ flux
    def feed(self, audio: np.ndarray) -> None:
        """Pousse de l'audio (float32 mono 16 kHz, n'importe quelle longueur)."""
        buf = audio.astype(np.float32).reshape(-1)
        for i in range(0, len(buf) - self.frame + 1, self.frame):
            self._frame(buf[i:i + self.frame])

    def _frame(self, f: np.ndarray) -> None:
        p = self.vad.is_speech(f)
        if self._segment:
            self._segment.append(f)
            if p >= self.cfg.vad_threshold:
                self._silence = 0
                self._speech += 1
            else:
                self._silence += 1
            if self._speech == self.cfg.barge_in_frames and self.speaking() and \
                    self.on_barge_in is not None:
                self.on_barge_in()
            long = len(self._segment) * self.cfg.frame_ms / 1000 >= self.cfg.max_segment_seconds
            if self._silence * self.cfg.frame_ms >= self.cfg.end_silence_ms or long:
                seg = np.concatenate(self._segment)
                self._segment, self._silence, self._speech = [], 0, 0
                self._close(seg)
        elif p >= self.cfg.vad_threshold:
            self._segment = list(self.preroll) + [f]
            self._speech, self._silence = 1, 0
            self.preroll.clear()
        else:
            self.preroll.append(f)

    # -------------------------------------------------------------- décision
    def engaged(self) -> bool:
        return self.clock() < self.engaged_until

    def _close(self, seg: np.ndarray) -> None:
        dur = len(seg) / SR
        if dur < self.cfg.min_segment_seconds:
            return
        self.stats["segments"] += 1
        if not self.engaged():
            if not self.wake.heard(seg):
                self.stats["ignored"] += 1      # parole qui ne s'adresse pas à Valdar : oubliée
                return
            self.stats["woken"] += 1
        tr = self.stt.transcribe(seg)
        self.stats["transcribed"] += 1
        if is_junk(tr.text) or tr.logprob < self.cfg.min_logprob or \
                tr.no_speech > self.cfg.max_no_speech:
            self.stats["junk"] += 1
            return
        self.engaged_until = self.clock() + self.cfg.engaged_seconds
        self.on_heard(Heard(tr.text.strip(), self.clock(), dur, tone(seg)))

    def keep_engaged(self) -> None:
        """Valdar vient de parler : on lui répond sans redire son nom."""
        self.engaged_until = max(self.engaged_until, self.clock() + self.cfg.engaged_seconds)

    def status(self) -> dict[str, Any]:
        return dict(self.stats, engaged=self.engaged())
