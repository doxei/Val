"""Le portier (avenant 3 §1) : Valdar écoute la pièce sans la transcrire.

Chaîne : micro → tampon circulaire **en mémoire vive** → détection de parole (VAD) → segment
de parole → mot d'éveil ? → seulement alors transcription et réponse. Pendant une conversation
engagée, l'éveil reste ouvert quelques secondes (comme avant).

Invariants (testés) :
1. rien n'est jamais écrit sur disque : ni audio, ni texte entendu par hasard ;
2. sans éveil et hors conversation engagée, aucun segment n'atteint la transcription
   « complète » ; le mot d'éveil peut être cherché par un petit modèle (voir `wake.py`), dont
   le texte est jeté aussitôt ;
3. quand quelqu'un parle pendant que Valdar parle, il se tait (coupure).
"""
from __future__ import annotations

import collections
import concurrent.futures
import time
from collections.abc import Callable
from dataclasses import dataclass, field
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
    # La phrase elle-même, en mémoire vive seulement, pour reconnaître la voix de qui parle.
    audio: np.ndarray | None = field(default=None, repr=False)
    # Chronométrage : fin de parole → texte prêt (secondes), transcription anticipée ou non.
    timing: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Decision:
    """Ce que le portier conclut d'un segment (calculé à part, appliqué ensuite)."""
    kind: str                       # ignored | junk | heard
    text: str = ""
    note: str = ""
    stats: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0            # temps de calcul (éveil + transcription)


_JUNK = ("sous-titres", "amara", "merci d'avoir regardé", "abonnez-vous",
         "n'oubliez pas de vous abonner")


def is_junk(text: str) -> bool:
    """Phrases que whisper invente sur du bruit (relevées sur l'ancienne installation)."""
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
                 clock: Callable[[], float] = time.time,
                 on_frame: Callable[[np.ndarray, float], None] | None = None,
                 on_note: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.vad, self.wake, self.stt = vad, wake, stt
        self.on_heard = on_heard
        self.on_barge_in = on_barge_in
        self.speaking = speaking or (lambda: False)
        self.clock = clock
        self.on_frame = on_frame      # voie basse : niveau du son vers le cœur (ambient.py)
        self.on_note = on_note        # diagnostic : ce que le portier décide (jamais le texte
        self.notes: collections.deque = collections.deque(maxlen=12)   # non adressé)
        self._rest = np.zeros(0, np.float32)
        self.tap: Callable[[np.ndarray], bool] | None = None
        # Écho : sans annulation d'écho, le micro entend Valdar dans les haut-parleurs. Pendant
        # qu'il parle, on apprend le niveau de cet écho ; seule une voix nettement plus forte
        # (quelqu'un près du micro) le coupe, et ce qu'il dit lui-même n'est jamais transcrit.
        self._echo: float | None = None
        self._loud = 0
        self._spoken = 0
        self._seg_during_speech = False
        self._barged = False
        frame = int(SR * cfg.frame_ms / 1000)
        self.frame = frame
        self.preroll: collections.deque = collections.deque(
            maxlen=max(1, int(cfg.preroll_seconds * 1000 / cfg.frame_ms)))
        self._segment: list[np.ndarray] = []
        self._silence = 0
        self._speech = 0
        self.engaged_until = 0.0
        self.stats = {"segments": 0, "woken": 0, "transcribed": 0, "ignored": 0, "junk": 0,
                      "anticipated": 0}
        # Transcription anticipée : dès que le silence commence (early_ms), on transcrit déjà
        # la phrase ; si la personne ne reprend pas, le texte est prêt à la fin de la phrase
        # au lieu de commencer à ce moment-là. Un seul fil fait tout (whisper n'est jamais
        # appelé deux fois en même temps).
        self._worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="valdar-stt")
        self._spec: tuple[int, int, concurrent.futures.Future] | None = None
        self._silence_at: float | None = None

    # ------------------------------------------------------------------ flux
    def feed(self, audio: np.ndarray) -> None:
        """Pousse de l'audio (float32 mono 16 kHz, n'importe quelle longueur)."""
        buf = np.concatenate([self._rest, audio.astype(np.float32).reshape(-1)])
        n = len(buf) // self.frame * self.frame
        for i in range(0, n, self.frame):
            self._frame(buf[i:i + self.frame])
        self._rest = buf[n:].copy()          # rien n'est perdu entre deux blocs du micro

    def _note(self, text: str) -> None:
        self.notes.append(text)
        if self.on_note is not None:
            self.on_note(text)

    def _frame(self, f: np.ndarray) -> None:
        p = self.vad.is_speech(f)
        if self.on_frame is not None:
            self.on_frame(f, p)
        speaking = self.speaking()
        level = 20.0 * float(np.log10(np.sqrt(np.mean(f.astype(np.float32) ** 2)) + 1e-6))
        if speaking:
            # Les premières 0,5 s, on apprend vite le niveau de l'écho et personne ne peut
            # le couper (sinon le début de sa propre phrase le ferait taire).
            self._spoken += 1
            learning = self._spoken * self.cfg.frame_ms < 500
            tau = 0.1 if learning else self.cfg.echo_tau_seconds
            a = 1.0 - np.exp(-self.cfg.frame_ms / 1000 / tau)
            self._echo = level if self._echo is None else self._echo + a * (level - self._echo)
            loud = (not learning and p >= self.cfg.vad_threshold
                    and level >= self._echo + self.cfg.barge_in_margin_db)
            self._loud = self._loud + 1 if loud else max(0, self._loud - 1)
            if self._loud == self.cfg.barge_in_frames and not self._barged:
                self._barged = True
                self._note("on me coupe la parole : je me tais")
                if self.on_barge_in is not None:
                    self.on_barge_in()
        else:
            self._echo, self._loud, self._spoken = None, 0, 0
        if self._segment:
            self._segment.append(f)
            if p >= self.cfg.vad_threshold:
                self._silence = 0
                self._speech += 1
                self._silence_at = None
            else:
                if self._silence == 0:
                    self._silence_at = time.perf_counter()     # la parole vient de s'arrêter
                self._silence += 1
                self._maybe_anticipate()
            long = len(self._segment) * self.cfg.frame_ms / 1000 >= self.cfg.max_segment_seconds
            if self._silence * self.cfg.frame_ms >= self.cfg.end_silence_ms or long:
                seg = np.concatenate(self._segment)
                spec, said = self._spec, self._speech
                self._segment, self._silence, self._speech, self._spec = [], 0, 0, None
                self._close(seg, spec if spec is not None and spec[0] == said else None)
        elif p >= self.cfg.vad_threshold:
            self._segment = list(self.preroll) + [f]
            self._speech, self._silence = 1, 0
            self._seg_during_speech = speaking
            self._barged = False
            self.preroll.clear()
        else:
            self.preroll.append(f)

    # -------------------------------------------------------------- décision
    def engaged(self) -> bool:
        return self.clock() < self.engaged_until

    def _ignorable(self, seg: np.ndarray) -> bool:
        return (len(seg) / SR < self.cfg.min_segment_seconds
                or (self._seg_during_speech and not self._barged))

    def _maybe_anticipate(self) -> None:
        early = self.cfg.early_ms
        if (early <= 0 or self._spec is not None or self.tap is not None
                or self._silence * self.cfg.frame_ms < early):
            return
        snap = np.concatenate(self._segment)
        if self._ignorable(snap):
            return
        self._spec = (self._speech, len(snap),
                      self._worker.submit(self._decide, snap, self.engaged()))

    def _decide(self, seg: np.ndarray, engaged: bool) -> _Decision:
        """Éveil puis transcription d'un segment. Ne touche à rien d'autre : le résultat est
        appliqué par `_close` (et jeté si la personne a repris la parole entre-temps)."""
        t0 = time.perf_counter()
        dur = len(seg) / SR
        st = {"segments": 1}
        if not engaged:
            if not self.wake.heard(seg):
                seen = getattr(self.wake, "last_text", None)
                return _Decision("ignored", note=(
                    f"parole {dur:.1f} s : pas de « {self.cfg.names[0]} » entendu"
                    + (f" (compris : « {seen.strip()} »)" if seen is not None else "")),
                    stats={**st, "ignored": 1}, seconds=time.perf_counter() - t0)
            st["woken"] = 1
        tr = self.stt.transcribe(seg)
        st["transcribed"] = 1
        if is_junk(tr.text) or tr.logprob < self.cfg.min_logprob or \
                tr.no_speech > self.cfg.max_no_speech:
            err = getattr(self.stt, "last_error", None)
            return _Decision("junk", note=(
                f"parole {dur:.1f} s adressée, mais transcription rejetée"
                + (f" (erreur : {err})" if err else " (vide)" if not tr.text.strip() else "")),
                stats={**st, "junk": 1}, seconds=time.perf_counter() - t0)
        used = getattr(self.stt, "used", "")
        return _Decision("heard", tr.text.strip(),
                         f"parole {dur:.1f} s comprise" + (f" (par {used})" if used else ""),
                         st, time.perf_counter() - t0)

    def _close(self, seg: np.ndarray, spec: tuple[int, int, Any] | None = None) -> None:
        dur = len(seg) / SR
        if self._ignorable(seg):
            if dur >= self.cfg.min_segment_seconds:
                self._note(f"parole {dur:.1f} s pendant que je parlais : ma propre voix, "
                           "ignorée")
            return
        if self.tap is not None and self.tap(seg):
            return            # enrôlement de voix en cours : la phrase sert d'échantillon
        silence_at = self._silence_at
        t_close = time.perf_counter()
        anticipated = spec is not None
        if anticipated:
            d = spec[2].result()       # souvent déjà prêt : calculé pendant le silence
            self.stats["anticipated"] += 1
        else:
            d = self._worker.submit(self._decide, seg, self.engaged()).result()
        for k, v in d.stats.items():
            self.stats[k] = self.stats.get(k, 0) + v
        if d.note:
            self._note(d.note + (" (anticipée)" if anticipated else ""))
        if d.kind != "heard":
            return
        self.engaged_until = self.clock() + self.cfg.engaged_seconds
        now = time.perf_counter()
        timing = {"transcription_s": round(d.seconds, 3), "anticipee": anticipated,
                  "attente_apres_silence_s": round(now - t_close, 3),
                  "fin_de_parole_au_texte_s": round(now - silence_at, 3)
                  if silence_at is not None else None}
        self.on_heard(Heard(d.text, self.clock(), dur, tone(seg), seg, timing))

    def keep_engaged(self) -> None:
        """Valdar vient de parler : on lui répond sans redire son nom."""
        self.engaged_until = max(self.engaged_until, self.clock() + self.cfg.engaged_seconds)

    def status(self) -> dict[str, Any]:
        return dict(self.stats, engaged=self.engaged(), notes=list(self.notes))
