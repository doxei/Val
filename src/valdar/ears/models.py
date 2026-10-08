"""Modèles de l'écoute, chargés seulement sur la machine (extra `[ears]`).

- VAD : Silero (ONNX, CPU, < 1 ms par trame).
- Mot d'éveil :
  * `OpenWakeWordWake` : modèle « Valdar » entraîné (fichier .onnx/.tflite) — la vraie solution ;
  * `TranscriptWake` : **compromis temporaire** tant que ce modèle n'existe pas. Un petit
    whisper (« tiny ») lit seulement le **début** du segment, en mémoire vive, cherche
    « Valdar » (et ses déformations), puis le texte est jeté. Rien n'est gardé ni journalisé.
- Transcription : faster-whisper (large-v3-turbo, int8). Sur le CPU par défaut : la carte
  graphique est pleine avec Gemma + la voix (mesure de la phase 3).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np

from valdar.config.loader import EarsConfig
from valdar.ears.gate import SR, Transcript


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _sound(w: str) -> str:
    """Forme sonore grossière : b/w → v, fins muettes (d, t, s, e, h) et « ar/arc/ard »
    ramenées à « ar ». « Baldar », « Valdard », « Val d'arc » sonnent comme « Valdar »."""
    w = w.replace("w", "v").replace("b", "v").replace("ph", "f")
    w = re.sub(r"(ar)(c|d|t|s|e|h)+$", r"\1", w)
    w = re.sub(r"(?<=[a-z])[dtsehx]+$", "", w) if not w.endswith("ar") else w
    return w


def says_name(text: str, names: list[str], tolerance: int = 1) -> bool:
    """« Valdar » entendu, même mal transcrit (« valdare », « val dar », « baldar »,
    « val d'arc ») : par la forme écrite, puis par la forme sonore."""
    t = _norm(text)
    words = re.findall(r"[a-z]+", t)
    pairs = [a + b for a, b in zip(words, words[1:], strict=False)]
    triples = ["".join(words[i:i + 3]) for i in range(len(words) - 2)]
    cands = words + pairs + triples
    for name in names:
        n = _norm(name).replace(" ", "")
        if n in t.replace(" ", ""):
            return True
        ns = _sound(n)
        for w in cands:
            if abs(len(w) - len(n)) > 3:
                continue
            if _lev(n, w) <= tolerance or _lev(ns, _sound(w)) <= tolerance:
                return True
    return False


class SileroVAD:
    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=True)

    def is_speech(self, frame: np.ndarray) -> float:
        import torch

        x = torch.from_numpy(frame.astype(np.float32))
        return float(self.model(x, SR).item())


class WhisperSTT:
    def __init__(self, model: str, device: str = "cpu", compute_type: str = "int8",
                 language: str = "fr", beam_size: int = 1, hint: str = ""):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size
        self.hint = hint      # un nom propre que whisper ne connaît pas : on le lui souffle

    def transcribe(self, segment: np.ndarray) -> Transcript:
        extra: dict[str, Any] = {}
        if self.hint:
            extra = {"initial_prompt": f"{self.hint}, tu m'entends ?", "hotwords": self.hint}
        try:
            segs, _ = self.model.transcribe(segment.astype(np.float32),
                                            language=self.language, beam_size=self.beam_size,
                                            condition_on_previous_text=False,
                                            vad_filter=False, **extra)
        except TypeError:          # faster-whisper trop ancien pour « hotwords »
            extra.pop("hotwords", None)
            segs, _ = self.model.transcribe(segment.astype(np.float32),
                                            language=self.language, beam_size=self.beam_size,
                                            condition_on_previous_text=False,
                                            vad_filter=False, **extra)
        texts, probs, nospeech = [], [], []
        for s in segs:
            if s.text.strip():
                texts.append(s.text.strip())
            probs.append(float(getattr(s, "avg_logprob", -1.0)))
            nospeech.append(float(getattr(s, "no_speech_prob", 0.0)))
        return Transcript(" ".join(texts), sum(probs) / len(probs) if probs else -99.0,
                          max(nospeech) if nospeech else 1.0)


class TranscriptWake:
    def __init__(self, stt: Any, names: list[str], head_seconds: float = 4.0):
        self.stt = stt
        self.names = names
        self.head = int(head_seconds * SR)
        self.peek = False          # diagnostic (--debug) : montrer ce que l'éveil a compris
        self.last_text: str | None = None

    def heard(self, segment: np.ndarray) -> bool:
        text = self.stt.transcribe(segment[: self.head]).text
        found = says_name(text, self.names)
        # Rien n'est gardé de ce qui ne s'adressait pas à Valdar, sauf à l'écran en mode
        # diagnostic, demandé par Olivier (en mémoire vive, jamais sur disque).
        self.last_text = text if self.peek else None
        del text
        return found


class OpenWakeWordWake:
    def __init__(self, model_path: str, threshold: float = 0.5):
        from openwakeword.model import Model

        self.model = Model(wakeword_models=[model_path], inference_framework="onnx")
        self.threshold = threshold

    def heard(self, segment: np.ndarray) -> bool:
        pcm = (np.clip(segment, -1, 1) * 32767).astype(np.int16)
        self.model.reset()
        best = 0.0
        for i in range(0, len(pcm) - 1280 + 1, 1280):       # trames de 80 ms
            scores = self.model.predict(pcm[i:i + 1280])
            best = max([best, *scores.values()])
        return best >= self.threshold


def build(cfg: EarsConfig, model_path: Any, llm: Any = None) -> tuple[Any, Any, Any]:
    """(vad, wake, stt) réels. Lève une exception claire si un morceau manque.

    `stt_backend` : « gemma » (oreilles natives du cerveau, rien de plus en mémoire) ou
    « whisper » (faster-whisper sur le CPU)."""
    vad = SileroVAD()
    if cfg.stt_backend == "gemma":
        from valdar.ears.gemma import GemmaSTT, check

        if llm is None:
            raise RuntimeError("oreilles Gemma : pas de modèle de langage")
        problem = check(llm)
        if problem:
            raise RuntimeError("Gemma n'entend pas (Ollama trop ancien pour l'audio de Gemma 4 ?"
                               f" il faut la 0.33.3 ou plus) : {problem}. Repli possible : "
                               "ears.stt_backend: whisper")
        stt: Any = GemmaSTT(llm)
    else:
        stt = WhisperSTT(cfg.stt_model, cfg.stt_device, cfg.stt_compute_type)
    wake_file = model_path(cfg.wake_model) if cfg.wake_model else None
    if wake_file is not None and wake_file.is_file():
        wake: Any = OpenWakeWordWake(str(wake_file), cfg.wake_threshold)
    elif cfg.transcript_wake:
        wake = TranscriptWake(WhisperSTT(cfg.wake_stt_model, "cpu", "int8",
                                         beam_size=2, hint=cfg.names[0]), cfg.names)
    else:
        raise RuntimeError("aucun mot d'éveil : entraîne le modèle « Valdar » ou active "
                           "ears.transcript_wake")
    return vad, wake, stt
