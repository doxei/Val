"""Oreilles natives : Gemma 4 12B transcrit lui-même (choix d'Olivier, 8 octobre 2026).

Gemma 4 (E2B, E4B, 12B) est entraîné à la reconnaissance de la parole : clips mono 16 kHz,
30 s au plus par appel. Ollama accepte l'audio pour Gemma 4 par le même champ que les images
(WAV encodé en base64). Avantages : aucun modèle de plus en mémoire, ni sur la carte (Gemma est
déjà chargé), ni sur le processeur.

Pas de probabilités comme whisper : on demande une transcription exacte, « [silence] » quand il
n'y a pas de parole, et les filtres anti-hallucinations du portier font le reste.
"""
from __future__ import annotations

import base64
import io
import re
import wave

import numpy as np

from valdar.ears.gate import SR, Transcript
from valdar.llm.backend import GenParams, LLMBackend, LLMError

PROMPT = ("Transcris mot pour mot ce qui est dit dans cet audio, en français, sans rien "
          "ajouter, sans traduire, sans commenter. S'il n'y a pas de parole compréhensible, "
          "réponds exactement : [silence]")
MAX_SECONDS = 30.0
_SILENCE = re.compile(r"^\W*\[?\s*silence\s*\]?\W*$", re.I)


def to_wav(segment: np.ndarray) -> bytes:
    pcm = (np.clip(segment.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _clean(text: str) -> str:
    text = text.strip().strip('"«»“”').strip()
    for prefix in ("Transcription :", "Transcription:", "Voici la transcription :"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text


class GemmaSTT:
    """Transcription par le cerveau lui-même (même modèle, mêmes réglages qu'Ollama)."""

    def __init__(self, llm: LLMBackend, max_tokens: int = 200):
        self.llm = llm
        self.max_tokens = max_tokens
        self.last_error: str | None = None

    def transcribe(self, segment: np.ndarray) -> Transcript:
        texts = []
        step = int(MAX_SECONDS * SR)
        for i in range(0, max(1, len(segment)), step):
            texts.append(self._one(segment[i:i + step]))
        text = " ".join(t for t in texts if t)
        if not text:
            return Transcript("", -99.0, 1.0)      # rien entendu : le portier l'écarte
        return Transcript(text, 0.0, 0.0)

    def _one(self, chunk: np.ndarray) -> str:
        if len(chunk) < SR // 4:
            return ""
        msg = {"role": "user", "content": PROMPT,
               "images": [base64.b64encode(to_wav(chunk)).decode("ascii")]}
        try:
            res = self.llm.chat([msg], system="", tools=None,
                                params=GenParams(temperature=0.0, max_tokens=self.max_tokens))
        except LLMError as exc:
            self.last_error = str(exc)
            return ""
        self.last_error = None
        text = _clean(res.content or "")
        return "" if _SILENCE.match(text) else text


def check(llm: LLMBackend) -> str | None:
    """Vérifie qu'Ollama accepte l'audio pour ce modèle. Renvoie None si ça marche, sinon la
    raison (version d'Ollama trop ancienne, modèle sans oreilles…)."""
    t = np.arange(SR) / SR
    tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    msg = {"role": "user", "content": PROMPT,
           "images": [base64.b64encode(to_wav(tone)).decode("ascii")]}
    try:
        llm.chat([msg], system="", tools=None, params=GenParams(temperature=0.0, max_tokens=20))
    except LLMError as exc:
        return str(exc)
    return None
