"""Voix de Valdar : celle de RAUB (XTTS v2 + chaîne « megatron »), avenant 3 §5."""
from __future__ import annotations

from valdar.config.loader import ValdarConfig
from valdar.voice.character import apply, speakable, to_int16
from valdar.voice.speaker import NullSink, SoundDeviceSink, Speaker
from valdar.voice.tts import FakeTTS, TTSError, XttsBackend, split_sentences


def make_tts(cfg: ValdarConfig) -> XttsBackend:
    v = cfg.voice
    if v.backend != "xtts":
        raise TTSError(f"backend de voix inconnu : {v.backend}")
    return XttsBackend(cfg.repo_path(v.xtts.model_dir), cfg.repo_path(v.xtts.reference),
                       language=v.xtts.language, device=v.xtts.device)


__all__ = ["FakeTTS", "NullSink", "SoundDeviceSink", "Speaker", "TTSError", "XttsBackend",
           "apply", "make_tts", "speakable", "split_sentences", "to_int16"]
