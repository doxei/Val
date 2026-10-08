"""Micro : capture continue (sounddevice), rééchantillonnée en 16 kHz mono, en mémoire vive.

Choix du micro comme RAUB : le K66 en priorité, en MME d'abord (mesuré stable sous Windows).
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from valdar.ears.gate import SR


def find_input(prefer: list[str]) -> tuple[int | None, str]:
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"]]
    for name in prefer:
        pool = [(i, d) for i, d in inputs if name.lower() in str(d["name"]).lower()]
        if pool:
            mme = [(i, d) for i, d in pool
                   if "mme" in str(sd.query_hostapis(d["hostapi"])["name"]).lower()]
            i, d = (mme or pool)[0]
            return i, str(d["name"])
    try:
        i = sd.default.device[0]
        return (i if i is not None and i >= 0 else None), "micro par défaut"
    except Exception:
        return None, "aucun micro"


class Mic:
    def __init__(self, on_audio: Callable[[np.ndarray], None], prefer: list[str]):
        self.on_audio = on_audio
        self.prefer = prefer
        self._stream: Any = None
        self.name = ""
        self._lock = threading.Lock()

    def start(self) -> str:
        import sounddevice as sd
        from scipy.signal import resample_poly

        idx, self.name = find_input(self.prefer)
        sr = int(sd.query_devices(idx, "input")["default_samplerate"]) if idx is not None \
            else SR

        def cb(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            x = indata[:, 0].astype(np.float32)
            if sr != SR:
                x = resample_poly(x, SR, sr).astype(np.float32)
            with self._lock:
                self.on_audio(x)

        self._stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                      device=idx, blocksize=int(sr * 0.1), callback=cb)
        self._stream.start()
        return self.name

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
