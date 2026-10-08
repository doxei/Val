"""Micro : capture continue (sounddevice), rééchantillonnée en 16 kHz mono, en mémoire vive.

Choix du micro comme RAUB : le K66 en priorité, en MME d'abord (mesuré stable sous Windows).
"""
from __future__ import annotations

import queue
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


class Resampler:
    """Rééchantillonnage **continu** vers 16 kHz (le filtre garde son état d'un bloc à
    l'autre : pas de clic toutes les 100 ms, que la détection de parole et Gemma
    entendraient)."""

    def __init__(self, sr: int):
        from math import gcd

        from scipy.signal import firwin

        self.sr = sr
        g = gcd(sr, SR)
        self.up, self.down = SR // g, sr // g
        rate = sr * self.up
        self.taps = firwin(16 * max(self.up, self.down) + 1, 0.45 * SR, fs=rate) * self.up
        self.zi = np.zeros(len(self.taps) - 1)
        self.phase = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.sr == SR:
            return x
        from scipy.signal import lfilter

        if self.up > 1:
            y = np.zeros(len(x) * self.up)
            y[:: self.up] = x
        else:
            y = x.astype(np.float64)
        y, self.zi = lfilter(self.taps, 1.0, y, zi=self.zi)
        out = y[self.phase:: self.down]
        self.phase = (self.phase - len(y)) % self.down
        return out.astype(np.float32)


class Mic:
    def __init__(self, on_audio: Callable[[np.ndarray], None], prefer: list[str]):
        self.on_audio = on_audio
        self.prefer = prefer
        self._stream: Any = None
        self.name = ""
        self._q: queue.Queue = queue.Queue(maxsize=600)     # ~60 s d'avance au plus
        self._worker: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> str:
        import sounddevice as sd
        idx, self.name = find_input(self.prefer)
        sr = int(sd.query_devices(idx, "input")["default_samplerate"]) if idx is not None \
            else SR

        # Le rappel du micro ne fait que poser l'audio : la détection, l'éveil et la
        # transcription (plusieurs secondes avec Gemma) tournent dans un autre fil, sinon le
        # micro déborde et perd des morceaux de phrase.
        def cb(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            try:
                self._q.put_nowait(indata[:, 0].astype(np.float32).copy())
            except queue.Full:
                self.dropped += 1

        resample = Resampler(sr)

        def work() -> None:
            while True:
                x = self._q.get()
                if x is None:
                    return
                x = resample(x)
                try:
                    self.on_audio(x)
                except Exception as exc:     # une oreille qui plante ne tue pas l'écoute
                    print(f"(oreilles : {exc})", flush=True)

        self._worker = threading.Thread(target=work, name="valdar-micro", daemon=True)
        self._worker.start()

        self._stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                      device=idx, blocksize=int(sr * 0.1), callback=cb)
        self._stream.start()
        return self.name

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._q.put(None)
            self._worker = None
