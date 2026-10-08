"""Valdar parle : synthèse phrase par phrase, la première est jouée pendant que la suivante
se calcule. Ne bloque jamais la conversation : `say()` rend la main tout de suite.
"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from valdar.config.loader import VoiceConfig
from valdar.voice.character import apply, speakable
from valdar.voice.tts import TTSBackend, split_sentences


class Sink(Protocol):
    def play(self, audio: np.ndarray, sr: int) -> None: ...   # bloquant, float [-1, 1]
    def stop(self) -> None: ...


class SoundDeviceSink:
    """Sortie audio par défaut de Windows (comme RAUB)."""

    def play(self, audio: np.ndarray, sr: int) -> None:
        import sounddevice as sd

        sd.play(audio.astype(np.float32), sr)
        sd.wait()

    def stop(self) -> None:
        import sounddevice as sd

        sd.stop()


class NullSink:
    def __init__(self) -> None:
        self.played: list[tuple[int, int]] = []

    def play(self, audio: np.ndarray, sr: int) -> None:
        self.played.append((len(audio), sr))

    def stop(self) -> None:
        pass


@dataclass
class SpeechStats:
    first_audio_seconds: float | None = None      # du texte reçu au premier son
    synth_seconds: list[float] = field(default_factory=list)
    audio_seconds: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_STOP = object()


class Speaker:
    def __init__(self, cfg: VoiceConfig, tts: TTSBackend, sink: Sink | None = None):
        self.cfg = cfg
        self.tts = tts
        self.sink = sink or SoundDeviceSink()
        self._texts: queue.Queue = queue.Queue()
        self._in_play = False
        self._play_until = 0.0
        self._audio: queue.Queue = queue.Queue(maxsize=4)
        self._generation = 0
        self._pending = 0
        self._count = threading.Lock()
        self._threads: list[threading.Thread] = []
        self.stats = SpeechStats()
        self.ready = threading.Event()
        self.load_error: str | None = None

    # ------------------------------------------------------------ cycle de vie
    def start(self, preload: bool = True) -> Speaker:
        for name, fn in (("valdar-voix-synthese", self._synth_loop),
                         ("valdar-voix-lecture", self._play_loop)):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        if preload:
            threading.Thread(target=self._preload, name="valdar-voix-chargement",
                             daemon=True).start()
        return self

    def _preload(self) -> None:
        try:
            apply(np.zeros(64, np.int16), self.tts.sample_rate, self.cfg.character)  # scipy
            self.tts.load()
        except Exception as exc:  # la voix ne doit jamais faire tomber Valdar
            self.load_error = str(exc)
        finally:
            self.ready.set()

    def close(self, timeout: float = 2.0) -> None:
        self.interrupt()
        self._texts.put(_STOP)
        with contextlib.suppress(queue.Full):
            self._audio.put(_STOP, timeout=timeout)
        for t in self._threads:
            t.join(timeout=timeout)

    # ------------------------------------------------------------------ usage
    def say(self, text: str) -> None:
        text = speakable(text)
        if text:
            with self._count:
                self._pending += 1
            self._texts.put((self._generation, time.time(), text))

    def interrupt(self) -> None:
        """Coupe la parole en cours et oublie ce qui restait à dire."""
        self._generation += 1
        with self._count:
            self._pending = 0
        for q in (self._texts, self._audio):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        self.sink.stop()

    def playing(self) -> bool:
        """Du son sort réellement des haut-parleurs (ou vient d'en sortir : la pièce résonne
        encore 0,3 s). Les oreilles s'en servent pour reconnaître l'écho de sa voix."""
        return self._in_play or time.time() < self._play_until

    def speaking(self) -> bool:
        with self._count:
            return self._pending > 0

    def wait_done(self, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if not self.speaking():
                return True
            time.sleep(0.02)
        return False

    # -------------------------------------------------------------- threads
    def _synth_loop(self) -> None:
        while True:
            item = self._texts.get()
            if item is _STOP:
                return
            gen, t_in, text = item
            first = True
            for sent in split_sentences(text, self.cfg.max_sentence_chars):
                if gen != self._generation:
                    break
                t0 = time.time()
                try:
                    raw = self.tts.synthesize(sent)
                    audio = apply(raw, self.tts.sample_rate, self.cfg.character)
                except Exception as exc:   # TTSError, CUDA, etc. : Valdar reste debout
                    self.stats.errors.append(str(exc))
                    break
                self.stats.synth_seconds.append(time.time() - t0)
                self.stats.audio_seconds.append(len(audio) / self.tts.sample_rate)
                self._audio.put((gen, t_in if first else None, audio))
                first = False
            self._audio.put((gen, None, None))   # fin de ce texte

    def _play_loop(self) -> None:
        while True:
            item = self._audio.get()
            if item is _STOP:
                return
            gen, t_in, audio = item
            if audio is None:
                if gen == self._generation:   # un texte interrompu ne compte plus
                    with self._count:
                        self._pending = max(0, self._pending - 1)
                continue
            if gen != self._generation:
                continue
            if t_in is not None:
                self.stats.first_audio_seconds = time.time() - t_in
            self._in_play = True
            try:
                self.sink.play(audio, self.tts.sample_rate)
            except Exception as exc:
                self.stats.errors.append(f"lecture : {exc}")
            finally:
                self._in_play = False
                self._play_until = time.time() + 0.3
