"""Synthèse vocale. Backend principal : XTTS v2 local, avec la voix clonée d'origine.

Même modèle, même référence, mêmes réglages que le code d'origine de la voix. Une seule
différence, sans effet sur le son : les latents de la voix sont calculés une fois au
chargement au lieu d'être recalculés à chaque phrase (l'ancien code passait par `synthesize`,
qui refait le clonage à chaque appel). Si cette voie échoue, on retombe sur l'appel d'origine,
à l'identique.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class TTSError(RuntimeError):
    pass


class TTSBackend(Protocol):
    sample_rate: int

    def load(self) -> None: ...
    def synthesize(self, text: str) -> np.ndarray: ...   # int16 mono


# --------------------------------------------------------------------------- phrases
_SENT = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, max_chars: int = 230) -> list[str]:
    """Découpe en phrases (XTTS limite à 273 caractères en français). Une phrase trop
    longue est recoupée aux virgules, puis aux espaces."""
    out: list[str] = []
    for sent in _SENT.split(text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        while len(sent) > max_chars:
            cut = sent.rfind(", ", 0, max_chars)
            if cut < max_chars // 3:
                cut = sent.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(sent[:cut + 1].strip())
            sent = sent[cut + 1:].strip()
        if sent:
            out.append(sent)
    return out


# --------------------------------------------------------------------------- XTTS
def _patch_transformers() -> None:
    """Même correctif qu'avant pour transformers 4.57 (coqui-tts teste torchcodec)."""
    import transformers.utils.import_utils as iu

    if not hasattr(iu, "is_torchcodec_available"):
        iu.is_torchcodec_available = lambda: False


def _default_loader(model_dir: Path, device: str) -> tuple[Any, Any]:
    _patch_transformers()
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise TTSError("XTTS demande la carte graphique (CUDA introuvable dans ce Python)")
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    cfg = XttsConfig()
    cfg.load_json(str(model_dir / "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=str(model_dir), eval=True)
    if device == "cuda":
        model.cuda()
    model.eval()
    return model, cfg


class XttsBackend:
    sample_rate = 24000
    _INFER_KEYS = ("temperature", "length_penalty", "repetition_penalty", "top_k", "top_p")

    def __init__(self, model_dir: Path, reference: Path | list[Path], language: str = "fr",
                 device: str = "cuda",
                 loader: Callable[[Path, str], tuple[Any, Any]] | None = None):
        self.model_dir = Path(model_dir)
        refs = reference if isinstance(reference, list) else [reference]
        self.references = [Path(r) for r in refs]
        self.reference = self.references[0]
        self.language = language
        self.device = device
        self._loader = loader or _default_loader
        self._lock = threading.Lock()
        self.model: Any = None
        self.cfg: Any = None
        self._voice: tuple[Any, Any] | None = None
        self._settings: dict[str, Any] = {}
        self.load_seconds: float | None = None
        self.mode = "non chargé"

    def check_files(self) -> list[str]:
        missing = []
        if not (self.model_dir / "config.json").is_file() or \
                not (self.model_dir / "model.pth").is_file():
            missing.append(f"modèle XTTS ({self.model_dir})")
        for ref in self.references:
            if not ref.is_file():
                missing.append(f"référence de la voix ({ref})")
        return missing

    def load(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            missing = self.check_files()
            if missing:
                raise TTSError("fichiers de voix absents : " + ", ".join(missing)
                               + " — lance tools\\valdar_voix.bat (il les reprend de "
                               "l'ancienne installation)")
            t0 = time.time()
            model, cfg = self._loader(self.model_dir, self.device)
            conf = getattr(model, "config", None) or cfg
            try:
                lat, emb = model.get_conditioning_latents(
                    audio_path=(str(self.reference) if len(self.references) == 1
                                else [str(r) for r in self.references]),
                    max_ref_length=_get(conf, "max_ref_len"),
                    gpt_cond_len=_get(conf, "gpt_cond_len"),
                    gpt_cond_chunk_len=_get(conf, "gpt_cond_chunk_len"),
                    sound_norm_refs=_get(conf, "sound_norm_refs"))
                self._voice = (lat, emb)
                self._settings = {k: _get(conf, k) for k in self._INFER_KEYS}
                self.mode = "latents calculés une fois"
            except Exception:
                self._voice = None
                self.mode = "appel d'origine à l'identique"
            self.model, self.cfg = model, cfg
            self.load_seconds = time.time() - t0

    def synthesize(self, text: str) -> np.ndarray:
        self.load()
        out = None
        if self._voice is not None:
            try:
                out = self._infer(lambda: self.model.inference(
                    text, self.language, self._voice[0], self._voice[1], **self._settings))
            except (TypeError, AttributeError):
                self._voice = None
                self.mode = "appel d'origine à l'identique"
        if out is None:
            out = self._infer(lambda: self.model.synthesize(
                text, self.cfg, language=self.language,
                speaker_wav=(str(self.reference) if len(self.references) == 1
                             else [str(r) for r in self.references])))
        wav = out["wav"] if isinstance(out, dict) else out[0]
        wav = np.asarray(wav).astype(np.float32)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        peak = peak or 1.0
        return (wav * (32767.0 / peak)).astype(np.int16)

    def stream(self, text: str, chunk_size: int = 20) -> Iterator[np.ndarray]:
        """La phrase morceau par morceau (float32 dans [-1, 1]), dès qu'XTTS les produit.

        Repli : si le flux n'existe pas (version de coqui-tts, latents absents), la phrase
        entière en un seul morceau."""
        self.load()
        if self._voice is None or not hasattr(self.model, "inference_stream"):
            yield self.synthesize(text).astype(np.float32) / 32768.0
            return
        try:
            import torch

            ctx: Any = torch.inference_mode()
        except ImportError:
            ctx = _Null()
        with ctx:
            gen = self.model.inference_stream(
                text, self.language, self._voice[0], self._voice[1],
                stream_chunk_size=chunk_size, enable_text_splitting=False,
                **self._settings)
            for chunk in gen:
                arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else chunk
                yield np.asarray(arr, dtype=np.float32).reshape(-1)

    def _infer(self, fn: Callable[[], Any]) -> Any:
        try:
            import torch
        except ImportError:   # tests : modèle simulé
            return fn()
        with torch.inference_mode():
            return fn()


def _get(conf: Any, key: str) -> Any:
    try:
        return conf[key]
    except (KeyError, TypeError, IndexError):
        return getattr(conf, key)


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


class FakeTTS:
    """Pour les tests : un bip par phrase, durée proportionnelle au texte."""
    sample_rate = 24000

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.texts: list[str] = []

    def load(self) -> None:
        if self.fail:
            raise TTSError("voix simulée en panne")

    def synthesize(self, text: str) -> np.ndarray:
        self.load()
        self.texts.append(text)
        n = max(240, 60 * len(text))
        t = np.arange(n) / self.sample_rate
        return (8000 * np.sin(2 * np.pi * 180 * t)).astype(np.int16)

    def stream(self, text: str, chunk_size: int = 20) -> Iterator[np.ndarray]:
        wav = self.synthesize(text).astype(np.float32) / 32768.0
        step = max(1, len(wav) // 3)
        for i in range(0, len(wav), step):
            yield wav[i:i + step]
