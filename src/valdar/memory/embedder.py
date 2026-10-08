"""Vrai modèle de plongement (embeddings) pour comprendre le sens, par Ollama.

Les vecteurs maison (`vectors.embed`, n-grammes hachés) retrouvent les mots, pas le sens :
« la vitre du bed » ne retrouve pas « le plateau en verre ». Un modèle de plongement
multilingue (bge-m3 par défaut, bon en français) comble ce manque.

- Il tourne sur le processeur (`num_gpu: 0`) : la carte graphique est déjà pleine.
- S'il manque ou ne répond pas, Valdar retombe sur les vecteurs maison et réessaie plus tard :
  rien ne casse.
- Les vecteurs sont rangés avec la signature du modèle : changer de modèle les refait.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np

from valdar.config.loader import EmbeddingsConfig


class Embedder:
    def __init__(self, cfg: EmbeddingsConfig, url: str, client: Any = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.url = url.rstrip("/")
        self.client = client
        self.clock = clock
        self._down_until = 0.0
        self.last_error = ""

    @property
    def signature(self) -> str:
        return f"{self.cfg.backend}:{self.cfg.model}"

    def available(self) -> bool:
        return self.cfg.backend == "ollama" and self.clock() >= self._down_until

    def embed_many(self, texts: list[str]) -> np.ndarray | None:
        """Vecteurs normés (une ligne par texte), ou None si le modèle est indisponible."""
        if not texts or not self.available():
            return None
        body: dict[str, Any] = {"model": self.cfg.model,
                                "input": [t[:self.cfg.max_chars] for t in texts],
                                "keep_alive": self.cfg.keep_alive}
        if self.cfg.cpu:
            body["options"] = {"num_gpu": 0}
        try:
            post = self.client.post if self.client is not None else httpx.post
            r = post(self.url + "/api/embed", json=body, timeout=self.cfg.timeout_seconds)
            r.raise_for_status()
            m = np.asarray(r.json()["embeddings"], dtype=np.float32)
            if m.ndim != 2 or len(m) != len(texts):
                raise ValueError("réponse inattendue")
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            self.last_error = str(exc)[:200]
            self._down_until = self.clock() + self.cfg.retry_seconds
            return None
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def embed_one(self, text: str) -> np.ndarray | None:
        m = self.embed_many([text])
        return None if m is None else m[0]


def pack(v: np.ndarray) -> bytes:
    return v.astype(np.float16).tobytes()


def unpack(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def deep_matrix(cells: list[tuple[bytes | None, str | None]], embedder: Any
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(vecteurs du vrai modèle, masque des lignes qui en ont un à la bonne signature)."""
    n = len(cells)
    sig = embedder.signature if embedder is not None else None
    vecs = [unpack(b) if b is not None and s == sig else None for b, s in cells]
    dim = next((len(v) for v in vecs if v is not None), 0)
    mat = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for i, v in enumerate(vecs):
        if v is not None and len(v) == dim:
            mat[i], mask[i] = v, True
    return mat, mask
