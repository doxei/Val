"""Représentation des textes pour la mémoire (avenant 3 §2.7, v1).

N-grammes de caractères (3 à 5) des mots normalisés, hachés sur `dim` dimensions avec un
signe : pas de modèle, déterministe, robuste aux conjugaisons et aux fautes de dictée.
"""
from __future__ import annotations

import zlib

import numpy as np

from valdar.memory.facts import _STOP, _norm

DIM = 512


def _hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def embed(text: str, dim: int = DIM, max_chars: int = 2000) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for word in _norm(text[:max_chars]).split():
        if len(word) < 3 or word in _STOP:
            continue
        h = _hash(word)
        v[h % dim] += 1.5 if (h >> 16) & 1 else -1.5
        w = f" {word} "
        for n in (3, 4, 5):
            for i in range(len(w) - n + 1):
                g = _hash(w[i:i + n])
                v[g % dim] += 1.0 if (g >> 16) & 1 else -1.0
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def cosine_rows(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosinus de chaque ligne de m avec q (lignes supposées normées ou nulles)."""
    if m.size == 0:
        return np.zeros(0, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn == 0:
        return np.zeros(len(m), dtype=np.float32)
    norms = np.linalg.norm(m, axis=1)
    norms[norms == 0] = 1.0
    return (m @ q) / (norms * qn)
