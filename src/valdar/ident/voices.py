"""Reconnaître qui parle à la voix (phase 4).

Un petit réseau (CAM++ entraîné sur VoxCeleb, ONNX, sur le processeur, via sherpa-onnx)
transforme quelques secondes de voix en un vecteur : deux phrases de la même personne
donnent des vecteurs proches, quel que soit ce qui est dit. On compare avec les empreintes
enregistrées pour chaque membre du foyer.

- **Enrôlement** : la personne parle ~15 s (n'importe quoi) ; on découpe en morceaux de 3 s,
  une empreinte par morceau.
- **Reconnaissance** : la meilleure personne doit dépasser le seuil ET devancer la deuxième
  d'un écart minimum (sinon « je ne suis pas sûr »). Les enfants d'une même famille ont des
  voix proches : l'écart compte autant que le seuil.
- **Apprentissage** : quand il est très sûr, il garde l'empreinte (la voix change avec la
  fatigue, le rhume, la pièce) ; les plus anciennes apprises sont oubliées.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from valdar.config.loader import IdentConfig
from valdar.ears.gate import SR
from valdar.ident.store import Prints

KIND = "voix"


class Embedder(Protocol):
    def embed(self, samples: np.ndarray) -> np.ndarray: ...


class SherpaEmbedder:
    def __init__(self, model: str, threads: int = 2):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=threads,
                                                          debug=False, provider="cpu")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        st = self.ex.create_stream()
        st.accept_waveform(SR, np.asarray(samples, dtype=np.float32))
        st.input_finished()
        if not self.ex.is_ready(st):
            raise ValueError("trop court pour reconnaître la voix")
        return np.asarray(self.ex.compute(st), dtype=np.float32)


@dataclass
class Guess:
    person: str | None
    score: float
    margin: float
    second: str | None = None

    @property
    def sure(self) -> bool:
        return self.person is not None

    def to_dict(self) -> dict[str, Any]:
        return {"person": self.person, "score": round(self.score, 3),
                "margin": round(self.margin, 3), "second": self.second}


class VoiceID:
    def __init__(self, cfg: IdentConfig, prints: Prints, embedder: Embedder):
        self.cfg = cfg
        self.prints = prints
        self.embedder = embedder

    def _voiced(self, seg: np.ndarray) -> np.ndarray:
        """Garde les trames où il y a de la voix (les blancs brouillent l'empreinte)."""
        x = np.asarray(seg, dtype=np.float32).reshape(-1)
        hop = SR // 50
        n = len(x) // hop
        if n < 5:
            return x
        e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
        keep = e >= max(1e-4, 0.15 * float(np.percentile(e, 95)))
        return x[: n * hop].reshape(n, hop)[keep].reshape(-1)

    def identify(self, seg: np.ndarray) -> Guess:
        x = self._voiced(seg)
        if len(x) < self.cfg.voice_min_seconds * SR:
            return Guess(None, 0.0, 0.0)
        try:
            v = self.embedder.embed(x)
        except Exception:
            return Guess(None, 0.0, 0.0)
        ranked = self.prints.match(KIND, v)
        if not ranked:
            return Guess(None, 0.0, 0.0)
        best, score = ranked[0]
        second, s2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
        margin = score - s2
        ok = score >= self.cfg.voice_threshold and margin >= self.cfg.voice_margin
        if ok and score >= self.cfg.voice_threshold + self.cfg.adapt_above:
            self.prints.add(KIND, best, v, learned=True, cap=self.cfg.max_prints)
        return Guess(best if ok else None, score, margin, second)

    def confidence(self, g: Guess) -> float:
        """Confiance d'identité (0 à 1) pour les permissions : la voix seule ne monte jamais
        jusqu'aux outils élevés (0,9) ; il faut la voix et le visage d'accord."""
        if not g.sure:
            return 0.0
        span = max(0.05, 1.0 - self.cfg.voice_threshold)
        return round(min(0.85, 0.55 + 0.3 * (g.score - self.cfg.voice_threshold) / span
                         + 0.5 * g.margin), 3)

    def enroll(self, person: str, seg: np.ndarray, chunk_seconds: float = 3.0) -> int:
        """Ajoute les empreintes d'un enregistrement ; renvoie combien ont été gardées."""
        x = self._voiced(seg)
        step = int(chunk_seconds * SR)
        kept = 0
        for i in range(0, max(1, len(x) - step // 2), step):
            part = x[i:i + step]
            if len(part) < SR:
                continue
            try:
                self.prints.add(KIND, person, self.embedder.embed(part),
                                cap=self.cfg.max_prints)
                kept += 1
            except Exception:
                continue
        return kept


class Enrollment:
    """Enrôlement en cours : accumule la parole de la personne jusqu'à la durée voulue."""

    def __init__(self, person: str, seconds: float):
        self.person = person
        self.need = int(seconds * SR)
        self.parts: list[np.ndarray] = []

    @property
    def have(self) -> int:
        return sum(len(p) for p in self.parts)

    @property
    def progress(self) -> float:
        return min(1.0, self.have / self.need)

    def add(self, seg: np.ndarray) -> bool:
        """True quand c'est assez."""
        self.parts.append(np.asarray(seg, dtype=np.float32))
        return self.have >= self.need

    def audio(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
