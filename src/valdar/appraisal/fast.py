"""Évaluation rapide d'un message (voie basse, v2 §5.4) : mots-clés → stimuli.

Instantanée et sans modèle de langage. Elle peut se tromper : la voie haute (évaluation par
le LLM, phase suivante) confirmera ou corrigera.
"""
from __future__ import annotations

import re
import unicodedata

from valdar.config.loader import AppraisalFastConfig


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower().replace("’", "'"))
    return "".join(c for c in t if not unicodedata.combining(c))


class FastAppraisal:
    def __init__(self, cfg: AppraisalFastConfig):
        self.rules = [
            (r.stimulus, r.scale,
             [re.compile(r"(?<![a-z0-9])" + re.escape(normalize(w)) + r"(?![a-z0-9])")
              for w in r.words])
            for r in cfg.rules
        ]

    def appraise(self, text: str) -> list[tuple[str, float]]:
        """Renvoie [(stimulus, échelle)] : le plus fort par stimulus."""
        t = normalize(text)
        found: dict[str, float] = {}
        for stimulus, scale, patterns in self.rules:
            if any(p.search(t) for p in patterns):
                found[stimulus] = max(found.get(stimulus, 0.0), scale)
        return sorted(found.items())
