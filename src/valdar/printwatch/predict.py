"""Prédiction d'échec à partir des scores image par image (logique d'Obico, reprise fidèlement).

Constantes vérifiées dans le code d'Obico (`backend/lib/prediction.py`, `settings.py`,
`failure_detection.py`, licence AGPL-3.0 ; voir docs/connaissances/detection_etat_art.md) :
- score d'une image = somme des confiances des détections ≥ 0,08 ;
- lissage exponentiel EWM (span 12) ;
- moyennes glissantes courte (310 images) et longue (7 200 images, **sur la vie de
  l'imprimante** : c'est la ligne de base propre à la machine, qui annule un faux positif
  permanent — câble, reflet, jupe) ;
- 30 premières images d'une impression jamais en échec ;
- ajusté = (EWM − moyenne longue) × sensibilité / facteur ;
  < 0,38 → non ; > 0,78 → oui ; sinon oui si ajusté > (courte − longue) × 3,8 ;
- alerte avec facteur 1, pause avec facteur 1,75.

Ce que PrintOS faisait de travers (et qui est corrigé ici) : seuil de détection à 0,25 au lieu
de 0,08, aucune ligne de base, aucune trace des scores. Ici chaque image laisse une trace.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from valdar.config.loader import PredictorConfig


@dataclass
class RollingMean:
    window: int
    values: deque = field(default_factory=deque)
    total: float = 0.0

    def push(self, x: float) -> None:
        self.values.append(x)
        self.total += x
        if len(self.values) > self.window:
            self.total -= self.values.popleft()

    @property
    def mean(self) -> float:
        return self.total / len(self.values) if self.values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window, "values": list(self.values)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RollingMean:
        r = cls(int(d["window"]))
        for v in d.get("values", [])[-r.window:]:
            r.push(float(v))
        return r


@dataclass
class Verdict:
    score: float            # score brut de l'image
    ewm: float
    adjusted: float         # (EWM − ligne de base) × sensibilité
    alert: bool             # niveau « alerte »
    pause: bool             # niveau « pause » (signal nettement plus fort)
    warming_up: bool        # encore dans les premières images de l'impression


class FailurePredictor:
    def __init__(self, cfg: PredictorConfig, lifetime: dict[str, Any] | None = None):
        self.cfg = cfg
        self.long = RollingMean.from_dict(lifetime) if lifetime else \
            RollingMean(cfg.rolling_win_long)
        self.reset_print()

    def reset_print(self) -> None:
        self.ewm: float | None = None
        self.short = RollingMean(self.cfg.rolling_win_short)
        self.frames = 0

    @staticmethod
    def frame_score(confidences: list[float], thresh: float) -> float:
        return float(sum(c for c in confidences if c >= thresh))

    def push(self, score: float) -> Verdict:
        c = self.cfg
        alpha = 2.0 / (c.ewm_span + 1.0)
        self.ewm = score if self.ewm is None else alpha * score + (1 - alpha) * self.ewm
        self.short.push(self.ewm)
        self.long.push(self.ewm)
        self.frames += 1
        warming = self.frames <= c.init_safe_frames
        adjusted = (self.ewm - self.long.mean) * c.sensitivity
        alert = not warming and self._failing(adjusted, 1.0)
        pause = not warming and self._failing(adjusted, c.escalating_factor)
        return Verdict(score, self.ewm, adjusted, alert, pause, warming)

    def _failing(self, adjusted: float, factor: float) -> bool:
        c = self.cfg
        a = adjusted / factor
        if a < c.threshold_low:
            return False
        if a > c.threshold_high:
            return True
        return a > (self.short.mean - self.long.mean) * c.short_multiple

    def lifetime_state(self) -> dict[str, Any]:
        """La ligne de base de l'imprimante, à sauvegarder entre deux impressions."""
        return self.long.to_dict()
