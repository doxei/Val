"""Intéroception : la charge cognitive (avenant 5 §3).

La fatigue (`energy`) et le stress (`cortisol`) sont déjà des variables du cœur. Il manquait
la **charge** : à quel point la tête est pleine. Elle est mesurée, pas inventée :

- les **flux actifs** : répondre, penser en fond, parler de lui-même, regarder une image ;
- la **lenteur** de Gemma : secondes par jeton produit, comparées à sa médiane récente ;
- la **mémoire de la carte graphique** (lue par les nocicepteurs).

Trois niveaux, avec hystérésis (il faut redescendre un peu sous le seuil pour en sortir) :
0 normal ; 1 chargé : la pensée de fond s'arrête ; 2 saturé : réponses courtes, et Valdar
peut le dire. Les seuils viennent de la configuration, bornés : l'habituation ne peut pas les
pousser au-delà.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from valdar.config.loader import InteroceptionConfig
from valdar.llm.backend import ChatResult, GenParams, LLMBackend

LEVELS = ("normal", "chargé", "saturé")


class Load:
    def __init__(self, cfg: InteroceptionConfig):
        self.cfg = cfg
        self._lat: deque[float] = deque(maxlen=cfg.latency_window)
        self._recent: float | None = None
        self._lock = threading.Lock()
        self.level = 0
        self.value = 0.0
        self.parts: dict[str, float] = {}

    # ------------------------------------------------------------- mesures
    def record_latency(self, seconds: float, tokens: int | None) -> None:
        per = seconds / max(1, tokens or 1)
        with self._lock:
            self._lat.append(per)
            a = 0.4
            self._recent = per if self._recent is None else (1 - a) * self._recent + a * per

    def slowness(self) -> float:
        """0 à la vitesse habituelle, 1 quand Gemma est `slow_ratio` fois plus lent."""
        with self._lock:
            if len(self._lat) < 5 or self._recent is None:
                return 0.0
            med = statistics.median(self._lat)
            recent = self._recent
        if med <= 0:
            return 0.0
        ratio = recent / med
        return max(0.0, min(1.0, (ratio - 1.0) / max(0.1, self.cfg.slow_ratio - 1.0)))

    def update(self, streams: int, gpu_mem: float | None) -> int:
        """Recalcule la charge ; renvoie le niveau (0, 1, 2)."""
        c = self.cfg
        parts = {
            "flux": min(1.0, streams / max(1, c.max_streams)),
            "lenteur": self.slowness(),
            "carte": 0.0 if gpu_mem is None else
            max(0.0, min(1.0, (gpu_mem - c.gpu_mem_from) / max(0.01, 1.0 - c.gpu_mem_from))),
        }
        total = sum(c.weights.get(k, 0.0) for k in parts) or 1.0
        value = sum(c.weights.get(k, 0.0) * v for k, v in parts.items()) / total
        level = self.level
        up = (c.busy_at, c.saturated_at)
        while level < 2 and value >= up[level]:
            level += 1
        while level > 0 and value < up[level - 1] - c.hysteresis:
            level -= 1
        self.value, self.parts, self.level = value, parts, level
        return level

    @property
    def label(self) -> str:
        return LEVELS[self.level]

    def line(self) -> str:
        if self.level == 0:
            return ""
        detail = ", ".join(f"{k} {v:.0%}" for k, v in self.parts.items() if v >= 0.3)
        tail = " — réponds court, et dis-le si on t'en demande trop" if self.level == 2 else ""
        return f"ta charge mentale : {self.label} ({detail}){tail}"


class TimedLLM:
    """Enveloppe du modèle de langage qui mesure sa lenteur pour l'intéroception."""

    def __init__(self, inner: LLMBackend, load: Load, clock: Callable[[], float] = time.monotonic):
        self.inner = inner
        self.load = load
        self.clock = clock

    def chat(self, messages: list[dict[str, Any]], system: str = "",
             tools: list[dict[str, Any]] | None = None,
             params: GenParams | None = None, **kw: Any) -> ChatResult:
        t0 = self.clock()
        res = self.inner.chat(messages, system=system, tools=tools, params=params, **kw)
        usage = getattr(res, "usage", None)
        tokens = usage.get("eval_count") if isinstance(usage, dict) else None
        self.load.record_latency(self.clock() - t0, tokens if isinstance(tokens, int) else None)
        return res

    def available(self) -> bool:
        return self.inner.available()

    def __getattr__(self, name: str) -> Any:     # le reste (modèle, réglages) passe tel quel
        return getattr(self.inner, name)
