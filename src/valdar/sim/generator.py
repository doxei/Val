"""Générateur d'événements pseudo-aléatoires (reproductible par graine).

Intervalles courants entre min_interval et max_interval, longues absences occasionnelles,
et pas d'événement pendant le sommeil si night_events est faux.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from valdar.config.loader import CircadianSpec, SimConfig
from valdar.heart.dynamics import in_window, parse_hours
from valdar.heart.heart import local_hour


class EventGenerator:
    def __init__(self, sim: SimConfig, circadian: CircadianSpec, seed: int):
        self.sim = sim
        self.circ = circadian
        self.rng = random.Random(seed)
        self.count: dict[str, int] = {}
        self.next_at = 0.0
        self.next_type = sim.event_types[0]

    def schedule(self, now: float) -> None:
        s = self.sim
        if self.rng.random() < s.absence_probability:
            gap = self.rng.uniform(s.absence_min, s.absence_max)
        else:
            gap = self.rng.uniform(s.min_interval, s.max_interval)
        t = now + gap
        if not s.night_events and not in_window(local_hour(t), self.circ.awake_start,
                                                 self.circ.sleep_start):
            t = self._next_wake(t) + self.rng.uniform(s.min_interval, s.max_interval)
        self.next_at = t
        self.next_type = self.rng.choices(s.event_types, weights=s.weights, k=1)[0]

    def _next_wake(self, t: float) -> float:
        d = datetime.fromtimestamp(t)
        h = parse_hours(self.circ.awake_start)
        wake = d.replace(hour=int(h), minute=int(round((h % 1) * 60)), second=0, microsecond=0)
        if wake <= d:
            wake += timedelta(days=1)
        return wake.timestamp()

    def due(self, now: float) -> bool:
        return now >= self.next_at

    def pop(self) -> str:
        ev = self.next_type
        self.count[ev] = self.count.get(ev, 0) + 1
        self.schedule(self.next_at)
        return ev
