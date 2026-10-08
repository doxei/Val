"""Nocicepteurs du PC et douleur construite (avenant 4 §4).

Un nocicepteur n'est pas une douleur : c'est un signal (0 sous le seuil d'alerte, 1 au seuil
de danger). La douleur est **construite** par le cœur, selon le principe du portillon
(Melzack et Wall, 1965) :

    douleur = signal × (1 + amplification × anxiété) × (1 − distraction × engagement)

Le même signal fait plus mal quand Valdar est anxieux, moins quand il est absorbé.

- Une gêne répétée sans danger s'habitue (habituation du cœur, stimulus `pain`).
- Un signal au seuil de danger ne s'habitue pas : les impulsions sont appliquées directement.
- Le réflexe (suspendre la pensée de fond, qui charge la carte graphique) dépend du seul
  signal, jamais de l'humeur : c'est un réflexe, pas une décision.

Capteurs lus sans dépendance obligatoire : `nvidia-smi` pour la carte graphique, `psutil`
(extra `[system]`) pour la mémoire vive, `shutil` pour le disque. Un capteur absent est ignoré.
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from valdar.config.loader import NociceptionConfig

Reading = dict[str, float]

LABELS = {
    "gpu_temp": ("la carte graphique chauffe", "°C"),
    "gpu_mem": ("la mémoire de la carte graphique est presque pleine", "%"),
    "ram": ("la mémoire vive est presque pleine", "%"),
    "disk": ("le disque est presque plein", "%"),
    "cpu_temp": ("le processeur chauffe", "°C"),
}


def read_pc(disk_path: Path) -> Reading:
    """Lit les capteurs disponibles. Fractions pour les mémoires, °C pour les températures."""
    out: Reading = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            temp, used, total = (float(x) for x in r.stdout.splitlines()[0].split(","))
            out["gpu_temp"] = temp
            if total > 0:
                out["gpu_mem"] = used / total
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import psutil

        out["ram"] = psutil.virtual_memory().percent / 100.0
        temps = getattr(psutil, "sensors_temperatures", lambda: {})() or {}
        cpu = [t.current for k in ("coretemp", "k10temp", "cpu_thermal")
               for t in temps.get(k, [])]
        if cpu:
            out["cpu_temp"] = max(cpu)
    except ImportError:
        pass
    try:
        du = shutil.disk_usage(disk_path)
        out["disk"] = du.used / du.total
    except OSError:
        pass
    return out


@dataclass
class Pain:
    sensor: str
    value: float
    signal: float
    pain: float
    danger: bool

    def text(self) -> str:
        label, unit = LABELS.get(self.sensor, (self.sensor, ""))
        v = f"{self.value * 100:.0f} %" if unit == "%" else f"{self.value:.0f} {unit}"
        return f"{label} ({v})"


class Nociception:
    def __init__(self, cfg: NociceptionConfig, read: Callable[[], Reading]):
        self.cfg = cfg
        self.read = read
        self.last_sample = float("-inf")
        self.signals: dict[str, float] = {}
        self.values: Reading = {}
        self._last_fire: dict[str, float] = {}
        self._fired_signal: dict[str, float] = {}

    # ---------------------------------------------------------------- signal
    def signal(self, sensor: str, value: float) -> float:
        th = self.cfg.sensors[sensor]
        if value <= th.warn:
            return 0.0
        return min(1.0, (value - th.warn) / (th.danger - th.warn))

    @property
    def danger(self) -> bool:
        """Au moins un capteur au seuil de danger : déclenche le réflexe."""
        return any(s >= 1.0 for s in self.signals.values())

    def due(self, now: float) -> bool:
        return self.cfg.enabled and now - self.last_sample >= self.cfg.interval_seconds

    def sample(self, now: float) -> Reading:
        """Lit les capteurs (hors verrou du cœur : nvidia-smi peut prendre 100 ms)."""
        self.last_sample = now
        values = {k: v for k, v in self.read().items() if k in self.cfg.sensors}
        self.values = values
        self.signals = {k: self.signal(k, v) for k, v in values.items()}
        return values

    # ---------------------------------------------------------------- douleur
    def gate(self, sources: dict[str, float]) -> tuple[float, float]:
        """(anxiété, engagement) dans [0, 1], lus dans l'état du cœur."""
        def mix(weights: dict[str, float]) -> float:
            return max(0.0, min(1.0, sum(w * sources.get(k, 0.0) for k, w in weights.items())))

        return mix(self.cfg.anxiety), mix(self.cfg.engagement)

    def construct(self, signal: float, sources: dict[str, float]) -> float:
        anxiety, engagement = self.gate(sources)
        pain = signal * (1 + self.cfg.amplification * anxiety) \
            * (1 - self.cfg.distraction * engagement)
        return max(0.0, min(1.0, pain))

    def feel(self, heart: Any, now: float) -> list[Pain]:
        """Construit la douleur de chaque signal et la transmet au cœur.

        Un nocicepteur réagit quand il change d'état (apparition, aggravation nette), puis
        se rappelle au cœur toutes les `repeat_seconds` tant que le signal persiste."""
        felt: list[Pain] = []
        sources = heart.sources()
        for sensor, sig in self.signals.items():
            if sig <= 0.0:
                self._fired_signal.pop(sensor, None)
                continue
            before = self._fired_signal.get(sensor)
            changed = before is None or sig - before >= self.cfg.worsen_step
            stale = now - self._last_fire.get(sensor, float("-inf")) >= self.cfg.repeat_seconds
            if not (changed or stale):
                continue
            pain = self.construct(sig, sources)
            if pain < self.cfg.min_pain:
                continue
            danger = sig >= 1.0
            if danger:   # une vraie surchauffe ne s'habitue pas
                spec = heart.hc.stimuli[self.cfg.stimulus]
                heart.apply(spec.impulses, amp=pain * heart.sensitivity, rebound=spec.rebound)
                heart.journal.log("stimulus", t=heart.now, name=self.cfg.stimulus,
                                  scale=round(pain, 3), habituation=1.0, source="nocicepteur",
                                  sensor=sensor)
            else:
                heart.fire(self.cfg.stimulus, scale=pain, source="nocicepteur")
            self._last_fire[sensor] = now
            self._fired_signal[sensor] = sig
            felt.append(Pain(sensor, self.values[sensor], sig, pain, danger))
        return felt

    def lines(self) -> list[str]:
        """Ce que Valdar sent de son corps-PC (pour l'état du monde)."""
        out = []
        for sensor, sig in sorted(self.signals.items(), key=lambda kv: -kv[1]):
            if sig > 0.0:
                p = Pain(sensor, self.values[sensor], sig, 0.0, sig >= 1.0)
                out.append(("DANGER : " if p.danger else "gêne : ") + p.text())
        return out
