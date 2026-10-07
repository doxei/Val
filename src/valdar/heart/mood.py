"""Humeur (modèle ALMA, Gebhard 2005) et tempérament (Big Five → PAD, Mehrabian 1996).

L'humeur est un point PAD attiré par l'émotion courante (constante pull_tau) et ramené vers
l'humeur par défaut du tempérament (constante return_tau). Les deux forces combinées ont une
solution exacte : relaxation vers un équilibre pondéré avec une constante de temps effective.
"""
from __future__ import annotations

import math

from valdar.config.loader import MoodSpec, TemperamentSpec
from valdar.heart.dynamics import clamp, relax

AXES = ("P", "A", "D")


def default_mood(temperament: TemperamentSpec, profile: str, scale: float) -> dict[str, float]:
    """Humeur par défaut = échelle × projection de Mehrabian des Big Five du profil."""
    b5 = temperament.profiles[profile].big_five
    out: dict[str, float] = {}
    for axis in AXES:
        v = sum(w * b5.get(trait, 0.0) for trait, w in temperament.mehrabian[axis].items())
        out[axis] = clamp(scale * v, -1.0, 1.0)
    return out


def update(
    mood: dict[str, float],
    emotion_pad: dict[str, float],
    default: dict[str, float],
    spec: MoodSpec,
    dt: float,
) -> dict[str, float]:
    """dm/dt = (émotion − m)/pull_tau + (défaut − m)/return_tau, intégré exactement."""
    kp, kr = 1.0 / spec.pull_tau, 1.0 / spec.return_tau
    tau_eff = 1.0 / (kp + kr)
    out: dict[str, float] = {}
    for axis in AXES:
        eq = (kp * emotion_pad[axis] + kr * default[axis]) * tau_eff
        out[axis] = clamp(relax(mood[axis], eq, dt, tau_eff), -1.0, 1.0)
    return out


def octant_key(mood: dict[str, float]) -> str:
    return "".join(("+" if mood[a] >= 0.0 else "-") + a for a in AXES)


def mood_label(mood: dict[str, float], spec: MoodSpec) -> str:
    """Étiquette d'humeur : "neutre" ou "<intensité> <octant>" (ex. "légèrement détendu")."""
    norm = math.sqrt(sum(mood[a] ** 2 for a in AXES)) / math.sqrt(3.0)
    if norm < spec.neutral_radius:
        return "neutre"
    word = ""
    for iw in spec.intensity_words:
        if norm >= iw.threshold:
            word = iw.word
    name = spec.octants[octant_key(mood)]
    return f"{word} {name}".strip()
