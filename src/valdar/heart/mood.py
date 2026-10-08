"""Humeur (modèle ALMA, Gebhard 2005) et tempérament (Big Five → PAD, Mehrabian 1996).

L'humeur est un point PAD attiré par l'émotion courante (constante pull_tau) et ramené vers
l'humeur par défaut du tempérament (constante return_tau). Les deux forces combinées ont une
solution exacte : relaxation vers un équilibre pondéré avec une constante de temps effective.

Hystérésis (avenant 3 §3) : sur l'axe plaisir s'ajoute un terme bistable
κ · (u − u³/w² + h), u = P − centre. Sous une pression négative durable apparaît un second
état stable, « moral bas » : il faut alors plus de positif pour remonter qu'il n'en a fallu
pour tomber. Au repos, un seul état (garde-fou 1) ; la nuit, κ est réduit (garde-fou 2).
Le terme cubique n'a pas de solution exacte : sous-pas de `substep` secondes au plus.
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


def _alma(mood: dict[str, float], emotion_pad: dict[str, float], default: dict[str, float],
          spec: MoodSpec, dt: float) -> dict[str, float]:
    """dm/dt = (émotion − m)/pull_tau + (défaut − m)/return_tau, intégré exactement."""
    kp, kr = 1.0 / spec.pull_tau, 1.0 / spec.return_tau
    tau_eff = 1.0 / (kp + kr)
    out: dict[str, float] = {}
    for axis in AXES:
        eq = (kp * emotion_pad[axis] + kr * default[axis]) * tau_eff
        out[axis] = clamp(relax(mood[axis], eq, dt, tau_eff), -1.0, 1.0)
    return out


def bistable_force(p: float, spec: MoodSpec, awake: bool = True) -> float:
    """κ · (u − u³/w² + h) sur l'axe plaisir (par seconde), seulement entre les deux puits
    (|u| ≤ w) : au-delà, le cube freinerait la joie comme la tristesse, et c'est ALMA seul qui
    gouverne. La force est nulle aux puits (u = ±w), donc continue."""
    b = spec.bistable
    kappa = (1.0 if awake else b.sleep_factor) / b.kappa_tau
    u = p - b.center
    if abs(u) > b.width:
        return kappa * b.bias
    return kappa * (u - u ** 3 / b.width ** 2 + b.bias)


def update(
    mood: dict[str, float],
    emotion_pad: dict[str, float],
    default: dict[str, float],
    spec: MoodSpec,
    dt: float,
    awake: bool = True,
) -> dict[str, float]:
    if not spec.bistable.enabled:
        return _alma(mood, emotion_pad, default, spec, dt)
    n = max(1, math.ceil(dt / spec.bistable.substep))
    h = dt / n
    out = mood
    for _ in range(n):   # séparation d'opérateurs : ALMA exact, puis terme bistable
        out = _alma(out, emotion_pad, default, spec, h)
        out["P"] = clamp(out["P"] + h * bistable_force(out["P"], spec, awake), -1.0, 1.0)
    return out


def is_low(mood: dict[str, float], spec: MoodSpec) -> bool:
    """Moral bas : sous le centre des deux puits (côté du puits bas)."""
    return spec.bistable.enabled and mood["P"] < spec.bistable.center


def lag1_autocorrelation(xs: list[float]) -> float:
    """Autocorrélation à un pas des écarts à la tendance (ralentissement critique)."""
    n = len(xs)
    if n < 6:
        return 0.0
    k = 3   # moyenne glissante centrée : retire la tendance lente
    trend = [sum(xs[max(0, i - k):i + k + 1]) / len(xs[max(0, i - k):i + k + 1])
             for i in range(n)]
    r = [x - t for x, t in zip(xs, trend, strict=True)]
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r)
    if var < 1e-12:
        return 0.0
    return sum((r[i] - mean) * (r[i + 1] - mean) for i in range(n - 1)) / var


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
