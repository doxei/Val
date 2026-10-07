"""Fonctions mathématiques pures du cœur (aucun état, aucun nombre en dur).

Toutes les intégrations sont exactes pour un pas dt quelconque, ce qui rend la simulation
accélérée et le rattrapage cohérents avec le fonctionnement à 1 Hz.
"""
from __future__ import annotations

import math
import random

_EPS = 1e-9


def decay(dt: float, tau: float) -> float:
    """Fraction restante d'un écart après dt secondes (e^(−dt/τ))."""
    if tau <= 0.0:
        return 0.0
    return math.exp(-max(0.0, dt) / tau)


def relax(x: float, target: float, dt: float, tau: float) -> float:
    """Retour exponentiel exact vers la cible : x ← cible + (x − cible)·e^(−dt/τ)."""
    return target + (x - target) * decay(dt, tau)


def ou_noise(rng: random.Random, dt: float, tau: float, std: float) -> float:
    """Incrément d'un processus d'Ornstein-Uhlenbeck exact d'écart-type stationnaire `std`.

    À ajouter après `relax` : la variance stationnaire ne dépend pas du pas dt.
    """
    if std <= 0.0 or tau <= 0.0:
        return 0.0
    return std * math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * dt / tau))) * rng.gauss(0.0, 1.0)


def soft_add(x: float, delta: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Ajoute delta avec saturation douce : ≈ x + delta loin des bornes, jamais au-delà.

    Vers le haut : x' = hi − (hi − x)·e^(−delta/(hi − x)) ; symétrique vers le bas.
    """
    x = min(hi, max(lo, x))
    if delta > 0.0:
        room = hi - x
        if room <= _EPS:
            return x
        return hi - room * math.exp(-delta / room)
    if delta < 0.0:
        room = x - lo
        if room <= _EPS:
            return x
        return lo + room * math.exp(delta / room)
    return x


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def logistic(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def approach_one(n: float, dt: float, tau: float) -> float:
    """Approche exponentielle exacte de 1 (montée d'un besoin)."""
    return 1.0 - (1.0 - n) * decay(dt, tau)


def parse_hours(s: str) -> float:
    hh, mm = s.split(":")
    return float(hh) + float(mm) / 60.0


def in_window(hour: float, start: str, end: str) -> bool:
    """Vrai si l'heure locale est dans [start, end[, fenêtres passant minuit comprises."""
    a, b = parse_hours(start), parse_hours(end)
    if a <= b:
        return a <= hour < b
    return hour >= a or hour < b
