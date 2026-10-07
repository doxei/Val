"""Organes virtuels : sensations ressenties dérivées de l'état, exprimées en français.

Ce sont des métaphores corporelles d'un état simulé, jamais des affirmations biologiques.
"""
from __future__ import annotations

from valdar.config.loader import HeartConfig
from valdar.heart.dynamics import clamp, relax


def activations(hc: HeartConfig, src: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, spec in hc.organs.items():
        v = spec.bias
        for s, w in spec.sources.items():
            v += w * src.get(s, 0.0)
        out[name] = clamp(v)
    return out


def integrate(hc: HeartConfig, body: dict[str, float], src: dict[str, float],
              dt: float) -> dict[str, float]:
    """Les organes suivent l'état avec leur propre inertie (exact pour tout dt) : ils se
    mettent en place en rise_tau et se dénouent en fall_tau. Ce sont les « tampons » qui font
    sentir le poids d'une émotion après coup."""
    target = activations(hc, src)
    out: dict[str, float] = {}
    for name, spec in hc.organs.items():
        x = body.get(name, target[name])
        tau = spec.rise_tau if target[name] > x else spec.fall_tau
        out[name] = clamp(relax(x, target[name], dt, tau))
    return out


def felt_texts(hc: HeartConfig, act: dict[str, float]) -> dict[str, str]:
    res: dict[str, str] = {}
    for name, spec in hc.organs.items():
        a = act[name]
        text = spec.texts[-1].text if a >= spec.texts[-1].hi else spec.texts[0].text
        for band in spec.texts:
            if band.lo <= a < band.hi:
                text = band.text
                break
        res[name] = text
    return res


def bpm(hc: HeartConfig, act: dict[str, float]) -> int | None:
    spec = hc.organs.get("coeur")
    if spec is None or spec.low is None or spec.high is None:
        return None
    return int(round(spec.low + (spec.high - spec.low) * act.get("coeur", 0.0)))
