"""Systèmes affectifs (Panksepp), PAD de l'émotion courante et étiquette d'émotion."""
from __future__ import annotations

import math
from typing import Any

from valdar.config.loader import EmotionSpec, HeartConfig, PADSpec
from valdar.heart.dynamics import clamp, logistic, logit


def drive_targets(
    hc: HeartConfig,
    deviations: dict[str, float],
    needs: dict[str, float],
    rest: dict[str, float],
) -> dict[str, float]:
    """Activation cible de chaque drive à partir des écarts des variables et des besoins."""
    out: dict[str, float] = {}
    for name, spec in hc.drives.items():
        z = logit(rest[name])
        for src, w in spec.weights.items():
            z += hc.drive_gain * w * _source(src, deviations, needs)
        out[name] = logistic(z)
    return out


def _source(name: str, deviations: dict[str, float], needs: dict[str, float]) -> float:
    if name in deviations:
        return deviations[name]
    if name.startswith("need_"):
        return needs.get(name[5:], 0.0)
    return 0.0


def flat_sources(
    variables: dict[str, float],
    deviations: dict[str, float],
    drives: dict[str, float],
    drive_rest: dict[str, float],
    needs: dict[str, float],
    need_rest: float,
    pad: dict[str, float] | None = None,
) -> dict[str, float]:
    """Vue aplatie de toutes les sources nommées (PAD, organes, envies)."""
    src: dict[str, float] = {}
    for k, v in variables.items():
        src[k] = v
        d = deviations[k]
        src[f"{k}_dev"] = d
        src[f"{k}_low"] = max(0.0, -d)
    for k, a in drives.items():
        r = drive_rest[k]
        src[k] = a
        src[f"{k}_delta"] = a - r
        src[f"{k}_exc"] = max(0.0, a - r) / (1.0 - r)
    for k, n in needs.items():
        src[f"need_{k}"] = n
    src["need_rest"] = need_rest
    if pad is not None:
        for axis in ("P", "A", "D"):
            v = pad[axis]
            src[axis] = v
            src[f"{axis}_pos"] = max(0.0, v)
            src[f"{axis}_neg"] = max(0.0, -v)
    return src


def compute_pad(spec: PADSpec, src: dict[str, float],
                offsets: dict[str, float] | None = None) -> dict[str, float]:
    """PAD de l'émotion courante, centré sur le repos : axe = tanh(gain × Σ poids × source).

    Pour les variables on utilise l'écart à la base, pour les drives l'écart au repos.
    `offsets` s'ajoute avant tanh (ex. baisse d'activation pendant le sommeil).
    """
    out: dict[str, float] = {}
    for axis, weights in (("P", spec.valence), ("A", spec.arousal), ("D", spec.dominance)):
        z = (offsets or {}).get(axis, 0.0)
        for name, w in weights.items():
            z += w * _pad_source(name, src)
        out[axis] = math.tanh(spec.gain * z)
    return out


def _pad_source(name: str, src: dict[str, float]) -> float:
    if f"{name}_dev" in src:
        return src[f"{name}_dev"]
    if f"{name}_delta" in src:
        return src[f"{name}_delta"]
    return src.get(name, 0.0)


def dominant_drive(
    drives: dict[str, float], drive_rest: dict[str, float], minimum: float
) -> str | None:
    best, best_v = None, minimum
    for k, a in drives.items():
        v = a - drive_rest[k]
        if v > best_v:
            best, best_v = k, v
    return best


def emotion_label(
    pad: dict[str, float],
    spec: EmotionSpec,
    cues: dict[str, float],
) -> tuple[str, float]:
    """Prototype PAD le plus proche. Un prototype avec indice n'est possible que si l'indice
    atteint son seuil, et il gagne alors cue_bonus. Renvoie (nom, distance)."""
    norm = math.sqrt(pad["P"] ** 2 + pad["A"] ** 2 + pad["D"] ** 2)
    if norm < spec.calm_radius and "calme" in spec.prototypes:
        return "calme", norm
    best, best_score, best_dist = "calme", math.inf, math.inf
    for name, p in spec.prototypes.items():
        bonus = 0.0
        if p.cue is not None:
            threshold = p.cue_min if p.cue_min is not None else spec.cue_min_default
            if cues.get(p.cue, 0.0) < threshold:
                continue
            bonus = spec.cue_bonus
        dist = math.sqrt((p.P - pad["P"]) ** 2 + (p.A - pad["A"]) ** 2 + (p.D - pad["D"]) ** 2)
        score = dist - bonus
        if score < best_score:
            best, best_score, best_dist = name, score, dist
    return best, best_dist


def intensity(pad: dict[str, Any]) -> float:
    """Norme PAD normalisée dans [0, 1]."""
    return clamp(math.sqrt(pad["P"] ** 2 + pad["A"] ** 2 + pad["D"] ** 2) / math.sqrt(3.0))
