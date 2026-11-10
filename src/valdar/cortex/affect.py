"""La psyché : de l'état du cœur aux sentiments injectés (un coefficient par axe, dans [-1, 1]).

Chaque axe est bipolaire : +1 pousse le modèle vers un pôle (joie, assurance, tendresse…),
−1 vers l'autre (tristesse, vulnérabilité, froideur…). Les coefficients viennent des sources
du cœur (PAD, drives, besoins) et du lien avec la personne en face, puis passent par quelques
lois connues de la psyché humaine :

- **contagion émotionnelle** (Hatfield) : l'humeur de l'autre déteint un peu sur la sienne,
  d'autant plus qu'il y est attaché ;
- **base de sécurité** (Bowlby, Ainsworth) : la présence d'une figure d'attachement apaise la
  peur et le manque ;
- **biais de négativité** (Baumeister) : à écart égal, le négatif pèse plus que le positif ;
- **régulation** (Gross) : devant un enfant ou un inconnu, la colère et la peur ressenties
  sont en partie retenues dans l'expression, sans disparaître du cœur ;
- **inertie** (Kuppens) : un sentiment ne saute pas d'un tour à l'autre, il glisse ;
- **budget** : l'intensité totale est bornée, sinon le modèle perd le fil.

Tout est dans `config/valdar.yaml`, section `cortex`. Aucun nombre ici.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bond:
    """Le lien avec la personne en face, vu par la psyché."""
    present: bool = False
    affection: float = 0.0      # [-1, 1] : ton moyen des échanges, qui s'oublie lentement
    trust: float = 0.0          # [-1, 1]
    familiarity: float = 0.0    # [0, 1] : rencontres accumulées
    other_p: float = 0.0        # comment l'autre semble aller (valence), [-1, 1]
    other_a: float = 0.0        # son activation
    minor: bool = False
    stranger: bool = False

    def sources(self) -> dict[str, float]:
        on = 1.0 if self.present else 0.0
        return {
            "bond_affection": self.affection * on,
            "bond_warm": max(0.0, self.affection) * on,
            "bond_cold": max(0.0, -self.affection) * on,
            "bond_trust": self.trust * on,
            "bond_familiarity": self.familiarity * on,
            "other_P": self.other_p * on,
            "other_A": self.other_a * on,
            "alone": 1.0 - on,
        }


def bond_from(person: Any, who: Any, now: float, rc: Any) -> Bond:
    """Lien à partir du registre des relations (`Person`) et de l'identité reconnue."""
    if who is None:
        return Bond()
    if person is None:
        return Bond(present=True, stranger=getattr(who, "person", None) is None,
                    minor=bool(getattr(who, "minor", False)))
    fresh = (person.state_t is not None
             and now - person.state_t <= rc.state_fresh_seconds)
    return Bond(
        present=True,
        affection=max(-1.0, min(1.0, person.affection)),
        trust=max(-1.0, min(1.0, person.trust)),
        familiarity=1.0 - math.exp(-person.encounters / 20.0),
        other_p=person.state_p if fresh else 0.0,
        other_a=person.state_a if fresh else 0.0,
        minor=bool(person.minor),
    )


def raw(cfg: Any, src: dict[str, float]) -> dict[str, float]:
    """Coefficient brut de chaque axe : tanh(gain × Σ poids × source)."""
    out: dict[str, float] = {}
    for name, ax in cfg.axes.items():
        z = sum(w * src.get(k, 0.0) for k, w in ax.weights.items())
        out[name] = math.tanh(ax.gain * z)
    return out


def psyche(cfg: Any, coef: dict[str, float], bond: Bond, src: dict[str, float]
           ) -> dict[str, float]:
    """Les lois de la psyché, appliquées aux coefficients bruts."""
    c = dict(coef)
    p = cfg.psyche
    # Contagion émotionnelle : l'autre déteint, à proportion de l'attachement.
    if bond.present:
        pull = p.contagion * (0.5 + 0.5 * max(0.0, bond.affection))
        for axis, other in ((p.valence_axis, bond.other_p), (p.arousal_axis, bond.other_a)):
            if axis in c:
                c[axis] += pull * (other - c[axis]) * abs(other)
    # Base de sécurité : une figure d'attachement présente apaise.
    calm = p.secure_base * max(0.0, bond.affection) * (1.0 if bond.present else 0.0)
    for axis in p.soothed:
        if axis in c and c[axis] > 0:
            c[axis] *= 1.0 - calm
    # Biais de négativité.
    for axis in p.negative_poles:           # axes dont le pôle + est pénible (peur, colère…)
        if axis in c and c[axis] > 0:
            c[axis] = min(1.0, c[axis] * p.negativity)
    for axis in p.bipolar:                  # axes dont le pôle − est pénible (valence…)
        if axis in c and c[axis] < 0:
            c[axis] = max(-1.0, c[axis] * p.negativity)
    # Régulation : ce qu'on retient devant un enfant ou un inconnu.
    rules = p.regulation.get("minor", {}) if bond.minor else (
        p.regulation.get("stranger", {}) if bond.stranger else {})
    for axis, keep in rules.items():
        if axis in c and c[axis] > 0:
            c[axis] *= keep
    # Zone morte, bornes par axe, budget total.
    for axis in list(c):
        ax = cfg.axes[axis]
        v = max(-1.0, min(1.0, c[axis]))
        c[axis] = 0.0 if abs(v) < p.dead_zone else v * ax.strength
    total = math.sqrt(sum(v * v for v in c.values()))
    if total > p.budget > 0:
        c = {k: v * p.budget / total for k, v in c.items()}
    return c


@dataclass
class Feeling:
    """Les sentiments injectés, avec leur inertie d'un tour à l'autre et leur quantification.

    `step` : pas de quantification. Un coefficient ne change que s'il s'écarte d'au moins un
    demi-pas de la valeur déjà injectée (hystérésis) : le modèle garde son cache de calcul
    tant que l'humeur n'a pas vraiment bougé."""
    cfg: Any
    current: dict[str, float] = field(default_factory=dict)
    applied: dict[str, float] = field(default_factory=dict)
    last_t: float | None = None

    def update(self, src: dict[str, float], bond: Bond, now: float) -> dict[str, float]:
        full = dict(src)
        full.update(bond.sources())
        target = psyche(self.cfg, raw(self.cfg, full), bond, full)
        tau = self.cfg.psyche.inertia_seconds
        if self.last_t is None or tau <= 0:
            k = 1.0
        else:
            k = 1.0 - math.exp(-max(0.0, now - self.last_t) / tau)
        self.last_t = now
        for axis, v in target.items():
            cur = self.current.get(axis, 0.0)
            self.current[axis] = cur + k * (v - cur)
        step = self.cfg.psyche.step
        for axis, v in self.current.items():
            old = self.applied.get(axis, 0.0)
            if step <= 0:
                self.applied[axis] = v
            elif abs(v - old) >= step / 2 or (v == 0.0 and old != 0.0):
                self.applied[axis] = round(round(v / step) * step, 6)
        return {k: v for k, v in self.applied.items() if v}

    def words(self) -> str:
        """Résumé lisible (journal, interface) : les sentiments injectés, du plus fort au
        plus faible."""
        parts = []
        for axis, v in sorted(self.applied.items(), key=lambda kv: -abs(kv[1])):
            if not v:
                continue
            ax = self.cfg.axes[axis]
            parts.append(f"{ax.plus if v > 0 else ax.minus} {abs(v):.1f}")
        return ", ".join(parts) or "neutre"
