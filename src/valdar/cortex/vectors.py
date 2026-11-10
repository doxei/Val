"""Les directions des sentiments dans le modèle, et leur combinaison en une injection.

Un fichier `.npz` par modèle :
- `axes`    : noms des axes ;
- `dirs`    : [axe, couche, dimension] — direction unitaire de chaque axe à chaque couche
              (pôle + moins pôle −, différence des moyennes des activations) ;
- `norms`   : [couche] — norme moyenne du flux résiduel à cette couche (texte neutre) ;
- `quality` : [axe, couche] — séparation des deux pôles (d de Cohen le long de la direction) ;
- `model`   : nom du modèle, pour refuser des vecteurs appris sur un autre.

La couche d'indice `i` du fichier est la couche `i + 1` du modèle (la couche 0 n'est jamais
pilotée, comme dans llama.cpp).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Vectors:
    axes: list[str]
    dirs: np.ndarray          # [n_axes, n_layers - 1, n_embd], float32
    norms: np.ndarray         # [n_layers - 1]
    quality: np.ndarray       # [n_axes, n_layers - 1]
    model: str = ""

    @property
    def n_layers(self) -> int:
        return self.dirs.shape[1] + 1

    @property
    def n_embd(self) -> int:
        return self.dirs.shape[2]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, axes=np.array(self.axes), dirs=self.dirs.astype(np.float16),
                            norms=self.norms.astype(np.float32),
                            quality=self.quality.astype(np.float32), model=np.array(self.model))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Vectors:
        with np.load(path, allow_pickle=False) as z:
            return cls(axes=[str(a) for a in z["axes"]], dirs=z["dirs"].astype(np.float32),
                       norms=z["norms"].astype(np.float32),
                       quality=z["quality"].astype(np.float32), model=str(z["model"]))

    def best_layers(self, first: int, last: int) -> tuple[int, int]:
        """Bornes de couches (du modèle, incluses), ramenées dans ce que le modèle a."""
        first = max(1, min(first, self.n_layers - 1))
        last = max(first, min(last, self.n_layers - 1))
        return first, last

    def injection(self, coefs: dict[str, float], alpha: float, first: int, last: int
                  ) -> np.ndarray | None:
        """Le tampon à injecter : [couche, dimension], zéro hors de [first, last].

        À chaque couche pilotée on ajoute `alpha × norme(couche) / nombre de couches` fois la
        somme pondérée des directions : `alpha` est donc la part *totale* du flux résiduel
        déplacée (les ajouts se cumulent d'une couche à l'autre), comme dans les essais
        d'Anthropic (0,05 à 0,1 : subtil ; 0,5 : très fort)."""
        active = {a: c for a, c in coefs.items() if c and a in self.axes}
        if not active or alpha <= 0:
            return None
        first, last = self.best_layers(first, last)
        n = last - first + 1
        buf = np.zeros((self.n_layers - 1, self.n_embd), dtype=np.float32)
        for axis, c in active.items():
            k = self.axes.index(axis)
            for layer in range(first, last + 1):
                i = layer - 1
                buf[i] += (c * alpha * self.norms[i] / n) * self.dirs[k, i]
        return buf


def build(axes: list[str], pos: dict[str, np.ndarray], neg: dict[str, np.ndarray],
          neutral: np.ndarray, model: str = "") -> Vectors:
    """Directions à partir des activations.

    `pos[axe]`, `neg[axe]` : [exemple, couche, dimension] (activations moyennes de chaque
    texte, couches 1..n−1). `neutral` : pareil pour des textes neutres (normes du flux)."""
    n_layers, n_embd = neutral.shape[1], neutral.shape[2]
    dirs = np.zeros((len(axes), n_layers, n_embd), dtype=np.float32)
    quality = np.zeros((len(axes), n_layers), dtype=np.float32)
    for k, axis in enumerate(axes):
        p, n = pos[axis].astype(np.float64), neg[axis].astype(np.float64)
        diff = p.mean(axis=0) - n.mean(axis=0)                      # [couche, dimension]
        norm = np.linalg.norm(diff, axis=1, keepdims=True)
        unit = diff / np.maximum(norm, 1e-12)
        dirs[k] = unit.astype(np.float32)
        pp = np.einsum("eld,ld->el", p, unit)                       # projections
        nn = np.einsum("eld,ld->el", n, unit)
        spread = np.sqrt((pp.var(axis=0) + nn.var(axis=0)) / 2.0)
        quality[k] = ((pp.mean(axis=0) - nn.mean(axis=0)) / np.maximum(spread, 1e-9)
                      ).astype(np.float32)
    norms = np.linalg.norm(neutral.astype(np.float64), axis=2).mean(axis=0).astype(np.float32)
    return Vectors(axes=list(axes), dirs=dirs, norms=norms, quality=quality, model=model)


def cosines(v: Vectors, layer: int) -> np.ndarray:
    """Cosinus entre axes à une couche du modèle (pour voir lesquels se recouvrent)."""
    d = v.dirs[:, layer - 1, :]
    return d @ d.T
