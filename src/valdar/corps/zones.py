"""Où Valdar peut lire et écrire sur le disque."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/").lower().rstrip("/") + "/"


@dataclass
class Zones:
    home: Path
    racine: Path
    atelier: Path
    lecture: list[Path]
    ecriture: list[Path]
    interdits: list[str]
    proteges: list[Path]

    @classmethod
    def from_cfg(cls, cc: Any, racine: Path, home: Path | None = None) -> Zones:
        home = Path(home or os.path.expanduser("~"))

        def expand(s: str) -> Path:
            s = s.replace("{racine}", str(racine))
            if s == "~" or s.startswith("~/") or s.startswith("~\\"):
                s = str(home) + s[1:]
            return Path(s)

        return cls(home=home, racine=racine, atelier=expand(cc.atelier),
                   lecture=[expand(x) for x in cc.lecture],
                   ecriture=[expand(x) for x in cc.ecriture],
                   interdits=[x.lower().replace("\\", "/") for x in cc.interdits],
                   proteges=[expand(x) for x in cc.proteges_ecriture])

    def resolve(self, chemin: str) -> Path:
        """Chemin donné par le modèle → chemin absolu (relatif = dans son atelier)."""
        c = (chemin or "").strip().strip('"').strip("'")
        if c in ("", "."):
            return self.atelier
        if c == "~" or c.startswith(("~/", "~\\")):
            return Path(str(self.home) + c[1:]).resolve()
        p = Path(c)
        if not p.is_absolute():
            p = self.atelier / p
        return p.resolve()

    @staticmethod
    def _inside(p: Path, roots: list[Path]) -> bool:
        n = _norm(p)
        return any(n.startswith(_norm(r.resolve())) for r in roots)

    def forbidden(self, p: Path) -> str | None:
        n = _norm(p)
        for motif in self.interdits:
            if motif in n:
                return f"zone interdite ({motif})"
        return None

    def can_read(self, p: Path) -> str | None:
        """None si permis, sinon la raison du refus."""
        bad = self.forbidden(p)
        if bad:
            return bad
        if not self._inside(p, [*self.lecture, self.atelier]):
            return "hors des dossiers que je peux lire"
        return None

    def can_write(self, p: Path) -> str | None:
        bad = self.forbidden(p)
        if bad:
            return bad
        if self._inside(p, self.proteges):
            return "c'est mon propre code ou un de mes garde-fous : je n'y touche pas"
        if not self._inside(p, [*self.ecriture, self.atelier]):
            return "hors des dossiers où je peux écrire"
        return None

    def in_workshop(self, p: Path) -> bool:
        return self._inside(p, [self.atelier])
