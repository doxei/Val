"""Le journal du corps : chaque action sur le PC, écrite avant d'être faite.

Avant, pas après : si Valdar plante au milieu, la trace existe quand même. Une ligne JSON
par action, pour qu'Olivier puisse relire ce qui a été touché.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class Journal:
    def __init__(self, path: Path, keep: int = 20_000):
        self.path, self.keep = path, keep

    def write(self, outil: str, args: dict[str, Any] | None = None, qui: str = "",
              resultat: str = "") -> None:
        row = {"quand": time.strftime("%Y-%m-%dT%H:%M:%S"), "outil": outil, "qui": qui,
               "args": _short(args or {}), "resultat": resultat[:200]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._rotate()
        except OSError:
            pass          # un journal muet ne doit pas empêcher d'agir

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text("utf-8", errors="replace").splitlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
        return out

    def _rotate(self) -> None:
        if self.path.stat().st_size < 4_000_000:
            return
        lines = self.path.read_text("utf-8", errors="replace").splitlines()[-self.keep // 2:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", "utf-8")
        os.replace(tmp, self.path)


def _short(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        s = v if isinstance(v, (int, float, bool)) or v is None else str(v)
        out[k] = s[:200] + "…" if isinstance(s, str) and len(s) > 200 else s
    return out
