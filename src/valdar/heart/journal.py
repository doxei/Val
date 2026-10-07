"""Journal d'observabilité : JSONL en ajout seul, avec rotation par taille."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Journal:
    """Chaque ligne : {"wall": horloge réelle, "t": temps du cœur, "type": ..., ...}."""

    def __init__(self, path: str | Path, max_bytes: int = 20_000_000, keep: int = 5,
                 enabled: bool = True):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.keep = keep
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, etype: str, t: float | None = None, **data: Any) -> None:
        if not self.enabled:
            return
        rec = {"wall": round(time.time(), 3), "t": None if t is None else round(t, 3),
               "type": etype, **data}
        self._rotate_if_needed()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _rotate_if_needed(self) -> None:
        try:
            if self.path.stat().st_size < self.max_bytes:
                return
        except FileNotFoundError:
            return
        for i in range(self.keep, 0, -1):
            src = self.path.with_name(f"{self.path.name}.{i - 1}") if i > 1 else self.path
            dst = self.path.with_name(f"{self.path.name}.{i}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        if self.keep == 0 and self.path.exists():
            self.path.unlink()
