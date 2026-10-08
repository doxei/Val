"""Base de pinouts des cartes (Octopus, SKR, TMC...) — reprise de l'ancienne installation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from valdar.atelier.checklist import norm


class Pinouts:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] | None = None

    def data(self) -> dict[str, Any]:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._data = {}
        return self._data

    def lookup(self, cible: str) -> str:
        d = self.data()
        if not d:
            return "ma base de pinouts est vide (importe l'ancienne : valdar import-ancien)."
        n = norm(cible).strip()
        if not n:
            return "dis-moi une carte ou un connecteur (ex : 'BTT Octopus v1.1', 'TMC2209')."
        scored: list[tuple[int, str]] = []
        for key, board in d.items():
            hay = norm(key) + " " + norm(str(board.get("titre", "")))
            score = 50 if n in hay else sum(1 for w in n.split() if w in hay)
            if score:
                scored.append((score, key))
        if not scored:
            known = ", ".join(str(v.get("titre", k)) for k, v in d.items())
            return f"rien sur « {cible} ». Je connais : {known}."
        key = max(scored)[1]
        return _render(key, d[key])


def _fmt(entries: Any) -> list[str]:
    if isinstance(entries, dict):
        entries = [{"broche": k, **v} if isinstance(v, dict) else {"broche": k, "role": v}
                   for k, v in entries.items()]
    out = []
    for b in entries or []:
        if isinstance(b, str):
            out.append(f"- {b}")
            continue
        name = b.get("broche", b.get("nom", "?"))
        role = b.get("role") or b.get("signal") or b.get("fonction") or ""
        extra = " ".join(f"{k}={v}" for k, v in b.items()
                         if k not in ("broche", "nom", "role", "signal", "fonction"))
        out.append(f"- {name} : {role}" + (f" ({extra})" if extra else ""))
    return out


def _render(key: str, board: dict[str, Any]) -> str:
    lines = [f"{board.get('titre', key)} [source : {board.get('source', '?')}, "
             f"confiance : {board.get('confiance', '?')}]"]
    lines += [f"* {n}" for n in board.get("notes") or []]
    moteurs = board.get("moteurs")
    if isinstance(moteurs, dict):
        lines.append("MOTEURS :")
        for name, m in moteurs.items():
            if isinstance(m, dict):
                detail = " ".join(f"{k}={v}" for k, v in m.items() if k != "note")
                note = f" // {m['note']}" if m.get("note") else ""
                lines.append(f"- {name} : {detail}{note}")
    for section in ("chauffage", "ventilateurs", "capteurs", "broches"):
        if section in board:
            lines.append(f"{section.upper()} :")
            lines += _fmt(board[section])
    lines += [f"! PIÈGE : {p}" for p in board.get("pieges") or []]
    return "\n".join(lines)
