"""Checklist de procédure (une seule active), affichée plus tard sur le projecteur."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Item:
    text: str
    done: bool = False


@dataclass
class State:
    titre: str = ""
    items: list[Item] = field(default_factory=list)

    def active(self) -> bool:
        return bool(self.titre and self.items)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


class Checklist:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.current = State()
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self.current = State(titre=d.get("titre", ""),
                                 items=[Item(**i) for i in d.get("items", [])])
        except (ValueError, TypeError):
            self.current = State()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.current), ensure_ascii=False),
                             encoding="utf-8")

    def create(self, titre: str, etapes: list[str]) -> str:
        self.current = State(titre=titre.strip(),
                             items=[Item(t.strip()) for t in etapes if t and t.strip()])
        self.save()
        return self.render()

    def check(self, faites: list[str], annulees: list[str] | None = None) -> str:
        for label in faites:
            for item in self._match(label):
                item.done = True
        for label in annulees or []:
            for item in self._match(label):
                item.done = False
        self.save()
        return self.render()

    def _match(self, label: str) -> list[Item]:
        if not self.current.active():
            return []
        m = re.match(r"\s*(\d+)\s*$", label)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(self.current.items):
                return [self.current.items[idx]]
            return []
        nl = norm(label).strip()
        return [it for it in self.current.items
                if nl and (nl in norm(it.text) or norm(it.text) in nl)]

    def clear(self) -> str:
        self.current = State()
        self.save()
        return "checklist effacée."

    def render(self) -> str:
        if not self.current.active():
            return "aucune checklist active."
        done = sum(1 for i in self.current.items if i.done)
        lines = [f"[{done}/{len(self.current.items)}] {self.current.titre}"]
        for n, item in enumerate(self.current.items, 1):
            lines.append(f"{'[x]' if item.done else '[ ]'} {n}. {item.text}")
        return "\n".join(lines)
