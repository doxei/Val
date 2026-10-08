"""Rappels et alarmes, avec compréhension des moments en français (repris de l'ancienne
installation, corrigé)."""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

MAX_TEXT = 200
_NUM_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
              "dix": 10, "quinze": 15, "vingt": 20, "trente": 30, "quarante": 40}


def parse_when(text: str, now: float | None = None) -> float | None:
    """Comprend : « dans 10 minutes », « dans une heure », « dans 2h30 », « à 15h30 »,
    « à 9h », « demain à 8h », « à midi », « à minuit », « dans un quart d'heure »."""
    now = time.time() if now is None else now
    t = " " + text.strip().lower().replace("’", "'") + " "
    for word, n in _NUM_WORDS.items():
        t = re.sub(rf"\b{word}\b", str(n), t)
    base = datetime.fromtimestamp(now)

    if "quart d'heure" in t:
        return now + 900
    if "demi-heure" in t or "demi heure" in t:
        return now + 1800

    m = re.search(r"dans\s+(\d+)\s*h(?:eures?)?\s*(\d{1,2})?(?:\s*min\w*)?", t)
    if m:
        return now + int(m.group(1)) * 3600 + int(m.group(2) or 0) * 60
    m = re.search(r"dans\s+(\d+)\s*(secondes?|sec|s)\b", t)
    if m:
        return now + int(m.group(1))
    m = re.search(r"dans\s+(\d+)\s*(minutes?|min|mn|m)\b", t)
    if m:
        return now + int(m.group(1)) * 60
    m = re.search(r"dans\s+(\d+)\s*jours?\b", t)
    if m:
        return now + int(m.group(1)) * 86400

    day = 1 if "demain" in t else (2 if "après-demain" in t or "apres-demain" in t else 0)
    if "après-demain" in t or "apres-demain" in t:
        day = 2
    hm: tuple[int, int] | None = None
    if "midi" in t:
        hm = (12, 0)
    elif "minuit" in t:
        hm = (0, 0)
        day = max(day, 1)
    else:
        m = re.search(r"(?:à|a|vers)\s*(\d{1,2})\s*(?:h|:)\s*(\d{2})?", t)
        if m:
            hm = (int(m.group(1)), int(m.group(2) or 0))
    if hm is None:
        return None
    h, mn = hm
    if not (0 <= h <= 23 and 0 <= mn <= 59):
        return None
    when = (base + timedelta(days=day)).replace(hour=h, minute=mn, second=0, microsecond=0)
    if day == 0 and when.timestamp() <= now:
        when += timedelta(days=1)
    return when.timestamp()


class Reminders:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return []
        return [r for r in data if isinstance(r, dict)]

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, texte: str, due: float) -> dict[str, Any]:
        items = self.load()
        r = {"id": uuid.uuid4().hex[:10], "texte": texte[:MAX_TEXT], "due": float(due),
             "etat": "attente", "cree": time.time()}
        items.append(r)
        self._save(items)
        return r

    def cancel(self, fragment: str) -> list[dict[str, Any]]:
        items = self.load()
        hay = fragment.strip().lower()
        gone = [r for r in items if hay and r.get("etat") == "attente"
                and hay in r.get("texte", "").lower()]
        if gone:
            ids = {r["id"] for r in gone}
            self._save([r for r in items if r.get("id") not in ids])
        return gone

    def upcoming(self, limit: int = 8) -> list[dict[str, Any]]:
        return sorted((r for r in self.load() if r.get("etat") == "attente"),
                      key=lambda r: r["due"])[:limit]

    def pop_due(self, now: float | None = None) -> list[dict[str, Any]]:
        """Renvoie les rappels échus et les marque comme faits."""
        now = time.time() if now is None else now
        items = self.load()
        due = [r for r in items if r.get("etat") == "attente" and r.get("due", 0) <= now]
        if due:
            for r in due:
                r["etat"] = "fait"
            self._save(items)
        return due

    def render(self) -> str:
        items = self.upcoming()
        if not items:
            return "aucun rappel en attente."
        return "\n".join(
            f"- {time.strftime('%d/%m %H:%M', time.localtime(r['due']))} : {r['texte']}"
            for r in items)
