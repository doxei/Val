"""Réserve de l'atelier (filaments, vis, composants) — repris de RAUB, même schéma SQLite."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        qty REAL NOT NULL DEFAULT 0,
        unit TEXT DEFAULT '',
        loc TEXT DEFAULT '',
        threshold REAL NOT NULL DEFAULT 0,
        updated TEXT)""",
    """CREATE TABLE IF NOT EXISTS moves(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT NOT NULL,
        delta REAL NOT NULL,
        ts TEXT NOT NULL,
        note TEXT DEFAULT '')""",
)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class Stock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as con:
            for stmt in _SCHEMA:
                con.execute(stmt)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def set_item(self, name: str, qty: float, unit: str = "", loc: str = "",
                 threshold: float | None = None, note: str = "") -> str:
        name = name.strip()
        if not name:
            return "il me faut le nom de l'article."
        with self._lock, self._conn() as con:
            row = con.execute("SELECT qty, threshold FROM items WHERE name=?",
                              (name,)).fetchone()
            if row:
                con.execute(
                    "UPDATE items SET qty=?, unit=?, loc=?, threshold=?, updated=? WHERE name=?",
                    (qty, unit, loc, threshold if threshold is not None else row[1], _ts(), name))
                delta = qty - (row[0] or 0)
            else:
                con.execute(
                    "INSERT INTO items(name,qty,unit,loc,threshold,updated) VALUES(?,?,?,?,?,?)",
                    (name, qty, unit, loc, threshold or 0, _ts()))
                delta = qty
            con.execute("INSERT INTO moves(item,delta,ts,note) VALUES(?,?,?,?)",
                        (name, delta, _ts(), note or "inventaire"))
        return f"stock « {name} » = {qty:g} {unit}".strip()

    def change(self, name: str, delta: float, note: str = "") -> str:
        with self._lock, self._conn() as con:
            row = con.execute("SELECT qty, unit, threshold FROM items WHERE name=?",
                              (name,)).fetchone()
            if not row:
                close = [i["name"] for i in self.find(name)[:3]]
                hint = f" Tu veux dire : {', '.join(close)} ?" if close else ""
                return f"article « {name} » inconnu.{hint}"
            qty = max(0.0, (row[0] or 0) + delta)
            con.execute("UPDATE items SET qty=?, updated=? WHERE name=?", (qty, _ts(), name))
            con.execute("INSERT INTO moves(item,delta,ts,note) VALUES(?,?,?,?)",
                        (name, delta, _ts(), note))
        out = f"{name} : {qty:g} {row[1]}".strip()
        if qty <= (row[2] or 0):
            out += " → à racheter !"
        return out

    def find(self, query: str = "") -> list[dict[str, Any]]:
        where, args = "", ()
        if query.strip():
            q = f"%{query.strip().lower()}%"
            where = "WHERE lower(name) LIKE ? OR lower(loc) LIKE ? OR lower(unit) LIKE ?"
            args = (q, q, q)
        with self._conn() as con:
            rows = con.execute(
                f"SELECT name, qty, unit, loc, threshold, updated FROM items {where} "
                "ORDER BY threshold-qty DESC, name", args).fetchall()
        return [{"name": r[0], "qty": r[1], "unit": r[2], "loc": r[3], "threshold": r[4],
                 "updated": r[5]} for r in rows]

    def low(self) -> list[dict[str, Any]]:
        return [i for i in self.find() if i["qty"] <= i["threshold"]]

    def render(self, query: str = "") -> str:
        items = self.find(query)
        if not items:
            return "rien dans la réserve" + (f" pour « {query} »." if query else ".")
        lines = []
        for i in items[:40]:
            alert = " (à racheter)" if i["qty"] <= i["threshold"] else ""
            loc = f" — {i['loc']}" if i["loc"] else ""
            lines.append(f"- {i['name']} : {i['qty']:g} {i['unit']}{loc}{alert}".replace("  ", " "))
        return "\n".join(lines)
