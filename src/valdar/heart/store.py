"""Persistance du cœur : stockage clé-valeur JSON dans SQLite."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, db_path: str | Path):
        self.db = Path(db_path)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def save(self, key: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            (key, json.dumps(payload, ensure_ascii=False)),
        )
        self._conn.commit()

    def load(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()
