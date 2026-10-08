"""Persistance du cœur : stockage clé-valeur JSON dans SQLite.

Utilisé depuis plusieurs fils (le cœur bat dans son propre fil, la console sauvegarde à
l'arrêt) : une connexion partagée, autorisée entre fils, protégée par un verrou.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, db_path: str | Path):
        self.db = Path(db_path)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.commit()

    def save(self, key: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                               (key, data))
            self._conn.commit()

    def load(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
