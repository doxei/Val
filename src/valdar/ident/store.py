"""Empreintes (voix, visage) : des vecteurs, rangés par personne, dans le dossier de données.

Jamais d'audio ni d'image : seulement des vecteurs, qui ne permettent pas de refaire la voix
ou le visage. « Oublie-moi » les efface. Le fichier n'est jamais versionné.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class Prints:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as con:
            con.execute("CREATE TABLE IF NOT EXISTS prints(id INTEGER PRIMARY KEY "
                        "AUTOINCREMENT, kind TEXT, person TEXT, vec BLOB, t REAL, "
                        "learned INTEGER DEFAULT 0)")
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def add(self, kind: str, person: str, vec: np.ndarray, learned: bool = False,
            cap: int = 40) -> int:
        v = _unit(vec)
        with self._lock, self._conn() as con:
            con.execute("INSERT INTO prints(kind,person,vec,t,learned) VALUES(?,?,?,?,?)",
                        (kind, person, v.astype(np.float32).tobytes(), time.time(),
                         int(learned)))
            # on garde les empreintes d'enrôlement, et les plus récentes apprises
            ids = [r[0] for r in con.execute(
                "SELECT id FROM prints WHERE kind=? AND person=? AND learned=1 ORDER BY t DESC",
                (kind, person))]
            for old in ids[cap:]:
                con.execute("DELETE FROM prints WHERE id=?", (old,))
            n = con.execute("SELECT COUNT(*) FROM prints WHERE kind=? AND person=?",
                            (kind, person)).fetchone()[0]
        self._cache.pop(kind, None)
        return int(n)

    def _all(self, kind: str) -> dict[str, np.ndarray]:
        with self._lock:
            if kind not in self._cache:
                by: dict[str, list[np.ndarray]] = {}
                with self._conn() as con:
                    for person, blob in con.execute(
                            "SELECT person, vec FROM prints WHERE kind=?", (kind,)):
                        by.setdefault(person, []).append(np.frombuffer(blob, np.float32))
                self._cache[kind] = {p: np.stack(vs) for p, vs in by.items()}
            return self._cache[kind]

    def match(self, kind: str, vec: np.ndarray) -> list[tuple[str, float]]:
        """Personnes triées par ressemblance (cosinus), la meilleure d'abord.

        Score d'une personne = moyenne de ses 3 empreintes les plus proches (plus stable
        qu'une seule, plus fin qu'une moyenne de toutes)."""
        v = _unit(vec)
        out = []
        for person, mat in self._all(kind).items():
            if mat.shape[1] != v.shape[0]:
                continue
            sims = np.sort(mat @ v)[::-1]
            out.append((person, float(sims[:3].mean())))
        out.sort(key=lambda x: -x[1])
        return out

    def counts(self, kind: str) -> dict[str, int]:
        with self._conn() as con:
            return {p: n for p, n in con.execute(
                "SELECT person, COUNT(*) FROM prints WHERE kind=? GROUP BY person", (kind,))}

    def forget(self, person: str, kind: str | None = None) -> int:
        with self._lock, self._conn() as con:
            if kind is None:
                n = con.execute("DELETE FROM prints WHERE person=?", (person,)).rowcount
            else:
                n = con.execute("DELETE FROM prints WHERE person=? AND kind=?",
                                (person, kind)).rowcount
        self._cache.clear()
        return n
