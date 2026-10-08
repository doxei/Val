"""Mémoire sémantique (v2 §7.4) : faits durables, avec source, personne et confiance.

Recherche par mots-clés + récence en phase 2 ; les embeddings arrivent en phase 7.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

_SCHEMA = """CREATE TABLE IF NOT EXISTS facts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    norm TEXT NOT NULL,
    person TEXT DEFAULT '',
    source TEXT DEFAULT 'valdar',
    confidence REAL DEFAULT 0.8,
    created REAL NOT NULL,
    last_used REAL NOT NULL,
    hits INTEGER DEFAULT 1,
    archived INTEGER DEFAULT 0)"""

_STOP = {"les", "des", "une", "est", "que", "qui", "pour", "dans", "sur", "avec", "pas", "mais",
         "son", "ses", "sa", "le", "la", "de", "du", "et", "il", "elle", "je", "tu", "on", "a",
         "au", "aux", "en", "un", "ce", "cette", "mon", "ma", "mes", "ton", "ta", "tes"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("’", "'"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\b(qu|[ljdcmnst])'", "", s)        # élisions : j'imprime → imprime
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+ ]", " ", s)).strip()


def _terms(s: str) -> list[str]:
    return [w for w in _norm(s).split() if len(w) > 2 and w not in _STOP]


class Facts:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as con:
            con.execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def remember(self, text: str, person: str = "", source: str = "valdar",
                 confidence: float = 0.8, when: float | None = None) -> str:
        text = text.strip()
        if not text:
            return "rien à retenir."
        now = time.time() if when is None else when
        n = _norm(text)
        with self._lock, self._conn() as con:
            row = con.execute("SELECT id FROM facts WHERE norm=? AND archived=0",
                              (n,)).fetchone()
            if row:
                con.execute("UPDATE facts SET hits=hits+1, last_used=? WHERE id=?",
                            (now, row[0]))
                return "je le savais déjà, c'est noté deux fois."
            con.execute(
                "INSERT INTO facts(text,norm,person,source,confidence,created,last_used) "
                "VALUES(?,?,?,?,?,?,?)", (text, n, person, source, confidence, now, now))
        return "noté."

    def forget_person(self, person: str) -> int:
        """« Oublie-moi » : efface tous les faits liés à la personne."""
        with self._lock, self._conn() as con:
            return con.execute("DELETE FROM facts WHERE person=?", (person,)).rowcount

    def forget(self, fragment: str) -> str:
        n = _norm(fragment)
        if not n:
            return "dis-moi quoi oublier."
        with self._lock, self._conn() as con:
            cur = con.execute("UPDATE facts SET archived=1 WHERE norm LIKE ? AND archived=0",
                              (f"%{n}%",))
            count = cur.rowcount
        return f"{count} souvenir(s) oublié(s)." if count else "je n'avais rien là-dessus."

    def recall(self, query: str = "", limit: int = 10, person: str | None = None
               ) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id,text,person,source,confidence,created,last_used,hits FROM facts "
                "WHERE archived=0").fetchall()
        items = [{"id": r[0], "text": r[1], "person": r[2], "source": r[3], "confidence": r[4],
                  "created": r[5], "last_used": r[6], "hits": r[7]} for r in rows]
        if person is not None:
            items = [i for i in items if i["person"] in ("", person)]
        terms = _terms(query)
        if not terms:
            return sorted(items, key=lambda i: (i["hits"], i["last_used"]), reverse=True)[:limit]
        scored = []
        for it in items:
            hay = _norm(it["text"])
            score = sum(1 for t in terms if t in hay)
            if score:
                scored.append((score, it["hits"], it["last_used"], it))
        scored.sort(key=lambda x: x[:3], reverse=True)
        return [s[3] for s in scored[:limit]]

    def context(self, query: str, limit: int = 8, relevant: int = 5,
                person: str | None = None) -> list[dict[str, Any]]:
        """Souvenirs pour le prompt : les plus pertinents, complétés par les plus importants."""
        out = self.recall(query, limit=relevant, person=person) if query.strip() else []
        seen = {f["id"] for f in out}
        for f in self.recall("", limit=limit, person=person):
            if len(out) >= limit:
                break
            if f["id"] not in seen:
                out.append(f)
                seen.add(f["id"])
        return out

    def touch(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock, self._conn() as con:
            con.executemany("UPDATE facts SET last_used=? WHERE id=?",
                            [(time.time(), i) for i in ids])

    def count(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM facts WHERE archived=0").fetchone()[0]
