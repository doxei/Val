"""Mémoire sémantique (v2 §7.4) : faits durables, avec source, personne et confiance.

Recherche par mots-clés + récence en phase 2 ; les embeddings arrivent en phase 7.

Avenant 5 §4 (le Surmoi critique) : chaque fait est une **croyance** avec sa fiche :
- `origine` : « dit » (quelqu'un l'a dit), « lu » (une source), « deduit » (Valdar l'a
  déduit). Une déduction reste une **hypothèse** tant qu'aucune source extérieure ne la
  confirme ;
- `refutation` : ce qui prouverait qu'elle est fausse (la question de Popper) ;
- `pour` / `contre` : les éléments rencontrés depuis ;
- `statut` : valide, hypothese, quarantaine (consultable, cité comme douteux, exclu de
  l'apprentissage de la nuit tant qu'il n'est pas levé).
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

# Colonnes de la fiche de croyance (ajoutées sans casser une base existante).
_BELIEF_COLS = {
    "origine": "TEXT DEFAULT 'dit'",
    "refutation": "TEXT DEFAULT ''",
    "pour": "INTEGER DEFAULT 0",
    "contre": "INTEGER DEFAULT 0",
    "statut": "TEXT DEFAULT 'valide'",
    "motif": "TEXT DEFAULT ''",
    "verifie": "REAL DEFAULT 0",
}
ORIGINES = ("dit", "lu", "deduit")
STATUTS = ("valide", "hypothese", "quarantaine")
_COLS = ("id,text,person,source,confidence,created,last_used,hits,"
         "origine,refutation,pour,contre,statut,motif,verifie")

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
            have = {r[1] for r in con.execute("PRAGMA table_info(facts)")}
            for col, decl in _BELIEF_COLS.items():
                if col not in have:
                    con.execute(f"ALTER TABLE facts ADD COLUMN {col} {decl}")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def remember(self, text: str, person: str = "", source: str = "valdar",
                 confidence: float = 0.8, when: float | None = None, origine: str = "dit",
                 refutation: str = "") -> str:
        text = text.strip()
        if not text:
            return "rien à retenir."
        origine = origine if origine in ORIGINES else "dit"
        statut = "hypothese" if origine == "deduit" else "valide"
        if origine == "deduit":
            confidence = min(confidence, 0.5)
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
                "INSERT INTO facts(text,norm,person,source,confidence,created,last_used,"
                "origine,refutation,statut) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (text, n, person, source, confidence, now, now, origine, refutation.strip(),
                 statut))
        if statut == "hypothese":
            return "noté comme hypothèse (c'est une déduction, pas encore un fait)."
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
        items = self._rows("archived=0")
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

    def _rows(self, where: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(f"SELECT {_COLS} FROM facts WHERE {where}", args).fetchall()
        keys = _COLS.split(",")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    # ------------------------------------------------- le Surmoi critique
    def find(self, fragment: str, person: str | None = None) -> dict[str, Any] | None:
        """La croyance qui correspond le mieux à un fragment de texte."""
        hits = self.recall(fragment, limit=1, person=person)
        return hits[0] if hits else None

    def evidence(self, fid: int, supports: bool, note: str = "",
                 when: float | None = None) -> dict[str, Any] | None:
        """Ajoute un élément pour ou contre. Contre : la confiance baisse ; trop de contre,
        quarantaine. Pour, d'une source extérieure : une hypothèse peut devenir un fait."""
        now = time.time() if when is None else when
        with self._lock, self._conn() as con:
            row = con.execute("SELECT confidence,pour,contre,statut,origine FROM facts "
                              "WHERE id=? AND archived=0", (fid,)).fetchone()
            if row is None:
                return None
            conf, pour, contre, statut, origine = row
            if supports:
                pour += 1
                conf = min(0.95, conf + 0.1)
                if statut == "hypothese" and pour >= 2:
                    statut = "valide"
            else:
                contre += 1
                conf = max(0.05, conf - 0.2)
            motif = ""
            if not supports and (contre > pour or conf < 0.3) and statut != "quarantaine":
                statut, motif = "quarantaine", (note or "contredit")[:200]
            con.execute("UPDATE facts SET confidence=?, pour=?, contre=?, statut=?, verifie=?"
                        + (", motif=?" if motif else "") + " WHERE id=?",
                        (conf, pour, contre, statut, now, *((motif,) if motif else ()), fid))
        return self._rows("id=?", (fid,))[0]

    def quarantine(self, fid: int, motif: str) -> bool:
        with self._lock, self._conn() as con:
            return con.execute("UPDATE facts SET statut='quarantaine', motif=? WHERE id=? "
                               "AND archived=0", (motif[:200], fid)).rowcount > 0

    def lift(self, fid: int, when: float | None = None) -> bool:
        """Lève la quarantaine (une preuve, ou Olivier)."""
        now = time.time() if when is None else when
        with self._lock, self._conn() as con:
            return con.execute(
                "UPDATE facts SET statut=CASE WHEN origine='deduit' AND pour<2 THEN 'hypothese'"
                " ELSE 'valide' END, motif='', verifie=? WHERE id=? AND statut='quarantaine'",
                (now, fid)).rowcount > 0

    def audit_sample(self, n: int = 3, now: float | None = None) -> list[dict[str, Any]]:
        """Les croyances à réexaminer d'abord : très utilisées et peu vérifiées."""
        now = time.time() if now is None else now
        items = self._rows("archived=0 AND statut!='quarantaine'")
        day = 86400.0

        def urgency(f: dict[str, Any]) -> float:
            age = (now - (f["verifie"] or f["created"])) / day
            return f["hits"] * (1 + age) * (1.5 if f["statut"] == "hypothese" else 1.0)

        return sorted(items, key=urgency, reverse=True)[:n]

    def mark_checked(self, fid: int, when: float | None = None) -> None:
        with self._lock, self._conn() as con:
            con.execute("UPDATE facts SET verifie=? WHERE id=?",
                        (time.time() if when is None else when, fid))

    def quarantined(self) -> list[dict[str, Any]]:
        return self._rows("archived=0 AND statut='quarantaine'")

    def consolidable(self) -> list[dict[str, Any]]:
        """Ce que la nuit a le droit d'apprendre : seulement les croyances valides."""
        return self._rows("archived=0 AND statut='valide'")

    def touch(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock, self._conn() as con:
            con.executemany("UPDATE facts SET last_used=? WHERE id=?",
                            [(time.time(), i) for i in ids])

    def count(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM facts WHERE archived=0").fetchone()[0]
