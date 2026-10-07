from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from valdar.memory.facts import _terms
from valdar.memory.vectors import cosine_rows, embed

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc TEXT NOT NULL, section TEXT DEFAULT '', title TEXT DEFAULT '', text TEXT NOT NULL,
        sha TEXT UNIQUE, source TEXT DEFAULT '', tags TEXT DEFAULT '', vec BLOB)""",
    "CREATE TABLE IF NOT EXISTS docs(doc TEXT PRIMARY KEY, sha TEXT, n INTEGER)",
]
_WORD_LIMIT = 1800


@dataclass
class Passage:
    id: int
    doc: str
    title: str
    text: str
    source: str
    score: float

    def line(self, max_chars: int = 600) -> str:
        txt = " ".join(self.text.split())
        if len(txt) > max_chars:
            txt = txt[:max_chars].rsplit(" ", 1)[0] + "…"
        head = f"{self.title} " if self.title else ""
        return f"{head}({self.doc}) : {txt}"


class Knowledge:
    def __init__(self, path: str | Path, dim: int = 512):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.RLock()
        with self._conn() as con:
            for s in _SCHEMA:
                con.execute(s)
        self._index: dict[str, Any] | None = None

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    # ------------------------------------------------------------- ingestion
    def ingest_file(self, path: Path, doc: str | None = None, source: str = "") -> int:
        path = Path(path)
        doc = doc or path.name
        raw = path.read_text(encoding="utf-8", errors="replace")
        sha = hashlib.sha256(raw.encode()).hexdigest()
        with self._conn() as con:
            row = con.execute("SELECT sha FROM docs WHERE doc=?", (doc,)).fetchone()
        if row and row[0] == sha:
            return 0
        is_yaml = path.suffix.lower() in (".yaml", ".yml")
        passages = _from_yaml(raw) if is_yaml else _from_text(raw)
        return self.ingest(doc, passages, source or str(path.name), sha)

    def ingest(self, doc: str, passages: list[dict[str, str]], source: str = "",
               sha: str = "") -> int:
        n = 0
        with self._lock, self._conn() as con:
            con.execute("DELETE FROM passages WHERE doc=?", (doc,))
            for p in passages:
                text = p["text"].strip()
                if len(text) < 20:
                    continue
                psha = hashlib.sha256((doc + "\n" + text).encode()).hexdigest()
                vec = embed(p.get("title", "") + " " + text, self.dim)
                con.execute("INSERT OR IGNORE INTO passages(doc,section,title,text,sha,source,"
                            "tags,vec) VALUES(?,?,?,?,?,?,?,?)",
                            (doc, p.get("section", ""), p.get("title", ""), text, psha,
                             p.get("source", source), p.get("tags", ""),
                             vec.astype(np.float16).tobytes()))
                n += 1
            con.execute("INSERT OR REPLACE INTO docs(doc,sha,n) VALUES(?,?,?)", (doc, sha, n))
        self._index = None
        return n

    def ingest_dir(self, folder: Path) -> dict[str, int]:
        out = {}
        folder = Path(folder)
        if not folder.is_dir():
            return out
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in (".md", ".txt", ".yaml", ".yml") and p.is_file():
                out[p.name] = self.ingest_file(p, doc=p.relative_to(folder).as_posix())
        return out

    # --------------------------------------------------------------- recherche
    def _load(self) -> dict[str, Any]:
        if self._index is not None:
            return self._index
        with self._conn() as con:
            rows = con.execute("SELECT id, title, text, vec FROM passages").fetchall()
        docs_terms = [_terms(r[1] + " " + r[2]) for r in rows]
        df: Counter = Counter()
        for t in docs_terms:
            df.update(set(t))
        n = len(rows)
        vec = np.zeros((n, self.dim), dtype=np.float32)
        for i, r in enumerate(rows):
            vec[i] = np.frombuffer(r[3], dtype=np.float16)
        avg = sum(len(t) for t in docs_terms) / n if n else 1.0
        self._index = {"ids": [r[0] for r in rows], "tf": [Counter(t) for t in docs_terms],
                       "len": [len(t) for t in docs_terms], "df": df, "n": n, "avg": avg,
                       "vec": vec}
        return self._index

    def search(self, query: str, k: int = 4, min_score: float = 0.15) -> list[Passage]:
        idx = self._load()
        if idx["n"] == 0:
            return []
        terms = _terms(query)
        bm = np.zeros(idx["n"], dtype=np.float32)
        k1, b = 1.5, 0.75
        for t in set(terms):
            dft = idx["df"].get(t, 0)
            if not dft:
                continue
            idf = math.log(1 + (idx["n"] - dft + 0.5) / (dft + 0.5))
            for i in range(idx["n"]):
                f = idx["tf"][i].get(t, 0)
                if f:
                    norm = 1 - b + b * idx["len"][i] / idx["avg"]
                    bm[i] += idf * f * (k1 + 1) / (f + k1 * norm)
        if bm.max() > 0:
            bm = bm / bm.max()
        sem = cosine_rows(idx["vec"], embed(query, self.dim))
        score = 0.6 * bm + 0.4 * np.clip(sem, 0, 1)
        order = np.argsort(-score)[:k]
        chosen = [(int(idx["ids"][i]), float(score[i])) for i in order if score[i] >= min_score]
        if not chosen:
            return []
        with self._conn() as con:
            rows = {r[0]: r for r in con.execute(
                "SELECT id, doc, title, text, source FROM passages WHERE id IN "
                f"({','.join('?' * len(chosen))})", [c[0] for c in chosen])}
        return [Passage(i, rows[i][1], rows[i][2], rows[i][3], rows[i][4], s)
                for i, s in chosen if i in rows]

    def count(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]


# ------------------------------------------------------------------ découpage
def _from_text(raw: str) -> list[dict[str, str]]:
    """Markdown / texte : un passage par section (titre #), recoupé par paragraphes."""
    out: list[dict[str, str]] = []
    title, buf = "", []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        words = text.split()
        for i in range(0, len(words), _WORD_LIMIT // 6):
            chunk = " ".join(words[i:i + _WORD_LIMIT // 6])
            out.append({"title": title, "text": chunk})

    for line in raw.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            flush()
            buf = []
            title = m.group(1).strip()
        else:
            buf.append(line)
    flush()
    return out


def _from_yaml(raw: str) -> list[dict[str, str]]:
    data = yaml.safe_load(raw)
    items = data if isinstance(data, list) else data.get("notes") or data.get("entries") or []
    out: list[dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "titre" in it:   # base de défauts de Valdar
            parts = [f"{it['titre']} (catégorie {it.get('categorie', '')}, gravité "
                     f"{it.get('gravite', '?')}/5, caméra : {it.get('detectable_camera', '?')})"]
            for key, label in (("symptomes_visuels", "Ce qu'on voit"),
                               ("signes_precurseurs", "Signes avant-coureurs"),
                               ("point_de_non_retour", "Point de non-retour"),
                               ("causes", "Causes"),
                               ("corrections_immediates", "Pendant l'impression"),
                               ("corrections_durables", "Pour la suite"),
                               ("specifique_cr10s", "CR-10S")):
                v = it.get(key)
                if not v:
                    continue
                v = " ; ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                parts.append(f"{label} : {v}")
            out.append({"title": str(it["titre"]), "section": str(it.get("id", "")),
                        "text": "\n".join(parts), "tags": str(it.get("categorie", "")),
                        "source": " ".join(it.get("sources") or [])[:500]})
        elif "text" in it:   # notes de PrintOS
            out.append({"title": str(it.get("topic", "")), "text": str(it["text"]),
                        "tags": " ".join(it.get("tags") or [])})
    return out
