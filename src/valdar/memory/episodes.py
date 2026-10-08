"""Mémoire épisodique de Valdar : son vécu (avenant 3 §2, avenant 4 §7).

- Chaque échange est gardé (texte brut = source de vérité), avec le **contexte temporel** du
  moment (plusieurs échelles, modèle de Howard et Kahana) et l'**état affectif**.
- Rappel : ressemblance + même moment + force du souvenir (rappels passés, loi de puissance)
  + activation en cours (amorçage, contiguïté) + humeur.
- Mémoire de travail : les quelques souvenirs les plus actifs.
- Reprise du fil : la dernière conversation, pour reprendre là où on s'était arrêté.
- Souvenirs hérités : conversations importées (RAUB, Claude), marquées par leur source.
"""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from valdar.config.loader import EpisodicConfig
from valdar.memory.vectors import cosine_rows, embed

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS episodes(
        id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, ext_id TEXT UNIQUE,
        title TEXT DEFAULT '', person TEXT DEFAULT '', started REAL NOT NULL,
        ended REAL NOT NULL, summary TEXT DEFAULT '')""",
    """CREATE TABLE IF NOT EXISTS turns(
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode INTEGER NOT NULL, idx INTEGER NOT NULL,
        t REAL NOT NULL, speaker TEXT NOT NULL, text TEXT NOT NULL,
        p REAL DEFAULT 0, a REAL DEFAULT 0, d REAL DEFAULT 0, vec BLOB, ctx BLOB)""",
    "CREATE INDEX IF NOT EXISTS turns_ep ON turns(episode, idx)",
    "CREATE TABLE IF NOT EXISTS accesses(turn INTEGER NOT NULL, t REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS acc_turn ON accesses(turn)",
    "CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value BLOB)",
]
_MAX_TEXT = 8000
_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
         "octobre", "novembre", "décembre"]


@dataclass
class Recollection:
    turn: int
    episode: int
    source: str
    title: str
    t: float
    speaker: str
    text: str
    score: float
    pad: tuple[float, float, float]

    def line(self, now: float, max_chars: int = 280) -> str:
        txt = " ".join(self.text.split())
        if len(txt) > max_chars:
            txt = txt[:max_chars].rsplit(" ", 1)[0] + "…"
        origin = {"claude": "conversation d'Olivier avec Claude",
                  "raub": "souvenir de RAUB"}.get(self.source, "")
        where = f"{when_text(self.t, now)}" + (f", {origin}" if origin else "")
        if self.title:
            where += f", « {self.title} »"
        return f"[{where}] {self.speaker} : {txt}"


def when_text(t: float, now: float) -> str:
    d = max(0.0, now - t)
    if d < 3600:
        return f"il y a {max(1, int(d // 60))} min"
    if d < 86400:
        return f"il y a {int(d // 3600)} h"
    days = int(d // 86400)
    if days == 1:
        return "hier"
    if days < 31:
        return f"il y a {days} jours"
    if days < 365:
        return f"il y a {days // 30} mois"
    lt = time.localtime(t)
    return f"en {_MOIS[lt.tm_mon - 1]} {lt.tm_year}"


def _allowed(idx: dict[str, Any], positions: np.ndarray, person: str | None) -> np.ndarray:
    who = idx["person"][positions]
    return (who == "") | (who == (person or ""))


class TemporalContext:
    """Contexte qui dérive à plusieurs échelles de temps : c_k ← norm(ρ_k c_k + (1−ρ_k) f),
    ρ_k = exp(−Δt/τ_k) (Δt au moins min_step : chaque entrée compte un peu)."""

    def __init__(self, scales: list[float], dim: int, min_step: float):
        self.taus = np.asarray(scales, dtype=np.float64)
        self.dim = dim
        self.min_step = min_step
        self.c = np.zeros((len(scales), dim), dtype=np.float32)
        self.t: float | None = None

    def update(self, f: np.ndarray, now: float) -> None:
        dt = self.min_step if self.t is None else max(self.min_step, now - self.t)
        rho = np.exp(-dt / self.taus).astype(np.float32)[:, None]
        if not np.any(self.c):
            rho = np.zeros_like(rho)
        c = rho * self.c + (1.0 - rho) * f[None, :]
        norms = np.linalg.norm(c, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.c = (c / norms).astype(np.float32)
        self.t = now

    def snapshot(self) -> np.ndarray:
        return self.c.copy()

    def to_bytes(self) -> bytes:
        return np.float32(self.t or 0.0).tobytes() + self.c.astype(np.float16).tobytes()

    def load_bytes(self, raw: bytes) -> None:
        t = float(np.frombuffer(raw[:4], dtype=np.float32)[0])
        c = np.frombuffer(raw[4:], dtype=np.float16).astype(np.float32)
        if c.size == self.c.size:
            self.c = c.reshape(self.c.shape)
            self.t = t or None


class Episodic:
    def __init__(self, path: str | Path, cfg: EpisodicConfig | None = None):
        self.cfg = cfg or EpisodicConfig()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as con:
            for stmt in _SCHEMA:
                con.execute(stmt)
        self.context = TemporalContext(self.cfg.scales_seconds, self.cfg.dim,
                                       self.cfg.min_step_seconds)
        raw = self._state("context")
        if raw:
            self.context.load_bytes(raw)
        self._index: dict[str, Any] | None = None
        self.activation: dict[int, tuple[float, float]] = {}   # tour → (niveau, instant)
        self._current: int | None = None

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    def _state(self, key: str) -> bytes | None:
        with self._conn() as con:
            row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set_state(self, con: sqlite3.Connection, key: str, value: bytes) -> None:
        con.execute("INSERT OR REPLACE INTO state(key,value) VALUES(?,?)", (key, value))

    # ================================================================ écriture
    def log_turn(self, speaker: str, text: str, now: float | None = None,
                 pad: tuple[float, float, float] = (0.0, 0.0, 0.0), person: str = "",
                 source: str = "valdar") -> int:
        """Ajoute un échange vécu (ouvre un nouvel épisode après un long silence)."""
        now = time.time() if now is None else now
        text = (text or "").strip()[:_MAX_TEXT]
        if not text:
            return -1
        with self._lock, self._conn() as con:
            ep = self._open_episode(con, now, person, source, text)
            idx = con.execute("SELECT COUNT(*) FROM turns WHERE episode=?", (ep,)).fetchone()[0]
            f = embed(text, self.cfg.dim)
            self.context.update(f, now)
            cur = con.execute(
                "INSERT INTO turns(episode,idx,t,speaker,text,p,a,d,vec,ctx) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (ep, idx, now, speaker, text, *pad, f.astype(np.float16).tobytes(),
                 self.context.snapshot().astype(np.float16).tobytes()))
            con.execute("UPDATE episodes SET ended=? WHERE id=?", (now, ep))
            self._set_state(con, "context", self.context.to_bytes())
            tid = int(cur.lastrowid)
        self._index = None
        self._activate(tid, now, 1.0)
        return tid

    def _open_episode(self, con: sqlite3.Connection, now: float, person: str, source: str,
                      first_text: str) -> int:
        if self._current is None:
            row = con.execute("SELECT id, ended FROM episodes WHERE source=? "
                              "ORDER BY ended DESC LIMIT 1", (source,)).fetchone()
            if row and now - row[1] <= self.cfg.episode_gap_seconds:
                self._current = int(row[0])
        if self._current is not None:
            ended = con.execute("SELECT ended FROM episodes WHERE id=?",
                                (self._current,)).fetchone()
            if ended and now - ended[0] <= self.cfg.episode_gap_seconds:
                return self._current
        title = " ".join(first_text.split())[:80]
        cur = con.execute("INSERT INTO episodes(source,title,person,started,ended) "
                          "VALUES(?,?,?,?,?)", (source, title, person, now, now))
        self._current = int(cur.lastrowid)
        return self._current

    def import_episode(self, source: str, ext_id: str, title: str, person: str,
                       turns: Iterable[tuple[float, str, str, tuple[float, float, float]]],
                       context: TemporalContext) -> int | None:
        """Ajoute un épisode hérité (déjà daté). `context` est rejoué dans l'ordre du temps
        pour que ces souvenirs aient, eux aussi, leur contexte. Ne fait rien si déjà importé."""
        rows = [(t, sp, (tx or "").strip()[:_MAX_TEXT], pad) for t, sp, tx, pad in turns]
        rows = [r for r in rows if r[2]]
        if not rows:
            return None
        with self._lock, self._conn() as con:
            if con.execute("SELECT 1 FROM episodes WHERE ext_id=?", (ext_id,)).fetchone():
                return None
            cur = con.execute(
                "INSERT INTO episodes(source,ext_id,title,person,started,ended) "
                "VALUES(?,?,?,?,?,?)", (source, ext_id, title[:120], person, rows[0][0],
                                        rows[-1][0]))
            ep = int(cur.lastrowid)
            batch = []
            for i, (t, sp, tx, pad) in enumerate(rows):
                f = embed(tx, self.cfg.dim)
                context.update(f, t)
                batch.append((ep, i, t, sp, tx, *pad, f.astype(np.float16).tobytes(),
                              context.snapshot().astype(np.float16).tobytes()))
            con.executemany("INSERT INTO turns(episode,idx,t,speaker,text,p,a,d,vec,ctx) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?)", batch)
        self._index = None
        return ep

    def adopt_context(self, context: TemporalContext) -> None:
        """Après un import, si Valdar n'a encore rien vécu, son contexte part de là."""
        if self.context.t is None or (context.t or 0) > (self.context.t or 0):
            self.context = context
            with self._lock, self._conn() as con:
                self._set_state(con, "context", self.context.to_bytes())

    # ================================================================== index
    def _load_index(self) -> dict[str, Any]:
        if self._index is not None:
            return self._index
        with self._conn() as con:
            rows = con.execute(
                "SELECT t.id, t.episode, t.idx, t.t, t.p, t.a, t.d, t.vec, t.ctx, e.source, "
                "e.person "
                "FROM turns t JOIN episodes e ON e.id=t.episode ORDER BY t.id").fetchall()
            acc = con.execute("SELECT turn, t FROM accesses").fetchall()
        k, dim = len(self.cfg.scales_seconds), self.cfg.dim
        n = len(rows)
        vec = np.zeros((n, dim), dtype=np.float32)
        ctx = np.zeros((n, k, dim), dtype=np.float32)
        for i, r in enumerate(rows):
            vec[i] = np.frombuffer(r[7], dtype=np.float16)
            ctx[i] = np.frombuffer(r[8], dtype=np.float16).reshape(k, dim)
        accesses: dict[int, list[float]] = {}
        for turn, t in acc:
            accesses.setdefault(turn, []).append(t)
        self._index = {
            "ids": np.array([r[0] for r in rows], dtype=np.int64),
            "pos": {r[0]: i for i, r in enumerate(rows)},
            "episode": np.array([r[1] for r in rows], dtype=np.int64),
            "idx": np.array([r[2] for r in rows], dtype=np.int64),
            "t": np.array([r[3] for r in rows], dtype=np.float64),
            "pad": np.array([[r[4], r[5], r[6]] for r in rows], dtype=np.float32).reshape(n, 3),
            "vec": vec, "ctx": ctx, "accesses": accesses,
            "person": np.array([r[10] or "" for r in rows], dtype=object),
        }
        return self._index

    # =================================================================== rappel
    def _activate(self, tid: int, now: float, level: float) -> None:
        cur = self._activation_level(tid, now)
        self.activation[tid] = (min(1.0, max(cur, level)), now)

    def _activation_level(self, tid: int, now: float) -> float:
        a = self.activation.get(tid)
        if a is None:
            return 0.0
        return a[0] * math.exp(-(now - a[1]) / self.cfg.activation_tau_seconds)

    def _spread(self, idx: dict[str, Any], i: int, now: float, level: float) -> None:
        """Contiguïté : un souvenir rappelé pré-active ses voisins, plus vers l'avant."""
        ep, pos = idx["episode"][i], idx["idx"][i]
        for j in range(max(0, i - self.cfg.spread_reach),
                       min(len(idx["ids"]), i + self.cfg.spread_reach + 1)):
            if j == i or idx["episode"][j] != ep:
                continue
            dist = int(idx["idx"][j] - pos)
            w = self.cfg.spread_forward if dist > 0 else self.cfg.spread_backward
            self._activate(int(idx["ids"][j]), now, level * w ** abs(dist))

    def base_level(self, tid: int, t_encoded: float, now: float,
                   accesses: dict[int, list[float]]) -> float:
        """Force du souvenir (Anderson et Schooler) : ln Σ (t − t_j)^−0,5."""
        times = [t_encoded] + accesses.get(tid, [])
        return math.log(sum(max(60.0, now - tj) ** -0.5 for tj in times))

    def recall(self, query: str, now: float | None = None,
               pad: tuple[float, float, float] | None = None, k: int | None = None,
               touch: bool = True, person: str | None = None) -> list[Recollection]:
        """`person` : seuls les souvenirs communs ou ceux de cette personne remontent (la vie
        privée d'Olivier ne sort pas devant un inconnu)."""
        now = time.time() if now is None else now
        k = self.cfg.recall_k if k is None else k
        with self._lock:
            idx = self._load_index()
            n = len(idx["ids"])
            if n == 0 or k == 0:
                return []
            q = embed(query, self.cfg.dim)
            sem = cosine_rows(idx["vec"], q)
            act = np.array([self._activation_level(int(t), now) for t in idx["ids"]])
            candidates = np.where((sem >= self.cfg.min_semantic) | (act > 0.05))[0]
            candidates = candidates[idx["t"][candidates] < now - self.cfg.exclude_recent_seconds]
            candidates = candidates[_allowed(idx, candidates, person)]
            if len(candidates) == 0:
                return []
            score = self.cfg.w_semantic * sem[candidates] + self.cfg.w_activation * act[candidates]
            ctx_now = self.context.snapshot()
            for s, w in enumerate(self.cfg.w_context):
                if w and np.any(ctx_now[s]):
                    score += w * cosine_rows(idx["ctx"][candidates, s, :], ctx_now[s])
            if pad is not None and self.cfg.w_mood:
                pv = np.asarray(pad, dtype=np.float32)
                if np.linalg.norm(pv) > 0.1:
                    score += self.cfg.w_mood * cosine_rows(idx["pad"][candidates], pv)
            base = np.array([self.base_level(int(idx["ids"][i]), float(idx["t"][i]), now,
                                             idx["accesses"]) for i in candidates])
            score += self.cfg.w_base * base
            order = np.argsort(-score)[:k]
            chosen = [int(candidates[o]) for o in order]
            out = self._recollections(idx, chosen, score[order])
            if touch:
                self._touch(idx, chosen, now)
            return out

    def _touch(self, idx: dict[str, Any], positions: list[int], now: float) -> None:
        tids = [int(idx["ids"][i]) for i in positions]
        with self._conn() as con:
            con.executemany("INSERT INTO accesses(turn,t) VALUES(?,?)", [(t, now) for t in tids])
        for i, t in zip(positions, tids, strict=True):
            idx["accesses"].setdefault(t, []).append(now)
            self._activate(t, now, 1.0)
            self._spread(idx, i, now, 1.0)

    def _recollections(self, idx: dict[str, Any], positions: list[int],
                       scores: Iterable[float]) -> list[Recollection]:
        if not positions:
            return []
        tids = [int(idx["ids"][i]) for i in positions]
        with self._conn() as con:
            rows = {r[0]: r for r in con.execute(
                "SELECT t.id, t.episode, e.source, e.title, t.t, t.speaker, t.text, "
                "t.p, t.a, t.d FROM turns t JOIN episodes e ON e.id=t.episode "
                f"WHERE t.id IN ({','.join('?' * len(tids))})", tids)}
        return [Recollection(turn=r[0], episode=r[1], source=r[2], title=r[3], t=r[4],
                             speaker=r[5], text=r[6], score=float(s), pad=(r[7], r[8], r[9]))
                for tid, s in zip(tids, scores, strict=True) if (r := rows.get(tid))]

    def working_memory(self, now: float | None = None, exclude: Iterable[int] = (),
                       person: str | None = None) -> list[Recollection]:
        """Ce qui « trotte dans la tête » : les souvenirs les plus actifs."""
        now = time.time() if now is None else now
        with self._lock:
            idx = self._load_index()
            skip = set(exclude)
            live = [(self._activation_level(t, now), t) for t in list(self.activation)]
            live = [(a, t) for a, t in live
                    if a >= self.cfg.working_threshold and t not in skip and t in idx["pos"]
                    and idx["t"][idx["pos"][t]] < now - self.cfg.exclude_recent_seconds
                    and idx["person"][idx["pos"][t]] in ("", person or "")]
            live.sort(reverse=True)
            positions = [idx["pos"][t] for _, t in live[:self.cfg.working_memory]]
            return self._recollections(idx, positions, [a for a, _ in live])

    # ============================================================== reprise du fil
    def last_thread(self, now: float | None = None, source: str = "valdar",
                    turns: int = 6) -> dict[str, Any] | None:
        now = time.time() if now is None else now
        with self._conn() as con:
            row = con.execute("SELECT id, title, started, ended FROM episodes WHERE source=? "
                              "ORDER BY ended DESC LIMIT 1", (source,)).fetchone()
            if not row or now - row[3] > self.cfg.resume_hours * 3600:
                return None
            tail = con.execute("SELECT speaker, text, t FROM turns WHERE episode=? "
                               "ORDER BY idx DESC LIMIT ?", (row[0], turns)).fetchall()
        return {"episode": row[0], "title": row[1], "started": row[2], "ended": row[3],
                "turns": [{"speaker": s, "text": t, "t": tt} for s, t, tt in reversed(tail)]}

    def forget_person(self, person: str) -> int:
        """« Oublie-moi » : efface les épisodes de la personne (tours et accès compris)."""
        with self._lock, self._conn() as con:
            eps = [r[0] for r in con.execute("SELECT id FROM episodes WHERE person=?",
                                             (person,)).fetchall()]
            for ep in eps:
                con.execute("DELETE FROM accesses WHERE turn IN "
                            "(SELECT id FROM turns WHERE episode=?)", (ep,))
                con.execute("DELETE FROM turns WHERE episode=?", (ep,))
                con.execute("DELETE FROM episodes WHERE id=?", (ep,))
        if self._current in eps:
            self._current = None
        self._index = None
        return len(eps)

    def count(self) -> dict[str, int]:
        with self._conn() as con:
            rows = con.execute("SELECT e.source, COUNT(t.id) FROM episodes e "
                               "LEFT JOIN turns t ON t.episode=e.id GROUP BY e.source").fetchall()
        return {s: n for s, n in rows}
