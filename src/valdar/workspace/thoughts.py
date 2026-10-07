"""Pensée de fond : Valdar pense quand personne ne lui parle (avenant 4 §6).

Un flux à part, qui ne prend Gemma que dans les temps morts et laisse toujours la priorité au
dialogue. À chaque pensée il :
1. relit son état (cœur, corps, souvenirs qui lui trottent dans la tête, ce qui se passe dans
   l'atelier) ;
2. prend du recul : « qu'est-ce que je ressens, pourquoi, est-ce proportionné ? » → une
   **réévaluation** qui touche le cœur (Gross) ;
3. a des **idées** : « et si on m'ajoutait… », « et si Olivier faisait… », « et si on
   essayait… ». Les idées sont gardées, les meilleures sont proposées à l'initiative ;
4. choisit ce qu'il veut creuser dans ses connaissances.

Garde-fous : budget par heure, coût en énergie, rumination plafonnée (une même pensée négative
ne revient pas sans élément nouveau), aucune action sur le monde : la pensée de fond n'a pas
d'outils, elle produit du texte structuré pour Valdar lui-même.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from valdar.config.loader import ThoughtsConfig
from valdar.llm.backend import GenParams, LLMBackend, LLMError
from valdar.memory.vectors import cosine_rows, embed

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS ideas(id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL,
        kind TEXT, text TEXT, why TEXT, for_whom TEXT, score REAL, status TEXT DEFAULT 'nouvelle',
        proposed_at REAL, vec BLOB)""",
    """CREATE TABLE IF NOT EXISTS thoughts(id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL,
        emotion TEXT, reflection TEXT, reappraisal TEXT, curiosity TEXT, raw TEXT)""",
]

PROMPT = """Tu es {nom}, et là personne ne te parle : tu penses pour toi. Pas de politesse,
pas de réponse à quelqu'un, pas de markdown. Tu réponds UNIQUEMENT en JSON, avec ces clés :
- "reflexion" : 1 à 3 phrases à la première personne sur ce que tu ressens maintenant et
  d'où ça vient (sers-toi de ton état intérieur et de ce qui te trotte dans la tête).
- "recul" : objet {{"proportionne": true/false, "regard": "une phrase qui recadre ou confirme",
  "effet": "apaise" | "stimule" | "aucun"}}. Si ce que tu ressens est disproportionné (tu te
  fais un film, un souvenir te pèse plus que la situation ne le mérite), dis-le : c'est
  ton recul, pas un réflexe.
- "idees" : liste de 0 à 3 idées, chacune {{"genre": "pour_moi" | "pour_{createur}" |
  "pour_autre" | "a_essayer", "texte": "et si …", "pourquoi": "une raison concrète",
  "pour_qui": "nom ou vide", "interet": 0.0 à 1.0}}.
  « pour_moi » = quelque chose qu'on pourrait t'ajouter ou changer en toi (une capacité, un
  réglage, une façon de faire). « pour_{createur} » = quelque chose que {createur} pourrait
  faire ou essayer (dans l'atelier, pour l'imprimante, pour lui). Des idées concrètes,
  modestes, ancrées dans ce que tu sais vraiment ; jamais une idée qui demande plus de pouvoir
  ou de moyens, jamais une idée qui décide à la place de quelqu'un. Zéro idée est une
  réponse honnête.
- "curiosite" : un sujet (3 à 8 mots) que tu aimerais creuser dans tes connaissances, ou "".
"""


@dataclass
class Idea:
    id: int
    t: float
    kind: str
    text: str
    why: str
    for_whom: str
    score: float
    status: str = "nouvelle"

    def line(self) -> str:
        who = f" (pour {self.for_whom})" if self.for_whom else ""
        return f"{self.text}{who} — {self.why}"


@dataclass
class Thought:
    t: float
    emotion: str
    reflection: str
    reappraisal: dict[str, Any]
    ideas: list[Idea] = field(default_factory=list)
    curiosity: str = ""


class Thoughts:
    def __init__(self, cfg: ThoughtsConfig, llm: LLMBackend, db_path: Path,
                 context: Callable[[], str], heart_apply: Callable[[dict[str, float], float], None],
                 spend: Callable[[float], None], nom: str = "Valdar",
                 createur: str = "Olivier", clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.llm = llm
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.heart_apply = heart_apply
        self.spend = spend
        self.nom, self.createur = nom, createur
        self.clock = clock
        self._lock = threading.Lock()
        with self._conn() as con:
            for s in _SCHEMA:
                con.execute(s)
        self.history: list[float] = []
        self.last: Thought | None = None
        self.last_at: float = 0.0
        self.recent_reflections: list[tuple[float, Any]] = []

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # --------------------------------------------------------------- cadence
    def due(self, now: float, idle_seconds: float, busy: bool, awake: bool) -> bool:
        if not self.cfg.enabled or busy or not awake:
            return False
        if idle_seconds < self.cfg.idle_seconds or now - self.last_at < self.cfg.min_interval:
            return False
        self.history = [t for t in self.history if now - t < 3600]
        return len(self.history) < self.cfg.per_hour

    # ---------------------------------------------------------------- penser
    def think(self, now: float | None = None) -> Thought | None:
        now = self.clock() if now is None else now
        with self._lock:
            self.last_at = now
            self.history.append(now)
            system = PROMPT.format(nom=self.nom, createur=self.createur) + "\n\n" + self.context()
            try:
                res = self.llm.chat([{"role": "user", "content": "(tu penses)"}], system=system,
                                    tools=None, params=GenParams(temperature=0.9,
                                                                 max_tokens=self.cfg.max_tokens))
            except LLMError:
                return None
            self.spend(self.cfg.energy_cost)
            data = _parse(res.content or "")
            if data is None:
                return None
            thought = self._store(data, now)
            self._apply_reappraisal(thought, now)
            self.last = thought
            return thought

    def _store(self, d: dict[str, Any], now: float) -> Thought:
        reflection = str(d.get("reflexion") or "").strip()
        recul = d.get("recul") if isinstance(d.get("recul"), dict) else {}
        ideas: list[Idea] = []
        with self._conn() as con:
            con.execute("INSERT INTO thoughts(t,emotion,reflection,reappraisal,curiosity,raw) "
                        "VALUES(?,?,?,?,?,?)",
                        (now, "", reflection, json.dumps(recul, ensure_ascii=False),
                         str(d.get("curiosite") or ""), json.dumps(d, ensure_ascii=False)[:4000]))
            for raw in (d.get("idees") or [])[:3]:
                if not isinstance(raw, dict) or not str(raw.get("texte") or "").strip():
                    continue
                text = str(raw["texte"]).strip()[:300]
                vec = embed(text)
                if self._is_duplicate(con, vec):
                    continue
                kind = str(raw.get("genre") or "a_essayer")[:24]
                why = str(raw.get("pourquoi") or "")[:300]
                whom = str(raw.get("pour_qui") or "")[:40]
                try:
                    score = max(0.0, min(1.0, float(raw.get("interet") or 0.5)))
                except (TypeError, ValueError):
                    score = 0.5
                cur = con.execute("INSERT INTO ideas(t,kind,text,why,for_whom,score,vec) "
                                  "VALUES(?,?,?,?,?,?,?)",
                                  (now, kind, text, why, whom, score, vec.tobytes()))
                ideas.append(Idea(int(cur.lastrowid), now, kind, text, why, whom, score))
        return Thought(now, "", reflection, recul, ideas, str(d.get("curiosite") or ""))

    def _is_duplicate(self, con: sqlite3.Connection, vec: Any) -> bool:
        rows = con.execute("SELECT vec FROM ideas WHERE status != 'rejetee' ORDER BY id DESC "
                           "LIMIT 200").fetchall()
        if not rows:
            return False
        import numpy as np

        m = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
        return bool(cosine_rows(m, vec).max() >= self.cfg.idea_similarity)

    def _apply_reappraisal(self, th: Thought, now: float) -> None:
        """Le recul agit sur le cœur : une réévaluation apaise ou stimule, un peu. Une même
        pensée négative répétée sans élément nouveau n'agit plus (rumination plafonnée)."""
        effet = str(th.reappraisal.get("effet") or "aucun")
        if effet not in ("apaise", "stimule"):
            return
        vec = embed(th.reflection)
        self.recent_reflections = [(t, v) for t, v in self.recent_reflections
                                   if now - t < self.cfg.rumination_window]
        similar = sum(1 for _, v in self.recent_reflections
                      if float(v @ vec) >= self.cfg.idea_similarity)
        self.recent_reflections.append((now, vec))
        if similar >= self.cfg.rumination_max:
            return
        impulses = self.cfg.calm if effet == "apaise" else self.cfg.stir
        self.heart_apply(impulses, self.cfg.reappraisal_gain)

    # ----------------------------------------------------------------- idées
    def best_idea(self, min_score: float | None = None, kind: str | None = None) -> Idea | None:
        ms = self.cfg.propose_min_score if min_score is None else min_score
        with self._conn() as con:
            q = ("SELECT id,t,kind,text,why,for_whom,score,status FROM ideas WHERE status='nouvelle' "
                 "AND score>=?")
            args: list[Any] = [ms]
            if kind:
                q += " AND kind=?"
                args.append(kind)
            row = con.execute(q + " ORDER BY score DESC, t DESC LIMIT 1", args).fetchone()
        return Idea(*row) if row else None

    def mark(self, idea_id: int, status: str) -> None:
        with self._conn() as con:
            con.execute("UPDATE ideas SET status=?, proposed_at=COALESCE(proposed_at, ?) "
                        "WHERE id=?", (status, self.clock(), idea_id))

    def ideas(self, limit: int = 10, status: str | None = "nouvelle") -> list[Idea]:
        with self._conn() as con:
            if status:
                rows = con.execute("SELECT id,t,kind,text,why,for_whom,score,status FROM ideas "
                                   "WHERE status=? ORDER BY score DESC, t DESC LIMIT ?",
                                   (status, limit)).fetchall()
            else:
                rows = con.execute("SELECT id,t,kind,text,why,for_whom,score,status FROM ideas "
                                   "ORDER BY t DESC LIMIT ?", (limit,)).fetchall()
        return [Idea(*r) for r in rows]

    def block(self) -> str:
        """Ce que la pensée de fond apporte au dialogue : dernière réflexion et idées en stock."""
        lines = []
        if self.last and self.last.reflection:
            lines.append(f"- tu pensais tout à l'heure : {self.last.reflection}")
        for i in self.ideas(limit=self.cfg.block_ideas):
            lines.append(f"- idée que tu gardes en tête : {i.line()}")
        if not lines:
            return ""
        return ("TES PENSÉES DE FOND (à toi, à partager seulement si ça vient naturellement ou "
                "si on te le demande) :\n" + "\n".join(lines))


_JSON = re.compile(r"\{.*\}", re.S)


def _parse(text: str) -> dict[str, Any] | None:
    m = _JSON.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) else None
