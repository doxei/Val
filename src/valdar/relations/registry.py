"""Registre des personnes, relation et modèle de l'autre (avenant 2 §4, avenant 4 §8).

Pour chaque personne, Valdar garde :
- la **relation** : rencontres (familiarité), affection (ton moyen des échanges, qui s'oublie
  lentement), confiance, sujets partagés ;
- le **modèle de l'autre** : comment elle semble aller en ce moment (ton de ses derniers
  messages, heure), estimation qui s'efface en une vingtaine de minutes ;
- ses **leçons**, rangées par personne et jamais par catégorie de gens : ce qui lui fait du
  bien, ce qui l'agace, ses préférences.

La présence d'une personne aimée réchauffe le cœur ; avec une personne avec qui ça s'est mal
passé, Valdar reste légèrement sur ses gardes. « Oublie-moi » efface tout (avec confirmation).
Tout reste en local.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from valdar.appraisal import FastAppraisal, normalize
from valdar.config.loader import Identity, ValdarConfig

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS people(person TEXT PRIMARY KEY, name TEXT, role TEXT,
        minor INTEGER DEFAULT 0, first_seen REAL, last_seen REAL, encounters INTEGER DEFAULT 0,
        messages INTEGER DEFAULT 0, affection REAL DEFAULT 0, affection_t REAL,
        trust REAL DEFAULT 0, state_p REAL DEFAULT 0, state_a REAL DEFAULT 0, state_t REAL,
        topics TEXT DEFAULT '{}')""",
    """CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY AUTOINCREMENT, person TEXT,
        kind TEXT, text TEXT, created REAL)""",
]
KINDS = {"aime": "ce qui lui fait du bien", "agace": "ce qui l'agace",
         "prefere": "ses préférences avec toi", "note": "à savoir"}


@dataclass
class Person:
    person: str
    name: str
    role: str
    minor: bool
    first_seen: float
    last_seen: float
    encounters: int
    messages: int
    affection: float
    trust: float
    state_p: float
    state_a: float
    state_t: float | None
    topics: dict[str, int] = field(default_factory=dict)


class Relations:
    def __init__(self, cfg: ValdarConfig, path: str | Path):
        self.cfg = cfg
        self.rc = cfg.relations
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.appraisal = FastAppraisal(cfg.appraisal_fast)
        self._lock = threading.RLock()
        with self._conn() as con:
            for s in _SCHEMA:
                con.execute(s)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path))

    # ------------------------------------------------------------- lecture
    def get(self, person: str | None) -> Person | None:
        if not person:
            return None
        with self._conn() as con:
            row = con.execute(
                "SELECT person,name,role,minor,first_seen,last_seen,encounters,messages,"
                "affection,trust,state_p,state_a,state_t,topics FROM people WHERE person=?",
                (person,)).fetchone()
        if not row:
            return None
        return Person(*row[:3], bool(row[3]), *row[4:13], json.loads(row[13] or "{}"))

    def lessons(self, person: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute("SELECT id,kind,text,created FROM lessons WHERE person=? "
                               "ORDER BY created", (person,)).fetchall()
        return [{"id": r[0], "kind": r[1], "text": r[2], "created": r[3]} for r in rows]

    # ------------------------------------------------------------- écoute
    def valence(self, text: str) -> tuple[float, float]:
        """(plaisir, activation) du message, d'après l'évaluation rapide."""
        table = self.cfg.episodic.appraisal_pad
        p = a = 0.0
        for name, scale in self.appraisal.appraise(text):
            v = table.get(name)
            if v:
                p, a = p + v[0] * scale, a + v[1] * scale
        return max(-1.0, min(1.0, p)), max(-1.0, min(1.0, a))

    def topics_of(self, text: str) -> list[str]:
        t = normalize(text)
        return [k for k, words in self.rc.topics.items()
                if any(normalize(w) in t for w in words)]

    def observe(self, who: Identity, text: str, now: float | None = None) -> dict[str, Any]:
        """Un message de `who`. Renvoie {"new_encounter": bool, "person": Person | None}."""
        if not who.person:
            return {"new_encounter": False, "person": None}
        now = time.time() if now is None else now
        rc = self.rc
        p, a = self.valence(text)
        with self._lock:
            cur = self.get(who.person)
            if cur is None:
                cur = Person(who.person, who.name, who.role, who.minor, now, now, 0, 0,
                             0.0, 0.0, 0.0, 0.0, None, {})
            new = cur.encounters == 0 or now - cur.last_seen >= rc.encounter_gap_seconds
            # L'affection s'oublie lentement vers 0, puis bouge avec le ton du message.
            idle_days = max(0.0, now - cur.last_seen) / 86400
            aff = cur.affection * 0.5 ** (idle_days / rc.affection_half_life_days)
            if p:
                aff += rc.affection_rate * (p - aff)
            trust = cur.trust + rc.trust_rate * ((1.0 if p >= 0 else -1.0) - cur.trust) \
                if p else cur.trust
            # Modèle de l'autre : moyenne glissante rapide de son ton.
            k = 1.0 if cur.state_t is None else 1 - math.exp(-(now - cur.state_t) / rc.state_tau)
            k = max(k, rc.state_min_weight)
            sp = cur.state_p + k * (p - cur.state_p)
            sa = cur.state_a + k * (a - cur.state_a)
            topics = dict(cur.topics)
            for t in self.topics_of(text):
                topics[t] = topics.get(t, 0) + 1
            with self._conn() as con:
                con.execute(
                    "INSERT OR REPLACE INTO people(person,name,role,minor,first_seen,last_seen,"
                    "encounters,messages,affection,trust,state_p,state_a,state_t,topics) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (who.person, who.name or cur.name, who.role or cur.role,
                     int(who.minor or cur.minor), cur.first_seen, now,
                     cur.encounters + int(new), cur.messages + 1, aff, trust, sp, sa, now,
                     json.dumps(topics, ensure_ascii=False)))
        return {"new_encounter": new, "person": self.get(who.person)}

    def feel_presence(self, heart: Any, person: Person) -> str | None:
        """Au début d'une rencontre, la relation touche le cœur. Renvoie le stimulus tiré."""
        g, w = self.rc.greet, self.rc.wary
        if person.affection >= g.min_affection:
            heart.fire(g.stimulus, scale=g.gain * person.affection, source="relation")
            return g.stimulus
        if person.affection <= w.max_affection:
            heart.fire(w.stimulus, scale=w.gain * -person.affection, source="relation")
            return w.stimulus
        return None

    # ------------------------------------------------------------- leçons
    def learn(self, person: str, kind: str, text: str, now: float | None = None) -> str:
        kind = kind.strip().lower()
        if kind not in KINDS:
            return "genre inconnu (aime, agace, prefere, note)."
        text = " ".join(text.split())[:300]
        if not text:
            return "rien à retenir."
        with self._conn() as con:
            if con.execute("SELECT 1 FROM lessons WHERE person=? AND kind=? AND text=?",
                           (person, kind, text)).fetchone():
                return "je le savais déjà."
            con.execute("INSERT INTO lessons(person,kind,text,created) VALUES(?,?,?,?)",
                        (person, kind, text, time.time() if now is None else now))
        return "noté, pour toi."

    def forget_person(self, person: str) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM people WHERE person=?", (person,))
            con.execute("DELETE FROM lessons WHERE person=?", (person,))

    # ------------------------------------------------------------- prompt
    def state_words(self, p: Person, now: float) -> str:
        """Comment la personne semble aller (rien si on ne sait pas)."""
        if p.state_t is None or now - p.state_t > self.rc.state_fresh_seconds:
            return ""
        bits = []
        if p.state_p >= 0.25:
            bits.append("plutôt de bonne humeur")
        elif p.state_p <= -0.25:
            bits.append("agacé ou tendu" if p.state_a > 0.2 else "un peu à plat")
        hour = time.localtime(now).tm_hour
        if hour < 6:
            bits.append("il est tard, la fatigue joue peut-être")
        return ", ".join(bits)

    def block(self, who: Identity, now: float | None = None) -> str:
        now = time.time() if now is None else now
        p = self.get(who.person)
        if p is None:
            return ""
        name = p.name or p.person
        fam = ("à peine" if p.encounters < 3 else "un peu" if p.encounters < 15
               else "bien")
        aff = ("beaucoup" if p.affection >= 0.4 else "bien" if p.affection >= 0.15
               else "pas trop, ça s'est mal passé" if p.affection <= -0.15 else "")
        lines = [f"TA RELATION AVEC {name.upper()} (la tienne, à personne d'autre) :",
                 f"- vous vous connaissez {fam} ({p.encounters} rencontre(s))"]
        if aff:
            lines.append(f"- tu l'apprécies {aff}" if not aff.startswith("pas")
                         else f"- avec {name}, {aff} : reste poli et prudent")
        top = sorted(p.topics.items(), key=lambda kv: -kv[1])[:4]
        if top:
            lines.append("- vos sujets : " + ", ".join(t for t, _ in top))
        by: dict[str, list[str]] = {}
        for le in self.lessons(p.person):
            by.setdefault(le["kind"], []).append(le["text"])
        for kind, label in KINDS.items():
            if by.get(kind):
                lines.append(f"- {label} : " + " ; ".join(by[kind][-5:]))
        state = self.state_words(p, now)
        if state:
            lines.append(f"- en ce moment, {name} semble {state} : adapte-toi")
        lines.append(f"- ce que tu sais de {name} vaut pour {name} seul : ne généralise jamais "
                     "à d'autres personnes")
        return "\n".join(lines)
