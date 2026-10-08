"""Avenant 5 : le Surmoi critique (croyances, quarantaine) et la nuit."""
from __future__ import annotations

import json

from valdar.config.loader import NightConfig
from valdar.expression.compose import facts_block
from valdar.llm.backend import ChatResult
from valdar.memory import Episodic, Facts
from valdar.workspace.night import Night

NOW = 1_790_000_000.0


def test_deduction_is_a_hypothesis_until_confirmed(tmp_path):
    f = Facts(tmp_path / "m.db")
    assert "hypothèse" in f.remember("Olivier préfère le PLA", origine="deduit",
                                     refutation="il imprime surtout en PETG")
    b = f.find("PLA")
    assert b["statut"] == "hypothese" and b["confidence"] <= 0.5
    f.evidence(b["id"], True, "il l'a dit")
    assert f.find("PLA")["statut"] == "hypothese"     # une seule preuve ne suffit pas
    f.evidence(b["id"], True, "il en a racheté")
    assert f.find("PLA")["statut"] == "valide"


def test_contradiction_sends_to_quarantine_and_lift(tmp_path):
    f = Facts(tmp_path / "m.db")
    f.remember("la buse est en 0.4")
    b = f.find("buse")
    g = f.evidence(b["id"], False, "il a monté une 0.6 hier")
    assert g["statut"] == "quarantaine" and "0.6" in g["motif"]
    assert b["id"] not in [x["id"] for x in f.consolidable()]
    assert "douteux" in facts_block(f.context("buse"))   # consultable, mais cité douteux
    assert f.lift(b["id"])
    assert f.find("buse")["statut"] == "valide"


def test_old_database_gets_belief_columns(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE facts(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,"
                " norm TEXT NOT NULL, person TEXT DEFAULT '', source TEXT DEFAULT 'valdar',"
                " confidence REAL DEFAULT 0.8, created REAL NOT NULL, last_used REAL NOT NULL,"
                " hits INTEGER DEFAULT 1, archived INTEGER DEFAULT 0)")
    con.execute("INSERT INTO facts(text,norm,created,last_used) VALUES('vieux fait','vieux "
                "fait',1,1)")
    con.commit()
    con.close()
    f = Facts(db)
    assert f.find("vieux")["statut"] == "valide"


def test_audit_prefers_used_unchecked_beliefs(tmp_path):
    f = Facts(tmp_path / "m.db")
    f.remember("fait peu utilisé", when=NOW - 86400)
    for _ in range(5):
        f.remember("fait très utilisé", when=NOW - 86400)
    assert f.audit_sample(1, NOW)[0]["text"] == "fait très utilisé"


def test_quarantine_window_excludes_turns_from_the_day(tmp_path):
    m = Episodic(tmp_path / "e.db")
    m.log_turn("Olivier", "bonjour", now=NOW, person="olivier")
    m.log_turn("Valdar", "réponse sous douleur", now=NOW + 10, person="olivier")
    assert m.quarantine_window(NOW + 5, NOW + 15, "douleur") == 1
    texts = [t["text"] for t in m.day_turns(NOW - 1, NOW + 100)]
    assert texts == ["bonjour"]
    assert len(m.day_turns(NOW - 1, NOW + 100, include_quarantined=True)) == 2
    m.forget_person("olivier")
    assert m.quarantined_turns() == {}


class Scripted:
    """Faux Gemma : répond selon le prompt système."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, system="", tools=None, params=None):
        self.calls.append(system[:40])
        if "doute méthodique" in system:
            return ChatResult(json.dumps({"verdict": "contredite",
                                          "element": "j'ai changé pour une buse 0.6"}))
        if "socle" in system:
            return ChatResult(json.dumps({"manquements": [{
                "point": "conseiller, pas diriger", "citation": "fais comme ça",
                "lecon": "proposer au lieu d'ordonner"}]}))
        return ChatResult("J'ai aidé Olivier avec sa buse.")

    def available(self):
        return True


def test_night_runs_in_order_and_consolidates_only_validated(tmp_path):
    facts = Facts(tmp_path / "m.db")
    facts.remember("la buse est en 0.4", when=NOW - 3600)
    facts.remember("Olivier aime le café", when=NOW - 3600)
    mem = Episodic(tmp_path / "e.db")
    mem.log_turn("Olivier", "j'ai changé pour une buse 0.6", now=NOW - 600, person="olivier")
    mem.log_turn("Valdar", "fais comme ça", now=NOW - 590, person="olivier")
    mem.log_turn("Valdar", "phrase dite en pleine panique", now=NOW - 500, person="olivier")
    mem.quarantine_window(NOW - 501, NOW - 499, "émotion extrême")
    llm = Scripted()
    night = Night(NightConfig(audit_size=1), llm, facts, mem, tmp_path / "conso",
                  {"nom": "Valdar", "socle": ["tu conseilles, tu ne diriges pas"]}, "olivier")
    assert not night.due(NOW, awake=True, idle=99999)
    assert night.due(NOW, awake=False, idle=99999)
    rep = night.run(NOW)
    assert rep.turns == 3 and rep.quarantined == 1
    assert rep.audited and rep.audited[0]["verdict"] == "contredite"
    assert rep.lapses[0]["lecon"] == "proposer au lieu d'ordonner"
    assert "buse" in rep.summary
    with open(rep.path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh]
    kinds = [r["genre"] for r in rows]
    assert kinds[0] == "vecu" and "lecon" in kinds
    texts = " ".join(r["texte"] for r in rows)
    assert "panique" not in texts                  # la quarantaine n'est jamais apprise
    contradicted = rep.audited[0]["text"]
    assert contradicted not in [r["texte"] for r in rows if r["genre"] == "croyance"]
    assert not night.due(NOW + 60, awake=False, idle=99999)    # une fois par nuit
    assert any(e["source"] == "nuit" for e in [
        {"source": t["source"]} for t in mem.day_turns(NOW - 1, NOW + 1)])


def test_runtime_quarantines_extreme_exchange(runtime_factory):
    rt = runtime_factory()
    rt.nociception.signals = {"gpu_temp": 1.0}     # douleur au seuil de danger
    rt.handle("salut")
    assert rt.memory.quarantined_turns()
    rt.nociception.signals = {}
    assert rt._extreme() == "" or "émotion" in rt._extreme()
