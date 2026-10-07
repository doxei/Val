"""Connaissances (RAG local) et pensée de fond (avenant 4 §6)."""
import json

import pytest

from valdar.knowledge import Knowledge
from valdar.llm.backend import ChatResult, LLMError
from valdar.llm.fake import FakeBackend
from valdar.workspace.thoughts import Thoughts

REPO_DOCS = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "connaissances"


# ------------------------------------------------------------------ connaissances
def test_knowledge_ingest_and_search(tmp_path):
    k = Knowledge(tmp_path / "k.db")
    md = tmp_path / "notes.md"
    md.write_text("# Warping\nLes coins se soulèvent du plateau quand la pièce refroidit trop "
                  "vite ; un plateau à 60 °C et une jupe aident.\n\n# Stringing\nDes fils fins "
                  "entre deux îlots : rétraction trop courte ou buse trop chaude.\n",
                  encoding="utf-8")
    assert k.ingest_file(md) == 2
    assert k.ingest_file(md) == 0                       # inchangé : rien à refaire
    hits = k.search("les coins se soulevent")           # sans accents, comme la dictée
    assert hits and hits[0].title == "Warping"
    assert k.search("retraction des fils")[0].title == "Stringing"
    assert k.search("zzz qqq") == []
    md.write_text("# Warping\nTexte complètement réécrit sur le décollement des coins.\n",
                  encoding="utf-8")
    assert k.ingest_file(md) == 1 and k.count() == 1    # un document modifié remplace l'ancien


def test_knowledge_reads_the_shipped_bases(tmp_path):
    k = Knowledge(tmp_path / "k.db")
    counts = k.ingest_dir(REPO_DOCS)
    assert counts["defauts_impression.yaml"] > 20 and counts["notes_printos.yaml"] > 0
    hits = k.search("pièce décollée du plateau spaghetti")
    assert hits and "spaghetti" in hits[0].line().lower()
    assert len(hits[0].line(100)) <= 100 + len(hits[0].title) + len(hits[0].doc) + 10


# ------------------------------------------------------------------ pensée de fond
def _answer(reflexion="je me sens un peu seul, c'est disproportionné", effet="apaise",
            idees=None, curiosite="adhérence du PETG"):
    return ChatResult(content="voilà : " + json.dumps({
        "reflexion": reflexion,
        "recul": {"proportionne": False, "regard": "Olivier dort, c'est tout.", "effet": effet},
        "idees": idees if idees is not None else [
            {"genre": "pour_olivier", "texte": "et si on nettoyait le plateau à l'IPA",
             "pourquoi": "les coins se décollaient", "pour_qui": "olivier", "interet": 0.8},
            {"genre": "pour_moi", "texte": "et si j'avais un capteur d'humidité",
             "pourquoi": "le filament prend l'eau", "interet": 0.4},
            {"genre": "a_essayer", "texte": ""}],
        "curiosite": curiosite}, ensure_ascii=False))


@pytest.fixture()
def thoughts_factory(cfg, tmp_path):
    def _make(script):
        applied, spent = [], []
        th = Thoughts(cfg.thoughts, FakeBackend(script), tmp_path / "pensees.db",
                      context=lambda: "ÉTAT : calme", spend=spent.append,
                      heart_apply=lambda imp, amp: applied.append((imp, amp)),
                      clock=lambda: 1000.0)
        return th, applied, spent
    return _make


def test_think_stores_ideas_and_reappraises(cfg, thoughts_factory):
    th, applied, spent = thoughts_factory([_answer()])
    t = th.think(1000.0)
    assert t is not None and t.curiosity == "adhérence du PETG"
    assert [i.kind for i in t.ideas] == ["pour_olivier", "pour_moi"]   # l'idée vide est jetée
    assert applied == [(cfg.thoughts.calm, cfg.thoughts.reappraisal_gain)]
    assert spent == [cfg.thoughts.energy_cost]
    best = th.best_idea()
    assert best is not None and best.text.startswith("et si on nettoyait")
    assert th.best_idea(kind="pour_moi") is None        # sous le seuil de proposition
    th.mark(best.id, "proposee")
    assert th.best_idea() is None
    assert "tu pensais tout à l'heure" in th.block()
    system = th.llm.calls[0]["system"]
    assert "ÉTAT : calme" in system and "pour_Olivier" in system
    assert th.llm.calls[0]["tools"] is None             # la pensée de fond n'agit pas


def test_duplicate_ideas_and_rumination_are_capped(cfg, thoughts_factory):
    th, applied, _ = thoughts_factory([_answer() for _ in range(5)])
    for i in range(5):
        th.think(1000.0 + i)
    assert len(th.ideas(limit=50, status=None)) == 2    # mêmes idées : gardées une fois
    assert len(applied) == cfg.thoughts.rumination_max  # même pensée : n'agit plus


def test_bad_answers_change_nothing(thoughts_factory):
    def boom(**_):
        raise LLMError("injoignable")

    th, applied, spent = thoughts_factory(boom)
    assert th.think() is None and spent == []
    th2, applied2, spent2 = thoughts_factory([ChatResult(content="je ne sais pas")])
    assert th2.think() is None and applied2 == [] and len(spent2) == 1
    th3, applied3, _ = thoughts_factory([_answer(effet="rien", idees=[])])
    assert th3.think() is not None and applied3 == [] and th3.ideas() == []


def test_due_respects_budget_and_priority(cfg, thoughts_factory):
    th, _, _ = thoughts_factory([])
    c = cfg.thoughts
    idle = c.idle_seconds + 1
    assert th.due(10_000, idle, busy=False, awake=True)
    assert not th.due(10_000, idle, busy=True, awake=True)       # le dialogue passe d'abord
    assert not th.due(10_000, idle, busy=False, awake=False)     # il dort
    assert not th.due(10_000, c.idle_seconds - 1, busy=False, awake=True)
    th.history = [10_000 - i for i in range(c.per_hour)]
    assert not th.due(10_000 + 1, idle, busy=False, awake=True)  # budget de l'heure épuisé
    th.history, th.last_at = [], 10_000
    assert not th.due(10_000 + c.min_interval - 1, idle, busy=False, awake=True)
