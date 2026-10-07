"""Briques de la phase 2 : permissions, évaluation rapide, atelier, mémoire, expression."""
import json
import re
import time
from datetime import datetime

import pytest

from valdar.appraisal import FastAppraisal
from valdar.atelier import Checklist, Pinouts, Reminders, Stock, parse_when
from valdar.config.loader import Identity, PermissionsConfig
from valdar.expression.compose import feeling_block, gen_params, person_block
from valdar.memory import Facts
from valdar.tools.registry import DANGEROUS, ELEVATED, SAFE, SAFETY, Tool, decide, params

PERMS = PermissionsConfig()


def tool(tier):
    return Tool("t", "t", params(), lambda: None, tier)


# ------------------------------------------------------------------ permissions
@pytest.mark.parametrize("who,tier,allowed,confirm", [
    (Identity(), SAFE, True, False),
    (Identity(), SAFETY, True, False),
    (Identity(), ELEVATED, False, False),
    (Identity(), DANGEROUS, False, False),
    (Identity(person="o", role="owner", confidence=0.8), ELEVATED, False, False),
    (Identity(person="o", role="owner", confidence=0.92), ELEVATED, True, False),
    (Identity(person="o", role="owner", confidence=0.92), DANGEROUS, False, False),
    (Identity(person="o", role="owner", confidence=0.97), DANGEROUS, True, True),
    (Identity(person="g", role="guest", confidence=1.0), ELEVATED, False, False),
    (Identity(person="o", role="owner", confidence=1.0, minor=True), ELEVATED, False, False),
])
def test_permission_matrix(who, tier, allowed, confirm):
    d = decide(tool(tier), who, PERMS)
    assert d.allowed is allowed
    assert d.needs_confirmation is confirm


# ----------------------------------------------------------- évaluation rapide
def test_fast_appraisal(cfg):
    fa = FastAppraisal(cfg.appraisal_fast)
    assert ("frustration", 0.5) in fa.appraise("T'ES NUL franchement")
    assert ("praise", 0.6) in fa.appraise("Génial, bien joué !")
    assert ("warmth", 0.6) in fa.appraise("merci mon pote")
    assert ("shame", 0.6) in fa.appraise("non c'est faux, je t'ai déjà dit")
    assert fa.appraise("stop, arrête") == [], "« top » ne doit pas être trouvé dans « stop »"
    assert fa.appraise("le bed leveling est fait") == []


# ------------------------------------------------------------------ atelier
def test_stock(tmp_path):
    st = Stock(tmp_path / "a.db")
    st.set_item("PLA+ noir", 2, "bobine", "étagère", threshold=1)
    assert "1" in st.change("PLA+ noir", -1) and "racheter" in st.change("PLA+ noir", 0)
    assert "inconnu" in st.change("PETG", -1)
    assert st.low()[0]["name"] == "PLA+ noir"
    assert "PLA+ noir" in st.render("pla")


def test_checklist_persists_and_matches(tmp_path):
    p = tmp_path / "c.json"
    c = Checklist(p)
    c.create("Changer la buse", ["chauffer à 240", "dévisser la buse", "serrer à chaud"])
    c.check(["2"])
    c.check(["serrer"])
    again = Checklist(p)
    assert [i.done for i in again.current.items] == [False, True, True]
    assert again.render().startswith("[2/3]")


def test_parse_when():
    now = datetime(2026, 10, 7, 18, 0).timestamp()
    assert parse_when("dans 10 minutes", now) == now + 600
    assert parse_when("dans une heure", now) == now + 3600
    assert parse_when("dans 2h30", now) == now + 9000
    assert parse_when("dans un quart d'heure", now) == now + 900
    assert parse_when("à 18h30", now) == datetime(2026, 10, 7, 18, 30).timestamp()
    assert parse_when("à 9h", now) == datetime(2026, 10, 8, 9, 0).timestamp()
    assert parse_when("demain à 8h15", now) == datetime(2026, 10, 8, 8, 15).timestamp()
    assert parse_when("à midi", now) == datetime(2026, 10, 8, 12, 0).timestamp()
    assert parse_when("quand tu veux", now) is None


def test_reminders_pop_due(tmp_path):
    r = Reminders(tmp_path / "r.json")
    r.add("sortir le chien", time.time() - 5)
    r.add("vérifier l'impression", time.time() + 3600)
    due = r.pop_due()
    assert [d["texte"] for d in due] == ["sortir le chien"]
    assert r.pop_due() == []
    assert len(r.upcoming()) == 1
    assert r.cancel("impression") and r.upcoming() == []


def test_pinouts(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"octopus_v11": {"titre": "BTT Octopus V1.1", "source": "doc",
                                             "confiance": "haute",
                                             "chauffage": {"HE0": "PA2"},
                                             "pieges": ["jumper USB"]}}), encoding="utf-8")
    out = Pinouts(p).lookup("octopus")
    assert "HE0" in out and "PIÈGE" in out
    assert "rien sur" in Pinouts(p).lookup("arduino uno")


# ------------------------------------------------------------------ mémoire
def test_facts(tmp_path):
    f = Facts(tmp_path / "m.db")
    assert f.remember("Olivier a une CR-10S passée sous Klipper") == "noté."
    assert "déjà" in f.remember("olivier a une cr-10s passée sous klipper")
    f.remember("Olivier aime le PLA silk or", person="olivier")
    f.remember("Le voisin s'appelle Manu", person="")
    assert f.recall("klipper")[0]["text"].startswith("Olivier a une CR-10S")
    assert f.recall("klipper")[0]["hits"] == 2
    assert "1 souvenir" in f.forget("silk")
    assert f.count() == 2


# ------------------------------------------------------------------ expression
def test_feeling_block_has_no_numbers(make_heart, cfg):
    h = make_heart()
    h.fire("threat", scale=2.0)
    h.step(5)
    block = feeling_block(h, cfg.expression, {"talk": 0.6, "explore": 0.1})
    assert "peur" in block or "anxiété" in block
    assert "envie de parler" in block
    assert not re.search(r"\d", block), "aucun chiffre sur l'état interne dans le prompt"


def test_state_changes_generation(make_heart, cfg):
    calm, playful, tired = make_heart(), make_heart(), make_heart()
    playful.fire("play", scale=2.0)
    playful.step(5)
    tired.variables["energy"] = 0.15
    tired._refresh()
    p0 = gen_params(cfg.expression, calm.sources())
    assert gen_params(cfg.expression, playful.sources()).temperature > p0.temperature
    assert gen_params(cfg.expression, tired.sources()).max_tokens < p0.max_tokens


def test_unknown_person_block_protects_privacy():
    text = person_block(Identity(), "Olivier")
    assert "ne reconnais pas" in text and "vie privée" in text


def test_value_core_reaches_the_prompt():
    from pathlib import Path

    from valdar.expression.compose import load_self_model, self_block

    sm = load_self_model(Path(__file__).resolve().parents[1] / "config" / "self_model.yaml")
    text = self_block(sm)
    assert "Ton socle" in text and "tu conseilles, tu ne diriges pas" in text
    assert "aucun besoin d'être supérieur" in text
