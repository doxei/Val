"""Tri visuel par Gemma : sobre, consultatif, jamais bloquant."""
import contextlib
import json
import time

from valdar.llm.backend import ChatResult
from valdar.llm.fake import FakeBackend
from valdar.printwatch import WatchEvent
from valdar.printwatch.triage import Triage, parse

ANSWER = {"lisible": True, "visible": "des fils emmêlés à côté de la pièce",
          "defaut": "decollement_spaghetti", "certitude": "forte", "gravite": 5,
          "conseil": "mettre en pause"}


def test_triage_sends_the_image_and_parses_json():
    llm = FakeBackend([ChatResult(content="Voilà : " + json.dumps(ANSWER))])
    res = Triage(llm).assess(b"\xff\xd8jpeg", "la vigie s'inquiète")
    call = llm.calls[0]
    assert call["messages"][0]["images"], "l'image part bien au modèle"
    assert call["tools"] is None and "UNIQUEMENT en JSON" in call["system"]
    assert res.severity == 5 and "fils emmêlés" in res.text()


def test_unreadable_or_garbage_is_honest():
    assert parse("je sais pas") is None
    r = parse(json.dumps({"lisible": False, "visible": "image noire", "certitude": "énorme"}))
    assert r.certainty == "faible" and "n'arrive pas à lire" in r.text()


def test_runtime_triage_is_async_logged_and_spoken(runtime_factory, tmp_path):
    from valdar.config.loader import PrintWatchConfig
    from valdar.printwatch import PrintWatch

    rt = runtime_factory([ChatResult(content=json.dumps(ANSWER)),
                          ChatResult(content="Olivier, ta pièce part en spaghetti !")])

    class P:
        class cfg:
            moonraker_url = "http://x"

    rt.printwatch = PrintWatch(PrintWatchConfig(), P(), tmp_path / "v.db", tmp_path / "f",
                               rt._on_watch, cameras=[])
    rt._on_watch(WatchEvent("pause", "ça part en vrille", "job1", 0.8, b"\xff\xd8"))
    kinds, deadline = [], time.time() + 5
    while time.time() < deadline and "initiative" not in kinds:
        with contextlib.suppress(Exception):
            kinds.append(rt.events.get(timeout=0.2).kind)
    assert "vigie_triage" in kinds and "initiative" in kinds
    with rt.printwatch._conn() as con:
        n = con.execute("SELECT COUNT(*) FROM alerts WHERE kind='triage'").fetchone()[0]
    assert n == 1
    assert rt.printwatch.records() == [], "l'avis de Gemma ne compte pas comme une alerte"
