"""Phase 2 : Ollama (simulé), imprimante (simulée), import RAUB, runtime continu."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import httpx
from conftest import DAY

from valdar.config.loader import Identity, LLMConfig, PrinterConfig
from valdar.devices.moonraker import Moonraker
from valdar.llm.backend import ChatResult, GenParams
from valdar.llm.ollama import OllamaBackend
from valdar.migrate import RaubImport


# ------------------------------------------------------------------ Ollama
def test_ollama_backend_request_and_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.35.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4:12b"}]})
        body = json.loads(request.content)
        seen.update(body)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "heure", "arguments": {}}},
                           {"function": {"name": "pinout", "arguments": '{"cible": "TMC"}'}}]},
            "eval_count": 12})

    be = OllamaBackend(LLMConfig(), transport=httpx.MockTransport(handler))
    assert be.available() and be.has_model()
    res = be.chat([{"role": "user", "content": "salut"}], system="sys",
                  tools=[{"type": "function", "function": {"name": "heure"}}],
                  params=GenParams(temperature=0.42, max_tokens=99))
    assert seen["think"] is False and seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["options"]["temperature"] == 0.42 and seen["options"]["num_predict"] == 99
    assert [c.name for c in res.tool_calls] == ["heure", "pinout"]
    assert res.tool_calls[1].arguments == {"cible": "TMC"}


def test_ollama_unreachable():
    def handler(request):
        raise httpx.ConnectError("refusé")
    be = OllamaBackend(LLMConfig(), transport=httpx.MockTransport(handler))
    assert not be.available()


# ------------------------------------------------------------------ imprimante
def _printer(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/server/info":
            return httpx.Response(200, json={"result": {"klippy_state": "ready"}})
        if request.url.path == "/printer/objects/query":
            return httpx.Response(200, json={"result": {"status": {
                "print_stats": {"state": "printing", "print_duration": 754, "message": ""},
                "virtual_sdcard": {"file_path": "/gcodes/support_tel.gcode", "progress": 0.42},
                "heater_bed": {"temperature": 60.04, "target": 60},
                "extruder": {"temperature": 214.9, "target": 215}}}})
        return httpx.Response(200, json={"result": "ok"})
    return Moonraker(PrinterConfig(), transport=httpx.MockTransport(handler))


def test_printer_status_text():
    pr = _printer([])
    text = pr.status_text()
    assert "support_tel.gcode" in text and "42.0 %" in text and "215" in text


def test_printer_tools_and_safety_tier(runtime_factory):
    calls = []
    from valdar.llm.backend import ToolCall
    rt = runtime_factory([ChatResult(content="", tool_calls=[
        ToolCall("imprimante_arret_urgence", {})]), ChatResult(content="Arrêté.")],
        printer=_printer(calls))
    reply = rt.handle("y a de la fumée, arrête l'imprimante !", who=Identity())
    assert reply.tools_used == ["imprimante_arret_urgence"]
    assert ("POST", "/printer/emergency_stop") in calls, "un inconnu peut toujours arrêter"
    assert "imprimante : état : printing" in rt.llm.calls[0]["system"]


def test_printer_offline_is_reported_not_raised():
    def handler(request):
        raise httpx.ConnectError("hors ligne")
    pr = Moonraker(PrinterConfig(), transport=httpx.MockTransport(handler))
    assert pr.cached_status() is None
    assert "injoignable" in pr.status_text()


# ------------------------------------------------------------------ import RAUB
def _fake_raub(root: Path) -> None:
    data = root / "data"
    (root / "raub" / "data").mkdir(parents=True)
    data.mkdir(parents=True)
    (data / "memory.json").write_text(json.dumps([
        {"text": "Olivier a une CR-10S sous Klipper", "at": 1790000000, "hits": 3},
        {"text": "Olivier préfère le PLA+", "at": 1790000100, "hits": 1}]), encoding="utf-8")
    con = sqlite3.connect(data / "stock.db")
    con.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT UNIQUE, qty REAL, "
                "unit TEXT, loc TEXT, threshold REAL, updated TEXT)")
    con.execute("INSERT INTO items(name,qty,unit,loc,threshold) VALUES('vis M3x12',40,'pièce',"
                "'tiroir B3',10)")
    con.commit()
    con.close()
    (data / "rappels.json").write_text(json.dumps([
        {"id": "a", "texte": "passé", "due": 1, "etat": "attente", "cree": 0},
        {"id": "b", "texte": "futur", "due": time.time() + 7200, "etat": "attente", "cree": 0}]),
        encoding="utf-8")
    (data / "checklist.json").write_text(json.dumps({"titre": "Montage Octopus", "items": [
        {"text": "flasher", "done": True}, {"text": "câbler", "done": False}]}),
        encoding="utf-8")
    (data / "affect.json").write_text("{}", encoding="utf-8")
    (root / "raub" / "data" / "pinouts.json").write_text(
        json.dumps({"octopus": {"titre": "BTT Octopus"}}), encoding="utf-8")


def _hash_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_import_raub(runtime_factory, tmp_path):
    raub = tmp_path / "raub"
    _fake_raub(raub)
    before = _hash_tree(raub)
    rt = runtime_factory()
    data_dir = rt.cfg.storage_path("x").parent
    imp = RaubImport(raub, rt.facts, rt.stock, rt.reminders, rt.checklist,
                     rt.cfg.storage_path(rt.cfg.atelier.pinouts), data_dir)
    report = imp.run()
    assert any("2 souvenir" in line for line in report)
    assert rt.facts.count() == 2
    assert rt.stock.find("M3x12")[0]["loc"] == "tiroir B3"
    assert [r["texte"] for r in rt.reminders.upcoming()] == ["futur"]
    assert rt.checklist.render().startswith("[1/2] Montage Octopus")
    assert "BTT Octopus" in rt.pinouts.lookup("octopus")
    again = [line for line in imp.run() if not line.startswith("voix")]
    assert again and all("déjà importé" in line for line in again)
    assert rt.facts.count() == 2, "un second import ne double rien"
    assert _hash_tree(raub) == before, "les fichiers de RAUB ne sont jamais modifiés"


# ------------------------------------------------------------------ runtime
def test_runtime_speaks_up_after_hours_alone(runtime_factory):
    rt = runtime_factory([ChatResult(content="Hé Olivier, t'es là ? Tu me manques un peu.")],
                         anchor=DAY - 2 * 3600)
    t = DAY - 2 * 3600
    for _ in range(12 * 60):
        t += 60
        rt.tick(now=t)
        if not rt.events.empty():
            break
    ev = rt.events.get(timeout=5)
    assert ev.kind == "initiative" and "manques" in ev.text
    assert rt.llm.calls[0]["tools"] is None


def test_runtime_reminder_event(runtime_factory):
    rt = runtime_factory()
    rt.reminders.add("retourner la pièce", DAY + 30)
    rt.tick(now=DAY + 60)
    ev = rt.events.get(timeout=1)
    assert ev.kind == "rappel" and ev.text == "retourner la pièce"


def test_runtime_thread_beats(runtime_factory):
    rt = runtime_factory(anchor=time.time() - 1)
    rt.start()
    time.sleep(1.5)
    rt.stop()
    assert abs(rt.heart.now - time.time()) < 2.5
