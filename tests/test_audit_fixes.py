"""Non-régression de l'audit du 8 octobre (voir DECISIONS.md)."""
import threading
import time

import pytest

from valdar.config.loader import EpisodicConfig, Identity, PrintWatchConfig
from valdar.heart.store import Store
from valdar.llm.backend import ChatResult, ToolCall
from valdar.memory import Episodic


def test_heart_store_saves_from_another_thread(tmp_path):
    st = Store(tmp_path / "v.db")
    errors = []

    def worker():
        try:
            st.save("k", {"a": 1})
        except Exception as exc:   # avant : ProgrammingError à chaque battement
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not errors and st.load("k") == {"a": 1}


def test_heart_save_failure_never_kills_the_beat(make_heart):
    h = make_heart()

    class Broken:
        def save(self, *a):
            raise OSError("disque plein")

    h.store = Broken()
    h.persist = True
    h._next_save = 0
    h.step(1.0)                       # ne lève pas
    assert h._next_save > h.now - 1


def _forget_rt(runtime_factory, script):
    rt = runtime_factory(script)
    rt.relations.observe(Identity(person="zoe", name="Zoé", role="guest", confidence=1.0),
                         "salut")
    return rt


def _ask_forget():
    return ChatResult(content="", tool_calls=[ToolCall("oublie_moi", {})])


def test_forget_needs_a_clear_yes(runtime_factory):
    zoe = Identity(person="olivier", name="Olivier", role="owner", confidence=0.8)
    rt = runtime_factory([_ask_forget(), ChatResult(content="ok")] * 3)
    rt.handle("oublie-moi", who=zoe)
    assert rt.agent.pending is not None and rt.agent.pending["strict"]
    r = rt.handle("ok non attends", who=zoe)
    assert "oublie_moi" not in r.tools_used, "un « ok non attends » n'efface rien"


def test_forget_confirmation_expires_and_initiative_clears_it(runtime_factory):
    me = Identity(person="olivier", name="Olivier", role="owner", confidence=0.8)
    rt = runtime_factory([_ask_forget(), ChatResult(content="tu veux une idée ?"),
                          ChatResult(content="cool")])
    rt.handle("oublie-moi", who=me)
    rt.agent.spontaneous("propose une idée")
    assert rt.agent.pending is None
    r = rt.handle("oui", who=me)
    assert "oublie_moi" not in r.tools_used

    rt2 = runtime_factory([_ask_forget(), ChatResult(content="d'accord")])
    rt2.handle("oublie-moi", who=me)
    rt2.agent.pending["asked_at"] -= 600
    assert "oublie_moi" not in rt2.handle("oui", who=me).tools_used


def test_one_episode_per_person_so_forget_is_exact(tmp_path):
    m = Episodic(tmp_path / "e.db", EpisodicConfig(exclude_recent_seconds=0))
    t = 1_790_000_000.0
    m.log_turn("Zoé", "j'habite rue des Lilas", t, person="zoe")
    m.log_turn("Olivier", "rdv dentiste mardi", t + 30, person="olivier")
    m.log_turn("Zoé", "et j'aime le chocolat", t + 60, person="zoe")
    m.forget_person("zoe")
    with m._conn() as con:
        texts = [r[0] for r in con.execute("SELECT text FROM turns")]
    assert texts == ["rdv dentiste mardi"]


def test_low_mood_week_accounting_is_split(make_heart):
    h = make_heart()
    h.mood["P"] = -0.8
    t0 = h.now
    h._watch_mood(t0, 20 * 86400)  # appel direct : 20 jours en un bloc
    assert all(v <= 7 * 86400 + 1 for v in h.low_seconds.values())


def test_watch_keeps_baseline_of_absent_camera_and_survives_purge_errors(tmp_path, monkeypatch):
    from valdar.printwatch import PrintWatch

    class P:
        class cfg:
            moonraker_url = "http://x"
        state = "printing"

        def status(self):
            return {"state": self.state, "file": "a", "duration_s": 10, "progress": 1,
                    "nozzle": (0, 0), "bed": (0, 0)}

        def webcams(self):
            return []

    pr, events = P(), []
    w = PrintWatch(PrintWatchConfig(), pr, tmp_path / "v.db", tmp_path / "f", events.append,
                   cameras=[])
    w._baseline = {"plateau": {"window": 7200, "values": [0.1, 0.2]}}
    w.step(now=100.0)

    def boom(*a, **k):
        raise PermissionError("verrouillé")

    monkeypatch.setattr(w, "purge", boom)
    pr.state = "complete"
    w.step(now=200.0)
    assert w.job is None and events[-1].kind == "fin"
    assert "plateau" in w._get_state("baseline")


@pytest.mark.parametrize("line,expect", [("45, [N/A], [N/A]", {"gpu_temp": 45.0}),
                                         ("50, 600, 12000", {"gpu_temp": 50.0, "gpu_mem": 0.05})])
def test_nvidia_smi_partial_fields(monkeypatch, tmp_path, line, expect):
    import subprocess

    from valdar.heart import nociception

    class R:
        returncode = 0
        stdout = line + "\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    out = nociception.read_pc(tmp_path)
    for k, v in expect.items():
        assert out[k] == pytest.approx(v)


def test_gpu_hang_keeps_last_reading(tmp_path):
    from valdar.config import load
    from valdar.heart.nociception import Nociception

    readings = [{"gpu_temp": 95.0}, {"gpu_hang": 1.0}]
    n = Nociception(load().nociception, lambda: readings.pop(0))
    n.sample(0.0)
    assert n.danger
    n.sample(20.0)
    assert n.danger, "une carte qui ne répond plus ne relâche pas le réflexe"
    assert time.time() > 0
