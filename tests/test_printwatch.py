"""Vigie d'impression : porte des 98 %, prédicteur d'Obico, boucle de surveillance."""
from types import SimpleNamespace

import numpy as np
import pytest

from valdar.devices.moonraker import PrinterError
from valdar.printwatch import (
    FailurePredictor,
    PrintRecord,
    PrintWatch,
    clopper_pearson_lower,
    events_needed,
    gate,
)
from valdar.printwatch.vision import QualityMeter, ScriptedDetector, nms, postprocess

T0 = 1_800_000_000.0


# ------------------------------------------------------------------ porte des 98 %
def test_clopper_pearson_and_events_needed():
    assert clopper_pearson_lower(149, 149) >= 0.98
    assert clopper_pearson_lower(148, 148) < 0.98
    assert events_needed() == 149
    assert events_needed(misses=1) > 149          # un raté coûte cher
    assert clopper_pearson_lower(0, 10) == 0.0
    lo = clopper_pearson_lower(9, 10)
    assert 0.55 < lo < 0.9                         # 90 % mesurés sur 10, presque rien de prouvé


def test_gate_counts_lead_time_and_false_pauses():
    # Raté attrapé 5 min avant le point de non-retour ; raté vu trop tard ; réussite avec
    # une fausse pause.
    caught = PrintRecord("a", T0, T0 + 3600, True, no_return=T0 + 1800, alerts=[T0 + 1500])
    late = PrintRecord("b", T0, T0 + 3600, True, no_return=T0 + 1800, alerts=[T0 + 1790])
    ok = PrintRecord("c", T0, T0 + 10 * 3600, False, pauses=[T0 + 100])
    rep = gate([caught, late, ok], min_ok_hours=5)
    assert (rep.failures, rep.caught) == (2, 1)
    assert rep.lead_median_s == 300
    assert rep.false_pauses == 1 and rep.false_pause_rate_per_100h == pytest.approx(10.0)
    assert not rep.open
    assert any("fausses pauses" in r for r in rep.reasons)
    assert "verrouillée" in rep.text()


def test_gate_opens_only_with_proof():
    fails = [PrintRecord(f"f{i}", T0, T0 + 3600, True, alerts=[T0 + 60]) for i in range(149)]
    oks = [PrintRecord(f"o{i}", T0, T0 + 3600 * 10, False) for i in range(30)]
    assert gate(fails + oks).open
    assert not gate(fails[:-1] + oks).open
    assert not gate(fails + oks[:10]).open          # pas assez d'heures réussies


# ------------------------------------------------------------------ prédicteur
@pytest.fixture()
def pcfg(cfg):
    return cfg.printwatch.predictor.model_copy(update={"init_safe_frames": 3})


def test_predictor_warms_up_then_alerts(pcfg):
    p = FailurePredictor(pcfg)
    verdicts = [p.push(0.0) for _ in range(5)] + [p.push(3.0) for _ in range(15)]
    assert all(not v.alert for v in verdicts[:5])
    assert any(v.alert for v in verdicts) and any(v.pause for v in verdicts)
    first_alert = next(i for i, v in enumerate(verdicts) if v.alert)
    first_pause = next(i for i, v in enumerate(verdicts) if v.pause)
    assert first_alert <= first_pause           # la pause exige un signal plus fort

    p.reset_print()
    assert not p.push(3.0).alert                # nouvelle impression : de nouveau en chauffe


def test_predictor_baseline_absorbs_permanent_false_positive(pcfg):
    # Un câble vu comme « spaghetti » sur toutes les images de la vie de l'imprimante.
    life = {"window": pcfg.rolling_win_long, "values": [2.0] * 500}
    p = FailurePredictor(pcfg, life)
    assert not any(p.push(2.0).alert for _ in range(40))
    assert FailurePredictor.frame_score([0.05, 0.1, 0.5], 0.08) == pytest.approx(0.6)


# ------------------------------------------------------------------ vision (sans opencv)
def test_quality_meter_flags_dark_blurry_frozen(cfg):
    q = QualityMeter(cfg.printwatch.quality)
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert q.assess(sharp).usable
    assert q.assess(np.full((120, 160, 3), 5, np.uint8)).dark
    assert q.assess(np.full((120, 160, 3), 128, np.uint8)).blurry
    for _ in range(cfg.printwatch.quality.frozen_frames):
        last = q.assess(sharp)
    assert last.frozen and "figée" in last.describe()


def test_postprocess_threshold_and_nms():
    boxes = np.array([[[[0.1, 0.1, 0.3, 0.3]], [[0.11, 0.1, 0.31, 0.3]],
                       [[0.6, 0.6, 0.8, 0.8]], [[0.5, 0.5, 0.6, 0.6]]]], dtype=np.float32)
    confs = np.array([[[0.9], [0.5], [0.1], [0.05]]], dtype=np.float32)
    dets = postprocess([boxes, confs], 0.08, 0.45)
    assert sorted(round(d.confidence, 2) for d in dets) == [0.1, 0.9]   # doublon fusionné
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.45) == []


# ------------------------------------------------------------------ boucle de vigie
class FakePrinter:
    def __init__(self):
        self.cfg = SimpleNamespace(moonraker_url="http://imprimante")
        self.st = {"state": "standby"}
        self.paused = 0
        self.down = False

    def status(self):
        if self.down:
            raise PrinterError("injoignable")
        return dict(self.st)

    def pause(self):
        self.paused += 1

    def webcams(self):
        return []


class FakeCamera:
    def __init__(self, name="plateau"):
        self.spec = SimpleNamespace(name=name)
        self.rng = np.random.default_rng(1)

    def grab(self):
        return self.rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def _printing(t: float, progress: float = 10.0, nozzle=(210, 210)):
    return {"state": "printing", "file": "cube.gcode", "progress": progress,
            "duration_s": 600 + t, "nozzle": nozzle, "bed": (60, 60)}


@pytest.fixture()
def watch_factory(cfg, tmp_path):
    def _make(script=None, cameras=None):
        pw = cfg.printwatch.model_copy(deep=True)
        pw.predictor = pw.predictor.model_copy(update={"init_safe_frames": 3})
        events = []
        printer = FakePrinter()
        det = ScriptedDetector(script or [])
        w = PrintWatch(pw, printer, tmp_path / "vigie.db", tmp_path / "vigie", events.append,
                       detector_factory=lambda: det,
                       cameras=[FakeCamera()] if cameras is None else cameras,
                       clock=lambda: T0)
        return w, printer, events
    return _make


def _run(w, printer, n, start=0, **kw):
    for i in range(start, start + n):
        printer.st = _printing(i * 10, progress=float(i), **kw)
        w.step(T0 + i * 10)


def test_watch_alerts_but_does_not_pause_at_level_0(watch_factory):
    script = [[]] * 5 + [[0.9, 0.9, 0.9]] * 20
    w, printer, events = watch_factory(script)
    _run(w, printer, 25)
    kinds = [e.kind for e in events]
    assert "alerte" in kinds and "pause" in kinds
    assert kinds.count("pause") == 1            # on ne répète pas la même alarme
    assert printer.paused == 0                  # niveau 0 : il prévient, il ne touche à rien
    assert w.level() == 0

    printer.st = {"state": "cancelled"}
    w.step(T0 + 300)
    assert events[-1].kind == "question" and w.job is None
    assert "raté" in w.label(True, minutes_before_end=1)
    rec = w.records()
    assert len(rec) == 1 and rec[0].failed and rec[0].alerts and rec[0].pauses
    assert w.gate_report().caught == 1


def test_watch_pauses_alone_only_when_unlocked_and_gate_open(watch_factory, monkeypatch):
    script = [[]] * 5 + [[0.9, 0.9, 0.9]] * 20
    w, printer, events = watch_factory(script)
    assert w.unlock().startswith("Pas encore")
    w._set_state("unlocked", "1")
    monkeypatch.setattr(w, "gate_report", lambda: SimpleNamespace(open=True))
    assert w.level() == 1
    _run(w, printer, 25)
    assert printer.paused == 1
    assert [e.kind for e in events].count("pause_faite") == 1
    assert "reverrouillée" in w.lock() and w.level() == 0


def test_watch_without_camera_and_telemetry(watch_factory):
    w, printer, events = watch_factory(cameras=[])
    _run(w, printer, 6, nozzle=(180, 210))
    texts = [e.text for e in events]
    assert sum("aucune caméra" in t for t in texts) == 1
    assert any("buse" in t and e.kind == "telemetrie" for t, e in zip(texts, events,
                                                                     strict=True))
    # Progression figée plus longtemps que `stall_seconds`.
    w2, p2, ev2 = watch_factory(cameras=[])
    for i in range(0, 2000, 100):
        p2.st = _printing(i, progress=42.0)
        w2.step(T0 + i)
    assert any("progression ne bouge plus" in e.text for e in ev2)


def test_watch_survives_unreachable_printer_and_success_end(watch_factory):
    w, printer, events = watch_factory()
    _run(w, printer, 3)
    printer.down = True
    w.step(T0 + 100)
    assert w.last_status is None and w.job is not None   # une coupure réseau ne clôt rien
    printer.down = False
    printer.st = {"state": "complete"}
    w.step(T0 + 200)
    assert events[-1].kind == "fin"
    assert "réussi" in w.label(False)
    assert w.records()[0].failed is False


def test_purge_removes_old_frames_only(watch_factory, tmp_path):
    w, _, _ = watch_factory()
    d = tmp_path / "vigie" / "vieux"
    d.mkdir(parents=True)
    old, new = d / "old.jpg", tmp_path / "vigie" / "new.jpg"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    with w._conn() as con:
        con.execute("INSERT INTO frames(job,t,camera,path) VALUES('j',?,'c',?)",
                    (T0 - 40 * 86400, str(old)))
        con.execute("INSERT INTO frames(job,t,camera,path) VALUES('j',?,'c',?)",
                    (T0 - 86400, str(new)))
    assert w.purge(T0) == 1
    assert not old.exists() and not d.exists() and new.exists()
    with w._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 2   # scores gardés
