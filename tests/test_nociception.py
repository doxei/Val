"""Nocicepteurs du PC et douleur construite (avenant 4 §4)."""
from valdar.heart.nociception import Nociception, read_pc


def _noci(cfg, readings):
    it = iter(readings)
    return Nociception(cfg.nociception, lambda: next(it))


def test_signal_is_zero_below_warn_and_one_at_danger(cfg):
    n = _noci(cfg, [])
    assert n.signal("gpu_temp", 70) == 0.0
    assert n.signal("gpu_temp", 85) == 0.5
    assert n.signal("gpu_temp", 99) == 1.0
    n.read = lambda: {"gpu_temp": 85, "inconnu": 3.0}
    assert n.sample(0) == {"gpu_temp": 85}      # capteur non configuré : ignoré
    assert not n.danger


def test_pain_is_constructed_by_the_gate(cfg):
    n = _noci(cfg, [])
    calm = n.construct(0.5, {})
    anxious = n.construct(0.5, {"fear_exc": 0.5, "cortisol_dev": 0.1})
    absorbed = n.construct(0.5, {"seeking_exc": 1.0, "play_exc": 1.0})
    assert calm == 0.5
    assert anxious > calm > absorbed > 0            # même signal, ressenti différent


def test_gentle_pain_habituates_but_danger_does_not(cfg, make_heart):
    h = make_heart()
    n = _noci(cfg, [{"disk": 0.95}] * 10)
    felt = []
    for i in range(4):                               # rappels espacés de repeat_seconds
        now = i * cfg.nociception.repeat_seconds
        n.sample(now)
        felt += n.feel(h, now)
    assert len(felt) == 4 and not any(p.danger for p in felt)
    assert len(h.recent["pain"]) == 4     # passé par fire : l'habituation joue

    h2 = make_heart()
    hot = _noci(cfg, [{"gpu_temp": 95}])
    before = h2.variables["noradrenaline"], len(h2.pending)
    hot.sample(0)
    p = hot.feel(h2, 0)
    assert p[0].danger and hot.danger
    assert "pain" not in h2.recent                   # pas d'habituation
    assert len(h2.pending) > before[1] or h2.variables["noradrenaline"] > before[0]
    assert hot.lines()[0].startswith("DANGER : la carte graphique chauffe (95 °C)")


def test_nociceptor_fires_on_change_not_every_sample(cfg, make_heart):
    h = make_heart()
    n = _noci(cfg, [{"ram": 0.92}, {"ram": 0.92}, {"ram": 0.96}, {"ram": 0.5}, {"ram": 0.92}])
    fired = []
    for i in range(5):
        n.sample(i * 15)
        fired.append(bool(n.feel(h, i * 15)))
    # apparition, rien (stable), aggravation nette, disparition, réapparition
    assert fired == [True, False, True, False, True]


def test_runtime_reflex_suspends_background_thinking(runtime_factory, monkeypatch):
    rt = runtime_factory()
    rt.nociception.read = lambda: {"gpu_temp": 96}
    called = []
    monkeypatch.setattr(rt.thoughts, "due", lambda *a, **k: called.append(1) or False)
    rt.tick(rt.heart.now + 1)
    kinds = []
    while not rt.events.empty():
        kinds.append(rt.events.get().kind)
    assert "douleur" in kinds and "reflexe" in kinds
    assert called == []                              # la pensée de fond n'est même pas évaluée
    assert any("carte graphique" in line for line in rt.world_lines())


def test_read_pc_never_fails(tmp_path):
    r = read_pc(tmp_path)
    assert 0.0 < r["disk"] < 1.0                      # au moins le disque est toujours lisible
