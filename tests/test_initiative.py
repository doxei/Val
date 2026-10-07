"""Initiative et garde-fous anti-insistance (cahier v2, §8)."""
from conftest import DAY

from valdar.workspace import Initiative


def run(h, init, hours, dt=60.0):
    events = []
    t = 0.0
    while t < hours * 3600:
        h.step(dt)
        t += dt
        ev = init.check(h)
        if ev:
            events.append(ev)
    return events


def test_valdar_speaks_up_after_hours_alone(make_heart, quiet_cfg):
    h = make_heart(anchor=DAY - 2 * 3600)  # 09:00
    init = Initiative(quiet_cfg.initiative)
    events = run(h, init, 10)
    assert events, "après des heures seul, Valdar doit finir par vouloir parler"
    assert events[0]["t"] - (DAY - 2 * 3600) > 3 * 3600, "pas d'initiative trop tôt"


def test_never_more_than_max_unanswered(make_heart, quiet_cfg):
    h = make_heart(anchor=DAY - 2 * 3600)
    init = Initiative(quiet_cfg.initiative)
    events = run(h, init, 13)
    assert len(events) <= quiet_cfg.initiative.max_unanswered


def test_user_message_rearms(make_heart, quiet_cfg):
    h = make_heart(anchor=DAY - 2 * 3600)
    init = Initiative(quiet_cfg.initiative)
    run(h, init, 13)
    assert init.unanswered == quiet_cfg.initiative.max_unanswered
    init.on_user_message()
    assert init.unanswered == 0


def test_silence_blocks_everything(make_heart, quiet_cfg):
    h = make_heart(anchor=DAY - 2 * 3600)
    init = Initiative(quiet_cfg.initiative)
    init.silence(True)
    assert run(h, init, 12) == []


def test_no_initiative_in_quiet_hours(make_heart, quiet_cfg):
    h = make_heart(anchor=DAY - 2 * 3600)
    h.needs = {k: 1.0 for k in h.needs}
    h.now = DAY + 11.75 * 3600  # 22:45, heures calmes
    init = Initiative(quiet_cfg.initiative)
    assert run(h, init, 0.5) == []


def test_ignored_initiative_leaves_a_trace(make_heart, quiet_cfg):
    h = make_heart(anchor=DAY - 2 * 3600)
    init = Initiative(quiet_cfg.initiative)
    events = run(h, init, 10)
    assert events
    assert "ignored" in h.recent, "un message sans réponse doit laisser une trace dans le cœur"


def test_state_roundtrip(quiet_cfg):
    a = Initiative(quiet_cfg.initiative)
    a.unanswered, a.last_at, a.silenced = 1, 123.0, True
    b = Initiative(quiet_cfg.initiative)
    b.load_dict(a.to_dict())
    assert b.to_dict() == a.to_dict()
