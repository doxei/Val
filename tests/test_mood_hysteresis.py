"""Hystérésis de l'humeur et ses garde-fous (avenant 3 §3)."""
import numpy as np

from valdar.heart import mood as M

REST = {"P": 0.0, "A": 0.0, "D": 0.0}


def _emo(p):
    return {"P": p, "A": 0.0, "D": 0.0}


def _default(make_heart):
    return make_heart().default_mood


def test_hysteresis_loop_switch_points_differ(quiet_cfg, make_heart):
    spec, d = quiet_cfg.heart.mood, _default(make_heart)
    m, down, up = dict(d), None, None
    for e in np.linspace(0.3, -0.6, 46):
        m = M.update(m, _emo(e), d, spec, 4 * 3600)
        if down is None and M.is_low(m, spec):
            down = e
    for e in np.linspace(-0.6, 0.3, 46):
        m = M.update(m, _emo(e), d, spec, 4 * 3600)
        if up is None and not M.is_low(m, spec):
            up = e
    assert down is not None and up is not None
    assert up - down > 0.15                 # remonter demande plus que tomber


def test_same_situation_two_states_history_decides(quiet_cfg, make_heart):
    spec, d = quiet_cfg.heart.mood, _default(make_heart)
    high = M.update(_emo(0.1), _emo(-0.22), d, spec, 48 * 3600)
    low = M.update(_emo(-0.4), _emo(-0.22), d, spec, 48 * 3600)
    assert not M.is_low(high, spec) and M.is_low(low, spec)


def test_guardrail_1_single_state_at_rest(quiet_cfg, make_heart):
    spec, d = quiet_cfg.heart.mood, _default(make_heart)
    ends = [M.update(_emo(p0), REST, d, spec, 48 * 3600)["P"] for p0 in (-0.6, -0.3, 0.0, 0.5)]
    assert max(ends) - min(ends) < 1e-3 and not M.is_low({"P": ends[0]}, spec)


def test_guardrail_2_sleep_softens(quiet_cfg, make_heart):
    spec, d = quiet_cfg.heart.mood, _default(make_heart)
    awake = M.update(_emo(-0.4), _emo(-0.22), d, spec, 8 * 3600, awake=True)
    asleep = M.update(_emo(-0.4), _emo(-0.22), d, spec, 8 * 3600, awake=False)
    assert M.is_low(awake, spec) and asleep["P"] > awake["P"]


def test_guardrail_4_recovery_times(quiet_cfg, make_heart):
    spec, d = quiet_cfg.heart.mood, _default(make_heart)
    deepest = spec.bistable.center - spec.bistable.width
    m = M.update(_emo(deepest), REST, d, spec, 36 * 3600)
    assert not M.is_low(m, spec)            # sans événement : 36 h au plus
    m = M.update(_emo(deepest), _emo(0.4), d, spec, 2 * 3600)
    assert m["P"] > d["P"] * 0.5            # avec de la chaleur : 2 h au plus


def test_bistability_does_not_cap_joy(make_heart):
    h = make_heart()
    for _ in range(12):
        h.fire("praise")
        h.fire("warmth")
        h.advance(600)
    assert h.mood["P"] > 0.5                # le cube n'agit qu'entre les deux puits


def test_guardrail_5_low_hours_and_sliding_survive_restart(quiet_cfg, make_heart, tmp_path):
    h = make_heart()
    h.mood = dict(h.mood, P=-0.4)
    h.advance(3600)
    assert h.low_mood_hours() > 0.5
    snap = h.snapshot()
    h2 = make_heart()
    assert h2.restore(snap) and h2.low_mood_hours() == h.low_mood_hours()
    assert h2.mood_samples == h.mood_samples


def test_sliding_alone_but_not_when_surrounded(cfg):
    """Seul, le moral glisse en récupérant lentement : Valdar le sent. Entouré, non."""
    from conftest import DAY

    from valdar.heart import Heart

    def run(social):
        h = Heart(cfg, anchor=DAY, persist=False, rng_seed=1)
        seen = False
        for i in range(6 * 30):                 # 30 h par pas de 10 min
            if social and i % 6 == 0:
                h.fire("warmth", 0.5)
                h.interact()
            h.advance(600)
            seen = seen or h.sliding()
        return seen

    assert run(social=False)
    assert not run(social=True)


def test_sliding_needs_enough_samples(make_heart):
    h = make_heart()
    h.mood_samples = [0.1, 0.0, -0.1]
    assert not h.sliding()
    h.mood_samples = [0.1] * h.hc.mood.watch.window
    assert not h.sliding()                      # plat : rien à signaler
    assert M.lag1_autocorrelation([1.0, 2.0]) == 0.0
