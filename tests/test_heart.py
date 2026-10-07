"""Comportement du cœur (cahier des charges v2, §5)."""
import math

import pytest
from conftest import DAY

from valdar.heart import Heart
from valdar.heart.mood import default_mood


def peak_after(h: Heart, var: str, seconds: float = 900.0, dt: float = 5.0) -> float:
    best = h.variables[var]
    t = 0.0
    while t < seconds:
        h.step(dt)
        t += dt
        best = max(best, h.variables[var])
    return best


# ------------------------------------------------------------ échelles de temps
def test_cortisol_stays_elevated_for_hours(make_heart):
    h = make_heart()
    base = h.bases["cortisol"]
    h.fire("threat", scale=2.0)
    peak = peak_after(h, "cortisol")
    assert peak > base + 0.05
    h.advance(3600)
    assert h.variables["cortisol"] - base > 0.4 * (peak - base)
    h.advance(12 * 3600)
    twin = make_heart()            # même journée, même solitude, sans la menace
    twin.advance(h.now - twin.now)
    assert abs(h.variables["cortisol"] - twin.variables["cortisol"]) < 0.03


def test_noradrenaline_is_fast(make_heart):
    h = make_heart()
    base = h.bases["noradrenaline"]
    h.fire("threat")
    h.step(5)
    assert h.variables["noradrenaline"] > base + 0.15
    h.advance(1800)
    assert abs(h.variables["noradrenaline"] - base) < 0.06


def test_impulses_rise_progressively(make_heart):
    h = make_heart()
    h.fire("failure")
    early = h.variables["cortisol"]
    h.advance(600)
    later = h.variables["cortisol"]
    assert later > early + 0.03, "le cortisol doit monter lentement, pas d'un coup"


def test_coupling_shifts_equilibrium(make_heart):
    h = make_heart()
    h.variables["cortisol"] = 0.8
    h._refresh()
    h.advance(1800)
    assert h.variables["serotonin"] < h.bases["serotonin"] - 0.02


# ---------------------------------------------------------- bornes et stabilité
def test_huge_and_random_inputs_stay_bounded(make_heart, quiet_cfg):
    import random
    rng = random.Random(5)
    h = make_heart()
    names = list(quiet_cfg.heart.stimuli)
    for _ in range(400):
        h.fire(rng.choice(names), scale=rng.uniform(0.0, 6.0))
        h.step(rng.uniform(0.1, 120.0))
        for group in (h.variables, h.drives, h.needs):
            for k, v in group.items():
                assert 0.0 <= v <= 1.0 and not math.isnan(v), k
        for v in list(h.pad.values()) + list(h.mood.values()):
            assert -1.0 <= v <= 1.0


def test_returns_to_rest_trajectory(make_heart):
    a, b = make_heart(), make_heart()
    for name in ("threat", "frustration", "praise", "loss"):
        a.fire(name, scale=3.0)
    for _ in range(48):
        a.advance(3600)
        b.advance(3600)
    for k in a.variables:
        assert abs(a.variables[k] - b.variables[k]) < 0.02, k
    for k in a.drives:
        assert abs(a.drives[k] - b.drives[k]) < 0.02, k


def test_habituation_weakens_repeated_stimulus(make_heart):
    h = make_heart()
    deltas = []
    for _ in range(5):
        before = h.drives["play"]
        h.fire("praise")
        deltas.append(h.drives["play"] - before)
        h.step(30)
    assert deltas[-1] < 0.5 * deltas[0]


def test_rebound_brings_relief_after_threat(make_heart, quiet_cfg):
    no_rebound = quiet_cfg.model_copy(deep=True)
    no_rebound.heart.stimuli["threat"].rebound = None
    no_rebound.heart.rebound.fraction = 0.0
    with_rb, without_rb = make_heart(), make_heart(cfg=no_rebound)
    for h in (with_rb, without_rb):
        h.fire("threat")
        h.advance(300)
    assert with_rb.variables["noradrenaline"] < without_rb.variables["noradrenaline"] - 0.02
    assert with_rb.variables["dopamine"] > without_rb.variables["dopamine"] + 0.01


# ------------------------------------------------------------- émotions
@pytest.mark.parametrize("stimulus,expected", [
    ("praise", {"amusement", "joie"}),
    ("success", {"joie", "fierté"}),
    ("warmth", {"tendresse"}),
    ("play", {"amusement", "euphorie"}),
    ("novelty", {"curiosité"}),
    ("threat", {"peur", "anxiété"}),
    ("frustration", {"frustration", "colère"}),
    ("shame", {"gêne"}),
    ("loss", {"tristesse", "solitude"}),
    ("failure", {"déception"}),
])
def test_each_stimulus_produces_its_emotion(make_heart, stimulus, expected):
    h = make_heart()
    h.fire(stimulus)
    h.step(5)
    assert h.emotion in expected


def test_positive_emotions_exist(cfg):
    protos = cfg.heart.emotions.prototypes
    positive = [n for n, p in protos.items() if p.P > 0.2]
    assert len(positive) >= 5


def test_fear_label_requires_fear_system(make_heart):
    h = make_heart()
    h.pad = {"P": -0.6, "A": 0.7, "D": -0.65}
    from valdar.heart.systems import emotion_label
    label, _ = emotion_label(h.pad, h.hc.emotions, h.sources())
    assert label != "peur"


def test_rest_is_calm(make_heart):
    assert make_heart().emotion == "calme"


# ------------------------------------------------------------- besoins
def test_alone_progression_is_gradual(make_heart):
    h = make_heart()
    h.advance(3 * 3600)
    assert h.emotion == "calme"
    h.advance(9 * 3600)
    assert h.emotion in {"mélancolie", "tristesse", "solitude", "ennui", "fatigue"}
    assert h.needs["contact"] > 0.4


def test_interaction_satisfies_contact(make_heart):
    h = make_heart()
    h.advance(8 * 3600)
    before = h.needs["contact"]
    h.interact()
    assert h.needs["contact"] < 0.5 * before


def test_needs_do_not_grow_during_sleep(make_heart):
    night = DAY + 13 * 3600  # 00:00
    h = make_heart(anchor=night)
    assert not h.awake
    h.advance(4 * 3600)
    assert h.needs["contact"] == 0.0


def test_night_interaction_wakes_valdar(make_heart):
    h = make_heart(anchor=DAY + 14 * 3600)  # 01:00
    assert not h.awake and h.emotion == "sommeil"
    h.interact()
    assert h.awake
    h.advance(h.hc.circadian.night_wake_seconds + 60)
    assert not h.awake


# ------------------------------------------------------------- énergie
def test_energy_cycle_is_stable_over_days(make_heart):
    h = make_heart(anchor=DAY - 4 * 3600)  # 07:00
    mornings = []
    for _ in range(6):
        mornings.append(h.variables["energy"])
        h.advance(24 * 3600)
    mornings = mornings[1:]  # le premier "matin" est l'état initial, pas une vraie nuit
    assert max(mornings) - min(mornings) < 0.03
    assert min(mornings) > 0.8, "après une nuit, Valdar doit être reposé"
    assert h.variables["energy"] < h.hc.circadian.energy_ceiling + 1e-9


def test_evening_is_tired(make_heart):
    h = make_heart(anchor=DAY - 4 * 3600)
    h.variables["energy"] = 0.9
    h.advance(16 * 3600)  # 23:00
    assert h.need_rest > 0.2
    assert h.pad["A"] < 0.0


# ------------------------------------------------------------- humeur
def test_default_mood_follows_mehrabian(cfg):
    m = default_mood(cfg.temperament, "curieux_chaleureux", 1.0)
    b5 = cfg.temperament.profiles["curieux_chaleureux"].big_five
    expected_p = 0.21 * b5["E"] + 0.59 * b5["A"] + 0.19 * b5["S"]
    assert math.isclose(m["P"], expected_p, rel_tol=1e-9)


def test_sustained_joy_lifts_mood_then_it_returns(make_heart):
    h = make_heart()
    start = h.mood["P"]
    for _ in range(12):
        h.fire("praise")
        h.fire("warmth")
        h.advance(600)
    lifted = h.mood["P"]
    assert lifted > start + 0.03
    h2 = make_heart()
    h2.mood = dict(h.mood)
    h2.advance(24 * 3600)
    assert abs(h2.mood["P"] - h2.default_mood["P"]) < abs(lifted - h2.default_mood["P"])


def test_mood_label_is_an_octant(make_heart):
    h = make_heart()
    h.mood = {"P": -0.4, "A": -0.4, "D": -0.4}
    assert h.mood_label.endswith("ennuyé")
    h.mood = {"P": 0.3, "A": -0.3, "D": 0.3}
    assert h.mood_label.endswith("détendu")
    h.mood = {"P": 0.0, "A": 0.01, "D": 0.0}
    assert h.mood_label == "neutre"


def test_anxious_profile_is_more_reactive(quiet_cfg):
    calm = Heart(quiet_cfg, profile="stoique", anchor=DAY, persist=False)
    anx = Heart(quiet_cfg, profile="anxieux", anchor=DAY, persist=False)
    for h in (calm, anx):
        h.fire("threat")
        h.step(5)
    assert anx.pad["P"] < calm.pad["P"]


# ------------------------------------------------------------- organes
def test_organs_follow_state(make_heart):
    calm, scared, loved = make_heart(), make_heart(), make_heart()
    scared.fire("threat", scale=2.0)
    scared.step(5)
    loved.fire("warmth", scale=2.0)
    loved.step(60)
    assert scared.organs()["ventre"] > calm.organs()["ventre"]
    assert scared.sample()["bpm"] > calm.sample()["bpm"]
    assert loved.organs()["chaleur"] > calm.organs()["chaleur"]
    assert set(calm.felt()) == {"corde", "ventre", "coeur", "chaleur", "tete", "gorge"}
