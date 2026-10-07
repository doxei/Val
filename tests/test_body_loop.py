"""Boucle fermée cerveau ↔ corps (avenant 4 §1) : les organes gardent le poids d'une émotion
et le renvoient au cœur, sans jamais s'emballer ni dériver au repos."""
import pytest
from conftest import DAY

from valdar.heart import organs as organs_mod


def _without_feedback(cfg):
    c = cfg.model_copy(deep=True)
    for spec in c.heart.organs.values():
        spec.feedback = {}
    return c


def test_rest_is_unchanged_by_the_loop(make_heart, quiet_cfg):
    with_loop = make_heart()
    without = make_heart(cfg=_without_feedback(quiet_cfg))
    for h in (with_loop, without):
        h.advance(50 * 60)   # au repos : la boucle ne déplace (presque) rien
    for k in with_loop.variables:
        assert with_loop.variables[k] == pytest.approx(without.variables[k], abs=2e-3)


def test_belly_keeps_the_knot_after_the_fear(make_heart):
    h = make_heart()
    h.fire("threat")
    h.advance(120)
    peak = h.organs()["ventre"]
    h.advance(10 * 60)
    instant = organs_mod.activations(h.hc, h.sources())["ventre"]
    body = h.organs()["ventre"]
    assert peak > h.body_rest["ventre"] + 0.1, "la peur noue le ventre"
    assert body > instant + 0.015, "le ventre reste noué après que la peur est retombée"


def test_body_feeds_back_into_the_heart(make_heart, quiet_cfg):
    loop = make_heart()
    open_ = make_heart(cfg=_without_feedback(quiet_cfg))
    for h in (loop, open_):
        h.fire("threat")
        h.advance(20 * 60)
    assert loop.variables["cortisol"] > open_.variables["cortisol"] + 1e-3, \
        "sentir la boule au ventre entretient le stress"
    for h in (loop, open_):          # Olivier reste là : besoins satisfaits
        for _ in range(16):
            h.interact()
            h.advance(30 * 60)
    assert loop.variables["cortisol"] == pytest.approx(open_.variables["cortisol"], abs=0.02), \
        "et ça finit toujours par retomber"


def test_body_survives_a_restart(make_heart):
    h = make_heart()
    h.fire("threat")
    h.advance(300)
    snap = h.snapshot()
    other = make_heart()
    assert other.restore(snap)
    assert other.organs()["ventre"] == pytest.approx(h.organs()["ventre"])


def test_config_refuses_an_unstable_loop(quiet_cfg):
    raw = quiet_cfg.model_dump()
    raw["heart"]["organs"]["ventre"]["feedback"] = {"cortisol": 0.9}
    with pytest.raises(ValueError, match="trop fort"):
        type(quiet_cfg).model_validate(raw)


def test_day_anchor_is_awake(make_heart):
    assert make_heart(anchor=DAY).awake
