"""Critères d'acceptation de la phase 1 (cahier v2, §16)."""
import pytest
from pydantic import ValidationError

from valdar.config import load
from valdar.sim.run import acceptance, recovery_seconds, run_sim


def test_sim_72h_no_variable_stuck_at_bounds(cfg):
    rep = run_sim(cfg, days=3.0, seed=42)
    for var, ratio in rep.stuck.items():
        assert ratio <= cfg.sim.stuck_limit, f"{var} collé {ratio:.1%} du temps"


def test_sim_has_varied_emotions(cfg):
    rep = run_sim(cfg, days=3.0, seed=42)
    assert len(rep.emotions) >= cfg.sim.min_distinct_emotions
    total = sum(rep.emotions.values())
    positive = sum(v for k, v in rep.emotions.items()
                   if k in {"joie", "amusement", "tendresse", "curiosité", "sérénité",
                            "euphorie", "fierté"})
    assert positive / total > 0.05


def test_sim_reproducible_same_seed(cfg):
    assert run_sim(cfg, days=0.5, seed=7).fingerprint == run_sim(cfg, days=0.5, seed=7).fingerprint


def test_sim_different_seed_differs(cfg):
    assert run_sim(cfg, days=0.5, seed=7).fingerprint != run_sim(cfg, days=0.5, seed=8).fingerprint


@pytest.mark.parametrize("stimulus", ["threat", "frustration", "loss", "praise", "warmth"])
def test_recovery_after_stimulus(cfg, stimulus):
    rec = recovery_seconds(cfg, stimulus)
    assert rec is not None and rec < 48 * 3600


def test_sim_never_touches_real_database(tmp_path, monkeypatch):
    cfg = load()
    real_db = cfg.storage_path(cfg.storage.db)
    before = real_db.stat().st_mtime if real_db.exists() else None
    run_sim(cfg, days=0.2, seed=1)
    after = real_db.stat().st_mtime if real_db.exists() else None
    assert before == after


def test_config_rejects_typos(tmp_path):
    from valdar.config import DEFAULT_CONFIG_PATH
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  tick_seconds: 1.0", "  tick_secondz: 1.0")
    bad = tmp_path / "config" / "valdar.yaml"
    bad.parent.mkdir()
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load(bad)


def test_phase1_acceptance(cfg):
    results = acceptance(cfg)
    failed = [(n, d) for n, ok, d in results if not ok]
    assert not failed, failed
