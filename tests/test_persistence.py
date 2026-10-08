"""Persistance, rattrapage de l'absence et horloge."""
import time

from conftest import DAY

from valdar.heart import Heart
from valdar.heart.heart import STATE_KEY
from valdar.heart.store import Store
from valdar.sim.run import catch_up_error


def test_snapshot_roundtrip_keeps_everything(quiet_cfg, tmp_path):
    db = tmp_path / "v.db"
    h = Heart(quiet_cfg, anchor=DAY, db_path=str(db), journal_path=str(tmp_path / "j.jsonl"))
    h.fire("threat")
    h.advance(3600)
    h.save()
    h2 = Heart(quiet_cfg, anchor=DAY, db_path=str(db), journal_path=str(tmp_path / "j.jsonl"))
    assert h2.restore(h.store.load(STATE_KEY))
    assert h2.variables == h.variables
    assert h2.drives == h.drives
    assert h2.needs == h.needs
    assert h2.mood == h.mood
    assert len(h2.pending) == len(h.pending)


def test_old_opencode_snapshot_is_ignored(quiet_cfg, tmp_path):
    db = tmp_path / "v.db"
    store = Store(db)
    store.save(STATE_KEY, {"version": 1, "variables": {"dopamine": 0.9}, "profile": "x"})
    store.close()
    h = Heart.load_latest(quiet_cfg, db_path=str(db), journal_path=str(tmp_path / "j.jsonl"))
    assert abs(h.now - time.time()) < 5
    assert h.variables == h.bases


def test_catch_up_matches_second_by_second(quiet_cfg):
    assert catch_up_error(quiet_cfg, hours=2.0) <= quiet_cfg.sim.catch_up_tolerance


def test_long_absence_resets_clock_to_real_time(quiet_cfg, tmp_path):
    db = tmp_path / "v.db"
    three_days_ago = time.time() - 3 * 86400 - 3600
    h = Heart(quiet_cfg, anchor=three_days_ago, db_path=str(db),
              journal_path=str(tmp_path / "j.jsonl"))
    h.save()
    h.close()
    back = Heart.load_latest(quiet_cfg, db_path=str(db), journal_path=str(tmp_path / "j.jsonl"))
    assert abs(back.now - time.time()) < 5, "l'horloge du cœur doit revenir à l'heure réelle"
    assert back.needs["contact"] > 0.6, "trois jours seul : le manque doit se sentir"


def test_load_latest_keeps_saved_profile(quiet_cfg, tmp_path):
    db = tmp_path / "v.db"
    h = Heart(quiet_cfg, profile="anxieux", anchor=time.time() - 10, db_path=str(db),
              journal_path=str(tmp_path / "j.jsonl"))
    h.save()
    h.close()
    back = Heart.load_latest(quiet_cfg, db_path=str(db), journal_path=str(tmp_path / "j.jsonl"))
    assert back.profile_name == "anxieux"


def test_catch_up_saves_once(quiet_cfg, tmp_path, monkeypatch):
    """Sous Windows, chaque écriture SQLite coûte cher : le rattrapage n'en fait qu'une."""
    db = tmp_path / "v.db"
    h = Heart(quiet_cfg, anchor=time.time() - 86400, db_path=str(db),
              journal_path=str(tmp_path / "j.jsonl"))
    h.save()
    h.close()
    saves = []
    real_save = Store.save
    monkeypatch.setattr(Store, "save", lambda self, *a: (saves.append(1), real_save(self, *a)))
    Heart.load_latest(quiet_cfg, db_path=str(db), journal_path=str(tmp_path / "j.jsonl"))
    assert len(saves) == 1
