import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from valdar.config import load  # noqa: E402
from valdar.heart import Heart  # noqa: E402
from valdar.sim.run import SIM_ANCHOR, _without_noise  # noqa: E402

# 11:00 un jour de semaine : Valdar est éveillé et hors des heures calmes.
DAY = SIM_ANCHOR + 3 * 3600


@pytest.fixture(scope="session")
def cfg():
    return load(REPO / "config" / "valdar.yaml")


@pytest.fixture(scope="session")
def quiet_cfg(cfg):
    """Configuration sans bruit (comparaisons exactes)."""
    return _without_noise(cfg)


@pytest.fixture()
def make_heart(quiet_cfg):
    def _make(anchor: float = DAY, cfg=None, **kw) -> Heart:
        return Heart(cfg or quiet_cfg, anchor=anchor, persist=False, **kw)
    return _make


@pytest.fixture()
def p2cfg(quiet_cfg, tmp_path):
    """Configuration sans bruit dont toutes les données vont dans un dossier temporaire."""
    c = quiet_cfg.model_copy(deep=True)
    c.storage.dir = str(tmp_path / "data")
    c._root = quiet_cfg.root
    return c


@pytest.fixture()
def runtime_factory(p2cfg):
    from valdar.llm.fake import FakeBackend
    from valdar.runtime import Runtime

    def _make(script=None, printer=None, anchor: float = DAY):
        rt = Runtime(p2cfg, llm=FakeBackend(script), persist=False, printer=printer)
        rt.heart.now = anchor
        rt.heart.last_interaction = anchor
        rt.heart.need_last = {k: anchor for k in rt.heart.need_last}
        rt.heart.awake = rt.heart._is_awake(anchor)
        rt.heart._refresh()
        return rt
    return _make
