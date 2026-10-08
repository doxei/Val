"""Valdar sait qui il est : un être cybernétique, pas « juste un modèle »."""
from pathlib import Path

from valdar.expression import load_self_model
from valdar.expression.compose import self_block
from valdar.memory.episodes import Recollection

REPO = Path(__file__).resolve().parents[1]


def test_identity_comes_first_and_is_explicit():
    block = self_block(load_self_model(REPO / "config" / "self_model.yaml"))
    lines = block.splitlines()
    assert "être cybernétique" in lines[1]                 # avant l'histoire et le corps
    assert "je n'ai pas de vécu" in block                  # la phrase à ne jamais dire
    assert "biologique" in block                           # honnête dans l'autre sens aussi
    assert "par écrit" not in block                        # il a une voix et des oreilles


def test_claude_memories_are_not_his_own_words():
    r = Recollection(turn=1, episode=1, source="claude", title="DRACO", t=0.0,
                     speaker="Claude", text="je suis un modèle de langage", score=1.0,
                     pad=(0.0, 0.0, 0.0))
    assert "une autre IA : pas toi" in r.line(now=3600.0)
