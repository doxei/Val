"""Phase 2c : mémoire épisodique, contexte temporel, amorçage, humeur, reprise du fil."""
import numpy as np
import pytest
from conftest import prompt_of

from valdar.config.loader import EpisodicConfig
from valdar.llm.backend import ChatResult
from valdar.memory import Episodic, TemporalContext
from valdar.memory.vectors import embed

T0 = 1_790_000_000.0


@pytest.fixture()
def mem(tmp_path):
    return Episodic(tmp_path / "ep.db", EpisodicConfig(exclude_recent_seconds=0))


def test_vectors_tolerate_conjugation_and_typos():
    a = embed("j'imprime une pièce en PETG")
    b = embed("on imprimait des pieces en petg")
    c = embed("la recette de la tarte aux pommes")
    assert float(a @ b) > 0.4 > float(a @ c)


def test_context_drifts_at_several_speeds():
    ctx = TemporalContext([30, 86400], 64, 10)
    f1, f2 = np.zeros(64, np.float32), np.zeros(64, np.float32)
    f1[0], f2[1] = 1, 1
    ctx.update(f1, T0)
    ctx.update(f2, T0 + 600)
    fast, slow = ctx.snapshot()
    assert fast[1] > 0.99, "l'échelle courte ne garde que le présent"
    assert slow[0] > 0.9, "l'échelle longue garde encore le passé"


def test_episodes_split_on_long_silence(mem):
    mem.log_turn("Olivier", "salut", T0)
    mem.log_turn("Valdar", "salut toi", T0 + 60)
    mem.log_turn("Olivier", "re", T0 + 4 * 3600)
    with mem._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 2


def test_recall_finds_by_meaning_and_recency_breaks_ties(mem):
    mem.log_turn("Olivier", "la buse de la CR-10S est bouchée, le PLA ne sort plus", T0)
    mem.log_turn("Olivier", "on parle du chien et de la promenade", T0 + 3600)
    mem.log_turn("Olivier", "la buse est encore bouchée, le PLA ne sort plus", T0 + 40 * 86400)
    now = T0 + 40 * 86400 + 600
    got = mem.recall("buse bouchée plus de PLA", now=now, k=2, touch=False)
    assert len(got) == 2 and all("buse" in r.text for r in got)
    assert got[0].t > got[1].t, "à ressemblance égale, le plus récent revient d'abord"


def test_contiguity_priming_is_forward_biased_and_fades(mem):
    texts = ["on monte la carte Octopus", "on flashe Klipper sur l'Octopus",
             "on câble les moteurs pas à pas", "on règle le courant des drivers TMC",
             "on teste les fins de course"]
    for i, t in enumerate(texts):
        mem.log_turn("Olivier", t, T0 + i * 60)
    mem.activation.clear()
    now = T0 + 86400
    hit = mem.recall("câbler les moteurs pas à pas", now=now, k=1)[0]
    assert "moteurs" in hit.text
    ids = {r.text: r.turn for r in [hit]}
    level = {t: mem._activation_level(tid, now) for tid, t in
             [(i + 1, txt) for i, txt in enumerate(texts)]}
    assert level[texts[3]] > level[texts[1]] > 0, "le suivant est plus amorcé que le précédent"
    assert ids
    wm = mem.working_memory(now + 30, exclude=[hit.turn])
    assert texts[3] in [r.text for r in wm]
    assert mem.working_memory(now + 3600, exclude=[hit.turn]) == [], "l'amorçage s'éteint"


def test_mood_congruent_recall(mem):
    mem.log_turn("Olivier", "l'impression a raté, je suis dégoûté", T0, pad=(-0.7, 0.3, -0.2))
    mem.log_turn("Olivier", "l'impression a réussi, je suis ravi", T0 + 60, pad=(0.7, 0.3, 0.2))
    now = T0 + 10 * 86400
    sad = mem.recall("l'impression", now=now, pad=(-0.6, 0.2, -0.2), k=1, touch=False)[0]
    happy = mem.recall("l'impression", now=now, pad=(0.6, 0.2, 0.2), k=1, touch=False)[0]
    assert "raté" in sad.text and "réussi" in happy.text


def test_rehearsed_memories_last_longer(mem):
    a = mem.log_turn("Olivier", "le mot de passe du wifi atelier est sur le frigo", T0)
    b = mem.log_turn("Olivier", "le code du portail atelier est sur le frigo", T0 + 1)
    idx = mem._load_index()
    for k in range(1, 6):   # rappelé à intervalles espacés
        idx["accesses"].setdefault(a, []).append(T0 + k * 86400)
    now = T0 + 30 * 86400
    assert mem.base_level(a, T0, now, idx["accesses"]) > mem.base_level(b, T0, now,
                                                                         idx["accesses"])


def test_thread_and_preload_after_restart(runtime_factory):
    rt = runtime_factory([ChatResult(content="Ok, on reprend demain.")])
    rt.handle("on arrête là pour ce soir, on finira le support de téléphone demain")
    rt2 = runtime_factory([ChatResult(content="Alors, ce support ?")])
    assert rt2.agent.thread is not None
    assert [m["role"] for m in rt2.agent.history] == ["user", "assistant"]
    rt2.handle("salut")
    system = prompt_of(rt2.llm.calls[0])
    assert "LE FIL" in system and "support de téléphone" in system


def test_reinstatement_moves_the_heart(runtime_factory):
    rt = runtime_factory([ChatResult(content="Ah oui…")])
    mem = rt.memory
    mem.log_turn("Olivier", "le plateau en verre a explosé ce matin, je suis effondré", T0,
                 pad=(-0.8, 0.4, -0.5))
    before = rt.heart.variables["cortisol"]
    rt.handle("tu te souviens du plateau en verre ?")
    assert "plateau en verre" in prompt_of(rt.llm.calls[0])
    assert rt.heart.variables["cortisol"] > before or any(
        p["target"] == "cortisol" for p in rt.heart.pending)
