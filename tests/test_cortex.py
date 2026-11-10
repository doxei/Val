"""Avenant 6 : le cortex — sentiments injectés dans les couches, psyché, format Gemma 4."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from valdar.config.loader import Identity
from valdar.cortex import gemma
from valdar.cortex.affect import Bond, Feeling, psyche, raw
from valdar.cortex.backend import CortexBackend
from valdar.cortex.vectors import Vectors, build
from valdar.llm.backend import ChatResult, GenParams, LLMError

NOW = 1_790_000_000.0


# ------------------------------------------------------------------ psyché
def test_heart_sources_become_bounded_coefficients(cfg):
    c = raw(cfg.cortex, {"P": 0.5, "A": -0.4, "fear_exc": 0.6, "need_rest": 0.0})
    assert 0 < c["valence"] < 1 and c["activation"] < 0 and c["peur"] > 0.5
    assert c["fatigue"] == 0 and set(c) == set(cfg.cortex.axes)


def test_secure_base_soothes_fear_only_with_a_loved_one_present(cfg):
    src = {"fear_exc": 0.5}
    alone = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(), src)
    loved = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(present=True, affection=0.9), src)
    cold = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(present=True, affection=-0.5), src)
    assert loved["peur"] < alone["peur"] and cold["peur"] == alone["peur"]


def test_contagion_pulls_toward_the_other_and_more_when_attached(cfg):
    src = {"P": 0.0}
    sad = dict(present=True, other_p=-0.8)
    far = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(affection=0.0, **sad), src)
    near = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(affection=1.0, **sad), src)
    assert near["valence"] < far["valence"] < 0


def test_negativity_bias_and_regulation_in_front_of_a_child(cfg):
    up = psyche(cfg.cortex, raw(cfg.cortex, {"P": 0.3}), Bond(), {})
    down = psyche(cfg.cortex, raw(cfg.cortex, {"P": -0.3}), Bond(), {})
    assert abs(down["valence"]) > abs(up["valence"])
    src = {"rage_exc": 0.5}
    adult = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(present=True), src)
    child = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(present=True, minor=True), src)
    assert child["colere"] < 0.5 * adult["colere"]


def test_total_intensity_is_capped_and_small_feelings_are_silent(cfg):
    src = {"P": -1, "A": 1, "D": -1, "fear_exc": 1, "rage_exc": 1, "panic_exc": 1,
           "need_rest": 1, "need_contact": 1}
    c = psyche(cfg.cortex, raw(cfg.cortex, src), Bond(), src)
    assert math.sqrt(sum(v * v for v in c.values())) <= cfg.cortex.psyche.budget + 1e-9
    tiny = psyche(cfg.cortex, raw(cfg.cortex, {"P": 0.02}), Bond(), {})
    assert tiny["valence"] == 0.0


def test_feeling_glides_and_only_moves_by_whole_steps(cfg):
    f = Feeling(cfg.cortex)
    first = f.update({"P": 0.6}, Bond(), NOW)
    step = cfg.cortex.psyche.step
    ratio = first["valence"] / step
    assert first["valence"] > 0 and abs(ratio - round(ratio)) < 1e-6
    soon = f.update({"P": -0.6}, Bond(), NOW + 2)        # 2 s plus tard : il n'a pas basculé
    assert soon["valence"] > 0
    later = f.update({"P": -0.6}, Bond(), NOW + 300)
    assert later["valence"] < 0
    same = f.update({"P": -0.6 + 0.01}, Bond(), NOW + 600)   # variation infime : rien ne bouge
    assert same == later
    assert "tristesse" in f.words()


# ------------------------------------------------------------------ vecteurs
def _synthetic(rng, n_layers=6, dim=32):
    true = rng.standard_normal((2, n_layers, dim))
    true /= np.linalg.norm(true, axis=2, keepdims=True)
    noise = lambda n: rng.standard_normal((n, n_layers, dim)) * 0.3   # noqa: E731
    pos = {"a": noise(8) + 2 * true[0], "b": noise(8) + 2 * true[1]}
    neg = {"a": noise(8) - 2 * true[0], "b": noise(8) - 2 * true[1]}
    return true, build(["a", "b"], pos, neg, noise(10) * 10, model="jouet")


def test_directions_are_recovered_and_survive_disk(tmp_path):
    true, v = _synthetic(np.random.default_rng(0))
    cos = np.einsum("ld,ld->l", v.dirs[0], true[0])
    assert cos.min() > 0.95 and v.quality[0].min() > 3
    v.save(tmp_path / "v.npz")
    w = Vectors.load(tmp_path / "v.npz")
    assert w.axes == ["a", "b"] and w.model == "jouet"
    assert np.allclose(w.dirs, v.dirs, atol=2e-3) and (w.n_layers, w.n_embd) == (7, 32)


def test_injection_moves_alpha_of_the_stream_in_total_and_only_where_asked():
    _, v = _synthetic(np.random.default_rng(1))
    assert v.injection({}, 0.1, 2, 4) is None and v.injection({"a": 1.0}, 0.0, 2, 4) is None
    buf = v.injection({"a": 1.0}, 0.2, 2, 4)
    assert buf.shape == (6, 32) and not buf[0].any() and not buf[4:].any()
    moved = sum(np.linalg.norm(buf[i]) / v.norms[i] for i in range(1, 4))
    assert moved == pytest.approx(0.2, rel=1e-3)
    both = v.injection({"a": 0.5, "b": -0.5, "inconnu": 1.0}, 0.2, 2, 4)
    assert np.allclose(both, 0.5 * buf + v.injection({"b": -0.5}, 0.2, 2, 4), atol=1e-6)


# ------------------------------------------------------------------ format Gemma 4
def test_history_is_folded_into_one_model_turn():
    h = [{"role": "user", "content": "salut"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"function": {"name": "fouiller", "arguments": {"q": "chien"}}}]},
         {"role": "tool", "tool_name": "fouiller", "content": "Sony"},
         {"role": "assistant", "content": "Ah oui, Sony."},
         {"role": "tool", "tool_name": "perdu", "content": "x"}]
    t = gemma.to_template(h)
    assert [m["role"] for m in t] == ["user", "assistant", "user"]
    assert t[1]["tool_responses"] == [{"name": "fouiller", "response": "Sony"}]
    assert t[1]["content"] == "Ah oui, Sony." and "perdu" in t[2]["content"]


def test_tool_calls_are_read_from_gemma_marks():
    text = ('Attends. <|tool_call>call:fouiller{n:3,question:<|"|>le chien, {Sony}<|"|>,'
            'ok:true,l:[1,<|"|>a<|"|>],o:{x:1.5}}<tool_call|><|tool_response>')
    said, calls = gemma.parse(text)
    assert said == "Attends."
    assert calls == [("fouiller", {"n": 3, "question": "le chien, {Sony}", "ok": True,
                                   "l": [1, "a"], "o": {"x": 1.5}})]
    assert gemma.parse("<|channel>thought\nhum<channel|>Salut.<turn|>") == ("Salut.", [])
    assert gemma.parse("<|tool_call>call:heure{}<tool_call|>")[1] == [("heure", {})]


def test_voice_never_hears_a_tool_call():
    out: list[str] = []
    f = gemma.SpeechFilter(out.append)
    for piece in ["Sal", "ut <", "b> toi <|to", "ol_call>call:x{}", "<tool_call|>"]:
        f.feed(piece)
    f.close()
    assert "".join(out) == "Salut <b> toi "
    out.clear()
    f = gemma.SpeechFilter(out.append)
    f.feed("Fini.<tur")
    f.feed("n|>")
    f.close()
    assert "".join(out) == "Fini."


# ------------------------------------------------------------------ le backend
TEMPLATE = ("{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}"
            "<turn|>\n{% endfor %}<|turn>model\n")


class FakeEngine:
    """Un moteur sans modèle : garde ce qu'on lui injecte, répond ce qu'on lui dit."""

    def __init__(self, tmp: Path, reply: str = "Salut.", n_layers: int = 7, dim: int = 32,
                 broken: bool = False):
        self.path = tmp / "modele.gguf"
        self.path.write_bytes(b"x")
        self.reply, self.n_layers, self.n_embd = reply, n_layers, dim
        self.template, self.bos, self.steered = TEMPLATE, "<bos>", False
        self.buf = None
        self.prompts: list[str] = []
        self.broken = broken

    def load(self) -> None:
        if self.broken:
            raise LLMError("pas de carte graphique")

    def steer(self, buf, first=1, last=None) -> None:
        self.buf, self.steered = buf, buf is not None
        self.range = (first, last)

    def generate(self, prompt, params, on_text=None):
        self.prompts.append(prompt)
        if on_text is not None:
            for i in range(0, len(self.reply), 5):
                on_text(self.reply[i:i + 5])
        return self.reply, {"eval_count": 3, "prompt_eval_count": 9}

    def close(self) -> None:
        pass


@pytest.fixture()
def ccfg(p2cfg, tmp_path):
    c = p2cfg.model_copy(deep=True)
    c._root = tmp_path
    c.llm.backend = "cortex"
    c.cortex.free_ollama = False
    c.cortex.layers = (2, 4)
    return c


def _vectors(c, dim=32):
    rng = np.random.default_rng(3)
    axes = list(c.cortex.axes)
    d = rng.standard_normal((len(axes), 6, dim)).astype(np.float32)
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return Vectors(axes, d, np.full(6, 10.0, dtype=np.float32),
                   np.ones((len(axes), 6), dtype=np.float32), "jouet")


def test_feelings_reach_the_layers_and_leave_them(ccfg, tmp_path):
    eng = FakeEngine(tmp_path)
    b = CortexBackend(ccfg, engine=eng, vectors=_vectors(ccfg))
    assert b.feel({"valence": 0.5, "peur": 0.2}) is True
    assert eng.buf.shape == (6, 32) and eng.range == (2, 4) and eng.buf[1].any()
    assert not eng.buf[0].any() and not eng.buf[4].any()
    before = eng.buf.copy()
    b.cc.alpha *= 2                                   # réglage changé à chaud
    b.feel({"valence": 0.5, "peur": 0.2})
    assert np.allclose(eng.buf, 2 * before)
    assert b.feel({}) is False and eng.steered is False
    b.cc.inject = False
    assert b.feel({"valence": 1.0}) is False and eng.steered is False


def test_missing_or_foreign_vectors_never_block_the_model(ccfg, tmp_path):
    eng = FakeEngine(tmp_path)
    b = CortexBackend(ccfg, engine=eng)
    assert b.feel({"valence": 0.5}) is False and "absents" in b.problem
    assert b.chat([{"role": "user", "content": "salut"}]).content == "Salut."
    other = _vectors(ccfg, dim=16)
    other.save(tmp_path / ccfg.cortex.vectors)
    b2 = CortexBackend(ccfg, engine=FakeEngine(tmp_path))
    assert b2.feel({"valence": 0.5}) is False and "autre modèle" in b2.problem


def test_chat_speaks_calls_tools_and_refuses_images(ccfg, tmp_path):
    reply = 'Je regarde. <|tool_call>call:heure{ville:<|"|>Thann<|"|>}<tool_call|>'
    eng = FakeEngine(tmp_path, reply=reply)
    b = CortexBackend(ccfg, engine=eng, vectors=_vectors(ccfg))
    heard: list[str] = []
    r = b.chat([{"role": "user", "content": "quelle heure ?"}], system="Tu es Valdar.",
               params=GenParams(), on_token=heard.append)
    assert r.content == "Je regarde." and "".join(heard) == "Je regarde. "
    assert [(c.name, c.arguments) for c in r.tool_calls] == [("heure", {"ville": "Thann"})]
    assert eng.prompts[0].startswith("<bos><|turn>system\nTu es Valdar.")
    with pytest.raises(LLMError):
        b.chat([{"role": "user", "content": "regarde", "images": ["abc"]}])


def test_without_a_working_cortex_he_still_thinks_through_ollama(ccfg, tmp_path, monkeypatch):
    import valdar.llm.ollama as ollama

    class Stub:
        def __init__(self, cfg):
            pass

        def chat(self, messages, system="", tools=None, params=None, **kw):
            return ChatResult("je pense quand même.")

        def available(self):
            return True

    monkeypatch.setattr(ollama, "OllamaBackend", Stub)
    b = CortexBackend(ccfg, engine=FakeEngine(tmp_path, broken=True))
    assert b.chat([{"role": "user", "content": "salut"}]).content == "je pense quand même."
    assert "repli sur Ollama" in b.problem and b.feel({"valence": 1.0}) is False


# ------------------------------------------------------------------ dans Valdar
def test_runtime_feels_before_speaking_and_can_drop_the_text_block(runtime_factory):
    rt = runtime_factory()
    felt: list[dict[str, float]] = []
    rt.llm.inner.supports_feeling = True
    rt.llm.inner.feel = lambda coefs: felt.append(dict(coefs))
    rt.heart.fire("warmth", scale=1.0)
    rt.heart.advance(30)
    rt.handle("salut")
    assert felt and set(felt[-1]) <= set(rt.cfg.cortex.axes)
    call = rt.llm.inner.calls[-1]
    assert "TON ÉTAT INTÉRIEUR" in call["messages"][-1]["content"]
    rt.cfg.llm.backend, rt.cfg.cortex.describe_state = "cortex", False
    try:
        rt.handle("et là ?")
        assert "TON ÉTAT INTÉRIEUR" not in rt.llm.inner.calls[-1]["messages"][-1]["content"]
    finally:
        rt.cfg.llm.backend, rt.cfg.cortex.describe_state = "ollama", True


def test_the_bond_with_the_person_colours_what_is_injected(runtime_factory):
    rt = runtime_factory()
    who = rt.cfg.agent.console_identity
    for _ in range(12):
        rt.relations.observe(who, "merci, t'es génial, j'adore bosser avec toi")
    alone = dict(rt.feel(None))
    rt.feeling.last_t = None                         # pas d'inertie entre les deux mesures
    together = rt.feel(who)
    assert together.get("attachement", 0) > alone.get("attachement", 0)
    stranger = Identity()
    rt.feeling.last_t = None
    assert rt.feel(stranger).get("confiance", 0) <= together.get("confiance", 0)


# ------------------------------------------------------------------ le vrai llama.cpp
@pytest.mark.skipif(not os.environ.get("VALDAR_TINY_GGUF"),
                    reason="VALDAR_TINY_GGUF : petit modèle GGUF pour essayer le vrai moteur")
def test_real_engine_reads_layers_and_steering_changes_the_output(cfg):
    from valdar.cortex.engine import Engine

    cc = cfg.cortex.model_copy(update={"model_path": os.environ["VALDAR_TINY_GGUF"],
                                       "gpu_layers": 0, "threads": 2})
    e = Engine(cc, Path("."), capture=True, n_ctx=256)
    e.load()
    acts = e.read("Bonjour, ", "voici la fin du texte")
    assert acts.shape == (e.n_layers - 1, e.n_embd) and acts[0].any()
    p = GenParams(temperature=0.0, max_tokens=6)
    plain, _ = e.generate("Bonjour", p)
    e.steer(np.random.default_rng(0).standard_normal(acts.shape).astype(np.float32) * 3, 1, 2)
    pushed, _ = e.generate("Bonjour", p)
    e.steer(None)
    again, _ = e.generate("Bonjour", p)
    assert pushed != plain and again == plain
    e.close()
