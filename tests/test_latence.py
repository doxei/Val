"""Latence : le cache d'Ollama réchauffé, les plongements jamais sur le serveur de Gemma."""
from __future__ import annotations

from valdar.config.loader import EmbeddingsConfig
from valdar.memory.embedder import Embedder


def test_warm_reads_the_stable_prompt_without_touching_history(runtime_factory):
    rt = runtime_factory()
    rt.handle("salut")
    before = list(rt.agent.history)
    n = len(rt.llm.inner.calls) if hasattr(rt.llm, "inner") else len(rt.llm.calls)
    assert rt.agent.warm()
    calls = rt.llm.inner.calls if hasattr(rt.llm, "inner") else rt.llm.calls
    assert len(calls) == n + 1
    last = calls[-1]
    assert last["params"].max_tokens == 1 and last["tools"]
    assert last["messages"][-1]["content"].endswith("…")
    assert rt.agent.history == before


def test_warm_never_competes_with_a_conversation(runtime_factory):
    rt = runtime_factory()
    rt.agent.busy.set()
    assert not rt.agent.warm()


def test_embeddings_refuse_gemmas_server_by_default(p2cfg):
    from valdar.runtime import Runtime

    c = p2cfg.model_copy(deep=True)
    c.embeddings = EmbeddingsConfig(url=c.llm.url)
    rt = Runtime(c, persist=False, printer=False)
    assert not rt.embedder.available()
    assert "Gemma" in rt.embedder.last_error


def test_embeddings_default_to_a_separate_cpu_server(p2cfg):
    from valdar.runtime import Runtime

    rt = Runtime(p2cfg, persist=False, printer=False)
    assert rt.embedder.url.endswith(":11435")
    assert rt.embedder.url != p2cfg.llm.url


def test_embedder_measures_each_request():
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[1.0, 0.0]]}

    class Client:
        def post(self, *a, **k):
            return Resp()

    e = Embedder(EmbeddingsConfig(), "http://x", client=Client())
    assert e.embed_one("a") is not None and e.last_seconds is not None


def test_sidecar_reuses_a_running_server(monkeypatch):
    from valdar.memory import sidecar

    monkeypatch.setattr(sidecar, "alive", lambda url, timeout=1.0: True)
    s = sidecar.Sidecar(11435)
    assert s.start() and s.proc is None          # déjà là : rien de lancé
