"""Vrai modèle de sens (embeddings) : compréhension au-delà des mots, repli sans casse."""
import httpx
import numpy as np

from valdar.config.loader import EmbeddingsConfig
from valdar.knowledge import Knowledge
from valdar.memory import Episodic
from valdar.memory.embedder import Embedder

# Faux modèle : des « concepts » communs à des mots différents (ce que fait un vrai modèle).
CONCEPTS = {"plateau": 0, "verre": 0, "vitre": 0, "bed": 0, "chat": 1, "minou": 1,
            "basilic": 2, "plante": 2}


def _vec(text):
    v = np.full(8, 0.01, dtype=np.float32)
    for w in text.lower().replace("'", " ").split():
        if w in CONCEPTS:
            v[CONCEPTS[w]] += 1.0
    return v.tolist()


def _client(calls, fail=False):
    def handler(req):
        import json
        body = json.loads(req.content)
        calls.append(body)
        if fail:
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json={"embeddings": [_vec(t) for t in body["input"]]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def _emb(calls, fail=False, clock=lambda: 0.0, **kw):
    cfg = EmbeddingsConfig(batch=4, min_similarity=0.5, **kw)
    return Embedder(cfg, "http://ollama", client=_client(calls, fail), clock=clock)


def test_embedder_runs_on_cpu_and_normalizes():
    calls = []
    m = _emb(calls).embed_many(["le plateau en verre", "mon chat"])
    assert calls[0]["options"] == {"num_gpu": 0} and calls[0]["model"] == "bge-m3"
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_failure_falls_back_and_retries_later():
    now = {"t": 0.0}
    e = _emb([], fail=True, clock=lambda: now["t"])
    assert e.embed_one("x") is None and not e.available()
    now["t"] = e.cfg.retry_seconds + 1
    assert e.available()
    assert not Embedder(EmbeddingsConfig(backend="hash"), "http://x").available()


def test_memory_understands_meaning_after_backfill(cfg, tmp_path):
    calls = []
    e = _emb(calls)
    mem = Episodic(tmp_path / "e.db", cfg.episodic, e)
    t0 = 1_790_000_000.0
    mem.log_turn("Olivier", "le plateau en verre a explosé", t0)
    mem.log_turn("Olivier", "mon chat dort sur l'imprimante", t0 + 100)
    q = "la vitre du bed"                       # aucun mot en commun avec le souvenir
    assert not any("explosé" in r.text for r in mem.recall(q, now=t0 + 9e5, touch=False))
    assert mem.backfill() == 2 and mem.backfill() == 0
    hits = mem.recall(q, now=t0 + 9e5, touch=False)
    assert hits and "explosé" in hits[0].text


def test_knowledge_understands_meaning_and_model_change_redoes(tmp_path):
    calls = []
    e = _emb(calls)
    k = Knowledge(tmp_path / "k.db", embedder=e)
    k.ingest("notes", [{"title": "Plateau", "text": "Nettoyer le plateau en verre à l'alcool "
                                                    "isopropylique avant chaque impression."},
                       {"title": "Basilic", "text": "Arroser le basilic quand la terre est "
                                                    "sèche en surface, jamais les feuilles."}])
    assert k.backfill() == 2
    hits = k.search("comment je lave la vitre du bed ?")
    assert hits and hits[0].title == "Plateau"
    e.cfg = EmbeddingsConfig(batch=4, model="autre-modele")
    assert k.backfill() == 2                    # autre modèle : tout est refait
