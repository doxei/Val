"""Le cortex comme modèle de langage de Valdar (même interface qu'Ollama), plus `feel` :
poser les sentiments du moment dans les couches avant de penser ou de parler."""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from valdar.cortex import gemma
from valdar.cortex.engine import Engine
from valdar.cortex.vectors import Vectors
from valdar.llm.backend import ChatResult, GenParams, LLMError, ToolCall


def free_ollama(cfg: Any) -> None:
    """Demande à Ollama de décharger Gemma : deux Gemma ne tiennent pas sur la carte."""
    with contextlib.suppress(Exception):
        import httpx

        httpx.post(cfg.llm.url.rstrip("/") + "/api/generate",
                   json={"model": cfg.llm.model, "keep_alive": 0}, timeout=3.0)


class CortexBackend:
    supports_stream = True
    supports_feeling = True

    def __init__(self, cfg: Any, engine: Engine | None = None, vectors: Vectors | None = None):
        self.cfg = cfg.llm
        self.cc = cfg.cortex
        self.root: Path = cfg.root
        self.engine = engine or Engine(self.cc, self.root, n_ctx=cfg.llm.num_ctx)
        self.vectors = vectors
        self._vectors_tried = vectors is not None
        self.feeling: dict[str, float] = {}
        self.problem = ""               # pourquoi l'injection est coupée, s'il y a lieu
        self._lock = threading.RLock()
        self._cfg = cfg
        self._freed = False
        self._applied: Any = None
        self._fallback: Any = None      # Ollama, si le cortex ne peut pas se charger

    # ------------------------------------------------------------ état
    def available(self) -> bool:
        if self._fallback is not None:
            return self._fallback.available()
        return self.engine.path.is_file()

    def has_model(self) -> bool:
        return self.available()

    def _load_vectors(self) -> None:
        if self._vectors_tried:
            return
        self._vectors_tried = True
        path = self.root / self.cc.vectors
        if not path.is_file():
            self.problem = "vecteurs absents (lance « valdar cortex apprendre »)"
            return
        try:
            v = Vectors.load(path)
        except Exception as exc:
            self.problem = f"vecteurs illisibles : {exc}"
            return
        self.engine.load()
        if (v.n_layers, v.n_embd) != (self.engine.n_layers, self.engine.n_embd):
            self.problem = ("vecteurs appris sur un autre modèle "
                            "(relance « valdar cortex apprendre »)")
            return
        self.vectors = v

    def _free_ollama(self) -> None:
        """Gemma dans Ollama et Gemma ici ne tiennent pas ensemble sur la carte."""
        if self._freed or not self.cc.free_ollama:
            return
        self._freed = True
        free_ollama(self._cfg)

    # ------------------------------------------------------------ sentiments
    def feel(self, coefs: dict[str, float]) -> bool:
        """Pose les sentiments (axe → coefficient). Renvoie True si l'injection est active."""
        with self._lock:
            coefs = {k: float(v) for k, v in coefs.items() if v}
            key = (tuple(sorted(coefs.items())), self.cc.alpha, self.cc.inject,
                   tuple(self.cc.layers))
            if key == self._applied:
                return self.engine.steered
            self._applied = key
            self.feeling = coefs
            if self._fallback is not None:
                return False
            if not self.cc.inject:
                with contextlib.suppress(LLMError):
                    if self.engine.steered:
                        self.engine.steer(None)
                return False
            try:
                self._free_ollama()
                self._load_vectors()
                if self.vectors is None:
                    return False
                first, last = self.vectors.best_layers(*self.cc.layers)
                buf = self.vectors.injection(coefs, self.cc.alpha, first, last)
                self.engine.steer(buf, first, last)
                return buf is not None
            except LLMError as exc:
                self.problem = str(exc)
                return False

    # ------------------------------------------------------------ dialogue
    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        params: GenParams | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResult:
        if any(m.get("images") for m in messages):
            raise LLMError("le cortex ne voit pas encore les images")
        with self._lock:
            if self._fallback is None:
                try:
                    self._free_ollama()
                    self.engine.load()
                    if not self.engine.template:
                        raise LLMError("le modèle du cortex n'a pas de gabarit de conversation")
                except LLMError as exc:     # pas de cortex : il pense quand même, via Ollama
                    self.problem = f"cortex indisponible, repli sur Ollama : {exc}"
                    from valdar.llm.ollama import OllamaBackend

                    self._fallback = OllamaBackend(self.cfg)
            if self._fallback is not None:
                kw = {"on_token": on_token} if on_token is not None else {}
                return self._fallback.chat(messages, system=system, tools=tools,
                                           params=params, **kw)
            prompt = gemma.render(self.engine.template, messages, system=system, tools=tools,
                                  bos=self.engine.bos)
            speech = gemma.SpeechFilter(on_token) if on_token is not None else None
            raw, usage = self.engine.generate(prompt, params or GenParams(),
                                              speech.feed if speech else None)
            if speech is not None:
                speech.close()
            content, calls = gemma.parse(raw)
            return ChatResult(
                content=content,
                tool_calls=[ToolCall(name=n, arguments=a, call_id=f"call_{i}")
                            for i, (n, a) in enumerate(calls)],
                usage=usage)

    def close(self) -> None:
        self.engine.close()
