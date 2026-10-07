"""Faux modèle de langage scripté, pour les tests (aucun réseau, aucun GPU)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from valdar.llm.backend import ChatResult, GenParams


class FakeBackend:
    """Renvoie des réponses prévues à l'avance, ou calculées par une fonction.

    Chaque appel est enregistré dans `calls` (messages, système, outils, paramètres).
    """

    def __init__(self, script: list[ChatResult] | Callable[..., ChatResult] | None = None):
        self.script = script if script is not None else []
        self.calls: list[dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        params: GenParams | None = None,
    ) -> ChatResult:
        self.calls.append({"messages": [dict(m) for m in messages], "system": system,
                           "tools": tools, "params": params})
        if callable(self.script):
            return self.script(messages=messages, system=system, tools=tools, params=params)
        if not self.script:
            return ChatResult(content="d'accord.")
        return self.script.pop(0)
