"""Interface commune des modèles de langage (Ollama aujourd'hui, llama-server demain)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Le modèle de langage est injoignable ou a renvoyé une erreur."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenParams:
    temperature: float = 0.6
    max_tokens: int = 220
    top_p: float = 0.9
    repeat_penalty: float = 1.1


class LLMBackend(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        params: GenParams | None = None,
    ) -> ChatResult: ...

    def available(self) -> bool: ...
