"""Modèles de langage locaux derrière une interface commune."""
from valdar.llm.backend import ChatResult, GenParams, LLMBackend, LLMError, ToolCall

__all__ = ["ChatResult", "GenParams", "LLMBackend", "LLMError", "ToolCall", "make_backend"]


def make_backend(cfg):
    """Construit le backend décrit par la configuration."""
    if cfg.llm.backend == "ollama":
        from valdar.llm.ollama import OllamaBackend

        return OllamaBackend(cfg.llm)
    raise ValueError(f"backend LLM inconnu : {cfg.llm.backend}")
