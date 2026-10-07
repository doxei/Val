"""Backend Ollama (API native /api/chat) : outils, mode réflexion désactivable, keep_alive."""
from __future__ import annotations

import json
from typing import Any

import httpx

from valdar.config.loader import LLMConfig
from valdar.llm.backend import ChatResult, GenParams, LLMError, ToolCall


class OllamaBackend:
    def __init__(self, cfg: LLMConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.client = httpx.Client(
            base_url=cfg.url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_seconds, connect=5.0),
            transport=transport,
        )

    def available(self) -> bool:
        try:
            r = self.client.get("/api/version", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def has_model(self) -> bool:
        try:
            r = self.client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m.get("name", "") for m in r.json().get("models", [])}
        want = self.cfg.model
        return want in names or f"{want}:latest" in names

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        params: GenParams | None = None,
    ) -> ChatResult:
        p = params or GenParams()
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": msgs,
            "stream": False,
            "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": p.temperature,
                "top_p": p.top_p,
                "repeat_penalty": p.repeat_penalty,
                "num_predict": p.max_tokens,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if tools:
            body["tools"] = tools
        try:
            r = self.client.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama injoignable ({self.cfg.url}) : {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"Ollama {r.status_code} : {r.text[:300]}")
        data = r.json()
        msg = data.get("message") or {}
        calls: list[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {}
            calls.append(ToolCall(name=name, arguments=args, call_id=tc.get("id") or f"call_{i}"))
        usage = {k: data.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration")
                 if k in data}
        return ChatResult(content=(msg.get("content") or "").strip(), tool_calls=calls,
                          usage=usage)

    def close(self) -> None:
        self.client.close()
