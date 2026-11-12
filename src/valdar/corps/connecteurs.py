"""Les connecteurs : des serveurs MCP, dont les outils deviennent les siens.

Chaque connecteur tourne dans son coin (sous-processus ou adresse http) ; une boucle asyncio
dans un fil à part, parce que le reste de Valdar est synchrone. Si un connecteur ne répond
pas, on le note et on continue : il ne doit jamais empêcher Valdar de parler.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RemoteTool:
    connector: str
    name: str
    description: str
    parameters: dict[str, Any]


class Loop:
    """Une boucle asyncio qui vit dans un fil, pour appeler de l'async depuis du sync."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="mcp", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro: Any, timeout: float) -> Any:
        fut: Future[Any] = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)


@dataclass
class Connector:
    """Un serveur MCP, gardé ouvert tant que Valdar tourne."""
    name: str
    cfg: Any
    loop: Loop
    session: Any = None
    tools: list[RemoteTool] = field(default_factory=list)
    error: str = ""
    _stack: Any = None

    async def _open(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        if self.cfg.commande.startswith(("http://", "https://")):
            from mcp.client.streamable_http import streamable_http_client

            streams = await stack.enter_async_context(
                streamable_http_client(self.cfg.commande))
        else:
            params = StdioServerParameters(command=self.cfg.commande, args=list(self.cfg.args),
                                           env=dict(self.cfg.env) or None)
            streams = await stack.enter_async_context(stdio_client(params))
        read, write = streams[0], streams[1]
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        self.session, self._stack = session, stack
        self.tools = [RemoteTool(self.name, t.name, (t.description or "")[:300],
                                 t.inputSchema or {"type": "object", "properties": {}})
                      for t in listed.tools]

    def open(self, timeout: float = 30.0) -> str:
        try:
            self.loop.call(self._open(), timeout)
            return ""
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return self.error

    def call(self, tool: str, args: dict[str, Any], timeout: float = 60.0) -> str:
        if self.session is None:
            return f"le connecteur « {self.name} » n'est pas branché ({self.error})"
        try:
            res = self.loop.call(self.session.call_tool(tool, args or {}), timeout)
        except Exception as exc:
            return f"échec de {self.name}.{tool} : {type(exc).__name__}: {exc}"
        parts = []
        for block in getattr(res, "content", []) or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else f"[{getattr(block, 'type', 'contenu')}]")
        out = "\n".join(p for p in parts if p) or "(rien)"
        if getattr(res, "isError", False):
            return f"{self.name}.{tool} a répondu une erreur : {out}"
        return out

    def close(self) -> None:
        if self._stack is not None:
            with contextlib.suppress(Exception):
                self.loop.call(self._stack.aclose(), 5.0)
            self._stack = self.session = None


class Connectors:
    def __init__(self, cfgs: dict[str, Any]):
        self.cfgs = cfgs
        self.loop: Loop | None = None
        self.items: dict[str, Connector] = {}

    def start(self) -> list[str]:
        """Branche tout ce qui est configuré ; renvoie les ennuis, sans lever."""
        if not self.cfgs:
            return []
        try:
            import mcp  # noqa: F401
        except ImportError:
            return ["le module « mcp » n'est pas installé : aucun connecteur"]
        self.loop = Loop()
        notes = []
        for name, cfg in self.cfgs.items():
            c = Connector(name, cfg, self.loop)
            err = c.open()
            self.items[name] = c
            notes.append(f"{name} : {err}" if err else
                         f"{name} : {len(c.tools)} outil(s)")
        return notes

    def tools(self) -> list[RemoteTool]:
        return [t for c in self.items.values() for t in c.tools]

    def call(self, connector: str, tool: str, args: dict[str, Any]) -> str:
        c = self.items.get(connector)
        return c.call(tool, args) if c else f"connecteur inconnu : {connector}"

    def close(self) -> None:
        for c in self.items.values():
            c.close()
        if self.loop is not None:
            self.loop.close()
            self.loop = None
