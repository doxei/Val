"""Le format de conversation de Gemma 4 : mise en forme du prompt et lecture des appels d'outils.

Le gabarit (Jinja) vient du fichier du modèle lui-même (`tokenizer.chat_template`) : on ne
recopie pas le format, on applique celui que Google a livré. Deux choses restent à faire ici :

- ranger notre historique (messages `tool` séparés, façon Ollama) comme le gabarit l'attend
  (appels, réponses d'outils et texte final dans le même tour du modèle) ;
- lire ce que le modèle écrit quand il appelle un outil :
  `<|tool_call>call:nom{cle:<|"|>texte<|"|>,n:3}<tool_call|>`.
"""
from __future__ import annotations

import re
from typing import Any

TOOL_OPEN = "<|tool_call>"
TOOL_CLOSE = "<tool_call|>"
QUOTE = '<|"|>'
TURN_END = "<turn|>"
STOPS = (TURN_END, "<|tool_response>", "<|turn>")
_THOUGHT = re.compile(r"<\|channel>.*?(<channel\|>|$)", re.S)
# Marques de contrôle sans place dans un texte (images, son, vidéo, remplissage…).
STRAY = re.compile(r"<\|?(?:image|audio|video|think|tool|pad|mask|unk|bos|eos)\|?>")


def strays(text: str) -> int:
    """Nombre de marques parasites dans un texte produit par le modèle."""
    return len(STRAY.findall(text))


def to_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Historique de Valdar → messages pour le gabarit de Gemma 4."""
    out: list[dict[str, Any]] = []
    waiting = False        # le dernier tour du modèle attend encore ses réponses d'outils
    for m in messages:
        role = m.get("role")
        if role == "tool":
            if waiting:
                out[-1].setdefault("tool_responses", []).append(
                    {"name": m.get("tool_name") or "outil", "response": m.get("content", "")})
            else:       # réponse d'outil orpheline : on la donne comme une information
                out.append({"role": "user",
                            "content": f"[résultat de {m.get('tool_name') or 'outil'}] "
                                       f"{m.get('content', '')}"})
            continue
        if (role == "assistant" and not m.get("tool_calls") and waiting
                and out[-1].get("tool_responses")):
            out[-1]["content"] = m.get("content", "")     # la suite du même tour
            waiting = False
            continue
        msg: dict[str, Any] = {"role": role, "content": m.get("content", "") or ""}
        waiting = False
        if m.get("tool_calls"):
            msg["tool_calls"] = [{"function": {
                "name": (c.get("function") or {}).get("name", ""),
                "arguments": (c.get("function") or {}).get("arguments") or {}}}
                for c in m["tool_calls"]]
            waiting = role == "assistant"
        out.append(msg)
    return out


def render(template: str, messages: list[dict[str, Any]], system: str = "",
           tools: list[dict[str, Any]] | None = None, bos: str = "<bos>") -> str:
    """Le prompt complet, prêt à être découpé en jetons (réflexion désactivée)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    msgs = ([{"role": "system", "content": system}] if system else []) + to_template(messages)
    return env.from_string(template).render(
        messages=msgs, tools=tools or None, add_generation_prompt=True,
        enable_thinking=False, bos_token=bos)


class _Reader:
    def __init__(self, s: str):
        self.s, self.i = s, 0

    def peek(self, n: int = 1) -> str:
        return self.s[self.i:self.i + n]

    def take(self, lit: str) -> bool:
        if self.s.startswith(lit, self.i):
            self.i += len(lit)
            return True
        return False

    def value(self) -> Any:
        if self.take(QUOTE):
            j = self.s.find(QUOTE, self.i)
            j = len(self.s) if j < 0 else j
            v = self.s[self.i:j]
            self.i = min(len(self.s), j + len(QUOTE))
            return v
        if self.take("{"):
            return self.mapping()
        if self.take("["):
            items = []
            while self.i < len(self.s) and not self.take("]"):
                items.append(self.value())
                self.take(",")
            return items
        j = self.i
        while j < len(self.s) and self.s[j] not in ",}]":
            j += 1
        raw = self.s[self.i:j].strip()
        self.i = j
        return _scalar(raw)

    def mapping(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while self.i < len(self.s) and not self.take("}"):
            if self.take(QUOTE):
                j = self.s.find(QUOTE, self.i)
                key = self.s[self.i:j]
                self.i = j + len(QUOTE)
            else:
                j = self.s.find(":", self.i)
                if j < 0:
                    break
                key = self.s[self.i:j].strip()
                self.i = j
            if not self.take(":"):
                break
            out[key] = self.value()
            self.take(",")
        return out


def _scalar(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw.strip("\"'")


def last_sentence(text: str) -> str:
    """Réponse coupée par la limite de longueur : on s'arrête à la dernière phrase finie."""
    cut = max(text.rfind(c) for c in ".!?…")
    return text[:cut + 1] if cut >= len(text) // 3 else text


def parse(text: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Texte écrit par le modèle → (ce qu'il dit, [(outil, arguments)])."""
    calls: list[tuple[str, dict[str, Any]]] = []
    said: list[str] = []
    pos = 0
    while True:
        i = text.find(TOOL_OPEN, pos)
        if i < 0:
            said.append(text[pos:])
            break
        said.append(text[pos:i])
        j = text.find(TOOL_CLOSE, i)
        body = text[i + len(TOOL_OPEN):j if j >= 0 else len(text)]
        pos = len(text) if j < 0 else j + len(TOOL_CLOSE)
        if body.startswith("call:"):
            body = body[5:]
        k = body.find("{")
        name = (body if k < 0 else body[:k]).strip()
        args = _Reader(body[k + 1:]).mapping() if k >= 0 else {}
        if name:
            calls.append((name, args))
    content = STRAY.sub("", _THOUGHT.sub("", "".join(said)))
    for stop in STOPS:
        content = content.replace(stop, "")
    return content.strip(), calls


class SpeechFilter:
    """Laisse passer vers la voix ce que le modèle *dit*, au fil de l'eau, et retient
    l'appel d'outil (jamais prononcé)."""

    def __init__(self, emit: Any):
        self.emit = emit
        self.buf = ""
        self.muted = False

    def feed(self, piece: str) -> None:
        if self.muted:
            return
        self.buf += piece
        i = self.buf.find(TOOL_OPEN)
        if i >= 0:
            self._out(self.buf[:i])
            self.buf, self.muted = "", True
            return
        keep = 0                      # un début de marque en fin de morceau : on attend la suite
        for mark in (TOOL_OPEN, *STOPS):
            for n in range(min(len(mark) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(mark[:n]):
                    keep = max(keep, n)
                    break
        cut = len(self.buf) - keep
        self._out(self.buf[:cut])
        self.buf = self.buf[cut:]

    def _out(self, text: str) -> None:
        for stop in STOPS:
            text = text.replace(stop, "")
        text = STRAY.sub("", text)
        if text:
            self.emit(text)

    def close(self) -> None:
        if not self.muted:
            self._out(self.buf)
        self.buf = ""
