"""Parler au fil de l'eau : la réponse de Gemma arrive morceau par morceau (streaming) ; chaque
phrase complète part tout de suite vers la voix, sans attendre la fin de la réponse."""
from __future__ import annotations

import re
from collections.abc import Callable

_END = re.compile(r"(.+?[.!?…:;](?:[»\"')\]]*)(?:\s+|$)|.+?\n+)", re.S)


class SentenceStream:
    def __init__(self, say: Callable[[str], None], show: Callable[[str], None] | None = None,
                 min_chars: int = 12):
        self.say = say
        self.show = show
        self.min_chars = min_chars
        self.buf = ""
        self.text = ""

    def feed(self, piece: str) -> None:
        if self.show is not None:
            self.show(piece)
        self.text += piece
        self.buf += piece
        while True:
            m = _END.match(self.buf)
            if not m or not m.group(0).endswith((" ", "\n", "\t")):
                break               # la phrase n'est peut-être pas finie (« 3.5 », « M. »)
            sentence = m.group(1).strip()
            if len(sentence) < self.min_chars and len(self.buf) < 200:
                # trop court pour valoir une phrase seule (« Oui. ») : on attend la suite
                nxt = _END.match(self.buf, m.end())
                if not nxt:
                    break
                sentence = (sentence + " " + nxt.group(1).strip()).strip()
                self.buf = self.buf[nxt.end():]
            else:
                self.buf = self.buf[m.end():]
            if sentence:
                self.say(sentence)

    def close(self) -> None:
        rest = self.buf.strip()
        self.buf = ""
        if rest:
            self.say(rest)

    @property
    def started(self) -> bool:
        return bool(self.text.strip())
