"""Tri visuel : quand la vigie s'inquiète, Gemma 4 regarde l'image (phase 5).

Sobre et sans personnalité (PrintOS laissait son caractère « Printos » répondre à la place de
l'analyse, et sa confiance ne valait rien). Le tri **ne décide rien** : il décrit ce qu'il voit,
propose le défaut le plus probable parmi ceux de la base de connaissances, et donne un conseil.
Il ne compte pas dans la porte des 98 % : seule la vigie, mesurée, compte.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from valdar.llm.backend import GenParams, LLMBackend, LLMError

PROMPT = """Tu analyses une image d'imprimante 3D FDM (CR-10S, Klipper) prise pendant une
impression, parce qu'un détecteur automatique a trouvé quelque chose de suspect.
Décris seulement ce qui est visible. Si l'image est trop sombre, floue ou ne montre pas la
pièce, dis-le. Réponds UNIQUEMENT en JSON :
{"lisible": true/false, "visible": "ce que tu vois, une phrase",
 "defaut": "un identifiant de la liste ci-dessous, ou aucun, ou autre",
 "certitude": "faible" | "moyenne" | "forte",
 "gravite": 1 à 5, "conseil": "une phrase : continuer, surveiller, mettre en pause, arrêter"}
Défauts possibles :
"""

_JSON = re.compile(r"\{.*\}", re.S)
_CERT = {"faible", "moyenne", "forte"}


@dataclass
class TriageResult:
    readable: bool
    seen: str
    defect: str
    certainty: str
    severity: int
    advice: str

    def text(self) -> str:
        if not self.readable:
            return f"je n'arrive pas à lire l'image ({self.seen})"
        what = "" if self.defect in ("aucun", "") else f" ; ça ressemble à « {self.defect} »"
        return (f"je vois : {self.seen}{what} (certitude {self.certainty}). "
                f"Mon conseil : {self.advice}")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Triage:
    def __init__(self, llm: LLMBackend, knowledge: Any = None, max_tokens: int = 300):
        self.llm = llm
        self.knowledge = knowledge
        self.max_tokens = max_tokens

    def _candidates(self) -> str:
        if self.knowledge is None:
            return "- decollement_spaghetti, sous_extrusion, decalage_couches, warping, autre"
        hits = []
        for q in ("spaghetti pièce décollée", "décalage de couches", "sous-extrusion buse",
                  "warping coins qui se soulèvent", "amas de plastique autour de la buse"):
            hits += self.knowledge.search(q, k=2)
        seen, lines = set(), []
        for h in hits:
            if h.title in seen:
                continue
            seen.add(h.title)
            lines.append(f"- {h.title}")
        return "\n".join(lines[:10]) or "- autre"

    def assess(self, jpeg: bytes, context: str = "") -> TriageResult | None:
        msg = {"role": "user", "content": (context or "Que vois-tu ?"),
               "images": [base64.b64encode(jpeg).decode("ascii")]}
        try:
            res = self.llm.chat([msg], system=PROMPT + self._candidates(), tools=None,
                                params=GenParams(temperature=0.1, max_tokens=self.max_tokens))
        except LLMError:
            return None
        return parse(res.content or "")


def parse(text: str) -> TriageResult | None:
    m = _JSON.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    cert = str(d.get("certitude") or "faible").lower()
    try:
        sev = max(1, min(5, int(d.get("gravite") or 1)))
    except (TypeError, ValueError):
        sev = 1
    return TriageResult(bool(d.get("lisible", True)), str(d.get("visible") or "")[:300],
                        str(d.get("defaut") or "aucun")[:80],
                        cert if cert in _CERT else "faible", sev,
                        str(d.get("conseil") or "")[:200])
