"""Registre d'outils et politique de permissions (avenant 2 §5).

Niveaux :
- safe       : tout le monde ;
- safety     : tout le monde, sans attendre (arrêter une machine ne doit jamais être bloqué) ;
- elevated   : propriétaire reconnu avec une confiance suffisante ;
- dangerous  : propriétaire reconnu + confirmation explicite.
Un inconnu n'a jamais accès à plus que safe et safety.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from valdar.config.loader import Identity, PermissionsConfig

SAFE, SAFETY, ELEVATED, DANGEROUS = "safe", "safety", "elevated", "dangerous"
TIERS = (SAFE, SAFETY, ELEVATED, DANGEROUS)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    tier: str = SAFE
    family: str = "divers"

    def schema(self) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def params(required: list[str] | None = None, **props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        if tool.tier not in TIERS:
            raise ValueError(f"niveau inconnu pour {tool.name} : {tool.tier}")
        if tool.name in self.tools:
            raise ValueError(f"outil en double : {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def schemas(self, allowed: Callable[[Tool], bool] | None = None) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values() if allowed is None or allowed(t)]


@dataclass
class Decision:
    allowed: bool
    needs_confirmation: bool = False
    reason: str = ""


def decide(tool: Tool, who: Identity, cfg: PermissionsConfig) -> Decision:
    """Qui a le droit d'utiliser cet outil, maintenant ?"""
    if tool.tier in (SAFE, SAFETY):
        return Decision(True)
    is_owner = who.role in cfg.owner_roles and who.person is not None and not who.minor
    if not is_owner:
        return Decision(False, reason=f"« {tool.name} » est réservé au propriétaire.")
    if tool.tier == ELEVATED:
        if who.confidence >= cfg.elevated_min_confidence:
            return Decision(True)
        return Decision(False, reason="je ne suis pas assez sûr de qui me parle pour faire ça.")
    if who.confidence >= cfg.dangerous_min_confidence:
        return Decision(True, needs_confirmation=True)
    return Decision(False, reason="je ne suis pas assez sûr de qui me parle pour faire ça.")
