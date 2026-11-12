"""Valdar qui parle le langage d'Ollama, pour prendre la place de Gemini sur le téléphone.

Android ne donne la place d'assistant par défaut qu'à une **appli installée**. Une page web
ne peut pas l'avoir. Mais l'appli Home Assistant, elle, peut : et Home Assistant sait
appeler un agent de conversation extérieur — à condition qu'il parle comme Ollama.

Alors Valdar parle comme Ollama. Deux routes suffisent :
- `GET /api/tags` : « voilà les modèles que j'ai » → un seul, lui ;
- `POST /api/chat` : un message, une réponse (avec ou sans flux).

Ce n'est pas un vrai Ollama : pas de `/api/generate`, pas d'embeddings, pas de modèles à
télécharger. C'est **Valdar entier** derrière — son cœur, sa mémoire, ses outils — et non
le modèle brut. Un tour qui arrive par là compte comme un vrai tour de conversation.

La route est protégée par le même jeton que le téléphone : Home Assistant l'envoie dans
l'adresse (`?j=…`), puisque son intégration Ollama ne sait pas poser d'en-tête.
"""
from __future__ import annotations

import json
import time
from typing import Any

MODELE = "valdar"


def tags() -> dict[str, Any]:
    """La liste des modèles, à la façon d'Ollama. Il n'y en a qu'un : lui."""
    maintenant = time.strftime("%Y-%m-%dT%H:%M:%S%z") or "2026-01-01T00:00:00Z"
    return {"models": [{
        "name": f"{MODELE}:latest", "model": f"{MODELE}:latest",
        "modified_at": maintenant, "size": 0,
        "digest": "valdar", "details": {"parent_model": "", "format": "gguf",
                                        "family": "valdar", "families": ["valdar"],
                                        "parameter_size": "12B",
                                        "quantization_level": "Q4_0"}}]}


def show() -> dict[str, Any]:
    return {"license": "privé", "modelfile": f"FROM {MODELE}", "parameters": "",
            "template": "{{ .Prompt }}",
            "details": {"family": "valdar", "parameter_size": "12B"},
            "model_info": {"general.architecture": "valdar"},
            "capabilities": ["completion"]}


def last_user_message(body: dict[str, Any]) -> str:
    """Ce qu'Olivier vient de dire. L'historique vient de Valdar lui-même, pas du client :
    c'est lui qui se souvient, Home Assistant n'a pas à lui réapprendre sa propre vie."""
    for m in reversed(body.get("messages") or []):
        if (m.get("role") or "") == "user":
            return str(m.get("content") or "").strip()
    return str(body.get("prompt") or "").strip()


def reply_body(text: str, done: bool = True) -> dict[str, Any]:
    return {"model": f"{MODELE}:latest",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": text},
            "done": done,
            **({"done_reason": "stop"} if done else {})}


def chat(api: Any, body: dict[str, Any]) -> tuple[str, bool]:
    """Un tour de conversation. Renvoie (réponse, en_flux_demande)."""
    texte = last_user_message(body)
    flux = bool(body.get("stream", True))
    if not texte:
        return "", flux
    who = api.access.identity() if api.access else api.s.owner
    reply = api.s.answer(texte, who=who, source="assistant du téléphone")
    return (getattr(reply, "text", "") or "").strip(), flux


def stream_lines(text: str) -> list[bytes]:
    """La réponse découpée comme Ollama la rend : une ligne JSON par morceau."""
    morceaux = [m for m in text.split(" ") if m != ""]
    out = []
    for i, m in enumerate(morceaux):
        bout = m if i == 0 else " " + m
        out.append(json.dumps(reply_body(bout, done=False), ensure_ascii=False).encode()
                   + b"\n")
    out.append(json.dumps({**reply_body("", done=True), "total_duration": 0,
                           "eval_count": len(morceaux)}, ensure_ascii=False).encode() + b"\n")
    return out
