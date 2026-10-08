"""Connecteurs par n8n : chaque workflow déclaré devient un outil de Valdar.

n8n reste en dehors de la boucle de conversation (il ajouterait des allers-retours) : Valdar
l'appelle comme n'importe quel outil, quand il en a besoin. Côté n8n, un workflow commence par
un nœud « Webhook » (POST, chemin = nom du workflow) et finit par « Respond to Webhook » ; ce
qu'il renvoie (texte ou JSON) revient à Valdar. Un workflow que tu crées dans n8n et que tu
déclares dans `config/valdar.yaml` (section `n8n.workflows`) = une capacité de plus, sans code.
"""
from __future__ import annotations

import contextlib
import json
import re
from typing import Any

import httpx

from valdar.config.loader import N8nConfig
from valdar.tools.registry import Registry, Tool, params

_NAME = re.compile(r"[^a-z0-9_]+")


def tool_name(workflow: str) -> str:
    return "n8n_" + _NAME.sub("_", workflow.lower()).strip("_")


def call(cfg: N8nConfig, workflow: str, data: dict[str, Any],
         client: httpx.Client | None = None) -> str:
    url = f"{cfg.url.rstrip('/')}/webhook/{workflow}"
    try:
        if client is not None:
            r = client.post(url, json=data, timeout=cfg.timeout_seconds)
        else:
            r = httpx.post(url, json=data, timeout=cfg.timeout_seconds)
    except httpx.HTTPError as exc:
        return f"n8n injoignable ({cfg.url}) : {exc}"
    if r.status_code >= 400:
        return f"le workflow « {workflow} » a échoué ({r.status_code}) : {r.text[:300]}"
    text = r.text.strip()
    with contextlib.suppress(ValueError):
        text = json.dumps(r.json(), ensure_ascii=False)
    text = text or "fait."
    return text[:cfg.max_chars]


def add_n8n_tools(registry: Registry, cfg: N8nConfig,
                  client: httpx.Client | None = None) -> list[str]:
    """Ajoute un outil par workflow déclaré. Renvoie les noms d'outils créés."""
    names = []
    for wf, spec in cfg.workflows.items():
        props = {k: {"type": "string", "description": d} for k, d in spec.params.items()}

        def run(_wf: str = wf, **kwargs: Any) -> str:
            return call(cfg, _wf, {k: v for k, v in kwargs.items() if v not in (None, "")},
                        client)

        name = tool_name(wf)
        registry.add(Tool(name, spec.description, params(list(spec.required), **props), run,
                          spec.tier, "automatisations", confirm=spec.confirm,
                          min_confidence=spec.min_confidence))
        names.append(name)
    return names
