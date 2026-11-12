"""Le harnais : exactement les outils et le prompt que Valdar a quand il parle pour de vrai.

Le banc doit mesurer ce qui tourne vraiment, pas une maquette : même liste d'outils, même
texte de personnage. Rien n'est exécuté ici — on ne construit que les descriptions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

FALLBACK = [{"type": "function", "function": {
    "name": "regler_plateau", "description": "Règle la température du plateau de l'imprimante.",
    "parameters": {"type": "object", "properties": {
        "temperature": {"type": "integer", "description": "température en degrés Celsius"}},
        "required": ["temperature"]}}}]


def registry(cfg: Any) -> Any:
    """Le registre d'outils du vrai Valdar, monté sans rien démarrer autour."""
    from valdar.atelier import Checklist, Pinouts, Reminders, Stock
    from valdar.memory import Facts
    from valdar.tools.standard import ToolContext, build_registry

    tmp = Path(tempfile.mkdtemp(prefix="valdar-harnais-"))
    # Les familles d'outils qui n'existent que si leur service est là : on met un témoin
    # muet, parce que seules les descriptions comptent ici (aucun outil n'est appelé).
    stub = object()
    ctx = ToolContext(heart=None, facts=Facts(tmp / "facts.db"), stock=Stock(tmp / "stock.db"),
                      checklist=Checklist(tmp / "check.db"), reminders=Reminders(tmp / "rem.db"),
                      pinouts=Pinouts(tmp / "pin.db"), printer=None, person="olivier",
                      knowledge=stub, thoughts=stub, memory=stub, relations=stub)
    reg = build_registry(ctx)
    if cfg.kiwix.enabled:       # les deux outils de savoir hors ligne, décrits comme en vrai
        from valdar.knowledge.kiwix import Kiwix
        from valdar.tools.registry import SAFE, Tool, params

        k = Kiwix(f"http://127.0.0.1:{cfg.kiwix.port}")
        s = {"type": "string"}
        reg.add(Tool("chercher_savoir", "Cherche dans l'encyclopédie hors ligne.",
                     params(["sujet"], sujet=s), lambda sujet: k.search(sujet), SAFE, "savoir"))
        reg.add(Tool("lire_article", "Lit un article de l'encyclopédie hors ligne.",
                     params(["titre"], titre=s), lambda titre: k.article(titre), SAFE, "savoir"))
    if cfg.corps.enabled:
        from valdar.corps.outils import Corps, add_corps_tools
        from valdar.corps.web import Web

        cc = cfg.corps.model_copy(deep=True)
        cc.atelier = str(tmp / "atelier")
        cc.connecteurs = {}      # on ne branche rien : seules les descriptions comptent
        corps = Corps(cc, cfg.root, tmp, web=Web() if cc.web else None)
        add_corps_tools(reg, corps)
    return reg


def tools(cfg: Any) -> list[dict[str, Any]]:
    """Les schémas d'outils tels que le modèle les lit. En cas de pépin : l'outil témoin."""
    try:
        return registry(cfg).schemas()
    except Exception:
        return list(FALLBACK)


def names(schemas: list[dict[str, Any]]) -> set[str]:
    return {s["function"]["name"] for s in schemas}
