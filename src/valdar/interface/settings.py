"""Réglages modifiables depuis l'interface (onglet Réglages).

Chaque réglage : un chemin dans la configuration, un libellé, un type, des bornes, et s'il
faut redémarrer. Un changement s'applique tout de suite quand c'est possible et est gardé
dans `data/reglages.json` (par-dessus `config/valdar.yaml`, qui ne bouge pas).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from valdar.config.loader import ValdarConfig, overrides_path

# (chemin, libellé, type, options, redémarrage, groupe, aide)
SPEC: list[dict[str, Any]] = [
    # --- cerveau
    {"path": "llm.model", "label": "Modèle", "type": "model", "group": "Cerveau",
     "help": "Le modèle d'Ollama qui pense (liste des modèles installés)."},
    {"path": "llm.num_ctx", "label": "Mémoire de travail (jetons)", "type": "int",
     "min": 2048, "max": 32768, "step": 1024, "group": "Cerveau",
     "help": "Plus grand = il tient plus de conversation, mais plus de mémoire de la carte."},
    {"path": "llm.think", "label": "Mode réflexion", "type": "bool", "group": "Cerveau",
     "help": "Gemma réfléchit avant de répondre : plus juste, beaucoup plus lent."},
    {"path": "llm.keep_alive", "label": "Garder le modèle chargé", "type": "text",
     "group": "Cerveau", "help": "ex. 30m, 2h, -1 (toujours)."},
    {"path": "llm.history_messages", "label": "Messages gardés en tête", "type": "int",
     "min": 4, "max": 60, "step": 2, "group": "Cerveau"},
    {"path": "expression.temperature.base", "label": "Créativité (température)",
     "type": "float", "min": 0.1, "max": 1.3, "step": 0.05, "group": "Cerveau",
     "help": "Base ; son humeur la fait varier autour."},
    {"path": "expression.max_tokens.base", "label": "Longueur des réponses", "type": "int",
     "min": 60, "max": 1200, "step": 20, "group": "Cerveau"},
    {"path": "expression.top_p", "label": "Top-p", "type": "float", "min": 0.3, "max": 1.0,
     "step": 0.05, "group": "Cerveau"},
    {"path": "expression.repeat_penalty", "label": "Pénalité de répétition", "type": "float",
     "min": 1.0, "max": 1.5, "step": 0.01, "group": "Cerveau"},
    # --- voix
    {"path": "voice.character.high_shelf_db", "label": "Aigus (dB)", "type": "float",
     "min": -20, "max": 6, "step": 1, "group": "Voix",
     "help": "Monte-les si la voix est trop grave ou sourde."},
    {"path": "voice.character.drive", "label": "Grain (saturation)", "type": "float",
     "min": 1.0, "max": 3.0, "step": 0.1, "group": "Voix"},
    {"path": "voice.character.pitch_shift", "label": "Filtre radio", "type": "float",
     "min": -6, "max": 0, "step": 0.5, "group": "Voix",
     "help": "0 = son plein ; -2 = timbre radio d'origine."},
    {"path": "voice.character.peak", "label": "Volume", "type": "float", "min": 0.2,
     "max": 1.0, "step": 0.05, "group": "Voix"},
    {"path": "voice.enabled", "label": "Voix activée", "type": "bool", "group": "Voix",
     "restart": True},
    # --- oreilles
    {"path": "ears.names", "label": "Son nom (et déformations)", "type": "list",
     "group": "Oreilles", "restart": True},
    {"path": "ears.vad_threshold", "label": "Sensibilité à la parole", "type": "float",
     "min": 0.2, "max": 0.9, "step": 0.05, "group": "Oreilles",
     "help": "Plus bas = il détecte des voix plus faibles (et plus de bruit)."},
    {"path": "ears.end_silence_ms", "label": "Silence de fin de phrase (ms)", "type": "int",
     "min": 300, "max": 2000, "step": 50, "group": "Oreilles"},
    {"path": "ears.engaged_seconds", "label": "Conversation ouverte (s)", "type": "float",
     "min": 0, "max": 60, "step": 1, "group": "Oreilles",
     "help": "Après sa réponse, on peut lui parler sans redire son nom pendant ce temps."},
    {"path": "ears.barge_in_margin_db", "label": "Pour le couper (dB au-dessus de sa voix)",
     "type": "float", "min": 3, "max": 25, "step": 1, "group": "Oreilles"},
    {"path": "ears.stt_model", "label": "Modèle de transcription", "type": "choice",
     "options": ["base", "small", "medium", "large-v3-turbo"], "group": "Oreilles",
     "restart": True, "help": "small = rapide ; large-v3-turbo = plus juste, plus lent."},
    {"path": "ambient.startle_floor_db", "label": "Seuil du sursaut (dB)", "type": "float",
     "min": -30, "max": -3, "step": 1, "group": "Oreilles"},
    # --- reconnaissance
    {"path": "ident.voice_threshold", "label": "Voix : ressemblance minimale",
     "type": "float", "min": 0.2, "max": 0.9, "step": 0.01, "group": "Reconnaissance"},
    {"path": "ident.voice_margin", "label": "Voix : écart avec le 2e", "type": "float",
     "min": 0.0, "max": 0.3, "step": 0.01, "group": "Reconnaissance",
     "help": "Monte-le si deux enfants sont confondus."},
    {"path": "ident.face_threshold", "label": "Visage : ressemblance minimale",
     "type": "float", "min": 0.2, "max": 0.9, "step": 0.01, "group": "Reconnaissance"},
    {"path": "vision.enabled", "label": "Caméra activée", "type": "bool",
     "group": "Reconnaissance", "restart": True},
    {"path": "vision.camera", "label": "Caméra (numéro ou adresse)", "type": "text",
     "group": "Reconnaissance", "restart": True},
    {"path": "vision.table_rule", "label": "Gronder le chien sur la table", "type": "bool",
     "group": "Reconnaissance"},
    # --- vie intérieure
    {"path": "thoughts.enabled", "label": "Pensée de fond", "type": "bool",
     "group": "Vie intérieure"},
    {"path": "thoughts.min_interval", "label": "Entre deux pensées (s)", "type": "float",
     "min": 60, "max": 7200, "step": 60, "group": "Vie intérieure"},
    {"path": "night.enabled", "label": "Travail de nuit", "type": "bool",
     "group": "Vie intérieure"},
    {"path": "initiative.quiet_start", "label": "Heures calmes : début", "type": "text",
     "group": "Vie intérieure"},
    {"path": "initiative.quiet_end", "label": "Heures calmes : fin", "type": "text",
     "group": "Vie intérieure"},
    {"path": "initiative.min_interval_seconds", "label": "Entre deux prises de parole (s)",
     "type": "float", "min": 300, "max": 14400, "step": 300, "group": "Vie intérieure"},
    # --- interface
    {"path": "interface.screen", "label": "Écran de l'interface", "type": "int", "min": 1,
     "max": 4, "step": 1, "group": "Interface", "restart": True},
    {"path": "interface.kiosk", "label": "Plein écran (kiosque)", "type": "bool",
     "group": "Interface", "restart": True},
]
_BY_PATH = {s["path"]: s for s in SPEC}


def _get(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part) if not isinstance(obj, dict) else obj[part]
    return obj


def _set(obj: Any, path: str, value: Any) -> None:
    *head, last = path.split(".")
    for part in head:
        obj = getattr(obj, part)
    if isinstance(obj, dict):
        obj[last] = value
    else:
        object.__setattr__(obj, last, value)


def coerce(spec: dict[str, Any], value: Any) -> Any:
    t = spec["type"]
    if t == "bool":
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "oui")
    if t == "int":
        v = int(float(value))
    elif t == "float":
        v = float(value)
    elif t == "list":
        items = value if isinstance(value, list) else str(value).split(",")
        return [str(x).strip() for x in items if str(x).strip()]
    elif t == "choice":
        if str(value) not in spec["options"]:
            raise ValueError(f"choix impossible : {value}")
        return str(value)
    elif spec["path"] == "vision.camera":
        s = str(value).strip()
        return int(s) if s.isdigit() else s
    else:
        return str(value).strip()
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ValueError(f"{spec['label']} : entre {lo} et {hi}")
    return v


class Settings:
    def __init__(self, cfg: ValdarConfig, path: Path | None = None):
        self.cfg = cfg
        self.path = path or overrides_path(cfg.root)
        self._lock = threading.Lock()

    def _saved(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def listing(self) -> list[dict[str, Any]]:
        out = []
        for s in SPEC:
            try:
                v = _get(self.cfg, s["path"])
            except (AttributeError, KeyError):
                continue
            out.append({**s, "value": v})
        return out

    def set(self, path: str, value: Any) -> dict[str, Any]:
        spec = _BY_PATH.get(path)
        if spec is None:
            raise KeyError(f"réglage inconnu : {path}")
        v = coerce(spec, value)
        with self._lock:
            _set(self.cfg, path, v)
            saved = self._saved()
            node = saved
            *head, last = path.split(".")
            for part in head:
                node = node.setdefault(part, {})
            node[last] = v
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(saved, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        return {"path": path, "value": v, "restart": bool(spec.get("restart"))}

    def reset(self) -> None:
        with self._lock:
            if self.path.is_file():
                self.path.unlink()
