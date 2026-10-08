"""Le foyer : les gens qui vivent avec Valdar, et les animaux.

Rangé dans le dossier de données (`data/foyer.json`), jamais dans le dépôt git : ce sont des
prénoms, des liens de famille, des enfants. Olivier les ajoute depuis l'interface (onglet
Personnes). Chaque membre a un identifiant court (`oceane`), un prénom, son lien avec
Olivier, et `mineur` (les outils élevés restent réservés au propriétaire).
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

from valdar.config.loader import Identity


def slug(name: str) -> str:
    s = unicodedata.normalize("NFD", name.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "inconnu"


class Foyer:
    def __init__(self, path: str | Path, owner: Identity | None = None):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.owner = owner
        self.data: dict[str, Any] = {"membres": [], "animaux": []}
        if self.path.is_file():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    self.data.update(d)
            except ValueError:
                pass
        if owner is not None and owner.person and self.member(owner.person) is None:
            self.data["membres"].insert(0, {"id": owner.person, "nom": owner.name,
                                            "lien": "ton créateur", "role": "owner",
                                            "mineur": False})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------ membres
    @property
    def members(self) -> list[dict[str, Any]]:
        return list(self.data["membres"])

    @property
    def animals(self) -> list[dict[str, Any]]:
        return list(self.data["animaux"])

    def member(self, pid: str | None) -> dict[str, Any] | None:
        return next((m for m in self.data["membres"] if m["id"] == pid), None)

    def add_member(self, nom: str, lien: str = "", mineur: bool = False) -> dict[str, Any]:
        nom = nom.strip()
        if not nom:
            raise ValueError("prénom vide")
        pid = slug(nom)
        with self._lock:
            m = self.member(pid)
            if m is None:
                m = {"id": pid, "nom": nom, "lien": lien.strip(), "role": "family",
                     "mineur": bool(mineur)}
                self.data["membres"].append(m)
            else:
                m.update({"nom": nom, "lien": lien.strip() or m.get("lien", ""),
                          "mineur": bool(mineur)})
            self._save()
        return m

    def remove_member(self, pid: str) -> bool:
        with self._lock:
            before = len(self.data["membres"])
            self.data["membres"] = [m for m in self.data["membres"]
                                    if m["id"] != pid or m.get("role") == "owner"]
            changed = len(self.data["membres"]) != before
            if changed:
                self._save()
        return changed

    def add_animal(self, nom: str, espece: str = "chien") -> dict[str, Any]:
        nom = nom.strip()
        if not nom:
            raise ValueError("nom vide")
        with self._lock:
            a = next((x for x in self.data["animaux"] if slug(x["nom"]) == slug(nom)), None)
            if a is None:
                a = {"nom": nom, "espece": espece}
                self.data["animaux"].append(a)
            else:
                a["espece"] = espece
            self._save()
        return a

    def remove_animal(self, nom: str) -> bool:
        with self._lock:
            before = len(self.data["animaux"])
            self.data["animaux"] = [a for a in self.data["animaux"]
                                    if slug(a["nom"]) != slug(nom)]
            changed = len(self.data["animaux"]) != before
            if changed:
                self._save()
        return changed

    def dog(self) -> dict[str, Any] | None:
        return next((a for a in self.data["animaux"] if a.get("espece") == "chien"), None)

    # ------------------------------------------------------------ identité
    def identity(self, pid: str | None, confidence: float) -> Identity:
        """L'identité d'un membre reconnu (par la voix ou le visage)."""
        m = self.member(pid)
        if m is None:
            return Identity(person=None, name="inconnu", role="unknown", confidence=0.0)
        role = "owner" if m.get("role") == "owner" else "family"
        return Identity(person=m["id"], name=m["nom"], role=role,
                        confidence=max(0.0, min(1.0, confidence)),
                        minor=bool(m.get("mineur")))

    def block(self) -> str:
        """Le foyer, pour le prompt : qui vit là (sans rien de sensible)."""
        if not self.data["membres"] and not self.data["animaux"]:
            return ""
        lines = ["TON FOYER (les gens et les bêtes avec qui tu vis) :"]
        for m in self.data["membres"]:
            if m.get("role") == "owner":
                lien = " — ton créateur"
            else:
                who = self.owner.name if self.owner else "ton créateur"
                lien = f" — pour {who} : « {m['lien']} »" if m.get("lien") else ""
            lines.append(f"- {m['nom']}{lien}{' (enfant)' if m.get('mineur') else ''}")
        for a in self.data["animaux"]:
            lines.append(f"- {a['nom']}, le {a.get('espece', 'animal')}")
        return "\n".join(lines)
