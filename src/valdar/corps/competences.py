"""Les compétences : des dossiers d'instructions, lus seulement quand ils servent.

data/competences/<nom>/SKILL.md, avec en tête :
    ---
    name: imprimer-une-piece
    description: Quand Olivier veut lancer une impression : vérifier, trancher, envoyer.
    ---
    (les étapes)

Seuls le nom et la description sont dans sa tête en permanence ; le reste, il le lit quand
la tâche le demande. Il peut en écrire de nouvelles (c'est une zone où il écrit librement).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def _front(text: str) -> tuple[dict[str, str], str]:
    m = FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip().strip('"').strip("'")
    return meta, text[m.end():]


class Competences:
    def __init__(self, folder: Path):
        self.folder = folder

    def all(self) -> list[Skill]:
        out = []
        if not self.folder.is_dir():
            return out
        for f in sorted(self.folder.glob("*/SKILL.md")):
            try:
                meta, _ = _front(f.read_text("utf-8"))
            except OSError:
                continue
            out.append(Skill(meta.get("name") or f.parent.name,
                             meta.get("description", "")[:300], f))
        return out

    def summary(self) -> str:
        skills = self.all()
        if not skills:
            return "aucune pour l'instant"
        return "; ".join(f"{s.name} ({s.description})" for s in skills)

    def lire(self, nom: str) -> str:
        key = (nom or "").strip().lower()
        for s in self.all():
            if key in (s.name.lower(), s.path.parent.name.lower()):
                _, body = _front(s.path.read_text("utf-8"))
                extra = [p.name for p in s.path.parent.iterdir() if p.name != "SKILL.md"]
                more = f"\n\n(fichiers joints dans {s.path.parent} : {', '.join(extra)})" \
                    if extra else ""
                return f"Compétence « {s.name} » :\n{body.strip()}{more}"
        return f"pas de compétence « {nom} » ; j'ai : {self.summary()}"
