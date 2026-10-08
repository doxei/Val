"""Téléchargement des modèles d'identification au premier usage (dans data/models)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def fetch(dest: Path, urls: list[str], say: Callable[[str], None] = print,
          min_bytes: int = 100_000) -> Path:
    """Télécharge le premier lien qui répond. Ne retélécharge pas un fichier présent."""
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return dest
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        say(f"(je télécharge {dest.name}…)")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        fh.write(chunk)
            if tmp.stat().st_size < min_bytes:
                tmp.unlink(missing_ok=True)
                errors.append(f"{url} : fichier trop petit")
                continue
            tmp.replace(dest)
            return dest
        except Exception as exc:        # réseau, 404… on essaie le lien suivant
            errors.append(f"{url} : {exc}")
    raise RuntimeError("téléchargement impossible : " + " ; ".join(errors))
