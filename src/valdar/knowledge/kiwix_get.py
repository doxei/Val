"""Installer Kiwix et ses bibliothèques (`valdar kiwix`), en demandant avant chaque gros
téléchargement. Reprend là où il s'était arrêté si la connexion coupe."""
from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

from valdar.knowledge.kiwix import CATALOGUE, MIRROR, TOOLS, Book, latest


def _get(url: str) -> str:
    import httpx

    r = httpx.get(url, follow_redirects=True, timeout=30.0)
    r.raise_for_status()
    return r.text


def download(url: str, dest: Path, say: Callable[[str], None] = print) -> Path:
    """Téléchargement reprenable (en-tête Range), avec l'avancement tous les 5 %."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(60.0, read=120.0)) as r:
        if r.status_code == 416:            # déjà complet
            part.replace(dest)
            return dest
        r.raise_for_status()
        if have and r.status_code != 206:   # le serveur ne reprend pas : on recommence
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        step, nxt = max(1, total // 20), have + max(1, total // 20)
        with part.open("ab" if have else "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                have += len(chunk)
                if total and have >= nxt:
                    say(f"  {dest.name} : {100 * have // total} % "
                        f"({have / 1e9:.1f} / {total / 1e9:.1f} Go)")
                    nxt += step
    part.replace(dest)
    return dest


def install_tools(base: Path, say: Callable[[str], None] = print) -> Path:
    """kiwix-serve pour Windows (archive officielle kiwix-tools)."""
    bin_dir = base / "bin"
    found = sorted(bin_dir.glob("**/kiwix-serve*"))
    if found:
        return found[0]
    name = latest(_get(TOOLS), "kiwix-tools_win", ".zip")
    if name is None:
        raise RuntimeError("archive kiwix-tools pour Windows introuvable sur download.kiwix.org")
    say(f"Je télécharge {name}…")
    z = download(TOOLS + name, base / name, say)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(bin_dir)
    z.unlink(missing_ok=True)
    found = sorted(bin_dir.glob("**/kiwix-serve*"))
    if not found:
        raise RuntimeError("kiwix-serve absent de l'archive")
    return found[0]


def install_book(book: Book, base: Path, say: Callable[[str], None] = print) -> Path | None:
    listing = _get(MIRROR + book.folder + "/")
    name = latest(listing, book.prefix)
    if name is None:
        say(f"  {book.title} : introuvable sur le miroir, je passe.")
        return None
    dest = base / name
    if dest.exists():
        say(f"  {book.title} : déjà là ({name}).")
        return dest
    for old in base.glob(book.prefix + "*.zim"):      # ancienne version remplacée
        old.unlink(missing_ok=True)
    say(f"Je télécharge {book.title} ({name}, ~{book.size_gb:g} Go)…")
    return download(MIRROR + book.folder + "/" + name, dest, say)


def interactive(base: Path, ask: Callable[[str], str] = input,
                say: Callable[[str], None] = print) -> int:
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free / 1e9
    say(f"Bibliothèques hors ligne pour Valdar. Place libre sur ce disque : {free:.0f} Go.\n")
    for i, b in enumerate(CATALOGUE, 1):
        mark = "*" if b.default else " "
        say(f" {mark}{i:2d}. {b.title} — ~{b.size_gb:g} Go [{b.theme}]")
    essentials = [b for b in CATALOGUE if b.default]
    total = sum(b.size_gb for b in essentials)
    say(f"\n* = sélection conseillée (~{total:.0f} Go).")
    rep = ask("Entrée = la sélection conseillée ; sinon les numéros (ex. 1 3 7 16) ; "
              "q = annuler : ").strip().lower()
    if rep == "q":
        return 0
    chosen = essentials if not rep else [
        CATALOGUE[int(x) - 1] for x in rep.replace(",", " ").split()
        if x.isdigit() and 0 < int(x) <= len(CATALOGUE)]
    need = sum(b.size_gb for b in chosen)
    if need > free * 0.9:
        say(f"Attention : ~{need:.0f} Go demandés pour {free:.0f} Go libres.")
        if ask("Continuer quand même ? (o/n) ").strip().lower() != "o":
            return 1
    install_tools(base, say)
    ok = 0
    for b in chosen:
        try:
            if install_book(b, base, say) is not None:
                ok += 1
        except Exception as exc:
            say(f"  {b.title} : échec ({exc}). Relance plus tard, ça reprendra où ça en était.")
    say(f"\n{ok} bibliothèque(s) prête(s). Valdar les ouvrira à son prochain démarrage.")
    return 0
