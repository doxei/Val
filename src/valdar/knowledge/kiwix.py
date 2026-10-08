"""Kiwix : des bibliothèques entières hors ligne (Wikipédia, Stack Overflow, médecine…).

Kiwix range un site complet dans un seul fichier `.zim`. `kiwix-serve` (programme officiel de
Kiwix) les sert en local ; Valdar y cherche et y lit comme sur un site web, sans Internet.

- `Catalogue` : la liste des bibliothèques proposées (taille, contenu), téléchargées depuis
  download.kiwix.org en prenant la version la plus récente de chacune ;
- `KiwixServe` : lance `kiwix-serve` sur les fichiers présents ;
- `Kiwix` : recherche plein texte et lecture d'un article (texte nettoyé).
"""
from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

MIRROR = "https://download.kiwix.org/zim/"
TOOLS = "https://download.kiwix.org/release/kiwix-tools/"


@dataclass(frozen=True)
class Book:
    key: str            # nom court
    folder: str         # dossier sur download.kiwix.org/zim/
    prefix: str         # début du nom de fichier (la date suit)
    title: str
    size_gb: float      # ordre de grandeur, pour prévenir avant de télécharger
    theme: str
    default: bool = False


CATALOGUE: list[Book] = [
    Book("wikipedia", "wikipedia", "wikipedia_fr_all_nopic_", "Wikipédia en français (sans images)",
         16, "tout", True),
    Book("wikipedia-images", "wikipedia", "wikipedia_fr_all_maxi_",
         "Wikipédia en français (avec images)", 40, "tout"),
    Book("vikidia", "vikidia", "vikidia_fr_all_nopic_",
         "Vikidia : l'encyclopédie des 8-13 ans (pour les enfants)", 1, "enfants", True),
    Book("wiktionnaire", "wiktionary", "wiktionary_fr_all_nopic_", "Wiktionnaire (français)",
         2, "langue", True),
    Book("wikilivres", "wikibooks", "wikibooks_fr_all_nopic_",
         "Wikilivres : manuels (cuisine, bricolage, sciences…)", 0.5, "pratique", True),
    Book("wikiversite", "wikiversity", "wikiversity_fr_all_nopic_",
         "Wikiversité : cours (sciences, médecine…)", 0.5, "sciences", True),
    Book("medecine", "wikipedia", "wikipedia_en_medicine_maxi_",
         "WikiMed : la médecine de Wikipédia (anglais, très complet)", 2, "médecine", True),
    Book("ifixit", "ifixit", "ifixit_fr_all_", "iFixit : guides de réparation (français)", 3,
         "bricolage", True),
    Book("bricolage", "stack_exchange", "diy.stackexchange.com_en_all_",
         "Questions-réponses bricolage maison (anglais)", 2, "bricolage", True),
    Book("electronique", "stack_exchange", "electronics.stackexchange.com_en_all_",
         "Questions-réponses électronique (anglais)", 4, "électronique", True),
    Book("impression3d", "stack_exchange", "3dprinting.stackexchange.com_en_all_",
         "Questions-réponses impression 3D (anglais)", 0.3, "impression 3D", True),
    Book("jardinage", "stack_exchange", "gardening.stackexchange.com_en_all_",
         "Questions-réponses jardinage et plantes (anglais)", 1, "plantes", True),
    Book("parents", "stack_exchange", "parenting.stackexchange.com_en_all_",
         "Questions-réponses parents et enfants (anglais)", 0.5, "enfants", True),
    Book("sante", "stack_exchange", "medicalsciences.stackexchange.com_en_all_",
         "Questions-réponses sciences médicales (anglais)", 0.5, "médecine", True),
    Book("arduino", "stack_exchange", "arduino.stackexchange.com_en_all_",
         "Questions-réponses Arduino et ESP32 (anglais)", 0.5, "électronique", True),
    Book("stackoverflow", "stack_exchange", "stackoverflow.com_en_all_",
         "Stack Overflow : toute la programmation (anglais, énorme)", 75, "code"),
]


def latest(listing_html: str, prefix: str, ext: str = ".zim") -> str | None:
    """Le fichier le plus récent qui commence par `prefix` dans un index de dossier."""
    names = set(re.findall(r'href="(' + re.escape(prefix) + r'[^"/]*?' + re.escape(ext)
                           + r')"', listing_html))
    return max(names) if names else None       # les dates AAAA-MM se trient comme du texte


def clean(page: str, max_chars: int = 4000) -> str:
    """Texte lisible d'une page HTML de Kiwix (sans menus, scripts ni notes)."""
    page = re.sub(r"(?is)<(script|style|nav|header|footer|table)[^>]*>.*?</\1>", " ", page)
    page = re.sub(r"(?is)<sup[^>]*>.*?</sup>", "", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</li>|</div>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


class Kiwix:
    """Client de kiwix-serve (recherche et lecture)."""

    def __init__(self, url: str = "http://127.0.0.1:8888", client: Any = None):
        import httpx

        self.url = url.rstrip("/")
        self.http = client or httpx.Client(timeout=10.0)

    def available(self) -> bool:
        try:
            return self.http.get(self.url + "/catalog/v2/entries?count=1").status_code == 200
        except Exception:
            return False

    def books(self) -> list[dict[str, str]]:
        """Les bibliothèques servies : nom et titre."""
        r = self.http.get(self.url + "/catalog/v2/entries?count=500")
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ElementTree.fromstring(r.text)
        out = []
        for e in root.findall("a:entry", ns):
            name = e.findtext("a:name", default="", namespaces=ns)
            title = e.findtext("a:title", default="", namespaces=ns)
            lang = e.findtext("a:language", default="", namespaces=ns)
            if name:
                out.append({"name": name, "title": title, "lang": lang})
        return out

    def search(self, query: str, book: str | None = None, k: int = 5) -> list[dict[str, str]]:
        """Recherche plein texte. Renvoie titre, chemin, extrait, bibliothèque."""
        books = [book] if book else [b["name"] for b in self.books()]
        hits: list[dict[str, str]] = []
        for name in books:
            try:
                r = self.http.get(self.url + "/search", params={
                    "books.name": name, "pattern": query, "format": "xml",
                    "pageLength": str(k)})
            except Exception:
                continue
            if r.status_code != 200:
                continue
            try:
                root = ElementTree.fromstring(r.text)
            except ElementTree.ParseError:
                continue
            for item in root.iter("item"):
                hits.append({"book": name, "title": item.findtext("title", "").strip(),
                             "path": item.findtext("link", "").strip(),
                             "extrait": clean(item.findtext("description", ""), 300)})
                if len(hits) >= k * 3:
                    break
        return hits[: max(k, 1) * 2]

    def read(self, path: str, max_chars: int = 4000) -> str:
        url = path if path.startswith("http") else self.url + "/" + path.lstrip("/")
        r = self.http.get(url, follow_redirects=True)
        r.raise_for_status()
        return clean(r.text, max_chars)


class KiwixServe:
    """Lance kiwix-serve sur les fichiers .zim présents (en tâche de fond, sans fenêtre)."""

    def __init__(self, exe: Path, zim_dir: Path, port: int = 8888):
        self.exe = exe
        self.zim_dir = zim_dir
        self.port = port
        self.proc: subprocess.Popen | None = None

    def zims(self) -> list[Path]:
        return sorted(self.zim_dir.glob("*.zim")) if self.zim_dir.is_dir() else []

    def start(self) -> bool:
        files = self.zims()
        if not self.exe.is_file() or not files:
            return False
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([str(self.exe), "--port", str(self.port),
                                      "--address", "127.0.0.1", *map(str, files)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=flags)
        return True

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
