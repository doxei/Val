"""Les artefacts : ce qu'il fabrique pour qu'on le garde ou qu'on le regarde.

Une page (HTML complet, ou un morceau qu'il habille), un document (Markdown mis en page) ou
un tableau (CSV, montré en tableau et gardé en .csv). Rangés dans son atelier
(`artefacts/`), ouverts dans le navigateur d'Olivier.
"""
from __future__ import annotations

import csv
import html
import io
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

STYLE = """
:root{--fond:#f7f5f0;--texte:#1d1d1b;--doux:#6b6b66;--trait:#dcd8cf;--accent:#2f5d8a}
@media (prefers-color-scheme:dark){:root{--fond:#1b1c1e;--texte:#e9e7e2;--doux:#a4a29c;
--trait:#3a3b3e;--accent:#8db7e3}}
body{background:var(--fond);color:var(--texte);font:16px/1.6 system-ui,Segoe UI,sans-serif;
max-width:860px;margin:2rem auto;padding:0 1rem}
h1,h2,h3{line-height:1.25}h1{font-size:1.8rem}a{color:var(--accent)}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border-bottom:1px solid var(--trait);padding:.45rem .6rem;text-align:left}
th{color:var(--doux);font-weight:600}tr:hover td{background:rgba(127,127,127,.08)}
code,pre{font-family:Consolas,monospace;background:rgba(127,127,127,.12);border-radius:4px}
pre{padding:.8rem;overflow:auto}code{padding:.1rem .3rem}
.pied{color:var(--doux);font-size:.85rem;margin-top:3rem}
"""


def slug(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:60] or "artefact"


def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\w)", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)
    return t


def markdown(md: str) -> str:
    """Le Markdown courant → HTML (titres, listes, gras, code, liens, tableaux simples)."""
    out: list[str] = []
    lines = md.replace("\r\n", "\n").split("\n")
    i, para, lst = 0, [], ""

    def flush() -> None:
        nonlocal para, lst
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para = []
        if lst:
            out.append(f"</{lst}>")
            lst = ""

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{2,}",
                                                                 lines[i + 1].strip()):
            flush()
            rows = [s]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j].strip())
                j += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0])
                       + "</tr>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r)
                                           + "</tr>" for r in cells[1:]) + "</table>")
            i = j
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
        elif re.match(r"^([-*+]|\d+[.)])\s+", s):
            kind = "ol" if s[0].isdigit() else "ul"
            if para:
                out.append("<p>" + _inline(" ".join(para)) + "</p>")
                para = []
            if lst != kind:
                if lst:
                    out.append(f"</{lst}>")
                out.append(f"<{kind}>")
                lst = kind
            out.append("<li>" + _inline(re.sub(r"^([-*+]|\d+[.)])\s+", "", s)) + "</li>")
        elif not s:
            flush()
        else:
            if lst:
                out.append(f"</{lst}>")
                lst = ""
            para.append(s)
        i += 1
    flush()
    return "\n".join(out)


def table(data: str) -> str:
    sep = ";" if data.count(";") > data.count(",") else ","
    rows = [r for r in csv.reader(io.StringIO(data.strip()), delimiter=sep) if r]
    if not rows:
        return "<p>(tableau vide)</p>"
    head = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in rows[0]) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
                   for r in rows[1:])
    return f"<table>{head}{body}</table>"


def page(titre: str, body: str) -> str:
    when = time.strftime("%d/%m/%Y à %H:%M")
    return (f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(titre)}</title><style>{STYLE}</style></head><body>"
            f"<h1>{html.escape(titre)}</h1>\n{body}\n"
            f'<p class="pied">Fait par Valdar, le {when}.</p></body></html>')


class Artefacts:
    def __init__(self, folder: Path, opener: Callable[[str], None] | None = None):
        self.folder = folder
        self.opener = opener

    def creer(self, titre: str, genre: str, contenu: str, ouvrir: bool = True) -> str:
        genre = (genre or "document").lower().strip()
        if genre not in ("page", "document", "tableau"):
            return "genre inconnu : page, document ou tableau"
        if not (contenu or "").strip():
            return "rien à mettre dedans"
        self.folder.mkdir(parents=True, exist_ok=True)
        base = f"{time.strftime('%Y-%m-%d')}_{slug(titre)}"
        dest = self.folder / f"{base}.html"
        n = 2
        while dest.exists():
            dest = self.folder / f"{base}-{n}.html"
            n += 1
        extra = ""
        if genre == "page":
            low = contenu.lstrip().lower()
            text = contenu if low.startswith(("<!doctype", "<html")) else page(titre, contenu)
        elif genre == "document":
            text = page(titre, markdown(contenu))
            dest.with_suffix(".md").write_text(contenu, "utf-8")
            extra = f" (+ {dest.with_suffix('.md').name})"
        else:
            text = page(titre, table(contenu))
            dest.with_suffix(".csv").write_text(contenu.strip() + "\n", "utf-8-sig")
            extra = f" (+ {dest.with_suffix('.csv').name}, ouvrable dans Excel)"
        dest.write_text(text, "utf-8")
        shown = ""
        if ouvrir and self.opener is not None:
            try:
                self.opener(str(dest))
                shown = ", ouvert dans le navigateur"
            except Exception as exc:
                shown = f", pas pu l'ouvrir ({exc})"
        return f"artefact créé : {dest}{extra}{shown}"
