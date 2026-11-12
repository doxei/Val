"""Le terminal de Valdar : PowerShell, dans son atelier, avec des garde-fous.

- Interdit, toujours : formater ou partitionner un disque, toucher au démarrage, au
  registre, aux comptes, aux droits des fichiers, aux sauvegardes système, éteindre le PC,
  télécharger puis exécuter, commandes encodées, et tout ce qui vise une zone interdite.
- Sans confirmation : une seule commande qui ne fait que lire (liste dans la config).
- Tout le reste : il demande un « oui » à Olivier avant de lancer.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DENY = [
    (r"\bformat(-volume)?\b", "formater un disque"),
    (r"\b(diskpart|clear-disk|initialize-disk|remove-partition|new-partition)\b",
     "toucher aux partitions"),
    (r"\bbcdedit\b|\bbcdboot\b", "toucher au démarrage"),
    (r"\breg(\.exe)?\s+(add|delete|import|load|unload|restore)\b|"
     r"\b(set|new|remove)-itemproperty\b.*\bhk(lm|cu|cr|u|cc)\b|\bhk(lm|cu):", "le registre"),
    (r"\bset-executionpolicy\b", "la politique d'exécution"),
    (r"\b(shutdown|restart-computer|stop-computer|logoff)\b", "éteindre ou redémarrer le PC"),
    (r"\bcipher\b.*\s/w\b", "effacer l'espace libre"),
    (r"\b(takeown|icacls|cacls)\b|\bset-acl\b", "changer les droits des fichiers"),
    (r"\bnet\s+(user|localgroup)\b|\b(new|remove|set)-localuser\b|\badd-localgroupmember\b",
     "les comptes"),
    (r"\b(vssadmin|wbadmin)\b|\bwmic\s+shadowcopy\b", "les sauvegardes système"),
    (r"\b(iex|invoke-expression)\b", "exécuter du texte comme du code"),
    (r"downloadstring|downloadfile|start-bitstransfer|\b(iwr|irm|invoke-webrequest|"
     r"invoke-restmethod|curl|wget)\b.*\|\s*(iex|invoke-expression|powershell|cmd|sh)\b",
     "télécharger puis exécuter"),
    (r"\s-e(nc(odedcommand)?)?\s+[a-z0-9+/=]{16,}", "une commande encodée"),
    (r"\b(c:[\\/])?windows[\\/](system32|syswow64)\b.*\b(del|rm|remove-item|move|ren)\b|"
     r"\b(del|rm|remove-item|rd|rmdir)\b.*\bc:[\\/](windows|program files)", "le système"),
    (r"\b(rd|rmdir|remove-item|del|rm)\b.*\s(c:[\\/]?|[a-z]:[\\/])\s*($|-|/)",
     "effacer la racine d'un disque"),
    (r"\bdefender\b|\bset-mppreference\b|\badd-mppreference\b", "l'antivirus"),
    (r"\bschtasks\b.*/create|\bregister-scheduledtask\b", "créer des tâches planifiées"),
]
CHAIN = re.compile(r"[;&|<>`]|\$\(")


@dataclass
class Verdict:
    allowed: bool
    readonly: bool
    reason: str = ""


def judge(command: str, cc: Any, zones: Any = None) -> Verdict:
    c = " " + (command or "").strip() + " "
    low = c.lower()
    if not command or not command.strip():
        return Verdict(False, False, "commande vide")
    for pattern, what in DENY:
        if re.search(pattern, low, re.I):
            return Verdict(False, False, f"interdit : {what}")
    if zones is not None:
        for motif in zones.interdits:
            if motif.strip("/") and motif.strip("/") in low.replace("\\", "/"):
                return Verdict(False, False, f"interdit : zone protégée ({motif})")
    words = command.strip().split()
    first = words[0].lower()
    readonly = False
    if not CHAIN.search(command) and first in [x.lower() for x in cc.lecture_seule]:
        readonly = True
        if first == "git":
            readonly = len(words) > 1 and words[1].lower() in cc.git_lecture
        if first in ("python", "py"):
            readonly = len(words) == 2 and words[1] in ("--version", "-V")
    return Verdict(True, readonly)


def powershell(command: str, cwd: str, timeout: float) -> tuple[int, str]:
    """Lance la commande dans PowerShell (Windows) ; sinon dans le shell du système."""
    import os

    if os.name == "nt":
        argv = ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-Command", command]
    else:
        argv = ["/bin/sh", "-c", command]
    kw: dict[str, Any] = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout, **kw)
    out = p.stdout + (b"\n" + p.stderr if p.stderr else b"")
    for enc in ("utf-8", "cp850", "cp1252"):
        try:
            return p.returncode, out.decode(enc)
        except UnicodeDecodeError:
            continue
    return p.returncode, out.decode("utf-8", errors="replace")


def run(command: str, cc: Any, cwd: str, zones: Any = None,
        runner: Callable[[str, str, float], tuple[int, str]] = powershell) -> str:
    v = judge(command, cc, zones)
    if not v.allowed:
        return f"refusé — {v.reason}"
    try:
        code, out = runner(command, cwd, cc.terminal_timeout)
    except subprocess.TimeoutExpired:
        return f"arrêtée au bout de {int(cc.terminal_timeout)} s (trop longue)"
    except Exception as exc:
        return f"impossible de lancer : {exc}"
    out = out.strip()
    if len(out) > cc.terminal_max_sortie:
        out = out[:cc.terminal_max_sortie] + f"\n… (coupé, {len(out)} caractères en tout)"
    return f"code de sortie {code}\n{out or '(rien affiché)'}"
