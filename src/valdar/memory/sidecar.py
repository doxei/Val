"""Un petit Ollama à part, sur le processeur, rien que pour les plongements (bge-m3).

Pourquoi : sur la machine d'Olivier, Ollama tourne avec OLLAMA_MAX_LOADED_MODELS=1. Un
plongement demandé au même serveur que Gemma éjecte Gemma de la carte graphique ; le message
suivant le recharge (~6 s) et repart sans cache (journal d'Ollama, 8 octobre : 22
rechargements en 40 min, chaque réponse à ~15 s). Un second serveur, sur un autre port et
sans carte graphique, garde chacun à sa place, sans toucher aux réglages d'Olivier.

Il partage le dossier des modèles du serveur principal (rien à retélécharger).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx


def ollama_exe() -> str | None:
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if local.is_file():
        return str(local)
    return shutil.which("ollama")


def alive(url: str, timeout: float = 1.0) -> bool:
    try:
        return httpx.get(url.rstrip("/") + "/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


class Sidecar:
    def __init__(self, port: int):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None
        self.error = ""

    def start(self, wait: float = 15.0) -> bool:
        if alive(self.url):
            return True
        exe = ollama_exe()
        if exe is None:
            self.error = "ollama introuvable"
            return False
        env = dict(os.environ)
        env.update({
            "OLLAMA_HOST": f"127.0.0.1:{self.port}",
            "CUDA_VISIBLE_DEVICES": "-1",       # aucune carte graphique : processeur seul
            "HIP_VISIBLE_DEVICES": "-1",
            "OLLAMA_LLM_LIBRARY": "cpu",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_KEEP_ALIVE": "-1",          # bge-m3 reste chargé (en mémoire vive)
        })
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen([exe, "serve"], env=env, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError as exc:
            self.error = str(exc)
            return False
        end = time.time() + wait
        while time.time() < end:
            if alive(self.url):
                return True
            time.sleep(0.3)
        self.error = "le serveur des plongements n'a pas démarré"
        return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
