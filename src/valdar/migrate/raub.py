"""Reprise des données de RAUB (avenant 2 §9). Lecture seule côté RAUB : rien n'y est modifié.

Importé : faits (memory.json), stock (stock.db), rappels à venir, checklist, base de pinouts,
marqueurs d'affect (copiés pour la phase 7), et **la voix de RAUB** (avenant 3 §5) : la
référence clonée, ses extraits sources et le modèle XTTS v2.
Jamais importé : profils voix / visage (trop faibles), enregistrements de calibrage des
personnes (`cal_*.wav`), clés et jetons. Les fichiers de voix sont pris sur une liste
fermée, rien d'autre n'est lu dans ces dossiers.
Chaque import n'est fait qu'une fois (journal data/imports.json).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from valdar.atelier import Checklist, Reminders, Stock
from valdar.memory import Facts


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# La voix de RAUB : liste fermée de fichiers.
VOICE_FILES = ("xtts_ref.wav", "ref_eleven_1.wav", "ref_eleven_2.wav", "ref_eleven_3.wav")
XTTS_FILES = ("config.json", "model.pth", "dvae.pth", "mel_stats.pth", "speakers_xtts.pth",
              "vocab.json", "hash.md5", "LICENSE.txt", "README.md")
XTTS_REQUIRED = ("config.json", "model.pth", "dvae.pth", "mel_stats.pth", "vocab.json")


class _NotNow(Exception):
    """Étape impossible pour l'instant : signalée, pas marquée comme faite."""


def _copy_checked(src: Path, dst: Path) -> bool:
    """Copie si absent ou différent (taille). Retourne True si copié."""
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink()
        raise OSError(f"copie incomplète de {src.name}")
    tmp.replace(dst)
    return True


class RaubImport:
    def __init__(self, raub_root: Path, facts: Facts, stock: Stock, reminders: Reminders,
                 checklist: Checklist, pinouts_path: Path, data_dir: Path,
                 person: str = "olivier", voice_ref: Path | None = None,
                 xtts_dir: Path | None = None):
        self.root = Path(raub_root)
        self.data = self.root / "data"
        self.facts = facts
        self.stock = stock
        self.reminders = reminders
        self.checklist = checklist
        self.pinouts_path = Path(pinouts_path)
        self.data_dir = Path(data_dir)
        self.person = person
        self.voice_ref = Path(voice_ref) if voice_ref else self.data_dir / "voice" / "xtts_ref.wav"
        self.xtts_dir = Path(xtts_dir) if xtts_dir else self.data_dir / "models" / "xtts-v2"
        self.log_path = self.data_dir / "imports.json"

    # --------------------------------------------------------------- journal
    def _done(self) -> dict[str, Any]:
        return _read_json(self.log_path) or {}

    def _mark(self, key: str, info: str) -> None:
        log = self._done()
        log[key] = {"at": time.time(), "info": info}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    def run(self, force: bool = False,
            on_step: Callable[[str], None] | None = None) -> list[str]:
        if not self.data.is_dir():
            return [f"dossier RAUB introuvable : {self.data}"]
        report = []
        done = self._done()
        for key, fn in (("memoire", self._memory), ("stock", self._stock),
                        ("rappels", self._reminders), ("checklist", self._checklist),
                        ("pinouts", self._pinouts), ("affect", self._affect),
                        ("voix", self._voice)):
            if key in done and not force:
                report.append(f"{key} : déjà importé, ignoré")
                continue
            if on_step is not None:
                on_step(key)
            try:
                info = fn()
            except _NotNow as exc:
                report.append(f"{key} : {exc} (sera retenté au prochain import)")
                continue
            self._mark(key, info)
            report.append(f"{key} : {info}")
        return report

    # --------------------------------------------------------------- étapes
    def _memory(self) -> str:
        items = _read_json(self.data / "memory.json")
        if not isinstance(items, list):
            return "aucun fichier memory.json"
        n = 0
        for it in items:
            if isinstance(it, dict) and str(it.get("text", "")).strip():
                self.facts.remember(str(it["text"]), person=self.person, source="raub",
                                    confidence=0.7, when=float(it.get("at") or time.time()))
                n += 1
        return f"{n} souvenir(s) repris"

    def _stock(self) -> str:
        db = self.data / "stock.db"
        if not db.is_file():
            return "aucune réserve"
        con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT name, qty, unit, loc, threshold FROM items").fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
        existing = {i["name"] for i in self.stock.find()}
        n = 0
        for name, qty, unit, loc, threshold in rows:
            if name not in existing:
                self.stock.set_item(name, float(qty or 0), unit or "", loc or "",
                                    float(threshold or 0), note="repris de RAUB")
                n += 1
        return f"{n} article(s) repris"

    def _reminders(self) -> str:
        items = _read_json(self.data / "rappels.json")
        if not isinstance(items, list):
            return "aucun rappel"
        now = time.time()
        n = 0
        for r in items:
            if isinstance(r, dict) and r.get("etat") == "attente" and float(r.get("due", 0)) > now:
                self.reminders.add(str(r.get("texte", "")), float(r["due"]))
                n += 1
        return f"{n} rappel(s) à venir repris"

    def _checklist(self) -> str:
        d = _read_json(self.data / "checklist.json")
        if not isinstance(d, dict) or not d.get("items"):
            return "aucune checklist"
        if self.checklist.current.active():
            return "checklist Valdar déjà active, celle de RAUB n'a pas été reprise"
        self.checklist.create(str(d.get("titre", "")), [str(i.get("text", ""))
                                                         for i in d["items"]])
        done = [str(n + 1) for n, i in enumerate(d["items"]) if i.get("done")]
        if done:
            self.checklist.check(done)
        return f"checklist « {d.get('titre', '')} » reprise"

    def _pinouts(self) -> str:
        src = self.root / "raub" / "data" / "pinouts.json"
        if not src.is_file():
            return "aucune base de pinouts"
        self.pinouts_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, self.pinouts_path)
        n = len(_read_json(self.pinouts_path) or {})
        return f"{n} carte(s) reprise(s)"

    def _affect(self) -> str:
        src = self.data / "affect.json"
        if not src.is_file():
            return "aucun affect"
        dst = self.data_dir / "raub_affect.json"
        shutil.copyfile(src, dst)
        return "marqueurs par personne et activité copiés (utilisés en phase 7)"

    def _voice(self) -> str:
        """La voix construite pour RAUB devient celle de Valdar (avenant 3 §5)."""
        vdir = self.data / "voice"
        mdir = self.data / "models" / "xtts-v2"
        ref = vdir / "xtts_ref.wav"
        missing = [f for f in XTTS_REQUIRED if not (mdir / f).is_file()]
        if not ref.is_file() or missing:
            raise _NotNow("voix de RAUB introuvable ("
                          + ", ".join((["xtts_ref.wav"] if not ref.is_file() else []) + missing)
                          + ")")
        copied = int(_copy_checked(ref, self.voice_ref))
        for name in VOICE_FILES[1:]:
            if (vdir / name).is_file():
                copied += _copy_checked(vdir / name, self.voice_ref.parent / name)
        size = 0
        for name in XTTS_FILES:
            src = mdir / name
            if src.is_file():
                copied += _copy_checked(src, self.xtts_dir / name)
                size += src.stat().st_size
        return (f"référence clonée + modèle XTTS v2 repris ({size / 1e9:.1f} Go, {copied} "
                "fichier(s) copié(s), tailles vérifiées) : Valdar parlera avec la voix de RAUB")
