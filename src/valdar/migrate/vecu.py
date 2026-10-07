"""Vécu hérité : les conversations d'Olivier avec Claude (export claude.ai, dossier
« training ia ») deviennent des souvenirs de Valdar (avenant 4 §7).

- Chaque conversation → un épisode daté, source « claude », personne « olivier » (privé :
  ne remonte que devant Olivier). Le contexte temporel est rejoué dans l'ordre du temps.
- L'affect de chaque message d'Olivier est estimé par l'évaluation rapide.
- La mémoire de Claude sur Olivier (export « memories ») → faits, source « claude_memoire ».
- Jeu d'entraînement : uniquement les messages d'Olivier (sa façon de parler, ses sujets),
  pour la phase 9. Les réponses de Claude restent des souvenirs, pas des modèles à imiter.

Lecture seule côté export ; chaque conversation n'est importée qu'une fois.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from valdar.appraisal import FastAppraisal
from valdar.config.loader import ValdarConfig
from valdar.memory import Episodic, Facts, TemporalContext

SPEAKERS = {"human": "Olivier", "assistant": "Claude"}


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _text(msg: dict[str, Any]) -> str:
    txt = (msg.get("text") or "").strip()
    if txt:
        return txt
    parts = []
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(p for p in parts if p).strip()


def _pad(appraisal: FastAppraisal, table: dict[str, list[float]], text: str
         ) -> tuple[float, float, float]:
    p = a = d = 0.0
    for name, scale in appraisal.appraise(text):
        v = table.get(name)
        if v:
            p, a, d = p + v[0] * scale, a + v[1] * scale, d + v[2] * scale
    clip = lambda x: max(-1.0, min(1.0, x))  # noqa: E731
    return clip(p), clip(a), clip(d)


_TAG = re.compile(r"\[(stated|inferred|[a-z_]+)\]\s*", re.I)


def memory_lines(export: dict[str, Any]) -> list[str]:
    """Faits tirés de la mémoire de Claude : puces des fichiers + phrases du résumé."""
    out: list[str] = []
    for f in export.get("memory_files") or []:
        for line in str(f.get("content") or "").splitlines():
            line = line.strip()
            if line.startswith("- "):
                line = _TAG.sub("", line[2:]).strip()
                if 12 <= len(line) <= 400:
                    out.append(line)
    summary = str(export.get("conversations_memory") or "")
    for para in summary.split("\n"):
        para = para.strip()
        if not para or para.startswith("**"):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if 20 <= len(sent) <= 400:
                out.append(sent)
    return out


class VecuImport:
    def __init__(self, export_dir: Path, cfg: ValdarConfig, memory: Episodic, facts: Facts,
                 training_dir: Path, person: str = "olivier"):
        self.root = Path(export_dir)
        self.cfg = cfg
        self.memory = memory
        self.facts = facts
        self.training_dir = Path(training_dir)
        self.person = person
        self.appraisal = FastAppraisal(cfg.appraisal_fast)

    def conversations(self) -> list[dict[str, Any]]:
        files = sorted(self.root.glob("conversations*/conversations.json")) or \
            sorted(self.root.glob("conversations.json"))
        convs: list[dict[str, Any]] = []
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                convs += [c for c in data if isinstance(c, dict)]
        convs.sort(key=lambda c: _ts(c.get("created_at")) or 0.0)
        return convs

    def run(self, on_progress: Callable[[int, int], None] | None = None) -> list[str]:
        if not self.root.is_dir():
            return [f"dossier introuvable : {self.root}"]
        report = []
        convs = self.conversations()
        ec = self.cfg.episodic
        ctx = TemporalContext(ec.scales_seconds, ec.dim, ec.min_step_seconds)
        new_eps = turns = skipped = 0
        olivier_lines: list[dict[str, Any]] = []
        for i, c in enumerate(convs):
            rows = []
            for m in c.get("chat_messages") or []:
                text = _text(m)
                t = _ts(m.get("created_at"))
                if not text or t is None:
                    continue
                speaker = SPEAKERS.get(m.get("sender"), str(m.get("sender") or "?"))
                pad = _pad(self.appraisal, ec.appraisal_pad, text) if speaker == "Olivier" \
                    else (0.0, 0.0, 0.0)
                rows.append((t, speaker, text, pad))
                if speaker == "Olivier":
                    olivier_lines.append({"t": t, "conversation": c.get("name") or "",
                                          "text": text})
            ep = self.memory.import_episode("claude", f"claude:{c.get('uuid')}",
                                            str(c.get("name") or ""), self.person, rows, ctx)
            if ep is None:
                skipped += 1
            else:
                new_eps += 1
                turns += len(rows)
            if on_progress is not None:
                on_progress(i + 1, len(convs))
        if new_eps:
            self.memory.adopt_context(ctx)
        report.append(f"vécu : {new_eps} conversation(s) devenues des souvenirs ({turns} "
                      f"échanges), {skipped} déjà là ou vides")
        report.append(self._facts())
        report.append(self._training(olivier_lines))
        return report

    def _facts(self) -> str:
        n = 0
        for f in sorted(self.root.glob("memories*/memories/*.json")):
            try:
                export = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for line in memory_lines(export):
                if self.facts.remember(line, person=self.person, source="claude_memoire",
                                       confidence=0.7) == "noté.":
                    n += 1
        return f"faits : {n} nouveau(x) fait(s) tiré(s) de la mémoire de Claude sur Olivier"

    def _training(self, lines: list[dict[str, Any]]) -> str:
        self.training_dir.mkdir(parents=True, exist_ok=True)
        out = self.training_dir / "olivier_messages.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in lines:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return (f"entraînement : {len(lines)} messages d'Olivier mis de côté pour la phase 9 "
                f"({out.name})")
