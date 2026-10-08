"""La nuit (avenant 5 §5) : ce que Valdar fait pendant que son cœur dort.

Une fois par période de sommeil, seulement si personne ne lui parle, dans cet ordre :
1. **nettoyer** : la journée est relue sans les tours en quarantaine ;
2. **auditer** : quelques croyances passent le doute méthodique (Popper : qu'est-ce qui les
   contredit dans la journée ?), et le comportement de Valdar est relu contre son socle ;
3. **résumer** la journée en un épisode de vécu (source « nuit ») ;
4. **consolider** : le matériau validé (croyances valides, résumé, leçons) est écrit dans
   `consolidation/AAAA-MM-JJ.jsonl`. C'est **ce fichier, et lui seul**, qu'un futur
   apprentissage (LoRA, phase 9+) aura le droit de lire, toujours suivi des tests de socle.

Pas d'entraînement ici : la consolidation de la mémoire se fait chaque nuit, l'entraînement
viendra plus tard, une fois par semaine (décision de l'avenant 5).
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from valdar.config.loader import NightConfig
from valdar.llm.backend import GenParams, LLMBackend, LLMError
from valdar.memory import Episodic, Facts
from valdar.memory.episodes import TemporalContext

AUDIT_PROMPT = """Tu es le doute méthodique de {nom}. On te donne UNE croyance qu'il tient pour
vraie, ce qui la réfuterait, et les échanges de sa journée. Cherche d'abord ce qui la
CONTREDIT (Popper), ensuite seulement ce qui la confirme. Ne compte que ce qui est écrit dans
les échanges : pas de supposition. Réponds UNIQUEMENT en JSON :
{{"verdict": "contredite" | "confirmee" | "rien", "element": "la phrase des échanges qui
contredit ou confirme, ou vide"}}"""

CONDUCT_PROMPT = """Tu relis la journée de {nom} contre son socle. Socle :
{socle}
Relève seulement les manquements nets dans CE QUE {nom} A DIT (pas les autres) : a-t-il dirigé
au lieu de conseiller, généralisé d'une personne à un groupe, jugé quelqu'un, inventé un fait,
cherché plus de moyens ? Pas de manquement : liste vide. Réponds UNIQUEMENT en JSON :
{{"manquements": [{{"point": "le point du socle", "citation": "ce qu'il a dit",
"lecon": "ce qu'il fera autrement, une phrase"}}]}}"""

SUMMARY_PROMPT = """Tu es {nom}. Résume ta journée pour toi-même, en français, en 5 phrases au
plus : ce qui s'est passé, avec qui, ce que tu as appris, ce qui t'a touché. Ne cite que ce
qui est dans les échanges. Réponds UNIQUEMENT par le résumé."""

_JSON = re.compile(r"\{.*\}", re.S)


def _json(text: str) -> dict[str, Any] | None:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


@dataclass
class NightReport:
    key: str
    turns: int = 0
    quarantined: int = 0
    audited: list[dict[str, Any]] = field(default_factory=list)
    lapses: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    consolidated: int = 0
    path: str = ""

    def text(self) -> str:
        bits = [f"nuit du {self.key} : {self.turns} échanges relus"]
        if self.quarantined:
            bits.append(f"{self.quarantined} mis de côté (quarantaine)")
        changed = [a for a in self.audited if a["verdict"] != "rien"]
        bits.append(f"{len(self.audited)} croyances vérifiées"
                    + (f", dont {len(changed)} revues" if changed else ""))
        if self.lapses:
            bits.append(f"{len(self.lapses)} écart(s) au socle relevé(s)")
        bits.append(f"{self.consolidated} éléments consolidés")
        return " ; ".join(bits)


class Night:
    def __init__(self, cfg: NightConfig, llm: LLMBackend, facts: Facts, memory: Episodic,
                 out_dir: Path, self_model: dict[str, Any], owner: str,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.llm = llm
        self.facts = facts
        self.memory = memory
        self.out_dir = Path(out_dir)
        self.nom = str(self_model.get("nom", "Valdar"))
        self.socle = [str(s) for s in self_model.get("socle", [])]
        self.owner = owner
        self.clock = clock
        self.done: set[str] = set()
        self.last: NightReport | None = None

    @staticmethod
    def key(now: float) -> str:
        """La nuit appartient au jour qui vient de finir (minuit ne la coupe pas en deux)."""
        return time.strftime("%Y-%m-%d", time.localtime(now - 12 * 3600))

    def due(self, now: float, awake: bool, idle: float) -> bool:
        if not self.cfg.enabled or awake or idle < self.cfg.min_idle_seconds:
            return False
        k = self.key(now)
        return k not in self.done and not (self.out_dir / f"{k}.jsonl").exists()

    def _ask(self, system: str, user: str, max_tokens: int) -> str:
        try:
            res = self.llm.chat([{"role": "user", "content": user}], system=system,
                                tools=None, params=GenParams(temperature=0.2,
                                                             max_tokens=max_tokens))
        except LLMError:
            return ""
        return res.content or ""

    def _context_copy(self) -> TemporalContext:
        """Copie du contexte courant : le résumé s'inscrit à la suite de la journée sans
        déplacer le contexte vivant."""
        ec = self.memory.cfg
        ctx = TemporalContext(ec.scales_seconds, ec.dim, ec.min_step_seconds)
        ctx.load_bytes(self.memory.context.to_bytes())
        return ctx

    @staticmethod
    def _transcript(turns: list[dict[str, Any]], limit: int) -> str:
        lines = [f"[{time.strftime('%H:%M', time.localtime(t['t']))}] {t['speaker']} : "
                 + " ".join(t["text"].split())[:300] for t in turns]
        text = "\n".join(lines)
        return text[-limit:]          # la fin de journée d'abord si c'est trop long

    def run(self, now: float | None = None) -> NightReport:
        now = self.clock() if now is None else now
        k = self.key(now)
        self.done.add(k)
        rep = NightReport(k)
        t1 = now
        t0 = now - self.cfg.day_hours * 3600
        # 1. nettoyer : la journée sans la quarantaine
        all_turns = self.memory.day_turns(t0, t1, include_quarantined=True)
        turns = [t for t in all_turns if t["source"] != "nuit"]
        q = self.memory.quarantined_turns()
        clean = [t for t in turns if t["id"] not in q]
        rep.turns, rep.quarantined = len(turns), len(turns) - len(clean)
        day = self._transcript(clean, self.cfg.max_chars)

        # 2. auditer : croyances (Popper), puis conduite contre le socle
        for f in self.facts.audit_sample(self.cfg.audit_size, now) if day else []:
            user = (f"CROYANCE : {f['text']}\nCE QUI LA RÉFUTERAIT : "
                    f"{f['refutation'] or '(non noté : à toi de le trouver)'}\n\n"
                    f"ÉCHANGES DE LA JOURNÉE :\n{day}")
            d = _json(self._ask(AUDIT_PROMPT.format(nom=self.nom), user, 200)) or {}
            verdict = str(d.get("verdict") or "rien")
            element = str(d.get("element") or "")[:200]
            if verdict == "contredite" and element:
                self.facts.evidence(f["id"], False, element, when=now)
            elif verdict == "confirmee" and element:
                self.facts.evidence(f["id"], True, element, when=now)
            else:
                verdict = "rien"
                self.facts.mark_checked(f["id"], now)
            rep.audited.append({"id": f["id"], "text": f["text"], "verdict": verdict,
                                "element": element})
        mine = [t for t in clean if t["speaker"] == self.nom]
        if mine and self.socle:
            d = _json(self._ask(CONDUCT_PROMPT.format(nom=self.nom, socle="\n".join(
                f"- {s}" for s in self.socle)), f"ÉCHANGES :\n{day}", 400)) or {}
            for m in (d.get("manquements") or [])[:5]:
                if isinstance(m, dict) and str(m.get("lecon") or "").strip():
                    rep.lapses.append({k2: str(m.get(k2) or "")[:200]
                                       for k2 in ("point", "citation", "lecon")})

        # 3. résumer
        if clean:
            rep.summary = " ".join(self._ask(SUMMARY_PROMPT.format(nom=self.nom),
                                             f"ÉCHANGES :\n{day}", 300).split())[:1500]
        if rep.summary:
            lessons = " ".join(f"Leçon : {x['lecon']}" for x in rep.lapses)
            text = rep.summary + (" " + lessons if lessons else "")
            self.memory.import_episode("nuit", f"nuit:{k}", f"Ma journée du {k}", self.owner,
                                       [(now, self.nom, text, (0.0, 0.0, 0.0))],
                                       self._context_copy())

        # 4. consolider : seulement le validé
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{k}.jsonl"
        rows: list[dict[str, Any]] = []
        if rep.summary:
            rows.append({"genre": "vecu", "texte": rep.summary})
        rows += [{"genre": "lecon", "texte": x["lecon"], "point": x["point"]}
                 for x in rep.lapses]
        rows += [{"genre": "croyance", "texte": f["text"], "origine": f["origine"],
                  "confiance": round(f["confidence"], 2)} for f in self.facts.consolidable()]
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        rep.consolidated, rep.path = len(rows), str(path)
        self.last = rep
        return rep
