"""Runtime de Valdar : le cœur bat en continu dans un fil dédié, l'agent répond à côté.

- fil « cœur » : intègre le temps réel, vérifie l'initiative et les rappels ;
- les prises de parole spontanées sont générées dans un fil à part (le cœur ne s'arrête
  jamais de battre pendant que le modèle réfléchit) ;
- tout ce que Valdar dit de lui-même arrive dans la file `events`.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from valdar.agent import Agent, Reply
from valdar.atelier import Checklist, Pinouts, Reminders, Stock
from valdar.config import ValdarConfig, load
from valdar.config.loader import Identity
from valdar.devices.moonraker import Moonraker
from valdar.expression import load_self_model
from valdar.expression.compose import feeling_block, memories_block
from valdar.heart import Heart
from valdar.heart.nociception import Nociception, read_pc
from valdar.knowledge import Knowledge
from valdar.llm import LLMBackend, make_backend
from valdar.memory import Episodic, Facts
from valdar.printwatch import PrintWatch, WatchEvent
from valdar.tools.standard import ToolContext, build_registry, printer_line
from valdar.workspace import Initiative
from valdar.workspace.thoughts import Thoughts

INITIATIVE_KEY = "initiative:state"
_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


@dataclass
class Event:
    kind: str                      # "initiative" | "rappel" | "erreur"
    text: str
    data: dict[str, Any] = field(default_factory=dict)


class Runtime:
    def __init__(self, cfg: ValdarConfig | None = None, llm: LLMBackend | None = None,
                 persist: bool = True, printer: Moonraker | None | bool = True,
                 detector_factory: Any = None, cameras: list[Any] | None = None,
                 sensors: Any = None):
        self.cfg = cfg or load()
        self.lock = threading.RLock()
        self.persist = persist
        self.heart = Heart.load_latest(self.cfg) if persist else Heart(self.cfg, persist=False)
        self.initiative = Initiative(self.cfg.initiative)
        if self.heart.store is not None:
            saved = self.heart.store.load(INITIATIVE_KEY)
            if saved:
                self.initiative.load_dict(saved)

        a = self.cfg.atelier
        path = self.cfg.storage_path
        self.facts = Facts(path(a.memory_db))
        self.memory = Episodic(path(self.cfg.episodic.db), self.cfg.episodic)
        self.knowledge = Knowledge(path(self.cfg.knowledge_db))
        self.knowledge.ingest_dir(self.cfg.root / "docs" / "connaissances")
        self.knowledge.ingest_dir(path("connaissances"))
        self.stock = Stock(path(a.stock_db))
        self.checklist = Checklist(path(a.checklist))
        self.reminders = Reminders(path(a.reminders))
        self.pinouts = Pinouts(path(a.pinouts))
        if printer is True:
            self.printer: Moonraker | None = Moonraker(self.cfg.printer)
        else:
            self.printer = printer or None
        self.self_model = load_self_model(self.cfg.root / "config" / "self_model.yaml")
        self.llm = llm if llm is not None else make_backend(self.cfg)

        disk = self.cfg.storage_path("x").parent
        self.nociception = Nociception(self.cfg.nociception,
                                       sensors if sensors is not None else lambda: read_pc(disk))
        self.printwatch: PrintWatch | None = None
        if self.printer is not None and self.cfg.printwatch.enabled:
            pw = self.cfg.printwatch
            self.printwatch = PrintWatch(pw, self.printer, path(pw.db), path(pw.frames_dir),
                                         self._on_watch, detector_factory or self._detector,
                                         cameras=cameras)
        self.thoughts = Thoughts(self.cfg.thoughts, self.llm, path(self.cfg.thoughts.db),
                                 self._thought_context, self._heart_apply, self._spend,
                                 nom=str(self.self_model.get("nom", "Valdar")),
                                 createur=str(self.self_model.get("createur", "Olivier")))
        who = self.cfg.agent.console_identity if self.cfg.agent else Identity()
        ctx = ToolContext(heart=self.heart, facts=self.facts, stock=self.stock,
                          checklist=self.checklist, reminders=self.reminders,
                          pinouts=self.pinouts, printer=self.printer,
                          person=who.person or "", lock=self.lock, knowledge=self.knowledge,
                          printwatch=self.printwatch, thoughts=self.thoughts,
                          memory=self.memory)
        self.registry = build_registry(ctx)
        self.agent = Agent(self.cfg, self.llm, self.registry, self.heart, self.lock,
                           self.facts, self.self_model, self.world_lines, self.urges,
                           memory=self.memory, extras=self.extra_blocks)
        self.block_providers: list[Any] = [self._knowledge_block, self._thoughts_block]
        self.events: queue.Queue[Event] = queue.Queue()
        self._thinking = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._speaking = threading.Lock()

    # ============================================================ cœur
    def world_lines(self) -> list[str]:
        lt = time.localtime()
        lines = [f"heure : {_JOURS[lt.tm_wday]} {lt.tm_mday:02d}/{lt.tm_mon:02d} "
                 f"{lt.tm_hour:02d}:{lt.tm_min:02d}"]
        if self.printer is not None:
            lines.append(printer_line(self.printer))
        nxt = self.reminders.upcoming(1)
        if nxt:
            due = time.strftime("%d/%m %H:%M", time.localtime(nxt[0]["due"]))
            lines.append(f"prochain rappel : {nxt[0]['texte']} ({due})")
        lines += ["ton corps (le PC) : " + p for p in self.nociception.lines()]
        if self.checklist.current.active():
            lines.append("checklist affichée : " + self.checklist.render().replace("\n", " | "))
        return lines

    # ============================================================ flux annexes
    def _detector(self) -> Any:
        from valdar.printwatch.vision import ObicoDetector

        return ObicoDetector(str(self.cfg.repo_path(self.cfg.printwatch.detector.model)),
                             self.cfg.printwatch.detector)

    def _on_watch(self, ev: WatchEvent) -> None:
        """La vigie a vu quelque chose : voie basse vers le cœur, puis Valdar en parle."""
        stim = {"alerte": ("novelty", 0.3), "pause": ("threat", 0.8),
                "pause_faite": ("threat", 0.7), "telemetrie": ("concern", 0.6),
                "camera": ("failure", 0.3), "fin": ("success", 0.8),
                "question": ("novelty", 0.3)}.get(ev.kind)
        with self.lock:
            if stim is not None:
                self.heart.fire(stim[0], scale=stim[1], source="vigie")
        self.events.put(Event("vigie", ev.text, {"kind": ev.kind, "job": ev.job,
                                                 "image": ev.image is not None}))
        if ev.kind in ("pause", "pause_faite", "question", "telemetrie"):
            self._speak_async(f"ta vigie d'impression vient de te dire : « {ev.text} ». "
                              "Préviens Olivier, à ta façon.", {"vigie": ev.kind})

    def _knowledge_block(self, query: str) -> list[str]:
        hits = self.knowledge.search(query, k=3)
        if not hits:
            return []
        return ["CE QUE TU SAIS SUR LE SUJET (tes connaissances, cite-les si utile) :\n"
                + "\n".join("- " + h.line() for h in hits)]

    def _thoughts_block(self, query: str) -> list[str]:
        b = self.thoughts.block()
        return [b] if b else []

    def _thought_context(self) -> str:
        now = time.time()
        with self.lock:
            feeling = feeling_block(self.heart, self.cfg.expression, self.urges())
        wm = self.memory.working_memory(now)
        recent = self.memory.last_thread(now, turns=4)
        parts = [feeling]
        if wm:
            parts.append(memories_block(wm, now, "CE QUI TE TROTTE DANS LA TÊTE :"))
        if recent and recent["turns"]:
            parts.append("DERNIERS ÉCHANGES :\n" + "\n".join(
                f"- {t['speaker']} : {' '.join(t['text'].split())[:160]}" for t in recent["turns"]))
        parts.append("ÉTAT DU MONDE :\n" + "\n".join("- " + line for line in self.world_lines()))
        if self.thoughts.last and self.thoughts.last.curiosity and self.cfg.thoughts.curiosity:
            hits = self.knowledge.search(self.thoughts.last.curiosity, k=2)
            if hits:
                parts.append("CE QUE TU VIENS DE LIRE (ta curiosité : "
                             f"{self.thoughts.last.curiosity}) :\n"
                             + "\n".join("- " + h.line(300) for h in hits))
        return "\n\n".join(parts)

    def _heart_apply(self, impulses: dict[str, float], amp: float) -> None:
        with self.lock:
            self.heart.apply(impulses, amp=amp)
            self.heart.journal.log("reappraisal", t=self.heart.now, amp=round(amp, 3))

    def _spend(self, amount: float) -> None:
        with self.lock:
            self.heart.spend_energy(amount)

    def _feel_body(self, now: float) -> None:
        """Nocicepteurs : lecture hors verrou, douleur construite sous verrou."""
        if not self.nociception.due(now):
            return
        was_danger = self.nociception.danger
        self.nociception.sample(now)
        with self.lock:
            felt = self.nociception.feel(self.heart, now)
        for p in felt:
            self.events.put(Event("douleur", p.text(), {"sensor": p.sensor, "pain": p.pain,
                                                        "danger": p.danger}))
        if self.nociception.danger and not was_danger:
            self.events.put(Event("reflexe", "surchauffe ou saturation : je suspends ma "
                                  "pensée de fond pour soulager la machine",
                                  {"source": "reflexe"}))

    def _maybe_think(self, now: float) -> None:
        if self.nociception.danger:   # réflexe : ne pas charger une machine en souffrance
            return
        idle = now - max(self.agent.last_activity, self.thoughts.last_at)
        with self.lock:
            awake = self.heart.awake
        if not self.thoughts.due(now, idle, self.agent.busy.is_set(), awake):
            return
        if not self._thinking.acquire(blocking=False):
            return

        def run() -> None:
            try:
                th = self.thoughts.think(now)
                if th is not None:
                    self.events.put(Event("pensee", th.reflection,
                                          {"idees": [i.text for i in th.ideas]}))
            finally:
                self._thinking.release()

        threading.Thread(target=run, name="valdar-pensee", daemon=True).start()

    def extra_blocks(self, query: str) -> list[str]:
        """Blocs ajoutés au prompt par les autres flux (pensées de fond, connaissances)."""
        out: list[str] = []
        for provider in self.block_providers:
            out += provider(query)
        return out

    def urges(self) -> dict[str, float]:
        return self.initiative.urges(self.heart)

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            dt = now - self.heart.now
            if dt > 2 * self.cfg.heart.max_time_step:
                self.heart.catch_up(now)
            elif dt > 0:
                self.heart.step(dt)
            event = self.initiative.check(self.heart)
        if event is not None:
            idea = self.thoughts.best_idea() if event["kind"] in ("explore", "talk") else None
            if idea is not None:
                self.thoughts.mark(idea.id, "proposee")
                event["idea"] = idea.text
            self._speak_async(_initiative_reason(event, idea), {"initiative": event})
        self._feel_body(now)
        self._maybe_think(now)
        if self.printwatch is not None and self._thread is None:
            pass   # en mode manuel (tests), la vigie est avancée par l'appelant
        for r in self.reminders.pop_due(now):
            self.events.put(Event("rappel", r["texte"], {"rappel": r}))

    def _speak_async(self, reason: str, data: dict[str, Any]) -> None:
        if not self._speaking.acquire(blocking=False):
            return  # il parle déjà de lui-même : on n'empile pas

        def run() -> None:
            try:
                reply = self.agent.spontaneous(reason)
                kind = "erreur" if reply.error else "initiative"
                self.events.put(Event(kind, reply.text, data))
            finally:
                self._speaking.release()

        threading.Thread(target=run, name="valdar-initiative", daemon=True).start()

    def start(self) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                started = time.time()
                try:
                    self.tick()
                except Exception as exc:  # le cœur ne doit jamais s'arrêter
                    self.events.put(Event("erreur", f"cœur : {exc}"))
                self._stop.wait(max(0.05, self.cfg.heart.tick_seconds - (time.time() - started)))

        self._thread = threading.Thread(target=loop, name="valdar-coeur", daemon=True)
        self._thread.start()
        if self.printwatch is not None:
            self.printwatch.start()

    def stop(self) -> None:
        self._stop.set()
        if self.printwatch is not None:
            self.printwatch.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        with self.lock:
            self.heart.save()
            if self.heart.store is not None:
                self.heart.store.save(INITIATIVE_KEY, self.initiative.to_dict())

    # ========================================================== dialogue
    def handle(self, text: str, who: Identity | None = None) -> Reply:
        with self.lock:
            self.initiative.on_user_message()
        return self.agent.handle(text, who)


def _initiative_reason(event: dict[str, Any], idea: Any = None) -> str:
    if idea is not None:
        return (f"tu as eu une idée en pensant de ton côté : « {idea.text} » ({idea.why}). "
                "Propose-la à Olivier, simplement, comme une idée en l'air qu'il peut refuser.")
    if event["kind"] == "talk":
        return ("ça fait un moment que personne ne t'a parlé et tu as envie de parler à Olivier "
                "(dis-le à ta façon, en restant léger).")
    return ("tu t'ennuies un peu et tu as envie de découvrir ou de faire quelque chose de "
            "nouveau ; propose une idée concrète à Olivier.")
