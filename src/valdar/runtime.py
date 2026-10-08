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
from valdar.heart.interoception import Load, TimedLLM
from valdar.heart.nociception import Nociception, read_pc
from valdar.knowledge import Knowledge
from valdar.llm import LLMBackend, make_backend
from valdar.memory import Episodic, Facts
from valdar.printwatch import PrintWatch, WatchEvent
from valdar.relations import Relations
from valdar.tools.standard import ToolContext, build_registry, printer_line
from valdar.workspace import Initiative
from valdar.workspace.night import Night
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
        from valdar.memory.embedder import Embedder

        ec = self.cfg.embeddings
        self._sidecar: Any = None
        emb_url = ec.url or f"http://127.0.0.1:{ec.sidecar_port}"
        same = emb_url.rstrip("/") == self.cfg.llm.url.rstrip("/")
        self.embedder = Embedder(ec, emb_url)
        if same and not ec.same_server_ok:
            # même serveur que Gemma : chaque plongement l'éjecterait (voir sidecar.py)
            self.embedder.disable("serveur partagé avec Gemma (embeddings.same_server_ok)")
        self.memory = Episodic(path(self.cfg.episodic.db), self.cfg.episodic, self.embedder)
        self.knowledge = Knowledge(path(self.cfg.knowledge_db), embedder=self.embedder)
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
        raw_llm = llm if llm is not None else make_backend(self.cfg)
        self.load = Load(self.cfg.interoception)
        self.llm: LLMBackend = (TimedLLM(raw_llm, self.load)   # type: ignore[assignment]
                                if self.cfg.interoception.enabled else raw_llm)
        self.ambient: Any = None          # double audition, branchée avec le micro
        self.gate: Any = None             # le portier, quand le micro tourne
        self._triaging = 0
        self._triage_lock = threading.Lock()

        disk = self.cfg.storage_path("x").parent
        self.nociception = Nociception(self.cfg.nociception,
                                       sensors if sensors is not None else lambda: read_pc(disk))
        self.printwatch: PrintWatch | None = None
        if self.printer is not None and self.cfg.printwatch.enabled:
            pw = self.cfg.printwatch
            self.printwatch = PrintWatch(pw, self.printer, path(pw.db), path(pw.frames_dir),
                                         self._on_watch, detector_factory or self._detector,
                                         cameras=cameras)
        self.relations = Relations(self.cfg, path(self.cfg.relations.db))
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
                          memory=self.memory, relations=self.relations)
        self.tool_ctx = ctx
        self._who: Identity = who
        self._handling = threading.RLock()
        self.registry = build_registry(ctx)
        self.kiwix: Any = None
        self._kiwix_serve: Any = None
        if self.cfg.kiwix.enabled:
            self._add_kiwix_tools()
        if self.cfg.n8n.enabled:
            from valdar.tools.n8n import add_n8n_tools

            add_n8n_tools(self.registry, self.cfg.n8n)
        self.agent = Agent(self.cfg, self.llm, self.registry, self.heart, self.lock,
                           self.facts, self.self_model, self.world_lines, self.urges,
                           memory=self.memory, extras=self.extra_blocks)
        self.tool_ctx.on_recall = self.agent._reinstate
        self.block_providers: list[Any] = [self._relation_block, self._knowledge_block,
                                           self._thoughts_block]
        self.night = Night(self.cfg.night, self.llm, self.facts, self.memory,
                           path(self.cfg.night.out_dir), self.self_model,
                           self._owner() or "olivier")
        self.events: queue.Queue[Event] = queue.Queue()
        self._thinking = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._speaking = threading.Lock()

    # ============================================================ Kiwix
    def _add_kiwix_tools(self) -> None:
        from valdar.knowledge.kiwix import Kiwix
        from valdar.tools.registry import SAFE, Tool, params

        kc = self.cfg.kiwix
        self.kiwix = Kiwix(f"http://127.0.0.1:{kc.port}")
        s = {"type": "string"}

        def chercher(question: str, bibliotheque: str = "") -> str:
            try:
                hits = self.kiwix.search(question, bibliotheque or None, k=5)
            except Exception as exc:
                return f"bibliothèque hors ligne injoignable : {exc}"
            if not hits:
                return "rien trouvé dans les bibliothèques hors ligne."
            return "\n".join(f"- [{h['book']}] {h['title']} (chemin : {h['path']}) : "
                             f"{h['extrait']}" for h in hits)

        def lire(chemin: str) -> str:
            try:
                return self.kiwix.read(chemin, kc.max_chars)
            except Exception as exc:
                return f"lecture impossible : {exc}"

        self.registry.add(Tool(
            "chercher_savoir",
            "Cherche dans tes bibliothèques hors ligne (Wikipédia, Vikidia pour les enfants, "
            "médecine, bricolage, électronique, impression 3D, jardinage, parents, Stack "
            "Overflow…). À utiliser pour vérifier un fait à la source au lieu de deviner.",
            params(["question"], question={**s, "description": "mots-clés"},
                   bibliotheque={**s, "description": "nom d'une bibliothèque (optionnel)"}),
            chercher, SAFE, "savoir"))
        self.registry.add(Tool(
            "lire_article", "Lit un article trouvé par chercher_savoir (son chemin).",
            params(["chemin"], chemin={**s, "description": "le chemin donné par la recherche"}),
            lire, SAFE, "savoir"))

    def start_kiwix(self) -> str:
        """Lance kiwix-serve sur les bibliothèques téléchargées (s'il y en a)."""
        from pathlib import Path

        from valdar.knowledge.kiwix import KiwixServe

        kc = self.cfg.kiwix
        base = self.cfg.repo_path(kc.dir)
        exe = next(iter(sorted(Path(base, "bin").glob("**/kiwix-serve*"))), None)
        if self.kiwix is not None and self.kiwix.available():
            return "bibliothèques hors ligne déjà servies"
        if exe is None:
            return "Kiwix pas installé (tools\\valdar_kiwix.bat)"
        self._kiwix_serve = KiwixServe(exe, Path(base), kc.port)
        n = len(self._kiwix_serve.zims())
        if not self._kiwix_serve.start():
            return "aucune bibliothèque téléchargée (tools\\valdar_kiwix.bat)"
        return f"{n} bibliothèque(s) hors ligne"

    def kiwix_status(self) -> str:
        if self.kiwix is None:
            return "bibliothèques hors ligne désactivées"
        try:
            books = self.kiwix.books()
        except Exception:
            return "bibliothèques hors ligne : aucune servie (tools\\valdar_kiwix.bat)"
        return "bibliothèques hors ligne : " + ", ".join(b["title"] for b in books)

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
        for extra in (self.load.line(), self.ambient.line() if self.ambient else ""):
            if extra:
                lines.append(extra)
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
        if (ev.image and self.cfg.printwatch.triage
                and ev.kind in ("alerte", "pause", "pause_faite")):
            self._triage_async(ev)
            return       # Valdar parlera avec ce qu'il a vu
        if ev.kind in ("pause", "pause_faite", "question", "telemetrie"):
            self._speak_async(f"ta vigie d'impression vient de te dire : « {ev.text} ». "
                              "Préviens Olivier, à ta façon.", {"vigie": ev.kind})

    def _triage_async(self, ev: WatchEvent) -> None:
        """Gemma regarde l'image hors du fil de la vigie (elle ne doit jamais attendre)."""
        from valdar.printwatch.triage import Triage

        def run() -> None:
            with self._triage_lock:
                self._triaging += 1
            try:
                body()
            finally:
                with self._triage_lock:
                    self._triaging -= 1

        def body() -> None:
            res = Triage(self.llm, self.knowledge).assess(ev.image or b"", ev.text)
            if res is not None and self.printwatch is not None:
                self.printwatch.record_triage(ev.job, res.to_dict())
            seen = res.text() if res is not None else "je n'ai pas réussi à analyser l'image"
            self.events.put(Event("vigie_triage", seen, {"job": ev.job, "kind": ev.kind}))
            if ev.kind != "alerte" or (res is not None and res.severity >= 4):
                self._speak_async(f"ta vigie d'impression vient de te dire : « {ev.text} ». "
                                  f"Tu as regardé l'image : {seen}. Préviens Olivier, à ta "
                                  "façon, sans dramatiser ni minimiser.", {"vigie": ev.kind})

        threading.Thread(target=run, name="valdar-tri", daemon=True).start()

    def _knowledge_block(self, query: str) -> list[str]:
        if not self.cfg.context.auto_knowledge:
            return []          # à la demande (outil connaissance_impression)
        hits = self.knowledge.search(query, k=3)
        if not hits:
            return []
        return ["CE QUE TU SAIS SUR LE SUJET (tes connaissances, cite-les si utile) :\n"
                + "\n".join("- " + h.line() for h in hits)]

    def _relation_block(self, query: str) -> list[str]:
        b = self.relations.block(self._who)
        return [b] if b else []

    def _owner(self) -> str:
        return (self.cfg.agent.console_identity.person if self.cfg.agent else "") or ""

    def _thoughts_block(self, query: str) -> list[str]:
        if (self._who.person or "") != self._owner():
            return []          # ses pensées de fond sont privées : pas devant un invité
        b = self.thoughts.block()
        return [b] if b else []

    def _thought_context(self) -> str:
        now = time.time()
        with self.lock:
            feeling = feeling_block(self.heart, self.cfg.expression, self.urges())
        owner = self._owner()   # la pensée de fond ne rumine que la vie d'Olivier et la sienne
        wm = self.memory.working_memory(now, person=owner)
        recent = self.memory.last_thread(now, turns=4, person=owner)
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

    def streams(self) -> int:
        """Nombre de flux de pensée actifs en ce moment."""
        return (int(self.agent.busy.is_set()) + int(self._thinking.locked())
                + int(self._speaking.locked()) + self._triaging)

    def _feel_load(self) -> None:
        """Intéroception : recalcule la charge et prévient quand Valdar sature."""
        if not self.cfg.interoception.enabled:
            return
        before = self.load.level
        level = self.load.update(self.streams(), self.nociception.values.get("gpu_mem"))
        if level != before:
            self.events.put(Event("interoception", f"charge mentale : {self.load.label}",
                                  {"level": level, "value": round(self.load.value, 2),
                                   "parts": dict(self.load.parts)}))

    def make_ambient(self, frame_seconds: float, speaking: Any = None) -> Any:
        """Crée la voie basse de l'audition (appelée quand le micro démarre)."""
        from valdar.ears.ambient import Ambient

        def fire(name: str, scale: float, source: str) -> None:
            with self.lock:
                self.heart.fire(name, scale=scale, source=source)
            if name == self.cfg.ambient.startle_stimulus:
                lvl = self.ambient.last_level if self.ambient is not None else None
                self.events.put(Event("sursaut", "un bruit soudain m'a fait sursauter"
                                      + (f" ({lvl:.0f} dB)" if lvl is not None else ""),
                                      {"scale": round(scale, 2), "source": source}))

        self.ambient = Ambient(self.cfg.ambient, frame_seconds, fire, speaking)
        return self.ambient

    def _maybe_think(self, now: float) -> None:
        if self.nociception.danger:   # réflexe : ne pas charger une machine en souffrance
            return
        if self.load.level >= 1:      # tête pleine : la pensée de fond attend
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
            if self.cfg.llm.warm_cache:   # la pensée a pris le cache : on le refait
                self.agent.warm()

        threading.Thread(target=run, name="valdar-pensee", daemon=True).start()

    def _maybe_sleep_work(self, now: float) -> None:
        """La nuit : nettoyer, auditer, résumer, consolider (une fois par sommeil)."""
        with self.lock:
            awake = self.heart.awake
        idle = now - self.agent.last_activity
        if self.agent.busy.is_set() or not self.night.due(now, awake, idle):
            return
        if not self._thinking.acquire(blocking=False):   # pas en même temps que la pensée
            return

        def run() -> None:
            try:
                rep = self.night.run(now)
                self.events.put(Event("nuit", rep.text(), {"cle": rep.key,
                                                           "fichier": rep.path}))
            except Exception as exc:    # la nuit ne doit jamais faire tomber Valdar
                self.events.put(Event("erreur", f"nuit : {exc}"))
            finally:
                self._thinking.release()
            if self.cfg.llm.warm_cache:
                self.agent.warm()

        threading.Thread(target=run, name="valdar-nuit", daemon=True).start()

    def _extreme(self) -> str:
        """Motif de quarantaine si Valdar vit l'échange dans un état extrême, sinon ""."""
        if not self.cfg.critique.enabled:
            return ""
        if self.nociception.danger:
            return "vécu pendant une douleur au seuil de danger"
        with self.lock:
            b = self.heart.brief()
        if b["intensity"] >= self.cfg.critique.extreme_intensity:
            return f"vécu dans une émotion extrême ({b['emotion']})"
        return ""

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
            done = None
            if idea is not None:
                event["idea"] = idea.text
                done = lambda iid=idea.id: self.thoughts.mark(iid, "proposee")  # noqa: E731
            self._speak_async(_initiative_reason(event, idea), {"initiative": event}, done)
        self._feel_body(now)
        self._feel_load()
        self._maybe_think(now)
        self._maybe_sleep_work(now)
        if self.printwatch is not None and self._thread is None:
            pass   # en mode manuel (tests), la vigie est avancée par l'appelant
        for r in self.reminders.pop_due(now):
            self.events.put(Event("rappel", r["texte"], {"rappel": r}))

    def _speak_async(self, reason: str, data: dict[str, Any],
                     on_done: Any = None) -> None:
        if not self._speaking.acquire(blocking=False):
            return  # il parle déjà de lui-même : on n'empile pas (l'idée reste en stock)

        def run() -> None:
            try:
                reply = self.agent.spontaneous(reason)
                kind = "erreur" if reply.error else "initiative"
                self.events.put(Event(kind, reply.text, data))
                if on_done is not None and not reply.error:
                    on_done()
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
        if self.cfg.llm.warm_cache:      # Gemma chargé et la partie stable lue d'avance
            threading.Thread(target=self.agent.warm, name="valdar-cache", daemon=True).start()
        if self.cfg.embeddings.backend == "ollama":
            threading.Thread(target=self._start_sense, name="valdar-sens", daemon=True).start()
        if self.printwatch is not None:
            self.printwatch.start()

    def _start_sense(self) -> None:
        """Lance l'Ollama des plongements (processeur seul) puis vectorise en arrière-plan."""
        ec = self.cfg.embeddings
        if not ec.url and self.persist:
            from valdar.memory.sidecar import Sidecar

            self._sidecar = Sidecar(ec.sidecar_port)
            if not self._sidecar.start():
                self.embedder.disable(f"serveur des plongements : {self._sidecar.error}")
                self.events.put(Event("erreur", "sens : " + self._sidecar.error
                                      + " (je garde mes vecteurs maison)"))
                return
        self._backfill_loop()

    def _backfill_loop(self) -> None:
        """Vectorise peu à peu (vrai modèle de sens) connaissances et souvenirs, sur le
        processeur, sans jamais gêner la conversation : lots courts, pause entre deux."""
        ec = self.cfg.embeddings
        while not self._stop.is_set():
            done, busy = 0, self.agent.busy.is_set()
            try:
                if not busy:
                    done = self.knowledge.backfill(limit=ec.batch * 4)
                    done += self.memory.backfill(limit=ec.batch * 4)
            except Exception as exc:     # le sens est un plus : jamais une panne
                self.events.put(Event("erreur", f"vectorisation : {exc}"))
            # du travail en retard : on enchaîne vite ; sinon, un coup d'œil de temps en temps
            self._stop.wait(1.0 if busy or done else ec.backfill_every_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._sidecar is not None:
            self._sidecar.stop()
        if self._kiwix_serve is not None:
            self._kiwix_serve.stop()
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
    def handle(self, text: str, who: Identity | None = None, on_text: Any = None) -> Reply:
        with self._handling:    # un seul interlocuteur traité à la fois (contexte des outils)
            return self._handle(text, who, on_text)

    def _handle(self, text: str, who: Identity | None, on_text: Any = None) -> Reply:
        who = who or (self.cfg.agent.console_identity if self.cfg.agent else Identity())
        self._who = who
        self.tool_ctx.person = who.person or ""   # les outils agissent pour celui qui parle
        seen = self.relations.observe(who, text)
        with self.lock:
            self.initiative.on_user_message()
            if seen["new_encounter"] and seen["person"] is not None:
                self.relations.feel_presence(self.heart, seen["person"])
        t0 = time.time()
        before = self._extreme()
        reply = self.agent.handle(text, who, on_text=on_text)
        motif = before or self._extreme()
        if motif:     # vécu à isoler avant la nuit (avenant 5 §4.3)
            self.memory.quarantine_window(t0, time.time(), motif)
        if who.person and "oublie_moi" in reply.tools_used:
            self.relations.forget_person(who.person)   # ce dernier message compris
        return reply


def _initiative_reason(event: dict[str, Any], idea: Any = None) -> str:
    if idea is not None:
        return (f"tu as eu une idée en pensant de ton côté : « {idea.text} » ({idea.why}). "
                "Propose-la à Olivier, simplement, comme une idée en l'air qu'il peut refuser.")
    if event["kind"] == "talk":
        return ("ça fait un moment que personne ne t'a parlé et tu as envie de parler à Olivier "
                "(dis-le à ta façon, en restant léger).")
    return ("tu t'ennuies un peu et tu as envie de découvrir ou de faire quelque chose de "
            "nouveau ; propose une idée concrète à Olivier.")
