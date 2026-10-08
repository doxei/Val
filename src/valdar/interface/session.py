"""Une session de Valdar : le runtime, sa voix, ses oreilles, son œil, et qui lui parle.

Tout ce qui se passe est publié sur un **bus** (`publish(kind, data)`) : la console l'affiche,
l'interface le reçoit en direct. Une seule façon de répondre (`answer`), qu'on lui écrive au
clavier, dans l'interface, ou qu'on lui parle.
"""
from __future__ import annotations

import contextlib
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from valdar.config.loader import Identity, ValdarConfig
from valdar.relations.foyer import Foyer
from valdar.voice.stream import SentenceStream

Listener = Callable[[str, dict[str, Any]], None]

SCOLD = ["{nom}, descends de la table !", "{nom} ! Descends, tout de suite.",
         "Eh, {nom}, la table c'est pas pour toi. Descends !", "{nom}, en bas !"]


class Bus:
    def __init__(self) -> None:
        self._subs: list[Listener] = []
        self._lock = threading.Lock()
        self.log: list[dict[str, Any]] = []        # derniers messages (nouvel onglet ouvert)

    def subscribe(self, fn: Listener) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def off() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._subs.remove(fn)
        return off

    def publish(self, kind: str, data: dict[str, Any] | None = None) -> None:
        msg = {"kind": kind, "data": data or {}, "t": time.time()}
        with self._lock:
            subs = list(self._subs)
            if kind in ("user", "reply", "event", "heard", "spontaneous"):
                self.log.append(msg)
                del self.log[:-200]
        for fn in subs:
            with contextlib.suppress(Exception):
                fn(kind, msg["data"])


class Session:
    def __init__(self, cfg: ValdarConfig, rt: Any, speaker: Any = None):
        self.cfg = cfg
        self.rt = rt
        self.speaker = speaker
        self.bus = Bus()
        self.muted = False
        owner = cfg.agent.console_identity if cfg.agent else Identity()
        self.owner = owner
        self.foyer = Foyer(cfg.storage_path("foyer.json"), owner)
        self.prints: Any = None
        self.voice_id: Any = None
        self.face_id: Any = None
        self.watch: Any = None
        self.gate: Any = None
        self.mic: Any = None
        self.enrolling: Any = None
        self.last_who: Identity = owner
        self._stop = threading.Event()
        rt.block_providers.append(self._home_block)
        self._threads: list[threading.Thread] = []

    # ============================================================ démarrage
    def start(self) -> None:
        self.rt.start()
        self._spawn(self._pump_events, "valdar-bus")
        self._spawn(self._pulse, "valdar-pouls")

    def _spawn(self, fn: Callable[[], None], name: str) -> None:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        if self.watch is not None:
            self.watch.stop()
        if self.mic is not None:
            with contextlib.suppress(Exception):
                self.mic.stop()
        if self.speaker is not None:
            self.speaker.close()
        self.rt.stop()

    # ============================================================ parole
    def say(self, text: str) -> None:
        if self.speaker is None or self.muted:
            return
        if self.speaker.ready.is_set() and self.speaker.load_error:
            return
        self.speaker.say(text)

    def answer(self, text: str, who: Identity | None = None, source: str = "clavier",
               show: Callable[[str], None] | None = None) -> Any:
        """Répond en streaming : chaque phrase part vers la voix dès qu'elle est complète."""
        who = who or self.owner
        self.last_who = who
        self.bus.publish("user", {"text": text, "who": who.name, "person": who.person,
                                  "confidence": who.confidence, "source": source})
        if self.speaker is not None:
            self.speaker.interrupt()          # on lui parle : il se tait

        def piece(p: str) -> None:
            self.bus.publish("token", {"text": p})
            if show is not None:
                show(p)

        stream = SentenceStream(self.say, piece)
        reply = self.rt.handle(text, who, on_text=stream.feed)
        if stream.started:
            stream.close()
            extra = reply.text.strip()
            if extra and extra not in stream.text:      # ex. confirmation après un outil
                self.say(extra)
                if show is not None:
                    show("\n" + extra)
        else:
            self.say(reply.text)
            if show is not None:
                show(reply.text)
        self.bus.publish("reply", {"text": reply.text, "tools": reply.tools_used,
                                   "error": reply.error, "to": who.name})
        return reply

    # ============================================================ oreilles
    def attach_ears(self, gate: Any, mic: Any) -> None:
        self.gate, self.mic = gate, mic

    def who_spoke(self, audio: Any) -> tuple[Identity, dict[str, Any]]:
        """Qui vient de parler : la voix d'abord, le visage pour confirmer ou départager."""
        info: dict[str, Any] = {"voix": None, "visage": None}
        guess = None
        if self.voice_id is not None and audio is not None:
            guess = self.voice_id.identify(audio)
            info["voix"] = guess.to_dict()
        now = time.time()
        seen = [p for p, t in (self.watch.seen.items() if self.watch else [])
                if now - t <= 15.0]
        info["visage"] = seen
        if guess is not None and guess.sure:
            conf = self.voice_id.confidence(guess)
            if guess.person in seen:
                conf = 0.95                  # voix et visage d'accord : il en est sûr
            return self.foyer.identity(guess.person, conf), info
        if len(seen) == 1:                   # voix incertaine, une seule personne en vue
            return self.foyer.identity(seen[0], 0.6), info
        if guess is not None and self.prints is not None and self.prints.counts("voix"):
            return Identity(person=None, name="quelqu'un (voix non reconnue)",
                            role="unknown", confidence=0.0), info
        return self.owner, info              # rien d'enrôlé : comme au clavier

    def on_heard(self, h: Any, show: Callable[[str], None] | None = None) -> Any:
        who, info = self.who_spoke(getattr(h, "audio", None))
        self.bus.publish("heard", {"text": h.text, "who": who.name, "person": who.person,
                                   "confidence": who.confidence, "ident": info})
        return self.answer(h.text, who, source="voix", show=show)

    # ------------------------------------------------------------ enrôlement
    def enroll_voice(self, person: str) -> str:
        from valdar.ident.voices import Enrollment

        if self.voice_id is None or self.gate is None:
            return "il me faut le micro et le modèle de voix (onglet Réglages)."
        m = self.foyer.member(person)
        if m is None:
            return "personne inconnue dans le foyer."
        self.enrolling = Enrollment(person, self.cfg.ident.voice_enroll_seconds)

        def tap(seg: Any) -> bool:
            e = self.enrolling
            if e is None:
                return False
            done = e.add(seg)
            self.bus.publish("enroll", {"kind": "voix", "person": person,
                                        "progress": round(e.progress, 2)})
            if done:
                self.enrolling = None
                self.gate.tap = None
                n = self.voice_id.enroll(person, e.audio())
                self.bus.publish("enroll", {"kind": "voix", "person": person,
                                            "progress": 1.0, "done": True, "prints": n})
            return True

        self.gate.tap = tap
        self.bus.publish("enroll", {"kind": "voix", "person": person, "progress": 0.0})
        return (f"{m['nom']} : parle-moi normalement pendant une quinzaine de secondes, "
                "sans dire mon nom (ce que tu dis n'est pas transcrit).")

    def cancel_enroll(self) -> None:
        self.enrolling = None
        if self.gate is not None:
            self.gate.tap = None
        if self.watch is not None:
            self.watch.enroll = None

    def enroll_face(self, person: str) -> str:
        if self.watch is None or self.face_id is None:
            return "il me faut la caméra et les modèles de visage."
        m = self.foyer.member(person)
        if m is None:
            return "personne inconnue dans le foyer."

        def done(p: str, n: int) -> None:
            self.bus.publish("enroll", {"kind": "visage", "person": p, "progress": 1.0,
                                        "done": True, "prints": n})

        self.watch.enroll_done = done
        self.watch.enroll = (person, self.cfg.ident.face_enroll_shots)
        return f"{m['nom']} : regarde la caméra, bouge un peu la tête, une dizaine de secondes."

    def forget_prints(self, person: str) -> int:
        return self.prints.forget(person) if self.prints is not None else 0

    # ============================================================ œil
    def on_scene(self, scene: Any) -> None:
        self.bus.publish("scene", scene.to_dict())

    def on_dog_table(self) -> None:
        dog = self.foyer.dog()
        nom = dog["nom"] if dog else "le chien"
        line = random.choice(SCOLD).format(nom=nom)
        with self.rt.lock:
            self.rt.heart.fire("concern", scale=0.5, source="oeil")
        if self.speaker is not None and not self.muted:
            self.speaker.say(line)
        self.bus.publish("event", {"kind": "chien", "text": f"{nom} sur la table : « {line} »"})

    # ============================================================ contexte
    def _home_block(self, query: str) -> list[str]:
        lines = []
        fb = self.foyer.block()
        if fb:
            lines.append(fb)
        if self.watch is not None:
            now = time.time()
            here = []
            for pid, t in sorted(self.watch.seen.items(), key=lambda kv: -kv[1]):
                if now - t <= self.cfg.vision.seen_memory_seconds:
                    m = self.foyer.member(pid)
                    ago = int(now - t)
                    here.append(f"{m['nom'] if m else pid} (vu il y a "
                                f"{ago // 60} min)" if ago >= 60 else
                                f"{m['nom'] if m else pid} (là, maintenant)")
            dog = self.foyer.dog()
            if dog and now - self.watch.dog_seen <= self.cfg.vision.seen_memory_seconds:
                last = self.watch.last
                here.append(dog["nom"] + (" (sur la table !)" if last and last.dog_on_table
                                          else ""))
            if here:
                lines.append("TU VOIS DANS LA PIÈCE (ta caméra) : " + ", ".join(here))
        who = self.last_who
        if who.person and who.confidence < 0.7:
            lines.append(f"Tu crois parler à {who.name}, sans en être sûr : si c'est "
                         "important, demande gentiment.")
        return ["\n".join(lines)] if lines else []

    # ============================================================ flux
    def _pump_events(self) -> None:
        while not self._stop.is_set():
            try:
                ev = self.rt.events.get(timeout=0.5)
            except Exception:
                continue
            if ev.kind == "initiative":
                self.bus.publish("spontaneous", {"text": ev.text})
                self.say(ev.text)
            elif ev.kind == "rappel":
                self.bus.publish("event", {"kind": "rappel", "text": ev.text})
                self.say(f"Rappel : {ev.text}")
            else:
                self.bus.publish("event", {"kind": ev.kind, "text": ev.text})

    def _pulse(self) -> None:
        """L'état du cœur une fois par seconde ; la bouche 20 fois par seconde quand il parle."""
        last_state = 0.0
        while not self._stop.is_set():
            now = time.time()
            if self.speaker is not None and hasattr(self.speaker, "level"):
                lvl = self.speaker.level()
                if lvl > 0 or getattr(self.speaker, "playing", lambda: False)():
                    self.bus.publish("mouth", {"level": round(lvl, 3)})
            if now - last_state >= 1.0:
                last_state = now
                with contextlib.suppress(Exception):
                    self.bus.publish("state", self.state())
            self._stop.wait(0.05)

    def state(self) -> dict[str, Any]:
        rt = self.rt
        with rt.lock:
            s = rt.heart.sample()
            felt = rt.heart.felt()
            organs = rt.heart.organs()
        amb = rt.ambient
        gate = self.gate
        return {
            "emotion": s["emotion"], "intensity": round(s["intensity"], 3),
            "mood": s["mood_label"], "pad": {k: round(v, 3) for k, v in s["pad"].items()},
            "awake": s["awake"], "bpm": s["bpm"], "dominant": s["dominant"],
            "variables": {k: round(v, 3) for k, v in s["variables"].items()},
            "needs": {k: round(v, 3) for k, v in s["needs"].items()},
            "organs": {k: round(v, 3) for k, v in organs.items()},
            "felt": list(felt.values()),
            "load": {"level": rt.load.label, "value": round(rt.load.value, 2)},
            "ambient": None if amb is None or amb.base is None else
            {"mood": amb.mood(), "db": round(amb.base, 1)},
            "ears": None if gate is None else {"engaged": gate.engaged(), **gate.stats},
            "speaking": bool(self.speaker and self.speaker.speaking()),
            "pain": rt.nociception.lines(),
            "who": {"name": self.last_who.name, "confidence": self.last_who.confidence},
            "muted": self.muted,
            "silenced": bool(getattr(rt.initiative, "silenced", False)),
        }
