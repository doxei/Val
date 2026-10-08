"""Agent de Valdar : dialogue + outils, piloté par le cœur (avenant 2 §2 et §5).

À chaque message :
1. le cœur l'entend (interaction, évaluation rapide → stimuli) ;
2. le prompt est composé à partir de l'état réel du cœur, de la personne et des souvenirs ;
3. le modèle répond ou appelle des outils (permissions selon la personne) ;
4. le résultat des outils touche le cœur (réussite, échec) et chaque réponse coûte de l'énergie.
"""
from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from valdar.appraisal import FastAppraisal, normalize
from valdar.config.loader import Identity, ValdarConfig
from valdar.expression.compose import (
    gen_params,
    memories_block,
    prompt_parts,
    thread_block,
    with_context,
)
from valdar.heart import Heart
from valdar.llm.backend import ChatResult, GenParams, LLMBackend, LLMError
from valdar.memory import Episodic, Facts
from valdar.tools.registry import Registry, decide

FORGET_TOOLS = ("oublie_moi",)   # après eux, rien de la personne ne doit rester
_YES = re.compile(r"^\s*(oui|ok|vas[- ]y|confirme|go|d'accord|c'est bon)\b")
# Pour une action irréversible : la réponse doit être un oui net, et rien d'autre.
_STRICT_YES = re.compile(r"^\s*(oui|confirme|oui confirme|oui vas[- ]y)\s*[.!]*\s*$")
_NO = re.compile(r"\b(non|annule|attends|pas|stop|arrete)\b")
PENDING_TTL = 120.0   # une demande de confirmation expire au bout de 2 minutes


@dataclass
class Reply:
    text: str
    tools_used: list[str] = field(default_factory=list)
    stimuli: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None


class Agent:
    def __init__(
        self,
        cfg: ValdarConfig,
        llm: LLMBackend,
        registry: Registry,
        heart: Heart,
        heart_lock: threading.RLock,
        facts: Facts,
        self_model: dict[str, Any],
        world: Callable[[], list[str]],
        urges: Callable[[], dict[str, float]],
        memory: Episodic | None = None,
        extras: Callable[[str], list[str]] | None = None,
    ):
        if cfg.expression is None or cfg.agent is None:
            raise ValueError("configuration incomplète : sections expression et agent requises")
        self.cfg = cfg
        self.llm = llm
        self.registry = registry
        self.heart = heart
        self.lock = heart_lock
        self.facts = facts
        self.self_model = self_model
        self.world = world
        self.urges = urges
        self.fast = FastAppraisal(cfg.appraisal_fast)
        self.history: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self._llm_lock = threading.Lock()
        self._world_sent: tuple[str, ...] | None = None   # état du monde déjà donné
        self._world_at = 0.0
        self._on_text: Any = None       # streaming : le texte part vers la voix au fil de l'eau
        self.memory = memory
        self.extras = extras
        self.thread: dict[str, Any] | None = None
        self.busy = threading.Event()        # un échange est en cours (pensées de fond : attendre)
        self._turn = threading.RLock()       # un seul échange à la fois (dialogue ou initiative)
        self.last_activity = time.time()
        if memory is not None:
            self.thread = memory.last_thread(
                person=cfg.agent.console_identity.person or "")
            self._preload()

    def _preload(self) -> None:
        """Après un redémarrage récent, la conversation reprend où elle s'était arrêtée."""
        th, ec = self.thread, self.cfg.episodic
        if not th or time.time() - th["ended"] > ec.preload_hours * 3600:
            return
        for t in th["turns"][-ec.preload_turns:]:
            role = "assistant" if t["speaker"] == "Valdar" else "user"
            self.history.append({"role": role, "content": t["text"]})
        while self.history and self.history[0]["role"] == "assistant":
            self.history.pop(0)

    # ================================================================ entrée
    def handle(self, text: str, who: Identity | None = None, on_text: Any = None) -> Reply:
        """`on_text(morceau)` : reçoit la réponse au fil de l'eau (streaming)."""
        who = who or self.cfg.agent.console_identity
        text = text.strip()
        self.busy.set()
        self._turn.acquire()
        self._on_text = on_text
        try:
            with self.lock:
                self.heart.interact()
                stimuli = self.fast.appraise(text)
                for name, scale in stimuli:
                    self.heart.fire(name, scale=scale, source="voie_basse")
            self._remember(_speaker(who), text, who)

            reply = None
            if self.pending is not None:
                if time.time() - self.pending.get("asked_at", 0.0) > PENDING_TTL:
                    self.pending = None           # trop tard : la question ne tient plus
                elif who.person != self.pending["person"]:
                    self.pending = None           # quelqu'un d'autre parle : on annule, puis on
                                                  # traite normalement son message
                else:
                    reply = self._resolve_pending(text, who, stimuli)
            if reply is None:
                self.history.append({"role": "user", "content": text})
                reply = self._loop(who, query=text)
                reply.stimuli = stimuli
            if any(t in FORGET_TOOLS for t in reply.tools_used):
                self.forget_session()         # ce qui a été dit avant ne doit pas resurgir
            elif not reply.error:
                self._remember("Valdar", reply.text, who)
            return reply
        finally:
            self._on_text = None
            self._turn.release()
            self.last_activity = time.time()
            self.busy.clear()

    def forget_session(self) -> None:
        """Après un « oublie-moi » : rien de la conversation en cours ne doit resurgir."""
        self.history.clear()
        self.thread = None
        self.pending = None

    def _remember(self, speaker: str, text: str, who: Identity) -> None:
        if self.memory is None:
            return
        with self.lock:
            pad = self.heart.pad
            pad_t = (pad["P"], pad["A"], pad["D"])
        with contextlib.suppress(Exception):   # la mémoire ne fait jamais tomber l'échange
            self.memory.log_turn(speaker, text, pad=pad_t, person=who.person or "")

    def spontaneous(self, reason: str, who: Identity | None = None) -> Reply:
        """Valdar prend la parole de lui-même (initiative, rappel)."""
        who = who or self.cfg.agent.console_identity
        with self._turn:
            self.pending = None    # une confirmation ne survit pas à un changement de sujet
            self.history.append({
                "role": "user",
                "content": (f"[note interne, personne n'a parlé] Tu prends la parole de "
                            f"toi-même : {reason} Une ou deux phrases naturelles, sans "
                            "insister ni culpabiliser."),
            })
            reply = self._loop(who, query=reason, allow_tools=False)
            if not reply.error:
                self._remember("Valdar", reply.text, who)
            return reply

    # ============================================================ boucle LLM
    def _loop(self, who: Identity, query: str, allow_tools: bool = True) -> Reply:
        used: list[str] = []
        rounds = self.cfg.llm.max_tool_rounds
        # Calculé une fois par échange : le même contexte d'un tour d'outil à l'autre, pour
        # qu'Ollama réutilise son cache.
        system, context, params = self._compose(who, query)
        for _ in range(rounds):
            tools = self._tool_schemas(who) if allow_tools else None
            try:
                stream = ({"on_token": self._on_text}
                          if self._on_text is not None and getattr(self.llm, "supports_stream",
                                                                   False) else {})
                with self._llm_lock:
                    result = self.llm.chat(with_context(self._window(), context),
                                           system=system, tools=tools, params=params,
                                           **stream)
            except LLMError as exc:
                self._feel("tool_failure")
                if self.history and self.history[-1]["role"] == "user":
                    self.history.pop()   # on pourra reposer la question proprement
                return Reply(text="mon cerveau ne répond pas (le modèle de langage est "
                                  f"injoignable) : {exc}", tools_used=used, error=str(exc))
            with self.lock:
                self.heart.spend_energy(self.cfg.expression.energy_cost_per_call)

            if not result.tool_calls:
                text = result.content or "…"
                self.history.append({"role": "assistant", "content": text})
                self._trim()
                return Reply(text=text, tools_used=used)

            self.history.append(_assistant_with_calls(result))
            for call in result.tool_calls:
                out = self._run_tool(call.name, call.arguments, who)
                if out is None:   # confirmation demandée : on s'arrête là
                    tool = self.registry.get(call.name)
                    what = f"« {call.name} »"
                    if tool is not None and tool.describe is not None:
                        with contextlib.suppress(Exception):
                            what = tool.describe(call.arguments)
                    ask = (f"Tu confirmes : {what} ? Dis « oui » pour lancer, n'importe "
                           "quoi d'autre pour annuler.")
                    self.history.append({"role": "assistant", "content": ask})
                    return Reply(text=ask, tools_used=used)
                used.append(call.name)
                self.history.append({"role": "tool", "content": out[:2000],
                                     "tool_name": call.name})
        text = "j'ai perdu le fil, redis-moi ça autrement."
        self.history.append({"role": "assistant", "content": text})
        return Reply(text=text, tools_used=used)

    def _compose(self, who: Identity, query: str):
        world = self._world_now()  # peut interroger l'imprimante : jamais sous le verrou du cœur
        facts: list[dict[str, Any]] = []
        if self.cfg.context.auto_facts:
            facts = self.facts.context(query, person=who.person or "")
            self.facts.touch([f["id"] for f in facts])
        blocks = self._memory_blocks(query, who)
        if self.extras is not None:
            with contextlib.suppress(Exception):
                blocks += [b for b in self.extras(query) if b]
        with self.lock:
            urges = self.urges()
            system, context = prompt_parts(self.cfg, self.self_model, self.heart, who, facts,
                                           world, urges, extra_blocks=blocks)
            params = gen_params(self.cfg.expression, self.heart.sources())
        return system, context, params

    def _world_now(self, now: float | None = None) -> list[str]:
        """L'heure à chaque tour ; le reste du monde au premier tour, puis une fois par
        `world_every_seconds`, ou tout de suite s'il a changé (imprimante, rappel, corps)."""
        lines = self.world()
        if not lines:
            return lines
        now = time.time() if now is None else now
        clock, rest = lines[0], tuple(lines[1:])
        due = (self._world_sent is None or rest != self._world_sent
               or now - self._world_at >= self.cfg.context.world_every_seconds)
        if not due:
            return [clock]
        self._world_sent, self._world_at = rest, now
        return lines

    def _memory_blocks(self, query: str, who: Identity | None = None) -> list[str]:
        if self.memory is None:
            return []
        now = time.time()
        if not self.cfg.context.auto_memory:
            # à la demande (outil fouiller_memoire) ; seul le fil de la dernière conversation
            # revient de lui-même au réveil, comme quand on se réveille
            if self.thread and len(self.history) <= 4 and who is not None and \
                    who.person == self.cfg.agent.console_identity.person:
                b = thread_block(self.thread, now)
                return [b] if b else []
            return []
        with self.lock:
            pad = self.heart.pad
            pad_t = (pad["P"], pad["A"], pad["D"])
        try:
            person = who.person if who is not None else None
            recalled = self.memory.recall(query, now=now, pad=pad_t, person=person)
            wm = self.memory.working_memory(now, exclude=[r.turn for r in recalled],
                                            person=person)
        except Exception:
            return []
        self._reinstate(recalled)
        blocks = [memories_block(recalled, now, "SOUVENIRS QUI TE REVIENNENT (ta mémoire est "
                                 "reconstruite : si c'est important, dis-le avec prudence) :"),
                  memories_block(wm, now, "CE QUI TE TROTTE ENCORE DANS LA TÊTE :")]
        if self.thread and len(self.history) <= 4 and who is not None and \
                who.person == self.cfg.agent.console_identity.person:
            blocks.insert(0, thread_block(self.thread, now))
        return [b for b in blocks if b]

    def _reinstate(self, recalled: list) -> None:
        """Repenser à un moment fort le fait un peu revivre (avenant 4 §2)."""
        r = self.cfg.episodic.reinstate
        if not recalled:
            return
        top = recalled[0]
        p = top.pad[0]
        if abs(p) < r.threshold or not r.gain:
            return
        impulses = r.positive if p > 0 else r.negative
        with self.lock:
            self.heart.apply(impulses, amp=abs(p) * r.gain)

    def warm(self, who: Identity | None = None) -> bool:
        """Réchauffe le cache d'Ollama : fait relire au modèle la partie stable (personnalité,
        outils, conversation en cours) quand personne ne parle, pour que le prochain message
        ne paie que ses propres jetons. Utile après une pensée de fond ou au démarrage, qui
        ont remplacé ce cache. Ne touche pas à l'historique."""
        if self.busy.is_set() or not self._llm_lock.acquire(blocking=False):
            return False
        try:
            who = who or self.cfg.agent.console_identity
            system, context, params = self._compose(who, "")
            msgs = with_context(self._window() + [{"role": "user", "content": "…"}], context)
            self.llm.chat(msgs, system=system, tools=self._tool_schemas(who),
                          params=GenParams(temperature=0.0, max_tokens=1))
            return True
        except Exception:
            return False
        finally:
            self._llm_lock.release()

    def _tool_schemas(self, who: Identity) -> list[dict[str, Any]]:
        perms = self.cfg.permissions
        return self.registry.schemas(lambda t: decide(t, who, perms).allowed)

    def _run_tool(self, name: str, args: dict[str, Any], who: Identity) -> str | None:
        tool = self.registry.get(name)
        if tool is None:
            return f"outil inconnu : {name}"
        decision = decide(tool, who, self.cfg.permissions)
        if not decision.allowed:
            return f"refusé : {decision.reason}"
        if decision.needs_confirmation:
            self.pending = {"tool": name, "args": args, "person": who.person,
                            "asked_at": time.time(),
                            "strict": name in FORGET_TOOLS or getattr(tool, "confirm", False)}
            return None
        return self._execute(tool, args)

    def _execute(self, tool, args: dict[str, Any]) -> str:
        try:
            out = str(tool.fn(**args))
        except TypeError as exc:
            self._feel("tool_failure")
            return f"mauvais arguments pour {tool.name} : {exc}"
        except Exception as exc:  # un outil ne doit jamais faire tomber Valdar
            self._feel("tool_failure")
            return f"échec de {tool.name} : {exc}"
        self._feel("tool_success")
        return out

    def _resolve_pending(self, text: str, who: Identity,
                         stimuli: list[tuple[str, float]]) -> Reply:
        p, self.pending = self.pending, None
        if p is None:
            return Reply(text="rien en attente.")
        said = normalize(text)
        ok = (_STRICT_YES.match(said) is not None if p.get("strict")
              else _YES.search(said) is not None and not _NO.search(said))
        if who.person != p["person"] or not ok:
            msg = f"ok, j'annule « {p['tool']} »."
            self.history.append({"role": "assistant", "content": msg})
            return Reply(text=msg, stimuli=stimuli)
        tool = self.registry.get(p["tool"])
        out = self._execute(tool, p["args"]) if tool else "outil disparu."
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "tool", "content": out[:2000], "tool_name": p["tool"]})
        reply = self._loop(who, query=text)
        reply.tools_used.insert(0, p["tool"])
        reply.stimuli = stimuli
        return reply

    # ================================================================ divers
    def _feel(self, which: str) -> None:
        ref = getattr(self.cfg.agent, which)
        with self.lock:
            self.heart.fire(ref.stimulus, scale=ref.scale, source="outil")

    def _window(self) -> list[dict[str, Any]]:
        n = self.cfg.llm.history_messages
        win = self.history[-n:]
        while win and win[0]["role"] == "tool":   # jamais un résultat d'outil orphelin
            win = win[1:]
        return win

    def _trim(self) -> None:
        keep = self.cfg.llm.history_messages * 2
        if len(self.history) > keep:
            self.history = self.history[-keep:]


def _speaker(who: Identity) -> str:
    return who.name or (who.person or "quelqu'un").capitalize()


def _assistant_with_calls(result: ChatResult) -> dict[str, Any]:
    return {"role": "assistant", "content": result.content or "",
            "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}}
                           for c in result.tool_calls]}
