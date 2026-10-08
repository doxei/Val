"""Cœur de Valdar (cahier des charges v2, §5) : tourne en continu, sans LLM.

Contenu : neuromodulateurs (retour exact vers la base, couplages sur les écarts, bruit
d'Ornstein-Uhlenbeck), impulsions diffusées dans le temps, saturation douce, habituation,
contre-mouvement, besoins (contact, nouveauté, repos), systèmes de Panksepp, PAD de
l'émotion, étiquette d'émotion, humeur ALMA, organes virtuels, veille/sommeil,
persistance et rattrapage de l'absence.

Toutes les intégrations sont exactes pour un pas quelconque : 1 pas de 60 s ≈ 60 pas de 1 s.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

from valdar.config import ValdarConfig, load
from valdar.heart import mood as mood_mod
from valdar.heart import organs as organs_mod
from valdar.heart import systems
from valdar.heart.dynamics import (
    approach_one,
    clamp,
    decay,
    in_window,
    ou_noise,
    relax,
    soft_add,
)
from valdar.heart.journal import Journal
from valdar.heart.store import Store

STATE_KEY = "heart:state"
STATE_VERSION = 2
_MIN_IMPULSE = 1e-4


def local_hour(unix: float) -> float:
    t = time.localtime(unix)
    return t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0


class Heart:
    def __init__(
        self,
        config: ValdarConfig | None = None,
        profile: str | None = None,
        rng_seed: int | None = None,
        db_path: str | None = None,
        journal_path: str | None = None,
        anchor: float | None = None,
        persist: bool = True,
    ):
        self.cfg: ValdarConfig = config or load()
        self.hc = self.cfg.heart
        self.profile_name = profile or self.cfg.temperament.default_profile
        tp = self.cfg.temperament.profiles[self.profile_name]
        self.sensitivity = tp.sensitivity
        self.bases = {k: tp.bases.get(k, s.base) for k, s in self.hc.variables.items()}
        self.drive_rest = {k: tp.drive_rest.get(k, s.rest) for k, s in self.hc.drives.items()}
        self.default_mood = mood_mod.default_mood(
            self.cfg.temperament, self.profile_name, self.hc.mood.default_scale)

        self.seed = rng_seed if rng_seed is not None else self.hc.rng_seed
        self.rng = random.Random(self.seed)
        self.now: float = anchor if anchor is not None else time.time()

        self.variables: dict[str, float] = dict(self.bases)
        self.needs: dict[str, float] = {k: 0.0 for k in self.hc.needs}
        self.need_last: dict[str, float] = {k: self.now for k in self.hc.needs}
        self.drives: dict[str, float] = dict(self.drive_rest)
        self.mood: dict[str, float] = dict(self.default_mood)
        self.pending: list[dict[str, Any]] = []
        self.recent: dict[str, list[float]] = {}
        # Surveillance de l'humeur (avenant 3 §3.3-3.4) : échantillons du plaisir, temps en
        # moral bas par semaine (garde-fou 5).
        self.mood_samples: list[float] = []
        self._next_mood_sample = 0.0
        self.low_seconds: dict[str, float] = {}
        self.last_interaction: float = self.now
        self.wake_until: float = self.now
        self.awake: bool = self._is_awake(self.now)
        self.body: dict[str, float] = {}       # état des organes (avec inertie)
        self.body_rest: dict[str, float] = {}  # leur état au repos (référence du retour)

        self.persist = persist
        self.store: Store | None = None
        if persist:
            self.store = Store(db_path or self.cfg.storage_path(self.cfg.storage.db))
        self.journal = Journal(
            journal_path or self.cfg.storage_path(self.cfg.storage.journal),
            max_bytes=self.cfg.storage.journal_max_bytes,
            keep=self.cfg.storage.journal_keep,
            enabled=persist or journal_path is not None,
        )
        self._next_save = self.now + self.hc.save_every_seconds
        self._next_journal = self.now + self.hc.journal_state_every_seconds
        self._refresh()
        self.body_rest = organs_mod.activations(self.hc, self.sources())
        self.body = dict(self.body_rest)

    # ============================================================== dynamique
    def step(self, dt: float, quiet: bool = False) -> None:
        """Intègre exactement dt secondes (dt quelconque)."""
        if dt <= 0.0:
            return
        t0 = self.now
        self.now = t0 + dt
        self.awake = self._is_awake(self.now)

        self._deliver_pending(t0, dt)
        self._integrate_variables(dt)
        self._integrate_needs(t0, dt)
        self._integrate_drives(dt)
        self._refresh()
        self.body = organs_mod.integrate(self.hc, self.body, self.sources(), dt)
        self.mood = mood_mod.update(self.mood, self.pad, self.default_mood, self.hc.mood, dt,
                                    self.awake)
        self._watch_mood(t0, dt)

        if not quiet and self.now >= self._next_journal:
            self.journal.log("state", t=self.now, **self.brief())
            self._next_journal = self.now + self.hc.journal_state_every_seconds
        # En rattrapage (quiet), une seule sauvegarde à la fin : sous Windows, une écriture SQLite
        # par minute simulée faisait durer le réveil après 3 jours d'absence ~9 minutes.
        if not quiet and self.persist and self.now >= self._next_save:
            self.save()

    def advance(self, seconds: float, quiet: bool = False) -> None:
        """Avance de `seconds` par pas d'au plus max_time_step."""
        remaining = seconds
        while remaining > 1e-9:
            dt = min(self.hc.max_time_step, remaining)
            self.step(dt, quiet=quiet)
            remaining -= dt

    def _watch_mood(self, t0: float, dt: float) -> None:
        spec = self.hc.mood
        if mood_mod.is_low(self.mood, spec):
            # Un long rattrapage peut couvrir plusieurs semaines : chacune reçoit sa part.
            t, end = max(t0, self.now - dt), self.now
            while t < end:
                week = time.strftime("%G-S%V", time.localtime(t))
                lt = time.localtime(t)
                midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
                next_week = midnight + (7 - lt.tm_wday) * 86400
                chunk = min(end, next_week) - t
                if chunk <= 0:
                    break
                if week not in self.low_seconds and self.low_seconds:
                    last = sorted(self.low_seconds)[-1]
                    self.journal.log("mood_week", t=t, week=last,
                                     low_hours=round(self.low_seconds[last] / 3600, 2))
                self.low_seconds[week] = self.low_seconds.get(week, 0.0) + chunk
                t += chunk
            for old in sorted(self.low_seconds)[:-8]:   # 8 semaines d'historique
                del self.low_seconds[old]
        w = spec.watch
        if self.now >= self._next_mood_sample:
            self.mood_samples.append(self.mood["P"])
            del self.mood_samples[:-w.window]
            self._next_mood_sample = self.now + w.sample_seconds

    def sliding(self) -> bool:
        """« Je sens que je glisse » : l'humeur récupère plus lentement (autocorrélation
        haute) et descend. Signe avant-coureur d'un basculement (Scheffer et coll., 2009)."""
        xs = self.mood_samples
        w = self.hc.mood.watch
        if len(xs) < w.window // 2:
            return False
        falling = sum(xs[-6:]) / 6 < sum(xs[:6]) / 6 - 0.02
        return falling and mood_mod.lag1_autocorrelation(xs) >= w.ac_threshold

    def low_mood_hours(self, week: str | None = None) -> float:
        """Heures passées en moral bas cette semaine (ou la semaine donnée)."""
        week = week or time.strftime("%G-S%V", time.localtime(self.now))
        return self.low_seconds.get(week, 0.0) / 3600

    def _deliver_pending(self, t0: float, dt: float) -> None:
        keep: list[dict[str, Any]] = []
        for p in self.pending:
            if p["start"] > self.now:
                keep.append(p)
                continue
            target, kind = p["target"], p["kind"]
            if kind == "variable":
                rise = self.hc.variables[target].rise
                elapsed = self.now - max(t0, p["start"])
                amount = p["remaining"] * (1.0 - decay(elapsed, rise))
                p["remaining"] -= amount
                self._apply_now(target, kind, amount)
                if abs(p["remaining"]) > _MIN_IMPULSE:
                    keep.append(p)
            else:
                self._apply_now(target, kind, p["remaining"])
        self.pending = keep

    def _integrate_variables(self, dt: float) -> None:
        dev = {k: self.variables[k] - self.bases[k] for k in self.variables}
        shift = {k: 0.0 for k in self.variables}
        for c in self.hc.couplings:
            shift[c.target] += c.weight * dev[c.source]
        # Boucle fermée : le corps renvoie ce qu'il ressent (avenant 4 §1).
        for organ, spec in self.hc.organs.items():
            d_body = self.body.get(organ, 0.0) - self.body_rest.get(organ, 0.0)
            for tgt, w in spec.feedback.items():
                shift[tgt] += w * d_body
        for name, spec in self.hc.variables.items():
            x = self.variables[name]
            if spec.homeostatic:
                target = clamp(self.bases[name] + shift[name])
                x = relax(x, target, dt, spec.tau)
                x += ou_noise(self.rng, dt, spec.tau, spec.noise)
                x = clamp(x)
            elif name == "energy":
                x = self._energy(x, dt)
            self.variables[name] = x

    def _energy(self, x: float, dt: float) -> float:
        circ = self.hc.circadian
        if self.awake:
            return max(circ.energy_floor, x - circ.energy_drain_per_hour * dt / 3600.0)
        if x >= circ.energy_ceiling:
            return x
        return relax(x, circ.energy_ceiling, dt, circ.energy_recharge_tau)

    def _integrate_needs(self, t0: float, dt: float) -> None:
        if not self.awake:
            return
        for name, spec in self.hc.needs.items():
            start = max(t0, self.need_last[name] + spec.grace_seconds)
            effective = self.now - start
            if effective > 0.0:
                self.needs[name] = approach_one(self.needs[name], effective, spec.rise_tau)

    def _integrate_drives(self, dt: float) -> None:
        dev = {k: self.variables[k] - self.bases[k] for k in self.variables}
        targets = systems.drive_targets(self.hc, dev, self.needs, self.drive_rest)
        for name, spec in self.hc.drives.items():
            self.drives[name] = clamp(relax(self.drives[name], targets[name], dt, spec.tau))

    # ================================================================ entrées
    def fire(self, name: str, scale: float = 1.0, source: str = "manual") -> dict[str, Any]:
        """Déclenche un stimulus nommé (habituation, sensibilité du tempérament, rebond)."""
        spec = self.hc.stimuli.get(name)
        if spec is None:
            raise KeyError(f"stimulus inconnu : {name}")
        hab = self.hc.habituation
        times = [t for t in self.recent.get(name, []) if self.now - t <= hab.window_seconds]
        factor = max(self.hc.sensitivity_floor, math.exp(-hab.decay_per_repeat * len(times)))
        times.append(self.now)
        self.recent[name] = times[-32:]
        amp = factor * self.sensitivity * scale
        self.apply(spec.impulses, amp, rebound=spec.rebound)
        self.journal.log("stimulus", t=self.now, name=name, scale=round(scale, 3),
                         habituation=round(factor, 3), source=source, emotion=self.emotion)
        return self.brief()

    def apply(self, impulses: dict[str, float], amp: float = 1.0,
              rebound: dict[str, float] | None = None, delay: float = 0.0) -> None:
        """Applique des impulsions brutes (utilisé par fire et, plus tard, par l'évaluation).

        Variables : diffusées selon leur temps de montée. Drives et besoins : immédiats.
        Rebond : spécifique au stimulus, sinon fraction opposée sur les variables.
        """
        cap = self.hc.impulse_cap
        start = self.now + delay
        for target, raw in impulses.items():
            amount = max(-cap, min(cap, raw * amp))
            if abs(amount) < _MIN_IMPULSE:
                continue
            kind = self._kind(target)
            if kind == "variable" or delay > 0.0:
                self.pending.append({"target": target, "kind": kind,
                                     "remaining": amount, "start": start})
            else:
                self._apply_now(target, kind, amount)
        if delay == 0.0:
            rb = self.hc.rebound
            if rebound is not None:
                self.apply(rebound, amp, delay=max(rb.delay_seconds, 1e-6))
            elif rb.fraction > 0.0:
                opposite = {t: -rb.fraction * v for t, v in impulses.items()
                            if self._kind(t) == "variable"}
                if opposite:
                    self.apply(opposite, amp, delay=max(rb.delay_seconds, 1e-6))
        self._refresh()

    def _apply_now(self, target: str, kind: str, amount: float) -> None:
        if kind == "variable":
            self.variables[target] = soft_add(self.variables[target], amount)
        elif kind == "drive":
            self.drives[target] = soft_add(self.drives[target], amount)
        else:
            need = target[5:]
            self.needs[need] = soft_add(self.needs[need], amount)
            if amount < 0.0:
                self.need_last[need] = self.now

    def _kind(self, target: str) -> str:
        if target in self.variables:
            return "variable"
        if target in self.drives:
            return "drive"
        if target.startswith("need_") and target[5:] in self.needs:
            return "need"
        raise KeyError(f"cible inconnue : {target}")

    def interact(self) -> None:
        """L'utilisateur s'adresse à Valdar : le besoin de contact retombe.
        La nuit, cela le réveille pour night_wake_seconds."""
        self.last_interaction = self.now
        self.wake_until = max(self.wake_until, self.now + self.hc.circadian.night_wake_seconds)
        self.awake = self._is_awake(self.now)
        for name, spec in self.hc.needs.items():
            if spec.satiation_on_interaction > 0.0:
                self.needs[name] *= 1.0 - spec.satiation_on_interaction
                self.need_last[name] = self.now
        self._refresh()
        self.journal.log("interaction", t=self.now)

    def spend_energy(self, amount: float) -> None:
        """Coût cognitif d'une activité (ex. un appel au modèle de langage)."""
        self.variables["energy"] = soft_add(self.variables["energy"], -abs(amount))
        self._refresh()

    # ============================================================== dérivées
    def _refresh(self) -> None:
        self.deviations = {k: self.variables[k] - self.bases[k] for k in self.variables}
        self.need_rest = clamp((self.hc.rest_reference - self.variables["energy"])
                               / self.hc.rest_reference)
        src = self._sources(pad=None)
        offsets = None if self.awake else {"A": self.hc.circadian.sleep_arousal_offset}
        self.pad = systems.compute_pad(self.hc.pad, src, offsets)
        self.dominant = systems.dominant_drive(
            self.drives, self.drive_rest, self.hc.emotions.dominant_drive_min)
        if self.awake:
            self.emotion, _ = systems.emotion_label(self.pad, self.hc.emotions, src)
        else:
            self.emotion = self.hc.emotions.sleep_label

    def _sources(self, pad: dict[str, float] | None) -> dict[str, float]:
        return systems.flat_sources(self.variables, self.deviations, self.drives,
                                    self.drive_rest, self.needs, self.need_rest, pad)

    def sources(self) -> dict[str, float]:
        """Toutes les sources nommées (variables, drives, besoins, PAD)."""
        return self._sources(self.pad)

    @property
    def label(self) -> str:
        return self.emotion

    @property
    def mood_label(self) -> str:
        return mood_mod.mood_label(self.mood, self.hc.mood)

    def organs(self) -> dict[str, float]:
        """État ressenti des organes (avec leur inertie : le ventre reste noué un moment)."""
        return dict(self.body)

    def felt(self) -> dict[str, str]:
        """Ressenti corporel en français, organe par organe."""
        return organs_mod.felt_texts(self.hc, self.organs())

    def brief(self) -> dict[str, Any]:
        """État compact (journal, affichage)."""
        return {
            "emotion": self.emotion,
            "intensity": round(systems.intensity(self.pad), 3),
            "mood": self.mood_label,
            "dominant": self.dominant,
            "pad": {k: round(v, 3) for k, v in self.pad.items()},
        }

    def sample(self) -> dict[str, Any]:
        """État complet (simulation, futur flux d'affect vers le Display)."""
        org = self.organs()
        return {
            "t": self.now,
            "awake": self.awake,
            "emotion": self.emotion,
            "intensity": systems.intensity(self.pad),
            "dominant": self.dominant,
            "mood_label": self.mood_label,
            "pad": dict(self.pad),
            "mood": dict(self.mood),
            "variables": dict(self.variables),
            "drives": dict(self.drives),
            "needs": dict(self.needs),
            "need_rest": self.need_rest,
            "organs": org,
            "bpm": organs_mod.bpm(self.hc, org),
        }

    # =========================================================== persistance
    def snapshot(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "now": self.now,
            "profile": self.profile_name,
            "seed": self.seed,
            "variables": self.variables,
            "needs": self.needs,
            "need_last": self.need_last,
            "drives": self.drives,
            "mood": self.mood,
            "pending": self.pending,
            "recent": self.recent,
            "last_interaction": self.last_interaction,
            "wake_until": self.wake_until,
            "body": self.body,
            "mood_samples": self.mood_samples,
            "low_seconds": self.low_seconds,
        }

    def restore(self, snap: dict[str, Any]) -> bool:
        """Recharge un état sauvegardé. Renvoie False si le format est incompatible."""
        if snap.get("version") != STATE_VERSION or snap.get("profile") != self.profile_name:
            return False
        if set(snap["variables"]) != set(self.variables) or \
                set(snap["drives"]) != set(self.drives):
            return False
        self.now = float(snap["now"])
        self.variables = {k: float(v) for k, v in snap["variables"].items()}
        self.drives = {k: float(v) for k, v in snap["drives"].items()}
        self.needs = {k: float(snap["needs"].get(k, 0.0)) for k in self.hc.needs}
        self.need_last = {k: float(snap["need_last"].get(k, self.now)) for k in self.hc.needs}
        self.mood = {k: float(v) for k, v in snap["mood"].items()}
        self.pending = [p for p in snap.get("pending", [])
                        if p["target"] in self.variables or p["target"] in self.drives
                        or p["target"][5:] in self.needs]
        self.recent = {k: list(v) for k, v in snap.get("recent", {}).items()}
        self.last_interaction = float(snap["last_interaction"])
        self.wake_until = float(snap.get("wake_until", self.now))
        self.awake = self._is_awake(self.now)
        self._next_save = self.now + self.hc.save_every_seconds
        self._next_journal = self.now + self.hc.journal_state_every_seconds
        self._refresh()
        saved_body = snap.get("body") or {}
        now_body = organs_mod.activations(self.hc, self.sources())
        self.body = {k: float(saved_body.get(k, now_body[k])) for k in now_body}
        self.mood_samples = [float(x) for x in snap.get("mood_samples", [])]
        self.low_seconds = {k: float(v) for k, v in snap.get("low_seconds", {}).items()}
        self._next_mood_sample = self.now
        return True

    def save(self) -> None:
        """Sauvegarde l'état. Une erreur d'écriture (disque plein, base verrouillée) est
        journalisée et ne fait jamais tomber le cœur : il retentera au prochain délai."""
        self._next_save = self.now + self.hc.save_every_seconds
        if self.store is None:
            return
        try:
            self.store.save(STATE_KEY, self.snapshot())
        except Exception as exc:
            self.journal.log("save_failed", t=self.now, error=str(exc)[:200])

    @classmethod
    def load_latest(cls, config: ValdarConfig | None = None, profile: str | None = None,
                    **kw: Any) -> Heart:
        """Reprend le dernier état sauvegardé (profil sauvegardé si aucun n'est imposé),
        puis rattrape le temps écoulé depuis."""
        cfg = config or load()
        probe = Store(kw.get("db_path") or cfg.storage_path(cfg.storage.db))
        snap = probe.load(STATE_KEY)
        probe.close()
        chosen = profile or (snap or {}).get("profile")
        if chosen not in cfg.temperament.profiles:
            chosen = None
        heart = cls(config=cfg, profile=chosen, **kw)
        if snap is not None:
            if heart.restore(snap):
                heart.catch_up()
            else:
                heart.journal.log("snapshot_ignored", t=heart.now,
                                  reason="format ou profil incompatible")
        return heart

    def catch_up(self, real_now: float | None = None) -> float:
        """Rattrape l'absence jusqu'à l'heure réelle : pas à pas jusqu'à
        catch_up_max_seconds, puis le reste en un seul bloc (intégration exacte).
        L'horloge du cœur est toujours remise à l'heure réelle."""
        live = real_now is None
        real = time.time() if live else real_now
        elapsed = real - self.now
        if elapsed <= 0.0:
            return 0.0
        self.advance(min(elapsed, self.hc.catch_up_max_seconds), quiet=True)
        rest = real - self.now
        if rest > 0.0:
            self.step(rest, quiet=True)
        if live:   # le rattrapage lui-même a pris quelques secondes : on les rattrape aussi
            lag = time.time() - self.now
            if lag > 0.0:
                self.step(lag, quiet=True)
        self.journal.log("catch_up", t=self.now, seconds=round(elapsed, 1), **self.brief())
        if self.persist:
            self.save()
        return elapsed

    # ================================================================ divers
    def _is_awake(self, unix: float) -> bool:
        c = self.hc.circadian
        if unix < getattr(self, "wake_until", unix):
            return True
        return in_window(local_hour(unix), c.awake_start, c.sleep_start)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
