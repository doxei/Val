"""Simulation accélérée du cœur et critères d'acceptation de la phase 1 (cahier v2 §16).

La simulation n'écrit jamais dans la base réelle de Valdar (persist=False).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

from valdar.config import ValdarConfig, load
from valdar.heart.heart import Heart
from valdar.sim.generator import EventGenerator
from valdar.workspace.initiative import Initiative

# Date fixe (lundi 08:00, heure locale) : la simulation est identique d'un jour à l'autre.
SIM_ANCHOR = datetime(2026, 1, 5, 8, 0, 0).timestamp()


@dataclass
class SimReport:
    days: float
    seed: int
    profile: str
    samples: int = 0
    events: dict[str, int] = field(default_factory=dict)
    emotions: dict[str, int] = field(default_factory=dict)
    moods: dict[str, int] = field(default_factory=dict)
    stuck: dict[str, float] = field(default_factory=dict)
    initiatives: dict[str, int] = field(default_factory=dict)
    max_unanswered_seen: int = 0
    energy_range: tuple[float, float] = (1.0, 0.0)
    fingerprint: str = ""
    final_emotion: str = ""
    final_mood: str = ""


def run_sim(
    config: ValdarConfig | None = None,
    days: float = 3.0,
    seed: int | None = None,
    profile: str | None = None,
    noise: bool = True,
) -> SimReport:
    cfg = config or load()
    if not noise:
        cfg = _without_noise(cfg)
    heart = Heart(cfg, profile=profile, rng_seed=seed, anchor=SIM_ANCHOR, persist=False)
    init = Initiative(cfg.initiative)
    gen = EventGenerator(cfg.sim, cfg.heart.circadian, seed=heart.seed)
    gen.schedule(heart.now)

    rep = SimReport(days=days, seed=heart.seed, profile=heart.profile_name)
    dt = cfg.sim.sampling_interval
    end = SIM_ANCHOR + days * 86400.0
    margin = cfg.sim.bound_margin
    stuck_counts = {k: 0 for k in heart.variables}
    trace: list[list] = []
    e_min, e_max = 1.0, 0.0

    while heart.now < end:
        heart.step(dt, quiet=True)
        while gen.due(heart.now):
            ev = gen.pop()
            if ev in cfg.sim.social_events:
                heart.interact()
                init.on_user_message()
            heart.fire(ev, source="sim")
        event = init.check(heart)
        if event is not None:
            rep.initiatives[event["kind"]] = rep.initiatives.get(event["kind"], 0) + 1
        rep.max_unanswered_seen = max(rep.max_unanswered_seen, init.unanswered)

        rep.samples += 1
        if heart.awake:
            rep.emotions[heart.emotion] = rep.emotions.get(heart.emotion, 0) + 1
        mood = heart.mood_label
        rep.moods[mood] = rep.moods.get(mood, 0) + 1
        for k, v in heart.variables.items():
            if v <= margin or v >= 1.0 - margin:
                stuck_counts[k] += 1
        e = heart.variables["energy"]
        e_min, e_max = min(e_min, e), max(e_max, e)
        trace.append([round(heart.now, 1), heart.emotion,
                      [round(v, 6) for v in heart.variables.values()]])

    rep.events = dict(gen.count)
    rep.stuck = {k: n / max(1, rep.samples) for k, n in stuck_counts.items()}
    rep.energy_range = (e_min, e_max)
    rep.fingerprint = hashlib.sha256(json.dumps(trace).encode()).hexdigest()
    rep.final_emotion = heart.emotion
    rep.final_mood = heart.mood_label
    return rep


def recovery_seconds(cfg: ValdarConfig, stimulus: str, scale: float = 3.0,
                     tol: float = 0.02, horizon_hours: float = 72.0) -> float | None:
    """Temps pour revenir à la trajectoire de repos après un stimulus (sans bruit).

    Compare un cœur perturbé à un cœur témoin identique : renvoie le premier instant où
    toutes les variables, drives et besoins sont à moins de `tol` du témoin.
    """
    c = _without_noise(cfg)
    a = Heart(c, anchor=SIM_ANCHOR, persist=False)
    b = Heart(c, anchor=SIM_ANCHOR, persist=False)
    a.fire(stimulus, scale=scale, source="test")
    step = c.sim.sampling_interval
    t = 0.0
    while t < horizon_hours * 3600.0:
        a.step(step, quiet=True)
        b.step(step, quiet=True)
        t += step
        if t > c.heart.rebound.delay_seconds + step and _max_diff(a, b) < tol:
            return t
    return None


def catch_up_error(cfg: ValdarConfig, hours: float = 6.0, fine_dt: float = 1.0) -> float:
    """Écart maximal entre un cœur avancé seconde par seconde et un cœur rattrapé d'un coup."""
    c = _without_noise(cfg)
    fine = Heart(c, anchor=SIM_ANCHOR, persist=False)
    coarse = Heart(c, anchor=SIM_ANCHOR, persist=False)
    for h in (fine, coarse):
        h.fire("threat", scale=2.0, source="test")
        h.fire("praise", scale=1.0, source="test")
    n = int(hours * 3600.0 / fine_dt)
    for _ in range(n):
        fine.step(fine_dt, quiet=True)
    coarse.catch_up(real_now=SIM_ANCHOR + n * fine_dt)
    return _max_diff(fine, coarse)


def _max_diff(a: Heart, b: Heart) -> float:
    d = 0.0
    for group_a, group_b in ((a.variables, b.variables), (a.drives, b.drives),
                             (a.needs, b.needs)):
        for k in group_a:
            d = max(d, abs(group_a[k] - group_b[k]))
    return d


def _without_noise(cfg: ValdarConfig) -> ValdarConfig:
    c = cfg.model_copy(deep=True)
    for spec in c.heart.variables.values():
        spec.noise = 0.0
    c._root = cfg.root
    return c


def acceptance(cfg: ValdarConfig | None = None, seed: int = 42) -> list[tuple[str, bool, str]]:
    """Critères de la phase 1 (cahier v2 §16). Renvoie [(critère, réussi, détail)]."""
    cfg = cfg or load()
    s = cfg.sim
    out: list[tuple[str, bool, str]] = []

    r1 = run_sim(cfg, days=3.0, seed=seed)
    worst = max(r1.stuck.items(), key=lambda kv: kv[1])
    out.append(("aucune variable aux bornes plus de 5 % du temps (72 h)",
                worst[1] <= s.stuck_limit, f"pire : {worst[0]} {worst[1]:.1%}"))
    distinct = len(r1.emotions)
    out.append((f"au moins {s.min_distinct_emotions} émotions différentes en 72 h",
                distinct >= s.min_distinct_emotions, f"{distinct} émotions"))
    r2 = run_sim(cfg, days=1.0, seed=seed)
    r3 = run_sim(cfg, days=1.0, seed=seed)
    out.append(("résultats identiques à graine égale", r2.fingerprint == r3.fingerprint,
                r2.fingerprint[:12]))
    worst_rec = 0.0
    failed = []
    for name in cfg.heart.stimuli:
        rec = recovery_seconds(cfg, name)
        if rec is None:
            failed.append(name)
        else:
            worst_rec = max(worst_rec, rec)
    out.append(("retour à la trajectoire de repos après chaque stimulus", not failed,
                f"pire : {worst_rec / 3600:.1f} h" if not failed else f"échec : {failed}"))
    err = catch_up_error(cfg)
    out.append(("rattrapage d'un bloc = simulation seconde par seconde",
                err <= s.catch_up_tolerance, f"écart max {err:.4f} (tolérance "
                f"{s.catch_up_tolerance})"))
    return out


def report_to_text(r: SimReport) -> str:
    def top(d: dict[str, int], n: int = 12) -> str:
        total = max(1, sum(d.values()))
        items = sorted(d.items(), key=lambda kv: -kv[1])[:n]
        return ", ".join(f"{k} {v / total:.0%}" for k, v in items)

    stuck = ", ".join(f"{k} {v:.1%}" for k, v in r.stuck.items())
    evts = ", ".join(f"{k}={v}" for k, v in sorted(r.events.items()))
    inits = ", ".join(f"{k}={v}" for k, v in r.initiatives.items()) or "aucune"
    return "\n".join([
        f"Simulation : {r.days:g} jours, graine {r.seed}, profil « {r.profile} »",
        f"Émotions   : {top(r.emotions)}",
        f"Humeurs    : {top(r.moods, 6)}",
        f"Aux bornes : {stuck}",
        f"Énergie    : de {r.energy_range[0]:.2f} à {r.energy_range[1]:.2f}",
        f"Événements : {evts}",
        f"Initiatives: {inits} (max sans réponse : {r.max_unanswered_seen})",
        f"Fin        : émotion {r.final_emotion}, humeur {r.final_mood}",
    ])
