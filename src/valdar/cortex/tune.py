"""La boucle de réglage autonome (`valdar cortex boucle`).

Elle tourne seule sur la machine : elle essaie des réglages, les mesure, garde les meilleurs,
passe le banc complet dessus, et ne s'arrête que quand le banc est au vert — ou quand elle
n'a plus rien à essayer dans le temps imparti. Dans ce cas elle le dit, sans maquiller.

Ce qu'elle a le droit de régler :
- la bande de couches où l'on injecte ;
- la force de l'injection (la plus grande qui ne coûte rien au raisonnement) ;
- l'épuration des sentiments (aucune, valence seule, valence et activation) ;
- la dose de chaque sentiment (plus forte pour une direction faible, bornée par le budget).

Ce qu'elle n'a **pas** le droit de toucher : les seuils, les questions de raisonnement, les
choix qui servent de mesure. Leur empreinte est notée à chaque tour ; si elle change, le
rapport le dit. Et pour ne pas régler sur la mesure elle-même, le réglage n'utilise que le
premier choix de chaque sentiment : le verdict, lui, compte les deux.

Entre deux tours sans piste, elle attend : si les situations d'apprentissage
(config/cortex_scenarios.yaml) ou la configuration changent sur le disque, elle réapprend et
repart. C'est par là qu'on lui donne de nouvelles idées sans la relancer.
Un fichier `data/cortex/STOP` l'arrête proprement.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from valdar.cortex.bench import SEUILS, Bench, gpu_memory
from valdar.cortex.engine import Engine

MODES = ("fond", "valence", "aucune")
EXTRA_BANDS = [(16, 28), (18, 30), (20, 28)]
DOSES = (1.0, 1.3, 1.6)
FROZEN = ("probes", "cognition", "stress", "journee", "interpretations", "etats", "paires",
          "validation")


def _say(line: str) -> None:
    print(line, flush=True)


def fingerprint(bench_section: dict[str, Any]) -> str:
    """Empreinte de tout ce qui sert de mesure (seuils, questions, choix)."""
    frozen = {k: bench_section.get(k) for k in FROZEN}
    blob = json.dumps({"seuils": SEUILS, "mesure": frozen}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class Tuner:
    def __init__(self, load_cfg: Any, engine: Engine | None = None, say: Any = _say,
                 hours: float = 10.0, clock: Any = time.time, sleep: Any = time.sleep,
                 poll_seconds: float = 30.0, top: int = 2):
        self.load_cfg, self.say = load_cfg, say
        self.engine = engine
        self.clock, self.sleep = clock, sleep
        self.deadline = clock() + hours * 3600
        self.poll, self.top = poll_seconds, top
        self.history: list[dict[str, Any]] = []
        self.tried: set[tuple[Any, ...]] = set()
        self.best: dict[str, Any] | None = None
        self.prints: list[str] = []
        self.rounds = 0
        self.cfg = load_cfg()
        self.out = (self.cfg.root / self.cfg.cortex.vectors).parent
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ état sur disque
    def _watched(self) -> float:
        files = [self.cfg.root / self.cfg.cortex.scenarios,
                 self.cfg.root / "config" / "valdar.yaml", self.out / "consigne.json"]
        return max((f.stat().st_mtime for f in files if f.is_file()), default=0.0)

    def _state(self, etat: str, **extra: Any) -> None:
        data = {"etat": etat, "tour": len(self.history), "heure": time.strftime("%H:%M:%S"),
                "empreinte_de_la_mesure": self.prints[-1] if self.prints else "",
                "mesure_modifiee_en_route": len(set(self.prints)) > 1,
                "meilleur": self.best, "historique": self.history[-12:], **extra}
        tmp = self.out / "boucle.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.out / "boucle.json")

    def _stop_asked(self) -> bool:
        return (self.out / "STOP").is_file() or self.clock() >= self.deadline

    # ------------------------------------------------------------ essai rapide
    def quick(self, b: Bench, band: tuple[int, int], mode: str, base: int) -> dict[str, Any]:
        """Un réglage, mesuré vite : force sans perte, doses, suivi du cœur."""
        b.first, b.last = band
        b.set_variant(mode)
        for axis in b.axes:
            b.cc.axes[axis].strength = self.start_doses[axis]
        alpha = 0.0
        for a in sorted(b.b["alphas"]):
            if min(b.stress_scores(a).values()) < base - SEUILS["cognition_perte_max"]:
                break
            alpha = a
        cand: dict[str, Any] = {"couches": list(band), "epuration": mode, "alpha": alpha}
        if alpha <= 0:
            return {**cand, "score": -99.0, "raison": "aucune force sans perte"}
        b.only = 0                                     # réglage sur le premier choix seulement
        try:
            doses, effects = {}, {}
            budget = b.cc.psyche.budget or 2.0
            for axis in b.axes:
                best_dose, best_eff = 1.0, None
                for dose in DOSES:
                    if dose > budget:
                        break
                    b.cc.axes[axis].strength = dose
                    b.steer(b.pole(axis, 1), alpha)
                    up = b.ask(axis)
                    b.steer(b.pole(axis, -1), alpha)
                    eff = up - b.ask(axis)
                    if best_eff is None or eff > best_eff:
                        best_dose, best_eff = dose, eff
                    if eff >= 2 * SEUILS["effet_min"]:
                        break
                b.cc.axes[axis].strength = best_dose
                doses[axis], effects[axis] = best_dose, round(float(best_eff or 0.0), 2)
            b.steer(None)
            suivi = b.tracking_single(alpha)["moyenne"]     # un sentiment à la fois
        finally:
            b.only = None
        ok = sum(1 for e in effects.values() if e >= SEUILS["effet_min"])
        inverted = [a for a, e in effects.items() if e < -0.2]
        score = ok - 3 * len(inverted) + 6 * max(suivi, -0.5) + alpha
        return {**cand, "doses": doses, "effets": effects, "axes_avec_effet": ok,
                "a_l_envers": inverted, "suivi": round(suivi, 2), "score": round(score, 2)}

    def bands(self, b: Bench, tour: int) -> list[tuple[int, int]]:
        lo, hi = b.base
        given = [tuple(x) for x in b.b.get("bandes") or []]
        pool = given + EXTRA_BANDS
        if tour >= 2 and self.best:                    # autour du meilleur
            f, last = self.best["reglages"]["couches"]
            pool = [(f, last), (f - 1, last), (f + 1, last), (f, last - 1), (f, last + 1),
                    (f - 2, last + 2), (f + 2, last - 2), *given]
        seen: list[tuple[int, int]] = []
        for band in pool:
            if lo <= band[0] < band[1] <= hi and band not in seen:
                seen.append(band)
        return seen or [b.base]

    # ------------------------------------------------------------ un tour
    def round(self) -> dict[str, Any] | None:
        tour = self.rounds + 1
        self.cfg = self.load_cfg()
        b = Bench(self.cfg, engine=self.engine, say=self.say, full=True)
        self.engine = b.eng
        self.prints.append(fingerprint(b.b))
        self.start_doses = {a: b.cc.axes[a].strength for a in b.axes}
        self.say(f"=== Tour {tour} — mesure {self.prints[-1]}"
                 + (" (ATTENTION : la mesure a changé depuis le début)"
                    if len(set(self.prints)) > 1 else "") + f" — carte : {gpu_memory()}")
        b.steer(None)
        base = b._answers()[0]
        self.say(f"  raisonnement sans injection : {base}/{len(b.b['cognition'])}")
        stamp = self._watched()
        cands = []
        for band in self.bands(b, tour):
            for mode in MODES:
                key = (band, mode, stamp)
                if key in self.tried:
                    continue
                if self._stop_asked():
                    break
                self.tried.add(key)
                c = self.quick(b, band, mode, base)
                cands.append(c)
                self.say(f"  couches {band[0]}–{band[1]}, épuration {mode:<7} : alpha "
                         f"{c['alpha']}, effet sur {c.get('axes_avec_effet', 0)}/11, à l'envers "
                         f"{', '.join(c.get('a_l_envers', [])) or 'aucun'}, suivi "
                         f"{c.get('suivi', 0):+.2f} → score {c['score']}")
                self._state("cherche", essais_du_tour=cands)
        cands.sort(key=lambda c: -c["score"])
        result = None
        for c in [c for c in cands if c["score"] > -99][:self.top]:
            if self._stop_asked():
                break
            self.say(f"  Banc complet sur couches {c['couches'][0]}–{c['couches'][1]}, "
                     f"épuration {c['epuration']}, alpha {c['alpha']}…")
            self._state("vérifie", candidat=c)
            for axis in b.axes:
                b.cc.axes[axis].strength = self.start_doses[axis]
            rep = b.run(preset=c)
            v = rep["verdict"]
            self._record(tour, rep, c)
            if v["operationnel"]:
                result = rep
                break
        if not cands:
            self.history.append({"tour": tour, "reglages": None, "operationnel": False,
                                 "requis_passes": 0, "echecs": ["rien de neuf à essayer"],
                                 "suivi": None})
        return result

    # ------------------------------------------------------------ la boucle
    def loop(self) -> dict[str, Any]:
        left = round((self.deadline - self.clock()) / 3600, 1)
        self.say(f"Boucle de réglage : jusqu'au vert, ou {left} h au plus. "
                 "Fichier data/cortex/STOP pour l'arrêter.")
        first = self.resume()
        if first is not None:
            self._apply(first)
            self._state("vert", reglages=first.get("reglages"))
            self.say("VERT : le banc passe. Réglages enregistrés, cortex activé.")
            return {"etat": "vert", "rapport": first}
        while not self._stop_asked():
            green = self.round()
            if green is not None:
                self._apply(green)
                self._state("vert", reglages=green.get("reglages"))
                self.say("VERT : le banc passe. Réglages enregistrés, cortex activé.")
                return {"etat": "vert", "rapport": green}
            if self._stop_asked():
                break
            self.rounds += 1
            if self.rounds >= 2:                       # plus de piste automatique : on attend
                stamp = self._watched()
                self._state("attend", besoin=self.needs())
                self.say("Pas encore vert. J'attends de nouvelles situations ou une nouvelle "
                         "configuration (je regarde le disque toutes les "
                         f"{int(self.poll)} s). Ce qui bloque : {self.needs()}")
                while not self._stop_asked() and self._watched() <= stamp:
                    self.sleep(self.poll)
                if not self._stop_asked():
                    self.say("Du nouveau sur le disque : je réapprends et je repars.")
        self._state("arrêté", besoin=self.needs())
        self.say("Arrêt sans vert. Meilleur réglage trouvé : "
                 + (json.dumps(self.best, ensure_ascii=False) if self.best else "aucun")
                 + ". Le cortex reste éteint.")
        return {"etat": "arrêté", "meilleur": self.best}

    def resume(self) -> dict[str, Any] | None:
        """Reprend le meilleur réglage d'une boucle précédente et le repasse au banc tout de
        suite, avant de chercher ailleurs."""
        try:
            old = json.loads((self.out / "meilleur_banc.json").read_text(encoding="utf-8"))
            preset = dict(old["reglages"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        self.cfg = self.load_cfg()
        b = Bench(self.cfg, engine=self.engine, say=self.say, full=True)
        self.engine = b.eng
        lo, hi = b.base
        if not (lo <= preset["couches"][0] < preset["couches"][1] <= hi):
            return None
        self.prints.append(fingerprint(b.b))
        self.say(f"Reprise du meilleur réglage connu : couches {preset['couches'][0]}–"
                 f"{preset['couches'][1]}, épuration {preset['epuration']}, alpha "
                 f"{preset['alpha']} — mesure {self.prints[-1]}")
        self._state("vérifie", candidat=preset)
        rep = b.run(preset=preset)
        self._record(0, rep, preset)
        return rep if rep["verdict"]["operationnel"] else None

    def _record(self, tour: int, rep: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
        v = rep["verdict"]
        jr = rep.get("journee", {})
        entry = {"tour": tour, "reglages": rep.get("reglages", cand),
                 "alpha_final": rep["alpha_retenu"], "operationnel": v["operationnel"],
                 "requis_passes": sum(1 for k in v["requis"] if v["criteres"][k]),
                 "echecs": v["echecs"],
                 "suivi": ((jr.get("suivi_multivarie") or {}).get("C") or {}).get("moyenne"),
                 "suivi_un_par_un": (jr.get("suivi_un_par_un") or {}).get("moyenne"),
                 "axes_avec_effet": rep.get("comportement", {}).get("axes_avec_effet"),
                 "a_l_envers": rep.get("comportement", {}).get("a_l_envers")}
        self.history.append(entry)
        if self.best is None or (entry["requis_passes"], entry["suivi"] or -1) > (
                self.best["requis_passes"], self.best["suivi"] or -1):
            self.best = entry
            for name in ("banc.md", "banc.json"):
                src = self.out / name
                if src.is_file():
                    (self.out / f"meilleur_{name}").write_bytes(src.read_bytes())
        return entry

    def needs(self) -> str:
        if not self.best:
            return "aucun réglage ne tient le raisonnement"
        return ", ".join(self.best["echecs"]) or "rien"

    def _apply(self, rep: dict[str, Any]) -> None:
        """Écrit les réglages gagnants dans data/reglages.json (par-dessus valdar.yaml)."""
        from valdar.config.loader import overrides_path

        r = rep["reglages"]
        path = overrides_path(self.cfg.root)
        try:
            saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except ValueError:
            saved = {}
        cx = saved.setdefault("cortex", {})
        cx.update({"alpha": rep["alpha_retenu"], "layers": list(r["couches"]),
                   "purify": r["epuration"] != "aucune",
                   "purify_core": {"fond": ["valence", "activation"], "valence": ["valence"]
                                   }.get(r["epuration"], ["valence", "activation"])})
        axes = cx.setdefault("axes", {})
        for axis, dose in r["doses"].items():
            axes.setdefault(axis, {})["strength"] = dose
        saved.setdefault("llm", {})["backend"] = "cortex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")


def run(load_cfg: Any, say: Any = _say, hours: float = 10.0) -> dict[str, Any]:
    return Tuner(load_cfg, say=say, hours=hours).loop()
