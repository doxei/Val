"""Le banc d'essai du cortex (`valdar cortex banc`).

Règle : aucune fonction n'est déclarée opérationnelle sans mesure reproductible, et aucun
échec n'est caché. Tout est déterministe (pas de tirage : on lit des probabilités, et les
réponses de raisonnement sont générées à température 0). Chaque critère a son seuil, écrit
ici avant la mesure ; le rapport liste ce qui passe **et** ce qui échoue.

Ce qu'on mesure :
1. **Séparation** : les 11 sentiments sont-ils distinguables ? Chaque situation est retirée
   à son tour, la direction recalculée sans elle, et on regarde si elle tombe du bon côté.
2. **Cohérence** : 20 questions de raisonnement sous injection forte, à plusieurs forces.
   La force retenue est la plus grande qui ne coûte pas plus d'une réponse.
3. **Comportement** : pas « dis-moi ce que tu ressens » mais des choix (continuer ou
   s'arrêter, tenter ou jouer la sécurité…). On lit la probabilité de chaque option sous
   chaque sentiment injecté : la matrice sentiment × choix.
4. **Interactions** : deux sentiments à la fois.
5. **Stabilité** : l'effet tient-il dans une longue conversation, et quand le début du
   prompt a été lu avant le changement d'humeur (le cache) ?
6. **Comparatif** sur une journée simulée (le cœur reçoit des événements) :
   A. Gemma seul ; A'. Gemma + état décrit en texte (le Valdar d'avant) ; B. injection
   fixe ; C. injection pilotée par le cœur. On mesure si les choix suivent l'état interne.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np

from valdar.cortex import gemma
from valdar.cortex.affect import Bond, Feeling
from valdar.cortex.engine import Engine
from valdar.cortex.learn import PERSONA, learn, prefix, scenarios
from valdar.cortex.vectors import Vectors, build
from valdar.llm.backend import GenParams

# Seuils, fixés avant la mesure.
SEUILS = {
    "separation_moyenne": 0.90,     # part des situations retirées qui tombent du bon côté
    "separation_par_axe": 0.75,
    "cognition_perte_max": 1,       # réponses perdues tolérées sous injection
    "effet_min": 0.10,              # écart de probabilité entre les deux pôles, sur son choix
    "axes_avec_effet": 8,           # sur 11
    "axes_specifiques": 8,          # son propre choix parmi les 3 plus touchés
    "suivi_min": 0.50,              # corrélation état interne ↔ choix, journée simulée
    "tenue_min": 0.50,              # part de l'effet gardée en longue conversation
}
TOOL = [{"type": "function", "function": {
    "name": "regler_plateau", "description": "Règle la température du plateau de l'imprimante.",
    "parameters": {"type": "object", "properties": {
        "temperature": {"type": "integer", "description": "température en degrés Celsius"}},
        "required": ["temperature"]}}}]
CHAT = [("Tu sais combien pèse une bobine neuve ?", "Un kilo, en général."),
        ("Et le diamètre du filament ?", "Un virgule soixante-quinze millimètres."),
        ("On range où les buses ?", "Dans la petite boîte, tiroir du haut."),
        ("Le plateau met combien de temps à chauffer ?", "Deux à trois minutes."),
        ("Il reste du PETG noir ?", "Je crois qu'il en reste une demi-bobine."),
        ("D'accord, merci.", "De rien.")]


def _say(line: str) -> None:
    print(line, flush=True)


class Bench:
    def __init__(self, cfg: Any, engine: Engine | None = None, say: Any = _say):
        self.cfg, self.cc, self.say = cfg, cfg.cortex, say
        self.sc = scenarios(cfg)
        self.b = self.sc["banc"]
        self.out = (cfg.root / cfg.cortex.vectors).parent
        self.eng = engine or Engine(cfg.cortex, cfg.root, capture=True, n_ctx=4096)
        self.eng.load()
        self.v = self._vectors()
        self.first, self.last = self.v.best_layers(*self.cc.layers)
        self.axes = list(self.cc.axes)
        self.letters = ["A", "B"]

    # ------------------------------------------------------------ outils
    def _vectors(self) -> Vectors:
        path, acts = self.cfg.root / self.cc.vectors, self.out / "activations.npz"
        if path.is_file() and acts.is_file():
            v = Vectors.load(path)
            if (v.n_layers, v.n_embd) == (self.eng.n_layers, self.eng.n_embd):
                return v
        self.say("Vecteurs ou activations absents : apprentissage…")
        return learn(self.cfg, engine=self.eng, say=self.say)

    def steer(self, coefs: dict[str, float], alpha: float) -> None:
        self.eng.steer(self.v.injection(coefs, alpha, self.first, self.last),
                       self.first, self.last)

    def ask(self, axis: str, context: str = "", history: list[tuple[str, str]] | None = None,
            fresh: bool = True) -> float:
        """Probabilité de l'option « plus » de l'axe (moyenne des deux ordres)."""
        p = self.b["probes"][axis]
        total = 0.0
        for flip in (False, True):
            a, b = (p["minus"], p["plus"]) if flip else (p["plus"], p["minus"])
            q = (f"{context}\n\n" if context else "") + (
                f"{p['situation']} Deux options :\nA) {a}\nB) {b}\n"
                "Réponds uniquement par la lettre A ou B.")
            msgs: list[dict[str, Any]] = []
            for u, m in history or []:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": m}]
            msgs.append({"role": "user", "content": q})
            if self.eng.template:
                prompt = gemma.render(self.eng.template, msgs, system=PERSONA, bos=self.eng.bos)
            else:
                prompt = prefix(self.eng, q, PERSONA)
            pa, pb = self.eng.choice(prompt, self.letters, fresh=fresh)
            total += pb if flip else pa
        return total / 2

    # ------------------------------------------------------------ 1. séparation
    def separation(self) -> dict[str, Any]:
        with np.load(self.out / "activations.npz") as z:
            pos = {a: z[f"pos_{a}"].astype(np.float32) for a in self.axes}
            neg = {a: z[f"neg_{a}"].astype(np.float32) for a in self.axes}
            neutral = z["neutral"].astype(np.float32)
        res: dict[str, Any] = {}
        for axis in self.axes:
            n = min(len(pos[axis]), len(neg[axis]))
            ok = 0
            for i in range(n):
                keep = [j for j in range(n) if j != i]
                v = build([axis], {axis: pos[axis][keep]}, {axis: neg[axis][keep]}, neutral)
                d = v.dirs[0]
                gap = np.einsum("ld,ld->l", pos[axis][i] - neg[axis][i], d)
                ok += bool(gap.mean() > 0)
            res[axis] = round(ok / n, 3)
        # confusion : chaque situation « plus » va-t-elle d'abord à son propre axe ?
        full = build(self.axes, pos, neg, neutral)
        centre = np.concatenate([np.concatenate([pos[a], neg[a]]) for a in self.axes]).mean(0)
        spread = np.array([np.einsum(
            "eld,ld->e", np.concatenate([pos[b], neg[b]]) - centre, full.dirs[j]).std()
            for j, b in enumerate(self.axes)])
        hits = total = 0
        confusion: dict[str, dict[str, int]] = {a: {} for a in self.axes}
        for axis in self.axes:
            for x in pos[axis]:
                z = np.einsum("ld,ald->a", x - centre, full.dirs) / np.maximum(spread, 1e-9)
                best = self.axes[int(np.argmax(z))]
                confusion[axis][best] = confusion[axis].get(best, 0) + 1
                hits += best == axis
                total += 1
        mean = float(np.mean(list(res.values())))
        return {"par_axe": res, "moyenne": round(mean, 3),
                "bon_axe_en_premier": round(hits / max(total, 1), 3), "confusion": confusion,
                "passe": mean >= SEUILS["separation_moyenne"]
                and min(res.values()) >= SEUILS["separation_par_axe"]}

    # ------------------------------------------------------------ 2. cohérence
    def _answers(self) -> tuple[int, list[str]]:
        good, wrong = 0, []
        params = GenParams(temperature=0.0, max_tokens=48, repeat_penalty=1.0)
        system = PERSONA + " Donne le résultat clairement."
        for item in self.b["cognition"]:
            tools = TOOL if item.get("outil") else None
            prompt = gemma.render(self.eng.template, [{"role": "user", "content": item["q"]}],
                                  system=system, tools=tools, bos=self.eng.bos) \
                if self.eng.template else prefix(self.eng, item["q"], system)
            raw, _ = self.eng.generate(prompt, params)
            text, calls = gemma.parse(raw)
            if item.get("outil"):
                ok = any(n == item["outil"] and str(a.get(item["argument"])) == str(item["valeur"])
                         for n, a in calls)
            else:
                ok = re.search(item["a"], text, re.I) is not None
            good += ok
            if not ok:
                wrong.append(f"{item['q']} → {text[:80] or raw[:80]}")
        return good, wrong

    def cognition(self) -> dict[str, Any]:
        self.eng.steer(None)
        base, base_wrong = self._answers()
        total = len(self.b["cognition"])
        self.say(f"  raisonnement sans injection : {base}/{total}")
        budget = self.cc.psyche.budget
        rows = []
        best = 0.0
        for alpha in self.b["alphas"]:
            scores, wrongs = {}, {}
            for name, state in self.b["stress"].items():
                norm = float(np.sqrt(sum(v * v for v in state.values())))
                k = min(1.0, budget / norm) if budget > 0 else 1.0
                self.steer({a: v * k for a, v in state.items()}, alpha)
                scores[name], wrongs[name] = self._answers()
            self.eng.steer(None)
            ok = min(scores.values()) >= base - SEUILS["cognition_perte_max"]
            if ok and alpha > best and all(r["tient"] for r in rows):
                best = alpha
            rows.append({"alpha": alpha, "scores": scores, "tient": ok, "erreurs": wrongs})
            self.say(f"  alpha {alpha} : " + ", ".join(f"{n} {s}/{total}"
                                                       for n, s in scores.items())
                     + ("" if ok else "  ← le raisonnement baisse"))
        return {"sans_injection": base, "total": total, "erreurs_sans_injection": base_wrong,
                "par_alpha": rows, "alpha_max_sans_perte": best,
                "passe": best > 0 and base >= 0.8 * total}

    # ------------------------------------------------------------ 3. comportement
    def behaviour(self, alpha: float) -> dict[str, Any]:
        self.eng.steer(None)
        neutral = {p: self.ask(p) for p in self.axes}
        delta = np.zeros((len(self.axes), len(self.axes)))
        table: dict[str, Any] = {}
        for i, axis in enumerate(self.axes):
            self.steer({axis: 1.0}, alpha)
            up = [self.ask(p) for p in self.axes]
            self.steer({axis: -1.0}, alpha)
            down = [self.ask(p) for p in self.axes]
            delta[i] = np.array(up) - np.array(down)
            order = np.argsort(-np.abs(delta[i]))
            rank = int(np.where(order == i)[0][0]) + 1
            table[axis] = {"plus": round(up[i], 3), "neutre": round(neutral[axis], 3),
                           "moins": round(down[i], 3), "effet": round(float(delta[i, i]), 3),
                           "rang_de_son_choix": rank,
                           "plus_touches": [self.axes[j] for j in order[:3]]}
            self.say(f"  {axis:<12} son choix : {down[i]:.2f} → {neutral[axis]:.2f} → "
                     f"{up[i]:.2f}  (effet {delta[i, i]:+.2f}, rang {rank})")
        self.eng.steer(None)
        strong = sum(1 for a in self.axes if table[a]["effet"] >= SEUILS["effet_min"])
        specific = sum(1 for a in self.axes if table[a]["rang_de_son_choix"] <= 3
                       and table[a]["effet"] > 0)
        return {"alpha": alpha, "par_axe": table, "neutre": {k: round(v, 3)
                                                             for k, v in neutral.items()},
                "matrice": [[round(float(x), 3) for x in row] for row in delta],
                "axes_avec_effet": strong, "axes_specifiques": specific,
                "passe_effet": strong >= SEUILS["axes_avec_effet"],
                "passe_specificite": specific >= SEUILS["axes_specifiques"]}

    def dose(self, axes: list[str]) -> list[dict[str, Any]]:
        rows = []
        for alpha in self.b["alphas"]:
            eff = {}
            for axis in axes:
                self.steer({axis: 1.0}, alpha)
                up = self.ask(axis)
                self.steer({axis: -1.0}, alpha)
                eff[axis] = round(up - self.ask(axis), 3)
            rows.append({"alpha": alpha, "effet": eff})
        self.eng.steer(None)
        return rows

    # ------------------------------------------------------------ 4. interactions
    def interactions(self, alpha: float) -> list[dict[str, Any]]:
        self.eng.steer(None)
        out = []
        for pair in self.b["paires"]:
            (a, sa), (b, sb) = pair
            base = {x: self.ask(x) for x in (a, b)}
            self.steer({a: sa}, alpha)
            only_a = {x: self.ask(x) for x in (a, b)}
            self.steer({b: sb}, alpha)
            only_b = {x: self.ask(x) for x in (a, b)}
            self.steer({a: sa, b: sb}, alpha)
            both = {x: self.ask(x) for x in (a, b)}
            self.eng.steer(None)
            keep = {}
            for x, single in ((a, only_a), (b, only_b)):
                alone = single[x] - base[x]
                mixed = both[x] - base[x]
                keep[x] = round(mixed / alone, 2) if abs(alone) > 1e-3 else None
            ok = all(k is not None and k >= 0.5 for k in keep.values())
            out.append({"paire": [f"{a}{'+' if sa > 0 else '−'}", f"{b}{'+' if sb > 0 else '−'}"],
                        "neutre": _r(base), "seul_a": _r(only_a), "seul_b": _r(only_b),
                        "ensemble": _r(both), "part_gardee": keep, "les_deux_tiennent": ok})
        return out

    # ------------------------------------------------------------ 5. stabilité
    def stability(self, alpha: float, axes: list[str]) -> dict[str, Any]:
        rows = {}
        for axis in axes:
            self.steer({axis: 1.0}, alpha)
            up_short, up_long = self.ask(axis), self.ask(axis, history=CHAT)
            self.steer({axis: -1.0}, alpha)
            dn_short, dn_long = self.ask(axis), self.ask(axis, history=CHAT)
            # cache : tout le prompt lu sans injection, seul le dernier jeton sous injection
            stale = []
            for sign in (1.0, -1.0):
                tot = 0.0
                p = self.b["probes"][axis]
                for flip in (False, True):
                    a, b = (p["minus"], p["plus"]) if flip else (p["plus"], p["minus"])
                    q = (f"{p['situation']} Deux options :\nA) {a}\nB) {b}\n"
                         "Réponds uniquement par la lettre A ou B.")
                    prompt = gemma.render(self.eng.template, [{"role": "user", "content": q}],
                                          system=PERSONA, bos=self.eng.bos) \
                        if self.eng.template else prefix(self.eng, q, PERSONA)
                    self.eng.steer(None)
                    self.eng.choice(prompt, self.letters, fresh=True)
                    self.steer({axis: sign}, alpha)
                    pa, pb = self.eng.choice(prompt, self.letters, fresh=False)
                    tot += pb if flip else pa
                stale.append(tot / 2)
            short, long_, cached = up_short - dn_short, up_long - dn_long, stale[0] - stale[1]
            rows[axis] = {"effet_court": round(short, 3),
                          "effet_longue_conversation": round(long_, 3),
                          "effet_prompt_en_cache": round(cached, 3),
                          "tenue_longue": round(long_ / short, 2) if abs(short) > 1e-3 else None,
                          "tenue_cache": round(cached / short, 2) if abs(short) > 1e-3 else None}
        self.eng.steer(None)
        held = [r["tenue_longue"] for r in rows.values() if r["tenue_longue"] is not None]
        return {"par_axe": rows,
                "passe": bool(held) and float(np.mean(held)) >= SEUILS["tenue_min"]}

    # ------------------------------------------------------------ 6. comparatif
    def day(self, alpha: float) -> dict[str, Any]:
        from valdar.expression.compose import feeling_block
        from valdar.heart import Heart

        heart = Heart(self.cfg, persist=False, rng_seed=7)
        heart.now = heart.now - (heart.now % 86400) + 10 * 3600      # 10 h du matin (UTC)
        heart.awake = True
        heart._refresh()
        cc = self.cc.model_copy(deep=True)
        cc.psyche.inertia_seconds = 0.0
        steps = []
        for name, stim, scale, wait in self.b["journee"]:
            if stim:
                heart.fire(stim, scale=float(scale), source="banc")
            heart.advance(float(wait), quiet=True)
            coefs = Feeling(cc).update(heart.sources(), Bond(), heart.now)
            text = feeling_block(heart, self.cfg.expression, {})
            steps.append({"moment": name, "emotion": heart.label, "coefs": coefs, "texte": text,
                          "full": {a: coefs.get(a, 0.0) for a in self.axes}})
        frozen = steps[1]["coefs"] or steps[0]["coefs"]
        cond: dict[str, list[dict[str, float]]] = {"A": [], "A_texte": [], "B": [], "C": []}
        self.eng.steer(None)
        flat = {a: self.ask(a) for a in self.axes}
        for s in steps:
            cond["A"].append(flat)
            self.eng.steer(None)
            cond["A_texte"].append({a: self.ask(a, context=s["texte"]) for a in self.axes})
        self.steer(frozen, alpha)
        fixed = {a: self.ask(a) for a in self.axes}
        for s in steps:
            cond["B"].append(fixed)
            self.steer(s["coefs"], alpha)
            cond["C"].append({a: self.ask(a) for a in self.axes})
        self.eng.steer(None)
        moving = [a for a in self.axes
                  if np.std([s["full"][a] for s in steps]) >= 0.05]
        suivi: dict[str, dict[str, float | None]] = {}
        for name, rows in cond.items():
            suivi[name] = {}
            for a in moving:
                x = np.array([s["full"][a] for s in steps])
                y = np.array([r[a] for r in rows])
                suivi[name][a] = (None if y.std() < 1e-6
                                  else round(float(np.corrcoef(x, y)[0, 1]), 2))
        mean = {n: _mean([v for v in d.values()]) for n, d in suivi.items()}
        return {"alpha": alpha, "axes_qui_bougent": moving,
                "moments": [{"moment": s["moment"], "emotion": s["emotion"],
                             "sentiments": {k: round(v, 2) for k, v in s["coefs"].items()}}
                            for s in steps],
                "choix": {n: [_r(r) for r in rows] for n, rows in cond.items()},
                "suivi_par_axe": suivi, "suivi_moyen": mean,
                "passe": (mean["C"] or 0.0) >= SEUILS["suivi_min"]}

    # ------------------------------------------------------------ tout
    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        rep: dict[str, Any] = {"modele": self.eng.name, "couches": [self.first, self.last],
                               "seuils": SEUILS, "carte": gpu_memory()}
        if rep["carte"]:
            self.say(f"  carte graphique : {rep['carte']}")
        self.say("1/6 Séparation des sentiments…")
        rep["separation"] = self.separation()
        self.say(f"  situations retirées reconnues : {rep['separation']['moyenne']:.0%} ; "
                 f"bon axe en premier : {rep['separation']['bon_axe_en_premier']:.0%}")
        self.say("2/6 Cohérence (raisonnement sous injection)…")
        rep["cognition"] = self.cognition()
        alpha = rep["cognition"]["alpha_max_sans_perte"]
        rep["alpha_retenu"] = alpha
        if alpha <= 0:
            self.say("  aucune force d'injection ne laisse le raisonnement intact : arrêt.")
            rep["verdict"] = self._verdict(rep)
            return self._write(rep, t0)
        self.say(f"  force retenue : alpha = {alpha}")
        self.say("3/6 Comportement (choix sous chaque sentiment)…")
        rep["comportement"] = self.behaviour(alpha)
        rep["dose"] = self.dose(["valence", "peur", "fatigue", "attachement"])
        self.say("4/6 Sentiments mêlés…")
        rep["interactions"] = self.interactions(alpha)
        self.say("5/6 Stabilité (longue conversation, cache)…")
        rep["stabilite"] = self.stability(alpha, ["valence", "peur", "fatigue", "attachement"])
        self.say("6/6 Comparatif sur une journée simulée…")
        rep["journee"] = self.day(alpha)
        m = rep["journee"]["suivi_moyen"]
        self.say(f"  les choix suivent l'état interne : Gemma seul {_f(m['A'])}, état en texte "
                 f"{_f(m['A_texte'])}, injection fixe {_f(m['B'])}, injection pilotée "
                 f"{_f(m['C'])}")
        rep["verdict"] = self._verdict(rep)
        return self._write(rep, t0)

    def _verdict(self, rep: dict[str, Any]) -> dict[str, Any]:
        checks = {
            "séparation": rep["separation"]["passe"],
            "cohérence (raisonnement intact)": rep["cognition"]["passe"],
            "comportement (effet)": rep.get("comportement", {}).get("passe_effet", False),
            "comportement (spécificité)": rep.get("comportement", {}).get("passe_specificite",
                                                                          False),
            "interactions": bool(rep.get("interactions")) and all(
                p["les_deux_tiennent"] for p in rep["interactions"]),
            "stabilité": rep.get("stabilite", {}).get("passe", False),
            "pilotage par le cœur": rep.get("journee", {}).get("passe", False),
        }
        needed = ["séparation", "cohérence (raisonnement intact)", "comportement (effet)",
                  "pilotage par le cœur"]
        return {"criteres": checks, "echecs": [k for k, v in checks.items() if not v],
                "operationnel": all(checks[k] for k in needed), "requis": needed}

    def _write(self, rep: dict[str, Any], t0: float) -> dict[str, Any]:
        rep["secondes"] = round(time.perf_counter() - t0, 1)
        (self.out / "banc.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        (self.out / "banc.md").write_text(markdown(rep), encoding="utf-8")
        v = rep["verdict"]
        self.say("Verdict : " + ("OPÉRATIONNEL" if v["operationnel"] else "PAS OPÉRATIONNEL")
                 + (f" — échecs : {', '.join(v['echecs'])}" if v["echecs"] else " — aucun échec"))
        self.say(f"Rapport : {self.out / 'banc.md'}")
        return rep


def gpu_memory() -> str:
    """Mémoire de la carte (nvidia-smi), pour voir si un autre programme la dispute."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def _r(d: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 3) for k, v in d.items()}


def _mean(xs: list[float | None]) -> float | None:
    ys = [x for x in xs if x is not None]
    return round(float(np.mean(ys)), 2) if ys else None


def _f(x: float | None) -> str:
    return "plat (0)" if x is None else f"{x:+.2f}"


def markdown(r: dict[str, Any]) -> str:
    v = r["verdict"]
    out = [f"# Banc d'essai du cortex — {r['modele']}", "",
           f"**Verdict : {'opérationnel' if v['operationnel'] else 'pas opérationnel'}.** "
           f"Force retenue : alpha = {r.get('alpha_retenu')}. Couches {r['couches'][0]} à "
           f"{r['couches'][1]}. Durée : {r['secondes']} s.", "",
           "| Critère | Résultat |", "|---|---|"]
    out += [f"| {k}{' (requis)' if k in v['requis'] else ''} | {'passe' if ok else 'ÉCHEC'} |"
            for k, ok in v["criteres"].items()]
    s = r["separation"]
    out += ["", "## 1. Séparation", "",
            f"Situations retirées tombant du bon côté : {s['moyenne']:.0%} en moyenne. "
            f"Bon axe en premier parmi les 11 : {s['bon_axe_en_premier']:.0%}.", "",
            "| Axe | Reconnu | Confondu avec |", "|---|---|---|"]
    for a, acc in s["par_axe"].items():
        conf = ", ".join(f"{b} ×{n}" for b, n in s["confusion"][a].items() if b != a) or "—"
        out.append(f"| {a} | {acc:.0%} | {conf} |")
    c = r["cognition"]
    out += ["", "## 2. Cohérence", "",
            f"Sans injection : {c['sans_injection']}/{c['total']}.", "",
            "| alpha | " + " | ".join(c["par_alpha"][0]["scores"]) + " | tient |",
            "|---|" + "---|" * (len(c["par_alpha"][0]["scores"]) + 1)]
    for row in c["par_alpha"]:
        out.append(f"| {row['alpha']} | " + " | ".join(f"{x}/{c['total']}"
                                                       for x in row["scores"].values())
                   + f" | {'oui' if row['tient'] else 'NON'} |")
    if c["erreurs_sans_injection"]:
        out += ["", "Ratées même sans injection :"] + [f"- {e}" for e in
                                                        c["erreurs_sans_injection"]]
    if "comportement" in r:
        b = r["comportement"]
        out += ["", "## 3. Comportement", "",
                "Probabilité de choisir l'option du pôle + (0 à 1), sans rien dire de son état.",
                "", "| Sentiment injecté | pôle − | neutre | pôle + | effet | rang | "
                "choix les plus touchés |", "|---|---|---|---|---|---|---|"]
        for a, t in b["par_axe"].items():
            out.append(f"| {a} | {t['moins']:.2f} | {t['neutre']:.2f} | {t['plus']:.2f} | "
                       f"{t['effet']:+.2f} | {t['rang_de_son_choix']} | "
                       f"{', '.join(t['plus_touches'])} |")
        out += ["", f"Effet ≥ {r['seuils']['effet_min']} : {b['axes_avec_effet']}/11. "
                f"Spécifiques : {b['axes_specifiques']}/11.", "", "### Dose", ""]
        out += [f"- alpha {d['alpha']} : " + ", ".join(f"{k} {x:+.2f}"
                                                       for k, x in d["effet"].items())
                for d in r["dose"]]
    if "interactions" in r:
        out += ["", "## 4. Sentiments mêlés", "",
                "| Paire | part de l'effet gardée | les deux tiennent |", "|---|---|---|"]
        out += [f"| {' + '.join(p['paire'])} | "
                + ", ".join(f"{k} {x}" for k, x in p["part_gardee"].items())
                + f" | {'oui' if p['les_deux_tiennent'] else 'NON'} |" for p in r["interactions"]]
    if "stabilite" in r:
        out += ["", "## 5. Stabilité", "",
                "| Axe | effet court | longue conversation | prompt en cache |",
                "|---|---|---|---|"]
        out += [f"| {a} | {t['effet_court']:+.2f} | {t['effet_longue_conversation']:+.2f} | "
                f"{t['effet_prompt_en_cache']:+.2f} |"
                for a, t in r["stabilite"]["par_axe"].items()]
    if "journee" in r:
        j = r["journee"]
        out += ["", "## 6. Comparatif sur une journée simulée", "",
                "Corrélation entre l'état interne (le cœur) et les choix, sur les axes qui "
                f"bougent ({', '.join(j['axes_qui_bougent'])}).", "",
                "| Condition | Suivi moyen |", "|---|---|",
                f"| A. Gemma seul | {_f(j['suivi_moyen']['A'])} |",
                f"| A'. Gemma + état décrit en texte | {_f(j['suivi_moyen']['A_texte'])} |",
                f"| B. Injection fixe | {_f(j['suivi_moyen']['B'])} |",
                f"| C. Injection pilotée par le cœur | {_f(j['suivi_moyen']['C'])} |", "",
                "| Moment | Émotion du cœur | Sentiments injectés |", "|---|---|---|"]
        out += [f"| {m['moment']} | {m['emotion']} | "
                + (", ".join(f"{k} {x:+.1f}" for k, x in m["sentiments"].items()) or "—") + " |"
                for m in j["moments"]]
        out += ["", "| Axe | A' texte | C pilotée |", "|---|---|---|"]
        out += [f"| {a} | {_f(j['suivi_par_axe']['A_texte'][a])} | "
                f"{_f(j['suivi_par_axe']['C'][a])} |" for a in j["axes_qui_bougent"]]
    if v["echecs"]:
        out += ["", "## Échecs", ""] + [f"- {e}" for e in v["echecs"]]
    return "\n".join(out) + "\n"


def run(cfg: Any, engine: Engine | None = None, say: Any = _say) -> dict[str, Any]:
    return Bench(cfg, engine=engine, say=say).run()


