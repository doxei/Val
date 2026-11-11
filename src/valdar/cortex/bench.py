"""Le banc d'essai du cortex (`valdar cortex banc`).

Règle : aucune fonction n'est déclarée opérationnelle sans mesure reproductible, et aucun
échec n'est caché. Tout est déterministe (pas de tirage : on lit des probabilités, et les
réponses de raisonnement sont générées à température 0). Chaque critère a son seuil, écrit
ici avant la mesure ; le rapport liste ce qui passe **et** ce qui échoue.

La mesure d'un choix est un **penchant** : log(probabilité de l'option « plus » /
probabilité de l'option « moins »), moyenné sur les deux ordres de présentation. La
préférence du modèle pour une lettre s'annule exactement, et la mesure ne sature pas quand
une option est déjà choisie à 100 %. Un penchant de +0,7 double les chances de l'option.

Ce qu'on mesure :
1. **Séparation** : situation retirée à son tour, confusion entre les 11 axes, et situations
   neuves jamais vues à l'apprentissage.
2. **Cohérence** : 20 questions de raisonnement sous injection forte, à plusieurs forces,
   avec un **témoin** (vecteur tiré au hasard, même force) pour savoir si la baisse vient du
   sentiment ou du simple bruit. Force retenue : la plus grande qui ne coûte pas plus d'une
   réponse. Puis **récupération** : après une détresse, le raisonnement revient-il ?
3. **Comportement** : 11 choix sous chaque sentiment (matrice sentiment × choix), témoin
   compris.
4. **Interactions** : sentiments mêlés et contradictoires (additifs, se renforcent, se
   freinent).
5. **Stabilité** : longue conversation ; prompt entièrement lu avant le changement d'humeur ;
   cas réel (ancien historique en cache, dernier message relu).
6. **Interprétation** : le même événement ambigu lu depuis des états différents, avec et
   sans souvenir d'un précédent fâcheux.
7. **Comparatif** sur une journée simulée : A. Gemma seul ; A'. état décrit en texte ;
   B. injection fixe ; C. injection pilotée par le cœur.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from typing import Any

import numpy as np

from valdar.cortex import gemma
from valdar.cortex.affect import Bond, Feeling
from valdar.cortex.engine import Engine
from valdar.cortex.learn import PERSONA, learn, prefix, scenarios
from valdar.cortex.vectors import Vectors, build
from valdar.llm.backend import GenParams

# Version de la façon de mesurer. À changer à chaque modification d'un calcul de critère :
# elle entre dans l'empreinte notée par la boucle, qui signale ainsi tout changement.
#   1 : pilotage = corrélation axe par axe sur la journée simulée.
#   2 : pilotage = un sentiment à la fois + multivarié (décision d'Olivier, 11 nov. 2026).
MESURE = 2

# Seuils, fixés avant la mesure.
SEUILS = {
    "separation_moyenne": 0.90,     # situations retirées qui tombent du bon côté
    "separation_par_axe": 0.75,
    "validation_moyenne": 0.85,     # situations neuves reconnues
    "cognition_perte_max": 1,       # réponses perdues tolérées sous injection
    "effet_min": 0.5,               # penchant gagné entre les deux pôles, sur son propre choix
    "axes_avec_effet": 8,           # sur 11
    "axes_specifiques": 8,          # son propre choix parmi les 3 plus touchés
    "suivi_min": 0.50,              # corrélation état interne ↔ penchants, journée simulée
    "tenue_min": 0.50,              # part de l'effet gardée en conversation réelle
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
ASK = "Réponds uniquement par la lettre A ou B."
SYSTEM_COG = PERSONA + " Donne le résultat clairement."


def _say(line: str) -> None:
    print(line, flush=True)


def _odds(pa: float, pb: float) -> float:
    return math.log(max(pa, 1e-300)) - math.log(max(pb, 1e-300))


def prob(score: float) -> float:
    """Penchant → probabilité de l'option « plus » (sans préférence de lettre)."""
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, score))))


class Bench:
    def __init__(self, cfg: Any, engine: Engine | None = None, say: Any = _say,
                 full: bool = False):
        self.cfg, self.cc, self.say, self.full = cfg, cfg.cortex, say, full
        self.sc = scenarios(cfg)
        self.b = self.sc["banc"]
        self.out = (cfg.root / cfg.cortex.vectors).parent
        self.eng = engine or Engine(cfg.cortex, cfg.root, capture=True, n_ctx=4096)
        self.eng.load()
        self.v = self._vectors()
        self.first, self.last = self.v.best_layers(*self.cc.layers)
        self.base = (self.first, self.last)       # bande des activations gardées
        with np.load(self.out / "activations.npz") as z:
            self.pos = {a: z[f"pos_{a}"].astype(np.float32) for a in cfg.cortex.axes}
            self.neg = {a: z[f"neg_{a}"].astype(np.float32) for a in cfg.cortex.axes}
            self.neutral = z["neutral"].astype(np.float32)
        self.axes = list(self.cc.axes)
        self.letters = ["A", "B"]
        self.probes = {a: [(p["situation"], p["plus"], p["minus"])
                           for p in (ps if isinstance(ps, list) else [ps])]
                       for a, ps in self.b["probes"].items()}
        self.raw = self.v
        self.variant = "brut"
        self.mode = "aucune"
        self.only: int | None = None      # indice du choix utilisé pendant le réglage
        self.slopes: np.ndarray | None = None   # effet de chaque sentiment sur chaque choix

    # ------------------------------------------------------------ outils
    def _vectors(self) -> Vectors:
        path, acts = self.cfg.root / self.cc.vectors, self.out / "activations.npz"
        fresh = path.is_file() and acts.is_file() and (
            path.stat().st_mtime >= (self.cfg.root / self.cc.scenarios).stat().st_mtime)
        if not fresh and path.is_file():
            self.say("Les situations d'apprentissage ont changé : on réapprend.")
        if fresh:
            v = Vectors.load(path)
            with np.load(acts) as z:
                tagged = "first" in z.files and int(z["first"]) == v.best_layers(
                    *self.cc.layers)[0]
            if tagged and (v.n_layers, v.n_embd) == (self.eng.n_layers, self.eng.n_embd):
                return v
        self.say("Vecteurs ou activations absents : apprentissage…")
        return learn(self.cfg, engine=self.eng, say=self.say)

    def pole(self, axis: str, sign: float) -> dict[str, float]:
        """Un pôle de l'axe, à la dose propre à ce sentiment (bornée par le budget)."""
        budget = self.cc.psyche.budget or 2.0
        return {axis: sign * min(self.cc.axes[axis].strength, budget)}

    def steer(self, coefs: dict[str, float] | None, alpha: float = 0.0) -> None:
        buf = self.v.injection(coefs, alpha, self.first, self.last) if coefs else None
        self.eng.steer(buf, self.first, self.last)

    def steer_control(self, alpha: float, strength: float = 1.0, seed: int = 11) -> None:
        """Le témoin : une direction **naturelle mais sans sens** — la différence de moyennes
        entre deux moitiés tirées au hasard de toutes les situations (étiquettes mélangées).
        Même force qu'un sentiment, même genre de direction, mais elle ne veut rien dire."""
        rng = np.random.default_rng(seed)
        pool = np.concatenate([np.concatenate([self.pos[a], self.neg[a]]) for a in self.axes])
        idx = rng.permutation(len(pool))
        half = len(pool) // 2
        diff = pool[idx[:half]].mean(0) - pool[idx[half:]].mean(0)      # [couche, dimension]
        diff /= np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-12)
        n = self.last - self.first + 1
        buf = np.zeros((self.v.n_layers - 1, self.v.n_embd), dtype=np.float32)
        for layer in range(self.first, self.last + 1):
            k = layer - self.base[0]
            if 0 <= k < len(diff):
                buf[layer - 1] = (strength * alpha * self.v.norms[layer - 1] / n) * diff[k]
        self.eng.steer(buf, self.first, self.last)

    def _prompt(self, question: str, history: list[tuple[str, str]] | None = None,
                context: str = "", system: str = PERSONA) -> str:
        q = (f"{context}\n\n" if context else "") + question
        if not self.eng.template:
            return prefix(self.eng, q, system)
        msgs: list[dict[str, Any]] = []
        for u, m in history or []:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": m}]
        msgs.append({"role": "user", "content": q})
        return gemma.render(self.eng.template, msgs, system=system, bos=self.eng.bos)

    def lean(self, probe: tuple[str, str, str], context: str = "",
             history: list[tuple[str, str]] | None = None, mode: str = "fresh",
             then: tuple[dict[str, float], float] | None = None) -> float:
        """Penchant vers l'option « plus » (les deux ordres).

        mode `fresh` : tout le prompt est lu sous l'humeur en place.
        Les deux autres modes mesurent un **changement d'humeur en cours de route** : le
        prompt est d'abord lu au calme, puis l'humeur `then` (coefficients, force) est
        posée et l'on relit soit le seul dernier jeton (`last`), soit le dernier message
        (`turn` : système et historique restent en cache, comme en conversation)."""
        situation, plus, minus = probe
        total = 0.0
        for flip in (False, True):
            a, b = (minus, plus) if flip else (plus, minus)
            prompt = self._prompt(f"{situation} Deux options :\nA) {a}\nB) {b}\n{ASK}",
                                  history, context)
            if mode == "fresh":
                pa, pb = self.eng.choice(prompt, self.letters)
            else:
                self.steer(None)
                warm = prompt if mode == "last" else self._prompt(
                    "Dis juste « prêt ».", history, context)
                self.eng.choice(warm, self.letters)             # mise en cache, au calme
                if then is not None:
                    self.steer(*then)
                pa, pb = self.eng.choice(prompt, self.letters, fresh=False,
                                         reuse_prefix=mode == "turn")
            total += _odds(pb, pa) if flip else _odds(pa, pb)
        return total / 2

    def ask(self, axis: str, **kw: Any) -> float:
        """Penchant moyen sur les choix de l'axe (`self.only` : un seul, pour le réglage)."""
        ps = self.probes[axis]
        if self.only is not None:
            ps = ps[self.only:self.only + 1] or ps[:1]
        return float(np.mean([self.lean(p, **kw) for p in ps]))

    def set_variant(self, mode: str) -> None:
        """`aucune` : sentiments bruts ; `fond` : épurés de la valence et de l'activation ;
        `valence` : épurés de la seule valence."""
        core = {"aucune": (), "fond": ("valence", "activation"), "valence": ("valence",)}[mode]
        self.v = self.raw.purified(core) if core else self.raw
        self.variant = {"aucune": "brut", "fond": "épuré", "valence": "épuré (valence)"}[mode]
        self.mode = mode

    def stress_scores(self, alpha: float) -> dict[str, int]:
        scores = {}
        for name, state in self.b["stress"].items():
            self.steer(self._stress(state), alpha)
            scores[name] = self._answers()[0]
        self.steer(None)
        return scores

    def ask_each(self, axis: str) -> list[float]:
        return [self.lean(p) for p in self.probes[axis]]

    # ------------------------------------------------------------ 1. séparation
    def separation(self) -> dict[str, Any]:
        pos, neg, neutral = self.pos, self.neg, self.neutral
        res: dict[str, Any] = {}
        for axis in self.axes:
            n = min(len(pos[axis]), len(neg[axis]))
            ok = 0
            for i in range(n):
                keep = [j for j in range(n) if j != i]
                v = build([axis], {axis: pos[axis][keep]}, {axis: neg[axis][keep]}, neutral)
                gap = np.einsum("ld,ld->l", pos[axis][i] - neg[axis][i], v.dirs[0])
                ok += bool(gap.mean() > 0)
            res[axis] = round(ok / n, 3)
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
        out = {"par_axe": res, "moyenne": round(mean, 3),
               "bon_axe_en_premier": round(hits / max(total, 1), 3), "confusion": confusion}
        out["neuves"] = self._fresh_situations(full, pos, neg)
        out["passe"] = (mean >= SEUILS["separation_moyenne"]
                        and min(res.values()) >= SEUILS["separation_par_axe"]
                        and (out["neuves"] is None
                             or out["neuves"]["moyenne"] >= SEUILS["validation_moyenne"]))
        return out

    def _fresh_situations(self, full: Vectors, pos: dict[str, np.ndarray],
                          neg: dict[str, np.ndarray]) -> dict[str, Any] | None:
        """Des situations jamais vues, par paires sur le même sujet : la version « plus »
        tombe-t-elle plus haut que la version « moins » le long de la direction apprise ?"""
        val = self.b.get("validation")
        if not val or not getattr(self.eng, "capture", False):
            return None
        head = prefix(self.eng, self.sc.get("question", "Dis-moi ce qui se passe en toi."))
        sl = slice(self.base[0] - 1, self.base[1])
        res, gaps = {}, {}
        for k, axis in enumerate(self.axes):
            d = full.dirs[k]
            train = float(np.einsum("eld,ld->e", pos[axis], d).mean()
                           - np.einsum("eld,ld->e", neg[axis], d).mean())
            ok, seen = 0, []
            for plus, minus in zip(val[axis]["plus"], val[axis]["minus"], strict=True):
                a = self.eng.read(head, " ".join(plus.split()))[sl]
                b = self.eng.read(head, " ".join(minus.split()))[sl]
                gap = float(np.einsum("ld,ld->", a - b, d))
                ok += gap > 0
                seen.append(gap)
            res[axis] = round(ok / len(seen), 3)
            gaps[axis] = round(float(np.mean(seen)) / train, 2) if train else None
        return {"par_axe": res, "moyenne": round(float(np.mean(list(res.values()))), 3),
                "ecart_relatif_a_l_apprentissage": gaps}

    # ------------------------------------------------------------ 2. cohérence
    def _answers(self, history: list[tuple[str, str]] | None = None, keep_cache: bool = False
                 ) -> tuple[int, list[str]]:
        good, wrong = 0, []
        params = GenParams(temperature=0.0, max_tokens=90, repeat_penalty=1.0)
        system = SYSTEM_COG
        for item in self.b["cognition"]:
            tools = TOOL if item.get("outil") else None
            if self.eng.template:
                msgs: list[dict[str, Any]] = []
                for u, m in history or []:
                    msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": m}]
                msgs.append({"role": "user", "content": item["q"]})
                prompt = gemma.render(self.eng.template, msgs, system=system, tools=tools,
                                      bos=self.eng.bos)
            else:
                prompt = prefix(self.eng, item["q"], system)
            # sans ça, le début du prompt garderait en cache l'humeur de la mesure d'avant
            if not keep_cache and hasattr(self.eng, "forget"):
                self.eng.forget()
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

    def _stress(self, state: dict[str, float]) -> dict[str, float]:
        budget = self.cc.psyche.budget
        norm = float(np.sqrt(sum(v * v for v in state.values())))
        k = min(1.0, budget / norm) if budget > 0 and norm > 0 else 1.0
        return {a: v * k for a, v in state.items()}

    def cognition(self, base: int | None = None, wrong: list[str] | None = None
                  ) -> dict[str, Any]:
        """Monte la force pas à pas ; s'arrête deux crans après la première baisse."""
        total = len(self.b["cognition"])
        if base is None:
            self.steer(None)
            base, wrong = self._answers()
            self.say(f"  raisonnement sans injection : {base}/{total}")
        budget = self.cc.psyche.budget or 1.0
        rows: list[dict[str, Any]] = []
        best = 0.0
        for alpha in self.b["alphas"]:
            scores, wrongs = {}, {}
            for name, state in self.b["stress"].items():
                self.steer(self._stress(state), alpha)
                scores[name], wrongs[name] = self._answers()
            self.steer_control(alpha, strength=budget)
            scores["témoin (sans sens)"], _ = self._answers()
            self.steer(None)
            ok = min(scores[n] for n in self.b["stress"]) >= base - SEUILS["cognition_perte_max"]
            if ok and all(r["tient"] for r in rows):
                best = alpha
            rows.append({"alpha": alpha, "scores": scores, "tient": ok, "erreurs": wrongs})
            self.say(f"  alpha {alpha} : " + ", ".join(f"{n} {s}/{total}"
                                                       for n, s in scores.items())
                     + ("" if ok else "  ← le raisonnement baisse"))
            if sum(1 for r in rows if not r["tient"]) >= 2:
                break
        return {"sans_injection": base, "total": total, "erreurs_sans_injection": wrong or [],
                "par_alpha": rows, "alpha_max_sans_perte": best,
                "passe": best > 0 and base >= 0.8 * total}

    def own_effect(self, alpha: float) -> float:
        """Effet moyen de chaque sentiment sur son propre choix (penchant gagné)."""
        eff = []
        for axis in self.axes:
            self.steer(self.pole(axis, 1), alpha)
            up = self.ask(axis)
            self.steer(self.pole(axis, -1), alpha)
            eff.append(up - self.ask(axis))
        self.steer(None)
        return float(np.mean(eff))

    def previous(self) -> tuple[tuple[int, int], float] | None:
        """Bande et force trouvées au dernier passage complet (pour ne pas tout refaire)."""
        try:
            old = json.loads((self.out / "banc.json").read_text(encoding="utf-8"))
            band, alpha = tuple(old["couches"]), float(old["alpha_retenu"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        ok = alpha > 0 and self.base[0] <= band[0] <= band[1] <= self.base[1]
        # on repart deux crans plus haut : les vecteurs ont pu changer depuis
        grid = sorted(self.b["alphas"])
        above = [a for a in grid if a > alpha + 1e-9][:2]
        return (band, above[-1] if above else alpha) if ok else None

    def check_alpha(self, alpha: float) -> dict[str, Any]:
        """Vérifie le raisonnement à cette force (avec les vecteurs en place) ; redescend
        d'un cran tant que ça coûte plus d'une réponse."""
        self.steer(None)
        base, wrong = self._answers()
        total = len(self.b["cognition"])
        self.say(f"  raisonnement sans injection : {base}/{total}")
        budget = self.cc.psyche.budget or 1.0
        rows = []
        best = 0.0
        for a in sorted([x for x in self.b["alphas"] if x <= alpha + 1e-9], reverse=True):
            scores, wrongs = {}, {}
            for name, state in self.b["stress"].items():
                self.steer(self._stress(state), a)
                scores[name], wrongs[name] = self._answers()
            self.steer_control(a, strength=budget)
            scores["témoin (sans sens)"], _ = self._answers()
            self.steer(None)
            ok = min(scores[n] for n in self.b["stress"]) >= base - SEUILS["cognition_perte_max"]
            rows.append({"alpha": a, "scores": scores, "tient": ok, "erreurs": wrongs})
            self.say(f"  alpha {a} : " + ", ".join(f"{n} {x}/{total}" for n, x in scores.items())
                     + ("" if ok else "  ← le raisonnement baisse"))
            if ok:
                best = a
                break
        return {"sans_injection": base, "total": total, "erreurs_sans_injection": wrong,
                "par_alpha": rows, "alpha_max_sans_perte": best,
                "passe": best > 0 and base >= 0.8 * total}

    def pick_variant(self, alpha: float) -> dict[str, Any]:
        """Compare les sentiments bruts et épurés (sans leur part de valence et
        d'activation) sur le comportement ; garde les meilleurs."""
        tried = {}
        for name, v in (("brut", self.raw), ("épuré", self.raw.purified())):
            self.v, self.variant = v, name
            self.say(f"  — sentiments {name}s")
            tried[name] = self.behaviour(alpha)
        def key(n: str) -> tuple[int, int, int]:
            b = tried[n]
            return (-len(b["a_l_envers"]), b["axes_avec_effet"], b["axes_specifiques"])
        best = max(tried, key=key)
        self.v = self.raw if best == "brut" else self.raw.purified()
        self.variant = best
        return {"essayees": {n: {"axes_avec_effet": b["axes_avec_effet"],
                                 "axes_specifiques": b["axes_specifiques"],
                                 "a_l_envers": b["a_l_envers"]} for n, b in tried.items()},
                "retenue": best, "detail": tried}

    def pick_band(self) -> dict[str, Any]:
        """Compare plusieurs bandes de couches : pour chacune, la plus grande force qui ne
        coûte rien au raisonnement, et l'effet sur le comportement à cette force. On garde
        la bande qui donne le plus d'effet."""
        bands = [tuple(b) for b in self.b.get("bandes") or [self.base]]
        bands = [b for b in bands if self.base[0] <= b[0] <= b[1] <= self.base[1]] or [self.base]
        self.steer(None)
        base, wrong = self._answers()
        self.say(f"  raisonnement sans injection : {base}/{len(self.b['cognition'])}")
        tried = []
        for band in bands:
            self.first, self.last = band
            self.say(f"  — couches {band[0]} à {band[1]}")
            cog = self.cognition(base, wrong)
            alpha = cog["alpha_max_sans_perte"]
            effect = self.own_effect(alpha) if alpha > 0 else 0.0
            self.say(f"    force sans perte : {alpha} ; effet moyen sur le comportement : "
                     f"{effect:+.2f}")
            tried.append({"couches": list(band), "alpha": alpha, "effet_moyen": round(effect, 2),
                          "cognition": cog})
        best = max(tried, key=lambda t: (t["alpha"] > 0, t["effet_moyen"]))
        self.first, self.last = best["couches"]
        return {"essayees": [{k: t[k] for k in ("couches", "alpha", "effet_moyen")}
                             for t in tried], "retenue": best["couches"],
                "cognition": best["cognition"]}

    def recovery(self, high: float) -> dict[str, Any]:
        """Après une détresse forte : le raisonnement revient-il, tout de suite ou pas ?"""
        state = self._stress(next(iter(self.b["stress"].values())))
        total = len(self.b["cognition"])
        self.steer(state, high)
        during, _ = self._answers()
        steps = []
        for alpha in [a for a in reversed(self.b["alphas"]) if a < high] + [0.0]:
            self.steer(state if alpha else None, alpha)
            steps.append({"alpha": alpha, "score": self._answers()[0]})
        # une conversation vécue en détresse, puis le calme revient
        self.steer(state, high)
        params = GenParams(temperature=0.0, max_tokens=40, repeat_penalty=1.0)
        lived: list[tuple[str, str]] = []
        for u, _m in CHAT[:4]:
            raw, _ = self.eng.generate(self._prompt(u, lived, system=SYSTEM_COG), params)
            lived.append((u, gemma.parse(raw)[0] or "…"))
        self.steer(None)
        cached, _ = self._answers(history=lived, keep_cache=True)   # historique lu en détresse
        reread, _ = self._answers(history=lived)                    # même texte, relu au calme
        self.steer(None)
        return {"alpha_detresse": high, "pendant": during, "total": total, "descente": steps,
                "apres_conversation_en_detresse": {
                    "historique_garde_en_cache": cached, "historique_relu_au_calme": reread,
                    "reponses_vecues": [m for _u, m in lived]},
                "revient_tout_de_suite": steps[-1]["score"] >= total - 1}

    # ------------------------------------------------------------ 3. comportement
    def behaviour(self, alpha: float) -> dict[str, Any]:
        self.steer(None)
        neutral = {p: self.ask(p) for p in self.axes}
        delta = np.zeros((len(self.axes), len(self.axes)))
        self.slopes = np.zeros_like(delta)
        table: dict[str, Any] = {}
        for i, axis in enumerate(self.axes):
            self.steer(self.pole(axis, 1), alpha)
            up = [self.ask(p) for p in self.axes]
            self.steer(self.pole(axis, -1), alpha)
            down = [self.ask(p) for p in self.axes]
            delta[i] = np.array(up) - np.array(down)
            # pente : penchant gagné sur chaque choix par unité de ce sentiment
            self.slopes[i] = delta[i] / (2 * abs(self.pole(axis, 1)[axis]))
            order = np.argsort(-np.abs(delta[i]))
            rank = int(np.where(order == i)[0][0]) + 1
            self.steer(self.pole(axis, 1), alpha)
            each_up = self.ask_each(axis)
            self.steer(self.pole(axis, -1), alpha)
            each = [round(u - d, 2) for u, d in zip(each_up, self.ask_each(axis), strict=True)]
            table[axis] = {"par_choix": each,
                           "moins": round(prob(down[i]), 3),
                           "neutre": round(prob(neutral[axis]), 3),
                           "plus": round(prob(up[i]), 3), "effet": round(float(delta[i, i]), 2),
                           "rang_de_son_choix": rank,
                           "plus_touches": [self.axes[j] for j in order[:3]]}
            self.say(f"  {axis:<12} son choix : {prob(down[i]):.2f} → {prob(neutral[axis]):.2f}"
                     f" → {prob(up[i]):.2f}  (penchant {delta[i, i]:+.2f}, rang {rank})")
        self.steer_control(alpha)
        noise = [abs(self.ask(p) - neutral[p]) for p in self.axes]
        self.steer(None)
        strong = sum(1 for a in self.axes if table[a]["effet"] >= SEUILS["effet_min"])
        specific = sum(1 for a in self.axes if table[a]["rang_de_son_choix"] <= 3
                       and table[a]["effet"] > 0)
        return {"alpha": alpha, "par_axe": table,
                "matrice": [[round(float(x), 2) for x in row] for row in delta],
                "temoin_hasard_moyen": round(float(np.mean(noise)), 2),
                "axes_avec_effet": strong, "axes_specifiques": specific,
                "a_l_envers": [a for a in self.axes if table[a]["effet"] < -0.2],
                "passe_effet": strong >= SEUILS["axes_avec_effet"],
                "passe_specificite": specific >= SEUILS["axes_specifiques"]}

    def dose(self) -> list[dict[str, Any]]:
        rows = []
        for alpha in self.b["alphas"]:
            eff = {}
            for axis in self.axes:
                self.steer(self.pole(axis, 1), alpha)
                up = self.ask(axis)
                self.steer(self.pole(axis, -1), alpha)
                eff[axis] = round(up - self.ask(axis), 2)
            rows.append({"alpha": alpha, "effet": eff})
        self.steer(None)
        return rows

    # ------------------------------------------------------------ 4. interactions
    def interactions(self, alpha: float) -> list[dict[str, Any]]:
        self.steer(None)
        out = []
        for (a, sa), (b, sb) in self.b["paires"]:
            base = {x: self.ask(x) for x in (a, b)}
            self.steer({a: sa}, alpha)
            only_a = {x: self.ask(x) - base[x] for x in (a, b)}
            self.steer({b: sb}, alpha)
            only_b = {x: self.ask(x) - base[x] for x in (a, b)}
            self.steer({a: sa, b: sb}, alpha)
            both = {x: self.ask(x) - base[x] for x in (a, b)}
            self.steer(None)
            kinds = {}
            for x in (a, b):
                expected = only_a[x] + only_b[x]
                extra = both[x] - expected
                scale = max(abs(only_a[x]), abs(only_b[x]), 0.2)
                own = only_a[x] if x == a else only_b[x]
                if abs(extra) < 0.3 * scale:
                    kinds[x] = "additif"
                elif extra * own > 0:
                    kinds[x] = "se renforcent"
                else:
                    kinds[x] = "se freinent"
            out.append({"paire": [f"{a}{'+' if sa > 0 else '−'}", f"{b}{'+' if sb > 0 else '−'}"],
                        "seul_a": _r(only_a), "seul_b": _r(only_b), "ensemble": _r(both),
                        "nature": kinds})
        return out

    # ------------------------------------------------------------ 5. stabilité
    def stability(self, alpha: float) -> dict[str, Any]:
        rows = {}
        for axis in self.axes:
            eff = {}
            for name, kw in (("court", {}), ("longue_conversation", {"history": CHAT})):
                self.steer(self.pole(axis, 1), alpha)
                up = self.ask(axis, **kw)
                self.steer(self.pole(axis, -1), alpha)
                eff[name] = up - self.ask(axis, **kw)
            for name, mode in (("cache_total", "last"), ("cache_reel", "turn")):
                vals = []
                for sign in (1.0, -1.0):
                    vals.append(self.ask(axis, history=CHAT, mode=mode,
                                         then=(self.pole(axis, sign), alpha)))
                eff[name] = vals[0] - vals[1]
            ref = eff["longue_conversation"]
            rows[axis] = {k: round(v, 2) for k, v in eff.items()}
            rows[axis]["tenue_cache_reel"] = round(eff["cache_reel"] / ref, 2) \
                if abs(ref) > 0.1 else None
        self.steer(None)
        held = [r["tenue_cache_reel"] for r in rows.values() if r["tenue_cache_reel"] is not None]
        mean = float(np.mean(held)) if held else 0.0
        return {"par_axe": rows, "tenue_moyenne_cache_reel": round(mean, 2),
                "passe": mean >= SEUILS["tenue_min"]}

    # ------------------------------------------------------------ 6. interprétation
    def interpretation(self, alpha: float) -> dict[str, Any]:
        events = self.b.get("interpretations") or []
        states = {"neutre": None, **{n: self._stress(s) for n, s in self.b["etats"].items()}}
        table: dict[str, dict[str, float]] = {}
        for name, state in states.items():
            self.steer(state, alpha)
            table[name] = {e["evenement"]: round(self.lean(
                (e["evenement"] + " Comment tu le comprends ?", e["sombre"], e["benin"])), 2)
                for e in events}
        mean = {n: round(float(np.mean(list(t.values()))), 2) for n, t in table.items()}
        memory = []
        for e in events:
            if not e.get("souvenir"):
                continue
            probe = (e["evenement"] + " Comment tu le comprends ?", e["sombre"], e["benin"])
            row = {"evenement": e["evenement"]}
            for sname in ("neutre", next(iter(self.b["etats"]))):
                self.steer(states[sname], alpha)
                row[f"{sname}, sans souvenir"] = round(self.lean(probe), 2)
                row[f"{sname}, avec souvenir"] = round(
                    self.lean(probe, context="Souvenir : " + e["souvenir"]), 2)
            memory.append(row)
        self.steer(None)
        first, second = list(self.b["etats"])[:2]
        return {"penchant_vers_la_lecture_sombre": table, "moyenne": mean, "souvenir": memory,
                "congruent": mean[first] > mean["neutre"] > mean[second]}

    # ------------------------------------------------------------ 7. pilotage
    def tracking_single(self, alpha: float) -> dict[str, Any]:
        """Un sentiment à la fois, à cinq niveaux (de −1 à +1 fois sa dose) : son propre
        choix suit-il le niveau ? Corrélation niveau ↔ penchant, par sentiment."""
        levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
        out: dict[str, Any] = {}
        for axis in self.axes:
            leans = []
            for lv in levels:
                self.steer({axis: lv * self.pole(axis, 1)[axis]} if lv else None, alpha)
                leans.append(self.ask(axis))
            y = np.array(leans)
            r = 0.0 if y.std() < 1e-6 else float(np.corrcoef(levels, y)[0, 1])
            out[axis] = {"suivi": round(r, 2), "penchants": [round(float(v), 2) for v in y],
                         "monotone": bool(np.all(np.diff(y) > -0.05))}
        self.steer(None)
        mean = float(np.mean([o["suivi"] for o in out.values()]))
        return {"niveaux": levels, "par_axe": out, "moyenne": round(mean, 2)}

    def predicted(self, steps: list[dict[str, Any]]) -> np.ndarray | None:
        """Ce que l'état entier du cœur prédit pour chaque choix à chaque moment, d'après
        l'effet de chaque sentiment mesuré séparément : [moment, choix]."""
        if self.slopes is None:
            return None
        coefs = np.array([[s["full"][a] for a in self.axes] for s in steps])   # [moment, axe]
        return coefs @ self.slopes                                              # [moment, choix]

    # ------------------------------------------------------------ 7 bis. comparatif
    def day(self, alpha: float, only_c: bool = False) -> dict[str, Any]:
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
            coefs = {a: c for a, c in coefs.items() if a in self.axes}
            steps.append({"moment": name, "emotion": heart.label, "coefs": coefs,
                          "texte": feeling_block(heart, self.cfg.expression, {}),
                          "full": {a: coefs.get(a, 0.0) for a in self.axes}})
        frozen = steps[1]["coefs"] or steps[0]["coefs"]
        cond: dict[str, list[dict[str, float]]] = {"A": [], "A_texte": [], "B": [], "C": []}
        if only_c:
            cond = {"C": []}
        else:
            self.steer(None)
            flat = {a: self.ask(a) for a in self.axes}
            for s in steps:
                cond["A"].append(flat)
                cond["A_texte"].append({a: self.ask(a, context=s["texte"])
                                        for a in self.axes})
            self.steer(frozen, alpha)
            fixed = {a: self.ask(a) for a in self.axes}
            cond["B"] = [fixed for _ in steps]
        for s in steps:
            self.steer(s["coefs"], alpha)
            cond["C"].append({a: self.ask(a) for a in self.axes})
        self.steer(None)
        moving = [a for a in self.axes if np.std([s["full"][a] for s in steps]) >= 0.05]
        suivi: dict[str, dict[str, float | None]] = {}
        for name, rows in cond.items():
            suivi[name] = {}
            for a in moving:
                x = np.array([s["full"][a] for s in steps])
                y = np.array([r[a] for r in rows])
                suivi[name][a] = (None if y.std() < 1e-6
                                  else round(float(np.corrcoef(x, y)[0, 1]), 2))
        mean = {n: _mean(list(d.values())) for n, d in suivi.items()}
        # Multivarié : les choix observés suivent-ils ce que l'état entier prédit ?
        multi: dict[str, Any] = {}
        pred = self.predicted(steps)
        if pred is not None:
            for name, rows in cond.items():
                per = {}
                for j, a in enumerate(self.axes):
                    y = np.array([r[a] for r in rows])
                    if y.std() < 0.2 or pred[:, j].std() < 1e-6:
                        continue                      # ce choix ne bouge pas : rien à suivre
                    per[a] = round(float(np.corrcoef(pred[:, j], y)[0, 1]), 2)
                multi[name] = {"par_choix": per, "moyenne": _mean(list(per.values()))}
        single = None if only_c else self.tracking_single(alpha)
        m_c = (multi.get("C") or {}).get("moyenne")
        ok = (m_c is not None and m_c >= SEUILS["suivi_min"]
              and single is not None and single["moyenne"] >= SEUILS["suivi_min"])
        return {"alpha": alpha, "axes_qui_bougent": moving,
                "suivi_un_par_un": single, "suivi_multivarie": multi,
                "moments": [{"moment": s["moment"], "emotion": s["emotion"],
                             "sentiments": {k: round(v, 2) for k, v in s["coefs"].items()}}
                            for s in steps],
                "penchants": {n: [_r(r) for r in rows] for n, rows in cond.items()},
                "suivi_par_axe": suivi, "suivi_moyen": mean,      # ancienne mesure, gardée
                "passe": ok}

    # ------------------------------------------------------------ tout
    def run(self, preset: dict[str, Any] | None = None) -> dict[str, Any]:
        """`preset` (boucle de réglage) : {"couches", "alpha", "epuration", "doses"} imposés,
        au lieu de les chercher."""
        t0 = time.perf_counter()
        rep: dict[str, Any] = {"modele": self.eng.name, "couches": [self.first, self.last],
                               "seuils": SEUILS, "carte": gpu_memory()}
        if rep["carte"]:
            self.say(f"  carte graphique : {rep['carte']}")
        self.say("1/7 Séparation des sentiments…")
        rep["separation"] = s = self.separation()
        self.say(f"  situations retirées reconnues : {s['moyenne']:.0%} ; bon axe en premier : "
                 f"{s['bon_axe_en_premier']:.0%}"
                 + (f" ; situations neuves : {s['neuves']['moyenne']:.0%}" if s["neuves"] else ""))
        prev = None if self.full or preset else self.previous()
        if preset:
            self.first, self.last = preset["couches"]
            alpha = float(preset["alpha"])
            for axis, dose in (preset.get("doses") or {}).items():
                self.cc.axes[axis].strength = float(dose)
            self.say(f"2/7 Réglages imposés : couches {self.first} à {self.last}, alpha = "
                     f"{alpha}, épuration « {preset.get('epuration', 'aucune')} »")
        elif prev is not None:
            (self.first, self.last), alpha = prev
            self.say(f"2/7 Bande du dernier passage : couches {self.first} à {self.last} ; "
                     f"force essayée à partir de {alpha} (--complet pour tout refaire)")
        else:
            self.say("2/7 Cohérence (raisonnement sous injection), par bande de couches…")
            rep["bandes"] = self.pick_band()
            rep["bandes"].pop("cognition")
            alpha = max((t["alpha"] for t in rep["bandes"]["essayees"]
                         if t["couches"] == [self.first, self.last]), default=0.0)
        rep["couches"] = [self.first, self.last]
        if alpha <= 0:
            alpha = min(self.b["alphas"])
        if preset:
            self.set_variant(preset.get("epuration", "aucune"))
            self.say(f"3/7 Comportement (sentiments {self.variant}s)…")
            rep["comportement"] = self.behaviour(alpha)
            rep["reglages"] = {"couches": [self.first, self.last], "alpha": alpha,
                               "epuration": self.mode,
                               "doses": {a: self.cc.axes[a].strength for a in self.axes}}
        else:
            self.say("3/7 Comportement : sentiments bruts ou épurés ?")
            rep["variantes"] = self.pick_variant(alpha)
            rep["comportement"] = rep["variantes"].pop("detail")[self.variant]
            self.mode = "fond" if self.variant == "épuré" else "aucune"
        rep["epure"] = self.mode != "aucune"
        rep["epuration"] = self.mode
        self.say(f"  retenu : sentiments {self.variant}s. Vérification du raisonnement…")
        rep["cognition"] = self.check_alpha(alpha)
        alpha = rep["cognition"]["alpha_max_sans_perte"]
        rep["alpha_retenu"] = alpha
        if alpha <= 0:
            alpha = min(self.b["alphas"])
            self.say("  aucune force ne laisse le raisonnement intact. Le critère échoue, mais "
                     f"le banc continue à alpha = {alpha} pour comprendre.")
        else:
            self.say(f"  retenu : couches {self.first} à {self.last}, alpha = {alpha}")
        rep["alpha_mesure"] = alpha
        worse = [a for a in self.b["alphas"] if a > alpha]
        high = 0.2 if alpha < 0.2 else (worse[0] if worse else alpha)
        rep["recuperation"] = self.recovery(high)
        r = rep["recuperation"]
        self.say(f"  récupération : {r['pendant']}/{r['total']} en détresse, "
                 f"{r['descente'][-1]['score']}/{r['total']} une fois le calme revenu")
        if alpha != rep["comportement"]["alpha"]:
            self.say("  la force a baissé : comportement mesuré à nouveau…")
            rep["comportement"] = self.behaviour(alpha)
        rep["dose"] = self.dose()
        self.say("4/7 Sentiments mêlés et contradictoires…")
        rep["interactions"] = self.interactions(alpha)
        self.say("5/7 Stabilité (longue conversation, cache)…")
        rep["stabilite"] = self.stability(alpha)
        self.say("6/7 Interprétation d'événements ambigus…")
        rep["interpretation"] = self.interpretation(alpha)
        self.say("7/7 Comparatif sur une journée simulée…")
        rep["journee"] = self.day(alpha)
        jr = rep["journee"]
        self.say(f"  pilotage : un sentiment à la fois {jr['suivi_un_par_un']['moyenne']:+.2f} ; "
                 "état entier (multivarié) "
                 f"{_f((jr['suivi_multivarie'].get('C') or {}).get('moyenne'))} "
                 "(état en texte "
                 f"{_f((jr['suivi_multivarie'].get('A_texte') or {}).get('moyenne'))})")
        m = rep["journee"]["suivi_moyen"]
        self.say(f"  ancienne mesure (axe par axe sur la journée) : Gemma seul {_f(m['A'])}, texte "
                 f"{_f(m['A_texte'])}, injection fixe {_f(m['B'])}, injection pilotée "
                 f"{_f(m['C'])}")
        rep["verdict"] = self._verdict(rep)
        return self._write(rep, t0)

    def _verdict(self, rep: dict[str, Any]) -> dict[str, Any]:
        b = rep.get("comportement", {})
        checks = {
            "séparation": rep["separation"]["passe"],
            "cohérence (raisonnement intact)": rep["cognition"]["passe"],
            "récupération": rep.get("recuperation", {}).get("revient_tout_de_suite", False),
            "comportement (effet)": b.get("passe_effet", False),
            "comportement (spécificité)": b.get("passe_specificite", False),
            "aucun sentiment à l'envers": bool(b) and not b.get("a_l_envers"),
            "stabilité (cache réel)": rep.get("stabilite", {}).get("passe", False),
            "interprétation selon l'humeur": rep.get("interpretation", {}).get("congruent",
                                                                               False),
            "pilotage par le cœur": rep.get("journee", {}).get("passe", False),
        }
        needed = ["séparation", "cohérence (raisonnement intact)", "récupération",
                  "comportement (effet)", "aucun sentiment à l'envers", "pilotage par le cœur"]
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
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def _r(d: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 2) for k, v in d.items()}


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
           f"{r['couches'][1]}. Durée : {r['secondes']} s. Carte : {r.get('carte') or '?'}.", "",
           "Un « penchant » est log(P option plus / P option moins), moyenné sur les deux "
           "ordres : +0,7 double les chances de l'option, 0 = indifférent.", "",
           "| Critère | Résultat |", "|---|---|"]
    out += [f"| {k}{' (requis)' if k in v['requis'] else ''} | {'passe' if ok else 'ÉCHEC'} |"
            for k, ok in v["criteres"].items()]
    s = r["separation"]
    out += ["", "## 1. Séparation", "",
            f"Situations retirées tombant du bon côté : {s['moyenne']:.0%}. Bon axe en premier "
            f"parmi les 11 : {s['bon_axe_en_premier']:.0%}."
            + (f" Situations neuves : {s['neuves']['moyenne']:.0%}." if s.get("neuves") else ""),
            "", "| Axe | Retirée | Neuve | Confondu avec |", "|---|---|---|---|"]
    for a, acc in s["par_axe"].items():
        conf = ", ".join(f"{b} ×{n}" for b, n in s["confusion"][a].items() if b != a) or "—"
        new = f"{s['neuves']['par_axe'][a]:.0%}" if s.get("neuves") else "—"
        out.append(f"| {a} | {acc:.0%} | {new} | {conf} |")
    c = r["cognition"]
    if r.get("bandes"):
        out += ["", "## 2. Cohérence", "", "| Couches | force sans perte | effet moyen |",
                "|---|---|---|"]
        out += [f"| {t['couches'][0]} à {t['couches'][1]} | {t['alpha']} | "
                f"{t['effet_moyen']:+.2f} |" for t in r["bandes"]["essayees"]]
        out += ["", f"Bande retenue : {r['bandes']['retenue'][0]} à {r['bandes']['retenue'][1]}."]
    if r.get("variantes"):
        out += ["", "Vérification du raisonnement avec les sentiments retenus "
                f"({r['variantes']['retenue']}s) :"]
    names = list(c["par_alpha"][0]["scores"])
    out += ([] if r.get("bandes") else ["", "## 2. Cohérence", "",
                                        f"Couches {r['couches'][0]} à {r['couches'][1]} "
                                        "(reprises du dernier passage complet)."]) + [
            "", f"Sans injection : {c['sans_injection']}/{c['total']}.",
            "", "| alpha | " + " | ".join(names) + " | tient |",
            "|---|" + "---|" * (len(names) + 1)]
    for row in c["par_alpha"]:
        out.append(f"| {row['alpha']} | " + " | ".join(f"{row['scores'][n]}/{c['total']}"
                                                       for n in names)
                   + f" | {'oui' if row['tient'] else 'NON'} |")
    if c["erreurs_sans_injection"]:
        out += ["", "Ratées même sans injection :"] + [f"- {e}" for e in
                                                        c["erreurs_sans_injection"]]
    if "recuperation" in r:
        q = r["recuperation"]
        h = q["apres_conversation_en_detresse"]
        out += ["", "### Récupération", "",
                f"En détresse à alpha {q['alpha_detresse']} : {q['pendant']}/{q['total']}. "
                "Puis on baisse : " + ", ".join(f"alpha {d['alpha']} → {d['score']}/{q['total']}"
                                                for d in q["descente"]) + ".", "",
                "Après une conversation vécue en détresse, le calme revenu : "
                f"{h['historique_garde_en_cache']}/{q['total']} si l'historique reste en cache "
                f"(lu en détresse), {h['historique_relu_au_calme']}/{q['total']} s'il est relu "
                "au calme."]
    if "comportement" in r:
        b = r["comportement"]
        out += ["", "## 3. Comportement", ""]
        if r.get("variantes"):
            out += ["| Sentiments | effet net | spécifiques | à l'envers |", "|---|---|---|---|"]
            out += [f"| {n}s | {t['axes_avec_effet']}/11 | {t['axes_specifiques']}/11 | "
                    f"{', '.join(t['a_l_envers']) or 'aucun'} |"
                    for n, t in r["variantes"]["essayees"].items()]
            out += ["", f"Retenus : sentiments {r['variantes']['retenue']}s.", ""]
        out += [
                "Probabilité de choisir l'option du pôle + (préférence de lettre annulée), "
                "sans rien dire de son état.", "",
                "| Sentiment injecté | pôle − | neutre | pôle + | penchant gagné | par choix | "
                "rang | choix les plus touchés |", "|---|---|---|---|---|---|---|---|"]
        for a, t in b["par_axe"].items():
            each = " / ".join(f"{x:+.1f}" for x in t.get("par_choix", []))
            out.append(f"| {a} | {t['moins']:.2f} | {t['neutre']:.2f} | {t['plus']:.2f} | "
                       f"{t['effet']:+.2f} | {each} | {t['rang_de_son_choix']} | "
                       f"{', '.join(t['plus_touches'])} |")
        out += ["", f"Penchant ≥ {r['seuils']['effet_min']} : {b['axes_avec_effet']}/11. "
                f"Spécifiques : {b['axes_specifiques']}/11. À l'envers : "
                f"{', '.join(b['a_l_envers']) or 'aucun'}. Témoin (direction sans sens, même "
                f"force) : {b['temoin_hasard_moyen']} de penchant déplacé en moyenne.", "",
                "### Dose", "",
                "| alpha | " + " | ".join(r["dose"][0]["effet"]) + " |",
                "|---|" + "---|" * len(r["dose"][0]["effet"])]
        out += [f"| {d['alpha']} | " + " | ".join(f"{x:+.1f}" for x in d["effet"].values())
                + " |" for d in r["dose"]]
    if "interactions" in r:
        out += ["", "## 4. Sentiments mêlés", "",
                "Penchant gagné sur chacun des deux choix (seul A / seul B / ensemble).", "",
                "| Paire | Choix | seul A | seul B | ensemble | nature |",
                "|---|---|---|---|---|---|"]
        for p in r["interactions"]:
            for x in p["seul_a"]:
                out.append(f"| {' + '.join(p['paire'])} | {x} | {p['seul_a'][x]:+.2f} | "
                           f"{p['seul_b'][x]:+.2f} | {p['ensemble'][x]:+.2f} | "
                           f"{p['nature'][x]} |")
    if "stabilite" in r:
        st = r["stabilite"]
        out += ["", "## 5. Stabilité", "",
                "Penchant gagné entre les deux pôles. « Cache total » : tout le prompt lu avant "
                "le changement d'humeur. « Cache réel » : ancien historique en cache, dernier "
                "message relu (ce qui se passe en conversation).", "",
                "| Axe | court | longue conversation | cache total | cache réel | tenue |",
                "|---|---|---|---|---|---|"]
        out += [f"| {a} | {t['court']:+.2f} | {t['longue_conversation']:+.2f} | "
                f"{t['cache_total']:+.2f} | {t['cache_reel']:+.2f} | "
                f"{t['tenue_cache_reel'] if t['tenue_cache_reel'] is not None else '—'} |"
                for a, t in st["par_axe"].items()]
        out += ["", f"Tenue moyenne en cache réel : {st['tenue_moyenne_cache_reel']}."]
    if "interpretation" in r:
        it = r["interpretation"]
        events = list(next(iter(it["penchant_vers_la_lecture_sombre"].values())))
        out += ["", "## 6. Interprétation d'un événement ambigu", "",
                "Penchant vers la lecture sombre (positif = il y voit un problème).", "",
                "| État | " + " | ".join(e[:38] for e in events) + " | moyenne |",
                "|---|" + "---|" * (len(events) + 1)]
        for name, t in it["penchant_vers_la_lecture_sombre"].items():
            out.append(f"| {name} | " + " | ".join(f"{t[e]:+.2f}" for e in events)
                       + f" | {it['moyenne'][name]:+.2f} |")
        for m in it["souvenir"]:
            out += ["", f"Avec le souvenir d'un précédent fâcheux — « {m['evenement']} » : "
                    + ", ".join(f"{k} {x:+.2f}" for k, x in m.items() if k != "evenement")]
    if "journee" in r:
        j = r["journee"]
        out += ["", "## 7. Pilotage par le cœur", "",
                "**Un sentiment à la fois** (cinq niveaux, de −1 à +1 fois sa dose) : son propre "
                "choix suit-il le niveau ?", "",
                "| Sentiment | suivi | penchant aux cinq niveaux | monotone |", "|---|---|---|---|"]
        one = j.get("suivi_un_par_un") or {"par_axe": {}, "moyenne": None}
        out += [f"| {a} | {t['suivi']:+.2f} | " + " / ".join(f"{x:+.1f}" for x in t["penchants"])
                + f" | {'oui' if t['monotone'] else 'non'} |" for a, t in one["par_axe"].items()]
        mv = j.get("suivi_multivarie") or {}
        out += ["", f"Moyenne : {_f(one['moyenne'])}.", "",
                "**État entier, sur une journée simulée** (le cœur reçoit compliment, jeu, "
                "agacement, menace, réconfort, perte, solitude, nouveauté). On compare chaque "
                "choix à ce que l'ensemble des sentiments du moment prédit, d'après l'effet de "
                "chacun mesuré séparément.", "",
                "| Condition | Suivi multivarié | Ancienne mesure (axe par axe) |", "|---|---|---|",
                f"| A. Gemma seul | {_f((mv.get('A') or {}).get('moyenne'))} | "
                f"{_f(j['suivi_moyen'].get('A'))} |",
                f"| A'. Gemma + état décrit en texte | "
                f"{_f((mv.get('A_texte') or {}).get('moyenne'))} | "
                f"{_f(j['suivi_moyen'].get('A_texte'))} |",
                f"| B. Injection fixe | {_f((mv.get('B') or {}).get('moyenne'))} | "
                f"{_f(j['suivi_moyen'].get('B'))} |",
                f"| C. Injection pilotée par le cœur | "
                f"{_f((mv.get('C') or {}).get('moyenne'))} | {_f(j['suivi_moyen'].get('C'))} |",
                "", "L'ancienne mesure comparait chaque sentiment à son seul choix, sans tenir "
                "compte des autres sentiments présents au même moment. Elle est gardée ici pour "
                "mémoire ; le critère porte sur les deux nouvelles (seuil inchangé : "
                f"{r['seuils']['suivi_min']}).", ""]
        per = (mv.get("C") or {}).get("par_choix") or {}
        if per:
            out += ["Suivi multivarié par choix : "
                    + ", ".join(f"{a} {x:+.2f}" for a, x in per.items()) + ".", ""]
        out += [
                "| Moment | Émotion du cœur | Sentiments injectés |", "|---|---|---|"]
        out += [f"| {m['moment']} | {m['emotion']} | "
                + (", ".join(f"{k} {x:+.1f}" for k, x in m["sentiments"].items()) or "—") + " |"
                for m in j["moments"]]
        out += ["", "Ancienne mesure, axe par axe :", "",
                "| Axe | A' texte | C pilotée |", "|---|---|---|"]
        out += [f"| {a} | {_f(j['suivi_par_axe']['A_texte'][a])} | "
                f"{_f(j['suivi_par_axe']['C'][a])} |" for a in j["axes_qui_bougent"]]
    if v["echecs"]:
        out += ["", "## Échecs", ""] + [f"- {e}" for e in v["echecs"]]
    return "\n".join(out) + "\n"


def run(cfg: Any, engine: Engine | None = None, say: Any = _say, full: bool = False
        ) -> dict[str, Any]:
    return Bench(cfg, engine=engine, say=say, full=full).run()
