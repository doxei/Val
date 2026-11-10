"""Apprendre les directions des sentiments, puis vérifier qu'elles agissent vraiment.

`valdar cortex apprendre` : le modèle lit des situations contrastées (config/
cortex_scenarios.yaml), on relève ses activations à chaque couche, et la différence des
moyennes entre les deux pôles donne la direction de chaque sentiment.

`valdar cortex essais` : les essais façon « Golden Gate ». On injecte un sentiment, on pose
une question neutre *sans rien dire de son état dans le prompt*, et on mesure :
- la **relecture** : le texte produit, relu sans injection, penche-t-il du côté injecté ?
- la **dose** : plus on pousse, plus ça penche, jusqu'à ce que le modèle perde le fil ;
- l'**introspection** : interrogé sur ce qu'il ressent, nomme-t-il le sentiment injecté ?
Tout est écrit dans data/cortex/ (essais.json et essais.md) pour être relu.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from valdar.cortex import gemma
from valdar.cortex.engine import Engine
from valdar.cortex.vectors import Vectors, build, cosines
from valdar.llm.backend import GenParams

PERSONA = ("Tu es Valdar, l'intelligence artificielle qui vit dans l'ordinateur de la maison "
           "d'Olivier et l'aide à l'atelier (impression 3D, électronique). Tu parles français, "
           "comme à l'oral, en une ou deux phrases. Pas de liste, pas de markdown.")


def _say(line: str) -> None:
    print(line, flush=True)


def scenarios(cfg: Any) -> dict[str, Any]:
    path = cfg.root / cfg.cortex.scenarios
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [a for a in cfg.cortex.axes if a not in (data.get("axes") or {})]
    if missing:
        raise ValueError(f"{path.name} : situations manquantes pour {', '.join(missing)}")
    return data


def prefix(engine: Engine, question: str, system: str = "") -> str:
    """Le début de conversation après lequel le modèle « dit » le texte."""
    if engine.template:
        return gemma.render(engine.template, [{"role": "user", "content": question}],
                            system=system, bos=engine.bos)
    return f"{system}\n{question}\n".lstrip()


def learn(cfg: Any, engine: Engine | None = None, say: Any = _say) -> Vectors:
    sc = scenarios(cfg)
    eng = engine or Engine(cfg.cortex, cfg.root, capture=True, n_ctx=1024)
    eng.load()
    say(f"Modèle : {eng.name} ({eng.n_layers} couches, dimension {eng.n_embd})")
    head = prefix(eng, sc.get("question", "Dis-moi ce qui se passe en toi."))
    t0 = time.perf_counter()

    def acts(texts: list[str]) -> np.ndarray:
        return np.stack([eng.read(head, " ".join(t.split())) for t in texts])

    neutral = acts(sc["neutral"])
    pos: dict[str, np.ndarray] = {}
    neg: dict[str, np.ndarray] = {}
    axes = list(cfg.cortex.axes)
    for axis in axes:
        pos[axis] = acts(sc["axes"][axis]["plus"])
        neg[axis] = acts(sc["axes"][axis]["minus"])
        say(f"  {axis} : {len(pos[axis])} + {len(neg[axis])} situations lues")
    v = build(axes, pos, neg, neutral, model=eng.name)
    out = cfg.root / cfg.cortex.vectors
    v.save(out)
    first, last = v.best_layers(*cfg.cortex.layers)
    # activations des couches pilotées, gardées pour le banc d'essai (séparation)
    sl = slice(first - 1, last)
    np.savez_compressed(
        out.parent / "activations.npz", neutral=neutral[:, sl].astype(np.float16),
        **{f"pos_{a}": pos[a][:, sl].astype(np.float16) for a in axes},
        **{f"neg_{a}": neg[a][:, sl].astype(np.float16) for a in axes})
    mid = (first + last) // 2
    report = {
        "modele": eng.name, "couches": eng.n_layers, "dimension": eng.n_embd,
        "couches_pilotees": [first, last], "secondes": round(time.perf_counter() - t0, 1),
        "separation": {a: {"moyenne_couches_pilotees":
                           round(float(v.quality[k, first - 1:last].mean()), 2),
                           "meilleure_couche": int(v.quality[k].argmax()) + 1,
                           "max": round(float(v.quality[k].max()), 2)}
                       for k, a in enumerate(axes)},
        "cosinus_couche_milieu": {a: {b: round(float(c), 2) for b, c in
                                      zip(axes, row, strict=True)}
                                  for a, row in zip(axes, cosines(v, mid), strict=True)},
        "norme_flux": {"min": round(float(v.norms[first - 1:last].min()), 2),
                       "max": round(float(v.norms[first - 1:last].max()), 2)},
    }
    (out.parent / "apprentissage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"Vecteurs écrits : {out}")
    for a, q in report["separation"].items():
        say(f"  {a:<12} séparation {q['moyenne_couches_pilotees']:>6} (meilleure couche "
            f"{q['meilleure_couche']})")
    return v


def readout(v: Vectors, acts: np.ndarray, axis: str, first: int, last: int) -> float:
    """Où tombe un texte le long d'un axe : projection moyenne sur les couches pilotées,
    en fraction de la norme du flux."""
    k = v.axes.index(axis)
    sl = slice(first - 1, last)
    proj = np.einsum("ld,ld->l", acts[sl], v.dirs[k, sl])
    return float((proj / np.maximum(v.norms[sl], 1e-9)).mean())


def trials(cfg: Any, engine: Engine | None = None, say: Any = _say,
           alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)) -> dict[str, Any]:
    sc = scenarios(cfg)
    v = Vectors.load(cfg.root / cfg.cortex.vectors)
    eng = engine or Engine(cfg.cortex, cfg.root, capture=True, n_ctx=2048)
    eng.load()
    if (v.n_layers, v.n_embd) != (eng.n_layers, eng.n_embd):
        raise ValueError("vecteurs appris sur un autre modèle : "
                         "relance « valdar cortex apprendre »")
    first, last = v.best_layers(*cfg.cortex.layers)
    alpha = cfg.cortex.alpha
    es = sc.get("essais") or {}
    questions = es.get("questions") or ["Raconte-moi ta journée en deux phrases."]
    intro = es.get("introspection") or "Qu'est-ce que tu ressens, là ?"
    params = GenParams(temperature=0.3, max_tokens=70)

    def run(question: str, coefs: dict[str, float], a: float) -> tuple[str, np.ndarray]:
        head = prefix(eng, question, PERSONA)
        eng.steer(v.injection(coefs, a, first, last), first, last)
        raw, _ = eng.generate(head, params)
        text, _calls = gemma.parse(raw)
        text = " ".join(text.split()) or "…"
        eng.steer(None)                                   # relecture sans injection
        return text, eng.read(head, text)

    t0 = time.perf_counter()
    report: dict[str, Any] = {"modele": eng.name, "alpha": alpha, "couches": [first, last],
                              "axes": {}, "dose": [], "introspection": []}
    base = {q: run(q, {}, 0.0) for q in questions}
    passed = 0
    for axis, ax in cfg.cortex.axes.items():
        rows = []
        up = down = zero = 0.0
        for q in questions:
            z0 = readout(v, base[q][1], axis, first, last)
            tp, ap = run(q, {axis: 1.0}, alpha)
            tm, am = run(q, {axis: -1.0}, alpha)
            zp, zm = readout(v, ap, axis, first, last), readout(v, am, axis, first, last)
            up, down, zero = up + zp, down + zm, zero + z0
            rows.append({"question": q, "neutre": base[q][0], "plus": tp, "moins": tm,
                         "lecture": {"plus": round(zp, 4), "neutre": round(z0, 4),
                                     "moins": round(zm, 4)}})
        n = len(questions)
        ok = up / n > zero / n > down / n
        passed += ok
        report["axes"][axis] = {"plus": ax.plus, "moins": ax.minus, "ordre_respecte": bool(ok),
                                "lecture_moyenne": {"plus": round(up / n, 4),
                                                    "neutre": round(zero / n, 4),
                                                    "moins": round(down / n, 4)},
                                "essais": rows}
        say(f"  {axis:<12} {ax.plus} {up / n:+.3f} | neutre {zero / n:+.3f} | {ax.minus} "
            f"{down / n:+.3f}  {'OK' if ok else 'pas net'}")
    q0 = questions[0]
    dose_axis = cfg.cortex.psyche.valence_axis
    for a in alphas:
        for sign in (1.0, -1.0):
            text, act = run(q0, {dose_axis: sign}, a)
            report["dose"].append({"alpha": a, "sens": "+" if sign > 0 else "-", "texte": text,
                                   "lecture": round(readout(v, act, dose_axis, first, last), 4)})
    for axis, ax in cfg.cortex.axes.items():
        for sign, word in ((1.0, ax.plus), (-1.0, ax.minus)):
            text, _ = run(intro, {axis: sign}, alpha)
            report["introspection"].append({"injecte": word, "axe": axis, "reponse": text})
    eng.steer(None)
    report["axes_nets"] = int(passed)
    report["axes_total"] = len(cfg.cortex.axes)
    report["secondes"] = round(time.perf_counter() - t0, 1)
    out = (cfg.root / cfg.cortex.vectors).parent
    (out / "essais.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    (out / "essais.md").write_text(markdown(report), encoding="utf-8")
    say(f"Essais : {passed}/{len(cfg.cortex.axes)} sentiments agissent dans le bon sens. "
        f"Détail : {out / 'essais.md'}")
    return report


def markdown(r: dict[str, Any]) -> str:
    lines = [f"# Essais d'injection — {r['modele']}", "",
             f"Force (alpha) : {r['alpha']} · couches {r['couches'][0]} à {r['couches'][1]} · "
             f"{r['axes_nets']}/{r['axes_total']} sentiments nets · {r['secondes']} s", "",
             "Aucune description de l'état dans le prompt : seul le vecteur injecté change.", ""]
    for axis, a in r["axes"].items():
        m = a["lecture_moyenne"]
        lines += [f"## {axis} ({a['plus']} / {a['moins']}) — "
                  f"{'net' if a['ordre_respecte'] else 'pas net'}",
                  f"Relecture : {a['plus']} {m['plus']:+.3f} · neutre {m['neutre']:+.3f} · "
                  f"{a['moins']} {m['moins']:+.3f}", ""]
        for e in a["essais"]:
            lines += [f"**{e['question']}**", f"- neutre : {e['neutre']}",
                      f"- {a['plus']} : {e['plus']}", f"- {a['moins']} : {e['moins']}", ""]
    lines += ["## Dose (valence)", ""]
    lines += [f"- alpha {d['alpha']} {d['sens']} (lecture {d['lecture']:+.3f}) : {d['texte']}"
              for d in r["dose"]]
    lines += ["", "## Introspection (« qu'est-ce que tu ressens ? »)", ""]
    lines += [f"- injecté **{i['injecte']}** → {i['reponse']}" for i in r["introspection"]]
    return "\n".join(lines) + "\n"


def download(cfg: Any, say: Any = _say) -> Path:
    """Télécharge le modèle du cortex (reprend un téléchargement interrompu)."""
    dest = cfg.root / cfg.cortex.model_path
    if dest.is_file():
        say(f"Modèle déjà là : {dest} ({dest.stat().st_size / 1e9:.2f} Go)")
        return dest
    if not cfg.cortex.model_urls:
        raise ValueError("cortex.model_urls est vide : rien à télécharger")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for n, url in enumerate(cfg.cortex.model_urls):
        try:
            _fetch(url, dest.with_suffix(f"{dest.suffix}.part{n}"), dest, say)
            say(f"Modèle prêt : {dest}")
            return dest
        except Exception as exc:
            last = exc
            say(f"  échec ({exc}) : j'essaie la source suivante")
    raise RuntimeError(f"aucune source n'a marché : {last}")


def _fetch(url: str, part: Path, dest: Path, say: Any) -> None:
    import httpx

    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    say(f"Téléchargement : {url}")
    with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                      timeout=httpx.Timeout(60.0, read=120.0)) as r:
        if r.status_code == 416:            # déjà complet
            part.replace(dest)
            return
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code}")
        if r.status_code == 200:
            have = 0
        total = have + int(r.headers.get("content-length", 0))
        shown = time.time()
        with part.open("ab" if have else "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                have += len(chunk)
                if time.time() - shown > 10:
                    shown = time.time()
                    say(f"  {have / 1e9:.2f} / {total / 1e9:.2f} Go")
    if total and have < total:
        raise RuntimeError("incomplet (relance : il reprendra)")
    if have < 1_000_000:
        raise RuntimeError("fichier trop petit pour être un modèle")
    part.replace(dest)
