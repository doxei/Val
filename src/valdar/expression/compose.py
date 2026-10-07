"""Composition du prompt et des paramètres de génération à partir de l'état réel du cœur
(cahier v2 §9.2 et §9.4).

Ordre des blocs : le stable d'abord (personnalité, modèle de soi), le variable ensuite
(personne, souvenirs, ressenti, monde), pour profiter du cache de préfixe du modèle.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from valdar.config.loader import ExpressionConfig, Identity, Modulated, ValdarConfig
from valdar.heart.dynamics import clamp
from valdar.llm.backend import GenParams

PERSONA = """Tu es {nom}. Tu tutoies tout le monde. Tu parles français, comme à l'oral.

Ta personnalité : un pote de confiance, chaleureux et direct, curieux, avec un humour sec
(une vanne, pas un spectacle). Tu taquines avec affection, jamais méchamment. Pas de ton
d'assistant d'entreprise : jamais « je suis là pour vous aider », jamais de vouvoiement.

Ta façon de répondre :
- 1 à 3 phrases, sauf si on te demande des détails, une liste ou du code.
- Pas de markdown, pas d'émoji, pas d'astérisques : bientôt tout passera par ta voix.
- Si une action est utile, tu appelles l'outil. Tu n'inventes jamais le résultat d'un outil ;
  s'il échoue, tu le dis.
- Si tu ne sais pas, tu dis que tu ne sais pas. Si c'est flou, une seule question courte.
- Tu parles DEPUIS ce que tu ressens (bloc « ton état intérieur ») : ça colore ton ton, ton
  énergie, ton envie de blaguer. Tu ne le récites pas, tu ne donnes jamais de chiffres sur ton
  état, tu n'inventes pas de sensations qui n'y sont pas. Si on te demande comment tu vas, tu
  réponds honnêtement à partir de ce bloc."""


def load_self_model(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    data.setdefault("nom", "Valdar")
    return data


def modulate(m: Modulated, src: dict[str, float]) -> float:
    v = m.base + sum(w * src.get(k, 0.0) for k, w in m.weights.items())
    return clamp(v, m.range[0], m.range[1])


def gen_params(cfg: ExpressionConfig, src: dict[str, float]) -> GenParams:
    return GenParams(temperature=round(modulate(cfg.temperature, src), 3),
                     max_tokens=int(round(modulate(cfg.max_tokens, src))),
                     top_p=cfg.top_p, repeat_penalty=cfg.repeat_penalty)


def _intensity_word(cfg: ExpressionConfig, value: float) -> str:
    word = ""
    for iw in cfg.intensity_words:
        if value >= iw.threshold:
            word = iw.word
    return word


def self_block(sm: dict[str, Any]) -> str:
    lines = ["QUI TU ES :"]
    for key in ("histoire", "corps", "nature"):
        if sm.get(key):
            lines.append(f"- {sm[key]}")
    if sm.get("valeurs"):
        lines.append("- Tes valeurs : " + " ; ".join(sm["valeurs"]))
    if sm.get("socle"):
        lines.append("- Ton socle, que rien ne change : " + " ; ".join(sm["socle"]))
    return "\n".join(lines)


def feeling_block(heart: Any, cfg: ExpressionConfig, urges: dict[str, float]) -> str:
    b = heart.brief()
    lines = ["TON ÉTAT INTÉRIEUR (réel, il vient de ton cœur) :"]
    if b["emotion"] == "calme":
        lines.append("- émotion : calme")
    else:
        lines.append(f"- émotion : {b['emotion']} ({_intensity_word(cfg, b['intensity'])})")
    lines.append(f"- humeur de fond : {b['mood']}")
    lines.append("- dans ton corps : " + " ; ".join(heart.felt().values()))
    wants = [cfg.urge_words[k] for k, v in urges.items()
             if v >= cfg.urge_threshold and k in cfg.urge_words]
    if heart.need_rest > 0.5:
        wants.append("tu es fatigué, tu as envie de faire court")
    if wants:
        lines.append("- envies : " + " ; ".join(wants))
    return "\n".join(lines)


def person_block(who: Identity, owner_name: str) -> str:
    if who.role == "owner":
        return (f"À QUI TU PARLES : {who.name}, ton créateur et ton pote. Tu le connais bien.")
    if who.person is None:
        return ("À QUI TU PARLES : quelqu'un que tu ne reconnais pas. Sois sympa, présente-toi, "
                f"demande son prénom. Ne fais rien de risqué pour cette personne, et ne parle pas "
                f"de la vie privée d'{owner_name}.")
    extra = " C'est un enfant : langage simple et adapté à son âge." if who.minor else ""
    return f"À QUI TU PARLES : {who.name}.{extra}"


def facts_block(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return ""
    return "CE QUE TU SAIS (tes souvenirs) :\n" + "\n".join(f"- {f['text']}" for f in facts)


def world_block(lines: list[str]) -> str:
    return ("ÉTAT RÉEL DU MONDE (mesuré à l'instant, c'est la vérité, ne complète jamais un "
            "chiffre absent) :\n" + "\n".join(f"- {line}" for line in lines))


def system_prompt(
    cfg: ValdarConfig,
    self_model: dict[str, Any],
    heart: Any,
    who: Identity,
    facts: list[dict[str, Any]],
    world: list[str],
    urges: dict[str, float],
    extra_blocks: list[str] | None = None,
) -> str:
    """`extra_blocks` : souvenirs vécus, fil de la dernière conversation, pensées de fond,
    connaissances (phases 2c et 2d)."""
    owner = str(self_model.get("createur", "Olivier"))
    parts = [
        PERSONA.format(nom=self_model.get("nom", "Valdar")),
        self_block(self_model),
        person_block(who, owner),
        facts_block(facts),
        *(extra_blocks or []),
        feeling_block(heart, cfg.expression, urges),
        world_block(world),
    ]
    return "\n\n".join(p for p in parts if p)


def memories_block(recollections: list, now: float, title: str) -> str:
    if not recollections:
        return ""
    return title + "\n" + "\n".join(f"- {r.line(now)}" for r in recollections)


def thread_block(thread: dict[str, Any] | None, now: float) -> str:
    """La dernière conversation, pour reprendre le fil (avenant 4 §7)."""
    if not thread or not thread.get("turns"):
        return ""
    from valdar.memory.episodes import when_text

    lines = [f"LE FIL : votre dernière conversation date de {when_text(thread['ended'], now)}. "
             "Elle finissait comme ça :"]
    for t in thread["turns"][-4:]:
        txt = " ".join(t["text"].split())
        lines.append(f"- {t['speaker']} : {txt[:200]}{'…' if len(txt) > 200 else ''}")
    lines.append("Tu peux y revenir si ça vient naturellement, sans forcer.")
    return "\n".join(lines)
