"""Outils de la phase 2 : ceux de RAUB sans risque, plus la lecture de son propre état.

Chaque outil reçoit ses dépendances par le `ToolContext` (aucun état global).
"""
from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import dataclass
from typing import Any

from valdar.atelier import Checklist, Pinouts, Reminders, Stock, parse_when
from valdar.devices.moonraker import Moonraker, PrinterError, describe
from valdar.memory import Facts
from valdar.tools.registry import ELEVATED, SAFE, SAFETY, Registry, Tool, params

_JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
         "septembre", "octobre", "novembre", "décembre"]


@dataclass
class ToolContext:
    heart: Any            # valdar.heart.Heart (verrouillé par l'appelant)
    facts: Facts
    stock: Stock
    checklist: Checklist
    reminders: Reminders
    pinouts: Pinouts
    printer: Moonraker | None
    person: str = "olivier"
    lock: Any = None      # verrou du cœur (lecture cohérente de l'état)
    knowledge: Any = None     # valdar.knowledge.Knowledge
    printwatch: Any = None    # valdar.printwatch.PrintWatch
    thoughts: Any = None      # valdar.workspace.thoughts.Thoughts
    memory: Any = None        # valdar.memory.Episodic
    relations: Any = None     # valdar.relations.Relations


def build_registry(ctx: ToolContext) -> Registry:
    reg = Registry()
    s = {"type": "string"}

    # ------------------------------------------------------------------ soi
    def t_heure() -> str:
        lt = time.localtime()
        return (f"{_JOURS[lt.tm_wday]} {lt.tm_mday} {_MOIS[lt.tm_mon - 1]} {lt.tm_year}, "
                f"{lt.tm_hour:02d}:{lt.tm_min:02d}")

    def t_systeme() -> str:
        du = shutil.disk_usage("C:\\" if os.name == "nt" else "/")
        parts = [f"{platform.system()} {platform.release()}",
                 f"disque {du.used / 1e9:.0f}/{du.total / 1e9:.0f} Go"]
        try:
            import psutil

            vm = psutil.virtual_memory()
            parts.insert(1, f"processeur {psutil.cpu_percent(interval=0.3):.0f} %, "
                            f"mémoire {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} Go")
        except ImportError:
            pass
        return " · ".join(parts)

    def t_mon_etat() -> str:
        h = ctx.heart
        with ctx.lock or _NoLock():
            b = h.brief()
            felt = "; ".join(h.felt().values())
            needs = ", ".join(f"{k} {v:.2f}" for k, v in h.needs.items())
            energy = h.variables["energy"]
        return (f"émotion {b['emotion']} (intensité {b['intensity']:.2f}), humeur {b['mood']}, "
                f"énergie {energy:.2f}, besoins : {needs}. Ressenti : {felt}.")

    reg.add(Tool("heure", "Date et heure locales.", params(), t_heure, SAFE, "soi"))
    reg.add(Tool("info_systeme", "État de l'ordinateur : processeur, mémoire, disque.",
                 params(), t_systeme, SAFE, "soi"))
    reg.add(Tool("mon_etat_interieur",
                 "Lit l'état réel de ton cœur (émotion, humeur, énergie, besoins, ressenti). "
                 "À utiliser quand on te demande précisément comment tu vas ou ce que tu ressens.",
                 params(), t_mon_etat, SAFE, "soi"))

    # ------------------------------------------------------------- mémoire
    def t_memoire(action: str = "relis", quoi: str = "", origine: str = "dit",
                  refutation: str = "") -> str:
        if action == "note":
            return ctx.facts.remember(quoi, person=ctx.person, origine=origine,
                                      refutation=refutation)
        if action == "oublie":
            return ctx.facts.forget(quoi)
        hits = ctx.facts.recall(quoi, person=ctx.person)
        if not hits:
            return "je ne sais rien là-dessus."
        return "\n".join(f"- {h['text']}" for h in hits)

    reg.add(Tool("se_souvenir",
                 "Mémorise un fait durable sur la personne ou le monde (action=note), relit ce "
                 "que tu sais (action=relis), ou oublie (action=oublie).",
                 params(["action"], action={"type": "string", "enum": ["note", "relis", "oublie"]},
                        quoi={**s, "description": "le fait à retenir, ou le sujet cherché"},
                        origine={"type": "string", "enum": ["dit", "lu", "deduit"],
                                 "description": "dit : on te l'a dit ; lu : une source ; "
                                                "deduit : ta propre déduction (hypothèse)"},
                        refutation={**s, "description": "ce qui prouverait que c'est faux"}),
                 t_memoire, SAFE, "mémoire"))

    # --------------------------------------- le Surmoi critique (avenant 5 §4)
    def t_douter(action: str, quoi: str, note: str = "") -> str:
        f = ctx.facts.find(quoi, person=ctx.person)
        if f is None:
            return "je n'ai pas de croyance là-dessus."
        if action == "quarantaine":
            ctx.facts.quarantine(f["id"], note or "douteux")
            return f"« {f['text']} » mis en quarantaine : je ne m'appuie plus dessus."
        g = ctx.facts.evidence(f["id"], action == "pour", note)
        if g is None:
            return "je n'ai pas de croyance là-dessus."
        etat = {"valide": "je le tiens pour vrai", "hypothese": "ça reste une hypothèse",
                "quarantaine": "je le mets en quarantaine"}[g["statut"]]
        return (f"« {g['text']} » : {g['pour']} pour, {g['contre']} contre, confiance "
                f"{g['confidence']:.0%} ; {etat}.")

    reg.add(Tool("douter",
                 "Doute méthodique : ajoute un élément pour ou contre une de tes croyances, "
                 "ou mets-la en quarantaine si elle te paraît douteuse. Cherche d'abord ce "
                 "qui la contredirait.",
                 params(["action", "quoi"],
                        action={"type": "string", "enum": ["pour", "contre", "quarantaine"]},
                        quoi={**s, "description": "la croyance concernée"},
                        note={**s, "description": "l'élément, ou le motif"}),
                 t_douter, SAFE, "mémoire"))

    def t_lever(quoi: str) -> str:
        f = next((x for x in ctx.facts.quarantined()
                  if x["id"] == (ctx.facts.find(quoi) or {}).get("id")), None)
        if f is None:
            hits = [x for x in ctx.facts.quarantined() if quoi.lower() in x["text"].lower()]
            f = hits[0] if hits else None
        if f is None or not ctx.facts.lift(f["id"]):
            return "rien en quarantaine là-dessus."
        return f"quarantaine levée pour « {f['text']} »."

    reg.add(Tool("lever_quarantaine",
                 "Lève la quarantaine d'une croyance (sur preuve, ou parce qu'Olivier le dit).",
                 params(["quoi"], quoi={**s, "description": "la croyance concernée"}),
                 t_lever, ELEVATED, "mémoire"))

    # ------------------------------------------------------------- atelier
    reg.add(Tool("etat_stock", "Contenu de la réserve de l'atelier (filtre optionnel).",
                 params(recherche={**s, "description": "mot à chercher (optionnel)"}),
                 lambda recherche="": ctx.stock.render(recherche), SAFE, "atelier"))
    reg.add(Tool("creer_stock", "Déclare un article et sa quantité dans la réserve.",
                 params(["nom", "quantite"], nom={**s, "description": "ex : 'vis M3x12'"},
                        quantite={"type": "number"}, unite=s,
                        localisation={**s, "description": "ex : 'tiroir B3'"},
                        seuil={"type": "number", "description": "alerte en dessous"}),
                 lambda nom, quantite, unite="", localisation="", seuil=0:
                 ctx.stock.set_item(nom, float(quantite), unite, localisation, float(seuil)),
                 SAFE, "atelier"))
    reg.add(Tool("ajouter_stock",
                 "Met à jour un article : delta négatif si consommé, positif si reçu.",
                 params(["nom", "delta"], nom=s, delta={"type": "number"}),
                 lambda nom, delta: ctx.stock.change(nom, float(delta)), SAFE, "atelier"))
    reg.add(Tool("creer_checklist",
                 "Crée la checklist de procédure affichée (étapes dans l'ordre). À utiliser dès "
                 "qu'Olivier décrit une suite d'étapes.",
                 params(["titre", "etapes"], titre=s,
                        etapes={"type": "array", "items": s}),
                 lambda titre, etapes: ctx.checklist.create(titre, list(etapes)), SAFE, "atelier"))
    reg.add(Tool("cocher_checklist", "Coche (ou décoche) des étapes par numéro ou par texte.",
                 params(["faites"], faites={"type": "array", "items": s},
                        annulees={"type": "array", "items": s}),
                 lambda faites, annulees=None: ctx.checklist.check(list(faites),
                                                                   list(annulees or [])),
                 SAFE, "atelier"))
    reg.add(Tool("etat_checklist", "État de la checklist active.", params(),
                 lambda: ctx.checklist.render(), SAFE, "atelier"))

    def t_rappel(texte: str, quand: str) -> str:
        due = parse_when(quand)
        if due is None:
            return (f"je n'ai pas compris le moment « {quand} ». Exemples : dans 10 minutes, "
                    "à 15h30, demain à 9h, à midi.")
        r = ctx.reminders.add(texte, due)
        when = time.strftime("%d/%m à %H:%M", time.localtime(due))
        return f"rappel posé : « {r['texte']} » le {when}."

    def t_annuler_rappel(fragment: str) -> str:
        gone = ctx.reminders.cancel(fragment)
        if not gone:
            return f"aucun rappel ne contient « {fragment} »."
        return "annulé : " + ", ".join(r["texte"] for r in gone)

    reg.add(Tool("planifier_rappel",
                 "Rappel ou alarme. « quand » en français : dans 10 minutes, dans 2h30, à 15h30, "
                 "demain à 9h, à midi.",
                 params(["texte", "quand"], texte=s, quand=s), t_rappel, SAFE, "atelier"))
    reg.add(Tool("etat_rappels", "Rappels à venir.", params(),
                 lambda: ctx.reminders.render(), SAFE, "atelier"))
    reg.add(Tool("annuler_rappel", "Annule les rappels contenant un fragment de texte.",
                 params(["fragment"], fragment=s), t_annuler_rappel, SAFE, "atelier"))
    reg.add(Tool("pinout",
                 "Pinout d'une carte ou d'un composant (BTT Octopus, SKR, TMC2209...).",
                 params(["cible"], cible=s), lambda cible: ctx.pinouts.lookup(cible),
                 SAFE, "atelier"))

    # ------------------------------------------------------------ imprimante
    if ctx.printer is not None:
        pr = ctx.printer

        def t_fichiers() -> str:
            try:
                files = sorted(pr.files(), key=lambda f: f.get("modified", 0), reverse=True)
            except PrinterError as exc:
                return str(exc)
            if not files:
                return "aucun fichier d'impression."
            return f"{len(files)} fichier(s), les plus récents :\n" + "\n".join(
                f"- {f['path']} ({f.get('size', 0) / 1e6:.1f} Mo)" for f in files[:15])

        def t_pause() -> str:
            pr.pause()
            return "impression mise en pause."

        def t_urgence() -> str:
            pr.emergency_stop()
            return ("arrêt d'urgence envoyé : moteurs et chauffes coupés. Il faudra un "
                    "FIRMWARE_RESTART pour repartir.")

        reg.add(Tool("imprimante_etat",
                     "État réel de l'imprimante 3D : fichier, progression, températures.",
                     params(), lambda: pr.status_text(), SAFE, "imprimante"))
        reg.add(Tool("imprimante_fichiers", "Fichiers d'impression disponibles.", params(),
                     t_fichiers, SAFE, "imprimante"))
        reg.add(Tool("imprimante_pause", "Met l'impression en cours en pause.", params(),
                     t_pause, SAFETY, "imprimante"))
        reg.add(Tool("imprimante_arret_urgence",
                     "ARRÊT D'URGENCE de l'imprimante (M112) : seulement en cas de danger réel "
                     "(fumée, feu, collision, bruit anormal).", params(), t_urgence, SAFETY,
                     "imprimante"))

    # ---------------------------------------------------------- connaissances
    if ctx.knowledge is not None:
        def t_connaissance(sujet: str) -> str:
            hits = ctx.knowledge.search(sujet, k=4)
            if not hits:
                return "rien dans mes connaissances là-dessus."
            return "\n".join("- " + h.line(700) for h in hits)

        reg.add(Tool("connaissance_impression",
                     "Cherche dans mes connaissances sur l'impression 3D (défauts, signes "
                     "avant-coureurs, corrections, Klipper, CR-10S, détection).",
                     params(sujet=s), t_connaissance, SAFE, "connaissances"))

    # ------------------------------------------------------------------ vigie
    if ctx.printwatch is not None:
        pw = ctx.printwatch

        def t_vigie() -> str:
            return pw.status_text()

        def t_etiquette(resultat: str, minutes_avant_fin: str = "") -> str:
            failed = resultat.strip().lower() in ("rate", "raté", "ratee", "ratée", "echec",
                                                   "échec", "fail", "failed")
            try:
                mins = float(minutes_avant_fin) if minutes_avant_fin else None
            except ValueError:
                mins = None
            return pw.label(failed, minutes_before_end=mins, by=ctx.person)

        reg.add(Tool("vigie_impression",
                     "État de ma vigie d'impression : caméras, images analysées, score de "
                     "confiance (porte des 98 %), niveau d'autonomie.", params(), t_vigie,
                     SAFE, "imprimante"))
        reg.add(Tool("vigie_etiqueter",
                     "Note le résultat de la dernière impression terminée (« réussie » ou "
                     "« ratée ») et, si ratée, combien de minutes avant la fin c'était fichu. "
                     "C'est ce qui me permet d'apprendre à voir venir les échecs.",
                     params(["resultat"], resultat=s, minutes_avant_fin=s), t_etiquette, SAFE,
                     "imprimante"))

    # ------------------------------------------------------------------ idées
    if ctx.thoughts is not None:
        th = ctx.thoughts

        def t_idees() -> str:
            ideas = th.ideas(limit=6)
            if not ideas:
                return "pas d'idée en stock pour l'instant."
            return "\n".join(f"- [{i.id}] {i.line()}" for i in ideas)

        def t_idee_statut(numero: str, statut: str) -> str:
            st = {"adoptee": "adoptee", "adoptée": "adoptee", "rejetee": "rejetee",
                  "rejetée": "rejetee", "plus tard": "plus_tard"}.get(statut.strip().lower())
            if st is None:
                return "statut inconnu (adoptée, rejetée, plus tard)."
            th.mark(int(numero), st)
            return "noté."

        reg.add(Tool("mes_idees", "Les idées que j'ai eues en pensant de mon côté.", params(),
                     t_idees, SAFE, "soi"))
        reg.add(Tool("idee_statut", "Marque une de mes idées : adoptée, rejetée, ou plus tard.",
                     params(["numero", "statut"], numero=s, statut=s), t_idee_statut, SAFE,
                     "soi"))
    # ------------------------------------------------------------ relations
    if ctx.relations is not None:
        rel = ctx.relations

        def t_retenir_sur_toi(genre: str, texte: str) -> str:
            if not ctx.person:
                return "je ne sais pas qui tu es : dis-moi ton prénom d'abord."
            return rel.learn(ctx.person, genre, texte)

        def t_oublie_moi() -> str:
            who = ctx.person
            if not who:
                return "je ne sais pas qui tu es, je n'ai rien à oublier."
            rel.forget_person(who)
            n_facts = ctx.facts.forget_person(who)
            n_eps = ctx.memory.forget_person(who) if ctx.memory is not None else 0
            return (f"c'est fait : j'ai oublié la personne « {who} » ({n_facts} fait(s), "
                    f"{n_eps} conversation(s), et tout ce que j'avais appris d'elle).")

        reg.add(Tool("retenir_sur_toi",
                     "Retiens quelque chose sur la personne qui te parle, pour elle seule : "
                     "genre « aime » (ce qui lui fait du bien), « agace », « prefere » (comment "
                     "elle veut que tu lui parles) ou « note ».",
                     params(["genre", "texte"], genre=s, texte=s), t_retenir_sur_toi, SAFE,
                     "relations"))
        reg.add(Tool("oublie_moi",
                     "La personne qui te parle te demande de l'oublier : efface sa relation, "
                     "ses faits et ses conversations. Demande confirmation.",
                     params(), t_oublie_moi, SAFE, "relations", confirm=True))
    return reg


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def printer_line(printer: Moonraker | None) -> str:
    if printer is None:
        return "imprimante : non configurée"
    st = printer.cached_status()
    if st is None:
        return "imprimante : injoignable"
    return "imprimante : " + describe(st).replace("\n", " · ")
