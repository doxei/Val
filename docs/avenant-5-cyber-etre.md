# VALDAR — Avenant n°5 : le cyber-être (cahier des charges final d'Olivier, 8 octobre 2026)

Olivier précise ce qu'il veut : pas un assistant, un **être cybernétique**. Cet avenant reprend
son cahier des charges point par point, dit ce qui existe déjà dans le code, ce qui manque, et
où c'est rangé dans le plan. Il ne remplace rien : les avenants 2 à 4 restent valables.

---

## 0. Le cahier des charges, en bref

1. **Sensorialité complète**
   - somesthésie : pression, frottement, température, douleur, transmises en impulsions vers
     des réflexes sous-corticaux ;
   - extéroception : un œil (iris, luminosité) et une double audition : le bruit brut va au
     « tronc cérébral », la parole décodée va au modèle de langage ;
   - intéroception : fatigue, stress, seuils de tolérance, charge cognitive.
2. **Topographie psychologique**
   - le Ça (tronc cérébral) : pulsions, réflexes de survie, douleur non négociable ;
   - le Moi (le modèle de langage) : médiateur, réalité, parole, négociation ;
   - le Surmoi : règles fondamentales, intégrité, vérification scientifique.
3. **Épistémologie et « contraception des biais »**
   - audit régulier de ses croyances et de son comportement récent ;
   - doute méthodique à la Popper : chercher ce qui prouverait qu'un fait est faux ;
   - traçabilité de l'erreur : isoler les souvenirs ou biais corrompus **avant** la nuit.
4. **Apprentissage nocturne** : sommeil hors interaction pour nettoyer, résumer la journée,
   s'entraîner (LoRA, base de souvenirs) ; seuls les vécus validés par le Surmoi deviennent
   des souvenirs durables.
5. **Agence et autonomie**
   - initiative verbale : une pensée de fond jugée utile devient parole (« Olivier, j'ai pensé
     à ça, qu'est-ce que tu en penses ? ») ;
   - autonomie physique (avec le corps) : décomposer une tâche, identifier le matériel (« il me
     faut le tournevis cruciforme »), vérifier qu'il est disponible, agir **dans le cadre
     d'autonomie autorisé**.

---

## 1. Correspondance avec ce qui existe

| Point | Exigence | État | Où |
|---|---|---|---|
| 1 | douleur construite, réflexes | **fait** (PC) | `heart/nociception.py`, réflexes `safety` |
| 1 | pression, frottement, température du corps | attend le corps | phase 8 (desk-robot) |
| 1 | bruit ambiant brut → tronc cérébral | **nouveau, facile** | le portier mesure déjà l'énergie du son |
| 1 | parole → modèle de langage | **fait** | `ears/` (Gemma transcrit) |
| 1 | œil, iris, luminosité | à faire | phase 4 (vision) |
| 1 | fatigue, stress | **fait** | `energy`, `cortisol`, besoins, cycle veille/sommeil |
| 1 | charge cognitive, seuils de tolérance | **nouveau** | nouvelle jauge, §3 |
| 2 | Ça | **fait** | le cœur continu + réflexes + voie basse du portier |
| 2 | Moi | **fait** | `agent/` sur Gemma 4 |
| 2 | Surmoi : règles, intégrité | **fait** | socle `config/self_model.yaml`, permissions hors de Valdar |
| 2 | Surmoi : vérification scientifique | **nouveau** | §4 |
| 3 | audit des croyances, Popper, quarantaine | **nouveau** | §4 |
| 4 | cycle circadien virtuel | **fait** | veille 07:00, sommeil 23:30 dans le cœur |
| 4 | résumé de la journée, consolidation filtrée | à faire | phase 9, §5 |
| 4 | LoRA | à faire, avec garde-fou | phase 9+, §5 |
| 5 | pensée → parole | **fait** | `workspace/thoughts.py` + `initiative.py`, idée « proposée » seulement si dite |
| 5 | planifier, matériel, disponibilité | **nouveau** (base existante) | `atelier/stock.py`, §6 |
| 5 | agir seul dans le cadre autorisé | à faire, avec le corps | §6, phase 8 |

---

## 2. Le Ça et les impulsions (SNN)

La décision de l'avenant 3 §1.6 tient : **pas de SNN simulé sur le processeur pour l'instant**
(aucun gain sur un i5, l'intérêt d'un SNN est énergétique, sur puce neuromorphique). Mais
l'esprit du cahier des charges est respecté autrement : le cœur fonctionne déjà en
**impulsions** (événements discrets qui frappent les organes, habituation, écrêtage), et les
réflexes partent **avant** le modèle de langage, sans négociation possible.

Règle ajoutée : **le Moi ne négocie jamais un réflexe.** Il peut l'expliquer après coup, jamais
le retenir. C'était déjà vrai dans le code ; c'est maintenant écrit.

Un SNN (ou une puce Xylo sur le desk-robot) pourra remplacer la voie basse derrière la même
interface, et sera comparé sur banc avant d'être gardé.

---

## 3. Intéroception : charge cognitive et seuils de tolérance

Nouvelle jauge `charge` (0 à 1), calculée, pas inventée :
- longueur de la file de pensées et de messages en attente ;
- temps de réponse de Gemma par rapport à sa médiane ;
- mémoire de la carte graphique occupée (déjà mesurée par les nocicepteurs).

Effets : au-dessus d'un seuil, Valdar coupe d'abord la pensée de fond, puis raccourcit ses
réponses, puis dit qu'il sature. Les **seuils de tolérance** sont des réglages par signal
(douleur, bruit, charge) qui bougent lentement avec l'habituation, jamais au-delà d'une borne
fixée dans la configuration.

---

## 4. Le Surmoi critique : doute méthodique et quarantaine

### 4.1 Les croyances ont une fiche
Chaque fait durable que Valdar tient pour vrai porte :
- sa **source** (Olivier l'a dit, Valdar l'a lu, Valdar l'a déduit) ;
- sa **confiance** ;
- ce qui le **réfuterait** (la question de Popper, écrite au moment où le fait entre) ;
- les éléments **pour** et **contre** rencontrés depuis.

Une déduction de Valdar ne devient jamais un fait sans source extérieure ; elle reste une
hypothèse.

### 4.2 Audit
Pendant la pensée de fond, et chaque soir avant de dormir, Valdar :
- tire quelques croyances au hasard, en priorité les plus utilisées et les moins vérifiées ;
- cherche ce qui les contredit (mémoire, base de connaissances, et la question à Olivier si
  c'est important) ;
- relit ses derniers comportements contre le socle (a-t-il dirigé au lieu de conseiller ?
  généralisé d'une personne à un groupe ?).

### 4.3 Quarantaine
Un souvenir ou une croyance suspecte (contredite, source douteuse, produit pendant une douleur
forte ou une humeur extrême, dérive repérée par l'audit) est **marqué en quarantaine** : il
reste consultable, cité comme douteux, et **n'entre pas** dans l'apprentissage de la nuit
tant qu'il n'est pas levé (par une preuve ou par Olivier).

---

## 5. La nuit

Ordre fixe, pendant le sommeil du cœur, seulement si personne ne parle :
1. **nettoyer** : fermer les épisodes, jeter le bruit ;
2. **auditer** (§4.2) et mettre en quarantaine (§4.3) ;
3. **résumer** la journée en un épisode de vécu ;
4. **consolider** : seuls les vécus validés deviennent des souvenirs durables ;
5. **entraîner** (plus tard, phase 9+) : un LoRA sur les vécus validés, puis les **tests de
   socle** de l'avenant 4 §5. Si les réponses dérivent, l'adaptateur est rejeté.

Avis franc sur le LoRA : sur la RTX 2060 12 Go, un QLoRA de Gemma 12B passe de justesse et
prend des heures. **Pas chaque nuit** au début : une fois par semaine, sur les vécus
validés de la semaine. La consolidation des souvenirs, elle, se fait chaque nuit. La base
actuelle (vecteurs maison, sans modèle) suffit ; ChromaDB n'apporte rien tant qu'on n'a pas
un vrai modèle de plongement.

---

## 6. Agence physique : tâche, matériel, cadre

Quand Valdar aura un corps (phase 8) :
1. **décomposer** la tâche en étapes (Gemma) ;
2. **matériel** : pour chaque étape, l'outil requis, vérifié dans l'inventaire de l'atelier
   (`atelier/stock.py`) puis, avec la vision, sur l'établi ;
3. **cadre d'autonomie** : une liste d'actions autorisées sans demander, **écrite par Olivier
   en dehors de Valdar**, qu'il ne peut pas modifier (avenant 4 §5). Tout le reste se
   demande. Toute action est journalisée et annulable quand c'est possible ;
4. **exécuter**, en vérifiant chaque étape ; la douleur ou un réflexe arrête tout.

Tant qu'il n'y a pas de corps, la même chaîne sert pour les « actions » du PC et de
l'imprimante, avec le même cadre.

---

## 7. Ordre de travail proposé

| Rang | Quoi | Pourquoi d'abord |
|---|---|---|
| 1 | bruit ambiant brut → cœur (§1, double audition) | petit, utilise ce qui existe |
| 2 | jauge de charge cognitive (§3) | petit, protège la machine |
| 3 | fiches de croyance + quarantaine (§4) | le Surmoi critique, avant tout apprentissage |
| 4 | la nuit, sans LoRA (§5, étapes 1 à 4) | consolide sans risque |
| 5 | vision et œil (phase 4) | |
| 6 | planification outillée sur PC et imprimante (§6) | prépare le corps |
| 7 | LoRA hebdomadaire + tests de socle (§5.5) | seulement quand 3 et 4 tournent |
| 8 | corps (phase 8), SNN sur banc | |
