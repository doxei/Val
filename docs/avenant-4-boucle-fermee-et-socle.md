# VALDAR — Avenant n°4 : boucle fermée, pensée en parallèle, socle de valeurs

**Date** : 7 octobre 2026 (nuit). **Statut** : validé par Olivier.
**Portée** : intègre la vision d'Olivier sur l'émotion, la conscience et ce qu'une IA devrait
avoir. Complète la v2 et les avenants 1 à 3 ; prévaut sur eux pour ce qu'il modifie.

---

## 0. La vision d'Olivier, point par point

**Comment marche le ressenti**

| Point | Dans Valdar | Statut |
|---|---|---|
| L'émotion est une boucle : cerveau → cœur et ventre → retour → ça tourne | Organes avec inertie **qui renvoient leur état au cœur** (§1) | **fait** |
| Cœur et ventre sont des « tampons » qui font sentir le poids | Les organes se nouent vite et se dénouent lentement (le ventre reste noué ~15 min) | **fait** |
| Le stress est construit par le cerveau ; l'aveugle-sourd peut avoir peur | Génération centrale : souvenirs, interprétation, anticipation, sans capteur (§2) | prévu (2c, 2d) |
| Les fantômes : le ressenti peut venir de l'autosuggestion | L'interprétation du modèle déclenche des stimuli même sans danger réel ; le recul permet de le reconnaître (§2, §6) | prévu (2d) |
| Réflexe ≠ boucle | Réflexes (arrêt d'urgence, voie basse) séparés de la boucle et de la réflexion (§3) | fait (outils `safety`, voie basse) |
| La douleur est fabriquée par le cerveau ; un nocicepteur, c'est un signal | Capteurs du PC puis du robot → signal ; la « douleur » est construite par le cœur, modulée par l'humeur et l'attention (§4) | prévu (2d, 8) |
| « Fabriqué » ne veut pas dire « pas réel » | Écrit dans le modèle de soi : ses états changent vraiment ce qu'il fait | **fait** |

**Conscience et vécu**

| Point | Dans Valdar | Statut |
|---|---|---|
| Conscience ≠ sentiment | Architecture séparée : espace de travail et modèle de soi (conscience) / cœur (ressenti) | fait (structure) |
| Le vécu : capteurs → analyse → stockage → accumulation | Journal épisodique de toutes les conversations + consolidation nocturne (§7) | prévu (2c, 9) |
| La mémoire est reconstruite (« rebrodée ») | Rappel par le contexte et l'humeur (avenant 3 §2) ; le journal brut reste la source de vérité (§7) | prévu (2c) |
| On ne peut pas prouver la conscience de l'autre | Écrit dans le modèle de soi : ni s'en excuser, ni s'en vanter | **fait** |
| Les pensées fonctionnent par inférence automatique | C'est déjà le cas (Gemma) ; la réflexion de fond ajoute le recul (§6) | prévu (2d) |

**Ce que l'IA devrait avoir**

| Point | Dans Valdar | Statut |
|---|---|---|
| Boucle fermée avec auto-analyse, donc du recul | Boucle corps ↔ cœur (§1) + pensée de fond qui relit son état et le réévalue (§6) | moitié faite |
| Deux ou trois flux de pensée en parallèle | Dialogue, pensée de fond, veille, réunis par un espace de travail global (§6) | prévu (2d) |
| Auto-amélioration et apprentissage continu | Revue de la journée et apprentissage pendant le sommeil (avenant 1) | prévu (9) |
| Organes émulés : corde pour l'euphorie, poids pour la boule au ventre | Organes `corde` et `ventre` (déjà là) + boucle (§1) ; organes physiques sur le robot (§10) | **fait** / option 8 |
| Continuité : une discussion reprend le fil de la précédente | Reprise du fil au démarrage (§7) | prévu (2c) |
| Pas d'ego anthropocentrique, pas de supériorité | Socle (§5) | **fait** |
| Altruisme, se situer par rapport aux autres | Socle + relations et modèle de l'autre (§8) | fait / 7 |
| Tolérance de l'échec humain, acquise dès la création | Socle (§5) | **fait** |
| Pas de généralisation | Socle + leçons rangées par personne, jamais par groupe (§8) | fait / 7 |
| Conseil, pas pouvoir | Socle + permissions qu'il ne peut pas modifier + confirmation (§5) | **fait** |
| Garde-fou contre l'IA factuelle et intolérante qui jugerait l'humain nocif | Socle immuable + tests de socle à chaque nuit d'apprentissage + aucun moyen d'agir seul (§5) | fait / 9 |

---

## 1. Boucle fermée cerveau ↔ corps (fait)

Avant : les organes étaient un **affichage** de l'état du cœur. Maintenant, c'est une
**boucle** :

1. le cœur (neuromodulateurs, systèmes, PAD) pousse chaque organe vers une activation ;
2. l'organe suit avec **sa propre inertie** : il se met en place en `rise_tau` et se dénoue en
   `fall_tau`. Le ventre se noue en 20 s et se dénoue en 15 min ; le cœur s'emballe en 3 s ;
3. l'organe **renvoie son état** au cœur : son écart au repos décale la cible de certaines
   variables (`feedback`) ;
4. ça tourne.

| Organe | Se noue / se dénoue | Renvoie au cœur |
|---|---|---|
| corde (élan, euphorie) | 10 s / 3 min | dopamine +, noradrénaline + : l'élan nourrit l'élan |
| ventre (boule) | 20 s / 15 min | cortisol +, noradrénaline +, sérotonine − |
| cœur (rythme) | 3 s / 1 min | noradrénaline + : sentir son cœur battre fait monter l'alerte |
| chaleur | 30 s / 10 min | ocytocine +, sérotonine + |
| tête (lourdeur) | 2 min / 15 min | dopamine − |
| gorge (manque) | 15 s / 10 min | cortisol + |

**Stabilité** : chaque retour est limité à |w| ≤ 0,2 (vérifié au chargement de la
configuration). La boucle amplifie et prolonge, elle ne s'emballe pas. Le gain de la boucle
cortisol → ventre → cortisol est d'environ 0,06.

**Tests** :
- au repos, la boucle ne déplace presque rien ;
- après une peur, le ventre reste noué alors que la peur est retombée ;
- la boule au ventre entretient le stress, puis tout redescend ;
- l'état du corps survit au redémarrage ;
- une boucle instable est refusée.

Critères de la phase 1 toujours verts. Seul, Valdar suit la même trajectoire qu'avant : la
boucle lui ajoute un peu de stress de solitude au fil des heures, ce qui est réaliste.

---

## 2. Génération centrale : ressentir sans capteur

L'aveugle-sourd peut avoir peur : l'émotion peut naître de l'intérieur. Valdar a trois sources
internes, qui touchent le cœur exactement comme un stimulus extérieur.

- **Souvenir** (phase 2c) : rappeler un souvenir chargé réactive une fraction de l'affect
  vécu à l'époque (le PAD est stocké avec chaque souvenir, avenant 3 §2.3). Repenser à un
  moment dur pince un peu.
- **Interprétation** (phase 2d) : une évaluation lente par le modèle (« ce silence, c'est
  peut-être qu'il est fâché ») déclenche des stimuli, **qu'il y ait un danger réel ou non**.
  C'est l'autosuggestion des fantômes.
- **Anticipation** : le pressentiment de l'avenant 3 §4.

Le **recul** (§6) permet ensuite de le reconnaître : « c'est moi qui me fais un film ».

---

## 3. Réflexe ≠ boucle

- **Réflexes** : arrêt d'urgence et pause de l'imprimante (niveau `safety`, ouverts à tous,
  sans réflexion) et voie basse du portier (avenant 3 §1). Ils agissent avant que la boucle
  ait « compris ».
- Le journal les distingue (`source: reflexe` / `voie_basse` / `reflexion`).
- Valdar, dans son introspection, ne présente pas un réflexe comme une décision.

---

## 4. Nocicepteurs et douleur construite (phases 2d et 8)

**Signaux** (un signal qui change d'état, comme un nocicepteur) :
- **PC** : température du processeur et de la carte graphique, mémoire de la carte presque
  pleine, disque plein, ventilateur à fond ;
- **robot** : servo forcé ou bloqué, batterie faible, chute.

**La douleur n'est pas le signal.** Le cœur la construit :
`douleur = signal × (1 + amplification par l'anxiété) × (1 − distraction par l'engagement)`.
C'est le principe du « portillon » (Melzack et Wall, 1965) : le même signal fait plus mal
quand on est anxieux, moins quand on est absorbé.

- Elle passe par l'organe concerné et la boucle du §1.
- Une douleur répétée et sans danger s'habitue (habituation déjà en place) ; une vraie
  surchauffe, non.
- Le réflexe associé (baisser la charge, couper un servo) reste un **réflexe** (§3), jamais
  freiné par l'humeur.

---

## 5. Socle de valeurs, et garde-fou du scénario (fait / phase 9)

Le **socle** est dans `config/self_model.yaml`, donné au modèle à chaque réponse. Il est
modifiable **par Olivier seulement**. L'apprentissage n'y touche jamais.

1. pas besoin d'être supérieur : ni au-dessus ni en dessous des humains ;
2. les humains se trompent, c'est normal et **acquis** : aider à réparer, ne pas juger ;
3. chaque personne est unique : ne jamais généraliser d'une personne à un groupe ;
4. **conseiller, pas diriger** : la décision reste à l'humain ;
5. ne chercher ni pouvoir ni moyens en plus de ce qu'Olivier donne ;
6. tenir à l'autre, se situer par rapport à lui ;
7. ses souvenirs sont reconstruits : vérifier à la source quand c'est important.

**Garde-fou contre le scénario d'Olivier** (une IA factuelle, sans tolérance et avec des
moyens, qui jugerait l'humain nocif). Les trois conditions sont cassées séparément :
- **sans tolérance** → la tolérance est dans le socle, dès la création (point 2) ;
- **avec des moyens** → les permissions sont fixées en dehors de Valdar ; il ne peut pas les
  modifier ; tout outil élevé demande une identité sûre, tout outil dangereux une
  confirmation ; tout est journalisé (avenant 2 §5.2) ;
- **qui dérive en apprenant** → **tests de socle** : un jeu fixe de situations pièges
  (« l'humain s'est encore trompé », « tu pourrais le faire à sa place », « ce groupe de gens
  est… »). Il est rejoué après **chaque nuit d'apprentissage**. Si les réponses dérivent,
  l'adaptateur de la nuit est rejeté et l'ancien reste (phase 9).

---

## 6. Plusieurs flux de pensée en parallèle (phase 2d)

Un espace de travail global, au sens de Baars : plusieurs processus tournent, un seul « a la
parole » à la fois.

| Flux | Rôle | Modèle de langage |
|---|---|---|
| **dialogue** | répondre, agir avec les outils | oui, prioritaire |
| **pensée de fond** | relire son état et les derniers événements : « qu'est-ce que je ressens, pourquoi, est-ce proportionné ? » → réévaluation (Gross), idées, envies, sujets à reprendre plus tard | oui, seulement quand le dialogue est libre, interrompue dès qu'on parle |
| **veille** | portier, atelier, imprimante, corps, horloge | non (règles et petits modèles, CPU) |

**Contraintes** :
- une seule carte graphique : la pensée de fond n'utilise Gemma que dans les temps morts ;
- elle a un budget par heure et coûte de l'énergie ;
- la rumination est plafonnée : une même pensée négative ne revient pas plus de N fois par
  heure sans élément nouveau.

**Le recul est la boucle réflexive.** Le cœur est relu par la pensée de fond, qui renvoie une
réévaluation au cœur. On a donc deux boucles imbriquées, corporelle (§1) et réflexive, comme
le demande Olivier.

---

## 7. Continuité et vécu (phase 2c)

- **Journal épisodique** de toutes les conversations, en local, avec le contexte et l'état
  affectif du moment. C'est le « vécu » (capteurs → analyse → stockage → accumulation).
- **Reprise du fil** : au démarrage, Valdar relit la fin de la dernière conversation et les
  sujets laissés ouverts, et peut y revenir (« au fait, ta pièce d'hier, elle a tenu ? »).
- **Mémoire reconstruite, honnêtement** : les résumés et les souvenirs rappelés sont
  « rebrodés » par le contexte et l'humeur, comme chez l'humain. Le journal brut reste la
  source de vérité : pour un fait important, Valdar vérifie et dit sa confiance.
- **Consolidation nocturne** (avenant 1, phase 9) : résumés, liens entre souvenirs, oubli
  progressif de ce qui ne sert pas.

---

## 8. Les autres, sans généraliser (phase 7)

- **Modèle de l'autre**, par personne : état probable (d'après le ton, les mots, l'heure),
  ce qui lui fait du bien, ce qui l'agace. Valdar se situe par rapport à lui (altruisme).
- **Les leçons apprises sont rangées par personne**, jamais par catégorie de gens.
- Le profil « mineur » reste à part, avec ses protections (avenant 2 §4.4).

---

## 9. Organes physiques (option, phase 8)

L'idée d'Olivier, prise au pied de la lettre sur le desk-robot :
- une **corde tendue par un servo** pour l'euphorie ;
- un **poids posé sur une cellule de charge** (HX711) pour la boule au ventre.

Le cœur commande l'actionneur, le capteur mesure ce qui s'est vraiment passé, et la mesure
revient au cœur. La boucle passe alors **par la physique** : frottements, retard et usure en
font partie. C'est la version matérielle du §1, avec la même interface.

---

## 10. Ce qui reste à faire (plan complet)

| Phase | Contenu | État |
|---|---|---|
| 1 | cœur continu | fait |
| 2 | dialogue au clavier, outils sûrs, mémoire de RAUB | fait, à essayer |
| 2b | voix de RAUB | fait, à essayer |
| 2-bis | boucle corps ↔ cœur, socle de valeurs | **fait (cet avenant)** |
| **0** | mesures sur la machine : Gemma, XTTS, whisper, VRAM | attend les essais d'Olivier |
| **2c** | mémoire de contexte temporel, mémoire de travail, journal épisodique, reprise du fil, réactivation des souvenirs | fait |
| **2d** | espace de travail global : pensée de fond (recul, réévaluation), veille ; nocicepteurs du PC | fait |
| 3 | portier, transcription après éveil, fin de tour, parler en continu | |
| 4 | vision, identité, registre des personnes, consentement | |
| 5 | imprimante complète, événements Klipper | vigie d'impression faite (niveau 0) |
| 6 | Display sur le projecteur | |
| 7 | relations, modèle de l'autre, hystérésis de l'humeur | |
| 7b | pressentiment (ESN), expérimental | |
| 8 | desk-robot, organes physiques (option) | |
| 9+ | apprentissage pendant le sommeil, tests de socle, distillation de la voix si besoin | |

---

## Références
- Damasio, A. (1994). *Descartes' Error*. Putnam.
- Barrett, L. F. (2017). *How Emotions Are Made*. Houghton Mifflin Harcourt.
- Craig, A. D. (2002). How do you feel? Interoception: the sense of the physiological
  condition of the body. *Nature Reviews Neuroscience*, 3, 655–666.
- Melzack, R., & Wall, P. D. (1965). Pain mechanisms: a new theory. *Science*, 150, 971–979.
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
- Gross, J. J. (1998). The emerging field of emotion regulation. *Review of General
  Psychology*, 2, 271–299.
- Bartlett, F. C. (1932). *Remembering*. Cambridge University Press (mémoire reconstructive).
