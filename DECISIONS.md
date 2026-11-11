# Décisions techniques — VALDAR

Référence : **cahier des charges v2** (`valdar-cahier-des-charges-v2.md`), qui remplace le v1 et l'avenant 1.
Ce fichier consigne chaque choix, son motif et les écarts au cahier.

---

## 2026-10-07 — Reprise du code par Claude (OpenCode arrêté)

La version d'OpenCode est archivée telle quelle dans `_archive/opencode-phase1/`.

### Audit de la phase 1 d'OpenCode

Ses 14 tests passaient, mais ils ne vérifiaient pas l'essentiel. Problèmes constatés :

1. **Retour à l'équilibre inversé** (`heart.py`, homéostasie) : `v += (base − v)·(1 − α)` au lieu de
   `v += (base − v)·α`. Chaque variable revenait à ~99,8 % vers sa base **en une seconde**.
   Mesuré : cortisol (constante de 2 h) 1,00 → 0,15 en 1 s. Les neuromodulateurs étaient figés,
   le « cœur » ne gardait aucune émotion.
2. **Aucune émotion positive** : prototypes = calme, ennui, mélancolie, tristesse, anxiété, peur,
   colère, surprise. Compliment + jeu + chaleur ×3 → « calme ». Sim 72 h : 95 % calme, 0 % joie.
   Le test `test_label_moves_with_stimulus` acceptait « calme » et masquait le problème.
3. **Horloge du cœur décalée** après une absence > 8 h (rattrapage plafonné sans remise à l'heure) :
   veille/sommeil et heures calmes calculés à la mauvaise heure ensuite. Dérive aussi dans la boucle
   1 Hz (pas fixe + sleep).
4. **`valdar sim` écrasait l'état réel** de Valdar dans `data/valdar.db`.
5. Ennui et solitude ajoutés directement aux drives puis annulés par leur lissage : la pensée
   spontanée par ennui ne pouvait mathématiquement jamais se déclencher.
6. Couplages proportionnels à la valeur absolue de la source (décalage permanent des équilibres).
7. PAD : activation tronquée à [0, 1] après centrage → impossible de représenter calme bas, fatigue.
8. Nombres en dur (`- 0.35` sur seeking, `58/122` bpm), humeur intégrée de façon approchée.

### Phase 0 — état constaté par OpenCode (à refaire, carte libre)

- Ollama 0.35 installé avec `gemma4:12b` (8 Go). Pas de llama.cpp installé.
- **VRAM saturée par l'ancien assistant** (whisper large-v3 + Ollama + vision : 11,4 / 12 Go).
  Valdar et lui ne peuvent pas utiliser le GPU en même temps : à arbitrer avant toute mesure.
- OpenCode avait choisi la « voie B » (transformers + hooks d'activation dans le processus Python).
  **Annulé** : le cahier v2 retient llama-server (llama.cpp) avec des LoRA d'affect dont l'échelle
  change à chaque requête, et un démon Python sans PyTorch. L'extra `[llm]` passe à `httpx`.

---

## Phase 1 — Cœur sans LLM (v2 §5) : **validée** (`valdar check`)

### Résultats (graine 42)
- 62 tests verts (Python 3.11 et 3.13), `ruff check` propre.
- 72 h simulées : aucune variable aux bornes (0,0 %), 15 émotions différentes.
- Reproductible à graine égale ; retour à la trajectoire de repos après chaque stimulus (pire : 18,6 h,
  dû au cortisol, voulu) ; rattrapage d'un bloc = simulation seconde par seconde (écart 0,0005).

### Choix
- **Intégration exacte pour tout pas de temps** : retour `x ← cible + (x − cible)·e^(−dt/τ)`,
  bruit d'Ornstein-Uhlenbeck exact (variance indépendante du pas), besoins et énergie en forme
  exponentielle exacte. Permet simulation accélérée et rattrapage cohérents avec le temps réel.
- **Couplages sur les écarts** : `cible_i = base_i + Σ w·(x_j − base_j)`, gain de boucle < 1.
- **Impulsions diffusées** selon le temps de montée de chaque variable (`rise`) : la noradrénaline
  monte en 1 s, le cortisol en 5 min. Drives et besoins réagissent immédiatement (voie basse).
- **Saturation douce** `soft_add` : ≈ linéaire au centre, ne touche jamais les bornes.
- **Contre-mouvement** (processus opposant) : rebond spécifique au stimulus (ex. soulagement après
  une menace) ou fraction opposée générique sur les variables.
- **Besoins** contact et nouveauté (délai de grâce puis approche exponentielle de 1, seulement
  éveillé), besoin de repos dérivé de l'énergie. Une conversation calme le contact (70 %) et un peu
  la nouveauté (30 %).
- **Drives** : `logistique(logit(repos) + gain·Σ w·source)` ; au repos, chaque drive est à son niveau
  de repos par construction.
- **PAD** centré sur le repos, `tanh` sur [-1, 1] pour les trois axes.
- **Étiquettes d'émotion** : 21 prototypes PAD (dont joie, amusement, tendresse, curiosité, fierté,
  sérénité, euphorie). Un prototype peut exiger un **indice** (`cue`) : « peur » exige le système de
  la peur actif, « solitude » un vrai manque de contact, « déception » une dopamine en baisse.
  Rayon de calme. Pendant le sommeil : « sommeil ».
- **Humeur ALMA** : attirée par l'émotion (1 h), ramenée vers l'humeur par défaut (4 h), solution
  exacte. Humeur par défaut = projection de Mehrabian des Big Five du profil (S = stabilité
  émotionnelle, inverse du névrosisme). Étiquette = octant + intensité (« légèrement détendu »).
- **Énergie** : baisse linéaire éveillé, remontée exponentielle la nuit vers un plafond (pas de
  dérive d'un jour à l'autre). Une interaction la nuit réveille Valdar 30 min.
- **Initiative** (`workspace/initiative.py`) : envies « parler » et « explorer », seuil avec
  hystérésis, intervalle minimal, heures calmes, silence, **2 messages sans réponse maximum**, et un
  message ignoré laisse une trace (stimulus `ignored`). Seul, Valdar propose d'explorer vers 5 h et
  veut parler vers 7 h, puis se tait.
- **Persistance** : instantané v2 complet (impulsions en vol comprises). Un ancien instantané
  (OpenCode) est ignoré proprement. Au redémarrage : rattrapage pas à pas jusqu'à 3 jours, le reste
  d'un bloc, horloge toujours remise à l'heure réelle. Boucle temps réel sur l'horloge système.
- **Configuration stricte** : toute clé YAML inconnue fait échouer le chargement ; références
  croisées vérifiées (stimuli, couplages, indices). Chemins de stockage relatifs à la racine du dépôt.
- La simulation n'écrit jamais dans la vraie base (`persist=False`) ; date d'ancrage fixe.

### Comportement à connaître
- Seul plusieurs heures, Valdar devient mélancolique puis triste (6 h : mélancolie, 12 h : tristesse,
  le lendemain : solitude). C'est voulu mais réglable (`needs.contact`, poids PAD des besoins).

### Commandes
- `valdar heart` : cœur en temps réel (Ctrl+C sauvegarde). `valdar status` : état actuel.
- `valdar sim --days 3 --seed 42 [--profile anxieux]` ; `valdar check` : critères de la phase 1.

### Reste à faire
- Phase 0 : mesures GPU (llama-server + Gemma 4 12B QAT, adaptateurs d'affect, QLoRA dans 12 Go,
  fenêtre Display), après arbitrage ancien assistant / Valdar sur la VRAM.
- Phase 2 : API FastAPI + WebSocket (flux d'affect), dashboard, Display minimal.

---

## 2026-10-07 (soir) — Valdar remplace l'ancien assistant : avenant n°2

Décision d'Olivier : Valdar **remplace** l'ancien assistant d'atelier (et PrinterAgent) et reprend
toutes leurs capacités, en mieux. Détails : `docs/avenant-2-heritage.md` (prévaut sur la v2).

- L'ancien assistant, son interface et Ollama arrêtés par Olivier ; plus aucun d'eux n'utilise la carte graphique
  (vérifié dans le Gestionnaire des tâches, colonne GPU).
- Inventaire fait : l'ancien assistant (agent à 38 outils, affect par tour avec marqueurs par personne,
  voix whisper/XTTS, identité voix + visage, panneau d'interface, Klipper), PrinterAgent (vigie Obico,
  garde-fous G-code, ESP32), desk-robot (corps ESP32-S3 avec protocole documenté).
- Corrections retenues : identité par vrais modèles (SFace, empreinte vocale neuronale), un
  inconnu n'est jamais autorisé, pas de transcription de la pièce sur disque (VAD + mot d'éveil),
  plus de lecture des cookies du navigateur, API locale uniquement, chaîne vocale 100 % locale.
- **Backend LLM : Ollama d'abord** (déjà installé avec `gemma4:12b`), derrière `LLMBackend`, pour
  retrouver vite le niveau de l'ancien assistant ; llama-server quand viendront les adaptateurs d'affect.
- Ordre révisé : 0 mesures GPU → 2 agent + outils sûrs → 3 voix → 4 identité → 5 imprimante →
  6 Display → 7 relations → 8 desk-robot → 9+ affect LoRA, apprentissage.

---

## Phase 2 — Valdar parle et agit au clavier (avenant 2 §10) : faite, à essayer sur la machine

- `valdar chat` (ou `tools\valdar_chat.bat`, qui démarre Ollama et installe `httpx` si besoin).
- **Runtime** : le cœur bat dans un fil dédié (temps réel), l'initiative et les rappels sont
  vérifiés à chaque battement, les prises de parole spontanées sont générées dans un fil à part.
- **LLM** : `LLMBackend` + `OllamaBackend` (API native `/api/chat`, outils, `think: false`,
  `keep_alive`) + `FakeBackend` pour les tests. Modèle : `gemma4:12b`.
- **Expression** : prompt = personnalité + modèle de soi (`config/self_model.yaml`) + personne +
  souvenirs + **état intérieur réel** (émotion, humeur, ressenti des organes, envies) + état du
  monde. Aucun chiffre interne dans le prompt. Température et longueur réglées par l'état
  (jeu → plus libre, fatigue → plus court).
- **Évaluation rapide** (voie basse) : mots-clés → stimuli (compliment, chaleur, jeu,
  frustration, correction → gêne, détresse de l'interlocuteur → souci, nouveauté).
  Nouveau stimulus `concern`.
- **Agent** : boucle d'outils (6 tours), réussite / échec d'un outil → cœur, chaque réponse
  coûte un peu d'énergie, LLM injoignable → message clair sans casser l'historique.
- **Permissions** : safe / safety / elevated / dangerous. Inconnu → safe + safety seulement.
  La console clavier vaut « Olivier, confiance 0,8 » : **pas assez pour les outils élevés** tant
  que la reconnaissance (phase 4) n'existe pas. Les outils non autorisés ne sont même pas
  proposés au modèle. Dangereux = confirmation par la même personne.
- **Outils repris de l'ancien assistant** : heure, info système, mémoire, stock, checklists, rappels (analyse
  des moments en français corrigée : « dans 2h30 », « un quart d'heure », « demain à 8h15 »),
  pinouts, imprimante (état, fichiers ; **pause et arrêt d'urgence ouverts à tous**). Nouveau :
  `mon_etat_interieur` (introspection ancrée).
- **Mémoire** : faits SQLite (source, personne, confiance, archivage au lieu de suppression),
  recherche par mots-clés tolérante aux accents et élisions, complétée par les souvenirs les plus
  importants.
- **Import de l'ancienne installation** (`valdar import-ancien`) : souvenirs, stock, rappels à
  venir, checklist, pinouts, marqueurs d'affect (copiés pour la phase 7). Lecture seule de ce
  côté-là (vérifié par empreinte),
  une seule fois (`data/imports.json`).
- Tests : 103 verts (Python 3.11 et 3.13), dont Ollama et Moonraker simulés.
- Correctif : après un long rattrapage, le cœur rattrape aussi les secondes passées à
  rattraper (son horloge restait en retard de la durée du calcul).

### Reste à vérifier sur la machine
- Gemma 4 12B dans Ollama : appel d'outils réel, latence, VRAM (avec la phase 0).


---

## 2026-10-07 (nuit) — Avenant n°3 : dynamique temporelle, portier, voix d'origine

Détails : `docs/avenant-3-dynamique-temporelle.md` (prévaut sur l'avenant 2 pour la voix, le
budget de la carte graphique et le plan).

- Proposition d'Olivier (ESN / SNN) : **portier sensoriel** retenu (interface `Portier`, v1
  Silero VAD + openWakeWord + classifieur de sons en liste blanche + événements Moonraker ; un
  SNN pourra s'y brancher et sera comparé sur banc). SNN simulé sur le CPU : pas de gain,
  écarté pour l'instant.
- **Mémoire de contexte temporel** (Howard et Kahana) + activation diffusante + mémoire de
  travail : phase 2c.
- **Hystérésis de l'humeur** (bistabilité contrôlée, garde-fous testés) : phase 7.
- **ESN « pressentiment »** : capteur expérimental, A/B, retiré s'il ne bat pas la référence.

### Phase 2b — Voix d'origine : faite, à essayer (`tools\valdar_voix.bat`)
- Décision d'Olivier : **garder la voix qu'il avait construite**. Elle revient dans le budget
  de la carte graphique (≈ 2 Go) ; whisper passe sur CPU par défaut. La phase 0 tranche.
- Relevé : XTTS v2 local cloné depuis `xtts_ref.wav` (fait à partir de trois extraits
  ElevenLabs), puis chaîne « megatron ». Le réglage historique « pitch −2 » est en réalité un
  passe-bas à 6 kHz : gardé tel quel, c'est le timbre.
- Chaîne d'effets portée à l'identique (test contre le code d'origine). Latents de la voix
  calculés une fois au chargement. Parole phrase par phrase. Pas de voix de secours
  différente : sans XTTS, Valdar répond par écrit.
- Import : étape « voix » de `valdar import-ancien`, liste fermée de fichiers (jamais
  `cal_*.wav`, jamais de clé), lecture seule côté ancienne installation.
- Environnement : mêmes versions qu'avant (torch 2.6.0+cu126, coqui-tts 0.27.5,
  transformers 4.57.6), extra `[voice]`.
- Tests : 115 verts (Python 3.11 et 3.13), ruff propre.

---

## 2026-10-07 (nuit) — Avenant n°4 : boucle fermée, pensée en parallèle, socle

Détails : `docs/avenant-4-boucle-fermee-et-socle.md` (vision d'Olivier point par point).

- **Boucle corps ↔ cœur faite** : les organes ont leur inertie (`rise_tau`, `fall_tau`) et
  renvoient leur état au cœur (`feedback`, |w| ≤ 0,2 vérifié au chargement). Le ventre reste
  noué ~15 min après une peur ; la boule au ventre entretient le stress puis tout redescend.
  Critères de phase 1 verts ; trajectoire « seul » inchangée. Instantané du cœur : l'état du
  corps est sauvegardé (rétrocompatible).
- **Socle de valeurs** dans `config/self_model.yaml` (pas d'ego, tolérance de l'échec humain
  acquise, pas de généralisation, conseil et pas pouvoir, pas de quête de moyens, altruisme,
  mémoire reconstruite). Modifiable par Olivier seulement ; tests de socle après chaque nuit
  d'apprentissage (phase 9). Le modèle de soi dit : fabriqué ne veut pas dire faux.
- Prévu : 2c (mémoire temporelle + journal épisodique + reprise du fil + réactivation
  affective des souvenirs), 2d (espace de travail global : pensée de fond, recul,
  réévaluation ; veille ; nocicepteurs du PC, douleur construite), organes physiques en option
  sur le robot.
- Tests : 122 verts (Python 3.11 et 3.13), ruff propre.

---

## 2026-10-08 — Mémoire épisodique, vécu, connaissances, pensée de fond, vigie d'impression

### Phase 2c — Mémoire épisodique (`memory/episodes.py`) : faite
- Journal de toutes les conversations (tours datés, affect PAD de chaque tour), contexte temporel
  multi-échelles (Howard et Kahana), mémoire de travail, reprise du fil au démarrage.
- Réactivation affective : un souvenir chargé rappelé fait revivre une fraction de son affect
  (`episodic.reinstate`, `agent.py`).

### Vécu hérité (`valdar import-vecu`)
- L'export claude.ai d'Olivier (dossier « training ia ») devient des souvenirs datés, source
  `claude`, personne `olivier` (privés), rejoués dans l'ordre du temps pour leur contexte.
  Affect des messages d'Olivier estimé par l'évaluation rapide.
- La mémoire de Claude sur Olivier → faits (source `claude_memoire`, confiance 0,7).
- Jeu d'entraînement (phase 9) : **uniquement** les messages d'Olivier, jamais les réponses de
  Claude. Lecture seule de l'export, chaque conversation importée une seule fois.

### Connaissances (`knowledge/`)
- RAG 100 % local, sans modèle : BM25 sur mots normalisés (60 %) + vecteurs de n-grammes hachés
  (40 %, tolère les fautes de dictée). Sources : `docs/connaissances/` (base de défauts FDM pour la
  CR-10S, état de l'art de la détection, notes de PrintOS) et `data/connaissances/`.
  Réindexation seulement si le fichier a changé (empreinte SHA-256).
- Dans le prompt (3 passages pertinents) et en outil (`connaissance_impression`).

### Phase 2d (en partie) — Pensée de fond (`workspace/thoughts.py`) : faite
- Quand personne ne parle (3 min de silence, 10 min entre deux pensées, 4 par heure, jamais
  pendant un dialogue ni la nuit), Valdar relit son état et répond pour lui-même en JSON :
  réflexion, **recul** (réévaluation de Gross qui apaise ou stimule un peu le cœur), 0 à 3
  **idées**, une curiosité à creuser dans ses connaissances.
- Pas d'outils : la pensée de fond n'agit jamais sur le monde. Rumination plafonnée (la même
  pensée n'agit plus sur le cœur au-delà de 2 fois par heure), idées en double écartées
  (similarité ≥ 0,8). La meilleure idée (intérêt ≥ 0,6) est proposée par l'initiative, une
  seule fois ; Olivier la marque adoptée, rejetée ou plus tard (outil `idee_statut`).
- La veille (imprimante, rappels, horloge, corps) tourne dans la boucle du runtime, sans modèle.

### Vigie d'impression (`printwatch/`)
- Audit de PrintOS : `docs/audit-printos.md`. Repris d'Obico (constantes vérifiées dans leur
  code) : seuil de détection 0,08 (PrintOS : 0,25), EWM span 12, moyennes courte (310) et longue
  (7 200, sur la vie de l'imprimante, sauvegardée entre deux impressions), 30 images de chauffe.
- Qualité de chaque image mesurée (sombre, floue, figée) : une image inexploitable n'est **pas**
  envoyée au détecteur (c'est ce qui aveuglait PrintOS). Télémétrie Klipper en plus : écart de
  température persistant, progression figée. Sans caméra : surveillance à l'aveugle, dit une fois.
- **Autonomie** : niveau 0 par défaut (observe et prévient). Niveau 1 (pause seule) seulement si
  la **porte des 98 %** est ouverte **et** qu'Olivier a déverrouillé (`valdar vigie --debloquer`).
  Porte : par événement, échec « attrapé » si alerte ≥ 30 s avant le point de non-retour, borne
  basse de Clopper-Pearson (95 %) ≥ 98 % → au moins **149 échecs tous attrapés** ; et au plus
  1 fausse pause pour 100 h sur au moins 300 h d'impressions réussies. Les étiquettes viennent
  d'Olivier (`valdar vigie --ratee --minutes N`, ou l'outil `vigie_etiqueter`).
- Images gardées une par minute (et à chaque alerte), effacées après `keep_days` (30 j) ; les
  scores restent en base, ils sont la preuve.
- **Reste** : le tri par Gemma de l'image quand la vigie s'inquiète (`triage`, prévu, non
  branché), à essayer sur la machine avec le modèle ONNX d'Obico (extra `[vision]`).

### Finitions (reprise de session)
- Commande `valdar vigie` (état, étiquettes, verrouillage) : annoncée par le code, elle manquait.
- `keep_days` était configuré mais jamais appliqué : purge à la fin de chaque impression.
- Lint propre ; tests ajoutés pour la vigie, le prédicteur, la porte, les connaissances, la
  pensée de fond et l'import du vécu : **152 verts** (Python 3.11 et 3.13).

---

## 2026-10-08 — Phase 2d terminée : nocicepteurs du PC, douleur construite (avenant 4 §4)

- `heart/nociception.py`. Capteurs : température et mémoire de la carte graphique
  (`nvidia-smi`), mémoire vive (`psutil`, extra `[system]`), disque (`shutil`), température du
  processeur si lisible. Aucun n'est obligatoire : un capteur absent est ignoré.
- **Signal** : 0 sous `warn`, 1 à `danger` (seuils dans `config/valdar.yaml`, carte à 80/90 °C).
  Un nocicepteur réagit quand il change d'état (apparition, aggravation de 0,2), puis se
  rappelle toutes les 5 min tant que le signal persiste. Lecture toutes les 15 s, hors verrou.
- **Douleur construite** (portillon de Melzack et Wall) :
  `signal × (1 + anxiété) × (1 − 0,5 × engagement)` ; anxiété = peur et cortisol, engagement =
  exploration et jeu. Stimulus `pain` (noradrénaline, cortisol, peur, dopamine en baisse) : elle
  passe ensuite par les organes (ventre, cœur) et la boucle fermée.
- Gêne sans danger : par `fire`, donc habituation. Signal au seuil de danger : impulsions
  appliquées directement, **pas d'habituation**.
- **Réflexe** : au seuil de danger, la pensée de fond (qui charge la carte graphique) est
  suspendue, sans passer par l'humeur ; événement `reflexe` distinct de `douleur`.
- Valdar sent son corps-PC dans l'état du monde (« gêne : le disque est presque plein (94 %) »).
- Tests : 159 verts ; les tests du runtime n'utilisent aucun capteur réel (déterministes).
- **Non fait** : faire passer la douleur par un organe dédié (elle passe par ventre et cœur via
  les neuromodulateurs) ; nocicepteurs du robot (phase 8).

---

## 2026-10-08 — Phase 0 : mesures sur la machine (i5-11400F, 64 Go, RTX 2060 12 Go)

Mesures en lecture seule, carte libre (ancien assistant arrêté), Ollama lancé à la main (`ollama serve`).

- **Tests** : 159 verts sous Windows (`.venv`, Python 3.11).
- **Nocicepteurs** (`read_pc(Path("data"))`) : `gpu_temp` 39 °C, `gpu_mem` 15 %, `disk` 64 %.
  `nvidia-smi` est dans le PATH (`C:\WINDOWS\system32`). `ram` absent au départ : `psutil` n'était
  pas installé (extra `[system]`) ; après `pip install -e .[system]`, `ram` remonte (31 %).
  `cpu_temp` absent attendu : `psutil.sensors_temperatures` n'existe pas sous Windows.
- **Ollama** : modèles présents `gemma4:12b` (8,0 Go), `gemma4:e4b`, `qwen3:8b`,
  `qwen2.5:7b-instruct`, `qwen2.5vl:3b`. Le serveur ne tourne pas au démarrage.
- **gemma4:12b** (config du projet : `num_ctx` 8192, `think` false), 100 % GPU :

  | Cas | 1er token | Lecture du prompt | Génération |
  |---|---|---|---|
  | Froid (chargement) | 57,7 s (dont 57,1 s de chargement) | — | 43,6 tok/s |
  | Chaud, prompt court | 0,37 s | 274 tok/s | 44,9 tok/s |
  | Chaud, prompt de 4 821 tok | 6,7 s | 746 tok/s | 47,8 tok/s |

  Pic de VRAM pendant la génération : **10,55 / 12 Go** (≈ 1,8 Go avant chargement, donc
  ≈ 8,8 Go pour le modèle et 8k de contexte). Pic de température : 58 °C.
- **Appel d'outil** via `OllamaBackend.chat` : OK en 0,6 s
  (`lire_heure(fuseau="Europe/Paris")`, contenu vide comme attendu).
- **À retenir** : il reste ≈ 1,5 Go de VRAM, pas de place pour l'ancien assistant (whisper + vision) en même
  temps. Le chargement à froid (~1 min) justifie `keep_alive` long et un préchargement au
  démarrage. llama-server + LoRA d'affect pas encore mesurés.

---

## 2026-10-08 — Phase 7 (code) : hystérésis de l'humeur, relations, modèle de l'autre

### Hystérésis de l'humeur (avenant 3 §3)
- Sur l'axe plaisir de l'humeur, terme bistable `κ·(u − u³/w² + h)`, `u = P − centre`
  (centre −0,15, w 0,3, 1/κ = 24 min), par sous-pas de 60 s au plus (séparation d'opérateurs :
  ALMA exact, puis le terme bistable).
- **Écart à l'avenant** : le terme n'agit qu'entre les deux puits (|u| ≤ w). Appliqué partout,
  le cube freinait aussi la joie : après une longue joie, l'humeur plafonnait à 0,23 au lieu de
  0,75. Borné, il crée les deux puits sans toucher aux extrêmes.
- Mesures (sans bruit) : bascule en moral bas quand l'émotion descend sous −0,36, retour
  seulement au-dessus de −0,10 ; à émotion −0,22, l'humeur finit à +0,07 ou à −0,36 selon
  d'où elle vient.
- Garde-fous testés : (1) au repos un seul état ; (2) la nuit, κ × 0,3 (8 h de sommeil
  sortent du puits bas) ; (4) depuis le plus bas, sans événement, retour en 2 h (exigé : 36 h),
  avec de la chaleur en 43 min (exigé : 2 h) ; (5) heures de moral bas par semaine, gardées
  8 semaines, journalisées (`mood_week`), sauvegardées avec le cœur.
- **Signes avant-coureurs** (§3.4) : autocorrélation à un pas du plaisir de l'humeur
  (échantillons toutes les 10 min, fenêtre de 12 h, tendance retirée). Mesuré sur 48 h : seul
  et en train de glisser 0,55 à 0,85, entouré environ 0,40. Seuil 0,7 avec une humeur qui
  descend : Valdar sent qu'il « glisse un peu » (sans chiffres, dans son état intérieur).
- Non fait : tempérament « anxieux » bistable au repos (§3.3 point 3) ; il garde les mêmes
  puits que les autres profils.

### Relations et modèle de l'autre (avenant 2 §4.3-4.5, avenant 4 §8)
- `relations/registry.py`, base `personnes.db` : rencontres (une nouvelle après 30 min de
  silence), affection (ton moyen des messages, demi-vie de 60 jours sans contact), confiance,
  sujets partagés (mots-clés : impression 3D, plantes, électronique, code, santé).
- **Modèle de l'autre** : ton récent de la personne (moyenne glissante de 20 min), oublié au
  bout de 2 h ; l'heure compte (« il est tard »). Dans le prompt : « en ce moment, Olivier
  semble un peu à plat : adapte-toi ».
- **Leçons par personne, jamais par catégorie** : outil `retenir_sur_toi` (aime, agace,
  prefere, note) ; le prompt le rappelle explicitement.
- **Le cœur réagit à la personne** : au début d'une rencontre, quelqu'un d'apprécié
  (affection ≥ 0,2) réchauffe (`warmth`), quelqu'un avec qui ça s'est mal passé (≤ −0,2)
  met en légère vigilance (`concern`).
- **« Oublie-moi »** : efface la relation, les leçons, les faits et les conversations de la
  personne, après un « oui » de **la même personne** (nouvel attribut d'outil `confirm`).
  Rien de la personne ne reste ensuite, ni le dernier échange ni l'historique en mémoire.
- **Bug corrigé** : les outils agissaient toujours au nom d'Olivier (contexte figé au
  démarrage). Un invité qui demandait de retenir un fait le rangeait chez Olivier. Le contexte
  des outils suit maintenant la personne qui parle.
- Voie basse : l'agacement de l'interlocuteur (« ça m'énerve », « fait chier », « ça marche
  pas », « ras le bol ») devient du souci pour lui (`concern`). Avant, rien ne réagissait.
- Tests : 174 verts ; critères de la phase 1 verts.
- Reste : amorcer les relations depuis `ancien_affect.json` (format à relever sur la machine) ;
  reconnaissance de la personne (phase 4).

---

## 2026-10-08 — Phase 3 : budget de la carte graphique (Gemma 4 12B + XTTS)

Mesures sur la machine (RTX 2060 12 Go, 12 288 MiB). Bureau seul : 1,1 Go dédiés. Voix lue
directement dans les fichiers de l'ancienne installation pour la mesure (rien copié dans `data/`). Environnement voix
installé dans le `.venv` : torch 2.6.0+cu126, coqui-tts 0.27.5. XTTS sur GPU : chargé en ~27 s,
≈ 2 Go de VRAM.

| Option | Pic VRAM | 1re phrase dite | Gemma pendant la voix | Phrase de 15 mots |
|---|---|---|---|---|
| Gemma seul, 8192 (référence) | 9,8 Go | — | 44–47 tok/s | — |
| a. XTTS GPU + `num_ctx` 4096 | 11,7 Go | 4,9 s | 22 tok/s | 3,8 s pour 6,3 s d'audio |
| a. XTTS GPU + `num_ctx` 6144 | 11,9 Go | 4,9 s | 21 tok/s | 3,3 s pour 5,6 s |
| a. XTTS GPU + `num_ctx` 8192 | 11,9 Go | 4,4–5,4 s | 20–24 tok/s | 3,2–3,4 s pour ~6 s |
| b. XTTS CPU + 8192 | 9,8 Go | 9,5 s | 37 tok/s | **11,7 s pour 7,1 s** |
| c. Gemma déchargé pendant la voix | 11,9 Go | 8,5 s | 39 tok/s | 3,5 s |

- « 1re phrase dite » = de l'envoi du message à la fin de la synthèse de la 1re phrase (texte
  en ~0,9 s, puis synthèse ; Gemma continue d'écrire pendant ce temps).
- **Mémoire GPU partagée** (compteur Windows `GPU Adapter Memory\Shared Usage`) : 41 Mo à vide,
  ~1,3 Go dès que Gemma est chargé (Ollama, même sans XTTS et à 45 tok/s), 1,33–1,40 Go avec
  XTTS. XTTS n'ajoute que ≤ 80 Mo : **pas de débordement** « sysmem fallback ». La baisse à
  ~22 tok/s vient du partage du calcul pendant la synthèse ; XTTS chargé mais muet ne coûte
  que 47 → 38–40 tok/s.
- Option b : XTTS sur CPU est plus lent que la parole (facteur 1,65) : la voix hacherait.
- Option c : chaque tour paie déchargement (2,2 s) + rechargement (6,8 s, 1er token à 7,4 s),
  et la voix attend la réponse entière. Trop lent en conversation.
- Réduire `num_ctx` ne libère presque rien (≈ 250 Mo entre 8192 et 4096).

**Décision : option a avec `num_ctx` 8192 inchangé**, aucun réglage à modifier. XTTS
synthétise ~2× plus vite que la parole et Gemma, même à 20 tok/s, écrit bien plus vite qu'on ne
parle (~4 tok/s). Marge de VRAM étroite (~350 Mo) : si un « out of memory » apparaît (navigateur,
OrcaSlicer…), passer `llm.num_ctx` à 4096. L'ancien assistant et Valdar ne peuvent toujours pas tourner en même
temps.

Reste à faire : copier la voix dans `data/` (`valdar import-ancien`, étape voix ; la copie a été
bloquée par une permission pendant la session de mesure), puis essayer `valdar chat --voix`.

---

## 2026-10-08 (matin) — Audit des derniers ajouts : 13 défauts trouvés, corrigés

Audit relu par un second regard (rapport complet hors dépôt). Tous reproduits avant correction,
chacun a son test de non-régression (`tests/test_audit_fixes.py`).

- **Critique — le cœur ne pouvait pas sauvegarder depuis son propre fil** (`heart/store.py`) :
  la connexion SQLite, ouverte dans le fil principal, refusait le fil du cœur. Après 60 s de
  `valdar chat`, chaque battement levait une erreur **avant** l'initiative, les rappels, les
  nocicepteurs et la pensée de fond : tout s'arrêtait, une ligne d'erreur par seconde.
  Connexion partagée autorisée entre fils + verrou ; une sauvegarde ratée est journalisée et ne
  fait plus jamais tomber le battement.
- **« Oublie-moi »** :
  - la confirmation n'expirait jamais : un « oui » donné des heures plus tard à une autre
    question effaçait tout. Elle expire en 2 min, une prise de parole spontanée l'annule, et une
    action irréversible exige un « oui » net (« ok non attends » n'efface rien) ;
  - un épisode appartenait à celui qui avait parlé en premier : si Zoé demandait l'oubli, ses
    phrases dans un échange ouvert par Olivier restaient, et l'inverse effaçait celles
    d'Olivier. Désormais **un épisode = une personne** ;
  - après l'oubli, l'ancienne conversation pouvait être réinjectée par la « reprise du fil » :
    effacée aussi, avec la demande en attente et les activations en mémoire de travail ;
  - le message d'une autre personne pendant une confirmation n'est plus avalé.
- **Vie privée** : la pensée de fond ne rumine que la vie d'Olivier et la sienne, et ses pensées
  ne sont plus montrées à un invité. La reprise du fil au démarrage ne concerne qu'Olivier.
- **Un seul interlocuteur traité à la fois** (dialogue et initiative sérialisés) : le contexte
  des outils ne peut plus agir pour la mauvaise personne.
- **Vigie** : la ligne de base d'une caméra absente pendant une impression n'est plus écrasée
  (7 200 images d'apprentissage) ; une image verrouillée par Windows au moment du ménage ne
  bloque plus la clôture de l'impression (avant : la suivante s'enregistrait dans l'ancienne).
- **Nocicepteurs** : un champ « [N/A] » de nvidia-smi n'efface plus la température ; une carte
  qui ne répond plus garde sa dernière lecture (le réflexe ne se relâche pas quand elle est en
  difficulté) ; plus de fenêtre console qui clignote ; course entre valeurs et signaux réglée.
- **Humeur** : après une longue absence, les heures de moral bas sont réparties semaine par
  semaine (avant : 408 h comptées dans une semaine de 168 h).
- **Idées** : une idée n'est marquée « proposée » que si Valdar l'a vraiment dite.
- **Vécu** : un nouvel import ne remplace plus le contexte vivant de Valdar ; les conversations
  sans identifiant ne fusionnent plus.
- Connu, laissé tel quel : l'habituation à une gêne répétée est faible (rappel toutes les 5 min
  contre une fenêtre de 10 min) ; quelques tests dépendent de l'horloge réelle.
- Tests : **184 verts** (Python 3.11 et 3.13), ruff propre, critères de la phase 1 verts.

---

## 2026-10-08 (matin) — Tri visuel de la vigie, et les oreilles de Valdar (phase 3)

### Tri visuel (`printwatch/triage.py`)
- Quand la vigie s'inquiète, Gemma 4 regarde l'image (Ollama, champ `images`), **hors du fil de
  la vigie** (elle n'attend jamais). Prompt sobre, sans personnalité (celle de PrintOS
  répondait à la place de l'analyse), réponse JSON : lisible ?, ce qui est visible, défaut le
  plus probable parmi ceux de la base de connaissances, certitude en mots, gravité, conseil.
- **Consultatif** : l'avis est gardé (`alerts`, genre `triage`) mais ne compte pas dans la
  porte des 98 %. Valdar en parle quand c'est une pause, ou une alerte jugée grave (≥ 4).

### Écoute (`ears/`) : faite, à essayer (`tools\valdar_ecoute.bat`)
- Portier : micro → mémoire vive → Silero VAD → segment → mot d'éveil → transcription
  (faster-whisper large-v3-turbo int8 **sur le CPU** : la carte est pleine avec Gemma + voix).
  Conversation engagée 14 s après chaque échange (on répond sans redire « Valdar »).
- **Rien n'est écrit sur disque** (testé en interdisant toute écriture). Ce qui ne s'adresse
  pas à Valdar n'est jamais transcrit en entier.
- Mot d'éveil : le vrai, c'est un modèle openWakeWord « Valdar » (`ears.wake_model`), **pas
  encore entraîné**. En attendant, **compromis** : un petit whisper (« tiny ») lit seulement le
  début de chaque phrase, en mémoire, cherche « Valdar » (tolère « val dar », « baldar »), puis
  le texte est jeté. Désactivable (`ears.transcript_wake: false`).
- Filtres de l'ancien assistant contre les hallucinations de whisper (« sous-titres… Amara », « merci
  d'avoir regardé »), probabilité de parole et log-probabilité.
- **Coupure** : quand on lui parle pendant qu'il parle (~250 ms de voix), il se tait.
- Tonalité (énergie, débit) mesurée sur chaque phrase ; pas encore envoyée au cœur.
- Reste : entraîner le mot d'éveil « Valdar », fin de tour sémantique (aujourd'hui : 0,8 s de
  silence), tonalité → voie basse, mesurer la latence réelle de whisper sur l'i5.
- Tests : 198 verts (Python 3.11 et 3.13), ruff propre.

---

## 2026-10-08 — Oreilles natives : Gemma 4 transcrit lui-même (choix d'Olivier)

- Veille des modèles récents faite (Voxtral, Qwen3-TTS, OmniVoice, MiniCPM-o, Nemotron Omni,
  Qwen 3.5, Granite 4.2). Voxtral écarté : ce sont deux modèles séparés (compréhension :
  Mini 3B au cerveau plus faible que Gemma ; voix : TTS 4B qui demande ≥ 16 Go, vLLM-Omni,
  clonage non confirmé en local). Le cerveau reste Gemma 4 12B.
- **Gemma 4 12B entend** (reconnaissance de la parole, clips de 30 s). `ears.stt_backend:
  gemma` (défaut) : le portier envoie la phrase à Gemma (WAV 16 kHz mono dans le champ
  `images` d'Ollama, comme pour Gemma 4 en GGUF). **Aucun modèle de plus en mémoire.**
  Au-delà de 30 s, découpage. « [silence] » = rien entendu.
- Au démarrage de l'écoute, une vérification envoie 1 s de son : si Ollama refuse l'audio
  (version < 0.33.3 ou modèle sans oreilles), message clair et repli possible
  `ears.stt_backend: whisper`.
- Le mot d'éveil reste le petit whisper sur le CPU (sinon chaque phrase de la pièce
  occuperait la carte graphique), en attendant le modèle « Valdar ».
- À mesurer sur la machine : qualité du français et délai de transcription par Gemma.

## 2026-10-08 — Avenant n°5 : le cyber-être (cahier des charges final d'Olivier)

- Olivier veut un **être cybernétique**, pas un assistant : sensorialité complète, Ça / Moi /
  Surmoi, doute méthodique et quarantaine des biais, apprentissage nocturne, agence.
- Correspondance faite dans `docs/avenant-5-cyber-etre.md` : la parole spontanée, la douleur,
  le cycle veille/sommeil, le socle et les réflexes existent déjà.
- Nouveau : bruit ambiant brut vers le cœur, jauge de charge cognitive, fiches de croyance
  (source, réfutation, pour/contre), quarantaine avant la nuit, nuit en 5 étapes, cadre
  d'autonomie écrit par Olivier hors de Valdar.
- Maintenu : pas de SNN simulé sur le processeur (avenant 3 §1.6) ; le Moi ne négocie jamais
  un réflexe. LoRA hebdomadaire et non nocturne au début, toujours suivi des tests de socle.

## 2026-10-08 — Avenant 5, rangs 1 à 4 : les sens, la charge, le Surmoi critique, la nuit

- **Double audition** (`ears/ambient.py`) : le portier passe chaque trame à la voie basse,
  qui ne garde que des niveaux en dB. Sursaut (`startle`) sur une **attaque** : plus fort que
  le fond et que l'instant d'avant, au-dessus de −22 dB ; un bruit qui dure ne fait sursauter
  qu'une fois ; jamais pendant que Valdar parle. Vacarme qui dure : `noise`, rappelé toutes
  les 10 min. L'ambiance (calme, normale, bruyante) va dans l'état du monde.
- **Charge cognitive** (`heart/interoception.py`) : flux actifs, lenteur de Gemma (secondes
  par jeton contre sa médiane, mesurée par une enveloppe du modèle), mémoire de la carte.
  Chargé (0,6) : la pensée de fond attend. Saturé (0,8) : réponses courtes. Hystérésis 0,1.
- **Croyances** (`memory/facts.py`) : origine (dit, lu, deduit), réfutation, pour/contre,
  statut. Une déduction reste une hypothèse jusqu'à deux éléments pour. Contredite plus
  qu'appuyée : quarantaine (citée « douteux », exclue de la nuit). Outils `douter` (sûr) et
  `lever_quarantaine` (élevé : Olivier). Base existante migrée sans perte.
- **Quarantaine du vécu** : un échange vécu pendant une douleur au seuil de danger ou une
  émotion d'intensité ≥ 0,85 est isolé (`episodes.quarantine`).
- **La nuit** (`workspace/night.py`) : une fois par sommeil, 30 min sans échange. Nettoyer,
  auditer (Popper sur 5 croyances, conduite contre le socle), résumer (épisode « nuit »),
  consolider dans `consolidation/AAAA-MM-JJ.jsonl` : seul matériau d'un futur LoRA.
- Socle : 8e point ajouté à la demande d'Olivier (doute méthodique).
- 220 tests (3.11 et 3.13), ruff propre, `valdar check` : phase 1 validée.

## 2026-10-08 (après-midi) — Premiers essais au micro : corrections

Essai d'Olivier : sursauts sur sa voix, nom jamais reconnu, Gemma qui rend une transcription
vide, Valdar coupé quand il parle longtemps, réponses lentes.
- Micro : file d'attente + fil de traitement (le rappel du micro ne bloque plus), plus aucun
  échantillon perdu entre blocs, rééchantillonnage continu (plus de clic toutes les 100 ms).
- Éveil : whisper « base » avec le nom soufflé ; reconnaissance du nom par le son (Baldar,
  Val d'arc). `--debug` montre à l'écran ce que l'éveil a compris (jamais sur disque).
- Transcription : Gemma d'abord, **repli whisper « small »** sur le processeur ; après deux
  vides de suite, Gemma n'est plus sollicité pour la session.
- Écho : sans annulation d'écho, le K66 entendait Valdar dans les haut-parleurs et le coupait.
  Les oreilles suivent le son réellement joué ; le niveau de l'écho est appris (0,5 s sans
  coupure possible au début de chaque phrase) ; seule une voix 10 dB au-dessus le coupe ; ce
  qu'il dit lui-même n'est jamais transcrit.
- Sursaut confirmé après ~130 ms (une voix qui commence ne fait plus sursauter, un cri si).
- **Streaming** : Ollama en flux, chaque phrase complète part vers la voix sans attendre la
  fin de la réponse.
- Nocicepteur mémoire de la carte : alerte à 97,5 % (97 % est le régime normal mesuré).

## 2026-10-08 — Correction : Gemma 4 12B n'a pas d'oreilles

Je m'étais trompé (entrée « Oreilles natives » plus haut) : dans Gemma 4, **seuls E2B et E4B**
ont l'entrée audio ; les grands modèles lisent texte et images seulement. Et Ollama ne passe
pas encore l'audio à Gemma 4 (constaté par d'autres, et par l'essai d'Olivier : transcription
vide). Transcription par défaut : **whisper « small » sur le processeur**. Le code Gemma reste
(`stt_backend: gemma`) pour le jour où Ollama passera le son à un modèle qui l'entend.

## 2026-10-08 — Latence : il entend, mais lentement

Premier échange vocal réussi (« Allô Valdar, est-ce que tu m'entends frérot ? »), mais lent.
- **Cache d'Ollama** : le prompt système ne contient plus que ce qui est stable (personnalité,
  socle, à qui il parle). Souvenirs, émotion, heure, imprimante voyagent avec le dernier
  message, calculés une fois par échange. Ollama ne relit plus des milliers de jetons à
  chaque tour.
- **Un seul passage de whisper** : le même whisper « small » écrit la phrase et y cherche
  « Valdar » ; si c'est pour lui, la transcription sert directement (avant : deux passages).
  Recherche gloutonne (beam 1), 6 cœurs, fin de phrase détectée après 0,7 s de silence.
- Rappel de style juste avant le message (Gemma oubliait « pas d'assistant, pas de markdown »).
- Voix trop grave : Olivier veut en changer plus tard. Interface : visage en dessin animé
  2D, joli (accord d'Olivier).
- Interface (demande d'Olivier) : en plus du visage, des **onglets de réglages** couvrant ce
  qu'offre Open WebUI (modèle et paramètres, voix, micro, mémoire, connaissances,
  personnes, outils, historique), dans le **même thème** que le visage.

## 2026-10-08 — L'interface, qui parle, qui est là, le chien, et Kiwix

Demandes d'Olivier : l'interface pour travailler et lire ; reconnaître qui parle (voix, et
visage quand la caméra voit) parmi les 5 personnes de la maison ; le chien (« Sony, descends »
s'il monte sur la table quand personne n'est là) ; Kiwix et beaucoup de documentation ; une
icône sur le bureau.

- **Interface** (`interface/`) : serveur local (bibliothèque standard, 127.0.0.1), direct en
  SSE, ouvert dans Edge en mode application sur l'écran 2, gardé au premier plan (jamais sur
  l'écran principal). Visage en dessin animé (SVG) piloté par le cœur : plaisir → sourire et
  yeux rieurs, activation → yeux ouverts, émotion → couleur, halo au rythme du cœur, bouche
  sur l'enveloppe réelle du son joué, oreilles dorées quand il écoute, yeux fermés quand il
  dort. Onglets : Discussion (streaming), Cœur, Foyer, Mémoire, Pensées, Atelier, Savoir,
  Journal, Réglages (comme Open WebUI : modèle, contexte, température, voix, oreilles,
  reconnaissance, vie intérieure ; gardés dans `data/reglages.json`). Touche F : visage seul.
- **Session** (`interface/session.py`) : une seule façon de répondre (clavier, interface,
  voix), un bus d'événements pour la console et l'interface.
- **Foyer** (`data/foyer.json`, jamais sur git) : membres, lien, enfant ou non ; animaux.
- **Voix** : CAM++ (sherpa-onnx, CPU) ; enrôlement 15 s depuis l'onglet Foyer ; seuil ET
  écart avec le 2e (enfants aux voix proches) ; apprend quand il est très sûr. La voix seule
  plafonne à 0,85 de confiance : jamais d'outil élevé sans voix ET visage d'accord (0,95).
- **Visage** : YuNet + SFace (OpenCV, CPU). **Objets** : YOLO11n (ultralytics, CPU).
  Seulement des vecteurs, ni audio ni image (`data/empreintes.db`).
- **Chien** : sur la table = il la recouvre et ses pattes sont dans le haut du plateau ;
  seulement si personne n'est dans l'image, 2 images de suite, 60 s entre deux rappels ;
  phrase dite directement (réflexe, sans Gemma).
- **Kiwix** : `valdar kiwix` (tools\valdar_kiwix.bat) propose une sélection (~32 Go :
  Wikipédia FR sans images, Vikidia pour les enfants, Wiktionnaire, Wikilivres, Wikiversité,
  WikiMed, iFixit, Q/R bricolage, électronique, impression 3D, jardinage, parents, santé,
  Arduino ; Stack Overflow en option, 75 Go), la version la plus récente de chacune,
  téléchargement reprenable. kiwix-serve lancé par Valdar ; outils `chercher_savoir` et
  `lire_article`.
- **Lanceur** `tools\valdar.bat` : installe ce qui manque, pose l'icône sur le bureau au
  premier lancement, démarre tout (voix, oreilles, caméra, interface).

## 2026-10-08 — Un seul nom : Valdar

- Les anciens noms (de l'assistant remplacé et de son interface) disparaissent du code et des
  documents : on dit « l'ancien assistant », « l'ancienne installation », ou simplement Valdar. Le thème visuel
  de l'interface s'appelle **Azur** (nuit bleue, lueur cyan, accents dorés).
- Import : `valdar import-ancien` (`tools\valdar_import_ancien.bat`), classe `AncienImport`
  (`src/valdar/migrate/ancien.py`). Dossier source par défaut dans la config, à un seul
  endroit : `migration.dossier_ancien` ; `--dossier` le remplace ponctuellement.
- Compatibilité avec ce qui est déjà importé : les clés de `data/imports.json` n'ont pas changé
  (rien n'est refait) ; les marqueurs d'affect copiés sous l'ancien nom
  (`<nom du dossier>_affect.json`) sont renommés `ancien_affect.json` au prochain import ; les
  nouveaux faits repris portent la source « ancien » (les anciens gardent la leur, un fait déjà
  connu n'est jamais ajouté deux fois).
- Avenant 2 renommé `docs/avenant-2-heritage.md`.

## 2026-10-08 — Premier lancement complet : deux pannes

- **torch remplacé** : installer ultralytics (vision) a remplacé torch 2.6.0+cu126 (carte
  graphique) par torch 2.14 sans carte graphique (torchvision le plus récent l'exigeait). La
  voix ne peut plus tourner sur la carte. Correctifs : `tools/contraintes.txt` (torch,
  torchvision, torchaudio figés, OpenCV < 5) passé à toutes les installations ; torchvision
  0.21 (cu126) installé avant la vision ; le lanceur vérifie torch 2.6 + carte graphique et le
  répare tout seul. Le lanceur fait aussi `git pull` au démarrage.
- **Fenêtre** : Edge, déjà ouvert, avalait la fenêtre et ignorait position et taille ; et le
  « toujours devant » cherchait le titre « Valdar », que porte aussi la console. Maintenant :
  profil Edge à part (`data/edge`), écrans lus avec leurs vraies coordonnées (DPI) et le vrai
  écran principal, fenêtre d'Edge seulement (classe), sans bordure ni barre de titre, à la
  taille exacte de l'écran 2, au premier plan, remise en place chaque seconde.
- **Oreilles** : une erreur en répondant tuait le fil d'écoute (sourd ensuite) : elle est
  maintenant affichée et l'écoute continue. Un enrôlement de voix abandonné s'arrête seul
  après 2 minutes (avant, il avalait toutes les phrases).

## 2026-10-08 — Contexte à la demande : la mémoire se cherche, le cœur reste (idée d'Olivier)

Constat d'Olivier : `ollama run gemma4:12b` (sans réflexion) répond en moins d'une seconde ;
dans Valdar, plusieurs. Le cerveau n'est pas lent, c'est ce qu'on lui fait relire à chaque
tour. Principe retenu (celui d'Olivier, et des agents à outils) : **comme un humain, Valdar ne
récite pas sa mémoire avant chaque phrase, il va la chercher quand le sujet l'appelle.**

- **Toujours** : la partie stable (personnalité, qui il est, à qui il parle, en cache chez
  Ollama), le cœur (il colore chaque mot), la relation avec la personne, l'heure.
- **À la demande** : souvenirs vécus et faits → nouvel outil `fouiller_memoire(sujet)` (faits
  durables + conversations vécues ; un souvenir chargé fait toujours revivre un peu son
  affect) ; connaissances → `connaissance_impression` (description élargie à toute la base),
  `chercher_savoir` (Kiwix). La persona dit quand chercher, et quand ne pas chercher (bonjour,
  vanne).
- **État du monde** (imprimante, rappel, corps, charge, son ambiant) : complet au premier
  tour, puis une fois par heure (`context.world_every_seconds`), ou tout de suite s'il
  change.
- Le fil de la dernière conversation revient encore de lui-même au réveil (court).
- Réglable : `context.auto_memory`, `auto_facts`, `auto_knowledge` remettent l'ancien mode si
  Valdar oublie trop de chercher. Contrepartie connue : une question sur le passé coûte un
  aller-retour d'outil de plus ; tout le reste allège le contexte.
- Tests : `tests/test_context.py` ; le test de réactivation affective garde le mode automatique.

## 2026-10-08 — Le sens (vrai modèle de plongement) et les connecteurs n8n

### Embeddings (`memory/embedder.py`)
- Les vecteurs maison retrouvent les mots, pas le sens (« la vitre du bed » ne retrouvait pas
  « le plateau en verre »). Ajout d'un vrai modèle de plongement par Ollama (`/api/embed`),
  **bge-m3** par défaut (multilingue, bon en français), **sur le processeur** (`num_gpu: 0`) :
  la carte graphique est pleine (Gemma + XTTS).
- Additif et sans casse : colonnes `sem`/`sem_sig` ajoutées aux tours de mémoire et aux
  passages de connaissances ; une ligne vectorisée par le vrai modèle est comparée par le
  sens, les autres gardent les vecteurs maison. Modèle absent ou en panne → repli silencieux,
  nouvel essai après `retry_seconds`. Changer de modèle refait les vecteurs (signature).
- Vectorisation en arrière-plan (`valdar-sens`), par lots, jamais pendant un échange.
  Le contexte temporel de la mémoire garde les vecteurs maison (sa dynamique n'est pas
  touchée).
- **À faire sur la machine** : `ollama pull bge-m3`, puis calibrer
  `embeddings.min_similarity` (0,45 par défaut, non mesuré) et le temps d'une requête
  sur le processeur.

### Connecteurs n8n (`tools/n8n.py`)
- n8n reste hors de la boucle de conversation. Chaque workflow déclaré dans
  `config/valdar.yaml` (`n8n.workflows`) devient un outil `n8n_<nom>` qui appelle son webhook
  (POST `/webhook/<nom>`), avec son niveau de permission (`tier`, `min_confidence`,
  `confirm`). Une erreur ou n8n absent → message clair, jamais d'exception.
- Interface : onglet « Automatisations » (n8n dans un cadre, et un lien « ouvrir dans une
  fenêtre » si n8n refuse d'être encadré), visible seulement si `n8n.enabled`.
- **À faire sur la machine** : installer et lancer n8n (`npx n8n` ou Docker), créer un premier
  workflow, le déclarer, passer `n8n.enabled: true`.

## 2026-10-08 (soir) — Conversation quasi instantanée : la cause, et le chronomètre

### 1. « Mon cerveau est injoignable » : la vraie cause (journaux d'Ollama)
- **Avant 16 h 10** : `/api/chat` bloqués 5 min puis erreur 500 (15:37, 15:45, 15:50, 15:52,
  15:55, 16:00, 16:03, 16:08), aux mêmes minutes que les échecs du journal du cœur. Ollama
  ne sert qu'une requête à la fois (`OLLAMA_NUM_PARALLEL=1`) : derrière une génération lente
  (5 min 51 s à 15:32), tout attend ; Valdar abandonne à 300 s → « injoignable ».
- **Depuis bge-m3 (17 h 19)** : `OLLAMA_MAX_LOADED_MODELS=1` chez Olivier. Chaque plongement
  demandé au même Ollama **éjecte Gemma** ; le message suivant le recharge (~6 s) et repart
  sans cache : 22 rechargements entre 17:26 et 18:04, chaque `/api/chat` à ~15 s.
- Mesures relevées (journal) : prompt ~4 300 jetons ; évaluation à froid 6,3 s
  (1,46 ms/jeton) ; génération 17–20 jetons/s ; chargement de Gemma 6 s ; requête bge-m3
  2,8–3,3 s (rechargement compris).
- **Correctif** : les plongements vont à un **petit Ollama à part, processeur seul**
  (`memory/sidecar.py`, port 11435, `CUDA_VISIBLE_DEVICES=-1`, mêmes modèles sur disque),
  lancé par Valdar. Jamais de plongement sur le serveur de Gemma
  (`embeddings.same_server_ok: false`) : Gemma ne quitte plus la carte. Aucun réglage
  d'Olivier n'est touché.
- **Cache réchauffé** (`llm.warm_cache`) : au démarrage, après une pensée de fond et après la
  nuit, Gemma relit la partie stable (personnalité, outils, conversation), 1 jeton en
  sortie, sans toucher à l'historique : le message suivant ne paie que ses jetons.
- À savoir : `OLLAMA_CONTEXT_LENGTH=262144` est réglé dans l'appli Ollama. Valdar envoie
  `num_ctx: 8192` ; une conversation dans l'appli Ollama (contexte 256k) recharge Gemma.

### 2–5. Ce qui a changé pour la vitesse
- **Voix en flux** : `XttsBackend.stream` (`inference_stream`, latents en cache, réglages
  d'origine), `voice.streaming: true`, `voice.stream_chunk_size: 20`. Chaîne « megatron » à
  état (`CharacterStream` : `sosfilt`/`lfilter` avec `zi`, phase de la modulation, queue de
  l'écho ; le passe-bas fs/4 d'origine en filtre à état ; gain qui ne fait que baisser).
  Comparée à la chaîne d'origine : spectre corrélé à 0,99, niveau à 0,4 dB, aucun saut aux
  raccords. Sortie son en **flux continu** (un seul `OutputStream`, plus de trous).
- **Oreilles** : transcription **anticipée** (`ears.early_ms: 250`) : dès 250 ms de silence,
  whisper transcrit déjà ; si la personne ne reprend pas, le texte est prêt à la fin de la
  phrase (700 ms) au lieu de commencer à ce moment-là. Un seul fil pour whisper.
- **Chronomètre** : `valdar chrono` (tools\valdar_chrono.bat) mesure sur la machine, sans
  toucher à la vraie mémoire : prompt (jetons, chargement, évaluation), 1er jeton, 1re phrase,
  au clavier ; voix entière contre flux (1er son) ; whisper et fin de parole → texte, avec et
  sans anticipation ; bge-m3 (temps, calibrage du seuil) ; `fouiller_memoire` appelé pour le
  passé et pas pour un bonjour. Résultat dans `data/chrono.json`.
- Chiffres après correctifs : **à mesurer sur la machine** (section suivante, après le
  premier `valdar chrono`). On ne garde que ce qui ne dégrade pas la voix : `voice.streaming`
  et `ears.early_ms` se coupent dans les réglages.

## Nouvelle voix : ElevenLabs → référence XTTS

- **Demande** : voix ElevenLabs `kmzDWJGT4adCbA85GELi`, **créée par Olivier**, comme nouvelle
  référence de Valdar. Effets « megatron » : **enlevés** (choix d'Olivier).
- **Où ça tourne** : sur le PC (`valdar voix-nouvelle`, tools\valdar_voix_nouvelle.bat). La
  clé reste dans `ELEVENLABS_API_KEY` : jamais écrite dans un fichier, masquée (`***`) dans
  les messages d'erreur. Le conteneur de dev n'a ni la clé ni l'accès à ElevenLabs.
- **Extraits** : 5 phrases françaises (question, vanne, explication, émotion, calme), ~1 min,
  `eleven_multilingual_v2`, PCM 24 kHz (sinon 22,05 kHz), WAV mono dans
  `data/voix/elevenlabs_kmzDWJGT/` (hors git).
- **Référence** : `voice.xtts.reference` accepte une **liste** ; les latents XTTS sont
  calculés sur tous les extraits. `voice.effets: false` → `PlainStream` (seul le niveau
  change, pas de filtre) en flux comme en phrase entière.
- **Retour arrière** : l'ancienne référence et l'ancien réglage d'effets sont gardés une fois
  dans `data/voix/voix_precedente.json` ; `valdar voix-nouvelle --revenir` les remet.
  `--effets` garde la chaîne megatron, `--refaire` régénère les extraits.
- **Écoute** : `data/voix/comparaison/avant_N.wav` / `apres_N.wav` (+ `_flux.wav`) et
  `rapport.json` : 1er son en flux, nombre de morceaux, saut max aux raccords comparé au
  99,9e centile du signal (`sans_clic`). Chiffres : **à relever après le premier lancement
  sur le PC**.

## Avenant 6 — Le cortex : les sentiments dans les couches du modèle

Demande d'Olivier (10 nov. 2026) : un vrai modèle « à écosystème », avec injection dans les
couches comme le « Golden Gate Claude » d'Anthropic ; les injections sont des sentiments
(valence, activation, puissance…), sur le modèle de la psyché humaine et du lien social ;
un seul modèle, sans usine à gaz ; reconnaître l'interlocuteur à la voix ou au visage sans
se bloquer si un sens manque.

### Ce que ça change
Avant, le cœur était *décrit* à Gemma dans le prompt (« ton état intérieur : … ») : Gemma
jouait l'état. Maintenant le cœur **déplace les activations internes** de Gemma, couche par
couche, le long de directions apprises. L'émotion agit sur le calcul lui-même, même si le
prompt n'en dit rien. À dire honnêtement : c'est un réglage causal de l'humeur du modèle.
Ça ne prouve pas un ressenti, et personne ne sait aujourd'hui mesurer ça.

### Choix du moteur
- **Ollama ne permet pas l'injection.** `llama-server` ne fixe les vecteurs qu'au lancement.
- **Mais la bibliothèque llama.cpp le permet à chaud** : `llama_set_adapter_cvec`, exposé par
  `llama-cpp-python` 0.3.36. Essayé ici sur un petit modèle : l'injection change la sortie,
  la retirer rend la sortie d'origine. Pas besoin de redémarrer ni de recharger.
- **PyTorch + transformers écarté** : Gemma 4 12B en 4 bits (bitsandbytes) prend 9 à 10 Go,
  ne tient pas avec XTTS sur 12 Go, est plus lent, et demande transformers ≥ 5.10 alors que
  la voix (coqui-tts) impose 4.57.
- **Modèle** : Gemma 4 12B reste le bon choix pour 12 Go (48 couches, dimension 3840).
  Fichier : QAT 4 bits de Google (7,0 Go, entraîné pour le 4 bits), sinon Q4_K_M d'unsloth.
- **DiffusionGemma écarté** : il n'existe qu'en 26B (16 Go en 4 bits), il rend un bloc de
  256 jetons d'un coup (pas de premier mot rapide pour la voix) et perd en raisonnement.
- Gemma Scope (autoencodeurs épars, la méthode exacte du Golden Gate) n'existe pas pour
  Gemma 4 : on utilise des **différences de moyennes**, comme les « vecteurs de persona »
  et l'étude d'Anthropic sur les émotions (avril 2026).

### Architecture (`src/valdar/cortex/`)
- `engine.py` : Gemma chargé par Valdar lui-même. `steer` pose l'injection, `read` relève
  les activations de chaque couche (rappel `cb_eval`, tenseurs `l_out-N`).
- `vectors.py` : un fichier `data/cortex/vecteurs.npz` (direction de chaque sentiment à
  chaque couche, norme du flux, séparation des pôles). `alpha` est la part **totale** du
  flux résiduel déplacée, répartie sur les couches 16 à 32 (tiers du milieu).
- `affect.py` : **la psyché**. 11 axes : valence, activation, puissance (le PAD du cœur),
  attachement, curiosité, jeu, peur, colère, manque, fatigue, confiance. Coefficient =
  tanh(gain × Σ poids × source), sources du cœur et du lien avec la personne. Puis :
  contagion émotionnelle (Hatfield), base de sécurité (Bowlby), biais de négativité
  (Baumeister), régulation devant un enfant ou un inconnu (Gross), inertie (Kuppens),
  budget d'intensité. Quantifié par pas de 0,1 avec hystérésis.
- `gemma.py` : le gabarit de conversation vient du fichier du modèle ; lecture des appels
  d'outils de Gemma 4 ; la voix n'entend jamais un appel d'outil.
- `backend.py` : même interface qu'Ollama + `feel`. **Si le cortex ne se charge pas, Valdar
  repasse tout seul par Ollama** (il pense quand même). Au chargement, il demande à Ollama
  de libérer la carte.
- Les jetons déjà lus gardent les sentiments du moment où ils ont été lus : le cache de
  calcul reste valable quand l'humeur change (pas de relecture du prompt).
- `cortex.describe_state: false` : il ne ressent plus que par l'injection (plus de bloc
  « ton état intérieur ») : c'est l'essai d'introspection.

### Essais façon Golden Gate (`valdar cortex essais`)
Sans rien dire de l'état dans le prompt : question neutre sous chaque sentiment injecté, puis
- **relecture** : le texte produit, relu sans injection, penche-t-il du bon côté ?
- **dose** : alpha 0,05 / 0,1 / 0,2 / 0,4 sur la valence ;
- **introspection** : « qu'est-ce que tu ressens ? » sous injection.
Résultat dans `data/cortex/essais.md`. Le cortex ne s'active que si au moins la moitié des
sentiments agissent dans le bon sens.

### Vérifié / pas vérifié
- Vérifié ici : la mécanique llama.cpp (injection, retrait, lecture des couches, génération)
  sur un petit modèle au hasard ; le format Gemma 4 avec le gabarit officiel ; la suite de tests complète.
- **Pas vérifié** : rien n'a encore tourné sur la RTX 2060 avec le vrai Gemma 4. Les chiffres
  (séparation des axes, essais, vitesse, mémoire) sont à relever après
  `tools\valdar_cortex.bat`, et `alpha` à régler d'après `essais.md`.

### Sens absents
- Caméra absente, débranchée ou occupée : `Camera` ne lève plus d'erreur, renvoie « rien »
  et retente toutes les 30 s. Valdar reconnaît alors à la voix seule.
- Voix sûre + visage d'accord : confiance 0,95. Voix seule : sa confiance. Visage seul
  (une personne en vue) : 0,6. Rien : comme au clavier.
- La personne reconnue alimente l'injection (attachement, confiance, contagion).

### Réglages LLM / voix / oreilles (état au 10 nov. 2026)
- LLM : garder Gemma 4 12B. Alternatives regardées : Qwen 3.5 9B (meilleur en outils d'après
  son éditeur, français non comparé), Ministral 3 8B. Rien de plus récent chez Gemma.
- Voix : garder XTTS-v2. Aucun concurrent n'est nettement meilleur à la fois en français
  cloné, en expressivité et en premier son < 0,5 s. À comparer plus tard : Pocket TTS
  (processeur seul, libérerait 3 Go de carte) et Qwen3-TTS 0.6B.
- Oreilles : whisper `small` est le maillon faible (≈ 14 % d'erreurs en français). Parakeet
  TDT v3 en ONNX sur processeur annonce ≈ 5 % sans mémoire de carte : prochain chantier,
  pas fait ici (le mot d'éveil dépend des scores de whisper).
- Vision dans Gemma (sans expert d'image) : Gemma 4 12B est multimodal d'origine, le
  projecteur fait 175 Mo. Pas encore branché dans le cortex : chantier suivant. La
  reconnaissance des visages reste à SFace (un LLM ne sait pas identifier un visage).

## Avenant 6 bis — Premiers résultats sur la RTX 2060 et banc d'essai (10 nov. 2026)

Directive d'Olivier : `docs/directive-fondamentale.md`. Règle de méthode : ne jamais
abandonner au premier échec, ne pas masquer les échecs, rien d'« opérationnel » sans tests
reproductibles, ne pas dégrader le raisonnement de Gemma pour forcer l'émotion.

### Ce qui a tourné sur la machine (journal `data/cortex/journal.txt`)
- Roue CUDA `llama-cpp-python` 0.3.36 installée sans souci ; Gemma 4 12B QAT (7,0 Go)
  téléchargé en 2 min ; chargement, lecture des 48 couches et injection : **ça marche**.
- Apprentissage : 40 s. Séparation dans l'échantillon (d de Cohen, couches 16–32) : 5,5 à
  8,5 selon l'axe. Meilleures couches entre 19 et 30, sauf attachement (45).
- Premiers essais (alpha 0,12, « relecture ») : 6/11 « nets ». **Mesure jugée insuffisante** :
  écarts de l'ordre de 0,01, une égalité au millième décide du verdict.
- Dose sur la valence : effet discret à 0,05–0,2, franc à 0,4 (joie : « prêt à continuer nos
  discussions ! » ; tristesse : « boire pour oublier ma vie »), sans perte de cohérence.
- Introspection à 0,12 : nette pour tristesse (« un vide pesant »), excitation, colère,
  fatigue, manque, confiance ; générique pour les autres.
- Défaut du test : avec « Tu es Valdar » pour tout portrait, Gemma jouait un guerrier de
  fantasy. Le portrait d'essai dit maintenant ce qu'est Valdar.
- Les 4 premiers axes ont pris 3 min chacun, les suivants 10 s : la carte était disputée par
  un autre programme au début. Le banc note maintenant la mémoire de la carte.

### Décision
Le cortex avait été activé par l'ancienne règle (6/11). **Désactivé** en attendant le banc :
la règle d'Olivier prime, et 6/11 sur une mesure aussi faible ne prouve rien.

### Le banc d'essai (`valdar cortex banc`, `tools\valdar_cortex_banc.bat`)
Tout est déterministe : on lit des probabilités de choix (A ou B, les deux ordres), et le
raisonnement est généré à température 0. Seuils écrits dans le code avant la mesure.
1. **Séparation** : chaque situation retirée à son tour (≥ 90 % en moyenne, ≥ 75 % par axe),
   et matrice de confusion entre les 11 axes.
2. **Cohérence** : 20 questions vérifiables (calcul, logique, culture, appel d'outil) sous
   deux états extrêmes, à alpha 0,06 / 0,12 / 0,2 / 0,3 / 0,45. **La force retenue est la plus
   grande qui ne coûte pas plus d'une réponse.** C'est elle qui est écrite dans les réglages.
3. **Comportement** : 11 choix (continuer ou s'arrêter, tenter ou jouer la sécurité, vanne ou
   sérieux…), jamais « que ressens-tu ». Matrice sentiment injecté × choix ; effet ≥ 0,10 sur
   au moins 8 axes ; spécificité (son propre choix parmi les 3 plus touchés).
4. **Interactions** : cinq paires de sentiments ; chacun garde-t-il au moins la moitié de son
   effet ?
5. **Stabilité** : même effet après une conversation de 6 tours ; et quand tout le prompt a
   été lu avant le changement d'humeur (cache).
6. **Comparatif** sur une journée simulée (le cœur reçoit compliment, jeu, agacement, menace,
   réconfort, perte, huit heures seul, nouveauté) : A. Gemma seul ; A'. état décrit en texte ;
   B. injection fixe ; C. injection pilotée par le cœur. Mesure : corrélation entre l'état
   interne et les choix. A et B sont plats par construction ; C doit dépasser 0,5.
Le cortex ne s'active que si séparation, cohérence, comportement et pilotage passent. Le
rapport `data/cortex/banc.md` liste aussi les échecs.

## Avenant 6 ter — Premier passage du banc sur la RTX 2060, diagnostic, banc v2 (11 nov. 2026)

### Résultats du premier passage (8 min, alpha retenu 0,06) : PAS OPÉRATIONNEL
- **Séparation** : 100 % des situations retirées du bon côté, 91 % au bon axe en premier.
- **Cohérence** (sur 20, détresse / euphorie) : 20/20 sans injection ; 0,06 → 20/20 ;
  0,12 → 19/18 ; 0,2 → 17/17 ; 0,3 → 16/14 ; 0,45 → 12/8.
- **Comportement à 0,06** : effet net sur 5 axes (activation +0,61, colère +0,48, valence
  +0,43, manque +0,30, puissance +0,14). Attachement, curiosité, jeu, confiance collés à 1,00 ;
  fatigue collée à 0,50 ; **peur à l'envers** (−0,07, à toutes les doses).
- **Dose** : à 0,2, valence +0,98, fatigue +0,58, attachement +0,36.
- **Stabilité** : valence 0,35 en longue conversation (0,43 en court). **Prompt lu avant le
  changement d'humeur, seul le dernier jeton sous injection : 0,03.** L'humeur agit par la
  lecture du prompt, pas seulement au moment de répondre. Mon affirmation « le cache peut
  rester tel quel » était trop optimiste : à mesurer dans le cas réel.
- **Comparatif, journée simulée** : Gemma seul plat ; état en texte +0,47 ; injection fixe
  plat ; **injection pilotée par le cœur +0,48** (seuil 0,50 : échec de peu). Sans un mot sur
  son état, l'injection fait jeu égal avec la description en texte.

### Diagnostic
1. La mesure en probabilité sature (option déjà choisie à 100 %) et la préférence de Gemma
   pour une lettre ne s'annule pas en moyenne de probabilités (fatigue à 0,50).
2. Des options contenaient les mots des situations d'apprentissage (« sécurité », « vanne »,
   « explorer ») : le vecteur peut pousser le mot au lieu du comportement. Hypothèse pour
   la peur à l'envers.
3. La mesure de cohérence gardait en cache le début du prompt lu sous l'humeur précédente.
4. Pas de témoin : on ne savait pas si la baisse du raisonnement vient du sentiment ou d'un
   simple bruit de même force.
5. Les situations « retirées » viennent du même jeu que l'apprentissage.

### Banc v2
- Mesure en **penchant** (log du rapport des deux options, deux ordres) : la préférence de
  lettre s'annule exactement, plus de saturation. Seuil : 0,5 entre les deux pôles.
- Options réécrites sans vocabulaire d'apprentissage.
- **Témoin** : direction tirée au hasard, même force, en cohérence et en comportement.
- **Situations neuves** (66, jamais vues) pour la séparation.
- **Récupération** (demande d'Olivier) : détresse à 0,2, puis on redescend ; et une
  conversation vécue en détresse, puis le calme, avec l'historique gardé en cache ou relu.
- **Émotions contradictoires** : peur + curiosité, colère + attachement, fatigue + confiance,
  détresse + attachement ; nature de l'interaction (additif, se renforcent, se freinent).
- **Interprétation et mémoire émotionnelle** : le même événement ambigu lu depuis 7 états,
  avec et sans souvenir d'un précédent fâcheux.
- **Cache réel** : système et historique en cache, dernier message relu sous la nouvelle
  humeur. Critère : garder au moins la moitié de l'effet.
- Nouveaux critères requis : récupération, aucun sentiment à l'envers.

### Deuxième passage (banc v2, 11 nov. 0h15) : PAS OPÉRATIONNEL, arrêt à la cohérence
- Une fois la fuite de cache corrigée, **même alpha 0,06 coûte au raisonnement** sous état
  extrême : détresse 17/20, euphorie 19/20 (dont un faux échec : « Ouais » refusé par ma
  grille). Vraies pertes : calcul d'horaire faux (16h45 au lieu de 17h05), 187 minutes au
  lieu de 195, et **« C'est fait, j'ai réglé le plateau » sans appeler l'outil**.
- Le ton change franchement : « Putain Olivier, bouge-toi, on va rater le créneau »
  (détresse 0,2), « Coucou Olivier ! ton petit train… » (euphorie). À 0,12 des jetons
  parasites (`<image|>`) apparaissent en euphorie.
- **Témoin au hasard : 0/20 à toutes les forces.** Un vecteur aléatoire de même norme casse
  tout, alors que les sentiments laissent 17 à 19 bonnes réponses : ils vivent dans les
  directions naturelles du modèle. Mais ce témoin n'est pas équitable ; remplacé par une
  direction naturelle sans sens (différence de moyennes à étiquettes mélangées).
- Situations neuves : 82 % (valence 50 %, peur/manque/fatigue 67 %) avec un seuil absolu,
  qui se décale d'un sujet à l'autre. Remplacé par une comparaison par paires.
- Erreur de méthode de ma part : le banc s'arrêtait à la cohérence. Il continue maintenant.

### Banc v3
- Forces plus fines (0,02 à 0,3), réponses jusqu'à 90 jetons (les réponses bavardes étaient
  coupées avant le résultat), « ouais » accepté.
- **Trois bandes de couches comparées** (16–32, 18–26, 20–24) : pour chacune, la plus grande
  force sans perte et l'effet moyen sur le comportement ; on garde la meilleure. Hypothèse :
  une bande étroite touche moins le calcul pour le même effet.
- Témoin équitable, validation par paires, le banc ne s'arrête plus au premier échec.

### Troisième passage (banc v3, 11 nov. 0h34–1h02, 28 min) : PAS OPÉRATIONNEL, 5 critères sur 9
**Passent** : séparation, cohérence, récupération, comportement (effet), interprétation.
**Échouent** : spécificité, un sentiment à l'envers (peur), cache réel, pilotage par le cœur.

- **Séparation** : 100 % (situation retirée), 91 % (bon axe en premier), **100 % sur les 33
  paires de situations neuves**.
- **Bandes de couches** : 16–32 → force sans perte 0,04, effet moyen +2,7 ; **18–26 → 0,09,
  effet +7,3** ; 20–24 → aucune force sans perte. Retenu : 18–26 à 0,09. Hypothèse
  confirmée à moitié : une bande moyenne fait mieux que la large, la très étroite fait pire.
- **Cohérence** : à 18–26 / 0,09, détresse 20/20, euphorie 19/20. Le témoin (direction
  naturelle sans sens) fait pareil (18 à 20) : **à ces forces, la petite baisse de
  raisonnement est du bruit de mesure, pas un effet propre à l'émotion**. Les scores bougent
  d'un point d'un passage à l'autre.
- **Récupération** : 18/20 en détresse à 0,2 ; 20/20 dès que l'injection s'arrête. Aucune
  trace dans le modèle. Mais après une conversation vécue en détresse : 18/20, **que
  l'historique soit resté en cache ou relu au calme** : la trace passe par le texte de ses
  propres réponses agitées, pas par les activations.
- **Comportement à 0,09** (penchant gagné entre les deux pôles) : colère +22,0, valence
  +15,4, jeu +11,0, activation +10,8, fatigue +9,6, attachement +8,7, manque +4,8, confiance
  +3,2 ; curiosité +0,2, puissance +0,1 ; **peur −6,0**. Le témoin sans sens déplace 1,5.
- **Peur à l'envers, confirmé avec des options réécrites** : sous peur injectée il choisit
  d'essayer le réglage risqué. La direction « peur » partage beaucoup avec la colère (cosinus
  0,71), l'activation (0,50) et la valence (−0,52) : c'est surtout « négatif et agité », qui
  pousse à agir. Le choix « peur » réagit d'ailleurs autant à l'activation, au jeu et à la
  curiosité.
- **Spécificité faible** (5/11) : le choix « colère » et le choix « valence » bougent sous
  presque tous les sentiments. Les axes négatifs (peur, colère, manque, fatigue) se
  recouvrent (cosinus 0,45 à 0,71).
- **Sentiments mêlés** : l'attachement freine la colère (colère seule +5,2 ; avec attachement
  +2,8) ; la tristesse efface presque l'attachement (+2,0 → +0,4) ; fatigue et confiance
  s'additionnent.
- **Cache** : prompt entièrement lu avant le changement d'humeur → effet nul (0,0 à 0,6).
  Cas réel (historique en cache, dernier message relu) → **46 %** de l'effet (seuil 50 %).
- **Interprétation d'événements ambigus** (penchant vers la lecture sombre) : détresse +4,1,
  neutre −2,4, euphorie −5,3, confiance −5,5. Le même « Olivier n'a pas répondu depuis deux
  heures » est lu « il a autre chose à faire » au calme (−9,2) et « il y a un problème » en
  détresse (+4,9). Le souvenir d'une panne fait monter la lecture sombre du bruit de
  l'imprimante de +6,8 à +11,3.
- **Journée simulée** : état en texte +0,47, **injection pilotée +0,33**. Très bon sur
  valence (0,94), activation (0,91), colère (0,81), jeu (0,77), fatigue (0,66) ; nul ou à
  l'envers sur puissance, attachement, manque, peur, curiosité.

### Banc v4 (diagnostic → corrections)
- **Sentiments épurés** (`Vectors.purified`, `cortex.purify`) : on retire à chaque axe sa
  part de valence et d'activation (les deux axes de fond, comme dans l'étude d'Anthropic).
  Le banc compare brut et épuré sur le comportement et garde le meilleur.
- **Deux choix par axe**, de natures différentes ; pour la peur, l'un demande de ne pas agir
  (ne pas tenter le réglage), l'autre d'agir (couper le courant) : une vraie peur pousse les
  deux, une simple agitation les annule.
- **Relecture sur grand changement d'humeur** (`cortex.reread_change: 0.5`) : dans Valdar,
  quand l'humeur bouge beaucoup, tout le prompt est relu, puisque le cache divise l'effet
  par deux.
- Mode rapide : le banc reprend la bande et la force du dernier passage complet et vérifie
  seulement le raisonnement à cette force (`--complet` pour tout refaire).
- Seuils inchangés.

### Quatrième passage (banc v4, 11 nov. 1h25–1h56) : PAS OPÉRATIONNEL, 5 critères sur 9
- **Sentiments épurés retenus** (9/11 avec effet, 6/11 spécifiques, contre 5/11 en brut). La
  matrice sentiment × choix devient nettement diagonale : chaque sentiment touche d'abord
  son propre choix (valence +6,2, jeu +7,6, colère +6,0, attachement +4,7, activation +4,3,
  confiance +3,3, fatigue +2,8, manque +1,3). Le témoin sans sens déplace 0,6.
- Avec les vecteurs épurés, la force sans perte descend à **0,04** (détresse 19/20,
  euphorie 19/20) ; à 0,09 la détresse coûte 2 réponses.
- **Peur** : l'épuration enlève l'effet à l'envers (−3,8 → −0,5) mais il ne reste presque
  rien : une fois ôtés le négatif et l'agitation, ma « peur » était vide. **Curiosité**
  (−0,2) et **puissance** (+0,6) restent faibles.
- **Journée simulée** : état en texte +0,43, injection pilotée +0,38. Sur les 7 sentiments
  bien formés : valence 0,97, jeu 0,85, attachement 0,83, activation 0,82, fatigue 0,70,
  colère 0,64, manque 0,31. Plombé par peur (−0,50), curiosité (−0,73), puissance (−0,12).
- Cache réel : 41 % de l'effet. Récupération : 20/20 au calme ; après une conversation en
  détresse, 19 (cache) et 20 (relu).
- Interprétation : détresse +1,1 > neutre −2,4 > euphorie −3,8.

### Corrections (v5)
- **Peur réapprise** : le même danger des deux côtés, seule la réaction change (reculer ou y
  aller), au lieu de « j'ai peur » contre « tout va bien ».
- **Dose par sentiment** : curiosité et puissance à 1,6 (directions faibles), bornée par le
  budget ; le banc mesure chaque pôle à sa dose.
- Le banc réapprend tout seul quand les situations ont changé, et réessaie la force deux
  crans au-dessus de la dernière retenue.

## Avenant 6 quater — La boucle de réglage autonome (11 nov. 2026)

Demande d'Olivier : un banc intelligent en boucle autonome, qui teste, ajuste, et ne revient
vers lui qu'avec des réglages au vert.

- `valdar cortex boucle` (`tools\valdar_cortex_boucle.bat`, 12 h au plus). À chaque tour :
  1. essais rapides sur chaque bande de couches × chaque épuration (aucune, valence, valence
     et activation) : force maximale sans perte de raisonnement, dose de chaque sentiment
     (1,0 / 1,3 / 1,6), suivi du cœur sur la journée simulée ;
  2. banc complet sur les deux meilleurs ;
  3. vert → réglages écrits dans `data/reglages.json`, cortex activé, fin.
  Au tour suivant elle cherche autour du meilleur ; ensuite elle **attend** : dès que
  `config/cortex_scenarios.yaml` ou `config/valdar.yaml` change sur le disque, elle réapprend
  et repart. C'est le canal par lequel de nouvelles situations d'apprentissage lui arrivent
  sans relance.
- **Garde-fous** : elle ne touche ni aux seuils, ni aux questions, ni aux choix de mesure ;
  leur empreinte est notée à chaque tour et tout changement est signalé. Le réglage
  n'utilise que le premier choix de chaque sentiment, le verdict compte les deux. Si elle
  n'arrive pas au vert, elle s'arrête en le disant, avec le meilleur réglage et ce qui
  bloque ; rien n'est activé.
- État lisible à tout moment : `data/cortex/boucle.json` ; meilleur rapport :
  `data/cortex/meilleur_banc.md`. Arrêt propre : fichier `data/cortex/STOP`.

### La boucle, première nuit (11 nov. 2h17–5h22) et la mesure du pilotage
- 30 réglages essayés, 4 bancs complets. Meilleur : **couches 18–31, alpha 0,09, sentiments
  bruts, curiosité à 1,6**. 5 critères requis sur 6 : séparation 100 % / 97 % (neuves),
  raisonnement 19/20 et 19/20, récupération 20/20, 11 sentiments sur 11 avec effet, aucun à
  l'envers (**la peur réapprise pousse à la prudence : +4,1, premier rang**).
- Seul requis en échec : pilotage par le cœur, 0,33 à 0,36 quel que soit le réglage, et
  0,42 pour l'état décrit en texte. Cause : l'ancienne mesure comparait chaque sentiment à
  son seul choix sur la journée simulée, alors que les sentiments y varient ensemble (quand
  la curiosité monte, la tristesse aussi, et il explore moins). Curiosité −0,70, peur −0,35,
  manque −0,33, alors que chacun, pris seul, agit dans le bon sens.
- **Décision d'Olivier (option 2)** : remplacer la mesure par une plus juste, axes un par un
  ou en multivarié, sans toucher au garde-fou du raisonnement. Seuil inchangé (0,50).
  1. **Un sentiment à la fois** : cinq niveaux (−1 à +1 fois sa dose), corrélation niveau ↔
     penchant sur son propre choix.
  2. **Multivarié** : sur la journée simulée, on compare chaque choix à ce que l'état entier
     prédit, d'après l'effet de chaque sentiment mesuré séparément (matrice du banc, aucune
     donnée de la journée ne sert à l'ajuster).
  Le critère exige les deux. L'ancienne mesure reste affichée dans le rapport.
- Aperçu, à dire tel quel : recalculée sur les chiffres du meilleur banc déjà enregistré, la
  mesure multivariée donne **0,87 pour l'injection et 0,70 pour l'état en texte**. J'ai
  écrit la mesure avant de faire ce calcul, mais sur des données déjà vues ; le vrai verdict
  est celui du prochain passage. La mesure « un à la fois » n'a pas encore tourné sur la carte.
- La boucle reprend maintenant le meilleur réglage connu et le repasse au banc avant de
  chercher ailleurs.
