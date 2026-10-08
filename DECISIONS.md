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
- **VRAM saturée par RAUB** (whisper large-v3 + Ollama + vision : 11,4 / 12 Go). Valdar et RAUB ne
  peuvent pas utiliser le GPU en même temps : à arbitrer avant toute mesure.
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
  fenêtre Display), après arbitrage RAUB / Valdar sur la VRAM.
- Phase 2 : API FastAPI + WebSocket (flux d'affect), dashboard, Display minimal.

---

## 2026-10-07 (soir) — Valdar remplace RAUB : avenant n°2

Décision d'Olivier : Valdar **remplace** RAUB (et PrinterAgent) et reprend toutes leurs capacités,
en mieux. Détails : `docs/avenant-2-heritage-raub.md` (prévaut sur la v2).

- RAUB, AURA et Ollama arrêtés par Olivier ; plus aucun d'eux n'utilise la carte graphique
  (vérifié dans le Gestionnaire des tâches, colonne GPU).
- Inventaire fait : RAUB (agent à 38 outils, affect par tour avec marqueurs par personne,
  voix whisper/XTTS, identité voix + visage, panneau AURA, Klipper), PrinterAgent (vigie Obico,
  garde-fous G-code, ESP32), desk-robot (corps ESP32-S3 avec protocole documenté).
- Corrections retenues : identité par vrais modèles (SFace, empreinte vocale neuronale), un
  inconnu n'est jamais autorisé, pas de transcription de la pièce sur disque (VAD + mot d'éveil),
  plus de lecture des cookies du navigateur, API locale uniquement, chaîne vocale 100 % locale.
- **Backend LLM : Ollama d'abord** (déjà installé avec `gemma4:12b`), derrière `LLMBackend`, pour
  retrouver vite le niveau de RAUB ; llama-server quand viendront les adaptateurs d'affect.
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
- **Outils repris de RAUB** : heure, info système, mémoire, stock, checklists, rappels (analyse
  des moments en français corrigée : « dans 2h30 », « un quart d'heure », « demain à 8h15 »),
  pinouts, imprimante (état, fichiers ; **pause et arrêt d'urgence ouverts à tous**). Nouveau :
  `mon_etat_interieur` (introspection ancrée).
- **Mémoire** : faits SQLite (source, personne, confiance, archivage au lieu de suppression),
  recherche par mots-clés tolérante aux accents et élisions, complétée par les souvenirs les plus
  importants.
- **Import RAUB** (`valdar import-raub`) : souvenirs, stock, rappels à venir, checklist, pinouts,
  marqueurs d'affect (copiés pour la phase 7). Lecture seule côté RAUB (vérifié par empreinte),
  une seule fois (`data/imports.json`).
- Tests : 103 verts (Python 3.11 et 3.13), dont Ollama et Moonraker simulés.
- Correctif : après un long rattrapage, le cœur rattrape aussi les secondes passées à
  rattraper (son horloge restait en retard de la durée du calcul).

### Reste à vérifier sur la machine
- Gemma 4 12B dans Ollama : appel d'outils réel, latence, VRAM (avec la phase 0).


---

## 2026-10-07 (nuit) — Avenant n°3 : dynamique temporelle, portier, voix de RAUB

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

### Phase 2b — Voix de RAUB : faite, à essayer (`tools\valdar_voix.bat`)
- Décision d'Olivier : **garder la voix construite pour RAUB**. Elle revient dans le budget
  de la carte graphique (≈ 2 Go) ; whisper passe sur CPU par défaut. La phase 0 tranche.
- Relevé : XTTS v2 local cloné depuis `xtts_ref.wav` (fait à partir de trois extraits
  ElevenLabs), puis chaîne « megatron ». Le réglage « pitch −2 » de RAUB est en réalité un
  passe-bas à 6 kHz : gardé tel quel, c'est le timbre.
- Chaîne d'effets portée à l'identique (test contre le code de RAUB). Latents de la voix
  calculés une fois au chargement. Parole phrase par phrase. Pas de voix de secours
  différente : sans XTTS, Valdar répond par écrit.
- Import : étape « voix » de `valdar import-raub`, liste fermée de fichiers (jamais
  `cal_*.wav`, jamais de clé), lecture seule côté RAUB.
- Environnement : mêmes versions que RAUB (torch 2.6.0+cu126, coqui-tts 0.27.5,
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

Mesures en lecture seule, carte libre (RAUB arrêté), Ollama lancé à la main (`ollama serve`).

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
- **À retenir** : il reste ≈ 1,5 Go de VRAM, pas de place pour RAUB (whisper + vision) en même
  temps. Le chargement à froid (~1 min) justifie `keep_alive` long et un préchargement au
  démarrage. llama-server + LoRA d'affect pas encore mesurés.
