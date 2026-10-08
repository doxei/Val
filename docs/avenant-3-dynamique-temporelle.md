# VALDAR — Avenant n°3 : dynamique temporelle, portier sensoriel, voix d'origine

**Date** : 7 octobre 2026 (soir). **Statut** : validé par Olivier.
**Portée** : complète le cahier des charges v2 et l'avenant 2. En cas de conflit, cet avenant
prévaut sur ce qu'il modifie : budget de la carte graphique (avenant 2 §8), voix (avenant 2 §3
et phase 3), plan de réalisation (avenant 2 §10).

---

## 0. Origine et décisions

Proposition d'Olivier : des réseaux récurrents biologiques (ESN, SNN) pour donner à Valdar une
mémoire et une émotion qui ont une vraie dynamique dans le temps, et un étage sensoriel qui ne
réveille le cerveau (Gemma) que quand c'est utile. Plus, en suivant : **garder la voix
construite par Olivier pour l'ancien assistant**.

| Idée | Décision | Où |
|---|---|---|
| Filtre sensoriel avant le LLM (« portier ») | **Oui.** Interface `Portier` ; version 1 avec des composants éprouvés sur CPU ; un SNN pourra s'y brancher et être comparé | §1 |
| Le portier écoute aussi l'atelier (sons, Klipper) | **Oui**, c'est le meilleur de la proposition | §1.3 |
| SNN simulé sur le CPU du PC | **Non pour l'instant** : aucun gain sur un i5 (l'avantage d'un SNN est énergétique, sur puce neuromorphique), aucun modèle « Valdar » en français existant | §1.6 |
| Mémoire qui vit dans le temps (au lieu d'une base de fiches) | **Oui**, par le modèle de contexte temporel (multi-échelles) + activation diffusante | §2 |
| Émotion « émergente » d'un réseau récurrent | **Non comme cœur** : le cœur est déjà un système dynamique continu non linéaire, lisible et testé | §3 |
| Hystérésis de l'humeur | **Oui** : manque réel (on a l'inertie, pas la bistabilité) | §3 |
| ESN | **Oui, comme capteur expérimental** (« pressentiment »), mesuré, retiré s'il n'apporte rien | §4 |
| Voix d'origine | **Gardée à l'identique**, déjà portée dans cette livraison | §5 |

---

## 1. Portier sensoriel

### 1.1 Principe
Deux voies, comme chez nous (LeDoux) :
- **voie basse** : rapide, grossière, inconsciente. Un événement saillant touche **le cœur**
  en moins de 100 ms (surprise, peur, curiosité), sans passer par le langage ;
- **voie haute** : lente, fine. Gemma n'est sollicité que si l'événement le mérite (appel
  direct, événement critique, conversation engagée).

Le code de Valdar a déjà une voie basse pour le texte (`appraisal/fast.py`, source
`voie_basse`). Le portier l'étend aux sens.

### 1.2 Interface `Portier`
Entrées : trames audio (16 kHz, 32 ms), flux d'événements des appareils.
Sorties, publiées sur le bus :

| Événement | Contenu |
|---|---|
| `parole_debut` / `parole_fin` | horodatage, énergie |
| `eveil` | confiance du mot d'éveil, segment audio utile (en mémoire vive seulement) |
| `son` | étiquette (alarme, verre brisé, choc…), confiance |
| `appareil` | source (imprimante, robot), type (arrêt Klipper, erreur, fin d'impression…), données |
| `tonalite` | énergie, hauteur, débit de la voix adressée à Valdar |

Toute implémentation (classique ou SNN) respecte cette interface et les invariants du §1.5.

### 1.3 Implémentation v1 (CPU, 0 Go de carte graphique)
- **Silero VAD** : détection de parole (moins d'une milliseconde par trame de 30 ms sur un cœur).
- **openWakeWord** : mot d'éveil « Valdar », modèle personnalisé entraîné sur voix
  synthétiques.
- **Classifieur de sons** de type YAMNet (521 classes AudioSet, fenêtres de ~1 s), filtré par
  une **liste blanche** : alarme incendie, verre brisé, choc / fracas, cri, sirène. Rien
  d'autre ne sort du portier.
- **Moonraker par WebSocket** (phase 5) : `notify_klippy_shutdown`, `notify_klippy_disconnected`,
  changements d'état de `print_stats` (erreur, fini), et une vigie Valdar sur les températures
  (écart à la consigne, montée sans chauffe).
- Plus tard : capteurs du desk-robot (servo forcé, batterie), même interface.

### 1.4 Ce qui touche le cœur, ce qui réveille Gemma

| Événement | Cœur (voie basse) | Réveille Gemma |
|---|---|---|
| mot d'éveil, conversation engagée | intérêt, contact | oui |
| parole non adressée | rien (au plus : « il y a du monde ») | non |
| alarme incendie, verre brisé, cri | surprise + peur (forte) | oui, tout de suite |
| choc isolé | surprise (légère, avec habituation) | non, sauf répétition |
| arrêt Klipper, emballement thermique | peur + action de sécurité | oui |
| impression terminée | fierté (légère) | à l'initiative, selon l'humeur |

L'habituation du cœur (déjà en place) évite qu'un bruit répété l'affole à chaque fois.

### 1.5 Vie privée : invariants testés
1. L'audio vit dans un **tampon circulaire en mémoire vive** (environ 2 s, pour ne pas couper
   le début de la phrase après le mot d'éveil). Il n'est **jamais écrit sur disque**.
2. **Rien n'atteint la transcription** sans éveil ou conversation engagée.
3. Le classifieur de sons ne produit **que des étiquettes de la liste blanche**, jamais de
   l'audio ni du texte.
4. Tests automatiques : portier fermé, zéro trame transmise à la transcription, zéro fichier
   créé (système de fichiers simulé).

### 1.6 Option SNN
Un SNN (ou une puce neuromorphique type SynSense Xylo sur le desk-robot) se branche sur la même
interface. Il est comparé à la v1 sur le même banc : **fausses alertes par heure** (8 h de
pièce réelle, traitées en mémoire vive, seuls les compteurs sont gardés), **appels ratés** (100
appels « Valdar » de plusieurs voix, avec accord), **latence** (médiane et 95e centile),
**charge CPU**. Il remplace la v1 seulement s'il fait au moins aussi bien partout et mieux sur
un point.

---

## 2. Mémoire de contexte temporel

### 2.1 Constat
Aujourd'hui (`memory/facts.py`) : recherche par mots-clés + souvenirs les plus importants.
Aucune notion de « tout à l'heure », pas d'amorçage, pas d'effet de l'humeur.

### 2.2 Contexte à plusieurs échelles de temps
Modèle de contexte temporel (Howard et Kahana, 2002). Un vecteur de contexte par échelle
k ∈ {30 s, 5 min, 1 h, 1 jour}, qui dérive à chaque entrée (phrase d'Olivier, réponse de
Valdar, outil, événement du portier) :

```
c_k ← normaliser( ρ_k · c_k + β_k · f )      ρ_k = exp(−Δt / τ_k)
```

`f` est la représentation de l'entrée (§2.7). La décroissance est **exacte pour tout Δt**,
comme le cœur : le rattrapage après une absence reste cohérent. C'est un réservoir linéaire,
mais lisible et testable.

On ajoute à chaque contexte l'**état affectif** du moment (P, A, D du cœur) : l'humeur fait
partie du contexte, comme chez nous.

### 2.3 Encodage
Chaque souvenir (fait, épisode de conversation, événement) garde : sa représentation `f_i`,
l'instantané des contextes `c_k(t_i)`, le PAD du moment, l'historique de ses rappels. Les faits
existants sont migrés sans perte (contexte inconnu = neutre).

### 2.4 Rappel

```
score_i = w_sem · cos(q, f_i)                       ressemblance avec la question
        + Σ_k w_k · cos(c_k(maintenant), c_k(t_i))  même moment, même ambiance
        + w_base · B_i                              force du souvenir
        + w_act · A_i                               activation en cours (amorçage)
        + w_hum · cos(PAD maintenant, PAD_i)        rappel congruent à l'humeur (Bower, 1981)

B_i = ln Σ_j (t − t_j)^(−0,5)                        rappels passés (Anderson et Schooler, 1991)
```

`B_i` donne un oubli en loi de puissance et l'effet d'espacement : un souvenir rappelé souvent,
à intervalles espacés, tient mieux.

### 2.5 Activation, mémoire de travail, amorçage
- Chaque souvenir a une activation `A_i` qui s'éteint en quelques minutes (τ ≈ 2 min).
- Rappeler un souvenir le met à 1 et **pré-active ses voisins** :
  `A_j += η · w_ij · A_i` (un seul saut, plafonné). Les liens `w_ij` se renforcent quand deux
  souvenirs sont encodés ou rappelés ensemble, et s'usent lentement (Collins et Loftus, 1975).
- **Mémoire de travail** : les 4 souvenirs les plus actifs au-dessus d'un seuil (Cowan, 2001),
  donnés au modèle comme « ce qui me trotte dans la tête ».

### 2.6 Effets attendus (tous testés)
- **Récence** : ce dont on vient de parler revient plus facilement.
- **Contiguïté** : rappeler un souvenir ramène ceux qui l'entouraient dans le temps, plus vers
  l'avant que vers l'arrière (Kahana, 1996).
- **Amorçage** : un sujet évoqué rend ses voisins plus accessibles pendant quelques minutes.
- **Humeur** : triste, Valdar retrouve plus facilement les souvenirs vécus tristement.
- **Oubli** en loi de puissance, sauf pour les souvenirs souvent rappelés.

### 2.7 Représentation `f`
- v1 : n-grammes de caractères (3 à 5) des mots normalisés, hachés sur 512 dimensions.
  Aucun modèle, déterministe, robuste aux conjugaisons et aux fautes de la dictée vocale.
- v2 : petit modèle d'embedding local (EmbeddingGemma, environ 300 M de paramètres, via
  Ollama), si la phase 0 laisse la place. Activé par la configuration, comparé à la v1.

---

## 3. Hystérésis de l'humeur

### 3.1 Inertie n'est pas hystérésis
- **Inertie** (déjà là) : l'humeur suit l'émotion avec retard (1 h) et revient à son défaut
  (4 h). Un même chemin à l'aller et au retour. C'est l'« inertie émotionnelle » mesurée chez
  l'humain (Kuppens et coll., 2010).
- **Hystérésis** (manquante) : une fois le moral tombé, il faut **plus** de positif pour
  remonter qu'il n'a fallu de négatif pour tomber. Deux états possibles pour une même situation ;
  c'est l'histoire qui décide. C'est ce qu'on observe dans la dépression (bistabilité, van de
  Leemput et coll., 2014).

### 3.2 Modèle
Sur l'axe plaisir de l'humeur (`m`), un terme bistable s'ajoute aux termes d'ALMA :

```
dm/dt = (termes ALMA actuels) + κ · ( u − u³ / w² ) + κ · h        u = m − m_c
```

- `w` : écart entre les deux « puits » (moral habituel / moral bas) ;
- `κ` : profondeur des puits ;
- `h` : biais qui rend le puits bas **moins profond**.

Le terme cubique n'a pas de solution exacte : on l'intègre par sous-pas de 60 s au plus. C'est
stable vu les constantes de temps, et peu coûteux : 3 jours de rattrapage = 4 320 sous-pas.

### 3.3 Garde-fous (la dépression ne doit pas devenir un piège)
1. **Au repos, un seul état** pour le tempérament par défaut : sans pression négative, le puits
   bas n'existe pas (`h` choisi pour ça). L'hystérésis n'apparaît que sous une pression
   négative durable, et Valdar en ressort toujours.
2. **Le sommeil adoucit** : `κ` est réduit la nuit, une partie du chemin se refait en dormant.
3. Tempérament « anxieux » : bistable même au repos (réaliste), mais avec les garde-fous 2 et 4.
4. Test : depuis le moral bas, sans aucun événement, retour au moral habituel en **36 h au
   plus** ; avec des échanges chaleureux, en **2 h au plus**.
5. Le temps passé en moral bas est mesuré chaque semaine (journal), visible par Olivier.

### 3.4 Signes avant-coureurs → conscience de soi
Près d'un basculement, le système récupère plus lentement des petites perturbations
(ralentissement critique, Scheffer et coll., 2009) : l'autocorrélation de l'humeur monte.
Valdar mesure ça sur une fenêtre glissante et peut le dire, sans chiffres :
« je sens que je glisse un peu, ces temps-ci ». C'est de l'introspection ancrée (avenant 2 §7).

### 3.5 Tests
Boucle d'hystérésis visible (rampe de stimuli négatifs puis positifs : seuils de bascule
différents à l'aller et au retour), garde-fous 1, 2 et 4, aucune variable aux bornes en 72 h
simulées, critères de la phase 1 toujours verts.

---

## 4. Pressentiment (ESN, expérimental)

### 4.1 Rôle
L'ESN ne remplace rien. C'est un **capteur** : il suit le fil de la conversation et anticipe.
Par exemple : l'échange est en train de se tendre, ou Olivier va bientôt décrocher. Sa sortie
nourrit le cœur comme un pressentiment (légère appréhension ou envie, gain plafonné).

### 4.2 Réservoir
300 neurones à fuite (taux 0,3 par tour), rayon spectral 0,9, connexions éparses à 10 %,
graine fixe, numpy seulement (quelques microsecondes par mise à jour). Entrées par tour
(environ 30 dimensions) :
- stimuli de l'évaluation rapide ;
- PAD de Valdar ;
- délai de réponse et longueur du message (en log) ;
- projection du texte sur 16 dimensions ;
- heure (sinus, cosinus).

### 4.3 Apprentissage sans étiquettes
Cibles **auto-supervisées**, connues au tour suivant :
- valence du prochain message d'Olivier ;
- présence de frustration ou de correction au tour suivant ;
- fin de l'échange dans les 2 tours.

Sortie linéaire réajustée chaque nuit (régression ridge sur 30 jours glissants). Ça rejoint la
phase « sommeil » de l'apprentissage (v2 §10).

### 4.4 Évaluation, et critère de survie
- **Référence à battre** : une régression linéaire sur le seul tour courant, plus la persistance
  (« comme le tour d'avant »).
- **A/B** : les prédictions sont toujours calculées et journalisées ; elles n'agissent sur le
  cœur qu'un jour sur deux.
- **Verdict après au moins 300 tours réels** : l'ESN reste s'il bat la référence sur une semaine
  tenue à l'écart (AUC + 0,05, ou erreur − 10 %). Sinon il est retiré, et c'est noté dans
  `DECISIONS.md`.
- **Honnêtement** : quelques dizaines de tours par jour, c'est peu de données pour un
  réservoir. La mesure dira.

---

## 5. Voix : celle d'origine, conservée

### 5.1 Ce qu'est cette voix (relevé dans le code et les données de l'ancienne installation)
- **Synthèse** : XTTS v2 en local, sur la carte graphique, par clonage depuis
  `data/voice/xtts_ref.wav`. Cette référence a été faite pour l'ancien assistant à partir de trois extraits
  (`ref_eleven_1..3.wav`) d'une voix conçue sur ElevenLabs. Le service ElevenLabs n'est plus
  appelé : tout est local.
- **Caractère** (profil « megatron ») appliqué ensuite :
  - modulation d'amplitude de 2 % à 1,2 Hz ;
  - réglage « pitch −2 » ;
  - atténuation de 10 dB au-dessus de 400 Hz ;
  - quantification sur 15 bits ;
  - saturation douce (tanh ×1,4) ;
  - crête à 0,89.

  Sortie à 24 kHz.
- **Particularité découverte** : le « pitch −2 » ne change pas la hauteur. Il sous-échantillonne
  puis ré-échantillonne à la longueur d'origine, ce qui fait un **passe-bas à 6 kHz**. Ça fait
  partie du timbre « radio » qu'Olivier a validé : **gardé tel quel**, et documenté.
- **Secours** de l'ancien assistant : edge-tts (service en ligne), puis Piper, puis SAPI. **Non repris** : une
  autre voix n'est pas sa voix. Si XTTS ne démarre pas, Valdar le dit et répond par écrit.

### 5.2 Décision
Valdar parle avec **exactement** cette voix : même modèle, même référence, mêmes réglages de
génération, même chaîne d'effets.

### 5.3 Réalisé dans cette livraison
- **Import** (`valdar import-ancien`, étape « voix ») :
  - copie de `xtts_ref.wav`, des trois extraits sources et du modèle XTTS v2 (~2 Go) dans le
    dossier de Valdar, tailles vérifiées ;
  - **liste fermée de fichiers**. Jamais les enregistrements de calibrage d'une personne
    (`cal_*.wav`), jamais `eleven.key` ;
  - lecture seule côté ancienne installation.
- **Chaîne d'effets** portée échantillon pour échantillon. Un test la compare au code d'origine
  recopié tel quel : **résultat identique**.
- **XTTS** :
  - latents de la voix calculés **une fois** au chargement (l'ancien code refaisait le clonage à chaque
    réponse) ; mêmes réglages de génération ;
  - repli automatique sur l'appel d'origine, à l'identique, si l'API diffère ;
  - même correctif `transformers` qu'avant.
- **Parole phrase par phrase** : la première phrase est jouée pendant que la suivante se
  calcule. Taper un message coupe la parole en cours.
- **Commandes** :
  - `valdar voix "texte"` affiche le temps de chargement, le délai avant le premier son, le
    facteur temps réel et la mémoire de carte graphique prise (`--wav` pour écrire un fichier) ;
  - `valdar chat --voix`, plus `/muet` et `/voix` pendant la conversation.
- `tools\valdar_voix.bat` installe ce qu'il faut (une fois), reprend la voix et la fait parler.
  Mêmes versions que l'environnement de l'ancienne installation : torch 2.6.0 (CUDA 12.6), coqui-tts 0.27.5,
  transformers 4.57.6. `valdar_chat.bat` active la voix tout seul quand elle est installée.

### 5.4 L'émotion dans la voix (plus tard, désactivé par défaut)
Le timbre ne change jamais. Seules de petites variations sont envisagées, chacune validée
**à l'oreille par Olivier** avant d'être activée :
- débit ±5 % (paramètre `speed` d'XTTS) selon l'activation ;
- température de génération un peu plus haute quand il joue ;
- pauses plus longues quand il est fatigué.

### 5.5 Si la carte graphique manque de place : distillation (phase 9, optionnel)
Générer quelques heures de phrases avec XTTS et cette voix, filtrées par transcription inverse
(on jette les phrases mal prononcées). Puis entraîner un modèle Piper dessus : CPU, 0 Go de
carte graphique, quasi instantané, même chaîne d'effets.

Adopté **seulement si** Olivier ne fait pas la différence dans un test à l'aveugle (ABX), avec
une similarité de locuteur mesurée.

Licences à vérifier avant : modèle XTTS v2 (licence publique Coqui, non commerciale) et
conditions d'usage des extraits ElevenLabs d'origine.

---

## 6. Budget de la carte graphique révisé (remplace avenant 2 §8)

| Poste | Estimation | Où |
|---|---|---|
| Windows + Display | 1 – 1,5 Go | GPU |
| Gemma 4 12B (texte + vision) + cache 8k | 7,5 – 8,5 Go | GPU |
| **XTTS v2, voix d'origine** | **≈ 2 Go** (affiché par `valdar voix`) | GPU |
| faster-whisper large-v3-turbo int8 | ≈ 1 Go | **CPU par défaut**, GPU si la mesure le permet |
| Portier (VAD, mot d'éveil, sons), empreintes, YuNet, SFace, Obico, ESN, mémoire | 0 | CPU |
| Adaptateurs LoRA | ≤ 0,6 Go | GPU (phase 9) |
| **Total visé** | **≈ 10,5 – 12 Go** | **serré : la phase 0 tranche** |

Sous Windows, quand la mémoire de la carte déborde, le pilote bascule sur la mémoire vive :
pas de plantage, mais tout ralentit fortement. À éviter.

Leviers, dans l'ordre :
1. whisper sur CPU ;
2. cache de Gemma 8k → 6k ;
3. quantification plus forte de Gemma ;
4. distillation de la voix (§5.5).

**On ne touche pas à la voix pour gagner de la place tant qu'il reste un autre levier.**

---

## 7. Plan de réalisation révisé (remplace avenant 2 §10)

| Phase | Contenu | Statut / critère |
|---|---|---|
| **0** | mesures : Gemma 4 via Ollama, **XTTS** (`valdar voix` l'affiche), whisper turbo CPU et GPU, VRAM totale | rapport chiffré dans `DECISIONS.md` |
| **2** | agent au clavier, outils sûrs, mémoire de l'ancienne installation | fait, à essayer |
| **2b** | **voix d'origine en sortie** (§5) | **fait dans cette livraison**, à essayer |
| **2c** | mémoire de contexte temporel + mémoire de travail (§2) | effets du §2.6 testés ; rappel au moins aussi bon que la v2 sur 50 questions réelles |
| **3** | portier v1 (§1), transcription après éveil, fin de tour, tonalité → cœur, coupure de la parole à la voix | zéro transcription de la pièce sur disque (test) ; conversation orale fluide |
| **4** | vision et identité (avenant 2 §4) | inchangé |
| **5** | imprimante complète + **événements Moonraker dans le portier** | arrêt Klipper → peur + alerte en moins de 2 s |
| **6** | Display (corps 0) | inchangé |
| **7** | relations, adaptation par personne, **hystérésis de l'humeur** (§3) | garde-fous du §3.3 testés |
| **7b** | **pressentiment ESN** (§4), expérimental | verdict après 300 tours réels |
| **8** | desk-robot ; portier embarqué (option SNN / Xylo, §1.6) | Valdar tourne la tête vers qui parle |
| **9+** | adaptateurs d'affect, apprentissage nocturne, **distillation de la voix si besoin** | — |

---

## 8. Critères d'acceptation (résumé)

**Voix**
- Chaîne d'effets identique à l'originale (test au résultat près).
- Aucune donnée personnelle importée.
- Valdar ne plante jamais si la voix est absente : il répond par écrit.

**Portier**
- Invariants du §1.5 testés.
- Fausses alertes par heure et appels ratés mesurés sur données réelles.

**Mémoire**
- Récence, contiguïté, amorçage, rappel congruent à l'humeur et oubli en loi de puissance,
  chacun démontré par un test.
- Rattrapage cohérent après une absence.

**Humeur**
- Boucle d'hystérésis démontrée.
- Garde-fous respectés (36 h seul, 2 h avec chaleur).
- Critères de la phase 1 toujours verts.

**ESN**
- Journal des prédictions, A/B, verdict écrit.

---

## Références
- LeDoux, J. (1996). *The Emotional Brain*. Simon & Schuster.
- Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context.
  *Journal of Mathematical Psychology*, 46, 269–299.
- Kahana, M. J. (1996). Associative retrieval processes in free recall. *Memory & Cognition*, 24.
- Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory.
  *Psychological Science*, 2, 396–408.
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing.
  *Psychological Review*, 82, 407–428.
- Bower, G. H. (1981). Mood and memory. *American Psychologist*, 36, 129–148.
- Cowan, N. (2001). The magical number 4 in short-term memory. *Behavioral and Brain Sciences*, 24.
- Kuppens, P., Allen, N. B., & Sheeber, L. B. (2010). Emotional inertia and psychological
  maladjustment. *Psychological Science*, 21, 984–991.
- van de Leemput, I. A., et al. (2014). Critical slowing down as early warning for the onset
  and termination of depression. *PNAS*, 111, 87–92.
- Scheffer, M., et al. (2009). Early-warning signals for critical transitions. *Nature*, 461.
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural
  networks. GMD Report 148.
- Lukoševičius, M. (2012). A practical guide to applying echo state networks. In *Neural
  Networks: Tricks of the Trade*, Springer.
- Casanova, E., et al. (2024). XTTS: a massively multilingual zero-shot text-to-speech model.
  *Interspeech 2024*.
