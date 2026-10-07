# Détection et prédiction précoce des échecs d'impression FDM — état de l'art

*Document de référence pour l'assistant local (CR-10S, Klipper/Moonraker, caméra). Rédigé en français, synthèse personnelle des sources citées. Dernière mise à jour : 2026-10-08.*

---

## 1. Ce qu'il faut retenir

- La détection d'échec « grand public » open source repose presque entièrement sur **un détecteur d'objets entraîné à reconnaître le spaghetti** (Obico), appliqué à une image toutes les 10 à 60 s et **lissé dans le temps**. Ça marche bien pour les échecs spectaculaires (décollement, spaghetti, blob), mal pour les défauts discrets (bouchon, sous-extrusion, première couche).
- Les travaux universitaires annoncent souvent de 85 à 99 % d'**exactitude**, mais presque toujours **par image ou par segment de signal**, sur des jeux de données équilibrés et sur une seule machine. Quasiment aucun ne mesure un **délai d'anticipation** (combien de temps avant le point de non-retour l'alarme arrive) ni un **rappel par événement**.
- Pour **anticiper** et pas seulement constater, il faut regarder les **précurseurs** : coins qui se soulèvent, première couche mal écrasée, claquement de l'extrudeur, défilement du filament plus lent que la commande, température qui ne suit plus la consigne, temps de couche anormal. La caméra seule détecte surtout l'échec **une fois là**.
- Un objectif comme « prédire ≥ 98 % des échecs avant qu'ils arrivent » ne peut pas se vérifier sans **au moins ~150 événements d'échec indépendants tous détectés à temps** (borne inférieure de Clopper-Pearson à 95 %), et nettement plus s'il y a quelques ratés. Voir la section 7.

---

## 2. Obico (ex-The Spaghetti Detective)

### 2.1 Fonctionnement général
- Le plugin de l'imprimante (OctoPrint ou Moonraker) envoie périodiquement une image de la webcam au serveur. Le serveur passe l'image dans un réseau de neurones convolutif qui renvoie des **boîtes de détection avec un score de confiance**. Quand le score dépasse un seuil, Obico prévient l'utilisateur et peut mettre l'impression en pause ([blog Obico](https://www.obico.io/blog/how-obico-ai-failure-detection-works/)).
- Selon le blog, l'image est capturée par défaut toutes les 30–60 s côté plugin, le modèle de 1re génération a été entraîné sur « des centaines de milliers » d'images et une 2e génération utilise un jeu plus vaste, enrichi par les images signalées par les utilisateurs. Obico **ne publie pas de taux de rappel ni de taux de fausses alarmes**. Le blog reconnaît des faiblesses sur les bouchons, la sous-extrusion, le stringing, les tout premiers stades d'échec et la première couche, et note qu'une **caméra de côté** voit mieux les décalages de couches qu'une vue de dessus.
- Sources de faux positifs citées : géométries complexes (supports, remplissage visible), éclairage mauvais ou variable, filaments transparents ou peu contrastés, caméra qui vibre, pièce sombre sur plateau sombre.

### 2.2 Ce que montre le code source (serveur auto-hébergé)
Lecture du dépôt [obico-server](https://github.com/TheSpaghettiDetective/obico-server) (licence **AGPL-3.0**) :

| Élément | Valeur dans le code | Fichier |
|---|---|---|
| Architecture | Réseau Darknet de type YOLOv2 (couche `[region]`, 5 ancres), **1 seule classe** (« échec »), entrée 416×416 | `ml_api/model/model.cfg` |
| Poids | Darknet, ONNX et RKNN (NPU RK3588), téléchargés à la construction de l'image Docker depuis les URL des fichiers `model/model-weights.*.url` (ex. `https://www.obico.io/static_pub/ml-models/model-weights-5a6b1be1fa.onnx`), avec vérification SHA-256 | `ml_api/Dockerfile`, `ml_api/model/*.url` |
| Intervalle de détection | 10 s par défaut côté serveur ; le commentaire indique que les hyperparamètres ont été **réglés pour 10 s** et déconseille de le changer | `backend/config/settings.py` (`MIN_DETECTION_INTERVAL`) |
| Score par image `p` | **somme des confiances** de toutes les boîtes détectées | `backend/lib/prediction.py` |
| Lissage | moyenne mobile exponentielle, `EWM_SPAN = 12` (α ≈ 0,154, demi-vie ≈ 4 images ≈ 40 s) | idem |
| Moyennes glissantes | courte : 310 images (≈ 50 min) ; longue : 7 200 images (≈ 20 h, sur la vie de l'imprimante, sert de « ligne de base » propre à la machine) | idem |
| Période de grâce | `INIT_SAFE_FRAME_NUM = 30` images en début d'impression jamais considérées comme en échec (≈ 5 min à 10 s) | idem |
| Règle de décision | on calcule `a = (EWM − moyenne_longue) × sensibilité / facteur`. Si `a < 0,38` → pas d'échec ; si `a > 0,78` → échec ; entre les deux → échec si `a > (moyenne_courte − moyenne_longue) × 3,8` | `prediction.py`, `settings.py` |
| Alerte vs pause | **alerte** avec facteur 1 ; **pause** si la même règle est vraie avec le score divisé par `ESCALATING_FACTOR = 1,75` (il faut donc un signal nettement plus fort pour mettre en pause) | `backend/lib/failure_detection.py` |
| Sensibilité utilisateur | multiplicateur de 0,8 à 1,2 par pas de 0,05, défaut 1,0 ; « Low » < 0,95, « High » > 1,05 | `frontend/src/views/PrinterSettingsPage.vue`, modèle `detective_sensitivity` |
| Seuil d'affichage des boîtes | 0,2 | `settings.py` |

Points intéressants à reprendre pour notre assistant :
1. **Le score est relatif à une ligne de base propre à la machine** (moyenne longue) : une caméra qui « voit toujours un peu de spaghetti » (supports, fond chargé) ne déclenche pas en permanence.
2. **Deux niveaux** (alerte puis pause) avec une marge de 1,75× : bon compromis rappel / fausses pauses.
3. **Lissage temporel** : une image isolée ne déclenche pas ; il faut plusieurs images cohérentes.
4. **Contre-partie** : avec une image toutes les 10 s et une demi-vie de ~40 s, le délai typique entre l'apparition visible du spaghetti et la pause est de l'ordre de la minute. C'est une détection **après** le décollement, pas une prédiction.

Note : le développeur de PrintGuard (section 3) rapporte, sur ses propres jeux de test, une exactitude de **53,8 %** pour le modèle Obico contre 93,6 % pour le sien ; c'est une **évaluation par l'auteur d'un outil concurrent**, sur des images statiques, à prendre avec prudence ([PrintGuard](https://github.com/oliverbravery/PrintGuard)).

---

## 3. Autres modèles et jeux de données ouverts

| Ressource | Contenu | Licence | Où |
|---|---|---|---|
| **CAXTON** (Brion & Pattinson, Cambridge, 2022) | ~1,27 M images de la zone buse, 192 impressions, 8 Creality CR-20 Pro, PLA ; étiquetage **automatique** de 4 paramètres (débit, vitesse, Z offset, température), chacun en 3 classes (bas / bon / haut) | **CC BY 4.0** | [Dépôt Apollo Cambridge, DOI 10.17863/CAM.84082](https://www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8/full) ; article [Nature Communications](https://www.nature.com/articles/s41467-022-31985-y) |
| **PrintGuard** | Encodeur ShuffleNetV2 + classification par prototype le plus proche (apprentissage « few-shot ») ; seuil d'alerte, durée minimale de persistance et temps de recharge réglables par moniteur ; intégration Moonraker, alertes, MQTT. Rapporte 93,6 % d'exactitude et ~15 images/s sur Raspberry Pi 4 (auto-évaluation) | **GPL-2.0** | [github.com/oliverbravery/PrintGuard](https://github.com/oliverbravery/PrintGuard) |
| **Edge-FDM-Fault-Detection** | Code d'entraînement/évaluation de PrintGuard, y compris une évaluation du modèle Obico ; agrège 7 jeux publics (3 Roboflow, 4 Kaggle) via un script de fusion | **GPL-2.0** (code) ; licences des jeux à vérifier une par une | [github.com/oliverbravery/Edge-FDM-Fault-Detection](https://github.com/oliverbravery/Edge-FDM-Fault-Detection) |
| Roboflow « 3d-fdm-failures » | 916 images, classes spaghetti et stringing (boîtes) | **CC BY 4.0** | [Roboflow Universe](https://universe.roboflow.com/computer-vision-uxpzu/3d-fdm-failures-4rd2r) |
| Hugging Face « failures-3D-print » | 73 images, 4 catégories de boîtes non documentées | **inconnue** (à éviter pour un usage redistribué) | [huggingface.co/datasets/Javiai/failures-3D-print](https://huggingface.co/datasets/Javiai/failures-3D-print) |
| 3D-EDM (Jeong & Yoo, 2022) | ~1,3 k images **récupérées sur Google/YouTube** (décalage, fils, sous-extrusion, warping, normal) | non précisée ; images de provenance web | [arXiv 2203.12147](https://arxiv.org/pdf/2203.12147) |
| Obico | Poids du modèle téléchargeables (Darknet/ONNX) ; données d'entraînement **non publiées** | code AGPL-3.0 | [obico-server](https://github.com/TheSpaghettiDetective/obico-server) |

Attention : sur Roboflow et Kaggle, chaque jeu a sa propre licence (souvent CC BY 4.0, parfois plus restrictive ou non précisée). Vérifier avant de réentraîner un modèle que l'on redistribue.

---

## 4. Résultats universitaires

### 4.1 Vision
- **CAXTON** ([Nature Communications 2022](https://www.nature.com/articles/s41467-022-31985-y)) : réseau à attention résiduelle (Attention-56) à 4 têtes, entrée 224×224 recadrée sur la buse. Exactitude test globale **84,3 %** (débit 87,1 %, vitesse 86,4 %, Z offset 85,5 %, température 78,3 %). Inférence à **2,5 Hz** et **correction en boucle fermée** des paramètres (découpage du parcours en segments de 1 mm). Généralise à une autre imprimante (Lulzbot Taz 6, buse 0,6, filament 2,85), d'autres caméras et matériaux (TPU, chargé carbone). Les auteurs reconnaissent que les **échecs mécaniques, fissures, warping et décollements ne sont pas couverts**. Intérêt : c'est l'une des rares approches qui **agit avant l'échec** en corrigeant la dérive de débit/température.
- **Petsiuk & Pearce 2020** ([arXiv 2003.05660](https://arxiv.org/pdf/2003.05660.pdf)) : analyse couche par couche avec une seule caméra (vue de dessus rectifiée + pseudo-vue de côté), comparaison au G-code, recalage ICP, analyse de texture. Étude de faisabilité **sans métriques de rappel** ; ~21 s d'analyse par couche (+47 % de temps d'impression).
- **Petsiuk & Pearce 2021** ([arXiv 2111.02703](https://www.arxiv.org/pdf/2111.02703)) : on génère par rendu Blender une image « idéale » de chaque couche à partir du G-code et on compare avec la photo via HOG ; aucune donnée d'entraînement nécessaire ; anomalies ≥ ~5 mm détectables ; six types d'échec testés qualitativement (spaghetti, décollement, décalage, corps étranger…). Idée réutilisable : **comparer à une référence issue du G-code** plutôt que d'apprendre ce qu'est un échec.
- **Delli & Chang 2018** ([résumé 3DPrint.com](https://3dprint.com/222051/machine-learning-process-monitor/)) : pauses à des points de contrôle géométriques, photo, SVM « bon / défectueux ». Inconvénients : il faut mettre en pause, la vue de dessus rate les défauts de paroi.
- **AbouelNour & Gupta 2023** ([Additive Manufacturing, version NSF](https://par.nsf.gov/servlets/purl/10451119)) : caméra optique + caméra infrarouge, défauts (vides) volontairement insérés dans le G-code ; la température moyenne des zones chaudes augmente avec le nombre de défauts ; pas de rappel/précision rapportés.
- **3D-EDM** ([arXiv 2203.12147](https://arxiv.org/pdf/2203.12147)) : petit CNN, 96,7 % (binaire) et 93,4 % (4 classes) **sur des images web statiques** ; malgré le titre, **aucune mesure de précocité**.

### 4.2 Son, vibration, courant
- **Émission acoustique — Wu, Yu & Wang 2017** ([Int. J. Adv. Manuf. Technol.](https://link.springer.com/doi/10.1007/s00170-016-9548-6)) : capteur d'émission acoustique, modèle semi-markovien caché pour reconnaître les états de la machine en temps réel ; une revue lui attribue ~91,9 % d'exactitude.
- **Rupture de filament par émission acoustique — Sensors 2018** ([MDPI](https://www.mdpi.com/1424-8220/18/3/749)) : capteur large bande collé sur le corps de l'extrudeur, indicateurs d'asymétrie (skewness) et de similarité (coefficient de Bhattacharyya) par rapport à une période de référence ; rupture identifiée, sans taux chiffré dans la partie lisible.
- **Vibrations — Li et al. 2019** ([Sensors](https://www.mdpi.com/1424-8220/19/11/2589)) : accéléromètres sur plateau et extrudeur ; LS-SVM pour le blocage du filament (91–97,5 % selon le motif de remplissage), réseau BP pour warping et amas de matière (> 95 %).
- **Courant moteur extrudeur — Tlegenov, Lu & Hong 2019** ([Progress in Additive Manufacturing](https://link.springer.com/article/10.1007/s40964-019-00089-3)) : modèle physique reliant le courant du moteur d'extrusion au diamètre effectif de la buse ; une baisse de 8 % du diamètre effectif fait monter le courant d'environ 23 %, 13 % → +38 %. **Signal précurseur intéressant du bouchon progressif**, mais il faut un moteur asservi en vitesse ou une mesure de charge (sur Klipper avec TMC, StallGuard/SG_RESULT peut servir d'indicateur grossier ; pas disponible sur la carte d'origine du CR-10S).
- **Multi-capteurs bas coût — Kumar et al. 2022** ([Sensors](https://www.mdpi.com/1424-8220/22/2/517), CC BY) : vibration + microphone + pince de courant sur Arduino ; défauts provoqués (nivellement, température d'extrusion, tension de courroie) ; ~94 % d'exactitude normal/défaut.
- **Fusion multimodale — Waheed et al. 2026** ([arXiv 2602.16108](https://arxiv.org/pdf/2602.16108)) : micro stéréo, accéléromètre, caméra thermique ; le son atteint ~90 % et le thermique 100 % en conditions contrôlées pour « extrusion / pas d'extrusion », mais **la fusion n'est pas évaluée**.
- **Revue — Sampedro et al. 2022** ([Sensors](https://www.mdpi.com/1424-8220/22/23/9446)) : synthèse 2017–2021. Problèmes récurrents : **données d'échec rares et déséquilibrées**, dérive des conditions, diagnostic de cause peu étudié, **latence rarement rapportée** ; recommande de publier précision, rappel et F1, pas seulement l'exactitude.

### 4.3 Comment « précoce » est (rarement) mesuré
Dans les travaux ci-dessus, la « détection précoce » est presque toujours un **argument de motivation**, pas une mesure. Les métriques sont calculées par image ou par fenêtre de signal, sans dire **combien de temps avant la perte de la pièce** l'alarme arrive. Exceptions partielles : CAXTON mesure la vitesse de **correction** (boucle fermée à 2,5 Hz), Obico fixe implicitement un délai par ses paramètres de lissage. Pour notre usage, il faut donc **définir nous-mêmes** la mesure du délai d'anticipation (section 7).

---

## 5. Approche multi-signaux pour un CR-10S sous Klipper

### 5.1 Signaux disponibles sans matériel supplémentaire (via Moonraker)
| Signal | Ce qu'il annonce | Comment l'exploiter |
|---|---|---|
| Température buse / plateau + consigne | défaut de chauffe, thermistance, courant d'air, ventilateur qui refroidit le bloc | écart à la consigne et pente ; Klipper arrête déjà via `verify_heater` (max_error 120, check_gain_time 20 s buse / 60 s plateau) — l'assistant peut alerter **avant** ce seuil ([Config Reference](https://www.klipper3d.org/Config_Reference.html)) |
| Puissance de chauffe (PWM) | pour une température stable, une puissance qui grimpe = perte de chaleur (chaussette, ventilateur, courant d'air) ; une puissance qui chute = apport de chaleur anormal | moyenne glissante, comparaison à la même phase d'impressions précédentes |
| Progression / position / hauteur Z / fichier | temps de couche anormal, impression bloquée | comparer le temps de chaque couche à l'estimation du trancheur |
| Capteur de filament (switch d'origine) | fin de filament | `pause_on_runout` ([Config Reference](https://www.klipper3d.org/Config_Reference.html)) |
| Objets (`EXCLUDE_OBJECT_DEFINE`) | où se trouve chaque pièce | recadrer l'image sur la zone de chaque objet, abandon ciblé ([Exclude Object](https://www.klipper3d.org/Exclude_Object.html)) |
| Vitesse/débit (`M220`/`M221`), PA en cours | contexte pour interpréter l'image | gating par phase |

### 5.2 Ajouts matériels peu coûteux à fort rendement
1. **Capteur de mouvement de filament (encodeur)** `[filament_motion_sensor]` : détecte l'**arrêt du défilement** (bouchon, grinding, nœud), pas seulement l'absence ; c'est le meilleur précurseur bon marché des échecs d'extrusion. Adapter `detection_length` (7 mm par défaut) à la longueur du Bowden et aux rétractions.
2. **Microphone près de l'extrudeur** : le clic de l'extrudeur qui saute des pas et les chocs buse/pièce sont audibles bien avant que l'échec soit visible (cf. travaux acoustiques ci-dessus).
3. **Deuxième caméra de côté**, à hauteur de plateau, éclairage fixe : voit le soulèvement des coins (warping) et les décalages de couches.
4. **Tachymètre du ventilateur de radiateur** (`tachometer_pin` dans `[heater_fan]`) : précurseur direct du heat creep.
5. **Prise pilotée** déclarée dans Moonraker (`[power]`) : coupure secteur à distance en cas d'événement de sécurité ([Moonraker](https://moonraker.readthedocs.io/en/latest/configuration/)).
6. Optionnel : capteur de température sur le radiateur (`[temperature_sensor]`), accéléromètre ADXL345 sur la tête (chocs).

### 5.3 Architecture de fusion recommandée
- **Couche 1 — règles physiques déterministes** (rapides, explicables) : capteur de filament, écart de température, ventilateur arrêté, erreur Klipper. Peu de faux positifs, déclenchement immédiat.
- **Couche 2 — scores d'anomalie par signal**, relatifs à une **ligne de base** (même machine, même matériau, même phase), comme le fait Obico avec sa moyenne longue.
- **Couche 3 — vision** : détecteur de spaghetti (Obico/PrintGuard) + comparaison à une **référence issue du G-code** (silhouette attendue de la couche, hauteur attendue) pour détecter qu'une pièce a bougé ou ne grandit plus.
- **Fusion** : un échec est déclaré quand **deux sources indépendantes** concordent (ex. image + défilement filament), ou quand une source « forte » seule dépasse un seuil élevé. **Gating par contexte** : ne pas interpréter la première couche, les supports ou une pause comme un spaghetti ; ignorer les images lors d'un changement d'éclairage.
- **Actions graduées** : notifier → réduire vitesse/ajuster (`M220`, `SET_PRESSURE_ADVANCE`, `M106`…) → `PAUSE` → `CANCEL_PRINT`/coupure. Une pause coûte peu ; une annulation ou une coupure se demande à l'humain, sauf urgence de sécurité.
- **Calcul** : ne pas faire tourner l'inférence sur la même machine que Klipper si elle est chargée — une surcharge de l'hôte provoque des arrêts « Timer too close » ([Klipper Discourse](https://klipper.discourse.group/t/diagnose-timer-too-close-event/2121)).

---

## 6. Conseils pratiques : rappel très élevé avec peu de fausses alarmes

1. **Séparer alerte et action** : seuil bas pour alerter (rappel), seuil haut + persistance pour mettre en pause (précision). Obico utilise un facteur 1,75 entre les deux.
2. **Persistance temporelle** : exiger *N* images sur *M* (ex. 3 sur 5) ou un score lissé (EWM), avec hystérésis pour ne pas osciller.
3. **Ligne de base par machine et par impression** : comparer au comportement « normal » de la même imprimante et de la même phase.
4. **Gating par phase** : première couche, supports, ponts, pauses et purges ont chacun leur logique.
5. **Image stable** : éclairage LED fixe (pas la lumière de la pièce), caméra rigide (pas sur le plateau mobile, sauf caméra dédiée), mise au point fixe, fond uni contrasté avec le filament.
6. **Recadrage** sur les objets (coordonnées issues du G-code) pour que le modèle voie la pièce en grand.
7. **Fréquence d'analyse** : 0,1 Hz suffit pour le spaghetti ; pour les précurseurs fins (coins qui se lèvent, première couche), viser plutôt 0,5–2 Hz sur une région recadrée.
8. **Plusieurs modalités indépendantes** : les fausses alarmes d'une caméra et d'un capteur de filament sont peu corrélées ; exiger leur accord fait chuter les faux positifs sans trop réduire le rappel, à condition qu'au moins une des modalités voie chaque type d'échec.
9. **Mesurer les fausses alarmes par heure d'impression**, pas par image : 1 % de faux positifs par image à 6 images/min = une fausse alarme toutes les ~17 min.
10. **Boucle humaine** : chaque alarme confirmée ou infirmée par l'utilisateur alimente le jeu de validation (comme les retours Obico).
11. **Calibrer les seuils sur un jeu de validation séparé** (par impression, pas par image) en visant un rappel cible, puis vérifier sur un jeu de test jamais vu.

---

## 7. Mesurer rigoureusement « prédit ≥ 98 % des échecs avant qu'ils arrivent »

### 7.1 Définir l'événement et l'instant critique
- **Unité de comptage = l'événement d'échec**, pas l'image. Un événement = une impression (ou un objet) qui devient irrécupérable.
- Pour chaque événement, annoter sur le timelapse et les journaux l'instant **t_PNR** (point de non-retour : pièce qui bouge, extrusion arrêtée depuis X couches, décalage visible…). Utiliser la colonne `point_de_non_retour` de la base de connaissances comme guide d'annotation, avec deux annotateurs et arbitrage.
- Fixer à l'avance un **délai d'anticipation minimal L** (ex. 60 s ou une couche) et une **fenêtre maximale W** (ex. 30 min) : une alarme est un **succès** si `t_PNR − W ≤ t_alarme ≤ t_PNR − L`. Une alarme plus tardive est une **détection tardive** (comptée à part, pas comme une prédiction). Une alarme plus précoce que la fenêtre sans échec qui suit est une **fausse alarme**.
- Les métriques « par plage » pour séries temporelles (récompense d'existence + recouvrement) formalisent cette idée ([Tatbul et al., NeurIPS 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf)).

### 7.2 Métriques à publier
- **Rappel par événement** = événements prédits dans la fenêtre / événements totaux, **global et par type d'échec** (un 98 % global peut cacher un 0 % sur les bouchons).
- **Distribution du délai d'anticipation** (médiane, 10e centile) — c'est elle qui dit si l'humain ou l'automate a le temps d'agir.
- **Taux de fausses alarmes** par 100 h d'impression normale, séparément pour « alerte » et « pause ».
- Rappel « détection tardive incluse », pour comparaison avec les outils existants.

### 7.3 Intervalles de confiance et nombre d'événements nécessaires
Le rappel se traite comme une proportion binomiale. La borne **Clopper-Pearson** (exacte, conservatrice) se calcule avec les quantiles de la loi bêta ([Wikipédia — intervalles pour une proportion](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)). La **règle de trois** donne l'approximation pour zéro raté : si aucun échec n'est manqué sur *n*, le taux de ratés est < 3/*n* à 95 % ([Wikipédia — rule of three](https://en.wikipedia.org/wiki/Rule_of_three_(statistics))).

Pour affirmer « rappel ≥ 98 % » avec 95 % de confiance (borne inférieure unilatérale de Clopper-Pearson) :

| Ratés tolérés | Événements nécessaires (unilatéral 95 %) | (bilatéral 95 %) |
|---|---|---|
| 0 | **149** | 183 |
| 1 | 236 | 277 |
| 2 | 313 | 359 |
| 3 | 386 | 436 |
| 5 | 523 | — |

Exemples : 50/50 réussis → borne inférieure 94,2 % seulement ; 100/100 → 97,1 % ; 150/150 → 98,0 %. La règle de trois donne le même ordre de grandeur (3/150 = 2 %).

Pour les fausses alarmes, même logique avec une loi de Poisson : **zéro fausse pause en T heures** permet d'affirmer un taux < 3/T par heure à 95 %. Pour garantir « moins d'une fausse pause pour 100 h », il faut ~**300 h d'impression normale sans aucune fausse pause**.

Pièges statistiques :
- **Non-indépendance** : plusieurs événements issus de la même impression ou du même réglage défectueux ne sont pas indépendants. Compter par impression et estimer l'incertitude par **bootstrap par impression**.
- **Fuite de données** : séparer entraînement/validation/test **par impression** (et idéalement par jour, bobine, éclairage), jamais par image.
- **Stratification** : viser un nombre minimal d'événements **par type d'échec** ; sinon annoncer la garantie seulement pour les types suffisamment représentés.
- **Pré-enregistrer** le protocole (L, W, seuils, définition de t_PNR) avant de regarder le jeu de test.
- **Tests séquentiels** : si l'on vérifie le résultat au fil de l'eau, utiliser une procédure séquentielle ou fixer *n* à l'avance.

### 7.4 Provoquer des échecs de façon sûre
Les échecs naturels sont trop rares pour atteindre 150 événements en un temps raisonnable : il faut en **provoquer**, sans jamais toucher aux protections thermiques.

| Échec visé | Méthode sûre (logicielle de préférence) |
|---|---|
| Décalage de couche | `SET_GCODE_OFFSET X_ADJUST=2` ou `Y_ADJUST=2` à une hauteur choisie (décalage purement logiciel, sans forcer les moteurs) |
| Sous-/sur-extrusion | `M221 S50`–`S70` / `M221 S130`–`S150` à une couche donnée |
| Arrêt d'extrusion / runout | couper le filament au-dessus de l'extrudeur, ou fin de bobine contrôlée |
| Décollement / spaghetti | pièce haute et étroite sans bordure, Z offset relevé (`SET_GCODE_OFFSET Z_ADJUST=+0.1`), plateau non nettoyé ou agent de démoulage, petite surface de contact |
| Warping | ABS/PETG sans bordure, plateau abaissé en cours (`SET_HEATER_TEMPERATURE HEATER=heater_bed TARGET=<plus bas>`) |
| Stringing, pillowing, surchauffe | température +15 °C, ventilateur coupé (`M106 S0`) sur pièces de test |
| Première couche ratée | Z offset volontairement faux de ±0,1 mm |
| Bouchon partiel | filament contenant une petite impureté (avec prudence), ou débit volumique au-delà de la capacité |

À **ne pas** provoquer : emballement thermique, thermistance débranchée, ventilateur de radiateur coupé longtemps, blob autour du bloc chauffant, défaut électrique. Pour ces cas, tester la **chaîne d'alerte** (envoi de fausses mesures dans un simulateur, rejouer des journaux enregistrés) plutôt que la machine. Toujours être présent, avoir un détecteur de fumée et un extincteur, plateau dégagé, `verify_heater` actif, `max_temp` raisonnable.

Compléments utiles :
- **Rejouer** les flux enregistrés (images + télémétrie) pour évaluer de nouveaux modèles sans imprimer.
- Mélanger échecs provoqués et **échecs naturels** : un modèle qui ne voit que des échecs induits apprend leurs artefacts (ex. saut brutal de M221) plutôt que les vraies dérives progressives.
- Varier filaments, couleurs, éclairage, géométries, position sur le plateau.
- Pour les cas très rares, compter aussi les **quasi-incidents** (coin relevé rattrapé) comme données d'entraînement, mais pas comme événements de test.

---

## 8. Sources principales
- Obico : [blog](https://www.obico.io/blog/how-obico-ai-failure-detection-works/), [code serveur (AGPL-3.0)](https://github.com/TheSpaghettiDetective/obico-server)
- PrintGuard : [github.com/oliverbravery/PrintGuard](https://github.com/oliverbravery/PrintGuard), [Edge-FDM-Fault-Detection](https://github.com/oliverbravery/Edge-FDM-Fault-Detection)
- CAXTON : [Nature Communications 2022](https://www.nature.com/articles/s41467-022-31985-y), [jeu de données CC BY 4.0](https://www.repository.cam.ac.uk/items/6d77cd6d-8569-4bf4-9d5f-311ad2a49ac8/full)
- Petsiuk & Pearce : [arXiv 2003.05660](https://arxiv.org/pdf/2003.05660.pdf), [arXiv 2111.02703](https://www.arxiv.org/pdf/2111.02703)
- AbouelNour & Gupta 2023 : [NSF PAR](https://par.nsf.gov/servlets/purl/10451119)
- Delli & Chang 2018 : [3DPrint.com](https://3dprint.com/222051/machine-learning-process-monitor/)
- 3D-EDM : [arXiv 2203.12147](https://arxiv.org/pdf/2203.12147)
- Acoustique : [Wu, Yu & Wang 2017](https://link.springer.com/doi/10.1007/s00170-016-9548-6), [Sensors 2018 rupture filament](https://www.mdpi.com/1424-8220/18/3/749)
- Vibration : [Li et al. 2019](https://www.mdpi.com/1424-8220/19/11/2589)
- Courant : [Tlegenov et al. 2019](https://link.springer.com/article/10.1007/s40964-019-00089-3)
- Multi-capteurs : [Kumar et al. 2022](https://www.mdpi.com/1424-8220/22/2/517), [Waheed et al. 2026](https://arxiv.org/pdf/2602.16108)
- Revue : [Sampedro et al. 2022](https://www.mdpi.com/1424-8220/22/23/9446)
- Jeux de données : [Roboflow 3d-fdm-failures](https://universe.roboflow.com/computer-vision-uxpzu/3d-fdm-failures-4rd2r), [HF failures-3D-print](https://huggingface.co/datasets/Javiai/failures-3D-print)
- Klipper : [Config Reference](https://www.klipper3d.org/Config_Reference.html), [Exclude Object](https://www.klipper3d.org/Exclude_Object.html), [FAQ](https://www.klipper3d.org/FAQ.html), [Discourse « Timer too close »](https://klipper.discourse.group/t/diagnose-timer-too-close-event/2121) ; Moonraker : [configuration](https://moonraker.readthedocs.io/en/latest/configuration/)
- Statistiques : [Clopper-Pearson](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval), [règle de trois](https://en.wikipedia.org/wiki/Rule_of_three_(statistics)), [Tatbul et al. 2018](https://papers.NeurIPS.cc/paper_files/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf)
- Sécurité : [Duet3D — fire safety](https://docs.duet3d.com/User_manual/Overview/Fire_safety), [Hackaday 2016](https://hackaday.com/2016/12/07/dont-leave-3d-printers-unattended-they-can-catch-fire/)
