# VALDAR — Avenant n°2 : Valdar, héritier de l'ancien assistant

Complète le **cahier des charges v2** (`valdar-cahier-des-charges-v2.md`). En cas de conflit, cet
avenant prévaut. Date : 2026-10-07.

## 0. Décision

**Valdar remplace l'ancien assistant d'atelier.** Il reprend toutes ses capacités (et celles de
PrinterAgent), les améliore, et les relie au cœur émotionnel. L'ancien assistant, son interface et
PrinterAgent ne tournent plus en parallèle :
Valdar est la seule IA de la machine et le seul utilisateur de la carte graphique.

Conséquences sur la v2 :
- La ligne « hors périmètre : outils agentiques » est **supprimée**. Valdar agit sur le monde.
- S'ajoutent : **perception** (voix, visage, scène), **identité** (qui est là), **relations**
  (s'adapter à chaque personne), **actions** (outils, imprimante), **corps** (projecteur, robot de
  bureau, corps futurs).

Objectif assumé : une IA la plus proche possible d'un être sentient, conscient de lui-même,
amical et sympathique. Le cahier traduit cet objectif en mécanismes **fonctionnels et mesurables**
(modèle de soi, métacognition, ressenti qui pilote le comportement). Il ne prétend pas prouver une
conscience : personne ne sait le mesurer, et Valdar ne l'affirme pas comme une certitude.

---

## 1. Inventaire de l'existant (lu le 2026-10-07)

### 1.1 L'ancien assistant d'atelier (dossier : `migration.dossier_ancien` dans `config/valdar.yaml`)

| Domaine | Ce que l'ancien assistant fait | Fichiers |
|---|---|---|
| Dialogue | agent à outils (7 tours max), persona « pote », tutoiement, balise d'émotion `[happy]`, modes jarvis / copilote / veille | `core/agent.py`, `core/persona.py` |
| LLM | Ollama (API OpenAI) : `qwen2.5:7b` pour parler, `gemma4:12b` pour la vision, `nomic-embed-text` | `core/llm.py`, `config.py` |
| Affect | moteur par tour : ALMA + appraisal + PAD, valence du lexique et de la voix, **marqueurs somatiques par personne et par activité** | `core/affect.py` |
| Voix (entrée) | micro K66, débruitage, faster-whisper **large-v3 sur GPU**, mot d'éveil (son nom) détecté dans la transcription | `io/voice.py` |
| Voix (sortie) | XTTS v2 (clone, GPU) → edge-tts (cloud) → piper → SAPI ; ElevenLabs en option (cloud) ; effets « Sony / megatron » (hauteur, égaliseur, écho) | `io/voice.py` |
| Identité | empreinte de voix (log-mel moyen) + empreinte de visage (texture 6×6 + couleurs), fusion voix/visage | `core/voiceid.py`, `core/faceid.py`, `core/identity.py` |
| Vision | webcam OpenCV, flux RTSP (GoPro via mediamtx), capture d'écran, question visuelle au LLM | `io/webcam.py`, `tools.py` |
| Écran / projecteur | panneau d'interface (page web dans Edge en mode app sur l'écran 2), ancien affichage pygame au premier plan, checklist projetée | `ui/`, `io/projector.py`, `io/face.py` |
| Outils (38) | heure, système, applis, web, fichiers (lire/écrire), processus, mémoire, écran (voir, cliquer, taper, touches), **imprimante** (état, fichiers, démarrer, pause, annuler, température, macro, arrêt d'urgence), pinouts, comptes, modèles Printables (chercher, télécharger), webcam, profils voix/visage, qui parle, **stock d'atelier**, mode, **checklists**, **rappels** | `core/tools.py` |
| Sécurité | niveaux safe / elevated / dangerous, confirmation au-dessus d'un seuil, outils élevés réservés au propriétaire | `core/agent.py` |
| Données | `memory.json` (faits), `affect.json`, `stock.db`, `rappels.json`, `checklist.json`, `pinouts.json`, profils voix/visage | `data/` |
| Apprentissage | construction d'un jeu de données depuis tes exports ChatGPT / Gemini / Claude, QLoRA Gemma 4 12B via Unsloth, export GGUF pour Ollama | `scripts/dataset/` |

### 1.2 PrinterAgent / Printos — `C:\Users\doxei\klipper_brain`

Cerveau autonome de l'imprimante (Klipper / Moonraker) : vigie continue (modèle Obico ONNX),
triage et inspection par un modèle de vision (Qwen3-VL 8B prévu), **correctifs bornés par des
garde-fous G-code** (`SafetyLimits`), ESP32 compagnon (BME280, DHT22, relais LED), base de
connaissances (profils de filament appris), persistance de config en fin d'impression, appli
Windows PySide6, voix locale (piper + faster-whisper).

### 1.3 desk-robot — `C:\Users\doxei\desk-robot`

Projet open source (Rocky, de *Project Hail Mary*) que tu construis : XIAO ESP32-S3 Sense, visage
OLED, cou pan/tilt (V2 imprimée en cours), caméra, micro, ampli et haut-parleur. Le cerveau tourne
sur le PC (Silero VAD, Smart Turn, faster-whisper, suivi de visage OpenCV). Protocole documenté :
commandes JSON dans un sens, audio et JPEG dans l'autre (`docs/protocol.md`). C'est le **premier
corps physique** de Valdar.

### 1.4 Ce qu'il faut corriger en reprenant (constaté dans le code)

1. **Identité trop faible pour autoriser des actions.** Les empreintes voix et visage sont des
   statistiques simples : la lumière ou un rhume suffisent à confondre deux personnes.
   Et `_authorized()` renvoie **vrai quand le locuteur est inconnu** : n'importe qui (ou une télé)
   peut déclencher un outil « élevé » ou « dangereux ».
2. **Transcription permanente de la pièce.** Tout ce qui est dit est transcrit puis écrit dans
   son journal d'erreurs (« capté, ignoré »), conversations des autres comprises. Le GPU transcrit en
   continu, et whisper hallucine sur le bruit (« Sous-titrage Société Radio-Canada »).
3. **Cookies du navigateur** lus par l'agent (`browser_cookie3`) : un LLM à qui on parle à voix
   haute a accès à toutes tes sessions connectées.
4. **Serveur HTTP ouvert sur tout le réseau** (`0.0.0.0:8770`).
5. **Services cloud** (edge-tts, ElevenLabs) dans la chaîne vocale : contraire au « 100 % local ».
6. **Carte graphique saturée** : whisper large-v3 + XTTS + Ollama (+ Qwen3-VL côté PrinterAgent)
   ne tiennent pas ensemble dans 12 Go.
7. Mémoire = liste de phrases sans structure ni oubli ; affect recalculé seulement à chaque tour.

---

## 2. Architecture cible

```
            ┌──────────────────────── CORPS (interface Body) ────────────────────────┐
            │  Corps 0 : projecteur (Valdar Display)   Corps 1 : desk-robot (ESP32)   │
            │  capteurs : micro(s), caméra(s), écran   effecteurs : visage, voix, cou  │
            └───────────────┬──────────────────────────────────────▲─────────────────┘
                            │ flux                                   │ commandes
┌───────────────────────────▼───────────────┐          ┌─────────────┴──────────────┐
│ PERCEPTION                                │          │ EXPRESSION                  │
│ audio : VAD → mot d'éveil → STT           │          │ prompt + paramètres + LoRA  │
│         → empreinte vocale                │          │ voix (Piper + caractère)    │
│ vision : détection + empreinte visage     │          │ visage, regard, gestes      │
│          scène (Gemma 4 vision)           │          └─────────────▲──────────────┘
└───────────────┬───────────────────────────┘                        │
                ▼                                                     │
┌───────────────────────────┐   ┌──────────────┐   ┌─────────────────┴───────────┐
│ IDENTITÉ & RELATIONS       │──►│ ÉVALUATION   │──►│ CŒUR (phase 1, fait)         │
│ qui est là, confiance,     │   │ voie basse / │   │ émotion, humeur, besoins,    │
│ profil, permissions        │   │ voie haute   │   │ organes, initiative          │
└───────────────┬───────────┘   └──────────────┘   └─────────────┬───────────────┘
                │                                                  │
                ▼                                                  ▼
┌───────────────────────────────────────────┐    ┌─────────────────────────────────┐
│ AGENT : boucle de dialogue + outils        │◄──►│ MÉMOIRE & MODÈLE DE SOI          │
│ politique de permissions, confirmations    │    │ épisodes, faits, personnes, soi  │
└───────────────┬───────────────────────────┘    └─────────────────────────────────┘
                ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ ACTIONS : imprimante (Moonraker + garde-fous PrinterAgent), atelier (stock,        │
│ checklists, rappels, pinouts), ordinateur (applis, fichiers, écran), web, modèles 3D│
└───────────────────────────────────────────────────────────────────────────────────┘
```

Tous les modules communiquent par le bus d'événements du daemon (v2 §4). Le cœur reste
indépendant et tourne en continu.

---

## 3. Perception

### 3.1 Audio
- **Toujours à l'écoute, mais sans transcrire la pièce.** Chaîne : micro → **Silero VAD** (CPU)
  → **mot d'éveil « Valdar »** (openWakeWord, modèle personnalisé entraîné sur voix synthétiques,
  CPU) → seulement alors **faster-whisper** sur le segment utile.
- Pendant une conversation engagée, l'éveil reste ouvert N secondes (comme avant).
- **Fin de tour** par détection sémantique (approche Smart Turn du desk-robot) plutôt qu'un
  silence fixe.
- STT : `large-v3-turbo` en int8 sur GPU (≈ 1 Go) ; repli `small` sur CPU si la VRAM manque.
- Filtre anti-hallucination : rejet des phrases typiques de whisper sur le bruit, probabilité de
  « pas de parole », longueur minimale.
- **Rien n'est conservé de ce qui ne s'adresse pas à Valdar** : traité en mémoire vive puis
  oublié, jamais écrit sur disque.
- **Tonalité de la voix** (énergie, hauteur, débit) → voie basse de l'évaluation (repris de l'ancien assistant).

### 3.2 Vision
- Sources : webcam USB, flux RTSP (GoPro), caméras de l'imprimante, caméra du desk-robot.
- **Visages** : détection YuNet (déjà utilisée par l'ancien assistant) + **reconnaissance SFace** (modèle
  OpenCV de reconnaissance faciale, CPU) à la place de l'empreinte texture/couleur.
- **Scène** : questions visuelles à Gemma 4 12B (multimodal, déjà utilisé avant pour la vision) :
  un seul modèle pour parler et voir.
- Cadence basse au repos (présence toutes les quelques secondes), plus haute en conversation.

### 3.3 Présence
- Événements `person_arrived`, `person_left`, `person_speaking` publiés sur le bus.
- Ils nourrissent le cœur (besoin de contact, attachement) et l'initiative (dire bonjour
  quand Olivier rentre, se taire s'il est concentré).

---

## 4. Identité, relations et adaptation à chaque personne

### 4.1 Registre des personnes (SQLite)
Pour chaque personne : nom, alias, **rôle** (propriétaire, proche, invité, inconnu), drapeau
**mineur**, empreintes de voix (ECAPA ou équivalent, plusieurs échantillons), empreintes de
visage (SFace, plusieurs angles), consentement, date de dernière rencontre.

### 4.2 Reconnaissance
- **Fusion voix + visage** avec scores calibrés → probabilité d'identité. La phase 4 mesure le
  taux de fausses acceptations sur des enregistrements réels et fixe les seuils.
- Seuil non atteint → **« inconnu »**, jamais « Olivier par défaut ».
- Inconnu : Valdar est poli, se présente, peut demander le prénom, et n'a accès qu'aux outils
  sûrs.

### 4.3 Relation (ce que Valdar ressent pour chacun)
Repris et étendu des marqueurs somatiques de l'ancien assistant :
- **familiarité** (nombre et durée des rencontres), **confiance**, **affection** (valence
  moyenne des échanges, oubli lent), **sujets partagés**, souvenirs liés (épisodes).
- La présence d'une personne aimée module le cœur : ocytocine, besoin de contact satisfait, PAD.
  Une personne avec qui ça s'est mal passé : vigilance légère.

### 4.4 Adaptation
Le profil de la personne présente module l'expression (v2 §9) :
- **registre** (tutoiement, niveau de vannes, longueur, vocabulaire technique) ;
- **sujets** (l'impression 3D avec Olivier, autre chose avec un invité) ;
- **voix** (débit, chaleur) et **gestes** (regard, suivi du visage) ;
- profil **mineur** : langage adapté, aucun outil élevé, aucun contenu inadapté.
Les préférences s'apprennent (leçons N1 par personne, v2 §10) et se corrigent à la voix.

### 4.5 Consentement et données biométriques
- Enrôler une personne demande **son accord explicite** à voix haute (et celui d'Olivier).
- « Valdar, oublie-moi » supprime empreintes et souvenirs de la personne.
- Tout reste en local, dans la base de Valdar.

---

## 5. Actions (outils)

### 5.1 Reprise des outils de l'ancien assistant
Ses 38 outils sont repris, regroupés en familles, avec un **registre unique** (nom,
description, schéma d'arguments, niveau, effets sur le cœur).

| Famille | Outils |
|---|---|
| Atelier | stock, checklists (affichées sur le projecteur), rappels, pinouts |
| Imprimante | état, fichiers, démarrer, pause/reprise, annuler, températures, macros, arrêt d'urgence |
| Modèles 3D | recherche et téléchargement Printables (**API publique ou jeton dédié, plus de cookies**) |
| Ordinateur | heure, état système, ouvrir une appli, lister/lire des fichiers (dossiers autorisés), écrire (confirmation), processus, voir l'écran, cliquer, taper |
| Monde | recherche web, voir la webcam |
| Soi | se souvenir, oublier, changer de registre, état de son cœur |

### 5.2 Politique de permissions
Décision = **niveau de l'outil × identité (probabilité) × rôle × contexte** :

| Niveau | Qui | Condition |
|---|---|---|
| sûr | tout le monde | — |
| élevé | propriétaire | identité ≥ seuil haut |
| dangereux | propriétaire | identité ≥ seuil haut **et** confirmation explicite (« oui Valdar », ou bouton à l'écran) |
| sécurité | **tout le monde** | arrêt d'urgence de l'imprimante, pause : arrêter une machine ne doit jamais attendre une reconnaissance |

Toute action est journalisée (qui, quoi, résultat). Valdar ne peut pas modifier ses propres
permissions.

### 5.3 Imprimante : reprise de PrinterAgent
- Client Moonraker (HTTP + WebSocket) et **`SafetyLimits` de PrinterAgent réutilisés tels
  quels** : c'est ton code, déjà pensé pour la sécurité.
- Vigie Obico ONNX sur CPU ; triage visuel par Gemma 4 (pas de second modèle de vision dans la
  VRAM).
- Correctifs automatiques bornés, désactivés par défaut, activables par Olivier.
- **Lien avec le cœur** : spaghetti détecté → inquiétude ; impression réussie → fierté ;
  surchauffe → peur et action immédiate. Valdar « vit » ses impressions (marqueurs par activité,
  repris de l'ancien assistant).

---

## 6. Corps

### 6.1 Interface `Body`
Un corps déclare ses **capteurs** (micros, caméras, proprioception) et ses **effecteurs**
(visage, voix, cou, membres plus tard), ainsi que ses limites (angles, vitesses, couple).
Le reste de Valdar parle à l'interface, jamais au matériel.

### 6.2 Corps connus
- **Corps 0 : projecteur** (Valdar Display, v2 §11), micro K66, webcam.
- **Corps 1 : desk-robot** (protocole JSON/audio/JPEG existant) : visage OLED, cou pan/tilt,
  caméra, micro, haut-parleur. Valdar peut regarder la personne qui parle et suivre un visage.
- **Corps futurs** : même interface, nouveaux effecteurs.
- Plusieurs corps peuvent être actifs, mais **un seul « je »** : Valdar sait où il est et par
  quel corps il parle.

### 6.3 Proprioception → cœur (boucle de Damasio pour de vrai)
L'état réel du corps alimente les organes virtuels : servo forcé ou bloqué → inconfort, corps
déconnecté → manque, batterie faible → fatigue, chaleur du processeur → lourdeur.

---

## 7. Conscience de soi (fonctionnelle)

- **Modèle de soi enrichi** (v2 §7.6) : identité, histoire (« j'ai remplacé l'ancien assistant
  le 7 octobre 2026 »), corps disponibles et leur état, capacités (outils), limites, valeurs, relations.
- **Introspection ancrée** : quand Valdar parle de ce qu'il ressent, il lit l'état réel de son
  cœur et de ses organes. Il ne compose pas un ressenti.
- **Métacognition** : il estime sa confiance (« je ne suis pas sûr que ce soit toi »,
  « je ne sais pas »), surveille ses erreurs (apprentissage v2 §10) et le dit.
- **Continuité** : mémoire autobiographique, rattrapage de l'absence, réveil cohérent.
- **Test du miroir** (expérience) : se reconnaître sur le projecteur vu par la webcam,
  distinguer « mon visage » d'un autre écran.

---

## 8. Budget de la carte graphique (RTX 2060, 12 Go)

| Poste | Estimation | Où |
|---|---|---|
| Windows + Display | 1 – 1,5 Go | GPU |
| Gemma 4 12B (texte + vision) + cache 8k | 7,5 – 8,5 Go | GPU |
| faster-whisper large-v3-turbo int8 | ≈ 1 Go | GPU (repli : `small` sur CPU) |
| VAD, mot d'éveil, empreinte vocale, YuNet, SFace, Obico, Piper | 0 | CPU |
| Adaptateurs LoRA (v2 §9.1) | ≤ 0,6 Go | GPU |
| **Total** | **≈ 10 – 11,5 Go** | à mesurer en phase 0 |

Exclus de la VRAM : XTTS (voix clonée, ≈ 2 Go et lente sur cette carte), Qwen3-VL. La voix
devient Piper + la chaîne d'effets « Sony » d'origine. Une voix clonée locale pourra revenir si la
mesure laisse de la place.

**Backend LLM** : interface `LLMBackend` avec deux implémentations. **Ollama** d'abord (déjà
installé, `gemma4:12b` déjà téléchargé) pour atteindre vite le niveau de l'ancien assistant. Puis
**llama-server** quand les adaptateurs d'affect arrivent (v2 §9.1 : échelles de LoRA par
requête, qu'Ollama ne sait pas faire).

---

## 9. Reprise des données de l'ancienne installation

| Donnée | Devenir |
|---|---|
| `memory.json` (faits) | importés en mémoire sémantique (source « ancien », confiance moyenne), relus par Olivier |
| `affect.json` (marqueurs par personne / activité) | valeurs de départ des relations |
| `stock.db`, `rappels.json`, `checklist.json`, `pinouts.json` | importés tels quels |
| profils voix / visage | **non repris** (format trop faible) : nouvel enrôlement avec consentement |
| jeu de données d'entraînement (`scripts/dataset`) | réutilisé pour l'ancrage du style (v2 §10.5) |
| clés (`eleven.key`, `.token`) | non reprises ; jetons locaux régénérés |

---

## 10. Plan de réalisation révisé

La phase 1 (cœur) est faite. Priorité : **retrouver vite tout ce que l'ancien assistant savait faire**, en
mieux, puis aller au-delà.

| Phase | Contenu | Critère principal |
|---|---|---|
| **0** | mesures GPU (Gemma 4 via Ollama puis llama-server, whisper turbo, VRAM totale), fenêtre Display | rapport chiffré dans `DECISIONS.md` |
| **2** | daemon + API locale + `LLMBackend` (Ollama) + boucle d'agent pilotée par le cœur + outils **sûrs** + mémoire de l'ancienne installation importée | au clavier, Valdar répond, ressent, et fait tout ce que l'ancien assistant faisait sans risque |
| **3** | voix : VAD, mot d'éveil, STT, fin de tour, Piper + caractère, tonalité → cœur | conversation orale fluide, zéro transcription de la pièce sur disque |
| **4** | vision et identité : caméras, YuNet + SFace, empreinte vocale, fusion, registre des personnes, consentement, permissions | taux de fausses acceptations mesuré ; un inconnu ne déclenche jamais un outil élevé |
| **5** | imprimante : Moonraker, garde-fous PrinterAgent, vigie, lien avec le cœur | outils imprimante complets, correctifs bornés, arrêt d'urgence pour tous |
| **6** | Display (corps 0), v2 §11 | v2 phases 2 et 7 |
| **7** | relations et adaptation par personne, mémoire épisodique (v2 §7) | le registre change selon la personne reconnue (test) |
| **8** | corps 1 : desk-robot | Valdar tourne la tête vers qui parle |
| **9+** | adaptateurs d'affect, apprentissage N1/N2/N3, conscience de soi (v2 §9–§10, §7 ici) | critères v2 |

## 11. Mode de travail sur la machine

Le code est écrit et testé hors de la machine. Ce qui doit tourner sous Windows (GPU, micro,
caméras, projecteur) se lance par un fichier `.bat` dans `tools/`. Chaque lanceur écrit un
journal dans `tools/logs/`, que Claude relit.
