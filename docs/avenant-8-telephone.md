# VALDAR — Avenant n°8 : le téléphone (12 novembre 2026)

Olivier veut lui parler depuis son téléphone — en écrit, en vocal, en vidéo —, le mettre à
la place de Gemini comme assistant du téléphone, et le laisser piloter l'appareil (afficher
une page, lancer de la musique). Chez lui sans Tailscale, et ailleurs avec.

Cet avenant ne remplace rien : les avenants 2 à 7 restent valables.

---

## 1. Ce n'est pas une appli du Play Store

Valdar **sert lui-même** sa page. Le téléphone l'ajoute à son écran d'accueil : icône,
plein écran, pas de barre de navigateur. Rien à publier, aucun compte, et tout reste chez
Olivier.

Pourquoi pas une appli hébergée (Lovable et compagnie) : une page en https n'a pas le droit
d'appeler un PC en http sur le réseau local, et ce PC n'est pas sur Internet. Ça demanderait
un tunnel et un certificat pour un résultat moins bon.

## 2. En https, et ce n'est pas un détail

**Le micro et la caméra d'un téléphone ne s'ouvrent que sur une page sûre.** En http, le
navigateur refuse, point. Donc Valdar fabrique son propre certificat (pour son adresse
Wi-Fi, son adresse Tailscale et son nom de machine) et sert la page en https sur le port
8766. La fenêtre du PC, elle, garde son http sur 127.0.0.1 : deux portes, deux règles.

- **Chez lui, sans Tailscale** : `https://192.168.x.x:8766`. Le téléphone dit « connexion
  non sécurisée » **la première fois** (le certificat est celui de Valdar, pas celui d'une
  autorité) : Avancé → Continuer, et c'est réglé pour de bon.
- **Ailleurs, par Tailscale** : l'adresse MagicDNS marche pareil. Et avec
  `tailscale serve`, c'est Tailscale qui fournit un vrai certificat : plus aucun
  avertissement.

## 3. Le jeton, et qui il croit avoir au bout du fil

Un jeton tiré au sort vit dans `data/tel.token` (jamais dans le dépôt). Il voyage dans le
QR code, le téléphone le garde, et le remet à chaque appel.

**Tout ce qui arrive par la porte du téléphone doit le montrer** — même ce qui semble venir
de la machine elle-même, parce que derrière un tunnel le trafic sort justement de là.

Le jeton dit « c'est Olivier au téléphone », avec une confiance de **0,7** par défaut. En
dessous de 0,75, le **corps numérique reste fermé** : depuis le téléphone il parle, écoute,
voit, se souvient, utilise ses outils d'atelier — mais il ne clique pas sur le PC. Mettre
`tel.confiance: 0.8` lui ouvre le corps à distance ; c'est une décision d'Olivier, pas un
défaut.

Et le téléphone n'a pas accès à tout : ni les réglages, ni la mémoire, ni le foyer, ni le
journal. Ça se fait devant le PC.

## 4. Ce que fait la page

| | |
|---|---|
| **Écrit** | ses phrases s'écrivent en direct, au fur et à mesure qu'il les pense |
| **Vocal (entrée)** | bouton « appuie et parle » : le son est enregistré, ramené à 16 kHz et empaqueté en WAV **par le téléphone** — le PC n'a aucun décodeur à installer |
| **Vocal (sortie)** | chaque phrase est dite par **sa** voix (XTTS) sur le PC et jouée dans le téléphone. Voix pas chargée ? il le dit, et le téléphone peut lire la réponse avec la voix du système |
| **Vidéo (lui)** | son visage animé en haut de l'écran, la bouche suivant vraiment le son joué |
| **Vidéo (toi)** | la caméra frontale envoie une image toutes les 2,5 s : il reconnaît qui c'est et voit la scène. Caméra refusée : rien ne se bloque |

Rien n'est mis en cache de ce qu'il dit : le service worker ne garde que la coquille de
l'appli.

## 5. À la place de Gemini

Android ne donne la place d'assistant par défaut qu'à une **appli installée**. Une page web
ne peut pas l'avoir — c'est comme ça, et le contourner demanderait de fabriquer un APK.

Le chemin honnête passe par une appli qu'Olivier a déjà : **Home Assistant**.

1. Valdar parle le langage d'Ollama (`GET /api/tags`, `POST /api/chat`) — ce n'est pas un
   vrai Ollama : c'est **Valdar entier** derrière, cœur, mémoire et outils compris.
2. Dans HA : intégration Ollama → l'adresse de Valdar → agent de conversation du pipeline
   Assist.
3. Android → Appli d'assistance numérique → Home Assistant.

Appui long sur power : c'est Valdar qui répond.

## 6. Piloter le téléphone

L'appli compagnon HA accepte des ordres. Valdar les a comme outils, réservés à Olivier :

| Outil | Ce qu'il fait | Demande un « oui » |
|---|---|---|
| `tel_notifier` | une notification | non |
| `tel_afficher` | ouvre une page web à l'écran | oui |
| `tel_parler` | parle à voix haute dans le téléphone | non |
| `musique` | sur une enceinte de la maison, sinon sur le tél | oui |
| `etat_maison` | lit un capteur, une lampe, une prise | non |
| `agir_maison` | allume, éteint — **seulement** ce qui est dans la liste | oui |

Les serrures et les alarmes ne sont pas dans la liste, et il ne peut pas les y mettre : la
liste est dans la configuration, qu'il n'a pas le droit de modifier.

Le jeton de Home Assistant vit dans `data/ha.token` (ou dans `VALDAR_HA_TOKEN`) :
`python -m valdar maison --jeton` le demande sans l'afficher.

## 7. Mise en route

```
tools\valdar_tel.bat          certificat + QR code, puis scanne avec le téléphone
python -m valdar maison --jeton   (pour Home Assistant, quand tu voudras)
```

Et sur le téléphone, une fois la page ouverte : « Ajouter à l'écran d'accueil ».
