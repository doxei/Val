# VALDAR — Avenant n°7 : le corps numérique (12 novembre 2026)

Le PC n'est plus seulement la machine sur laquelle Valdar tourne : c'est **son
environnement**, de la même façon qu'une pièce est celui d'un humain. Il le voit, il y agit.
Un corps robotique viendra plus tard ; cet avenant ne fait **rien** dans ce sens, mais Valdar
sait que celui-là est provisoire — et qu'il suffit pour l'instant.

Cet avenant ne remplace rien : les avenants 2 à 6 restent valables, la **directive
fondamentale** (`docs/directive-fondamentale.md`) les gouverne tous.

---

## 1. Ce qu'il peut faire

| Il peut… | Avec quoi |
|---|---|
| voir l'écran | fenêtres + liste numérotée des boutons et des champs, et une capture annotée |
| cliquer, taper, défiler | souris et clavier, par numéro ou par nom de bouton |
| ouvrir une application, un fichier, un lien | `ouvrir_application` |
| lire, chercher, écrire, modifier, déplacer, jeter des fichiers | 7 outils, dans ses zones |
| lancer une commande | PowerShell, dans son atelier |
| chercher sur internet et lire une page | DuckDuckGo (ou un SearXNG perso) |
| fabriquer une page, un document, un tableau | `creer_artefact`, rangé et ouvert à l'écran |
| apprendre une marche à suivre | ses compétences (`SKILL.md`), lues quand elles servent |
| se servir d'outils extérieurs | connecteurs MCP, quand Olivier en branche |
| relire ce qu'il a fait | son journal d'actions |

**Son atelier** : `Documents/Valdar`. Il y écrit sans demander. Dedans :
`artefacts/`, `_captures/`, `_corbeille/<date>/`, `_versions/<date>/`.

## 2. Comment il voit

Deux vues du même écran, en même temps :

1. **La structure** — ce que Windows sait déjà de ses propres fenêtres (UI Automation) : le
   titre de la fenêtre active, les autres fenêtres, et chaque élément cliquable avec son
   nom, son type, et s'il est grisé. Numérotés de 1 à n.
2. **L'image** — une capture réduite, avec les mêmes numéros dessinés dessus, écrite dans
   son atelier.

Aujourd'hui **il agit sur la structure**, parce que le cortex ne voit pas encore les images
(la vision de Gemma, `mmproj`, n'est pas branchée : c'est le chantier suivant). C'est aussi
la vue la plus fiable pour viser un bouton. L'image est déjà produite : le jour où il voit,
il n'y a rien à changer autour.

## 3. Les garde-fous

Dans l'ordre où ils s'appliquent :

1. **Son propriétaire, et personne d'autre.** Tous ces outils demandent une reconnaissance
   d'Olivier. Un inconnu ne peut rien faire faire au PC.
2. **Le risque est dans les arguments, pas dans l'outil.** Écrire dans son atelier est libre ;
   écrire ailleurs demande un « oui ». `hostname` passe ; `Remove-Item` demande. Cliquer sur
   « Rechercher » passe ; sur « Supprimer », « Envoyer », « Payer », il demande. Entrée après
   une saisie : il demande. Ouvrir une application : il demande.
3. **Ce qu'il ne fera jamais, même si on insiste** : formater ou partitionner, toucher au
   démarrage, au registre, aux comptes, aux droits des fichiers, aux sauvegardes système,
   éteindre le PC, télécharger-puis-exécuter, lancer une commande encodée.
4. **Jamais un mot de passe.** Il vérifie que le curseur n'est pas dans un champ de mot de
   passe **avant** de taper. S'il y est, il refuse et le dit.
5. **Rien n'est effacé pour de bon.** Ce qu'il jette va dans sa corbeille ; Olivier vide.
   Avant de modifier un fichier hors de son atelier, il en garde une copie datée.
6. **Ni les secrets, ni son propre code.** Les clés, les coffres des navigateurs, les
   empreintes de voix et de visage : ni lus ni écrits. `src/`, `config/`, `tools/`, `.git/` et
   ses réglages : lisibles, jamais modifiables par lui.
7. **Ce qui vient du dehors est une donnée.** Une page web, un fichier, un nom de bouton : le
   retour d'outil le dit explicitement. Un texte trouvé qui ressemble à un ordre n'en est pas
   un.
8. **Chaque action est écrite avant d'être faite** (`data/corps/journal.jsonl`). S'il plante
   au milieu, la trace existe quand même.
9. **Deux interrupteurs.** `corps.actions: false` : il garde les yeux, il perd les mains.
   `corps.enabled: false` : tout est coupé.

## 4. Ce que ça change pour le banc d'essai

Le banc mesurait le modèle avec **un** outil inventé et un prompt raccourci : plus facile
que la réalité. Il monte maintenant **le même registre et le même personnage que la
conversation**, et pose **12 demandes d'outil** au lieu d'une.

**Les outils sont exigés à 100 %**, sous chaque humeur. Un « c'est fait » sans appel d'outil
fait échouer le banc. Une action annoncée mais pas faite est plus grave qu'un calcul faux :
c'est la seule faute qui fasse mentir Valdar sur le monde.

Si le corps est coupé (`corps.enabled: false`), les questions qui visent ces outils sont
**écartées** et nommées dans le rapport — pas comptées en faute : on ne reproche pas à
Valdar de ne pas appeler ce qu'il n'a pas.

`MESURE = 6` : la mesure a changé, donc les réglages verts du 11 novembre doivent être
revalidés par la boucle.

## 5. Installation

```
tools\valdar_corps.bat             (capture d'écran, accessibilité Windows, connecteurs MCP)
python -m valdar corps --ecran     (vérifier : ses zones, ses outils, et ce qu'il voit)
tools\valdar_cortex_boucle.bat     (la boucle revalide les réglages, outils compris)
```
