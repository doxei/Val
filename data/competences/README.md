# Les compétences de Valdar

Un dossier par compétence, avec un `SKILL.md` dedans. Valdar ne garde en tête que le **nom**
et la **description** ; il lit le reste seulement quand la tâche le demande (outil
`competence`). Il peut en écrire lui-même : ce dossier est dans sa zone d'écriture.

```
data/competences/
  imprimer-une-piece/
    SKILL.md          <- la marche à suivre
    gabarit.gcode     <- les fichiers joints sont listés quand il lit la compétence
```

En-tête d'un `SKILL.md` :

```
---
name: imprimer-une-piece
description: Quand Olivier veut lancer une impression : vérifier, trancher, envoyer.
---

1. Vérifier que le plateau est libre (regarder_ecran sur l'écran de l'imprimante).
2. …
```

La `description` est ce qui décide s'il va lire la compétence : écris-la comme un
« quand », pas comme un titre.
