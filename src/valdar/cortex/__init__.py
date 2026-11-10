"""Le cortex : le modèle de langage de Valdar, avec ses sentiments injectés dans les couches.

Le cœur (neuromodulateurs, drives, humeur, lien avec la personne) ne se contente plus d'être
*décrit* au modèle dans le prompt : il déplace directement ses activations internes, couche
par couche, le long de directions apprises (valence, activation, puissance, attachement,
peur…). C'est la technique du « Golden Gate Claude » d'Anthropic, pilotée ici par un état
affectif qui vit en continu.

- `affect`  : l'état du cœur et de la relation → un coefficient par sentiment (la psyché).
- `vectors` : les directions dans le modèle et leur combinaison.
- `scenarios` : les situations contrastées qui servent à trouver ces directions.
- `engine`  : le modèle chargé, avec l'injection (moteur PyTorch).
- `server`  : le cortex en service, parlant le même protocole qu'Ollama.
"""
