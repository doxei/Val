# Valdar

IA locale (Gemma 4 via Ollama) avec un **cœur émotionnel continu** : neuromodulateurs, organes
virtuels en boucle fermée avec le cœur, humeur, besoins, initiative. Valdar remplace RAUB :
atelier, imprimante 3D (Klipper/Moonraker), rappels, mémoire, et la voix de RAUB (XTTS v2).

- `DECISIONS.md` : journal des choix techniques.
- `docs/` : avenants au cahier des charges (2 : héritage RAUB, 3 : dynamique temporelle et voix,
  4 : boucle fermée et socle de valeurs), audit de PrintOS, base de connaissances impression.

## Démarrer (Windows)
- `tools\valdar_chat.bat` : parler à Valdar au clavier (démarre Ollama, importe RAUB).
- `tools\valdar_voix.bat` : installer et tester la voix.
- `python -m valdar --help` : toutes les commandes (`chat`, `voix`, `import-raub`,
  `import-vecu`, `heart`, `status`, `sim`, `check`).

## Tests
`pip install -e .[dev]` puis `pytest` et `ruff check src tests`.
