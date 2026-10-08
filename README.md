# Valdar

IA locale (Gemma 4 via Ollama) avec un **cœur émotionnel continu** : neuromodulateurs, organes
virtuels en boucle fermée avec le cœur, humeur, besoins, initiative. Valdar remplace l'ancien
assistant d'atelier : atelier, imprimante 3D (Klipper/Moonraker), rappels, mémoire, et sa voix
d'origine (XTTS v2). Interface : thème Azur (nuit bleue, lueur cyan, accents dorés).

- `DECISIONS.md` : journal des choix techniques.
- `docs/` : avenants au cahier des charges (2 : héritage de l'ancien assistant, 3 : dynamique
  temporelle et voix, 4 : boucle fermée et socle de valeurs), audit de PrintOS, base de connaissances impression.
  Le vécu importé, la vigie d'impression, les connaissances et la pensée de fond sont décrits
  dans `DECISIONS.md`.

## Démarrer (Windows)
- `tools\valdar_chat.bat` : parler à Valdar au clavier (démarre Ollama, reprend les données de
  l'ancienne installation).
- `tools\valdar_import_ancien.bat` : refaire cet import à la main (`--force` pour tout reprendre ;
  dossier source : `migration.dossier_ancien` dans `config/valdar.yaml`).
- `tools\valdar_voix.bat` : installer et tester la voix.
- `python -m valdar --help` : toutes les commandes (`chat`, `voix`, `import-ancien`,
  `import-vecu`, `vigie`, `heart`, `status`, `sim`, `check`).
- `python -m valdar vigie` : état de la vigie d'impression ; `--reussie` / `--ratee [--minutes N]`
  pour étiqueter la dernière impression ; `--debloquer` / `--verrouiller` pour l'autonomie.

## Tests
`pip install -e .[dev]` puis `pytest` et `ruff check src tests`.
