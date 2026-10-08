@echo off
rem Nouvelle voix de Valdar depuis ElevenLabs : extraits, reference XTTS, avant/apres.
rem La cle doit etre dans la variable d'environnement ELEVENLABS_API_KEY (jamais dans un fichier).
rem Ferme Valdar avant (la carte graphique doit etre libre).
cd /d "%~dp0.."
.venv\Scripts\python -m valdar voix-nouvelle %*
if exist data\voix\comparaison start "" data\voix\comparaison
pause
