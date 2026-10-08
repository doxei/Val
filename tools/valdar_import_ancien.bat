@echo off
rem Reprend les donnees de l'ancienne installation (lecture seule de ce cote-la). Ajoute --force pour refaire.
cd /d "%~dp0.."
.venv\Scripts\python -c "import httpx" >nul 2>&1 || .venv\Scripts\python -m pip install -e .[system]
.venv\Scripts\python -m valdar import-ancien %*
pause
