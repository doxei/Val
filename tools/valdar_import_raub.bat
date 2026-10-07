@echo off
rem Reprend les donnees de RAUB (lecture seule cote RAUB). Ajoute --force pour refaire.
cd /d "%~dp0.."
.venv\Scripts\python -c "import httpx" >nul 2>&1 || .venv\Scripts\python -m pip install -e .[system]
.venv\Scripts\python -m valdar import-raub %*
pause
