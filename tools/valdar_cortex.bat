@echo off
rem Installe le cortex de Valdar : Gemma charge par Valdar lui-meme, sentiments injectes
rem dans les couches. A lancer Valdar ferme. Tout est note dans data\cortex\journal.txt.
chcp 65001 >nul
cd /d "%~dp0.."
if not exist tools\logs mkdir tools\logs
set LOG=tools\logs\cortex_install.log
echo === Cortex : installation === > %LOG%
echo.
echo [1/2] Moteur llama.cpp pour la carte graphique (llama-cpp-python, CUDA)...
.venv\Scripts\python -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 "llama-cpp-python>=0.3.36" "jinja2>=3.1" >> %LOG% 2>&1
if errorlevel 1 (
  echo ECHEC : le moteur ne s'installe pas. Detail dans %LOG%
  goto fin
)
.venv\Scripts\python -m pip install -q -e . >> %LOG% 2>&1
echo [2/2] Modele (7 Go), apprentissage des sentiments, essais, activation...
echo       Compte 20 a 40 minutes selon la connexion. Tu peux laisser tourner.
.venv\Scripts\python -X utf8 -u -m valdar cortex tout
if errorlevel 2 (
  echo Les essais ne sont pas assez nets : le cortex n'est PAS active. Valdar reste comme avant.
  goto fin
)
if errorlevel 1 (
  echo ECHEC. Detail dans data\cortex\journal.txt
  goto fin
)
echo.
echo Cortex pret. Relance Valdar.
:fin
echo.
pause
