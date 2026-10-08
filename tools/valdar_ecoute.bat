@echo off
rem Donne des oreilles a Valdar : installe l'ecoute (une fois) puis lance la conversation.
rem Rien n'est enregistre : le micro reste en memoire vive, seul ce qui suit "Valdar" est transcrit.
cd /d "%~dp0.."
if not exist tools\logs mkdir tools\logs
if not exist .venv\Scripts\python.exe (
  echo Environnement Python introuvable : .venv
  pause
  exit /b 1
)
.venv\Scripts\python -c "import faster_whisper, silero_vad, sounddevice" >nul 2>&1
if not errorlevel 1 goto installe
echo Installation de l'ecoute : faster-whisper et Silero. Journal : tools\logs\ears_install.log
.venv\Scripts\python -m pip install -e .[system,ears] -c tools\contraintes.txt > tools\logs\ears_install.log 2>&1
if errorlevel 1 goto echec
:installe
call tools\valdar_chat.bat %*
exit /b 0
:echec
echo Echec de l'installation, voir tools\logs\ears_install.log
pause
exit /b 1
