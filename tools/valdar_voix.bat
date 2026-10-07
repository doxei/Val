@echo off
rem Donne a Valdar la voix de RAUB : installe XTTS (une seule fois), reprend la voix, la teste.
cd /d "%~dp0.."
if not exist tools\logs mkdir tools\logs
if not exist .venv\Scripts\python.exe (
  echo Environnement Python introuvable : .venv
  pause
  exit /b 1
)
.venv\Scripts\python -c "import TTS, torch, sounddevice, scipy; assert torch.cuda.is_available()" >nul 2>&1
if not errorlevel 1 goto installe
echo Installation de la voix : torch pour la carte graphique puis XTTS.
echo Gros telechargement, environ 3 Go. Journal : tools\logs\voice_install.log
.venv\Scripts\python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 > tools\logs\voice_install.log 2>&1
if errorlevel 1 goto echec
.venv\Scripts\python -m pip install -e .[system,voice] >> tools\logs\voice_install.log 2>&1
if errorlevel 1 goto echec
:installe
.venv\Scripts\python -m valdar import-raub
.venv\Scripts\python -m valdar voix %*
pause
exit /b 0
:echec
echo Echec de l'installation, voir tools\logs\voice_install.log
pause
exit /b 1
