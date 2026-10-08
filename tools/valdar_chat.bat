@echo off
rem Parler a Valdar au clavier. Demarre Ollama si besoin, installe ce qui manque.
rem Si la voix est installee (tools\valdar_voix.bat), il repond aussi a voix haute.
cd /d "%~dp0.."
if not exist tools\logs mkdir tools\logs
if not exist .venv\Scripts\python.exe (
  echo Environnement Python introuvable : .venv
  pause
  exit /b 1
)
.venv\Scripts\python -c "import httpx" >nul 2>&1
if errorlevel 1 (
  echo Installation des dependances de Valdar...
  .venv\Scripts\python -m pip install -e .[system] > tools\logs\pip_install.log 2>&1
  if errorlevel 1 (
    echo Echec de l'installation, voir tools\logs\pip_install.log
    pause
    exit /b 1
  )
)
curl.exe -s http://127.0.0.1:11434/api/version >nul 2>&1
if errorlevel 1 (
  echo Demarrage d'Ollama...
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
  ) else (
    start "Ollama" /min ollama serve
  )
  timeout /t 8 >nul
)
if not exist data\imports.json (
  echo Premiere fois : je reprends les donnees de l'ancienne installation...
  .venv\Scripts\python -m valdar import-ancien
)
set VOIX=
if exist data\models\xtts-v2\model.pth (
  .venv\Scripts\python -c "import TTS" >nul 2>&1
  if not errorlevel 1 set VOIX=--voix
)
set ECOUTE=
.venv\Scripts\python -c "import faster_whisper, silero_vad, sounddevice" >nul 2>&1
if not errorlevel 1 set ECOUTE=--ecoute
.venv\Scripts\python -m valdar chat %VOIX% %ECOUTE% %*
pause
