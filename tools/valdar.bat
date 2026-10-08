@echo off
rem Valdar, en entier : coeur, voix, oreilles, camera et interface sur le projecteur.
rem Double-clic (ou l'icone du bureau, creee au premier lancement).
cd /d "%~dp0.."
title Valdar
if not exist tools\logs mkdir tools\logs
if not exist .venv\Scripts\python.exe (
  echo Environnement Python introuvable : .venv
  pause
  exit /b 1
)

rem --- derniere version du code (sans bloquer si ca echoue)
where git >nul 2>&1
if not errorlevel 1 (
  git checkout -q main >nul 2>&1
  git pull -q --ff-only origin main >nul 2>&1
)

rem --- icone sur le bureau (une seule fois ; le Bureau peut etre dans OneDrive)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $l=Join-Path $d 'Valdar.lnk'; if (-not (Test-Path $l)) { $s=(New-Object -ComObject WScript.Shell).CreateShortcut($l); $s.TargetPath='%~dp0valdar.bat'; $s.WorkingDirectory='%~dp0..'; $s.IconLocation='%~dp0valdar.ico'; $s.Description='Valdar'; $s.Save(); Write-Host 'Icone Valdar posee sur le bureau.' }"

rem --- dependances (installees une fois, journal dans tools\logs)
.venv\Scripts\python -c "import httpx, psutil" >nul 2>&1
if errorlevel 1 (
  echo Installation des dependances de base...
  .venv\Scripts\python -m pip install -e .[system] > tools\logs\pip_install.log 2>&1
)
.venv\Scripts\python -c "import faster_whisper, silero_vad, sounddevice" >nul 2>&1
if errorlevel 1 (
  echo Installation de l'ecoute : faster-whisper et Silero...
  .venv\Scripts\python -m pip install -e .[system,ears] -c tools\contraintes.txt > tools\logs\ears_install.log 2>&1
)
.venv\Scripts\python -c "import sherpa_onnx" >nul 2>&1
if errorlevel 1 (
  echo Installation de la reconnaissance des voix...
  .venv\Scripts\python -m pip install -e .[identite] -c tools\contraintes.txt > tools\logs\identite_install.log 2>&1
)
.venv\Scripts\python -c "import cv2, ultralytics" >nul 2>&1
if errorlevel 1 (
  echo Installation de la vision : visages, chien...
  .venv\Scripts\python -m pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 > tools\logs\vision_install.log 2>&1
  .venv\Scripts\python -m pip install -e .[vision] -c tools\contraintes.txt --extra-index-url https://download.pytorch.org/whl/cu126 >> tools\logs\vision_install.log 2>&1
)

rem --- la voix a besoin de torch 2.6 pour la carte graphique (une installation a pu le changer)
if exist data\models\xtts-v2\model.pth (
  .venv\Scripts\python -c "import torch, torchvision, torchaudio; assert torch.__version__.startswith('2.6.0') and torch.cuda.is_available()" >nul 2>&1
  if errorlevel 1 (
    echo Reparation de torch pour la carte graphique, quelques minutes...
    .venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 > tools\logs\torch_repair.log 2>&1
  )
)

rem --- Ollama
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

set OPTIONS=--interface
if exist data\models\xtts-v2\model.pth (
  .venv\Scripts\python -c "import TTS" >nul 2>&1
  if not errorlevel 1 set OPTIONS=%OPTIONS% --voix
)
.venv\Scripts\python -c "import faster_whisper, silero_vad, sounddevice" >nul 2>&1
if not errorlevel 1 set OPTIONS=%OPTIONS% --ecoute
.venv\Scripts\python -c "import cv2" >nul 2>&1
if not errorlevel 1 set OPTIONS=%OPTIONS% --camera
.venv\Scripts\python -m valdar chat %OPTIONS% %*
pause
