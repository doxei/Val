@echo off
chcp 65001 >nul
cd /d "%~dp0.."
title Valdar au telephone
echo === Valdar sur ton telephone : ecrit, vocal, video ===
echo.
if not exist ".venv\Scripts\python.exe" (
  echo [X] Pas de .venv : lance d'abord tools\valdar_install.bat
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
echo [1/2] Ce qu'il faut (certificat https + QR code)
python -m pip install -q "cryptography>=42.0" "qrcode>=7.4"
if errorlevel 1 (
  echo [X] L'installation a echoue. Regarde le message au-dessus.
  pause & exit /b 1
)
echo.
echo [2/2] Appairage
echo.
python -m valdar tel
echo.
echo Rappel : Valdar doit tourner (tools\valdar_demarrer.bat) pour que la page reponde.
echo.
pause
