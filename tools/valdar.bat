@echo off
rem Valdar, en entier : coeur, voix, oreilles, camera et interface sur le projecteur.
rem Double-clic (ou l'icone du bureau, creee au premier lancement).
rem Ce fichier ne fait que mettre le code a jour puis passe la main : un fichier .bat
rem modifie pendant qu'il tourne se lit de travers, donc le reste est dans valdar_demarrer.bat.
cd /d "%~dp0.."
where git >nul 2>&1
if not errorlevel 1 (
  echo Mise a jour de Valdar...
  git checkout -q main >nul 2>&1
  git pull -q --ff-only origin main >nul 2>&1
)
tools\valdar_demarrer.bat %*
