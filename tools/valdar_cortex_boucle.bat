@echo off
rem Boucle de reglage autonome du cortex : essaie, mesure, ajuste, et ne s'arrete que quand
rem le banc est au vert (ou apres la duree maximale). A lancer Valdar ferme, sans jeu ouvert.
rem Pour l'arreter proprement : creer un fichier vide data\cortex\STOP (ou fermer la fenetre).
chcp 65001 >nul
cd /d "%~dp0.."
if exist data\cortex\STOP del data\cortex\STOP
.venv\Scripts\python -X utf8 -u -m valdar cortex boucle --heures 12
if errorlevel 2 (
  echo Pas de vert : le cortex reste eteint. Etat dans data\cortex\boucle.json
  goto fin
)
if errorlevel 1 (
  echo ECHEC. Detail dans data\cortex\journal.txt
  goto fin
)
echo VERT : cortex active avec les reglages trouves. Relance Valdar.
:fin
echo.
pause
