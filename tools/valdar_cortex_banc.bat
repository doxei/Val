@echo off
rem Banc d'essai du cortex : mesures reproductibles (separation, coherence, comportement,
rem interactions, stabilite, comparatif). A lancer Valdar ferme. Rapport : data\cortex\banc.md
chcp 65001 >nul
cd /d "%~dp0.."
.venv\Scripts\python -X utf8 -u -m valdar cortex banc
if errorlevel 2 (
  echo Le banc ne passe pas : le cortex reste eteint. Detail dans data\cortex\banc.md
  goto fin
)
if errorlevel 1 (
  echo ECHEC. Detail dans data\cortex\journal.txt
  goto fin
)
echo Banc passe : cortex active. Relance Valdar.
:fin
echo.
pause
