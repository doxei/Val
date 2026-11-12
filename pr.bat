@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Valdar - merger la PR
echo ===========================================================
echo   Merger la pull request de Valdar dans main
echo ===========================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
  echo [X] git n'est pas installe, ou pas dans le PATH.
  echo     Git pour Windows : https://git-scm.com/download/win
  goto :fin
)
git rev-parse --git-dir >nul 2>nul
if errorlevel 1 (
  echo [X] Ce dossier n'est pas un depot git : %CD%
  echo     Ce fichier doit rester a la racine du depot Valdar.
  goto :fin
)

echo [1/5] Je regarde ou en est GitHub...
git fetch origin --prune
if errorlevel 1 (
  echo [X] Je n'arrive pas a joindre GitHub. Verifie la connexion.
  goto :fin
)

rem --- Quelle branche ? celle donnee en argument, sinon la derniere branche claude/... poussee
set "BRANCHE=%~1"
if not "!BRANCHE!"=="" if "!BRANCHE:/=!"=="!BRANCHE!" set "BRANCHE=claude/!BRANCHE!"
if "!BRANCHE!"=="" (
  for /f "tokens=1" %%b in ('git branch -r --sort=-committerdate --list "origin/claude/*"') do (
    if not defined BRANCHE set "BRANCHE=%%b"
  )
  if defined BRANCHE set "BRANCHE=!BRANCHE:origin/=!"
)
if "!BRANCHE!"=="" (
  echo [X] Aucune branche « claude/... » sur GitHub : il n'y a rien a merger.
  goto :fin
)
git rev-parse --verify --quiet "origin/!BRANCHE!" >nul
if errorlevel 1 (
  echo [X] La branche « !BRANCHE! » n'existe pas sur GitHub. Les branches disponibles :
  git branch -r --list "origin/claude/*"
  echo.
  echo     Relance en donnant son nom : pr.bat nom-de-la-branche
  goto :fin
)

echo.
echo [2/5] Branche a merger : !BRANCHE!
git log --oneline -3 "origin/!BRANCHE!"
for /f %%n in ('git rev-list --count "origin/main..origin/!BRANCHE!"') do set "NOUVEAUX=%%n"
echo.
if "!NOUVEAUX!"=="0" (
  echo [i] Elle est deja dans main : il n'y a rien a merger.
  echo     Je mets juste ton main a jour.
  goto :maj
)
echo     !NOUVEAUX! commit^(s^) a faire passer dans main.
echo.
set "REPONSE="
set /p "REPONSE=Je merge dans main et j'envoie sur GitHub ? (oui / non) : "
if /i not "!REPONSE!"=="oui" (
  echo.
  echo Annule. Je n'ai touche a rien.
  goto :fin
)

echo.
echo [3/5] Je verifie que ton dossier est propre...
set "SALE="
for /f "delims=" %%s in ('git status --porcelain') do set "SALE=1"
if defined SALE (
  echo [i] Tu as des modifications non enregistrees ici :
  git status --short
  echo.
  echo     Souvent ce ne sont que des fins de ligne Windows. Je peux les mettre
  echo     de cote ^(git stash^) : rien n'est perdu, « git stash pop » les rend.
  set "GARDE="
  set /p "GARDE=Je les mets de cote et je continue ? (oui / non) : "
  if /i not "!GARDE!"=="oui" (
    echo.
    echo Annule. Je n'ai touche a rien.
    goto :fin
  )
  git stash push -u -m "mis de cote par pr.bat"
  if errorlevel 1 (
    echo [X] Je n'ai pas pu les mettre de cote. Dis-le moi dans le chat.
    goto :fin
  )
  echo     Mis de cote. Pour les recuperer plus tard : git stash pop
)
git checkout main
if errorlevel 1 (
  echo [X] Je n'ai pas pu passer sur main.
  goto :fin
)
git merge --ff-only origin/main
if errorlevel 1 (
  echo [X] Ton main local a pris un chemin different de GitHub.
  echo     Dis-le moi dans le chat, je regarde avec toi.
  goto :fin
)

echo.
echo [4/5] Merge de !BRANCHE! dans main...
git merge --no-ff "origin/!BRANCHE!" -m "Merge de !BRANCHE!"
if errorlevel 1 (
  echo [X] Conflit pendant le merge : je ne decide pas tout seul.
  echo     « git merge --abort » pour revenir en arriere, puis dis-le moi dans le chat.
  goto :fin
)

echo.
echo [5/5] J'envoie sur GitHub ^(ca ferme la PR^)...
git push origin main
if errorlevel 1 (
  echo [X] Le push a echoue. Dis-le moi dans le chat.
  goto :fin
)
echo.
echo === C'est merge, la PR est fermee, ton PC est a jour. ===
goto :suite

:maj
set "SALE="
for /f "delims=" %%s in ('git status --porcelain') do set "SALE=1"
if defined SALE (
  echo [i] Tu as des modifications non enregistrees : je ne touche a rien.
  goto :fin
)
git checkout main >nul 2>nul
git merge --ff-only origin/main
echo === Ton main est a jour. ===

:suite
echo.
echo Ensuite, dans l'ordre :
echo   tools\valdar_corps.bat            ses yeux et ses mains ^(une seule fois^)
echo   python -m valdar corps --ecran    verifier ce qu'il voit
echo   tools\valdar_cortex_boucle.bat    la boucle revalide les reglages
echo.

:fin
echo.
pause
endlocal
