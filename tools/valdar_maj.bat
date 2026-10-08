@echo off
rem Met Valdar a jour en un double-clic : code (git pull), dependances, Ollama.
cd /d "%~dp0"
if exist tools\valdar_chat.bat goto racine
cd ..
:racine
echo === Code de Valdar (GitHub doxei/Val, branche main) ===
git checkout main
if errorlevel 1 (
  echo Impossible de passer sur main : modifications locales ? Lance "git status".
  pause
  exit /b 1
)
git pull origin main
if errorlevel 1 (
  echo git pull a echoue : modifications locales ? Lance "git status" pour voir.
  pause
  exit /b 1
)
echo.
echo === Dependances Python ===
.venv\Scripts\python -m pip install -q -e .[system,ears]
echo.
echo === Ollama ===
ollama --version
echo Il faut Ollama 0.33.3 ou plus pour que Gemma entende.
where winget >nul 2>&1
if errorlevel 1 (
  echo winget absent : mets Ollama a jour depuis https://ollama.com/download
) else (
  winget upgrade --id Ollama.Ollama -e
)
ollama --version
echo.
echo === Tests ===
.venv\Scripts\python -m pytest -q
pause
