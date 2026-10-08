@echo off
rem Chronometre un tour complet de Valdar (clavier, voix, oreilles, sens).
rem Ferme Valdar avant : la carte graphique doit etre libre pour mesurer juste.
cd /d "%~dp0.."
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
.venv\Scripts\python -m valdar chrono %*
pause
