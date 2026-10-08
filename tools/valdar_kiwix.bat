@echo off
rem Bibliotheques hors ligne de Valdar (Wikipedia, Vikidia, medecine, bricolage...).
rem Demande avant de telecharger ; reprend la ou ca s'etait arrete.
cd /d "%~dp0.."
.venv\Scripts\python -m valdar kiwix
pause
