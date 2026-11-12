@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo === Corps numerique de Valdar : voir l'ecran et agir sur le PC ===
echo.
if not exist ".venv\Scripts\python.exe" (
  echo [X] Pas de .venv : lance d'abord tools\valdar_install.bat
  pause & exit /b 1
)
call .venv\Scripts\activate.bat
echo [1/2] Installation de ce qu'il faut (capture d'ecran, accessibilite Windows, connecteurs)
python -m pip install -q --upgrade pip
python -m pip install -q "pillow>=10.2" "mss>=9.0" "pywinauto>=0.6.9" "pywin32>=306" "mcp>=1.9"
if errorlevel 1 (
  echo [X] L'installation a echoue. Regarde le message au-dessus.
  pause & exit /b 1
)
echo [2/2] Verification
python -c "import mss, pywinauto, PIL, mcp; print('   capture, accessibilite, images, connecteurs : OK')"
if errorlevel 1 (
  echo [X] Quelque chose manque encore.
  pause & exit /b 1
)
echo.
echo === C'est bon. Valdar peut voir l'ecran et agir sur le PC. ===
echo     Son atelier : %%USERPROFILE%%\Documents\Valdar
echo     Pour couper ses mains sans couper ses yeux : corps.actions: false dans config\valdar.yaml
echo     Pour tout couper : corps.enabled: false
echo.
pause
