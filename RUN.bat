@echo off
REM Double-click this file to launch the BODHI dashboard in your browser.
cd /d "%~dp0"
echo.
echo  ============================================
echo    BODHI Dashboard
echo    Opening http://127.0.0.1:5000/ in browser
echo  ============================================
echo.
start "" http://127.0.0.1:5000/
python chat_ui.py
pause
