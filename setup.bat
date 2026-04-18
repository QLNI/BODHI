@echo off
REM ============================================================
REM  BODHI Setup — Windows
REM  Run this once after cloning.
REM ============================================================
echo.
echo  ██████╗  ██████╗ ██████╗ ██╗  ██╗██╗
echo  ██╔══██╗██╔═══██╗██╔══██╗██║  ██║██║
echo  ██████╔╝██║   ██║██║  ██║███████║██║
echo  ██╔══██╗██║   ██║██║  ██║██╔══██║██║
echo  ██████╔╝╚██████╔╝██████╔╝██║  ██║██║
echo  ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝
echo  BODHI Setup for Windows
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install from https://python.org/downloads
    echo         Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

REM Install pip packages
echo [1/3] Installing Python packages...
pip install numpy pillow cryptography torch sentencepiece --quiet
IF ERRORLEVEL 1 (
    echo [ERROR] pip install failed. Try running as Administrator.
    pause
    exit /b 1
)

REM Check if fingerprint data is present (LFS stub = tiny file)
echo [2/3] Checking data files...
FOR %%F IN (data\fingerprints_img.npz) DO (
    IF %%~zF LSS 1000000 (
        echo  Data files not downloaded yet. Running download_data.py...
        python download_data.py
    ) ELSE (
        echo  Data files OK.
    )
)

echo [3/3] All done!
echo.
echo  To start BODHI:
echo    python bodhi.py
echo.
echo  To chat with BODHI:
echo    python bodhi.py
echo    ^> then type anything
echo.
pause
