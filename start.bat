@echo off
title EDU Agent - AI Study Assistant
echo.
echo  ========================================
echo     EDU Agent - AI Study Assistant
echo  ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed!
    echo  Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Create venv if not exists
if not exist "venv_win\Scripts\activate.bat" (
    echo  [1/3] Creating virtual environment...
    python -m venv venv_win
    echo  Done.
) else (
    echo  [1/3] Virtual environment found.
)

:: Activate venv
call venv_win\Scripts\activate.bat

:: Install dependencies
echo  [2/3] Installing dependencies (first time may take a few minutes)...
pip install -r requirements.txt -q

:: Check .env
if not exist ".env" (
    echo.
    echo  ============================================
    echo   WARNING: No .env file found!
    echo   Creating one now...
    echo  ============================================
    echo GROQ_API_KEY=your_groq_api_key_here> .env
    echo GROQ_REASONING_EFFORT=medium>> .env
    echo GROQ_MAX_COMPLETION_TOKENS=4096>> .env
    echo.
    echo  Please edit .env and add your Groq API key.
    echo  Get a free key from: https://console.groq.com
    echo.
    notepad .env
    pause
)

:: Start server
echo  [3/3] Starting EDU Agent...
echo.
echo  ==========================================
echo   Open in your browser:
echo   http://localhost:8000
echo  ==========================================
echo.
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

pause
