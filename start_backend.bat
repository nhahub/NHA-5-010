@echo off
title MediVision-AI Backend Service
echo =====================================================================
echo           Starting MediVision-AI Multimodal Backend API
echo =====================================================================
echo.

cd /d "%~dp0backend"

if not exist ".env" if not exist "..\\.env" (
    echo [INFO] No .env file found. To enable Gemini AI Chat, create backend\.env with GEMINI_API_KEY=your_key
)

echo [INFO] Once started, access the application at:
echo        - Web App:   http://localhost:8000/app/
echo        - API Docs:  http://localhost:8000/docs
echo        - Local file: frontend\index.html
echo.

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Launching with uv on http://0.0.0.0:8000 ...
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
) else if exist ".venv\Scripts\python.exe" (
    echo [INFO] Launching with local virtualenv on http://0.0.0.0:8000 ...
    .venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
) else (
    echo [INFO] Launching with python on http://0.0.0.0:8000 ...
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
)

pause
