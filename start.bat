@echo off
title Multimodal RAG Launcher
setlocal enabledelayedexpansion

echo ========================================================
echo       Starting Multimodal RAG Application
echo ========================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "backend\app\main.py" (
    set "PROJECT_DIR=%SCRIPT_DIR%"
) else if exist "multimodal-rag\backend\app\main.py" (
    set "PROJECT_DIR=%SCRIPT_DIR%multimodal-rag\"
) else (
    echo [ERROR] Could not find backend directory.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_DIR%venv\Scripts\python.exe"
) else (
    echo [ERROR] Virtual environment not found at %PROJECT_DIR%venv
    pause
    exit /b 1
)

echo [1/3] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "Multimodal RAG - Backend (Port 8000)" cmd /k "cd /d "%PROJECT_DIR%backend" && "%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

echo [2/3] Launching Vite React Frontend on http://127.0.0.1:5173 ...
start "Multimodal RAG - Frontend (Port 5173)" cmd /k "cd /d "%PROJECT_DIR%frontend" && npm run dev -- --host 127.0.0.1"

echo [3/3] Waiting for servers to initialize...
timeout /t 3 /nobreak >nul

echo Opening browser at http://127.0.0.1:5173 ...
start http://127.0.0.1:5173

echo.
echo ========================================================
echo   Multimodal RAG is now running!
echo   - Frontend: http://127.0.0.1:5173
echo   - Backend:  http://127.0.0.1:8000
echo   - API Docs: http://127.0.0.1:8000/docs
echo ========================================================
echo.
