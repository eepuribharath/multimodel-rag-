@echo off
title Multimodal RAG - Backend
cd /d "%~dp0"

echo ========================================================
echo  Starting FastAPI Backend on http://127.0.0.1:8000
echo ========================================================
echo.

:: Clear any stale process currently holding port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Freeing port 8000 from stale process PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
) else (
    echo [ERROR] Virtual environment not found!
    echo Looked for: ..\venv\Scripts\python.exe
)

echo.
echo Backend server stopped.
pause
