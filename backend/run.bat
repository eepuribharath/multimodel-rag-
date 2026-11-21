@echo off
title Multimodal RAG - Backend
cd /d "%~dp0"

echo ========================================================
echo  Starting FastAPI Backend on http://127.0.0.1:8000
echo ========================================================
echo.

if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) else (
    echo [ERROR] Virtual environment not found!
    echo Looked for: ..\venv\Scripts\python.exe
)

echo.
echo Backend server stopped.
pause
