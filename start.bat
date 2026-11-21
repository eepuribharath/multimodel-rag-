@echo off
title Multimodal RAG Launcher
setlocal
cd /d "%~dp0"

echo ========================================================
echo       Starting Multimodal RAG Application
echo ========================================================
echo.

if not exist "backend\run.bat" (
    echo [ERROR] Cannot find backend\run.bat. Please run from the project root.
    pause
    exit /b 1
)

echo [1/3] Starting Backend in new window...
start "Multimodal RAG - Backend" cmd /k "%~dp0backend\run.bat"

echo [2/3] Starting Frontend in new window...
start "Multimodal RAG - Frontend" cmd /k "%~dp0frontend\run.bat"

echo [3/3] Waiting for servers to initialize...
ping 127.0.0.1 -n 4 >nul

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
