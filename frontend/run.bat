@echo off
title Multimodal RAG - Frontend
cd /d "%~dp0"

echo ========================================================
echo  Starting React Frontend on http://127.0.0.1:5173
echo ========================================================
echo.

:: Clear any stale process currently holding port 5173
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo Freeing port 5173 from stale process PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

set "PATH=C:\Program Files\nodejs;%PATH%"
npm run dev -- --host 127.0.0.1

echo.
echo Frontend server stopped.
pause
