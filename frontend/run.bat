@echo off
title Multimodal RAG - Frontend
cd /d "%~dp0"

echo ========================================================
echo  Starting React Frontend on http://127.0.0.1:5173
echo ========================================================
echo.

set "PATH=C:\Program Files\nodejs;%PATH%"
npm run dev -- --host 127.0.0.1

echo.
echo Frontend server stopped.
pause
