<#
.SYNOPSIS
    Starts the Multimodal RAG backend and frontend services.
#>
[CmdletBinding()]
param()

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (Test-Path "$ScriptDir\backend\app\main.py") {
    $ProjectDir = $ScriptDir
} elseif (Test-Path "$ScriptDir\multimodal-rag\backend\app\main.py") {
    $ProjectDir = Join-Path $ScriptDir "multimodal-rag"
} else {
    Write-Error "Could not locate project directory with backend/app/main.py."
    exit 1
}

$PythonExe = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python virtual environment not found at $PythonExe"
    exit 1
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Starting Multimodal RAG Application              " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure Node / npm is in PATH
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    if (Test-Path "C:\Program Files\nodejs") {
        $env:PATH = "C:\Program Files\nodejs;$env:PATH"
    }
}

Write-Host "[1/3] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$ProjectDir\backend`" && `"$PythonExe`" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

Write-Host "[2/3] Starting React Frontend on http://127.0.0.1:5173 ..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/k cd /d `"$ProjectDir\frontend`" && npm run dev -- --host 127.0.0.1"

Write-Host "[3/3] Waiting for servers to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Opening http://127.0.0.1:5173 in default browser..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:5173"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Multimodal RAG is running!" -ForegroundColor Green
Write-Host "  - Frontend: http://127.0.0.1:5173" -ForegroundColor White
Write-Host "  - Backend:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - API Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Green
Write-Host "Close the opened terminal windows to stop the servers." -ForegroundColor Gray
