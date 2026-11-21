<#
.SYNOPSIS
    Starts the Multimodal RAG backend and frontend services.
#>
[CmdletBinding()]
param()

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not (Test-Path "$ScriptDir\backend\run.bat")) {
    Write-Error "Cannot find backend\run.bat. Please ensure you are running from the project root."
    exit 1
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Starting Multimodal RAG Application              " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Starting Backend in new window..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$ScriptDir\backend\run.bat`""

Write-Host "[2/3] Starting Frontend in new window..." -ForegroundColor Green
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$ScriptDir\frontend\run.bat`""

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
