@echo off
REM In PowerShell do NOT paste this file. Run:  .\start_api.ps1   OR   .\start_api.bat
REM Optional port:  start_api.bat 8001
title Heart Disease API
cd /d "%~dp0"
set PORT=8000
if not "%~1"=="" set PORT=%~1
echo.
echo  Starting API: http://127.0.0.1:%PORT%
echo  Keep this window open while you use the site.
echo  Press Ctrl+C to stop the server.
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port %PORT%
if errorlevel 1 pause
