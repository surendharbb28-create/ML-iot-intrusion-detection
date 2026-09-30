@echo off
echo ============================================
echo   MLCS-HACK-26 - IoT IDS API Server
echo ============================================
cd /d "%~dp0"
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
pause
