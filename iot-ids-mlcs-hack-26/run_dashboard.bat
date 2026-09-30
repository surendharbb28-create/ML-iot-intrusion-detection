@echo off
echo ============================================
echo   MLCS-HACK-26 - IoT IDS Dashboard
echo ============================================
cd /d "%~dp0"
streamlit run app/dashboard.py
pause
