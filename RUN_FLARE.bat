@echo off
title FLARE Municipal Engine
cd /d "c:\Users\Dell\OneDrive\Desktop\New folder\antigravity\legacy_civic_dedupe"
echo ========================================================
echo    Starting FLARE Municipal Engine Dashboard
echo ========================================================
echo.
echo Localhost URL: http://localhost:8501
echo NOTE: Keep this terminal window open while using FLARE!
echo.
py -m streamlit run app.py
if %errorlevel% neq 0 (
    python -m streamlit run app.py
)
pause
