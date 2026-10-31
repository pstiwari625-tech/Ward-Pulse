@echo off
echo Starting Ward Pulse Civic Demand Intelligence Dashboard...
cd /d "%~dp0"
if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)
pause
