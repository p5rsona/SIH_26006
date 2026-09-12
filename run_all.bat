@echo off
REM Starts the API (new window) and then the dashboard.
REM API docs: http://localhost:8000/docs   Dashboard: http://localhost:8501
cd /d "%~dp0"
start "SIH API" cmd /k python -m uvicorn api:app --port 8000 --reload
timeout /t 4 >nul
python -m streamlit run dashboard.py
