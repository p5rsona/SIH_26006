#!/usr/bin/env bash
# Starts the API in the background and then the dashboard.
cd "$(dirname "$0")"
python -m uvicorn api:app --port 8000 &
API_PID=$!
trap "kill $API_PID" EXIT
sleep 4
python -m streamlit run dashboard.py
