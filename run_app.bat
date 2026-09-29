@echo off
cd /d C:\fish-app
call venv\Scripts\activate
python -m streamlit run app.py
pause
