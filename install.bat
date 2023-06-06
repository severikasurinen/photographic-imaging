@echo off
Pushd "%~dp0"
python -m pip install --upgrade pip
python -m pip install --user virtualenv
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt
PAUSE
