@echo off
cd /d C:\Users\Claydwin\Documents\String Art Générator
start cmd /k "python app.py"
timeout /t 1 >nul
start "" chrome.exe "http://127.0.0.1:5000"