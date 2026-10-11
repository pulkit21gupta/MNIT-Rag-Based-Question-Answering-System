@echo off
REM MNIT RAG Assistant — Launcher
REM Uses Python 3.14 where all packages are installed

set PYTHON="C:\Users\Dell\AppData\Local\Python\pythoncore-3.14-64\python.exe"
set SCRIPT_DIR=%~dp0

echo.
echo  MNIT RAG Assistant
echo  ==================
echo  Chat UI  : http://localhost:5000
echo  API Docs : http://localhost:5000/docs
echo.

cd /d "%SCRIPT_DIR%"
%PYTHON% app.py
pause
