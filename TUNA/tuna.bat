@echo off
if "%1" == "--version" (
    echo 0.4.1
) else (
    python "%~dp0tuna.py" %*
)