@echo off
setlocal

cd /d "%~dp0"

set VENV_DIR=.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found on this system.
    exit /b 1
)

set PYTHON_CMD=py -3.11

if exist "%VENV_DIR%" (
    "%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Recreating virtual environment with Python 3.11...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment with Python 3.11...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Installing project dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)

if not exist "models\saved\metrics.json" (
    echo Training model files are missing. Training models now...
    python scripts\train_models.py
    if errorlevel 1 (
        echo Model training failed.
        exit /b 1
    )
)

echo Starting Smart Energy Prediction app...
python backend\app.py

endlocal
