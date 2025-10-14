@echo off
REM Activation script for Face Emoji Matcher virtual environment (Windows)

echo 🎭 Activating Face Emoji Matcher virtual environment...

if exist "venv" (
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated!
    echo.
    echo To run the application:
    echo   Web interface: python app.py
    echo   Command line:  python simple_demo.py
    echo.
    echo To deactivate: deactivate
) else (
    echo ❌ Virtual environment not found!
    echo Run 'python setup.py' first to create it.
)
