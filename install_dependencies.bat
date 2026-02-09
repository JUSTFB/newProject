@echo off
echo ==================================================
echo      Installing Project Dependencies
echo ==================================================
echo.

echo [1/2] Installing Backend Dependencies (Node.js)...
cd backend
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo    X Failed to install backend dependencies.
    pause
    exit /b %ERRORLEVEL%
)
echo    V Backend dependencies installed.
cd ..
echo.

echo [2/2] Installing Python Dependencies...
cd python
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo    X Failed to install Python dependencies.
    echo      Make sure Python and Pip are installed and in your PATH.
    pause
    exit /b %ERRORLEVEL%
)
echo    V Python dependencies installed.
cd ..

echo.
echo ==================================================
echo      Installation Complete!
echo ==================================================
pause
