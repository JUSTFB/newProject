@echo off
setlocal
echo ==================================================
echo      Checking System Requirements
echo ==================================================
echo.

echo [1/4] Checking Node.js...
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto node_fail
echo    V Node.js is installed.
node -v
goto check_python

:node_fail
echo    X Node.js is NOT installed or not in PATH.

:check_python
echo.
echo [2/4] Checking Python...
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 goto py_found
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 goto python_found
goto python_fail

:py_found
echo    V Python is installed (command: py).
py --version
goto check_ffmpeg

:python_found
echo    V Python is installed (command: python).
python --version
goto check_ffmpeg

:python_fail
echo    X Python is NOT installed or not in PATH.

:check_ffmpeg
echo.
echo [3/4] Checking FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto ffmpeg_fail
echo    V FFmpeg is installed.
ffmpeg -version | findstr "ffmpeg version"
goto check_pip

:ffmpeg_fail
echo    X FFmpeg is NOT installed or not in PATH.

:check_pip
echo.
echo [4/4] Checking PIP...
where pip >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto pip_fail
echo    V PIP is installed.
pip --version
goto end

:pip_fail
echo    X PIP is NOT installed or not in PATH.

:end
echo.
echo ==================================================
echo      End of Check
echo ==================================================
pause
