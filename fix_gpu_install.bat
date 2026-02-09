@echo off
echo ===========================================
echo       Fixing GPU Installation (Torch)
echo ===========================================
echo.

echo [1/4] Killing potential locking processes...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pip.exe /T 2>nul

echo.
echo [2/4] Purging pip cache to remove corrupt downloads...
pip cache purge

echo.
echo [3/4] Installing PyTorch with CUDA 12.1...
echo       (Using --no-cache-dir and timeout flags)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir --timeout 1000 --force-reinstall

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo    X Installation FAILED.
    echo      Please try restarting your computer if this persists.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Verifying installation...
python check_gpu.py

echo.
echo ===========================================
echo       GPU Setup Fixed & Complete!
echo ===========================================
pause
