@echo off
echo ==================================================
echo      Setting up GPU Support (NVIDIA CUDA)
echo ==================================================
echo.

cd python

echo [1/3] Uninstalling existing CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo [2/3] Installing PyTorch with CUDA 12.1 support...
echo      This may take a while (approx 2.5GB download).
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
if %ERRORLEVEL% NEQ 0 (
    echo    X Failed to install CUDA PyTorch.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Verifying installation...
python ..\check_gpu.py

echo.
echo ==================================================
echo      GPU Setup Complete!
echo ==================================================
pause
