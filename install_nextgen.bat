@echo off
echo ============================================================
echo DubAI Studio - Next-Gen Pipeline Installer
echo ============================================================
echo This script installs the heavy ML frameworks for RVC (Voice) 
echo and LatentSync (Lip Sync), plus GFPGAN (Face Restore).
echo.
echo Requirements: Python 3.10+, NVIDIA GPU (8GB+ VRAM), Git
echo ============================================================
pause

echo.
echo [1/4] Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [2/4] Installing RVC (Voice Conversion) and GFPGAN...
echo Temporarily downgrading pip to fix omegaconf metadata bug...
python -m pip install "pip<24.1"
pip install rvc-python gfpgan insightface ffmpeg-python
python -m pip install --upgrade pip

echo.
echo [3/4] Cloning LatentSync Repository...
if not exist "latentsync" (
    git clone https://github.com/bytedance/LatentSync.git latentsync
) else (
    echo LatentSync folder already exists. Skipping clone.
)

echo.
echo [4/4] Installing LatentSync Dependencies...
cd latentsync
pip install -r requirements.txt
cd ..

echo.
echo ============================================================
echo Setup Complete! 
echo You can now run your DubAI backend and it will automatically
echo detect and use RVC + LatentSync for cinematic video quality!
echo ============================================================
pause
