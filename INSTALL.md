# Manual Installation Commands

## 0. Check Setup
First, check what is already installed on your PC.
We have created a script for this. Double-click `check_env.bat` in the project folder, or run:
```bash
check_env.bat
```

If you prefer manual commands:
```bash
node -v
npm -v
python --version
ffmpeg -version
```

## 1. Automatic Installation
We have created a script to install everything for you.
Double-click `install_dependencies.bat` in the project folder, or run:
```bash
install_dependencies.bat
```

## 2. Manual Installation (Backend)
Open a terminal in the `backend` folder and run:

```bash
cd backend
npm install
```

## 3. Python Requirements
Open a terminal in the `python` folder and run:

```bash
cd python
pip install -r requirements.txt
```

**Note**: You must have Python 3.10+ installed.

## 4. System Requirements (FFmpeg)
You need to have FFmpeg installed and added to your system PATH.

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**Windows (Manual):**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
2. Extract the files.
3. Add the `bin` folder to your System Environment Variables -> Path.

## 5. Running the Project
**Terminal 1 (Backend):**
```bash
cd backend
npm start
```
The server will start on port 5000.

**Terminal 2 (Frontend):**
Open `frontend/index.html` in your browser, or use a live server:
```bash
npx serve frontend
```

---

## 6. Colab Setup (Emotion-Aware v6.0)

Run this in your Colab notebook's install cell **instead of** the old pip install:

```bash
# System libs
!sudo apt-get -y install espeak-ng build-essential cmake libsndfile1 ffmpeg

# TTS (voice cloning)
!pip install --no-build-isolation git+https://github.com/coqui-ai/TTS

# Core dubbing
!pip install -q faster-whisper deep-translator google-generativeai ffmpeg-python edge-tts gtts demucs flask pyngrok

# 🆕 Emotion detection (required for v6.0)
!pip install -q librosa soundfile
```

### How the v6.0 Emotion Pipeline Works (Colab `cloud_api.py`):
1. **Extract audio** → FFmpeg
2. **Separate vocals** → Demucs (htdemucs, high quality)
3. **Transcribe** → Whisper large-v3
4. **🆕 Emotion pre-scan** → librosa analyzes each segment (pitch, energy, ZCR) → returns `{label, confidence, ssml_style, rate, pitch, volume}`
5. **🆕 Gemini professional rewrite** → sends all segments WITH emotion labels → Gemini rewrites (not just translates) the script to match the emotional energy
6. **🆕 Edge TTS with SSML** → uses `mstts:express-as style='cheerful/sad/angry/...'` + prosody adjustments for emotion-matching voice
7. **Prosody transfer + loudnorm + mix** → final video
