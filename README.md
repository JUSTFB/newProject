# 🎬 AI Dubbing Studio

> Translate and dub any video into multiple languages using AI — with voice cloning, emotion-matching, and professional audio mixing.

AI Dubbing Studio is an open-source tool that takes a video, transcribes it, translates it, and generates dubbed audio using cloned voices. It supports a **hybrid cloud architecture** — run the heavy AI on Google Colab's free GPU while keeping your dev server local.

---

## ✨ Features

- **Voice Cloning** — XTTS v2 clones the original speaker's voice
- **Prosody Transfer** — Matches pitch and energy of original speech using librosa
- **Vocal Separation** — Demucs isolates vocals from background music
- **Multi-Language** — Hindi, Spanish, French, German, Japanese, Korean, and more
- **Professional Mixing** — Normalized voice + ducked background music
- **High-Quality Time Stretch** — Rubberband-based speed sync (no robotic artifacts)
- **Live Progress** — Real-time progress bar in the browser
- **Cloud GPU Support** — Offload processing to Google Colab (free T4 GPU)

---

## 📁 Project Structure

```
├── frontend/              # Browser UI
│   ├── index.html         # Main page with embedded JS
│   ├── style.css          # Styles
│   └── script.js          # (Legacy, main logic is in index.html)
│
├── backend/               # Node.js server
│   ├── server.js          # Express API (upload, status, result)
│   ├── services/
│   │   ├── pythonBridge.js # Bridges Node ↔ Python (local or cloud)
│   │   └── jobs.js        # In-memory job tracking
│   └── package.json
│
├── python/                # AI dubbing scripts
│   ├── cloud_api.py       # Flask server for Colab (GPU processing)
│   ├── dubbing_service.py # Local dubbing (CPU mode)
│   └── requirements.txt   # Python dependencies
│
├── models/rvc/            # (Optional) RVC voice models
├── colab_cloud_server.ipynb  # Colab notebook for cloud mode
├── .env                   # Environment variables
└── nodemon.json           # Dev server config
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| **Node.js** | 18+ | `node -v` |
| **Python** | 3.10+ | `python --version` |
| **FFmpeg** | Any | `ffmpeg -version` |

> **FFmpeg** is required. Install from [ffmpeg.org](https://ffmpeg.org/download.html) or `choco install ffmpeg` (Windows) / `brew install ffmpeg` (Mac).

---

### Option A: Cloud Mode (Recommended — Free GPU)

This runs the heavy AI on Google Colab and your local machine just serves the UI.

#### 1. Clone & Install (Local)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-dubbing-studio.git
cd ai-dubbing-studio

# Install backend dependencies
cd backend
npm install
cd ..
```

#### 2. Set Up Colab (Cloud GPU)

1. Open `colab_cloud_server.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells **in order**:
   - **Step 1**: GPU check
   - **Step 2**: Install dependencies (takes ~3-5 min)
   - **Step 3**: Paste your [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken) (free signup)
   - **Step 4**: Upload `python/cloud_api.py` from this repo
   - **Step 5**: Start the server — copy the ngrok URL it prints

#### 3. Connect Local → Cloud

Create a `.env` file in the project root:

```env
CLOUD_API_URL=https://your-ngrok-url.ngrok-free.app
```

#### 4. Start Local Server

```bash
cd backend
npm start
```

#### 5. Open the UI

Go to **http://localhost:5000/frontend/index.html**

Upload a video, select language, and hit **🚀 Start Dubbing**!

---

### Option B: Local Mode (No Cloud, CPU Only)

This runs everything locally. Slower but no cloud setup needed.

#### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ai-dubbing-studio.git
cd ai-dubbing-studio

# Install backend
cd backend
npm install
cd ..

# Install Python dependencies
pip install -r python/requirements.txt
pip install edge-tts gtts librosa soundfile
```

#### 2. Start Server (No .env needed)

```bash
cd backend
npm start
```

Without `CLOUD_API_URL` in `.env`, the server automatically uses **local mode** and runs `python/dubbing_service.py`.

#### 3. Open the UI

Go to **http://localhost:5000/frontend/index.html**

> ⚠️ Local mode uses CPU only — expect **3-5x longer** processing times compared to cloud mode.

---

## 🌍 Supported Languages

| Language | Code | XTTS Voice Cloning |
|----------|------|--------------------|
| Hindi | `hi` | ✅ |
| English | `en` | ✅ |
| Spanish | `es` | ✅ |
| French | `fr` | ✅ |
| German | `de` | ✅ |
| Italian | `it` | ✅ |
| Portuguese | `pt` | ✅ |
| Japanese | `ja` | ✅ |
| Korean | `ko` | ✅ |
| Russian | `ru` | ✅ |
| Chinese | `zh-cn` | ✅ |

---

## ⚙️ How It Works

```
Video Input
    │
    ▼
┌─────────────────┐
│  1. Extract Audio│  (FFmpeg)
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Separate     │  (Demucs) → Vocals + Background
│     Vocals       │
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Transcribe   │  (Faster Whisper)
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Translate    │  (Google Translator)
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. Voice Clone  │  (XTTS v2 — clones original speaker)
│  + TTS           │
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Prosody Match│  (librosa — pitch & energy transfer)
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Time Sync    │  (Rubberband — broadcast quality)
└────────┬────────┘
         ▼
┌─────────────────┐
│  8. Mix Audio    │  (FFmpeg — voice + ducked background)
└────────┬────────┘
         ▼
    Dubbed Video Output
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `CLOUD_API_URL` | No | ngrok URL of your Colab server. If not set, uses local mode. |

### Optional: RVC Voice Models

For even better voice quality, place `.pth` RVC model files in `models/rvc/`. The dubbing service will automatically use them.

---

## 📊 Performance

| Video Length | Cloud (T4 GPU) | Local (CPU) |
|-------------|----------------|-------------|
| 10 sec | ~30 sec | ~2 min |
| 1 min | ~3 min | ~10 min |
| 5 min | ~10 min | ~40 min |
| 25 min | ~30 min | ~2+ hrs |

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| `No file uploaded` error | Make sure `form-data` and `node-fetch` are installed: `cd backend && npm install` |
| Colab disconnects mid-process | Use a shorter video or keep the Colab tab active |
| `FFmpeg not found` | Install FFmpeg and add it to your system PATH |
| Progress bar stuck | Check the terminal for errors, restart with `npm start` |
| ngrok tunnel error | Free tier has limits — restart the Colab notebook to get a new URL |

---

## 📝 License

MIT License — Use freely, modify, and distribute.

---

## 🙏 Credits

Built with:
- [XTTS v2](https://github.com/coqui-ai/TTS) — Voice cloning
- [Demucs](https://github.com/facebookresearch/demucs) — Vocal separation
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) — Speech recognition
- [librosa](https://github.com/librosa/librosa) — Audio analysis
- [Edge TTS](https://github.com/rany2/edge-tts) — Fallback text-to-speech
