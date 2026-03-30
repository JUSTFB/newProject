# AI Dubbing Studio - Agent Instructions

## Project Overview
This is an AI-powered video dubbing application that translates and dubs videos into multiple languages using voice cloning, emotion matching, and professional audio mixing.

## Tech Stack
- **Frontend**: Vanilla HTML/CSS/JS (no framework), located in `frontend/`
- **Backend**: Node.js + Express (ES modules), located in `backend/`
- **Python**: AI processing scripts, located in `python/`
- **Key Libraries**: FFmpeg, Demucs, Faster Whisper, XTTS v2, Edge TTS, librosa, Gemini AI

## Architecture
```
Browser (frontend/) → Express API (backend/) → Python AI (python/)
```
- `backend/server.js` is the Express server on port 5000
- `backend/services/pythonBridge.js` decides local vs cloud processing
- `python/dubbing_service.py` runs locally (CPU mode)
- `python/cloud_api.py` runs on Google Colab (GPU mode via ngrok)
- `.env` contains `CLOUD_API_URL` and `GEMINI_API_KEY`

## Key Rules
- Backend uses ES modules (`import` not `require`) — `package.json` has `"type": "module"`
- Python communicates with Node via JSON lines on stdout (one JSON object per line)
- Never commit `.env` files (they contain API keys)
- FFmpeg is a system dependency, not an npm package
- Python requires: torch, faster-whisper, TTS (coqui), edge-tts, librosa, deep-translator, google-generativeai, flask, demucs

## File Structure
- `frontend/index.html` — Main UI with embedded JS logic
- `frontend/script.js` — Legacy, main logic is in index.html
- `frontend/style.css` — Styles with light/dark theme
- `backend/server.js` — Express API (upload, status, result endpoints)
- `backend/services/jobs.js` — In-memory job tracking (Map-based)
- `backend/services/pythonBridge.js` — Bridges Node ↔ Python
- `python/cloud_api.py` — Flask server for Colab (GPU processing)
- `python/dubbing_service.py` — Local dubbing (CPU mode)

## API Endpoints
- `POST /api/process` — Upload video, returns jobId
- `GET /api/status/:jobId` — Poll progress (returns {status, progress, stage})
- `GET /api/result/:jobId` — Get final video URL
- `POST /api/compare` — Compare multiple voice engines

## Running the Project
```bash
cd backend && npm start    # Start server on port 5000
# Open http://localhost:5000/frontend/index.html
```

## Common Tasks
- Adding a new language: Update `VOICE_MAPPING` in both `python/cloud_api.py` and `python/dubbing_service.py`, plus the `<select>` in `frontend/index.html`
- Changing TTS engine: Edit `voice_engine` parameter handling in `pythonBridge.js` and `cloud_api.py`
- Debugging: Check `backend/server.js` console logs and Python stdout JSON messages
