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
