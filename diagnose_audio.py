import torch
import torchaudio
import soundfile as sf
import numpy as np
import os

log_file = "d:\\newProject\\diagnosis_patch_test.txt"

def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# --- MONKEY PATCH START ---
def custom_load(filepath, **kwargs):
    try:
        data, samplerate = sf.read(filepath)
        # soundfile returns (frames, channels) or (frames,)
        wav_tensor = torch.from_numpy(data).float()
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        else:
            wav_tensor = wav_tensor.t()
        return wav_tensor, samplerate
    except Exception as e:
        log(f"Custom load failed: {e}")
        raise e

log("Applying monkey-patch...")
torchaudio.load = custom_load
# --- MONKEY PATCH END ---

# Create dummy wav
dummy_wav = "d:\\newProject\\test_audio_patch.wav"
sr = 16000
data = np.random.uniform(-1, 1, sr).astype(np.float32)
sf.write(dummy_wav, data, sr)
log(f"Created dummy wav at {dummy_wav}")

try:
    log("Attempt 1: torchaudio.load() with patch")
    y, s = torchaudio.load(dummy_wav)
    log(f"Success! Shape: {y.shape}, SR: {s}")
except Exception as e:
    log(f"Failed Attempt 1: {e}")

# Cleanup
if os.path.exists(dummy_wav):
    os.remove(dummy_wav)
