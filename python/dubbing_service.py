import argparse
import sys
import os
import time
import json
import torch
import ffmpeg
from deep_translator import GoogleTranslator
from TTS.api import TTS
import whisper

# Ensure output is unbuffered for real-time Node.js parsing
import functools
print = functools.partial(print, flush=True)

# FORCE UTF-8 ENCODING FOR WINDOWS CONSOLE
# This prevents crashes when printing non-ASCII characters (e.g. from TTS logs or translations)
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Force torchaudio backend to avoid TorchCodec requirement
import torchaudio
import torch
import numpy as np
try:
    import soundfile as sf
    print(f"Soundfile library version: {sf.__version__}")
    
    # Monkey-patch torchaudio.load to usage soundfile directly
    # This bypasses the new torchaudio 2.10+ requirement for TorchCodec or ffmpeg
    def custom_load(filepath, **kwargs):
        # soundfile.read returns (data, samplerate)
        # data is (frames, channels) for multi-channel, or (frames,) for mono
        try:
            wav, sr = sf.read(filepath)
            wav_tensor = torch.from_numpy(wav).float()
            
            # Torchaudio expects (channels, frames)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0) # (1, frames)
            else:
                wav_tensor = wav_tensor.t() # (channels, frames)
                
            return wav_tensor, sr
        except Exception as e:
            print(f"Error in custom_load for {filepath}: {e}")
            raise e

    # Apply monkey-patch
    print("Overriding torchaudio.load with custom soundfile-based loader (Monkey-Patch)")
    torchaudio.load = custom_load
    
    # Also patch save just in case, though Coqui usually handles save internally or we use ffmpeg
    def custom_save(filepath, src, sample_rate, **kwargs):
        # src is (channels, frames)
        if src.ndim == 2:
            src = src.t() # (frames, channels)
        
        sf.write(filepath, src.numpy(), sample_rate)
        
    torchaudio.save = custom_save

except ImportError:
    print("Error: 'soundfile' library not found. Please install it via pip.")
except Exception as e:
    print(f"Error setting audio backend: {e}")

# AUTO-AGREE TO COQUI LICENSE
os.environ["COQUI_TOS_AGREED"] = "1"

def update_status(progress, stage, data=None):
    """Print status in a JSON format that Node.js can easily parse."""
    message = {
        "progress": progress,
        "stage": stage
    }
    if data:
        message.update(data)
    print(json.dumps(message))

def process_video(input_path, target_lang, output_path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        update_status(5, f"Initializing using {device.upper()}...")

        # --- STEP 1: Extract Audio ---
        update_status(10, "Extracting audio")
        audio_path = input_path.replace(".mp4", ".wav")
        # Extract audio using ffmpeg
        (
            ffmpeg
            .input(input_path)
            .output(audio_path, ac=1, ar=16000) # Mono 16kHz for Whisper
            .run(cmd="ffmpeg", overwrite_output=True, quiet=True)
        )

        # --- STEP 2: STT (Whisper) ---
        update_status(20, "Loading Whisper (STT)...")
        # Load small model for speed, use "medium" or "large" for better accuracy
        model = whisper.load_model("small", device=device)
        
        update_status(30, "Transcribing...")
        result = model.transcribe(audio_path)
        original_text = result["text"]
        
        # --- STEP 3: Translation ---
        update_status(50, "Translating...")
        # Map simple language codes to Google Translate codes if needed
        # deep_translator usually handles standard codes well
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(original_text)
        

        # --- STEP 4: TTS & Voice Cloning (Coqui XTTS) ---
        update_status(60, "Loading TTS Model...")
        
        # XTTS supported languages
        xtts_supported_langs = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
        
        dub_audio_path = input_path.replace(".mp4", f"_dub_{target_lang}.wav")

        if target_lang in xtts_supported_langs:
            update_status(65, f"Using Coqui XTTS (Voice Cloning) for {target_lang}...")
            # Note: This will download the model (~2GB) on first run
            
            # Workaround for PyTorch 2.6+ weights_only=True default
            # We need to allow TTS config classes
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            
            with torch.serialization.safe_globals([XttsConfig, BaseDatasetConfig, BaseAudioConfig, XttsAudioConfig, XttsArgs]):
                 tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            
            update_status(70, "Synthesizing dub with Voice Cloning...")
            
            # XTTS requires a speaker wav to clone. Use the extracted audio.
            tts.tts_to_file(
                text=translated_text,
                speaker_wav=audio_path,
                language=target_lang,
                file_path=dub_audio_path
            )
        else:
            update_status(65, f"Target language '{target_lang}' not supported by XTTS. Using Google TTS (Standard Voice)...")
            from gtts import gTTS
            
            update_status(70, "Synthesizing dub with Google TTS...")
            # slow=False for normal speed
            tts = gTTS(text=translated_text, lang=target_lang, slow=False)
            tts.save(dub_audio_path)
        
        # --- STEP 5: Merge Audio & Video ---
        update_status(90, "Merging video...")
        
        # Merge original video (without audio) + new dubbed audio
        # Using 'shortest=True' to cut if audio/video lengths differ slightly
        video = ffmpeg.input(input_path)
        audio = ffmpeg.input(dub_audio_path)
        
        (
            ffmpeg
            .output(video.video, audio, output_path, vcodec='copy', acodec='aac', strict='experimental')
            .run(cmd="ffmpeg", overwrite_output=True, quiet=True)
        )
        
        # Cleanup temp files
        if os.path.exists(audio_path):
             os.remove(audio_path)
        if os.path.exists(dub_audio_path):
             os.remove(dub_audio_path)

        update_status(100, "Completed", {
            "result": {
                "transcript": original_text,
                "translatedText": translated_text,
                "finalVideo": os.path.basename(output_path)
            }
        })

    except Exception as e:
        # Print error in a way Node.js can capture as a failure
        import traceback
        traceback.print_exc() # Print full stack trace to stderr for debugging
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--target_lang", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    process_video(args.input_video, args.target_lang, args.output_path)
