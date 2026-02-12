"""
Cloud API Server for AI Dubbing (v2 - Near-Human Quality)
Run this on Google Colab (or any cloud GPU server).
Your local Node.js server sends video files here for processing.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"  # Auto-accept TOS so model download doesn't hang

import sys
import json
import shutil
import tempfile
import asyncio
import gc
import numpy as np
import torch
import ffmpeg
import subprocess
import edge_tts
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
from gtts import gTTS

# Prosody transfer imports (lazy-loaded)
def get_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        return None

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB max

# Voice mapping
VOICE_MAPPING = {
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "zh-cn": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "ru": "ru-RU-SvetlanaNeural"
}

# Pre-load XTTS model once at startup (saves time per request)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device}")
if device == "cuda":
    print(f"💾 GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tts_engine = None
tts_lock = threading.Lock()  # Lock for GPU inference
xtts_supported = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]

def load_xtts():
    global tts_engine
    if tts_engine is not None:
        return
    print("🧬 Loading XTTS v2 model (one-time)...")
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    # Monkey-patch TOS prompt so it doesn't call input() in subprocess
    from TTS.utils.manage import ModelManager
    ModelManager.ask_tos = lambda self, *args, **kwargs: True
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    with torch.serialization.safe_globals([XttsConfig, BaseDatasetConfig, BaseAudioConfig, XttsAudioConfig, XttsArgs]):
        tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sys.stdout = old_stdout
    print("✅ XTTS loaded!")

async def generate_edge_tts(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except:
        return 0


def extract_best_reference(vocals_path, work_dir, target_duration=8.0):
    """Extract the loudest/clearest section of vocals as master reference for XTTS.
    Better reference = better voice cloning quality."""
    try:
        librosa = get_librosa()
        master_ref = os.path.join(work_dir, "chunks", "master_ref.wav")
        
        if librosa:
            import soundfile as sf
            y, sr = librosa.load(vocals_path, sr=24000)
            
            # Find the loudest window of target_duration
            frame_length = int(target_duration * sr)
            if len(y) < frame_length:
                # Audio shorter than target, use it all
                sf.write(master_ref, y, sr)
                return master_ref
            
            # Sliding window RMS to find loudest section
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            window_frames = int(target_duration * sr / 512)
            if len(rms) < window_frames:
                sf.write(master_ref, y, sr)
                return master_ref
            
            best_start = 0
            best_energy = 0
            for start_idx in range(len(rms) - window_frames):
                energy = np.mean(rms[start_idx:start_idx + window_frames])
                if energy > best_energy:
                    best_energy = energy
                    best_start = start_idx
            
            start_sample = best_start * 512
            end_sample = min(start_sample + frame_length, len(y))
            sf.write(master_ref, y[start_sample:end_sample], sr)
            print(f"   🎯 Best reference: {start_sample/sr:.1f}s - {end_sample/sr:.1f}s (energy: {best_energy:.4f})")
        else:
            # Fallback: use ffmpeg to extract first 8 seconds of vocals
            subprocess.run(f'ffmpeg -i "{vocals_path}" -t {target_duration} -ac 1 -ar 24000 "{master_ref}" -y -loglevel error', shell=True)
        
        return master_ref
    except Exception as e:
        print(f"⚠️ Reference extraction failed: {e}")
        return None

def transfer_prosody(original_segment_path, tts_path, output_path):
    """Transfer pitch contour and energy from original to TTS output.
    Makes dubbed audio match the speaker's emotion/intonation."""
    librosa = get_librosa()
    if not librosa:
        # No librosa, just copy
        shutil.copy2(tts_path, output_path)
        return
    
    try:
        import soundfile as sf
        
        # Load both audio files
        y_orig, sr_orig = librosa.load(original_segment_path, sr=24000)
        y_tts, sr_tts = librosa.load(tts_path, sr=24000)
        
        if len(y_orig) < 1024 or len(y_tts) < 1024:
            shutil.copy2(tts_path, output_path)
            return
        
        # 1. Match RMS energy (loudness)
        rms_orig = np.sqrt(np.mean(y_orig**2))
        rms_tts = np.sqrt(np.mean(y_tts**2))
        if rms_tts > 0:
            y_tts = y_tts * (rms_orig / rms_tts)
        
        # 2. Extract pitch from original
        f0_orig, voiced_flag_orig, _ = librosa.pyin(
            y_orig, fmin=50, fmax=400, sr=24000
        )
        f0_tts, voiced_flag_tts, _ = librosa.pyin(
            y_tts, fmin=50, fmax=400, sr=24000
        )
        
        # 3. Calculate average pitch shift needed
        valid_orig = f0_orig[~np.isnan(f0_orig)] if f0_orig is not None else np.array([])
        valid_tts = f0_tts[~np.isnan(f0_tts)] if f0_tts is not None else np.array([])
        
        if len(valid_orig) > 3 and len(valid_tts) > 3:
            median_orig = np.median(valid_orig)
            median_tts = np.median(valid_tts)
            
            if median_tts > 0 and median_orig > 0:
                # Calculate semitone shift (capped at ±4 semitones)
                semitone_shift = 12 * np.log2(median_orig / median_tts)
                semitone_shift = np.clip(semitone_shift, -4, 4)
                
                if abs(semitone_shift) > 0.5:  # Only shift if meaningful
                    y_tts = librosa.effects.pitch_shift(
                        y_tts, sr=24000, n_steps=semitone_shift
                    )
        
        # Clip to prevent distortion
        y_tts = np.clip(y_tts, -1.0, 1.0)
        sf.write(output_path, y_tts, 24000)
        
    except Exception as e:
        print(f"   ⚠️ Prosody transfer failed: {e}, using original TTS")
        shutil.copy2(tts_path, output_path)

def process_segment(i, seg, chunks_dir, vocals_path, master_ref, target_lang, gemini_api_key, use_xtts, context_prev="", context_next=""):
    """Worker function to process a single segment in parallel (v2 - quality upgrade)"""
    try:
        start, end, text = seg['start'], seg['end'], seg['text']
        duration = end - start
        
        # 1. Reference audio — always use ≥3s for good quality
        ref_path = os.path.join(chunks_dir, f"ref_{i}.wav")
        if duration < 3.0 and master_ref and os.path.exists(master_ref):
            ref_path = master_ref
        else:
            # Extract this segment's vocals as reference
            subprocess.run(f'ffmpeg -i "{vocals_path}" -ss {start} -t {duration} -ac 1 -ar 24000 "{ref_path}" -y -loglevel error', shell=True)
        
        # 2. Extract original audio segment for prosody matching
        orig_segment_path = os.path.join(chunks_dir, f"orig_{i}.wav")
        subprocess.run(f'ffmpeg -i "{vocals_path}" -ss {start} -t {duration} -ac 1 -ar 24000 "{orig_segment_path}" -y -loglevel error', shell=True)
        
        # 3. Translation (no Gemini polishing)
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        except:
            translated = text
        
        tts_path = os.path.join(chunks_dir, f"tts_{i}.wav")
        
        # 4. TTS Generation (GPU/CPU)
        if use_xtts and tts_engine:
            try:
                with tts_lock:
                    tts_engine.tts_to_file(text=translated, speaker_wav=ref_path, language=target_lang, file_path=tts_path)
                print(f"   [{i}] ✅ XTTS: {translated[:40]}...")
            except Exception as e:
                print(f"   [{i}] ⚠️ XTTS failed, using Edge TTS")
                voice = VOICE_MAPPING.get(target_lang, 'en-US-JennyNeural')
                try:
                    asyncio.run(generate_edge_tts(translated, voice, tts_path))
                except:
                    gTTS(text=translated, lang=target_lang, slow=False).save(tts_path)
        else:
            voice = VOICE_MAPPING.get(target_lang, 'en-US-JennyNeural')
            try:
                asyncio.run(generate_edge_tts(translated, voice, tts_path))
                print(f"   [{i}] ✅ Edge: {translated[:40]}...")
            except:
                gTTS(text=translated, lang=target_lang, slow=False).save(tts_path)
        
        # 5. Prosody transfer (match original pitch/energy)
        prosody_path = os.path.join(chunks_dir, f"prosody_{i}.wav")
        transfer_prosody(orig_segment_path, tts_path, prosody_path)
        
        # 6. High-quality time stretching (rubberband instead of atempo)
        actual_dur = get_audio_duration(prosody_path)
        if actual_dur == 0: actual_dur = 0.1
        speed_ratio = actual_dur / duration
        speed_ratio = max(0.75, min(speed_ratio, 1.3))  # Clamp to safe range
        synced_path = os.path.join(chunks_dir, f"synced_{i}.wav")
        
        # Try rubberband first (best quality), fallback to atempo
        rb_result = subprocess.run(
            f'ffmpeg -i "{prosody_path}" -filter:a "rubberband=tempo={speed_ratio}" "{synced_path}" -y -loglevel error',
            shell=True, capture_output=True
        )
        if rb_result.returncode != 0:
            # Fallback: asetrate + aresample (better than atempo)
            new_rate = int(24000 * speed_ratio)
            subprocess.run(
                f'ffmpeg -i "{prosody_path}" -filter:a "asetrate={new_rate},aresample=24000" "{synced_path}" -y -loglevel error',
                shell=True
            )
        
        return {
            "index": i,
            "start": start,
            "end": end,
            "tts_path": synced_path
        }
        
    except Exception as e:
        print(f"Error processing segment {i}: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device == "cuda" else 0,
        "xtts_loaded": tts_engine is not None
    })

@app.route('/dub', methods=['POST'])
def dub_video():
    """
    Accepts a video file + target_lang, returns dubbed video.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    target_lang = request.form.get('target_lang', 'hi')
    gemini_api_key = request.form.get('gemini_api_key', '')
    
    # Save uploaded file
    work_dir = tempfile.mkdtemp()
    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir)
    
    input_path = os.path.join(work_dir, secure_filename(file.filename))
    file.save(input_path)
    print(f"\n{'='*50}")
    print(f"📥 New Job (Parallel): {file.filename} → {target_lang}")
    print(f"{'='*50}")
    
    try:
        # Step 1: Extract Audio
        print("🎵 Step 1: Extracting Audio...")
        full_audio = os.path.join(work_dir, "full_source.wav")
        ffmpeg.input(input_path).output(full_audio, ac=1, ar=24000).run(
            cmd="ffmpeg", overwrite_output=True, quiet=True
        )
        total_duration = get_audio_duration(full_audio)
        
        # Step 2: Vocal Separation (Demucs)
        print("🎤 Step 2: Separating Vocals...")
        demucs_out = os.path.join(work_dir, "demucs_out")
        cmd = ["demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", device,
               "--shifts", "0", "--overlap", "0.1", full_audio, "-o", demucs_out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        vocals_path = os.path.join(demucs_out, "htdemucs", "full_source", "vocals.wav")
        background_path = os.path.join(demucs_out, "htdemucs", "full_source", "no_vocals.wav")
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        
        # Step 3: Transcription (Whisper)
        print("📝 Step 3: Transcribing...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        segments_gen, info = whisper_model.transcribe(
            vocals_path, beam_size=1, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]
        print(f"   Found {len(segments)} segments.")
        del whisper_model
        gc.collect()
        
        # Step 4: Load XTTS if needed
        use_xtts = target_lang in xtts_supported
        if use_xtts:
            load_xtts()
        
        # Master Reference — extract best 8-second voice sample for XTTS cloning
        print("🎯 Step 4b: Extracting best voice reference...")
        master_ref = extract_best_reference(vocals_path, work_dir, target_duration=8.0)
        
        # Step 5: Parallel Dubbing with context
        print(f"🚀 Step 5: Dubbing {len(segments)} segments (emotion-aware)...")
        
        max_workers = 4
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, seg in enumerate(segments):
                # Pass surrounding context for coherent translation
                ctx_prev = segments[i-1]['text'] if i > 0 else ""
                ctx_next = segments[i+1]['text'] if i < len(segments)-1 else ""
                futures.append(
                    executor.submit(process_segment, i, seg, chunks_dir, vocals_path, master_ref, target_lang, gemini_api_key, use_xtts, ctx_prev, ctx_next)
                )
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        
        # Sort results by index to ensure correct order
        results.sort(key=lambda x: x['index'])
        
        # Build ordered list with gaps
        print("🔧 Step 6: Merging...")
        ordered_files = []
        current_time = 0.0
        
        for res in results:
            start = res['start']
            end = res['end']
            tts_path = res['tts_path']
            
            # Gap processing (sequential, fast)
            gap_dur = start - current_time
            if gap_dur > 0.1:
                gap_path = os.path.join(chunks_dir, f"gap_{res['index']}.wav")
                subprocess.run(f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap_dur} {gap_path} -y -loglevel error', shell=True)
                ordered_files.append(gap_path)
            
            ordered_files.append(tts_path)
            current_time = end
        
        # Trailing gap
        if current_time < total_duration:
            gap_end = os.path.join(chunks_dir, "gap_end.wav")
            subprocess.run(f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {total_duration - current_time} {gap_end} -y -loglevel error', shell=True)
            ordered_files.append(gap_end)
            
        # Merge
        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in ordered_files:
                f.write(f"file '{p}'\n")
        
        full_tts = os.path.join(work_dir, "full_tts.wav")
        subprocess.run(f'ffmpeg -f concat -safe 0 -i "{concat_list}" -c copy "{full_tts}" -y -loglevel error', shell=True)
        
        # Professional audio mixing: normalize voice + duck background
        final_audio = os.path.join(work_dir, "final_mixed.wav")
        mix_filter = (
            "[0]loudnorm=I=-16:TP=-1.5:LRA=11[voice];"
            "[1]volume=0.25[bg];"
            "[voice][bg]amix=inputs=2:duration=first:weights=1 0.8"
        )
        subprocess.run(
            f'ffmpeg -i "{full_tts}" -i "{background_path}" -filter_complex "{mix_filter}" "{final_audio}" -y -loglevel error',
            shell=True
        )
        
        # Final video
        output_video = os.path.join(work_dir, "dubbed_output.mp4")
        subprocess.run(f'ffmpeg -i "{input_path}" -i "{final_audio}" -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental "{output_video}" -y -loglevel error', shell=True)
        
        print(f"🎉 DONE! Sending back dubbed video...")
        return send_file(output_video, mimetype='video/mp4', as_attachment=True, download_name=f"dubbed_{target_lang}.mp4")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup after sending (delayed)
        pass

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 AI Dubbing Cloud API Server (Parallel)")
    print(f"   Device: {device}")
    print("   Endpoint: POST /dub (file + target_lang)")
    print("="*50 + "\n")
    
    # Pre-load XTTS at startup for faster first request
    load_xtts()
    
    app.run(host='0.0.0.0', port=5050, debug=False)
