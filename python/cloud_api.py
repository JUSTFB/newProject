"""
Cloud API Server for AI Dubbing (v5.3 - CLEAN EDGE TTS - NO SSML)
Run this on Google Colab (or any cloud GPU server).
Your local Node.js server sends video files here for processing.
Engines: Edge TTS (free) | XTTS (voice clone) | ElevenLabs | Azure Neural
!!! IF YOU HEAR XML/SSML BEING SPOKEN, THIS FILE IS NOT LOADED !!!
Priority: QUALITY over speed. Every segment gets full attention.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"

import sys
import json
import shutil
import tempfile
import asyncio
import gc  
import math
import numpy as np
import torch
import ffmpeg
import subprocess
import edge_tts
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from gtts import gTTS

# Lazy imports for optional deps
def get_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        return None

def get_genai():
    try:
        import google.generativeai as genai
        return genai
    except ImportError:
        return None

def get_elevenlabs():
    try:
        from elevenlabs.client import ElevenLabs
        return ElevenLabs
    except ImportError:
        return None

def get_azure_speech():
    try:
        import azure.cognitiveservices.speech as speechsdk
        return speechsdk
    except ImportError:
        return None

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB max

# ── Language & Voice Config ───────────────────────────────────────────────────
# Edge TTS voices (Microsoft Neural — free, unlimited)
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

# ElevenLabs voice IDs (use actual IDs, NOT names — names cause API errors)
ELEVENLABS_VOICE_MAPPING = {
    "hi": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "es": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "fr": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "de": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "it": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "pt": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "zh-cn": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "ja": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "ko": "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "ru": "21m00Tcm4TlvDq8ikWAM"   # Rachel
}

# Azure Neural TTS voices (500K chars free/month, emotion styles)
AZURE_VOICE_MAPPING = {
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

LANG_NAMES = {
    "hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "zh-cn": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian"
}

# ── GPU & Model Setup ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device}")
if device == "cuda":
    print(f"💾 GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tts_engine = None
tts_lock = threading.Lock()
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
    from TTS.utils.manage import ModelManager
    ModelManager.ask_tos = lambda self, *args, **kwargs: True
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    with torch.serialization.safe_globals([XttsConfig, BaseDatasetConfig, BaseAudioConfig, XttsAudioConfig, XttsArgs]):
        tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sys.stdout = old_stdout
    print("✅ XTTS loaded!")

# ── Edge TTS (plain text, no SSML) ───────────────────────────────────────────
async def generate_edge_tts(text, voice, output_file):
    """Generate TTS with Edge TTS. Plain text only - NO SSML, NO XML, NO PROSODY."""
    print(f"     🟢 [v5.3] Edge TTS PLAIN TEXT: '{text[:60]}...'")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

# ── ElevenLabs TTS ───────────────────────────────────────────────────────────
def generate_elevenlabs_tts(text, target_lang, output_file, api_key, emotion="neutral"):
    """Generate TTS using ElevenLabs API (eleven_multilingual_v2).
    Free tier: 10,000 chars/month. Best quality available."""
    ElevenLabsClient = get_elevenlabs()
    if not ElevenLabsClient or not api_key:
        raise Exception("ElevenLabs not available")
    
    try:
        client = ElevenLabsClient(api_key=api_key)
        voice_id = ELEVENLABS_VOICE_MAPPING.get(target_lang, "21m00Tcm4TlvDq8ikWAM")  # Rachel ID
        
        # Map emotion to voice settings
        stability_map = {
            "excited": 0.3, "happy": 0.4, "angry": 0.3,
            "sad": 0.6, "whisper": 0.7, "serious": 0.6, "neutral": 0.5
        }
        similarity_map = {
            "excited": 0.8, "happy": 0.75, "angry": 0.85,
            "sad": 0.7, "whisper": 0.6, "serious": 0.8, "neutral": 0.75
        }
        
        stability = stability_map.get(emotion, 0.5)
        similarity = similarity_map.get(emotion, 0.75)
        
        # Generate audio — use simple text-to-speech convert
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # Write to temp mp3 first, then convert to wav
        mp3_temp = output_file + ".mp3"
        with open(mp3_temp, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        
        # Convert mp3 to wav for compatibility with rest of pipeline
        subprocess.run(
            f'ffmpeg -i "{mp3_temp}" -ac 1 -ar 24000 "{output_file}" -y -loglevel error',
            shell=True
        )
        os.remove(mp3_temp)
        
        print(f"     🟣 ElevenLabs generated: {text[:40]}...")
        return True
    except Exception as e:
        print(f"     ⚠️ ElevenLabs error: {e}")
        raise

# ── Azure Neural TTS ─────────────────────────────────────────────────────────
def generate_azure_tts(text, target_lang, output_file, api_key, region="eastus", emotion="neutral"):
    """Generate TTS using Azure Cognitive Services (Neural voices with emotion styles).
    Free tier: 500,000 chars/month. Excellent quality with SSML emotion styles."""
    speechsdk = get_azure_speech()
    if not speechsdk or not api_key:
        raise Exception("Azure Speech SDK not available")
    
    try:
        speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
        voice_name = AZURE_VOICE_MAPPING.get(target_lang, "en-US-JennyNeural")
        speech_config.speech_synthesis_voice_name = voice_name
        
        # Set output format to wav
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )
        
        # Build SSML with emotion styles (Azure-specific)
        azure_emotion_map = {
            "excited": "excited", "happy": "cheerful", "angry": "angry",
            "sad": "sad", "whisper": "whispering", "serious": "serious",
            "neutral": "neutral"
        }
        style = azure_emotion_map.get(emotion, "neutral")
        
        # Build SSML with express-as for emotion
        ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
            xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
            <voice name='{voice_name}'>
                <mstts:express-as style='{style}'>
                    {text}
                </mstts:express-as>
            </voice>
        </speak>"""
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"     🔵 Azure Neural generated ({style}): {text[:40]}...")
            return True
        else:
            # Fallback: try without SSML styles
            result = synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print(f"     🔵 Azure Neural (plain): {text[:40]}...")
                return True
            raise Exception(f"Azure TTS failed: {result.reason}")
    except Exception as e:
        print(f"     ⚠️ Azure TTS error: {e}")
        raise

# ── Audio Utilities ──────────────────────────────────────────────────────────
def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except:
        return 0

# ── 1. Gemini Scene Translation ──────────────────────────────────────────────
def translate_full_scene_with_gemini(segments, target_lang, api_key):
    """Translate ALL segments together for contextual coherence.
    This is the single biggest quality improvement."""
    genai = get_genai()
    if not genai or not api_key:
        print("   ⚠️ No Gemini API, falling back to Google Translate")
        return translate_with_google(segments, target_lang)
    
    try:
        genai.configure(api_key=api_key)
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        
        # Build the full dialogue script
        dialogue_lines = []
        for i, seg in enumerate(segments):
            dialogue_lines.append(f"[{i}] {seg['text']}")
        
        full_script = "\n".join(dialogue_lines)
        
        prompt = f"""You are a professional dubbing translator for cinema and TV.

ORIGINAL DIALOGUE:
{full_script}

TARGET LANGUAGE: {lang_name}

DUBBING RULES (follow strictly):
1. Translate the ENTIRE script into {lang_name}
2. Use NATURAL spoken dialogue — how a real person talks, not formal writing
3. Match the EMOTION of each line (excited = !, sad = ..., question = ?)
4. Keep SIMILAR word count per line for lip-sync timing
5. Adapt cultural references — don't just translate literally
6. Preserve character personality and speaking style
7. Add natural filler words where appropriate (um, well, you know, etc. in {lang_name})
8. Each line must start with its number in brackets: [0], [1], [2]...
9. Output ONLY the translated lines, nothing else
10. Do NOT merge or split lines — keep the same number of lines

OUTPUT FORMAT (one line per segment):
[0] translated text here
[1] translated text here
..."""

        model = None
        response = None
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
        except Exception as e1:
            print(f"   ⚠️ gemini-2.0-flash failed: {e1}")
            try:
                model = genai.GenerativeModel('gemini-2.0-flash-lite')
                response = model.generate_content(prompt)
            except Exception as e2:
                print(f"   ⚠️ gemini-2.0-flash-lite also failed: {e2}")
                return translate_with_google(segments, target_lang)
        
        if not response or not response.text:
            return translate_with_google(segments, target_lang)
        
        # Parse response into translations
        translations = {}
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Parse [N] text format
            if line.startswith("["):
                bracket_end = line.find("]")
                if bracket_end > 0:
                    try:
                        idx = int(line[1:bracket_end])
                        text = line[bracket_end+1:].strip()
                        translations[idx] = text
                    except ValueError:
                        continue
        
        # Build result array, falling back to Google Translate for missing lines
        result = []
        for i, seg in enumerate(segments):
            if i in translations and len(translations[i]) > 0:
                result.append(translations[i])
            else:
                try:
                    from deep_translator import GoogleTranslator
                    result.append(GoogleTranslator(source='auto', target=target_lang).translate(seg['text']))
                except:
                    result.append(seg['text'])
        
        print(f"   🎬 Gemini translated {len(translations)}/{len(segments)} lines successfully")
        return result
        
    except Exception as e:
        print(f"   ⚠️ Gemini scene translation failed: {e}")
        return translate_with_google(segments, target_lang)

def translate_with_google(segments, target_lang):
    """Fallback: translate each segment individually with Google Translate"""
    from deep_translator import GoogleTranslator
    result = []
    for seg in segments:
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(seg['text'])
            result.append(translated)
        except:
            result.append(seg['text'])
    return result

# ── 2. Emotion Detection ────────────────────────────────────────────────────
def detect_emotion(audio_path):
    """Analyze audio segment to detect speaker emotion using librosa.
    Returns: emotion string (excited, happy, sad, angry, whisper, serious, neutral)"""
    librosa = get_librosa()
    if not librosa:
        return "neutral"
    
    try:
        y, sr = librosa.load(audio_path, sr=24000)
        if len(y) < 512:
            return "neutral"
        
        # Extract features
        rms = np.mean(librosa.feature.rms(y=y)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # Pitch analysis
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        
        if len(valid_f0) > 0:
            pitch_mean = np.mean(valid_f0)
            pitch_std = np.std(valid_f0)
            pitch_range = np.max(valid_f0) - np.min(valid_f0)
        else:
            pitch_mean, pitch_std, pitch_range = 150, 10, 20
        
        # Speaking rate (approximate via zero crossing rate)
        speaking_rate = zcr
        
        # Classify emotion based on audio features
        if rms < 0.01:
            return "whisper"
        elif rms > 0.15 and pitch_std > 40 and speaking_rate > 0.1:
            return "excited"
        elif rms > 0.12 and pitch_mean > 200:
            return "happy"
        elif rms > 0.1 and pitch_std < 15 and speaking_rate > 0.08:
            return "angry"
        elif rms < 0.04 and pitch_mean < 150:
            return "sad"
        elif rms > 0.05 and pitch_std < 20:
            return "serious"
        else:
            return "neutral"
    
    except Exception as e:
        return "neutral"

# ── 3. Best Reference Extraction ─────────────────────────────────────────────
def extract_best_reference(vocals_path, work_dir, target_duration=12.0):
    """Extract the loudest/clearest section of vocals as master reference for XTTS."""
    try:
        librosa = get_librosa()
        master_ref = os.path.join(work_dir, "chunks", "master_ref.wav")
        
        if librosa:
            import soundfile as sf
            y, sr = librosa.load(vocals_path, sr=24000)
            
            frame_length = int(target_duration * sr)
            if len(y) < frame_length:
                sf.write(master_ref, y, sr)
                return master_ref
            
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
            subprocess.run(f'ffmpeg -i "{vocals_path}" -t {target_duration} -ac 1 -ar 24000 "{master_ref}" -y -loglevel error', shell=True)
        
        return master_ref
    except Exception as e:
        print(f"⚠️ Reference extraction failed: {e}")
        return None

# ── 4. Prosody Transfer (pitch/energy matching) ──────────────────────────────
def transfer_prosody(original_segment_path, tts_path, output_path):
    """Transfer pitch contour and energy from original to TTS output."""
    librosa = get_librosa()
    if not librosa:
        shutil.copy2(tts_path, output_path)
        return
    
    try:
        import soundfile as sf
        y_orig, _ = librosa.load(original_segment_path, sr=24000)
        y_tts, _ = librosa.load(tts_path, sr=24000)
        
        if len(y_orig) < 1024 or len(y_tts) < 1024:
            shutil.copy2(tts_path, output_path)
            return
        
        # Match RMS energy
        rms_orig = np.sqrt(np.mean(y_orig**2))
        rms_tts = np.sqrt(np.mean(y_tts**2))
        if rms_tts > 0:
            y_tts = y_tts * (rms_orig / rms_tts)
        
        # Pitch matching
        f0_orig, _, _ = librosa.pyin(y_orig, fmin=50, fmax=400, sr=24000)
        f0_tts, _, _ = librosa.pyin(y_tts, fmin=50, fmax=400, sr=24000)
        
        valid_orig = f0_orig[~np.isnan(f0_orig)] if f0_orig is not None else np.array([])
        valid_tts = f0_tts[~np.isnan(f0_tts)] if f0_tts is not None else np.array([])
        
        if len(valid_orig) > 3 and len(valid_tts) > 3:
            median_orig = np.median(valid_orig)
            median_tts = np.median(valid_tts)
            
            if median_tts > 0 and median_orig > 0:
                semitone_shift = 12 * np.log2(median_orig / median_tts)
                semitone_shift = np.clip(semitone_shift, -2, 2)
                
                if abs(semitone_shift) > 1.0:
                    y_tts = librosa.effects.pitch_shift(y_tts, sr=24000, n_steps=semitone_shift)
        
        y_tts = np.clip(y_tts, -1.0, 1.0)
        sf.write(output_path, y_tts, 24000)
        
    except Exception as e:
        print(f"   ⚠️ Prosody transfer failed: {e}")
        shutil.copy2(tts_path, output_path)

# ── 5. Audio Post-Processing Chain ───────────────────────────────────────────
def post_process_audio(input_path, output_path, room_tone_path=None):
    """Professional audio post-processing: compression, de-essing, EQ matching."""
    try:
        filters = []
        
        # 1. De-esser (reduce harsh sibilance)
        filters.append("highpass=f=80")  # Remove sub-bass rumble
        filters.append("equalizer=f=6000:t=q:w=2:g=-3")  # Gentle de-ess at 6kHz
        
        # 2. Compression (even out loud/quiet parts)
        filters.append("acompressor=threshold=-20dB:ratio=3:attack=5:release=50:makeup=2dB")
        
        # 3. Warmth (slight mid boost for voice presence)
        filters.append("equalizer=f=2500:t=q:w=1:g=2")  # Presence boost
        filters.append("equalizer=f=200:t=q:w=1:g=1")   # Warmth
        
        # 4. Limiter (prevent clipping)
        filters.append("alimiter=limit=0.95:attack=5:release=50")
        
        filter_chain = ",".join(filters)
        
        result = subprocess.run(
            f'ffmpeg -i "{input_path}" -af "{filter_chain}" "{output_path}" -y -loglevel error',
            shell=True, capture_output=True
        )
        
        if result.returncode != 0:
            # Fallback: just copy
            shutil.copy2(input_path, output_path)
            
    except Exception as e:
        print(f"   ⚠️ Post-processing failed: {e}")
        shutil.copy2(input_path, output_path)

# ── 6. Micro-Pause Insertion ─────────────────────────────────────────────────
def add_micro_pauses(tts_path, output_path, emotion="neutral"):
    """Add subtle fade-in/fade-out and breathing-like pauses to make speech natural."""
    try:
        # Fade durations based on emotion
        fade_map = {
            "excited": (10, 15),      # ms - short fades for energy
            "happy": (15, 20),
            "angry": (5, 10),          # Very sharp attack
            "sad": (40, 60),           # Long, gentle fades
            "whisper": (30, 50),
            "serious": (20, 30),
            "neutral": (20, 25),
        }
        
        fade_in, fade_out = fade_map.get(emotion, (20, 25))
        
        # Add fade-in, fade-out, and a tiny silence pad (breathing room)
        filters = [
            f"afade=t=in:d={fade_in/1000}",
            f"afade=t=out:st=-1:d={fade_out/1000}",  # Will be recalculated
        ]
        
        # Get duration to properly set fade-out start
        dur = get_audio_duration(tts_path)
        if dur > 0:
            fade_out_start = max(0, dur - fade_out/1000)
            filter_str = f"afade=t=in:d={fade_in/1000},afade=t=out:st={fade_out_start}:d={fade_out/1000}"
        else:
            filter_str = f"afade=t=in:d={fade_in/1000}"
        
        # Add 30ms silence pad at start and end (breathing room)
        pad_filter = f"adelay={30}|{30},apad=pad_dur=0.03"
        full_filter = f"{filter_str},{pad_filter}"
        
        result = subprocess.run(
            f'ffmpeg -i "{tts_path}" -af "{full_filter}" "{output_path}" -y -loglevel error',
            shell=True, capture_output=True
        )
        
        if result.returncode != 0:
            shutil.copy2(tts_path, output_path)
    except:
        shutil.copy2(tts_path, output_path)

# ── Segment Processing (v3 - Cinema Quality) ─────────────────────────────────
def xtts_best_of_n(tts_engine, text, ref_path, target_lang, output_path, n=2):
    """Generate TTS N times and pick the one with best audio quality (highest RMS, clearest).
    This dramatically improves XTTS output consistency."""
    librosa = get_librosa()
    candidates = []
    
    for attempt in range(n):
        candidate_path = output_path.replace(".wav", f"_candidate_{attempt}.wav")
        try:
            tts_engine.tts_to_file(text=text, speaker_wav=ref_path, language=target_lang, file_path=candidate_path)
            
            # Score: prefer higher RMS (louder = more confident) and longer duration
            if librosa and os.path.exists(candidate_path):
                y, sr = librosa.load(candidate_path, sr=24000)
                rms = np.sqrt(np.mean(y**2))
                dur = len(y) / sr
                # Penalize very short outputs (likely garbled)
                score = rms * min(dur, 10.0)
                candidates.append((candidate_path, score))
            else:
                candidates.append((candidate_path, 1.0))
        except:
            continue
    
    if not candidates:
        return False
    
    # Pick best candidate
    best_path, best_score = max(candidates, key=lambda x: x[1])
    shutil.copy2(best_path, output_path)
    
    # Cleanup
    for path, _ in candidates:
        if path != best_path and os.path.exists(path):
            try: os.remove(path)
            except: pass
    if os.path.exists(best_path) and best_path != output_path:
        try: os.remove(best_path)
        except: pass
    
    return True

def process_segment(i, seg, chunks_dir, vocals_path, master_ref, target_lang, translated_text, use_xtts, emotion="neutral",
                    voice_engine="edge", elevenlabs_key=None, azure_key=None, azure_region="eastus"):
    """Process a single segment with maximum quality pipeline (v5 - Multi-Engine).
    voice_engine: 'edge' | 'xtts' | 'elevenlabs' | 'azure'
    """
    try:
        start, end = seg['start'], seg['end']
        duration = end - start
        
        # 1. Reference audio — used by XTTS only, but extract for prosody matching too
        ref_path = os.path.join(chunks_dir, f"ref_{i}.wav")
        if duration < 4.0 and master_ref and os.path.exists(master_ref):
            ref_path = master_ref
        else:
            subprocess.run(f'ffmpeg -i "{vocals_path}" -ss {start} -t {duration} -ac 1 -ar 24000 "{ref_path}" -y -loglevel error', shell=True)
            if duration < 5.0 and master_ref and os.path.exists(master_ref):
                combined_ref = os.path.join(chunks_dir, f"ref_combined_{i}.wav")
                subprocess.run(
                    f'ffmpeg -i "{ref_path}" -i "{master_ref}" -filter_complex "[0][1]concat=n=2:v=0:a=1" "{combined_ref}" -y -loglevel error',
                    shell=True
                )
                if os.path.exists(combined_ref):
                    ref_path = combined_ref
        
        # 2. Extract original segment for prosody matching
        orig_segment_path = os.path.join(chunks_dir, f"orig_{i}.wav")
        subprocess.run(f'ffmpeg -i "{vocals_path}" -ss {start} -t {duration} -ac 1 -ar 24000 "{orig_segment_path}" -y -loglevel error', shell=True)
        
        # 3. Detect emotion from original audio
        emotion = detect_emotion(orig_segment_path)
        print(f"   [{i}] 🎭 Emotion: {emotion}")
        
        tts_path = os.path.join(chunks_dir, f"tts_{i}.wav")
        tts_success = False
        
        # 4. TTS Generation — Route to selected engine
        if voice_engine == "elevenlabs":
            try:
                generate_elevenlabs_tts(translated_text, target_lang, tts_path, elevenlabs_key, emotion)
                tts_success = True
            except Exception as e:
                print(f"   [{i}] ⚠️ ElevenLabs failed: {e}, falling back to Edge TTS")
        
        elif voice_engine == "azure":
            try:
                generate_azure_tts(translated_text, target_lang, tts_path, azure_key, azure_region, emotion)
                tts_success = True
            except Exception as e:
                print(f"   [{i}] ⚠️ Azure failed: {e}, falling back to Edge TTS")
        
        elif voice_engine == "xtts" and tts_engine:
            try:
                with tts_lock:
                    success = xtts_best_of_n(tts_engine, translated_text, ref_path, target_lang, tts_path, n=2)
                if success:
                    print(f"   [{i}] ✅ XTTS (best-of-2): {translated_text[:40]}...")
                    tts_success = True
                else:
                    raise Exception("XTTS generation failed")
            except Exception as e:
                print(f"   [{i}] ⚠️ XTTS failed: {e}, falling back to Edge TTS")
        
        # Default / fallback: Edge TTS (plain text — no SSML)
        if not tts_success:
            voice = VOICE_MAPPING.get(target_lang, 'en-US-JennyNeural')
            try:
                asyncio.run(generate_edge_tts(translated_text, voice, tts_path))
                print(f"   [{i}] ✅ Edge TTS: {translated_text[:40]}...")
            except:
                gTTS(text=translated_text, lang=target_lang, slow=False).save(tts_path)
        
        # 5. Prosody transfer (match original pitch/energy)
        #    SKIP for XTTS — voice clone already captures speaker characteristics
        prosody_path = os.path.join(chunks_dir, f"prosody_{i}.wav")
        if voice_engine == "xtts":
            # For XTTS: only match energy level, skip pitch shifting
            try:
                librosa = get_librosa()
                if librosa:
                    import soundfile as sf
                    y_orig, _ = librosa.load(orig_segment_path, sr=24000)
                    y_tts, _ = librosa.load(tts_path, sr=24000)
                    rms_orig = np.sqrt(np.mean(y_orig**2))
                    rms_tts = np.sqrt(np.mean(y_tts**2))
                    if rms_tts > 0:
                        y_tts = y_tts * (rms_orig / rms_tts)
                        y_tts = np.clip(y_tts, -1.0, 1.0)
                    sf.write(prosody_path, y_tts, 24000)
                else:
                    shutil.copy2(tts_path, prosody_path)
            except:
                shutil.copy2(tts_path, prosody_path)
            print(f"   [{i}] 🎯 XTTS: energy-only match (no pitch shift)")
        else:
            transfer_prosody(orig_segment_path, tts_path, prosody_path)
        
        # 6. Add micro-pauses and breathing
        breathed_path = os.path.join(chunks_dir, f"breathed_{i}.wav")
        add_micro_pauses(prosody_path, breathed_path, emotion)
        
        # 7. Audio post-processing (compression, de-ess, EQ)
        #    SKIP for XTTS — preserve the natural cloned voice characteristics
        processed_path = os.path.join(chunks_dir, f"processed_{i}.wav")
        if voice_engine == "xtts":
            # For XTTS: just apply a gentle limiter to prevent clipping
            result = subprocess.run(
                f'ffmpeg -i "{breathed_path}" -af "alimiter=limit=0.95:attack=5:release=50" "{processed_path}" -y -loglevel error',
                shell=True, capture_output=True
            )
            if result.returncode != 0:
                shutil.copy2(breathed_path, processed_path)
            print(f"   [{i}] 🎵 XTTS: gentle limiter only (no heavy post-processing)")
        else:
            post_process_audio(breathed_path, processed_path)
        
        # 8. Time stretching — fit dubbed audio into original timing slot
        #    Use gentler range to avoid artifacts (esp. for XTTS cloned voice)
        actual_dur = get_audio_duration(processed_path)
        if actual_dur == 0: actual_dur = 0.1
        speed_ratio = actual_dur / duration
        speed_ratio = max(0.7, min(speed_ratio, 2.0))  # Gentler range: 0.7x-2.0x
        synced_path = os.path.join(chunks_dir, f"synced_{i}.wav")
        
        print(f"   [{i}] ⏱️ Time sync: {actual_dur:.1f}s TTS → {duration:.1f}s slot (tempo={speed_ratio:.2f}x)")
        
        rb_result = subprocess.run(
            f'ffmpeg -i "{processed_path}" -filter:a "rubberband=tempo={speed_ratio}:pitch=highconsistency" "{synced_path}" -y -loglevel error',
            shell=True, capture_output=True
        )
        if rb_result.returncode != 0:
            # Fallback: use atempo (preserves pitch, unlike asetrate)
            # atempo only supports 0.5-2.0, chain for larger ratios
            tempo_val = max(0.5, min(speed_ratio, 2.0))
            subprocess.run(
                f'ffmpeg -i "{processed_path}" -filter:a "atempo={tempo_val}" "{synced_path}" -y -loglevel error',
                shell=True
            )
        
        return {
            "index": i,
            "start": start,
            "end": end,
            "tts_path": synced_path,
            "emotion": emotion
        }
        
    except Exception as e:
        print(f"Error processing segment {i}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ── Room Tone Extraction ─────────────────────────────────────────────────────
def extract_room_tone(background_path, work_dir, duration=2.0):
    """Extract a short sample of room ambience for blending."""
    room_tone = os.path.join(work_dir, "room_tone.wav")
    try:
        subprocess.run(
            f'ffmpeg -i "{background_path}" -t {duration} -ac 1 -ar 24000 "{room_tone}" -y -loglevel error',
            shell=True
        )
        return room_tone
    except:
        return None

# ── API Routes ───────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "version": "v3-cinema",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if device == "cuda" else 0,
        "xtts_loaded": tts_engine is not None
    })

@app.route('/dub', methods=['POST'])
def dub_video():
    """Multi-engine video dubbing endpoint (v5)."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    target_lang = request.form.get('target_lang', 'hi')
    gemini_api_key = request.form.get('gemini_api_key', '')
    voice_engine = request.form.get('voice_engine', 'edge')
    elevenlabs_key = request.form.get('elevenlabs_key', '')
    azure_key = request.form.get('azure_key', '')
    azure_region = request.form.get('azure_region', 'eastus')
    
    work_dir = tempfile.mkdtemp()
    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir)
    
    input_path = os.path.join(work_dir, secure_filename(file.filename))
    file.save(input_path)
    engine_label = {"edge": "Edge TTS", "xtts": "XTTS Clone", "elevenlabs": "ElevenLabs", "azure": "Azure Neural"}.get(voice_engine, voice_engine)
    print(f"\n{'='*60}")
    print(f"🎬 Dub Job: {file.filename} → {LANG_NAMES.get(target_lang, target_lang)} [{engine_label}]")
    print(f"{'='*60}")
    
    try:
        # ── Step 1: Extract Audio ─────────────────────────────────────
        print("🎵 Step 1: Extracting Audio...")
        full_audio = os.path.join(work_dir, "full_source.wav")
        ffmpeg.input(input_path).output(full_audio, ac=1, ar=24000).run(
            cmd="ffmpeg", overwrite_output=True, quiet=True
        )
        total_duration = get_audio_duration(full_audio)
        
        # ── Step 2: Vocal Separation (Demucs — high quality) ──────────
        print("🎤 Step 2: Separating Vocals (Demucs HQ — shifts=5)...")
        demucs_out = os.path.join(work_dir, "demucs_out")
        cmd = ["demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", device,
               "--shifts", "5", "--overlap", "0.25", full_audio, "-o", demucs_out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        vocals_path = os.path.join(demucs_out, "htdemucs", "full_source", "vocals.wav")
        background_path = os.path.join(demucs_out, "htdemucs", "full_source", "no_vocals.wav")
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        
        # ── Step 3: Transcription (Whisper large-v3 — maximum accuracy) ─
        print("📝 Step 3: Transcribing (Whisper large-v3, beam=5)...")
        whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        segments_gen, info = whisper_model.transcribe(
            vocals_path, beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]
        print(f"   Found {len(segments)} segments (detected: {info.language})")
        del whisper_model
        gc.collect()
        
        # ── Step 4: Load XTTS (only if engine is xtts) ─────────────────
        use_xtts = (voice_engine == "xtts") and (target_lang in xtts_supported)
        if use_xtts:
            load_xtts()
        
        # ── Step 4b: Best voice reference (12s for maximum quality) ────
        print("🎯 Step 4b: Extracting best 12s voice reference...")
        master_ref = extract_best_reference(vocals_path, work_dir, target_duration=12.0)
        
        # ── Step 4c: Extract room tone for blending ───────────────────
        room_tone = extract_room_tone(background_path, work_dir)
        
        # ── Step 5: Gemini Scene Translation ──────────────────────────
        print(f"🌍 Step 5: Translating full scene with Gemini...")
        translations = translate_full_scene_with_gemini(segments, target_lang, gemini_api_key)
        
        # ── Step 6: Sequential Quality Dubbing (no parallel — full GPU per segment)
        print(f"🎬 Step 6: Dubbing {len(segments)} segments [{engine_label}] SEQUENTIALLY...")
        
        results = []
        for i, seg in enumerate(segments):
            print(f"\n   ── Segment {i+1}/{len(segments)} [{engine_label}] ──")
            translated_text = translations[i] if i < len(translations) else seg['text']
            result = process_segment(
                i, seg, chunks_dir, vocals_path, master_ref, target_lang, translated_text, use_xtts,
                voice_engine=voice_engine, elevenlabs_key=elevenlabs_key,
                azure_key=azure_key, azure_region=azure_region
            )
            if result:
                results.append(result)
            # Clear GPU cache between segments for best quality
            if device == "cuda":
                torch.cuda.empty_cache()
        
        results.sort(key=lambda x: x['index'])
        
        # ── Step 7: Merge with intelligent gaps ───────────────────────
        print("🔧 Step 7: Merging with micro-pauses...")
        ordered_files = []
        current_time = 0.0
        
        for res in results:
            start = res['start']
            end = res['end']
            tts_path = res['tts_path']
            
            gap_dur = start - current_time
            if gap_dur > 0.1:
                gap_path = os.path.join(chunks_dir, f"gap_{res['index']}.wav")
                # Use room tone instead of silence for natural gaps
                if room_tone and os.path.exists(room_tone):
                    # Loop room tone to fill gap
                    subprocess.run(
                        f'ffmpeg -stream_loop -1 -i "{room_tone}" -t {gap_dur} -ac 1 -ar 24000 "{gap_path}" -y -loglevel error',
                        shell=True
                    )
                else:
                    subprocess.run(f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap_dur} "{gap_path}" -y -loglevel error', shell=True)
                ordered_files.append(gap_path)
            
            ordered_files.append(tts_path)
            current_time = end
        
        # Trailing gap
        if current_time < total_duration:
            gap_end = os.path.join(chunks_dir, "gap_end.wav")
            if room_tone and os.path.exists(room_tone):
                subprocess.run(
                    f'ffmpeg -stream_loop -1 -i "{room_tone}" -t {total_duration - current_time} -ac 1 -ar 24000 "{gap_end}" -y -loglevel error',
                    shell=True
                )
            else:
                subprocess.run(f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {total_duration - current_time} "{gap_end}" -y -loglevel error', shell=True)
            ordered_files.append(gap_end)
        
        # Concat all segments
        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in ordered_files:
                f.write(f"file '{p}'\n")
        
        full_tts = os.path.join(work_dir, "full_tts.wav")
        subprocess.run(f'ffmpeg -f concat -safe 0 -i "{concat_list}" -c copy "{full_tts}" -y -loglevel error', shell=True)
        
        # ── Step 8: Two-Pass Loudnorm + Professional Mixing ────────────
        print("🎚️ Step 8: Two-pass loudnorm + professional mixing...")
        
        # Pass 1: Measure loudness stats
        loudnorm_pass1 = os.path.join(work_dir, "loudnorm_stats.txt")
        measure_result = subprocess.run(
            f'ffmpeg -i "{full_tts}" -af "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json" -f null -',
            shell=True, capture_output=True, text=True
        )
        
        # Parse loudnorm stats for pass 2
        measured_I = "-16"
        measured_TP = "-1.5"
        measured_LRA = "11"
        measured_thresh = "-27"
        measured_offset = "0"
        if measure_result.stderr:
            import re
            stderr_text = measure_result.stderr
            for param, pattern in [
                ("measured_I", r'"input_i"\s*:\s*"([^"]+)"'),
                ("measured_TP", r'"input_tp"\s*:\s*"([^"]+)"'),
                ("measured_LRA", r'"input_lra"\s*:\s*"([^"]+)"'),
                ("measured_thresh", r'"input_thresh"\s*:\s*"([^"]+)"'),
                ("measured_offset", r'"target_offset"\s*:\s*"([^"]+)"'),
            ]:
                match = re.search(pattern, stderr_text)
                if match:
                    if param == "measured_I": measured_I = match.group(1)
                    elif param == "measured_TP": measured_TP = match.group(1)
                    elif param == "measured_LRA": measured_LRA = match.group(1)
                    elif param == "measured_thresh": measured_thresh = match.group(1)
                    elif param == "measured_offset": measured_offset = match.group(1)
        
        # Pass 2: Apply precise loudnorm with measured values
        normalized_tts = os.path.join(work_dir, "normalized_tts.wav")
        subprocess.run(
            f'ffmpeg -i "{full_tts}" -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={measured_I}:measured_TP={measured_TP}:measured_LRA={measured_LRA}:measured_thresh={measured_thresh}:offset={measured_offset}:linear=true" "{normalized_tts}" -y -loglevel error',
            shell=True
        )
        if not os.path.exists(normalized_tts):
            normalized_tts = full_tts  # Fallback
        
        # Mix: Normalized voice + ducked background
        final_audio = os.path.join(work_dir, "final_mixed.wav")
        mix_filter = (
            "[1]volume=0.20[bg];"
            "[0][bg]amix=inputs=2:duration=first:weights=1 0.7"
        )
        subprocess.run(
            f'ffmpeg -i "{normalized_tts}" -i "{background_path}" -filter_complex "{mix_filter}" "{final_audio}" -y -loglevel error',
            shell=True
        )
        
        # ── Step 9: Final Video ───────────────────────────────────────
        output_video = os.path.join(work_dir, "dubbed_output.mp4")
        subprocess.run(f'ffmpeg -i "{input_path}" -i "{final_audio}" -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental "{output_video}" -y -loglevel error', shell=True)
        
        # Print emotion summary
        emotions = [r.get('emotion', 'neutral') for r in results]
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        print(f"\n🎭 Emotion breakdown: {emotion_counts}")
        print(f"🎉 CINEMA DUB COMPLETE! Sending back dubbed video...")
        
        return send_file(output_video, mimetype='video/mp4', as_attachment=True, download_name=f"dubbed_{target_lang}_{voice_engine}.mp4")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        pass

# ── Preview Endpoint — Fast sentence-based dubbing ───────────────────────────
PREVIEW_MAX_SEGMENTS = 8  # Process first N sentences for preview

@app.route('/dub-preview', methods=['POST'])
def dub_preview():
    """Fast preview dubbing: first N sentences only, lightweight pipeline.
    Skips: Demucs, prosody matching, rubberband, post-processing.
    Uses: Whisper base model, Edge TTS (fastest), simple mixing.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    target_lang = request.form.get('target_lang', 'hi')
    gemini_api_key = request.form.get('gemini_api_key', '')
    
    work_dir = tempfile.mkdtemp()
    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir)
    
    input_path = os.path.join(work_dir, secure_filename(file.filename))
    file.save(input_path)
    
    print(f"\n{'='*60}")
    print(f"⚡ PREVIEW Job: {file.filename} → {LANG_NAMES.get(target_lang, target_lang)}")
    print(f"   Mode: Fast (first {PREVIEW_MAX_SEGMENTS} sentences, no Demucs/prosody)")
    print(f"{'='*60}")
    
    try:
        # ── Step 1: Extract Audio (same as full) ──────────────────────
        print("⚡ Step 1: Extracting Audio...")
        full_audio = os.path.join(work_dir, "full_source.wav")
        ffmpeg.input(input_path).output(full_audio, ac=1, ar=24000).run(
            cmd="ffmpeg", overwrite_output=True, quiet=True
        )
        total_duration = get_audio_duration(full_audio)
        
        # ── Step 2: SKIP Demucs — use raw audio as "vocals" ───────────
        print("⚡ Step 2: Skipping vocal separation (preview mode)...")
        vocals_path = full_audio  # Use raw audio directly
        
        # ── Step 3: Fast Transcription (Whisper base, beam=1) ─────────
        print("⚡ Step 3: Transcribing (Whisper base, beam=1 — fast)...")
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        segments_gen, info = whisper_model.transcribe(
            full_audio, beam_size=1, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        all_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]
        print(f"   Found {len(all_segments)} total segments (detected: {info.language})")
        del whisper_model
        gc.collect()
        
        # ── Take only first N segments for preview ────────────────────
        preview_segments = all_segments[:PREVIEW_MAX_SEGMENTS]
        preview_end_time = preview_segments[-1]["end"] if preview_segments else 0
        print(f"   ⚡ Preview: Processing {len(preview_segments)} sentences (~{preview_end_time:.1f}s)")
        
        # ── Step 4: Translate only preview segments ───────────────────
        print(f"⚡ Step 4: Translating {len(preview_segments)} sentences...")
        translations = translate_full_scene_with_gemini(preview_segments, target_lang, gemini_api_key)
        
        # ── Step 5: Fast TTS (Edge TTS only, no XTTS) ────────────────
        print(f"⚡ Step 5: Generating voice (Edge TTS — fast)...")
        
        results = []
        for i, seg in enumerate(preview_segments):
            translated_text = translations[i] if i < len(translations) else seg['text']
            
            tts_path = os.path.join(chunks_dir, f"tts_{i}.wav")
            voice = VOICE_MAPPING.get(target_lang, 'en-US-JennyNeural')
            
            try:
                asyncio.run(generate_edge_tts(translated_text, voice, tts_path))
                print(f"   [{i}] ✅ {translated_text[:50]}...")
            except:
                try:
                    gTTS(text=translated_text, lang=target_lang, slow=False).save(tts_path)
                except Exception as e:
                    print(f"   [{i}] ❌ TTS failed: {e}")
                    continue
            
            if os.path.exists(tts_path):
                results.append({
                    "index": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "tts_path": tts_path
                })
        
        results.sort(key=lambda x: x['index'])
        
        # ── Step 6: Simple merge (no room tone, no loudnorm) ──────────
        print("⚡ Step 6: Merging preview audio...")
        
        ordered_files = []
        current_time = 0.0
        
        for res in results:
            gap_dur = res['start'] - current_time
            if gap_dur > 0.05:
                gap_path = os.path.join(chunks_dir, f"gap_{res['index']}.wav")
                subprocess.run(
                    f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap_dur} "{gap_path}" -y -loglevel error',
                    shell=True
                )
                ordered_files.append(gap_path)
            
            ordered_files.append(res['tts_path'])
            current_time = res['end']
        
        # Add silence for the rest of the video (after preview segments)
        remaining_duration = total_duration - current_time
        if remaining_duration > 0:
            gap_end = os.path.join(chunks_dir, "gap_end.wav")
            subprocess.run(
                f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {remaining_duration} "{gap_end}" -y -loglevel error',
                shell=True
            )
            ordered_files.append(gap_end)
        
        if not ordered_files:
            return jsonify({"error": "No segments were processed"}), 500
        
        # Concat preview audio
        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in ordered_files:
                f.write(f"file '{p}'\n")
        
        preview_tts = os.path.join(work_dir, "preview_tts.wav")
        subprocess.run(
            f'ffmpeg -f concat -safe 0 -i "{concat_list}" -c copy "{preview_tts}" -y -loglevel error',
            shell=True
        )
        
        # ── Step 7: Mix preview audio with original audio ─────────────
        # For the preview portion: use dubbed voice
        # For the rest: keep original audio
        print("⚡ Step 7: Mixing preview with original audio...")
        
        # Extract original audio for the non-preview portion
        original_audio = os.path.join(work_dir, "original_audio.wav")
        ffmpeg.input(input_path).output(original_audio, ac=1, ar=24000).run(
            cmd="ffmpeg", overwrite_output=True, quiet=True
        )
        
        # Mix: preview dubbed audio fades into original audio after preview_end_time
        # Use amix with the preview TTS audio taking priority for the dubbed portion
        final_audio = os.path.join(work_dir, "final_preview_audio.wav")
        
        if preview_end_time < total_duration:
            # Create a mixed audio: dubbed for first N sentences, original for the rest
            # The preview_tts has silence after the dubbed portion, so we mix:
            # - High volume preview TTS for the preview portion
            # - Original audio for the rest
            mix_filter = (
                f"[0]volume=1.0[dubbed];"
                f"[1]volume=0.15,afade=t=in:st={max(0, preview_end_time - 0.5)}:d=1.0[orig];"
                f"[dubbed][orig]amix=inputs=2:duration=longest:weights=1 0.5"
            )
            subprocess.run(
                f'ffmpeg -i "{preview_tts}" -i "{original_audio}" -filter_complex "{mix_filter}" "{final_audio}" -y -loglevel error',
                shell=True
            )
        else:
            # All segments were in preview — just use the dubbed audio
            shutil.copy2(preview_tts, final_audio)
        
        if not os.path.exists(final_audio):
            final_audio = preview_tts  # Fallback
        
        # ── Step 8: Create preview video ──────────────────────────────
        print("⚡ Step 8: Encoding preview video...")
        output_video = os.path.join(work_dir, "preview_output.mp4")
        subprocess.run(
            f'ffmpeg -i "{input_path}" -i "{final_audio}" -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental "{output_video}" -y -loglevel error',
            shell=True
        )
        
        print(f"\n⚡ PREVIEW COMPLETE! {len(results)} sentences dubbed (~{preview_end_time:.1f}s)")
        print(f"   Total video: {total_duration:.1f}s | Preview covers: {preview_end_time:.1f}s")
        
        return send_file(
            output_video, mimetype='video/mp4', as_attachment=True,
            download_name=f"preview_{target_lang}.mp4"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        pass


# ── Compare Endpoint — Dubs with ALL engines and returns URLs ─────────────────
@app.route('/dub-compare', methods=['POST'])
def dub_compare():
    """Dub the same video with all available engines for side-by-side comparison.
    Returns JSON with paths to each dubbed video."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    target_lang = request.form.get('target_lang', 'hi')
    gemini_api_key = request.form.get('gemini_api_key', '')
    elevenlabs_key = request.form.get('elevenlabs_key', '')
    azure_key = request.form.get('azure_key', '')
    azure_region = request.form.get('azure_region', 'eastus')
    
    compare_dir = tempfile.mkdtemp()
    
    # Save input file once
    input_path = os.path.join(compare_dir, secure_filename(file.filename))
    file.save(input_path)
    
    print(f"\n{'='*60}")
    print(f"🔬 COMPARE MODE: {file.filename} → {LANG_NAMES.get(target_lang, target_lang)}")
    print(f"{'='*60}")
    
    # ── Shared preprocessing (do ONCE) ─────────────────────────────
    try:
        # Step 1: Extract Audio
        print("🎵 [Shared] Extracting Audio...")
        full_audio = os.path.join(compare_dir, "full_source.wav")
        ffmpeg.input(input_path).output(full_audio, ac=1, ar=24000).run(
            cmd="ffmpeg", overwrite_output=True, quiet=True
        )
        total_duration = get_audio_duration(full_audio)
        
        # Step 2: Vocal Separation
        print("🎤 [Shared] Separating Vocals (Demucs HQ)...")
        demucs_out = os.path.join(compare_dir, "demucs_out")
        cmd = ["demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", device,
               "--shifts", "5", "--overlap", "0.25", full_audio, "-o", demucs_out]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        vocals_path = os.path.join(demucs_out, "htdemucs", "full_source", "vocals.wav")
        background_path = os.path.join(demucs_out, "htdemucs", "full_source", "no_vocals.wav")
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        
        # Step 3: Transcription
        print("📝 [Shared] Transcribing (Whisper large-v3)...")
        whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        segments_gen, info = whisper_model.transcribe(
            vocals_path, beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]
        print(f"   Found {len(segments)} segments (detected: {info.language})")
        del whisper_model
        gc.collect()
        
        # Step 4: Voice reference
        master_ref = extract_best_reference(vocals_path, compare_dir, target_duration=12.0)
        room_tone = extract_room_tone(background_path, compare_dir)
        
        # Step 5: Translation
        print(f"🌍 [Shared] Translating full scene with Gemini...")
        translations = translate_full_scene_with_gemini(segments, target_lang, gemini_api_key)
        
        # ── Now dub with each engine ──────────────────────────────────
        engines_to_try = ["edge"]  # Always include Edge (free)
        
        # Check XTTS availability
        if target_lang in xtts_supported:
            load_xtts()
            engines_to_try.append("xtts")
        
        # Check ElevenLabs
        if elevenlabs_key:
            engines_to_try.append("elevenlabs")
        else:
            print("   ℹ️ No ElevenLabs key — skipping ElevenLabs")
        
        # Check Azure
        if azure_key:
            engines_to_try.append("azure")
        else:
            print("   ℹ️ No Azure key — skipping Azure")
        
        results_map = {}
        
        for engine in engines_to_try:
            engine_label = {"edge": "Edge TTS", "xtts": "XTTS Clone", "elevenlabs": "ElevenLabs", "azure": "Azure Neural"}[engine]
            print(f"\n{'─'*40}")
            print(f"🎙️ Dubbing with {engine_label}...")
            print(f"{'─'*40}")
            
            engine_dir = os.path.join(compare_dir, f"engine_{engine}")
            engine_chunks = os.path.join(engine_dir, "chunks")
            os.makedirs(engine_chunks, exist_ok=True)
            
            use_xtts = (engine == "xtts")
            
            engine_results = []
            for i, seg in enumerate(segments):
                translated_text = translations[i] if i < len(translations) else seg['text']
                result = process_segment(
                    i, seg, engine_chunks, vocals_path, master_ref, target_lang, translated_text, use_xtts,
                    voice_engine=engine, elevenlabs_key=elevenlabs_key,
                    azure_key=azure_key, azure_region=azure_region
                )
                if result:
                    engine_results.append(result)
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            engine_results.sort(key=lambda x: x['index'])
            
            # Merge segments for this engine
            ordered_files = []
            current_time = 0.0
            for res in engine_results:
                gap_dur = res['start'] - current_time
                if gap_dur > 0.1:
                    gap_path = os.path.join(engine_chunks, f"gap_{res['index']}.wav")
                    if room_tone and os.path.exists(room_tone):
                        subprocess.run(f'ffmpeg -stream_loop -1 -i "{room_tone}" -t {gap_dur} -ac 1 -ar 24000 "{gap_path}" -y -loglevel error', shell=True)
                    else:
                        subprocess.run(f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap_dur} "{gap_path}" -y -loglevel error', shell=True)
                    ordered_files.append(gap_path)
                ordered_files.append(res['tts_path'])
                current_time = res['end']
            
            if current_time < total_duration:
                gap_end = os.path.join(engine_chunks, "gap_end.wav")
                if room_tone and os.path.exists(room_tone):
                    subprocess.run(f'ffmpeg -stream_loop -1 -i "{room_tone}" -t {total_duration - current_time} -ac 1 -ar 24000 "{gap_end}" -y -loglevel error', shell=True)
                else:
                    subprocess.run(f'ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {total_duration - current_time} "{gap_end}" -y -loglevel error', shell=True)
                ordered_files.append(gap_end)
            
            # Concat
            concat_list = os.path.join(engine_dir, "concat.txt")
            with open(concat_list, "w") as f:
                for p in ordered_files:
                    f.write(f"file '{p}'\n")
            
            full_tts = os.path.join(engine_dir, "full_tts.wav")
            subprocess.run(f'ffmpeg -f concat -safe 0 -i "{concat_list}" -c copy "{full_tts}" -y -loglevel error', shell=True)
            
            # Loudnorm + mix
            normalized_tts = os.path.join(engine_dir, "normalized_tts.wav")
            measure_result = subprocess.run(
                f'ffmpeg -i "{full_tts}" -af "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json" -f null -',
                shell=True, capture_output=True, text=True
            )
            measured_I, measured_TP, measured_LRA, measured_thresh, measured_offset = "-16", "-1.5", "11", "-27", "0"
            if measure_result.stderr:
                import re
                for param, pattern in [
                    ("I", r'"input_i"\s*:\s*"([^"]+)"'), ("TP", r'"input_tp"\s*:\s*"([^"]+)"'),
                    ("LRA", r'"input_lra"\s*:\s*"([^"]+)"'), ("T", r'"input_thresh"\s*:\s*"([^"]+)"'),
                    ("O", r'"target_offset"\s*:\s*"([^"]+)"')
                ]:
                    m = re.search(pattern, measure_result.stderr)
                    if m:
                        if param == "I": measured_I = m.group(1)
                        elif param == "TP": measured_TP = m.group(1)
                        elif param == "LRA": measured_LRA = m.group(1)
                        elif param == "T": measured_thresh = m.group(1)
                        elif param == "O": measured_offset = m.group(1)
            
            subprocess.run(
                f'ffmpeg -i "{full_tts}" -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={measured_I}:measured_TP={measured_TP}:measured_LRA={measured_LRA}:measured_thresh={measured_thresh}:offset={measured_offset}:linear=true" "{normalized_tts}" -y -loglevel error',
                shell=True
            )
            if not os.path.exists(normalized_tts):
                normalized_tts = full_tts
            
            final_audio = os.path.join(engine_dir, "final_mixed.wav")
            mix_filter = '[1]volume=0.20[bg];[0][bg]amix=inputs=2:duration=first:weights=1 0.7'
            subprocess.run(
                f'ffmpeg -i "{normalized_tts}" -i "{background_path}" -filter_complex "{mix_filter}" "{final_audio}" -y -loglevel error',
                shell=True
            )
            
            # Final video
            output_video = os.path.join(engine_dir, f"dubbed_{engine}.mp4")
            subprocess.run(
                f'ffmpeg -i "{input_path}" -i "{final_audio}" -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental "{output_video}" -y -loglevel error',
                shell=True
            )
            
            if os.path.exists(output_video):
                results_map[engine] = output_video
                print(f"   ✅ {engine_label} dub complete!")
            else:
                print(f"   ❌ {engine_label} dub failed!")
        
        print(f"\n🔬 COMPARISON COMPLETE! {len(results_map)} engines produced output.")
        
        # Return all videos as downloadable files via a results endpoint
        # Save paths for retrieval
        results_json = {}
        for engine, path in results_map.items():
            # Copy to a served directory
            served_name = f"compare_{engine}_{target_lang}.mp4"
            served_path = os.path.join(UPLOAD_FOLDER, served_name)
            shutil.copy2(path, served_path)
            results_json[engine] = served_name
        
        return jsonify({
            "success": True,
            "engines": results_json,
            "segments_count": len(segments),
            "duration": total_duration
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Serve comparison files ────────────────────────────────────────────────────
@app.route('/compare-result/<filename>', methods=['GET'])
def serve_compare_result(filename):
    """Serve a comparison result video file."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='video/mp4')
    return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI Dubbing Cloud API (v5.4 - Preview Mode + Multi-Engine)")
    print(f"   Device: {device}")
    print("   Engines: Edge TTS | XTTS Clone | ElevenLabs | Azure Neural")
    print("   Whisper: large-v3 (full) | base (preview)")
    print("   Endpoints: POST /dub | POST /dub-preview | POST /dub-compare")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5050, debug=False)

