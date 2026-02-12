import argparse
import sys
import os
import time
import json
import torch
import ffmpeg
import math
import subprocess
from deep_translator import GoogleTranslator
from TTS.api import TTS
# import whisper  <-- Removed
import google.generativeai as genai
from faster_whisper import WhisperModel
import shutil
import functools
import asyncio
import edge_tts

# High Quality Microsoft Edge Voices (Default: Female)
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

# Parse args first to set up logging? No, keep standard flow.
# Ensure output is unbuffered
print = functools.partial(print, flush=True)

def update_status(progress, stage, data=None):
    message = {
        "progress": progress,
        "stage": stage
    }
    if data:
        message.update(data)
    print(json.dumps(message))

def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except:
        return 0

def create_m3u8(output_dir, segments, target_duration=10):
    """Generate the HLS playlist file."""
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{target_duration + 5}", # Allow some buffer
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:EVENT" # Event means we append to it
    ]
    
    for seg_file, duration in segments:
        lines.append(f"#EXTINF:{duration:.6f},")
        lines.append(seg_file)
    
    playlist_path = os.path.join(output_dir, "playlist.m3u8")
    with open(playlist_path, "w") as f:
        f.write("\n".join(lines))

import gc

async def generate_edge_tts(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

# RVC Integration
def apply_rvc(input_audio_path):
    """
    Applies RVC extraction if a model is found in models/rvc.
    Returns path to converted audio, or original if no model found.
    """
    rvc_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "rvc")
    if not os.path.exists(rvc_models_dir): return input_audio_path
    
    # Find first .pth file
    models = [f for f in os.listdir(rvc_models_dir) if f.endswith(".pth")]
    if not models: return input_audio_path
    
    model_name = models[0]
    model_path = os.path.join(rvc_models_dir, model_name)
    output_path = input_audio_path.replace(".wav", "_rvc.wav")
    
    # Index file (optional)
    index_path = None
    indices = [f for f in os.listdir(rvc_models_dir) if f.endswith(".index")]
    if indices: index_path = os.path.join(rvc_models_dir, indices[0])
    
    print(f"🎤 Applying RVC ({model_name})...")
    
    try:
        from rvc_python.infer import RVCInference
        rvc = RVCInference(device="cuda:0" if torch.cuda.is_available() else "cpu")
        rvc.load_model(model_path)
        rvc.infer_file(input_audio_path, output_path, index_path=index_path)
        return output_path
    except ImportError:
        print("⚠️ rvc-python not installed. Skipping RVC.")
        return input_audio_path
    except Exception as e:
        print(f"⚠️ RVC Failed: {e}")
        return input_audio_path

def polish_text_with_gemini(text, target_language, api_key):
    if not text or len(text.strip()) < 2: return text
    
    try:
        genai.configure(api_key=api_key)
        
        # PROMPT
        prompt = f'''Refine this text for a video dubbing script in {target_language}.
Rules:
1. Correct grammar/syntax.
2. Make it sound natural and conversational.
3. Do NOT change meaning.
4. Output ONLY the polished text.

Input: "{text}"'''

        model = None
        response = None

        # Attempt 1: Gemini 1.5 Flash
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
        except Exception as e1:
            print(f"⚠️ gemini-1.5-flash failed, trying gemini-pro...")
            try:
                # Attempt 2: Gemini Pro
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
            except Exception as e2:
                print(f"⚠️ gemini-pro also failed: {e2}")
                return text

        if response and response.text:
            cleaned = response.text.strip().replace('"', '').replace('\n', ' ')
            # simplistic check to ensure it didn't return a huge explanation
            if len(cleaned) < len(text) * 3:
                # print(f"✨ Gemini Polished: '{text}' -> '{cleaned}'")
                return cleaned
        return text
    except Exception as e:
        print(f"⚠️ Gemini System Error: {e}")
        return text

def run_worker_process(args):
    """Worker: Repurpose process_video_turbo to handle a specific slice."""
    # Create worker-specific work dir
    work_dir = args.output_path.replace(".mp4", f"_work_w{args.worker_id}")
    
    if os.path.exists(work_dir): shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    
    slice_audio_path = os.path.join(work_dir, "slice_source.wav")
    duration = args.end_time - args.start_time
    
    # Extract just this slice of audio
    subprocess.run(
        f'ffmpeg -i "{args.input_video}" -ss {args.start_time} -t {duration} -ac 1 -ar 24000 "{slice_audio_path}" -y -loglevel error',
        shell=True, check=True
    )
    
    target_output = args.output_path.replace(".mp4", f"_part{args.worker_id}.wav")
    
    # Pass gemini key if present
    final = process_video_turbo(
        slice_audio_path, 
        args.target_lang, 
        work_dir, 
        target_output, 
        audio_only=True, 
        gemini_api_key=args.gemini_api_key
    )
    print(f"Worker {args.worker_id} finished: {final}")

def run_master_process(args):
    """Master: Spawns workers and merges results."""
    
    update_status(0, "Initializing Parallel Turbo Mode (Master)...")
    
    # 1. Get Duration
    duration = get_audio_duration(args.input_video)
    if duration == 0: duration = 10 # Fallback
    
    # Reducing to 1 worker for stability (prevents VRAM OOM on consumer GPUs)
    num_workers = 1 
    chunk_len = duration / num_workers
    
    processes = []
    
    update_status(5, f"Spawning {num_workers} Worker (Safe Turbo Mode)...")
    
    script_path = os.path.abspath(__file__)
    
    for i in range(num_workers):
        # Stagger start to avoid VRAM spike
        if i > 0: time.sleep(3)
        
        start = i * chunk_len
        end = (i + 1) * chunk_len
        if i == num_workers - 1: end = duration + 1 # Ensure coverage
        
        cmd = [
            sys.executable, script_path,
            "--input_video", args.input_video,
            "--target_lang", args.target_lang,
            "--output_path", args.output_path,
            "--worker_id", str(i),
            "--start_time", str(start),
            "--end_time", str(end)
        ]
        
        if args.gemini_api_key:
            cmd.extend(["--gemini_api_key", args.gemini_api_key])
        
        # New console for workers? No, inherit for logs.
        p = subprocess.Popen(cmd)
        processes.append(p)
        
    # Wait for all
    for i, p in enumerate(processes):
        p.wait()
        if p.returncode != 0:
             print(f"Worker {i} failed!")
             sys.exit(1)
             
    update_status(90, "Merging Worker Audio Chunks...")
    
    # Collect parts
    concat_list_path = args.output_path.replace(".mp4", "_concat.txt")
    with open(concat_list_path, "w") as f:
        for i in range(num_workers):
            part_path = args.output_path.replace(".mp4", f"_part{i}.wav")
            f.write(f"file '{part_path}'\n")
            
    final_combined_audio = args.output_path.replace(".mp4", "_final_combined.wav")
    subprocess.run(
        f'ffmpeg -f concat -safe 0 -i "{concat_list_path}" -c copy "{final_combined_audio}" -y -loglevel error',
        shell=True, check=True
    )
    
    # Merge with Original Video
    update_status(95, "Finalizing Video...")
    subprocess.run(
        f'ffmpeg -i "{args.input_video}" -i "{final_combined_audio}" -map 0:v -map 1:a -c:v copy -c:a aac -strict experimental "{args.output_path}" -y -loglevel error',
        shell=True, check=True
    )
    
    update_status(100, "Completed", {
        "result": {
            "finalVideo": os.path.basename(args.output_path)
        }
    })

# Modified process_video_turbo to support Audio Only return and cleanup
def process_video_turbo(input_path, target_lang, work_dir, final_output_path, audio_only=False, gemini_api_key=None):
    # Check if input is WAV or Video to allow skipping extraction
    is_wav = input_path.lower().endswith(".wav")
    
    try:
        # Cleanup and Setup
        # If worker mode (audio_only), we must NOT delete work_dir because it contains our input!
        if not audio_only and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        
        chunks_dir = os.path.join(work_dir, "chunks")
        if os.path.exists(chunks_dir): shutil.rmtree(chunks_dir)
        os.makedirs(chunks_dir, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Worker log prefix
        log_prefix = f"[Worker]" if audio_only else "[Main]"
        print(f"{log_prefix} Initializing on {device}...")

        # --- Step 1: Extract Audio (Skip if likely WAV input from worker) ---
        full_audio_path = os.path.join(work_dir, "full_source.wav")
        
        if is_wav:
             shutil.copy(input_path, full_audio_path)
        else:
            (
                ffmpeg
                .input(input_path)
                .output(full_audio_path, ac=1, ar=24000) 
                .run(cmd="ffmpeg", overwrite_output=True, quiet=True)
            )
        
        total_duration = get_audio_duration(full_audio_path)

        # --- Step 2: Vocal Separation (Demucs) ---
        print(f"{log_prefix} Separating Vocals...")
        demucs_out = os.path.join(work_dir, "demucs_out")
        
        # Optimization: --shifts 0, --overlap 0.1 for speed
        cmd = [
            "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", device,
            "--shifts", "0", "--overlap", "0.1",
            full_audio_path, "-o", demucs_out
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"{log_prefix} ⚠️ Demucs CUDA failed (VRAM/Driver issue), switching to CPU...")
            # Fallback to CPU
            cmd[cmd.index("-d") + 1] = "cpu"
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        demucs_source_folder = os.path.join(demucs_out, "htdemucs", "full_source")
        vocals_path = os.path.join(demucs_source_folder, "vocals.wav")
        background_path = os.path.join(demucs_source_folder, "no_vocals.wav")
        
        if not os.path.exists(vocals_path): raise Exception("Demucs failed")

        # Cleanup VRAM after Demucs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # --- Step 3: Transcription ---
        # Switch to CPU for stability 
        print(f"{log_prefix} Transcribing (CPU)...")
        
        compute_type = "int8"
        try:
            model = WhisperModel("small", device="cpu", compute_type=compute_type)
        except:
             model = WhisperModel("tiny", device="cpu", compute_type=compute_type)
             
        # Use VAD
        segments_generator, info = model.transcribe(
            vocals_path, 
            beam_size=1, 
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_generator]
        print(f"{log_prefix} Found {len(segments)} segments.")
        
        # Cleanup RAM
        del model
        gc.collect()

        # --- Step 4: Initialize TTS ---
        print(f"{log_prefix} Loading TTS...")
        xtts_supported = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
        use_xtts = target_lang in xtts_supported
        
        tts_engine = None
        if use_xtts:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            # Redirect stdout to suppress logs
            sys.stdout = open(os.devnull, 'w')
            with torch.serialization.safe_globals([XttsConfig, BaseDatasetConfig, BaseAudioConfig, XttsAudioConfig, XttsArgs]):
                 tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            sys.stdout = sys.__stdout__ # Restore
        else:
             from gtts import gTTS

        # --- Step 5: Parallel Pipeline Loop ---
        print(f"{log_prefix} Dubbing Loop... (Gemini: {'Active' if gemini_api_key else 'Inactive'})")
        
        # A. Find Master Reference (Longest Speech Segment)
        longest_seg = max(segments, key=lambda s: s["end"] - s["start"]) if segments else None
        master_ref_path = os.path.join(chunks_dir, "master_ref.wav")
        
        if longest_seg:
            dur = longest_seg["end"] - longest_seg["start"]
            if dur > 3:
                # Extract reliable reference
                subprocess.run(f'ffmpeg -i "{vocals_path}" -ss {longest_seg["start"]} -t {dur} -ac 1 -ar 24000 "{master_ref_path}" -y -loglevel error', shell=True)
            else:
                # Fallback if even longest is short
                master_ref_path = None
        else:
             master_ref_path = None

        from concurrent.futures import ThreadPoolExecutor
        
        temp_tts_files = [None] * len(segments) 
        executor = ThreadPoolExecutor(max_workers=4)
        futures = []

        def slice_audio(start, end, out_path):
            if os.path.exists(out_path): return
            subprocess.run(f'ffmpeg -i "{vocals_path}" -ss {start} -t {end-start} -ac 1 -ar 24000 "{out_path}" -y -loglevel error', shell=True)

        def sync_segment(idx, tts_path, target_dur):
            actual_dur = get_audio_duration(tts_path)
            if actual_dur == 0: actual_dur = 0.1
            
            # Speed Clamping to prevent artifacts
            speed_factor = actual_dur / target_dur
            speed_factor = max(0.75, min(speed_factor, 1.3)) # Clamp between 0.75x and 1.3x
            
            synced_path = os.path.join(chunks_dir, f"synced_{idx}.wav")
            # atempo filter limit is 0.5 to 2.0 usually, but quality degrades outside 0.8-1.2
            subprocess.run(f'ffmpeg -i "{tts_path}" -filter:a "atempo={speed_factor}" "{synced_path}" -y -loglevel error', shell=True)
            return idx, synced_path

        # B. TTS Loop
        current_time = 0.0
        for i, seg in enumerate(segments):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]
            duration = end - start
            
            # Determine Reference Audio
            ref_path = os.path.join(chunks_dir, f"ref_{i}.wav")
            use_master = False
            
            if duration < 2.5 and master_ref_path and os.path.exists(master_ref_path):
                 ref_path = master_ref_path
                 use_master = True
            else:
                 executor.submit(slice_audio, start, end, ref_path)
            
            # Gap
            gap_duration = start - current_time
            if gap_duration > 0.1:
                gap_path = os.path.join(chunks_dir, f"gap_{i}.wav")
                subprocess.run(f"ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap_duration} {gap_path} -y -loglevel error", shell=True)
                temp_tts_files[i] = {"gap": gap_path, "tts": None}
            else:
                 temp_tts_files[i] = {"gap": None, "tts": None}

            # Translate
            try:
                translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
                if gemini_api_key:
                    translated_text = polish_text_with_gemini(translated_text, target_lang, gemini_api_key)
            except:
                translated_text = text

            tts_chunk_path = os.path.join(chunks_dir, f"tts_{i}.wav")
            
            # Ensure ref exists before TTS
            if not use_master and not os.path.exists(ref_path):
                 # Wait briefly or synchronous slice
                 slice_audio(start, end, ref_path)

            if use_xtts:
                try:
                    # XTTS (Zero-Shot Cloning) - Tier 1 Priority
                    tts_engine.tts_to_file(text=translated_text, speaker_wav=ref_path, language=target_lang, file_path=tts_chunk_path)
                except Exception as e:
                    print(f"⚠️ XTTS Error seg {i} (VRAM/Speed issue): {e}. Falling back to Edge TTS...")
                    # FALLBACK LAYER 1: EDGE TTS
                    voice = VOICE_MAPPING.get(target_lang, "en-US-JennyNeural")
                    try:
                        asyncio.run(generate_edge_tts(translated_text, voice, tts_chunk_path))
                        tts_chunk_path = apply_rvc(tts_chunk_path) # Try RVC if available
                    except Exception as e2:
                        print(f"⚠️ Edge TTS Failed: {e2}. Falling back to gTTS...")
                        # FALLBACK LAYER 2: gTTS
                        try:
                            tts = gTTS(text=translated_text, lang=target_lang, slow=False)
                            tts.save(tts_chunk_path)
                        except:
                            pass
            else:
                # EDGE TTS INTEGRATION (Tier 1 Priority if XTTS disabled)
                voice = VOICE_MAPPING.get(target_lang, "en-US-JennyNeural") 
                if target_lang not in VOICE_MAPPING: 
                     print(f"⚠️ No specific voice for {target_lang}, using fallback.")

                try:
                    asyncio.run(generate_edge_tts(translated_text, voice, tts_chunk_path))
                    tts_chunk_path = apply_rvc(tts_chunk_path)
                except Exception as e:
                    print(f"EdgeTTS Error: {e}, falling back to gTTS")
                    # Fallback to gTTS if Edge fails
                    try:
                        tts = gTTS(text=translated_text, lang=target_lang, slow=False)
                        tts.save(tts_chunk_path)
                    except:
                        pass

            futures.append(executor.submit(sync_segment, i, tts_chunk_path, duration))
            current_time = end

        # C. Collect
        for f in futures:
            idx, path = f.result()
            if temp_tts_files[idx]: temp_tts_files[idx]["tts"] = path
            
        executor.shutdown(wait=True)
        
        ordered_files = []
        for item in temp_tts_files:
            if item and item["gap"]: ordered_files.append(item["gap"])
            if item and item["tts"]: ordered_files.append(item["tts"])
            
        if current_time < total_duration:
             gap_duration = total_duration - current_time
             gap_path = os.path.join(chunks_dir, "gap_end.wav")
             subprocess.run(f"ffmpeg -f lavfi -i anullsrc=r=24000:cl=mono -t {gap_duration} {gap_path} -y -loglevel error", shell=True)
             ordered_files.append(gap_path)

        # --- Step 6: Merge ---
        list_path = os.path.join(work_dir, "concat_list.txt")
        with open(list_path, "w") as f:
            for path_ in ordered_files:
                f.write(f"file '{path_}'\n")
        
        full_tts_path = os.path.join(work_dir, "full_tts.wav")
        subprocess.run(f'ffmpeg -f concat -safe 0 -i "{list_path}" -c copy "{full_tts_path}" -y -loglevel error', shell=True)
        
        final_audio = os.path.join(work_dir, "final_mixed.wav")
        subprocess.run(f"ffmpeg -i {full_tts_path} -i {background_path} -filter_complex amix=inputs=2:duration=first {final_audio} -y -loglevel error", shell=True)
        
        if audio_only:
             shutil.copy(final_audio, final_output_path)
             return final_output_path
        
        # Original single-video merge logic would go here if not audio_only
        return final_audio

    except Exception as e:
        import traceback
        traceback.print_exc()
        if not audio_only:
             print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--target_lang", required=True)
    parser.add_argument("--output_path", required=True)
    
    # Worker arguments
    parser.add_argument("--worker_id", type=int, default=None)
    parser.add_argument("--start_time", type=float, default=0)
    parser.add_argument("--end_time", type=float, default=0)
    
    # Optional API Key
    # Proactively adding default for user convenience
    parser.add_argument("--gemini_api_key", type=str, default="AIzaSyBp-5CcBjg6LQKMaSI8j531z-3dcFSA80M")
    
    args = parser.parse_args()

    if args.worker_id is not None:
        # I AM A WORKER
        run_worker_process(args)
    else:
        # I AM THE MASTER
        run_master_process(args)
