import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { updateJob } from "./jobs.js";

// Helper to get absolute path relative to project root
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const PYTHON_SCRIPT = path.join(PROJECT_ROOT, "python", "dubbing_service.py");

/**
 * Cloud Mode: Send video to remote GPU server for processing.
 * The cloud server runs cloud_api.py on Colab/RunPod with a T4/A100 GPU.
 */
const runCloudDubbing = async (jobId, videoPath, targetLang, geminiApiKey = null, cloudUrl, voiceEngine = "edge", elevenlabsKey = null, azureKey = null, azureRegion = "eastus") => {
    console.log(`[Job ${jobId}] ☁️ CLOUD MODE: Sending to ${cloudUrl}/dub`);

    updateJob(jobId, {
        status: "processing",
        progress: 10,
        stage: "Uploading to Cloud GPU..."
    });

    try {
        const FormData = (await import('form-data')).default;
        const { default: fetch } = await import('node-fetch');

        const form = new FormData();
        form.append('file', fs.createReadStream(path.resolve(videoPath)));
        form.append('target_lang', targetLang);
        if (geminiApiKey) form.append('gemini_api_key', geminiApiKey);
        form.append('voice_engine', voiceEngine);
        if (elevenlabsKey) form.append('elevenlabs_key', elevenlabsKey);
        if (azureKey) form.append('azure_key', azureKey);
        if (azureRegion) form.append('azure_region', azureRegion);

        updateJob(jobId, {
            status: "processing",
            progress: 15,
            stage: "Uploading to Cloud GPU..."
        });

        // Simulated progress ticker for long cloud processing
        let simProgress = 15;
        const progressStages = [
            { at: 20, stage: "Extracting audio track..." },
            { at: 28, stage: "Separating vocals with Demucs HQ (shifts=5)..." },
            { at: 40, stage: "Transcribing speech (Whisper large-v3)..." },
            { at: 50, stage: "Translating full scene with Gemini AI..." },
            { at: 58, stage: "Voice cloning — dubbing segments (XTTS best-of-2)..." },
            { at: 70, stage: "Processing emotions & prosody transfer..." },
            { at: 80, stage: "Mixing & mastering final audio..." },
            { at: 88, stage: "Encoding final video..." },
        ];
        let stageIdx = 0;

        const simInterval = setInterval(() => {
            if (simProgress >= 92) { clearInterval(simInterval); return; }
            simProgress += 0.5 + Math.random() * 0.8;

            while (stageIdx < progressStages.length && simProgress >= progressStages[stageIdx].at) {
                updateJob(jobId, {
                    status: "processing",
                    progress: Math.round(simProgress),
                    stage: progressStages[stageIdx].stage
                });
                stageIdx++;
            }
        }, 2000);


        const response = await fetch(`${cloudUrl}/dub`, {
            method: 'POST',
            body: form,
            headers: form.getHeaders ? form.getHeaders() : {},
            timeout: 3600000 // 60 minute timeout
        });

        clearInterval(simInterval);

        if (!response.ok) {
            const rawText = await response.text();
            console.error(`[Job ${jobId}] ☁️ RAW Cloud Error (${response.status}):`, rawText.substring(0, 500));
            let errorMsg = `Cloud API returned ${response.status}`;
            try { 
                const j = JSON.parse(rawText);
                if (j.error) errorMsg = j.error;
            } catch(e) {}
            throw new Error(errorMsg);
        }

        // Save the returned video
        const outputPath = videoPath.replace(path.extname(videoPath), `_dubbed_${targetLang}.mp4`);
        const buffer = Buffer.from(await response.arrayBuffer());
        fs.writeFileSync(outputPath, buffer);

        console.log(`[Job ${jobId}] ☁️ Cloud processing complete! Saved to ${outputPath}`);

        updateJob(jobId, {
            status: "completed",
            progress: 100,
            stage: "Completed",
            result: {
                finalVideo: "uploads/" + path.basename(outputPath)
            }
        });

    } catch (err) {
        console.error(`[Job ${jobId}] ☁️ Cloud Error:`, err.message);
        updateJob(jobId, {
            status: "error",
            stage: "Cloud Error: " + err.message,
            result: { error: err.message }
        });
        throw err;
    }
};

/**
 * Cloud Preview Mode: Two-phase processing.
 * Phase 1: Call /dub-preview → fast preview (first N sentences, lightweight pipeline)
 * Phase 2: Call /dub → full quality (runs in background after preview is returned)
 */
const runCloudPreview = async (jobId, videoPath, targetLang, geminiApiKey = null, cloudUrl, voiceEngine = "edge", elevenlabsKey = null, azureKey = null, azureRegion = "eastus") => {
    console.log(`[Job ${jobId}] ⚡ CLOUD PREVIEW MODE: Two-phase processing`);

    const FormData = (await import('form-data')).default;
    const { default: fetch } = await import('node-fetch');

    // ── Phase 1: Fast Preview ──────────────────────────────────────
    try {
        updateJob(jobId, {
            status: "processing",
            progress: 5,
            stage: "⚡ Preview: Uploading to Cloud GPU..."
        });

        const previewForm = new FormData();
        previewForm.append('file', fs.createReadStream(path.resolve(videoPath)));
        previewForm.append('target_lang', targetLang);
        if (geminiApiKey) previewForm.append('gemini_api_key', geminiApiKey);

        // Preview progress simulation (faster stages)
        let previewProgress = 5;
        const previewStages = [
            { at: 15, stage: "⚡ Preview: Extracting audio..." },
            { at: 30, stage: "⚡ Preview: Transcribing (fast model)..." },
            { at: 50, stage: "⚡ Preview: Translating first sentences..." },
            { at: 65, stage: "⚡ Preview: Generating voice..." },
            { at: 80, stage: "⚡ Preview: Merging audio with video..." },
        ];
        let previewStageIdx = 0;

        const previewSimInterval = setInterval(() => {
            if (previewProgress >= 90) { clearInterval(previewSimInterval); return; }
            previewProgress += 1.5 + Math.random() * 1.0;

            while (previewStageIdx < previewStages.length && previewProgress >= previewStages[previewStageIdx].at) {
                updateJob(jobId, {
                    status: "processing",
                    progress: Math.round(previewProgress),
                    stage: previewStages[previewStageIdx].stage
                });
                previewStageIdx++;
            }
        }, 1000);

        console.log(`[Job ${jobId}] ⚡ Phase 1: Calling ${cloudUrl}/dub-preview`);

        const previewResponse = await fetch(`${cloudUrl}/dub-preview`, {
            method: 'POST',
            body: previewForm,
            headers: previewForm.getHeaders ? previewForm.getHeaders() : {},
            timeout: 600000 // 10 minute timeout for preview
        });

        clearInterval(previewSimInterval);

        if (!previewResponse.ok) {
            const errorData = await previewResponse.json().catch(() => ({ error: "Preview failed" }));
            throw new Error(errorData.error || `Preview API returned ${previewResponse.status}`);
        }

        // Save preview video
        const previewOutputPath = videoPath.replace(path.extname(videoPath), `_preview_${targetLang}.mp4`);
        const previewBuffer = Buffer.from(await previewResponse.arrayBuffer());
        fs.writeFileSync(previewOutputPath, previewBuffer);

        console.log(`[Job ${jobId}] ⚡ Preview ready! Saved to ${previewOutputPath}`);

        // Update job to preview_ready
        updateJob(jobId, {
            status: "preview_ready",
            progress: 100,
            stage: "⚡ Preview Ready!",
            previewResult: {
                finalVideo: "uploads/" + path.basename(previewOutputPath)
            }
        });

    } catch (err) {
        console.error(`[Job ${jobId}] ⚡ Preview Error:`, err.message);
        updateJob(jobId, {
            status: "error",
            stage: "Preview Error: " + err.message,
            result: { error: err.message }
        });
        throw err;
    }

    // ── Phase 2: Full Processing in Background ─────────────────────
    // Don't await — this runs async while user watches preview
    runCloudFullInBackground(jobId, videoPath, targetLang, geminiApiKey, cloudUrl, voiceEngine, elevenlabsKey, azureKey, azureRegion);
};

/**
 * Background full processing after preview is delivered.
 */
const runCloudFullInBackground = async (jobId, videoPath, targetLang, geminiApiKey, cloudUrl, voiceEngine, elevenlabsKey, azureKey, azureRegion) => {
    try {
        console.log(`[Job ${jobId}] 🎬 Phase 2: Starting full processing in background...`);

        updateJob(jobId, {
            status: "full_processing",
            progress: 0,
            stage: "Full quality processing started..."
        });

        const FormData = (await import('form-data')).default;
        const { default: fetch } = await import('node-fetch');

        const fullForm = new FormData();
        fullForm.append('file', fs.createReadStream(path.resolve(videoPath)));
        fullForm.append('target_lang', targetLang);
        if (geminiApiKey) fullForm.append('gemini_api_key', geminiApiKey);
        fullForm.append('voice_engine', voiceEngine);
        if (elevenlabsKey) fullForm.append('elevenlabs_key', elevenlabsKey);
        if (azureKey) fullForm.append('azure_key', azureKey);
        if (azureRegion) fullForm.append('azure_region', azureRegion);

        // Background progress simulation
        let bgProgress = 0;
        const bgStages = [
            { at: 8, stage: "🎬 Full: Extracting audio..." },
            { at: 18, stage: "🎬 Full: Separating vocals (Demucs HQ)..." },
            { at: 35, stage: "🎬 Full: Transcribing (Whisper large-v3)..." },
            { at: 45, stage: "🎬 Full: Translating full scene..." },
            { at: 55, stage: "🎬 Full: Voice cloning all segments..." },
            { at: 70, stage: "🎬 Full: Prosody & emotion transfer..." },
            { at: 82, stage: "🎬 Full: Mixing & mastering..." },
            { at: 90, stage: "🎬 Full: Encoding final video..." },
        ];
        let bgStageIdx = 0;

        const bgSimInterval = setInterval(() => {
            if (bgProgress >= 92) { clearInterval(bgSimInterval); return; }
            bgProgress += 0.4 + Math.random() * 0.5;

            while (bgStageIdx < bgStages.length && bgProgress >= bgStages[bgStageIdx].at) {
                updateJob(jobId, {
                    status: "full_processing",
                    progress: Math.round(bgProgress),
                    stage: bgStages[bgStageIdx].stage
                });
                bgStageIdx++;
            }
        }, 2500);

        const fullResponse = await fetch(`${cloudUrl}/dub`, {
            method: 'POST',
            body: fullForm,
            headers: fullForm.getHeaders ? fullForm.getHeaders() : {},
            timeout: 3600000
        });

        clearInterval(bgSimInterval);

        if (!fullResponse.ok) {
            const errorData = await fullResponse.json().catch(() => ({ error: "Full processing failed" }));
            throw new Error(errorData.error || `Full API returned ${fullResponse.status}`);
        }

        const outputPath = videoPath.replace(path.extname(videoPath), `_dubbed_${targetLang}.mp4`);
        const buffer = Buffer.from(await fullResponse.arrayBuffer());
        fs.writeFileSync(outputPath, buffer);

        console.log(`[Job ${jobId}] 🎬 Full processing complete! Saved to ${outputPath}`);

        updateJob(jobId, {
            status: "completed",
            progress: 100,
            stage: "✅ Full quality version ready!",
            result: {
                finalVideo: "uploads/" + path.basename(outputPath)
            }
        });

    } catch (err) {
        console.error(`[Job ${jobId}] 🎬 Full Processing Error:`, err.message);
        // Don't overwrite preview — user still has it. Just mark error on full.
        updateJob(jobId, {
            status: "completed",  // Keep as completed so user keeps preview
            progress: 100,
            stage: "⚠️ Full processing failed, preview still available",
            result: { error: err.message, fallbackToPreview: true }
        });
    }
};

/**
 * Local Mode: Spawn Python process on local machine.
 */
const runLocalDubbing = (jobId, videoPath, targetLang, geminiApiKey = null, voiceEngine = "edge", elevenlabsKey = null, azureKey = null, azureRegion = "eastus", previewMode = false) => {
    return new Promise((resolve, reject) => {
        const suffix = previewMode ? `_preview_${targetLang}` : `_dubbed_${targetLang}`;
        const outputPath = videoPath.replace(path.extname(videoPath), `${suffix}.mp4`);

        console.log(`[Job ${jobId}] 💻 LOCAL MODE${previewMode ? ' (PREVIEW)' : ''}: Spawning ${PYTHON_SCRIPT}`);

        const args = [
            "-3.10",
            PYTHON_SCRIPT,
            "--input_video", path.resolve(videoPath),
            "--target_lang", targetLang,
            "--output_path", path.resolve(outputPath)
        ];

        if (geminiApiKey) {
            console.log(`[Job ${jobId}] Using Gemini API Key for Polishing ✨`);
            args.push('--gemini_api_key', geminiApiKey);
        }

        if (previewMode) {
            args.push('--preview');
        }

        const pythonProcess = spawn("py", args, {
            env: { ...process.env, PYTHONIOENCODING: "utf-8" },
            shell: false // Prevent Command Injection by avoiding shell evaluation
        });

        let stdoutBuffer = "";
        pythonProcess.stdout.on("data", (data) => {
            stdoutBuffer += data.toString();
            let newlineIndex;
            
            while ((newlineIndex = stdoutBuffer.indexOf("\n")) !== -1) {
                const line = stdoutBuffer.slice(0, newlineIndex);
                stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
                
                if (!line.trim()) continue;

                try {
                    const message = JSON.parse(line);

                    if (message.error) {
                        console.error(`[Job ${jobId}] Python Error:`, message.error);
                        updateJob(jobId, {
                            status: "error",
                            stage: "Processing Error",
                            result: { error: message.error }
                        });
                    } else if (message.progress) {
                        console.log(`[Job ${jobId}] Progress: ${message.stage} (${message.progress}%)`);
                        let updates = {
                            status: "processing",
                            progress: message.progress,
                            stage: message.stage
                        };

                        // Handle preview_ready status from Python
                        if (message.status === "preview_ready") {
                            updates.status = "preview_ready";
                            if (message.result) {
                                updates.previewResult = {
                                    finalVideo: "uploads/" + message.result.finalVideo
                                };
                            }
                        }

                        if (message.status === "full_processing") {
                            updates.status = "full_processing";
                        }

                        if (message.playlist) {
                            const hlsDirName = path.basename(outputPath).replace(path.extname(outputPath), "_hls");
                            const playlistUrl = `uploads/${hlsDirName}/${message.playlist}`;
                            updates.result = {
                                playlistUrl: playlistUrl,
                                currentSegment: message.segment
                            };
                        }

                        updateJob(jobId, updates);

                        if (message.result && message.status !== "preview_ready") {
                            const resultData = message.result;
                            let finalResult = { ...resultData };

                            if (resultData.playlist_dir && resultData.playlist_file) {
                                finalResult.playlistUrl = `uploads/${resultData.playlist_dir}/${resultData.playlist_file}`;
                            } else if (resultData.finalVideo) {
                                finalResult.finalVideo = "uploads/" + resultData.finalVideo;
                            }

                            updateJob(jobId, {
                                status: "completed",
                                progress: 100,
                                stage: "Completed",
                                result: finalResult
                            });
                        }
                    }
                } catch (e) {
                    console.log(`[Job ${jobId}] Python Log: ${line}`);
                }
            }
        });

        pythonProcess.stderr.on("data", (data) => {
            process.stderr.write(data);
        });

        pythonProcess.on("close", (code) => {
            if (code !== 0) {
                console.error(`[Job ${jobId}] Python process exited with code ${code}`);
                updateJob(jobId, {
                    status: "error",
                    stage: "Failed",
                    result: { error: "Processing failed internally" }
                });
                reject(new Error(`Python process exited with code ${code}`));
            } else {
                console.log(`[Job ${jobId}] Python process finished successfully`);
                resolve();
            }
        });
    });
};

/**
 * Main export: Routes to Cloud or Local based on CLOUD_API_URL env var.
 * Supports previewMode for two-phase processing.
 */
export const runPythonDubbing = (jobId, videoPath, targetLang, geminiApiKey = null, voiceEngine = "edge", elevenlabsKey = null, azureKey = null, azureRegion = "eastus", previewMode = false) => {
    // Check CLOUD_API_URL at runtime (after dotenv loads)
    const CLOUD_API_URL = process.env.CLOUD_API_URL || null;

    if (CLOUD_API_URL) {
        console.log(`[Job ${jobId}] 🌐 Cloud API detected: ${CLOUD_API_URL}`);
        if (previewMode) {
            console.log(`[Job ${jobId}] ⚡ Preview mode: Two-phase processing`);
            return runCloudPreview(jobId, videoPath, targetLang, geminiApiKey, CLOUD_API_URL, voiceEngine, elevenlabsKey, azureKey, azureRegion);
        }
        return runCloudDubbing(jobId, videoPath, targetLang, geminiApiKey, CLOUD_API_URL, voiceEngine, elevenlabsKey, azureKey, azureRegion);
    } else {
        console.log(`[Job ${jobId}] 💻 No Cloud API set, running locally`);
        return runLocalDubbing(jobId, videoPath, targetLang, geminiApiKey, voiceEngine, elevenlabsKey, azureKey, azureRegion, previewMode);
    }
};



