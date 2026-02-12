import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { updateJob } from "./jobs.js";

// Helper to get absolute path relative to project root
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const PYTHON_SCRIPT = path.join(PROJECT_ROOT, "python", "dubbing_service.py");

// Cloud API URL (set this to use remote GPU processing)
// Example: CLOUD_API_URL=https://abc123.ngrok-free.app
// Cloud API URL is checked in runPythonDubbing to ensure process.env is loaded
// const CLOUD_API_URL = process.env.CLOUD_API_URL || null;

/**
 * Cloud Mode: Send video to remote GPU server for processing.
 * The cloud server runs cloud_api.py on Colab/RunPod with a T4/A100 GPU.
 */
const runCloudDubbing = async (jobId, videoPath, targetLang, geminiApiKey = null, cloudUrl) => {
    console.log(`[Job ${jobId}] ☁️ CLOUD MODE: Sending to ${cloudUrl}/dub`);

    updateJob(jobId, {
        status: "processing",
        progress: 10,
        stage: "Uploading to Cloud GPU..."
    });

    try {
        // MUST use node-fetch with form-data package (Node built-in fetch is incompatible)
        const FormData = (await import('form-data')).default;
        const { default: fetch } = await import('node-fetch');

        const form = new FormData();
        form.append('file', fs.createReadStream(path.resolve(videoPath)));
        form.append('target_lang', targetLang);
        if (geminiApiKey) form.append('gemini_api_key', geminiApiKey);

        updateJob(jobId, {
            status: "processing",
            progress: 20,
            stage: "Processing on Cloud GPU (XTTS Cloning)..."
        });


        const response = await fetch(`${cloudUrl}/dub`, {
            method: 'POST',
            body: form,
            headers: form.getHeaders ? form.getHeaders() : {},
            timeout: 3600000 // 60 minute timeout
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: "Unknown cloud error" }));
            throw new Error(errorData.error || `Cloud API returned ${response.status}`);
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
 * Local Mode: Spawn Python process on local machine.
 */
const runLocalDubbing = (jobId, videoPath, targetLang, geminiApiKey = null) => {
    return new Promise((resolve, reject) => {
        const outputPath = videoPath.replace(path.extname(videoPath), `_dubbed_${targetLang}.mp4`);

        console.log(`[Job ${jobId}] 💻 LOCAL MODE: Spawning ${PYTHON_SCRIPT}`);

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

        const pythonProcess = spawn("py", args, {
            env: { ...process.env, PYTHONIOENCODING: "utf-8" },
            shell: true
        });

        pythonProcess.stdout.on("data", (data) => {
            const lines = data.toString().split("\n");
            for (const line of lines) {
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

                        if (message.playlist) {
                            const hlsDirName = path.basename(outputPath).replace(path.extname(outputPath), "_hls");
                            const playlistUrl = `uploads/${hlsDirName}/${message.playlist}`;
                            updates.result = {
                                playlistUrl: playlistUrl,
                                currentSegment: message.segment
                            };
                        }

                        updateJob(jobId, updates);

                        if (message.result) {
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
 */
export const runPythonDubbing = (jobId, videoPath, targetLang, geminiApiKey = null) => {
    // Check CLOUD_API_URL at runtime (after dotenv loads)
    const CLOUD_API_URL = process.env.CLOUD_API_URL || null;

    if (CLOUD_API_URL) {
        console.log(`[Job ${jobId}] 🌐 Cloud API detected: ${CLOUD_API_URL}`);
        return runCloudDubbing(jobId, videoPath, targetLang, geminiApiKey, CLOUD_API_URL);
    } else {
        console.log(`[Job ${jobId}] 💻 No Cloud API set, running locally`);
        return runLocalDubbing(jobId, videoPath, targetLang, geminiApiKey);
    }
};
