import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { updateJob } from "./jobs.js";

// Helper to get absolute path relative to project root
// Assuming server.js is in /backend, so root is ../
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const PYTHON_SCRIPT = path.join(PROJECT_ROOT, "python", "dubbing_service.py");

export const runPythonDubbing = (jobId, videoPath, targetLang) => {
    return new Promise((resolve, reject) => {
        // Construct output path
        const outputPath = videoPath.replace(path.extname(videoPath), `_dubbed_${targetLang}.mp4`);
        const relativeOutputPath = path.relative(path.join(process.cwd()), outputPath);

        // Spawn Python process
        // Use 'py -3.10' launcher on Windows to ensure we use the compatible version
        const pythonProcess = spawn("py", [
            "-3.10",
            PYTHON_SCRIPT,
            "--input_video", path.resolve(videoPath),
            "--target_lang", targetLang,
            "--output_path", path.resolve(outputPath)
        ]);

        console.log(`[Job ${jobId}] Spawning Python script: ${PYTHON_SCRIPT}`);

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
                        // Don't reject yet, let the process exit ensure cleanup if needed, 
                        // or just wait for exit code.
                    } else if (message.progress) {
                        console.log(`[Job ${jobId}] Progress: ${message.stage} (${message.progress}%)`);
                        updateJob(jobId, {
                            status: "processing",
                            progress: message.progress,
                            stage: message.stage
                        });

                        if (message.result) {
                            // Python sent completion data
                            updateJob(jobId, {
                                status: "completed",
                                progress: 100,
                                stage: "Completed",
                                result: {
                                    ...message.result,
                                    finalVideo: "uploads/" + message.result.finalVideo // ensure correct relative path for frontend
                                }
                            });
                        }
                    }
                } catch (e) {
                    // Non-JSON output (maybe debug prints from libraries)
                    console.log(`[Job ${jobId}] Python Log: ${line}`);
                }
            }
        });

        pythonProcess.stderr.on("data", (data) => {
            // Pipe stderr directly to console so user sees download progress (tqdm bars)
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
