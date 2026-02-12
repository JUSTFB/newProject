import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Explicitly load .env from project root to ensure CLOUD_API_URL is found
dotenv.config({ path: resolve(__dirname, '../.env') });
import express from "express";
import multer from "multer";
import path from "path";

// app.use("/uploads", express.static("uploads"));

const app = express();
import cors from "cors";
app.use(cors());

app.use("/uploads", express.static("uploads"));
// Multer storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, "uploads"),
    filename: (req, file, cb) =>
        cb(null, Date.now() + "-" + file.originalname)
});

const upload = multer({ storage });

app.get("/", (req, res) => {
    res.send("ROOT OK");
});

import { createJob, updateJob, getJob } from "./services/jobs.js";

// Upload → Extract Audio → Transcribe
// NOW: Non-blocking, returns jobId immediately
app.post("/api/process", upload.single("file"), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: "No file received"
            });
        }

        const jobId = createJob();
        const videoPath = req.file.path;
        const targetLang = req.body?.lang || "hi";
        const geminiApiKey = req.body.gemini_api_key || null;

        // Start processing in background (do not await)
        processJob(jobId, videoPath, targetLang, geminiApiKey);

        res.json({
            success: true,
            jobId,
            message: "Processing started"
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({
            success: false,
            message: err.message
        });
    }
});

// Status Endpoint
app.get("/api/status/:jobId", (req, res) => {
    const job = getJob(req.params.jobId);
    if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
    }
    res.json({ success: true, status: job.status, progress: job.progress, stage: job.stage });
});

// Result Endpoint
app.get("/api/result/:jobId", (req, res) => {
    const job = getJob(req.params.jobId);
    if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
    }
    // Allow early result (streaming playlist) even if status is 'processing'
    if (job.status !== "completed" && !job.result) {
        return res.status(400).json({ success: false, message: "Job not ready" });
    }
    res.json({
        success: true,
        ...job.result
    });
});

// Background Worker Function
import { runPythonDubbing } from "./services/pythonBridge.js";

async function processJob(jobId, videoPath, targetLang, geminiApiKey = null) {
    try {
        updateJob(jobId, { status: "processing", progress: 0, stage: "Starting AI Engine..." });

        // The bridge handles all updates via stdout parsing
        await runPythonDubbing(jobId, videoPath, targetLang, geminiApiKey);

    } catch (err) {
        console.error(`Job ${jobId} failed to start:`, err);
        updateJob(jobId, {
            status: "error",
            progress: 0,
            stage: "Error: " + err.message
        });
    }
}

const server = app.listen(5000, () => {
    console.log("SERVER STARTED ON PORT 5000");
});
server.setTimeout(3600000); // 60 minutes












