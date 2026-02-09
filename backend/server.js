import "dotenv/config";
import express from "express";
import multer from "multer";
import extractAudio from "./services/extractAudio.js";
import transcribeAudio from "./services/transcribe.js";
import translateText from "./services/translate.js";
import textToSpeech from "./services/textToSpeech.js";
import mergeAudioVideo from "./services/mergeAudioVideo.js";
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

        // Start processing in background (do not await)
        processJob(jobId, videoPath, targetLang);

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
    if (job.status !== "completed") {
        return res.status(400).json({ success: false, message: "Job not ready" });
    }
    res.json({
        success: true,
        ...job.result
    });
});

// Background Worker Function
import { runPythonDubbing } from "./services/pythonBridge.js";

async function processJob(jobId, videoPath, targetLang) {
    try {
        updateJob(jobId, { status: "processing", progress: 0, stage: "Starting AI Engine..." });

        // The bridge handles all updates via stdout parsing
        await runPythonDubbing(jobId, videoPath, targetLang);

    } catch (err) {
        console.error(`Job ${jobId} failed to start:`, err);
        updateJob(jobId, {
            status: "error",
            progress: 0,
            stage: "Error: " + err.message
        });
    }
}

app.listen(5000, () => {
    console.log("SERVER STARTED ON PORT 5000");
});












