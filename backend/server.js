import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import express from "express";
import multer from "multer";
import path from "path";
import cors from "cors";
import fs from "fs";
import fetch from "node-fetch";
import FormData from "form-data";
import { createJob, updateJob, getJob } from "./services/jobs.js";
import { runPythonDubbing } from "./services/pythonBridge.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Explicitly load .env from project root to ensure CLOUD_API_URL is found
dotenv.config({ path: resolve(__dirname, '../.env') });

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));


// Ensure uploads directory exists
const uploadDir = "uploads";
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Multer storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, "uploads"),
    filename: (req, file, cb) => {
        const safeName = path.basename(file.originalname);
        cb(null, Date.now() + "-" + safeName);
    }
});
const upload = multer({ storage });



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

        const previewMode = req.body?.preview === "true" || req.body?.preview === true;
        const jobId = createJob({ previewMode });
        const videoPath = req.file.path;
        const targetLang = req.body?.lang || "hi";
        const geminiApiKey = req.body.gemini_api_key || process.env.GEMINI_API_KEY || null;
        const voiceEngine = req.body.voice_engine || "xtts";
        const elevenlabsKey = req.body.elevenlabs_key || null;
        const azureKey = req.body.azure_key || null;
        const azureRegion = req.body.azure_region || "eastus";

        // Start processing in background (do not await)
        processJob(jobId, videoPath, targetLang, geminiApiKey, voiceEngine, elevenlabsKey, azureKey, azureRegion, previewMode);

        res.json({
            success: true,
            jobId,
            previewMode,
            message: previewMode ? "Preview processing started" : "Processing started"
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
    res.json({
        success: true,
        status: job.status,
        progress: job.progress,
        stage: job.stage,
        previewMode: job.previewMode || false
    });
});

// Result Endpoint
app.get("/api/result/:jobId", (req, res) => {
    const job = getJob(req.params.jobId);
    if (!job) {
        return res.status(404).json({ success: false, message: "Job not found" });
    }

    const type = req.query.type || "full";

    // Preview result requested
    if (type === "preview") {
        if (!job.previewResult) {
            return res.status(400).json({ success: false, message: "Preview not ready" });
        }
        return res.json({ success: true, type: "preview", ...job.previewResult });
    }

    // Full result requested — allow early result (streaming playlist) even if status is 'processing'
    if (job.status !== "completed" && !job.result) {
        return res.status(400).json({ success: false, message: "Job not ready" });
    }
    res.json({
        success: true,
        type: "full",
        ...job.result
    });
});



const jobQueue = [];
let activeJobsCount = 0;
const MAX_CONCURRENT_JOBS = parseInt(process.env.MAX_CONCURRENT_JOBS || "1", 10);

async function processJob(jobId, videoPath, targetLang, geminiApiKey = null, voiceEngine = "edge", elevenlabsKey = null, azureKey = null, azureRegion = "eastus", previewMode = false) {
    const executeTask = async () => {
        try {
            updateJob(jobId, { status: "processing", progress: 0, stage: "Starting AI Engine..." });
            await runPythonDubbing(jobId, videoPath, targetLang, geminiApiKey, voiceEngine, elevenlabsKey, azureKey, azureRegion, previewMode);
        } catch (err) {
            console.error(`Job ${jobId} failed to start:`, err);
            updateJob(jobId, {
                status: "error",
                progress: 0,
                stage: "Error: " + err.message
            });
        } finally {
            activeJobsCount--;
            processNextJob();
        }
    };

    if (activeJobsCount < MAX_CONCURRENT_JOBS) {
        activeJobsCount++;
        executeTask();
    } else {
        updateJob(jobId, { status: "pending", progress: 0, stage: "Queued (Waiting for GPU)..." });
        jobQueue.push(executeTask);
    }
}

function processNextJob() {
    if (activeJobsCount < MAX_CONCURRENT_JOBS && jobQueue.length > 0) {
        activeJobsCount++;
        const nextTask = jobQueue.shift();
        nextTask();
    }
}

// Compare Endpoint — proxies to cloud /dub-compare
app.post("/api/compare", upload.single("file"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, message: "No file received" });
        }

        const cloudUrl = process.env.CLOUD_API_URL;
        if (!cloudUrl) {
            return res.status(500).json({ success: false, message: "CLOUD_API_URL not configured" });
        }

        // Forward to cloud /dub-compare
        const formData = new FormData();
        formData.append("file", fs.createReadStream(req.file.path), req.file.originalname);
        formData.append("target_lang", req.body?.lang || "hi");
        const compareGeminiKey = req.body.gemini_api_key || process.env.GEMINI_API_KEY;
        if (compareGeminiKey) formData.append("gemini_api_key", compareGeminiKey);
        if (req.body.elevenlabs_key) formData.append("elevenlabs_key", req.body.elevenlabs_key);
        if (req.body.azure_key) formData.append("azure_key", req.body.azure_key);
        if (req.body.azure_region) formData.append("azure_region", req.body.azure_region);

        console.log(`\n🔬 Compare request → ${cloudUrl}/dub-compare`);

        const response = await fetch(`${cloudUrl}/dub-compare`, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders(),
            timeout: 7200000 // 2 hours for long videos
        });

        const data = await response.json();

        // Remap filenames to our proxy URLs
        if (data.success && data.engines) {
            // Store cloudUrl for proxying
            global.__cloudUrl = cloudUrl;
            global.__compareEngines = data.engines;
        }

        res.json(data);

    } catch (err) {
        console.error("Compare error:", err);
        res.status(500).json({ success: false, error: err.message });
    }
});

// Proxy comparison video files from cloud
app.get("/api/compare-video/:filename", async (req, res) => {
    try {
        const cloudUrl = process.env.CLOUD_API_URL || global.__cloudUrl;
        if (!cloudUrl) {
            return res.status(500).json({ error: "No cloud URL" });
        }

        // Sanitize filename to prevent SSRF and path traversal
        const safeFilename = encodeURIComponent(path.basename(req.params.filename));

        const response = await fetch(`${cloudUrl}/compare-result/${safeFilename}`);
        if (!response.ok) {
            return res.status(response.status).json({ error: "File not found on cloud" });
        }

        res.set('Content-Type', 'video/mp4');
        response.body.pipe(res);

    } catch (err) {
        console.error("Compare video proxy error:", err);
        res.status(500).json({ error: err.message });
    }
});

// Static file serving — AFTER all API routes to prevent 405 on POST requests
app.use("/uploads", express.static("uploads"));
const frontendPath = resolve(__dirname, '../frontend');
app.use(express.static(frontendPath));

// Global JSON error handler — prevents Express from ever returning HTML errors
app.use((err, req, res, next) => {
    console.error('Express error:', err.message);

    // Handle multer-specific errors
    if (err instanceof multer.MulterError) {
        return res.status(400).json({
            success: false,
            message: `Upload error: ${err.message}`
        });
    }

    res.status(err.status || 500).json({
        success: false,
        message: err.message || 'Internal server error'
    });
});

// 404 handler for API routes
app.use('/api', (req, res) => {
    res.status(404).json({ success: false, message: 'API endpoint not found' });
});

const server = app.listen(5000, () => {
    console.log("SERVER STARTED ON PORT 5000");
});
server.setTimeout(3600000); // 60 minutes

