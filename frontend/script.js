const dubBtn = document.getElementById("dubBtn");
const videoInput = document.getElementById("videoFile");
const dropZone = document.getElementById("dropZone");
const uploadText = document.getElementById("uploadText");
const fileNameDisplay = document.getElementById("fileName");

const statusContainer = document.getElementById("statusContainer");
const statusText = document.getElementById("status");

const resultArea = document.getElementById("resultArea");
const translatedTextDiv = document.getElementById("translatedText");
const outputVideo = document.getElementById("outputVideo");
const downloadLink = document.getElementById("downloadLink");
const speakBtn = document.getElementById("speakBtn");
const newBtn = document.getElementById("newBtn");

let isProcessing = false;

// --- File Upload Handling ---
videoInput.addEventListener("change", handleFileSelect);

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--primary)";
    dropZone.style.background = "rgba(99, 102, 241, 0.1)";
});

dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--glass-border)";
    dropZone.style.background = "rgba(255, 255, 255, 0.02)";
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--glass-border)";
    dropZone.style.background = "rgba(255, 255, 255, 0.02)";

    if (e.dataTransfer.files.length) {
        videoInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = videoInput.files[0];
    if (file) {
        uploadText.classList.add("hidden");
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.classList.remove("hidden");
    }
}

// --- Main Process ---
dubBtn.addEventListener("click", async (e) => {
    if (e) e.preventDefault();
    if (isProcessing) return;

    const file = videoInput.files[0];
    const lang = document.getElementById("language").value;

    if (!file) {
        alert("Please upload a video first");
        return;
    }

    isProcessing = true;
    setLoadingState(true);
    updateStatus("Uploading video to server...", 0);
    resultArea.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("lang", lang);

    try {
        // 1. Start Job
        const response = await fetch("http://localhost:5000/api/process", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Server error");
        const data = await response.json();

        if (!data.success) throw new Error(data.message || "Processing failed");

        const jobId = data.jobId;
        console.log("Job started:", jobId);

        // 2. Poll Status
        pollStatus(jobId);

    } catch (err) {
        console.error(err);
        updateStatus("❌ Error: " + err.message);
        setLoadingState(false);
        isProcessing = false;
    }
});

// Prevent accidental reload
window.addEventListener("beforeunload", (e) => {
    if (isProcessing) {
        e.preventDefault();
        e.returnValue = "Processing in progress. Are you sure you want to leave?";
    }
});

async function pollStatus(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`http://localhost:5000/api/status/${jobId}`);
            const data = await res.json();

            if (!data.success) {
                clearInterval(pollInterval);
                throw new Error(data.message);
            }

            const { status, progress, stage } = data;

            updateStatus(stage, progress);

            if (status === "completed") {
                clearInterval(pollInterval);
                fetchResult(jobId);
            } else if (status === "error") {
                clearInterval(pollInterval);
                throw new Error("Job failed on server");
            }

        } catch (err) {
            clearInterval(pollInterval);
            console.error(err);
            updateStatus("❌ Error: " + err.message);
            setLoadingState(false);
            isProcessing = false;
        }
    }, 1000);
}

async function fetchResult(jobId) {
    try {
        const res = await fetch(`http://localhost:5000/api/result/${jobId}`);
        const data = await res.json();

        if (!data.success) throw new Error(data.message);

        showResults(data);
        setLoadingState(false);
        isProcessing = false;

    } catch (err) {
        console.error(err);
        updateStatus("❌ Error fetching results");
        setLoadingState(false);
        isProcessing = false;
    }
}

// --- Helper Functions ---

function setLoadingState(loading) {
    dubBtn.disabled = loading;
    if (loading) {
        dubBtn.innerHTML = '<span class="loader" style="width: 16px; height: 16px; border-width: 2px;"></span> Processing...';
        statusContainer.classList.remove("hidden");
    } else {
        dubBtn.innerHTML = '<span>Start Dubbing</span>';
        // Keep status container hidden if we are done, logic handled in showResults
    }
}

const progressBar = document.getElementById("progressBar");
const progressPercentage = document.getElementById("progressPercentage");

function updateStatus(message, percent = 0) {
    statusText.textContent = message;
    progressBar.style.width = `${percent}%`;
    progressPercentage.textContent = `${percent}%`;
}

function showResults(data) {
    statusContainer.classList.add("hidden");
    resultArea.classList.remove("hidden");

    // Populate Data
    if (data.translatedText) {
        translatedTextDiv.textContent = data.translatedText;
    } else {
        translatedTextDiv.textContent = "Text content not available in this version.";
    }

    const videoUrl = `http://localhost:5000/${data.finalVideo}`;
    outputVideo.src = videoUrl;
    downloadLink.href = videoUrl;

    // Smooth scroll to results
    resultArea.scrollIntoView({ behavior: 'smooth' });
}

// Reset Button
newBtn.addEventListener("click", () => {
    videoInput.value = "";
    uploadText.classList.remove("hidden");
    fileNameDisplay.classList.add("hidden");
    fileNameDisplay.textContent = "";

    resultArea.classList.add("hidden");
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Speak Button (Mock functionality if no specific TTS endpoint for just playing audio)
speakBtn.addEventListener("click", () => {
    alert("Audio preview would play here.");
});
