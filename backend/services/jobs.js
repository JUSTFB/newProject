const jobs = new Map();

export const createJob = () => {
    const jobId = Math.random().toString(36).substring(7);
    jobs.set(jobId, {
        id: jobId,
        status: "pending", // pending, processing, completed, error
        progress: 0,
        stage: "Queued",
        result: null,
        startTime: Date.now()
    });
    return jobId;
};

export const startJob = async (jobId, videoPath, targetLang, geminiApiKey = null) => {
    const job = jobs.get(jobId); // Use jobs.get for Map
    if (!job) return;

    job.status = 'processing';
    job.stage = 'Initializing...';

    try {
        // Assuming runPythonDubbing is defined elsewhere or will be imported
        await runPythonDubbing(jobId, videoPath, targetLang, geminiApiKey);
        // Update job status upon completion (assuming runPythonDubbing handles progress/result)
        job.status = 'completed';
        job.stage = 'Finished';
    } catch (error) {
        job.status = 'error';
        job.stage = error.message;
    } finally {
        // Ensure job state is updated in the map
        jobs.set(jobId, job);
    }
};

export const updateJob = (jobId, updates) => {
    if (!jobs.has(jobId)) return;
    const job = jobs.get(jobId);
    jobs.set(jobId, { ...job, ...updates });
};

export const getJob = (jobId) => {
    return jobs.get(jobId);
};
