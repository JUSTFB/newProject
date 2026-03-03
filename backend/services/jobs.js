const jobs = new Map();

/**
 * Create a new job with optional preview mode.
 * Statuses: pending → processing → preview_ready → full_processing → completed | error
 */
export const createJob = (options = {}) => {
    const jobId = Math.random().toString(36).substring(7);
    jobs.set(jobId, {
        id: jobId,
        previewMode: options.previewMode || false,
        status: "pending", // pending, processing, preview_ready, full_processing, completed, error
        progress: 0,
        stage: "Queued",
        result: null,         // Full video result
        previewResult: null,  // Preview video result (first N sentences)
        startTime: Date.now()
    });
    return jobId;
};

export const startJob = async (jobId, videoPath, targetLang, geminiApiKey = null) => {
    const job = jobs.get(jobId);
    if (!job) return;

    job.status = 'processing';
    job.stage = 'Initializing...';

    try {
        await runPythonDubbing(jobId, videoPath, targetLang, geminiApiKey);
        job.status = 'completed';
        job.stage = 'Finished';
    } catch (error) {
        job.status = 'error';
        job.stage = error.message;
    } finally {
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
