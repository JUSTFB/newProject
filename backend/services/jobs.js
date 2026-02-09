
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

export const updateJob = (jobId, updates) => {
    if (!jobs.has(jobId)) return;
    const job = jobs.get(jobId);
    jobs.set(jobId, { ...job, ...updates });
};

export const getJob = (jobId) => {
    return jobs.get(jobId);
};
