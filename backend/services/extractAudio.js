import { exec } from "child_process";
import path from "path";

export default function extractAudio(videoPath) {
    return new Promise((resolve, reject) => {
        const dir = path.dirname(videoPath);
        const base = path.basename(videoPath, path.extname(videoPath));
        const audioPath = path.join(dir, `${base}.wav`);

        const cmd = `ffmpeg -y -i "${videoPath}" "${audioPath}"`;

        exec(cmd, (error) => {
            if (error) {
                reject(error);
            } else {
                resolve(audioPath);
            }
        });
    });
}
