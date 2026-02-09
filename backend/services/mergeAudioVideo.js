import { exec } from "child_process";
import path from "path";

export default function mergeAudioVideo(videoPath, audioPath) {
    return new Promise((resolve, reject) => {
        const dir = path.dirname(videoPath);
        const base = path.basename(videoPath, path.extname(videoPath));
        const outputPath = path.join(dir, `${base}-dubbed.mp4`);

        const cmd = `ffmpeg -y -i "${videoPath}" -i "${audioPath}" -map 0:v:0 -map 1:a:0 -c:v copy "${outputPath}"`;

        exec(cmd, (error) => {
            if (error) {
                reject(error);
            } else {
                resolve(outputPath);
            }
        });
    });
}
