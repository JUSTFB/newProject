import { exec } from "child_process";
import path from "path";

export default async function textToSpeech(text, lang) {
    const outputPath = path.join(
        "uploads",
        `dubbed-${Date.now()}-${lang}.wav`
    );

    // Create a REAL silent wav using ffmpeg
    const cmd = `ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 5 "${outputPath}"`;

    return new Promise((resolve, reject) => {
        exec(cmd, (error) => {
            if (error) {
                reject(error);
            } else {
                resolve(outputPath);
            }
        });
    });
}
