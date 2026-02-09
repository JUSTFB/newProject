// import fs from "fs";
// import OpenAI from "openai";

// const openai = new OpenAI({
//     apiKey: process.env.OPENAI_API_KEY
// });

// export default async function transcribeAudio(audioPath) {
//     const response = await openai.audio.transcriptions.create({
//         file: fs.createReadStream(audioPath),
//         model: "whisper-1"
//     });

//     return response.text;
// }
export default async function transcribeAudio(audioPath) {
    console.log("MOCK TRANSCRIPTION for:", audioPath);

    // Simulate processing delay
    await new Promise((r) => setTimeout(r, 1000));

    return "This is a mock transcript. The real speech-to-text will be enabled later.";
}
