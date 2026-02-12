import dotenv from "dotenv";
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log("Current directory:", process.cwd());
console.log("Expected .env path:", resolve(__dirname, '../.env'));

dotenv.config({ path: resolve(__dirname, '../.env') });

console.log("CLOUD_API_URL:", process.env.CLOUD_API_URL);
