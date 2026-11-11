// Copies the MediaPipe WASM runtime into public/wasm so the browser can load
// the Face Landmarker without a bundler-specific import.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const target = path.join(root, "public", "wasm");

if (!fs.existsSync(source)) {
  console.warn("[copy-wasm] @mediapipe/tasks-vision is not installed - skipping.");
  process.exit(0);
}

fs.mkdirSync(target, { recursive: true });
for (const file of fs.readdirSync(source)) {
  if (file.endsWith(".wasm") || file.endsWith(".js")) {
    fs.copyFileSync(path.join(source, file), path.join(target, file));
  }
}
console.log(`[copy-wasm] MediaPipe runtime copied to ${target}`);
