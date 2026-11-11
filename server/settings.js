// Settings persistence for the Study Tracker web app.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(__dirname, "..");
export const ASSETS_DIR = path.join(ROOT, "assets");
export const MODELS_DIR = path.join(ROOT, "models");
const SETTINGS_FILE = path.join(ROOT, "config", "web-settings.json");

export const DEFAULT_SETTINGS = {
  // Sleep detection
  ear_closed_threshold: 0.21,
  ear_open_threshold: 0.28,
  min_eye_closure_duration: 2.5,
  sleep_alert_cooldown: 30,
  require_both_eyes: true,
  blink_threshold: 0.55,
  use_adaptive_threshold: true,
  adaptive_ratio: 0.62,
  // Phone detection
  phone_confidence_threshold: 0.4,
  min_phone_detection_duration: 1.5,
  phone_alert_cooldown: 30,
  // Performance
  phone_detection_interval: 500, // ms between two phone detections
  detection_size: 320,
  // Audio
  sleep_audio: "sleep_alert.mp3",
  phone_audio: "phone_alert.mp3",
};

function clampSettings(input) {
  const s = { ...DEFAULT_SETTINGS, ...(input || {}) };
  s.ear_closed_threshold = Math.min(Math.max(Number(s.ear_closed_threshold) || 0.21, 0.05), 0.6);
  s.ear_open_threshold = Math.min(Math.max(Number(s.ear_open_threshold) || 0.28, s.ear_closed_threshold + 0.01), 0.8);
  s.min_eye_closure_duration = Math.max(Number(s.min_eye_closure_duration) || 2.5, 0.1);
  s.sleep_alert_cooldown = Math.max(Number(s.sleep_alert_cooldown) || 0, 0);
  s.blink_threshold = Math.min(Math.max(Number(s.blink_threshold) || 0.55, 0.1), 0.95);
  s.adaptive_ratio = Math.min(Math.max(Number(s.adaptive_ratio) || 0.62, 0.2), 0.95);
  s.phone_confidence_threshold = Math.min(Math.max(Number(s.phone_confidence_threshold) || 0.4, 0.05), 0.99);
  s.min_phone_detection_duration = Math.max(Number(s.min_phone_detection_duration) || 1.5, 0.1);
  s.phone_alert_cooldown = Math.max(Number(s.phone_alert_cooldown) || 0, 0);
  s.phone_detection_interval = Math.min(Math.max(Number(s.phone_detection_interval) || 500, 100), 5000);
  s.detection_size = Math.min(Math.max(Number(s.detection_size) || 320, 160), 640);
  s.require_both_eyes = Boolean(s.require_both_eyes);
  s.use_adaptive_threshold = Boolean(s.use_adaptive_threshold);
  return s;
}

export function loadSettings() {
  try {
    if (fs.existsSync(SETTINGS_FILE)) {
      const raw = JSON.parse(fs.readFileSync(SETTINGS_FILE, "utf8"));
      return clampSettings(raw);
    }
  } catch (err) {
    console.warn("Could not read settings file:", err.message);
  }
  return { ...DEFAULT_SETTINGS };
}

export function saveSettings(settings) {
  const clean = clampSettings(settings);
  try {
    fs.mkdirSync(path.dirname(SETTINGS_FILE), { recursive: true });
    fs.writeFileSync(SETTINGS_FILE, JSON.stringify(clean, null, 2));
  } catch (err) {
    console.warn("Could not save settings file:", err.message);
  }
  return clean;
}

export function audioFilePath(name) {
  const safe = path.basename(name || "");
  return path.join(ASSETS_DIR, safe);
}

export function listAudioFiles() {
  try {
    return fs
      .readdirSync(ASSETS_DIR)
      .filter((f) => /\.(mp3|wav|ogg|m4a)$/i.test(f))
      .map((f) => ({
        name: f,
        url: `/assets/${encodeURIComponent(f)}`,
        exists: fs.existsSync(path.join(ASSETS_DIR, f)),
      }));
  } catch {
    return [];
  }
}
