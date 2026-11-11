// Camera manager: captures webcams, RTSP IP cameras and local video files
// with ffmpeg and pushes JPEG frames to the connected browser clients.
//
// Every camera runs in its own process so one broken RTSP stream can never
// stop the others.  The stream reconnects automatically when it breaks.
import { EventEmitter } from "node:events";
import { spawn } from "node:child_process";
import ffmpegPath from "ffmpeg-static";
import fs from "node:fs";
import path from "node:path";
import { ROOT } from "./settings.js";

let cameraCounter = 0;

// ffmpeg rejects unknown query options in an RTSP URL
// ("Error splitting the argument list: Option not found").
// Only a few options are supported - everything else is removed here.
function sanitizeRtspUrl(rawUrl) {
  let url = String(rawUrl || "").trim();
  let transport = "";
  let user = "";
  let password = "";
  let timeout = "";

  const queryIndex = url.indexOf("?");
  if (queryIndex >= 0) {
    const query = url.slice(queryIndex + 1);
    url = url.slice(0, queryIndex);
    for (const part of query.split("&")) {
      if (!part) continue;
      const [rawKey, rawValue = ""] = part.split("=");
      const key = decodeURIComponent(rawKey).trim().toLowerCase();
      const value = decodeURIComponent(rawValue);
      if (key === "transport") transport = value.trim().toLowerCase();
      else if (key === "user" || key === "username") user = value;
      else if (key === "pass" || key === "password") password = value;
      else if (key === "timeout") timeout = value.trim();
    }
  }

  // Credentials given inside the URL stay untouched.
  if (!user && /^[a-z]+:\/\/[^/@]+@/i.test(url)) {
    const match = url.match(/^([a-z]+:\/\/)([^/@]+)@/i);
    if (match) {
      const creds = match[2];
      const colon = creds.indexOf(":");
      user = colon >= 0 ? creds.slice(0, colon) : creds;
      password = colon >= 0 ? creds.slice(colon + 1) : "";
    }
  }
  if (user) {
    const creds = `${encodeURIComponent(user)}:${encodeURIComponent(password || "")}@`;
    url = url.replace(/^([a-z]+:\/\/)/i, `$1${creds}`);
  }

  return { url, transport, timeout };
}

function buildFfmpegArgs(source, opts = {}) {
  const args = ["-hide_banner", "-loglevel", "error"];
  const input = source.url;

  if (source.type === "webcam") {
    // Windows cameras are addressed by their DirectShow device name.
    // Strip a possible "video=" prefix and quotes so ffmpeg can parse it.
    const device = String(input).replace(/^video=/i, "").replace(/"/g, "").trim();
    args.push("-f", "dshow", "-i", `video=${device}`);
  } else {
    if (source.type === "rtsp") {
      const { url, transport, timeout } = sanitizeRtspUrl(input);
      const mode = ["tcp", "udp", "udp_multicast"].includes(transport) ? transport : "tcp";
      args.push("-rtsp_transport", mode, "-timeout", timeout || "5000000");
      args.push("-i", url);
    } else {
      args.push("-i", input);
    }
  }

  if (source.type === "file") {
    args.push("-stream_loop", "-1");
  }
  args.push(
    "-an",
    "-vf", `fps=${opts.fps || 12},scale=${opts.width || 960}:-2`,
    "-q:v", String(opts.quality || 6),
    "-f", "image2pipe",
    "-vcodec", "mjpeg",
    "-"
  );
  return args;
}

export class Camera extends EventEmitter {
  constructor(source) {
    super();
    cameraCounter += 1;
    this.id = `cam_${Date.now().toString(36)}_${cameraCounter}`;
    this.source = source; // { type, url, name }
    this.status = "connecting";
    this.error = "";
    this.process = null;
    this.buffer = Buffer.alloc(0);
    this.frame = null; // latest JPEG buffer
    this.frameCount = 0;
    this.fps = 0;
    this._fpsWindow = Date.now();
    this._framesInWindow = 0;
    this._stopping = false;
    this._reconnectDelay = 3000;
  }

  start() {
    this._stopping = false;
    this._spawn();
  }

  async _spawn() {
    if (this._stopping) return;
    this.status = "connecting";
    this.emit("status", this.publicStatus());

    let args;
    try {
      const source = { ...this.source };
      if (source.type === "webcam") {
        source.url = await resolveWebcamDevice(source.url);
        this.source.url = source.url;
      }
      args = buildFfmpegArgs(source, {});
    } catch (err) {
      this.error = err.message;
      this.status = "error";
      this.emit("status", this.publicStatus());
      return;
    }

    this.process = spawn(ffmpegPath, args, { stdio: ["ignore", "pipe", "pipe"] });

    this.process.stdout.on("data", (chunk) => this._onData(chunk));
    this.process.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      if (text.trim()) {
        this.error = text.trim().split("\n").slice(-1)[0];
        console.warn(`[camera ${this.id}] ${this.error}`);
      }
    });
    this.process.on("error", (err) => {
      this.error = err.message;
      this.status = "error";
      this.emit("status", this.publicStatus());
    });
    this.process.on("close", () => {
      if (this._stopping) {
        this.status = "disconnected";
        this.emit("status", this.publicStatus());
        return;
      }
      this.status = "reconnecting";
      this.error = this.error || "Stream interrupted";
      this.emit("status", this.publicStatus());
      setTimeout(() => this._spawn(), this._reconnectDelay);
    });  }

  _onData(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    // JPEG frames start with FFD8 and end with FFD9.
    const start = this.buffer.indexOf(Buffer.from([0xff, 0xd8]));
    if (start < 0) {
      if (this.buffer.length > 1024 * 1024) this.buffer = Buffer.alloc(0);
      return;
    }
    const end = this.buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2);
    if (end < 0) return;

    const frame = this.buffer.subarray(start, end + 2);
    this.buffer = this.buffer.subarray(end + 2);
    this.frame = frame;
    this.frameCount += 1;
    this._framesInWindow += 1;

    const now = Date.now();
    if (now - this._fpsWindow >= 1000) {
      this.fps = (this._framesInWindow * 1000) / (now - this._fpsWindow);
      this._framesInWindow = 0;
      this._fpsWindow = now;
    }

    if (this.status !== "connected") {
      this.status = "connected";
      this.error = "";
      this.emit("status", this.publicStatus());
    }
    this.emit("frame", frame);
  }

  stop() {
    this._stopping = true;
    if (this.process) {
      this.process.kill("SIGKILL");
      this.process = null;
    }
    this.status = "disconnected";
  }

  publicStatus() {
    return {
      id: this.id,
      name: this.source.name,
      type: this.source.type,
      url: this.source.url,
      status: this.status,
      error: this.error,
      fps: Number(this.fps.toFixed(1)),
      frameCount: this.frameCount,
    };
  }
}

export class CameraManager {
  constructor({ onFrame = null, onStatus = null } = {}) {
    this.cameras = new Map();
    this.onFrame = onFrame;
    this.onStatus = onStatus;
  }

  add(source) {
    const camera = new Camera(source);
    this.cameras.set(camera.id, camera);
    if (this.onFrame) camera.on("frame", (frame) => this.onFrame(camera, frame));
    if (this.onStatus) camera.on("status", (status) => this.onStatus(camera, status));
    camera.start();
    return camera;
  }

  remove(id) {
    const camera = this.cameras.get(id);
    if (!camera) return false;
    camera.stop();
    this.cameras.delete(id);
    return true;
  }

  get(id) {
    return this.cameras.get(id);
  }

  list() {
    return [...this.cameras.values()].map((c) => c.publicStatus());
  }

  stopAll() {
    for (const camera of this.cameras.values()) camera.stop();
    this.cameras.clear();
  }
}

// Resolve a webcam index to a Windows DirectShow device name for ffmpeg.
export async function listWebcamDevices() {
  const devices = await enumerateWebcams();
  return devices.map((d, index) => ({ index, name: d }));
}

export async function enumerateWebcams() {
  if (!ffmpegPath) return [];
  return new Promise((resolve) => {
    const proc = spawn(
      ffmpegPath,
      ["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
      { stdio: ["ignore", "pipe", "pipe"] }
    );
    let out = "";
    proc.stderr.on("data", (chunk) => (out += chunk.toString()));
    proc.on("close", () => {
      const devices = [];
      // ffmpeg prints one line per device:
      //   [dshow @ ...] "Integrated Camera" (video)
      for (const line of out.split(/\r?\n/)) {
        const match = line.match(/"([^"]+)"\s*\(video\)/i);
        if (match) devices.push(match[1]);
      }
      resolve(devices);
    });
    proc.on("error", () => resolve([]));
  });
}

// Turn the user input ("0", "Integrated Camera", ...) into a device name.
export async function resolveWebcamDevice(value) {
  const devices = await enumerateWebcams();
  const input = String(value ?? "").trim();
  if (!input) return devices[0] || "";
  if (/^\d+$/.test(input)) {
    const index = Number(input);
    return devices[index] || devices[0] || input;
  }
  // Exact match, then a case insensitive "contains" match.
  const exact = devices.find((d) => d.toLowerCase() === input.toLowerCase());
  if (exact) return exact;
  const partial = devices.find((d) => d.toLowerCase().includes(input.toLowerCase()));
  return partial || input;
}

// Build a capture URL for the given source description.
export function resolveSourceUrl(type, value, username = "", password = "") {
  if (type === "webcam") {
    const device = String(value?.trim() || "0");
    return { type, url: device, name: `Webcam ${device}` };
  }
  if (type === "rtsp") {
    let url = value.trim();
    if (!/^rtsp:\/\//i.test(url)) url = `rtsp://${url}`;
    if (username) {
      url = url.replace(/^rtsp:\/\//i, `rtsp://${encodeURIComponent(username)}:${encodeURIComponent(password || "")}@`);
    }
    let host = "";
    try {
      host = new URL(url.replace(/^rtsp:\/\//i, "http://")).host;
    } catch {
      host = "";
    }
    return { type, url, name: `IP Camera ${host}` };
  }
  const filePath = path.isAbsolute(value) ? value : path.join(ROOT, value);
  return { type, url: filePath, name: path.basename(filePath) };
}

export function fileExists(p) {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false;
  }
}
