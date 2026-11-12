// Study Tracker - web server.
//
//   * REST  : camera management, settings, audio files, alert history
//   * WS    : live frames (JPEG), detection status and alert events
//
// The camera capture and the YOLO phone detection run here; the MediaPipe
// face/eye detection runs in the browser (WebAssembly) and is reported back
// over the WebSocket.
import express from "express";
import http from "node:http";
import path from "node:path";
import fs from "node:fs";
import { Worker } from "node:worker_threads";
import { WebSocketServer } from "ws";
import { CameraManager, listWebcamDevices, resolveSourceUrl, fileExists, enumerateWebcams } from "./cameraManager.js";
import { AlertEngine } from "./alertEngine.js";
import { loadSettings, saveSettings, listAudioFiles, ASSETS_DIR, ROOT } from "./settings.js";

const PORT = Number(process.env.PORT) || 5174;
const allowedOrigins = new Set(
  (process.env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean)
);
// When no origin is configured the API stays open, so a split deployment
// (Vercel frontend + Render backend) works without extra configuration.
const allowAnyOrigin = allowedOrigins.size === 0;

const app = express();
app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin && (allowAnyOrigin || allowedOrigins.has("*") || allowedOrigins.has(origin))) {
    res.setHeader("Access-Control-Allow-Origin", allowAnyOrigin ? "*" : origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  }
  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }
  next();
});
app.use(express.json({ limit: "25mb" }));
app.use(express.raw({ type: ["audio/*", "application/octet-stream", "image/jpeg"], limit: "25mb" }));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

let settings = loadSettings();
const alertEngine = new AlertEngine(settings);

const cameraManager = new CameraManager({
  onFrame: (camera, jpegBuffer) => {
    // Push the frame to every connected browser.  A small binary header with
    // the camera id keeps multi-camera streams apart.
    const header = Buffer.from(JSON.stringify({ cameraId: camera.id }), "utf8");
    const length = Buffer.alloc(4);
    length.writeUInt32BE(header.length, 0);
    const payload = Buffer.concat([length, header, jpegBuffer]);
    for (const client of clients) {
      if (client.readyState === 1) client.send(payload, { binary: true });
    }

    // Run the phone detector a few times per second (it is the slow part).
    const now = Date.now();
    if (now - lastPhoneRun >= settings.phone_detection_interval) {
      lastPhoneRun = now;
      detector.detect(camera.id, jpegBuffer).then((boxes) => {
        broadcast({
          type: "phone",
          cameraId: camera.id,
          boxes,
          count: boxes.length,
          cameraName: camera.source.name,
        });
        const event = alertEngine.handlePhoneState({
          cameraId: camera.id,
          cameraName: camera.source.name,
          count: boxes.length,
        });
        if (event) broadcast({ type: "alert", event });
      });
    }
  },
  onStatus: (camera, status) => broadcast({ type: "camera-status", camera: status }),
});

// The YOLO inference runs in a worker thread so it never blocks the video
// stream.  If the worker cannot start, the server keeps running without it.
const detector = {
  ready: false,
  worker: null,
  busy: false,
  queue: new Map(),
  detect(cameraId, jpegBuffer) {
    return new Promise((resolve) => {
      if (!this.worker) return resolve([]);
      this._pending.set(cameraId, resolve);
      this.queue.set(cameraId, jpegBuffer);
      this._pump();
    });
  },
  _pump() {
    if (this.busy || this.queue.size === 0) return;
    const [cameraId, buffer] = this.queue.entries().next().value;
    this.queue.delete(cameraId);
    this.busy = true;
    const copy = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    this.worker.postMessage(
      {
        id: cameraId,
        buffer: copy,
        confidence: settings.phone_confidence_threshold,
        size: settings.detection_size,
      },
      [copy]
    );
  },
  _finish(cameraId, boxes) {
    this.busy = false;
    this._pending.get(cameraId)?.(boxes);
    this._pending.delete(cameraId);
    this._pump();
  },
  _pending: new Map(),
};

const workerPath = path.join(ROOT, "server", "detectorWorker.js");
try {
  const worker = new Worker(workerPath);
  worker.on("message", (msg) => detector._finish(msg.id, msg.boxes || []));
  worker.on("error", (err) => console.warn("[phone-detector] worker error:", err.message));
  worker.on("exit", () => {
    detector.worker = null;
  });
  detector.worker = worker;
  detector.ready = true;
} catch (err) {
  console.warn("[phone-detector] could not start the worker:", err.message);
}

const clients = new Set();
let lastPhoneRun = 0;

function broadcast(payload) {
  const text = JSON.stringify(payload);
  for (const client of clients) {
    if (client.readyState === 1) client.send(text);
  }
}

// ---------------------------------------------------------------- cameras
app.get("/api/state", (_req, res) => {
  res.json({
    settings,
    cameras: cameraManager.list(),
    alerts: alertEngine.events.slice(0, 50),
    audio: listAudioFiles(),
    detector: { available: detector.ready, error: detector.worker ? "" : "Detector worker not running" },
  });
});

app.get("/api/webcams", async (_req, res) => {
  const devices = await listWebcamDevices();
  res.json({ devices });
});

// Inform the frontend about the capture capabilities of this host.
app.get("/api/capabilities", async (_req, res) => {
  const devices = await enumerateWebcams();
  res.json({
    platform: process.platform,
    serverWebcams: devices,
    browserCamera: true,
  });
});

app.post("/api/cameras", (req, res) => {
  const { type, value, name, username, password } = req.body || {};
  const validTypes = ["browser", "webcam", "rtsp", "file"];
  if (!validTypes.includes(type)) {
    return res.status(400).json({ error: "type must be one of: browser, webcam, rtsp, file" });
  }
  if (type !== "browser" && !value) {
    return res.status(400).json({ error: "type and value are required" });
  }
  if (type === "file" && !fileExists(value)) {
    return res.status(400).json({ error: `File not found: ${value}` });
  }
  if (type === "rtsp" && !/rtsp:\/\//i.test(value)) {
    return res.status(400).json({ error: "The RTSP URL has to start with rtsp://" });
  }

  const source = resolveSourceUrl(type, value || "", username || "", password || "");
  if (name) source.name = name;
  const camera = cameraManager.add(source);
  res.json({ camera: camera.publicStatus() });
});

// Frames captured in the browser are posted here so a hosted backend can run
// the phone detection on the user's laptop camera.
app.post("/api/cameras/:id/frame", (req, res) => {
  const camera = cameraManager.get(req.params.id);
  if (!camera) return res.status(404).json({ error: "Camera not found" });
  if (!req.body || !req.body.length) {
    return res.status(400).json({ error: "No image data received." });
  }
  camera.pushBrowserFrame(req.body);
  res.json({ ok: true, frameCount: camera.frameCount });
});

app.delete("/api/cameras/:id", (req, res) => {
  const ok = cameraManager.remove(req.params.id);
  res.json({ ok });
});

// --------------------------------------------------------------- settings
app.get("/api/settings", (_req, res) => res.json(settings));

app.put("/api/settings", (req, res) => {
  settings = saveSettings({ ...settings, ...req.body });
  alertEngine.updateSettings(settings);
  broadcast({ type: "settings", settings });
  res.json(settings);
});

// ------------------------------------------------------------------ audio
app.get("/api/audio", (_req, res) => res.json({ files: listAudioFiles() }));

app.post("/api/audio/:name", (req, res) => {
  const fileName = path.basename(req.params.name);
  if (!/\.(mp3|wav|ogg|m4a)$/i.test(fileName)) {
    return res.status(400).json({ error: "Only mp3, wav, ogg and m4a files are supported." });
  }
  if (!req.body || !req.body.length) {
    return res.status(400).json({ error: "No audio data received." });
  }
  fs.mkdirSync(ASSETS_DIR, { recursive: true });
  fs.writeFileSync(path.join(ASSETS_DIR, fileName), req.body);
  res.json({ ok: true, files: listAudioFiles() });
});

// ---------------------------------------------------------------- history
app.get("/api/alerts", (_req, res) => res.json({ alerts: alertEngine.events }));

app.delete("/api/alerts", (_req, res) => {
  alertEngine.clear();
  res.json({ ok: true });
});

// ------------------------------------------------------------- static files
app.use("/assets", express.static(ASSETS_DIR));
app.use("/models", express.static(path.join(ROOT, "models")));

// The frontend is served by the Vite dev server on port 5173 (see package.json).
// The production build in dist/ is served here when it exists.
const DIST_DIR = path.join(ROOT, "dist");
if (fs.existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR));
  app.get(/^\/(?!api|assets|models|ws).*/, (_req, res) => {
    res.sendFile(path.join(DIST_DIR, "index.html"));
  });
}

// --------------------------------------------------------------- websocket
wss.on("connection", (socket) => {
  clients.add(socket);
  socket.send(
    JSON.stringify({
      type: "hello",
      settings,
      cameras: cameraManager.list(),
      alerts: alertEngine.events.slice(0, 50),
    })
  );

  socket.on("message", async (raw) => {
    let payload;
    try {
      payload = JSON.parse(raw.toString());
    } catch {
      return;
    }

    if (payload.type === "eye-state") {
      const event = alertEngine.handleEyeState(payload);
      if (event) broadcast({ type: "alert", event });
      return;
    }

    if (payload.type === "ping") {
      socket.send(JSON.stringify({ type: "pong", t: Date.now() }));
    }
  });

  socket.on("close", () => clients.delete(socket));
});

server.listen(PORT, () => {
  console.log(`[study-tracker] server ready on http://localhost:${PORT}`);
});

process.on("SIGINT", () => {
  cameraManager.stopAll();
  process.exit(0);
});
process.on("SIGTERM", () => {
  cameraManager.stopAll();
  process.exit(0);
});
