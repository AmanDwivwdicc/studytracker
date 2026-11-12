# Study Tracker – Web Edition

Study Tracker is a web application that monitors a study environment through a laptop webcam, USB camera, RTSP IP camera, or local test video. It warns the user when it detects closed eyes/sleep or mobile phone use.

The repository contains only the React/Vite frontend and Node.js backend. There is no Python runtime or desktop application.

| Capability | Implementation |
| --- | --- |
| Sleep detection | MediaPipe Face Landmarker in the browser, using Eye Aspect Ratio and the `eyeBlink` blendshape |
| Phone detection | YOLO11-nano ONNX model executed by ONNX Runtime in a Node.js worker thread |
| Cameras | Webcam/USB device discovery, RTSP streams, and local video files through FFmpeg |
| Alerts | Browser audio playback, per-type cooldown, and timestamped alert history |
| Dashboard | React + Vite single-page application with REST and WebSocket communication |

## Requirements

- Node.js 20.19 or newer
- npm
- A supported camera source
- Internet access during the first dependency installation only

## Local development

```powershell
cd C:\Users\LENOVO\study_tracker
npm install
npm run dev
```

Open `http://localhost:5173`.

The development command starts:

| Process | Port | Purpose |
| --- | --- | --- |
| `dev:web` | 5173 | Vite and the React dashboard |
| `dev:server` | 5174 | Camera capture, ONNX detection, REST API, and WebSocket server |

Other commands:

```powershell
npm run build      # Build the frontend into dist/
npm run preview    # Preview the frontend build
npm start          # Start the Node server; it also serves dist/ when present
npm test           # Camera/detection/WebSocket self-test (requires a camera)
```

## Using the dashboard

1. Select a laptop/USB webcam, RTSP camera, or local video file.
2. Enter the camera index, device name, RTSP URL, or file path.
3. Select **Connect Camera**.
4. Use **Start/Stop Monitoring** to pause or resume browser-side eye detection.
5. Use **Apply Settings** to save detection settings to `config/web-settings.json`.
6. Use the audio panel to upload and test alert sounds.

RTSP URLs normally look like `rtsp://192.168.1.100:554/stream1`. Credentials entered in the dashboard are URL-encoded before being passed to FFmpeg.

## Detection settings

| Setting | Default | Meaning |
| --- | --- | --- |
| Eye closed threshold (EAR) | 0.21 | EAR below which an eye counts as closed |
| Eye open threshold (EAR) | 0.28 | Re-open threshold with hysteresis |
| Blink score threshold | 0.55 | MediaPipe `eyeBlink` score above which an eye counts as closed |
| Adaptive EAR ratio | 0.62 | Closed-eye threshold relative to the learned open-eye baseline |
| Minimum eye closure | 2.5 seconds | How long eyes must remain closed before a sleep alert |
| Sleep alert cooldown | 30 seconds | Minimum delay between sleep sounds |
| Phone confidence | 0.40 | Minimum YOLO confidence for a phone detection |
| Minimum phone duration | 1.5 seconds | How long a phone must remain visible before an alert |
| Phone alert cooldown | 30 seconds | Minimum delay between phone sounds |
| Phone check interval | 500 milliseconds | Delay between server-side ONNX inference runs |

## Project layout

```text
study_tracker/
├── package.json
├── package-lock.json
├── vite.config.js
├── index.html
├── src/                      # React dashboard and browser MediaPipe detection
├── server/                   # Express, WebSocket, FFmpeg, and ONNX Runtime backend
├── public/wasm/              # MediaPipe browser runtime copied during npm install
├── models/
│   ├── face_landmarker.task  # Browser MediaPipe model
│   ├── yolo11n.onnx          # Node ONNX phone-detection model
│   └── yolo11n.onnx.data     # External ONNX model data
├── assets/                   # Browser-played alert sounds
├── config/web-settings.json  # Persisted web settings
└── scripts/                  # WASM copy and web self-test scripts
```

## Deployment architecture

The browser and server communicate through REST endpoints under `/api` and a WebSocket at `/ws`. The frontend can be hosted on Vercel while the persistent Node backend runs on Render, but the frontend must be given the backend URLs.

### Vercel: frontend deployment

1. Push the cleaned repository to GitHub.
2. In Vercel, select **Add New → Project** and import the repository.
3. Vercel should detect the Vite project automatically.
4. Use these build settings:
   - Install command: `npm install`
   - Build command: `npm run build`
   - Output directory: `dist`
5. Add environment variables if the backend is hosted separately:
   - `VITE_API_BASE_URL=https://your-render-service.onrender.com`
   - `VITE_WS_BASE_URL=wss://your-render-service.onrender.com/ws`
   - `VITE_MODEL_BASE_URL=https://your-render-service.onrender.com`
6. Redeploy the Vercel project after changing environment variables.

Vercel serves the React frontend only. The camera server, ONNX Runtime worker, and persistent WebSocket connection must run on a service that supports long-running Node processes.

### Render: full application or backend

To serve the complete application from one domain:

1. In Render, select **New → Web Service** and connect the repository.
2. Use:
   - Runtime: Node
   - Build command: `npm ci && npm run build`
   - Start command: `npm start`
3. Add `ALLOWED_ORIGINS` with the frontend origin when the frontend is on Vercel, for example `https://your-project.vercel.app`. Multiple origins can be comma-separated.
4. Deploy and use the generated `https://your-render-service.onrender.com` URL.
5. If Vercel hosts the frontend, set the three `VITE_*` variables in Vercel using the Render URL.

When the frontend and backend share the Render domain, no `VITE_*` variables are needed. The Node server serves the built frontend from `dist/` and handles `/api`, `/ws`, `/models`, and `/assets` on the same origin.

A Render-hosted backend can access reachable RTSP cameras and test videos stored on its filesystem. It cannot directly access a webcam attached to the user's laptop; use the local Node server for laptop/USB webcam monitoring.

## Important model and asset behavior

- `models/face_landmarker.task` is loaded by the browser through `/models/face_landmarker.task`.
- `models/yolo11n.onnx` and `models/yolo11n.onnx.data` are loaded by the Node ONNX worker.
- `assets/sleep_alert.mp3` and `assets/phone_alert.mp3` are played by the browser.
- `npm install` runs `scripts/copy-wasm.mjs`, copying the MediaPipe WASM runtime into `public/wasm/`.

Do not remove the model or asset files from the repository unless the corresponding detection or alert feature is being removed as well.

## Credits

- [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe)
- [Ultralytics YOLO](https://www.ultralytics.com/)
- [ONNX Runtime Node](https://onnxruntime.ai/)
- [FFmpeg](https://ffmpeg.org/)
- [React](https://react.dev/) and [Vite](https://vite.dev/)
