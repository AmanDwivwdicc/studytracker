# Study Tracker – Web Edition

AI-powered study monitoring **web application**. It watches your study environment
through a laptop webcam, a USB camera or a network IP camera (RTSP) and warns you when you

* fall asleep / keep your eyes closed, and
* start using a mobile phone.

Everything runs **locally and offline** – no cloud services, no paid APIs.

| | |
| --- | --- |
| **Sleep detection** | MediaPipe Face Landmarker (WebAssembly, runs in the browser) → Eye Aspect Ratio for both eyes + the `eyeBlink` blendshape as a second signal. Hysteresis + minimum closed duration means normal blinks never trigger an alert. |
| **Phone detection** | YOLO11-nano ONNX model executed with ONNX Runtime on the server. Only the COCO class **cell phone** (class 67) is reported. |
| **Cameras** | Laptop/USB webcam (DirectShow device names), RTSP IP cameras with credentials, local video files. One capture process per camera with automatic reconnection. |
| **Alerts** | Red banner on the live feed, your own `sleep_alert.mp3` / `phone_alert.mp3`, per-type cooldown, timestamped alert history. |
| **Dashboard** | React + Vite single page app: source selection, live feeds, colour coded status, EAR/blink readout, settings, audio upload + test buttons. |

---

## 1. Requirements

* Windows 11 (also works on Windows 10)
* **Node.js 20 or newer** – check with `node --version`
* A camera (laptop webcam, USB camera or RTSP IP camera)
* Internet access **only for the first install** (npm packages) and the first
  start (model files)

---

## 2. Install & run (VS Code terminal / PowerShell)

```powershell
cd C:\Users\LENOVO\study_tracker
npm install
npm run dev
```

Then open **http://localhost:5173** in Chrome/Edge.

`npm run dev` starts two processes:

| Process | Port | Task |
| --- | --- | --- |
| `dev:web` | 5173 | Vite + React dashboard (what you open) |
| `dev:server` | 5174 | camera capture, YOLO phone detection, alert engine, REST + WebSocket API |

Other commands:

```powershell
npm run build      # production build of the frontend (output: dist/)
npm run preview    # preview the production build
npm test           # end-to-end self test (camera + detection + alerts)
```

> If a port is still in use from an earlier run, stop it first:
> `Get-NetTCPConnection -LocalPort 5173,5174 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }`

---

## 3. Using the dashboard

1. **CAMERA SOURCE** – choose *Laptop / USB webcam*, *IP camera (RTSP)* or *Video file*.
   * Webcam: enter the index (`0`, `1`, …). The detected device names are listed
     underneath – the index is resolved to the real DirectShow device automatically.
   * RTSP: enter e.g. `rtsp://192.168.1.100:554/stream1` plus user name/password
     (they are inserted into the URL automatically).
   * File: absolute path to a video (it is replayed in a loop).
2. **Connect Camera** – the capture starts immediately and the live feed appears.
3. **Start/Stop Monitoring** (top right) – switches the detection on and off.
4. The status strip shows **green** = no distraction, **yellow** = camera/monitoring
   problem, **red** = sleeping or phone detected.
5. The camera card shows the live EAR values, the blink score and whether the eyes
   are open or closed.

### Audio alerts

* Put your files into `assets/` (`sleep_alert.mp3`, `phone_alert.mp3`) **or**
  upload them with the **Upload** button in the *AUDIO ALERTS* panel.
* **Test** plays the file immediately.
* If the browser blocks the sound, click once anywhere on the page (browsers
  require a user gesture before playing audio) and make sure *Sound on* is
  checked in the header.

---

## 4. Detection settings

All values are live-editable in the *DETECTION SETTINGS* panel and are stored in
`config/web-settings.json`.

| Setting | Default | Meaning |
| --- | --- | --- |
| Eye closed threshold (EAR) | 0.21 | EAR below which an eye counts as closed |
| Eye open threshold (EAR) | 0.28 | Re-open threshold (hysteresis) |
| Blink score threshold | 0.55 | MediaPipe `eyeBlink` score above which an eye counts as closed |
| Adaptive EAR ratio | 0.62 | Closed = baseline × this ratio (adapts to your face) |
| Min. eye closure (s) | 2.5 | How long the eyes must stay closed before the sleep alert fires |
| Sleep alert cooldown (s) | 30 | Minimum time between two sleep sounds |
| Phone confidence | 0.40 | Minimum YOLO confidence for a phone detection |
| Min. phone duration (s) | 1.5 | How long a phone must stay visible before the alert fires |
| Phone alert cooldown (s) | 30 | Minimum time between two phone sounds |
| Phone check interval (ms) | 500 | How often the server runs YOLO (lower = more accurate, slower) |

Typical tuning: too many sleep alarms → raise *Eye closed threshold* to 0.23 and
*Min. eye closure* to 3.5; sleep not detected → lower the threshold to 0.19 and
the duration to 1.5; phone not detected → lower *Phone confidence* to 0.30.

---

## 5. RTSP IP cameras – practical guide

1. Open the camera's app/web interface and find the **Network / RTSP / Live View**
   page. Typical URLs:
   * Hikvision: `rtsp://user:pass@IP:554/Streaming/Channels/101`
   * Dahua: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
   * Tapo: `rtsp://user:pass@IP:554/stream1`
   * Reolink: `rtsp://user:pass@IP:554/h264Preview_01_main`
2. Check that the PC can reach the camera:
   ```powershell
   ping 192.168.1.100
   Test-NetConnection -ComputerName 192.168.1.100 -Port 554
   ```
3. Enter the URL **without** credentials, type the user name and password into the
   two fields below – the application builds the final URL itself.
4. Press **Connect Camera**. If the stream cannot be opened, the camera card
   turns yellow and the ffmpeg error message is shown under the camera name.
   The app retries automatically every 3 seconds.

---

## 6. Project layout

```
study_tracker/
├── package.json              # npm scripts (dev / build / test)
├── vite.config.js            # dev server + proxy to the API server
├── index.html
├── src/                      # React dashboard
│   ├── App.jsx               # UI, WebSocket client, frame pipeline
│   ├── faceDetection.js      # MediaPipe Face Landmarker + EAR + blink logic
│   ├── api.js                # REST helpers
│   └── styles.css
├── server/
│   ├── index.js              # Express + WebSocket server
│   ├── cameraManager.js      # ffmpeg capture (webcam / RTSP / file) + reconnect
│   ├── phoneDetector.js      # YOLO ONNX Runtime (cell phone, class 67)
│   ├── alertEngine.js        # cooldowns + alert history
│   └── settings.js           # settings persistence
├── public/wasm/              # MediaPipe WASM runtime (copied by postinstall)
├── models/
│   ├── face_landmarker.task  # MediaPipe face landmarks (3.6 MB)
│   ├── yolo11n.onnx          # YOLO nano for the phone detection (11 MB)
│   └── yolo11n.onnx.data
├── assets/                   # your alert sounds (sleep_alert.mp3, phone_alert.mp3)
├── config/web-settings.json  # saved settings
├── scripts/                  # copy-wasm.mjs, test.mjs
└── desktop-version/          # the original Python/Tkinter desktop app (optional)
```

### Where does what run?

| Task | Runs in | Why |
| --- | --- | --- |
| Camera capture (webcam/RTSP/file) | server (ffmpeg) | ffmpeg handles RTSP and DirectShow reliably |
| Phone detection (YOLO) | server (ONNX Runtime) | no browser download, very fast on CPU |
| Face/EAR detection | browser (MediaPipe WASM) | runs at ~60 fps on the GPU, no server load |
| Alert decision + history | server | single source of truth for cooldowns |
| Sound playback | browser | plays your local files directly |

---

## 7. Common problems

| Problem | Solution |
| --- | --- |
| `Error opening input files` / camera stays yellow | Another app (Zoom/Teams/Chrome) holds the camera – close it and reconnect. The device list under the source field shows the real camera names. |
| `Could not connect to the RTSP camera` | Wrong URL/credentials, wrong port (554) or the RTSP service is disabled in the camera settings. Test with `Test-NetConnection ... -Port 554`. |
| No sound | Click once on the page (browser autoplay policy), check *Sound on*, and make sure the file is in `assets/`. |
| Phone not detected | Lower *Phone confidence* to 0.30 and *Phone check interval* to 250 ms; the phone must be clearly visible. |
| Slow / low fps | Raise *Phone check interval* to 1000 ms; close other camera apps. |
| `EADDRINUSE` on 5174 | An older instance is running – kill it with the command in section 2. |
| Nothing happens after `npm install` | The `postinstall` script copies the MediaPipe WASM runtime to `public/wasm`; run `node scripts/copy-wasm.mjs` manually if it was skipped. |

---

## 8. Desktop version (Python/Tkinter)

The original Tkinter desktop application is kept in `desktop-version/` and still
works with `python main.py` (it uses the same `assets/` and `models/` folders).
The web edition above is the primary application.

---

## 9. Credits

* [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe) – face landmarks (Apache 2.0)
* [Ultralytics YOLO](https://www.ultralytics.com/) – YOLO11 nano weights (AGPL-3.0 / Enterprise)
* [ONNX Runtime Node](https://onnxruntime.ai/) – inference (MIT)
* [FFmpeg](https://ffmpeg.org/) – camera capture (LGPL/GPL)
* [React](https://react.dev/) + [Vite](https://vite.dev/) – frontend (MIT)
#   s t u d y t r a c k e r  
 