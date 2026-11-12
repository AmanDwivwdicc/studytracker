import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, assetUrl, initBackend } from "./api.js";
import { backendWsUrl } from "./backend.js";
import { loadFaceLandmarker, SleepTracker } from "./faceDetection.js";

const LEVEL_COLOR = { green: "#3cc85a", yellow: "#f2c744", red: "#f23c3c" };

// Insert or update a camera without creating duplicate list entries.
function upsertCamera(list, camera) {
  const index = list.findIndex((c) => c.id === camera.id);
  if (index < 0) return [...list, camera];
  const copy = [...list];
  copy[index] = { ...copy[index], ...camera };
  return copy;
}

function useWebSocket(onMessage) {
  const socketRef = useRef(null);
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  useEffect(() => {
    let closed = false;
    let retry;

    const connect = async () => {
      await initBackend();
      if (closed) return;
      const socketUrl = backendWsUrl();
      if (!socketUrl) return;
      const socket = new WebSocket(socketUrl);
      socket.binaryType = "arraybuffer";
      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            handlerRef.current(JSON.parse(event.data));
          } catch {
            /* ignore malformed messages */
          }
        } else if (event.data instanceof ArrayBuffer) {
          handlerRef.current({ type: "binary", buffer: event.data });
        }
      };
      socket.onclose = () => {
        if (!closed) retry = setTimeout(connect, 2000);
      };
      socket.onerror = () => socket.close();
      socketRef.current = socket;
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      socketRef.current?.close();
    };
  }, []);

  return socketRef;
}

export default function App() {
  const [settings, setSettings] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [audioFiles, setAudioFiles] = useState([]);
  const [hasSignal, setHasSignal] = useState({});
  const [phoneBoxes, setPhoneBoxes] = useState({});
  const [eyeStatus, setEyeStatus] = useState({});
  const [statusMsg, setStatusMsg] = useState("");
  const [monitoring, setMonitoring] = useState(true);
  const [sourceType, setSourceType] = useState("browser");
  const [sourceValue, setSourceValue] = useState("0");
  const [sourceName, setSourceName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [webcams, setWebcams] = useState([]);
  const [audioReady, setAudioReady] = useState({});
  const [soundEnabled, setSoundEnabled] = useState(true);

  const landmarkerRef = useRef(null);
  const trackersRef = useRef(new Map());
  const frameTimesRef = useRef(new Map());
  const detectorsRef = useRef(new Map());
  const settingsRef = useRef(null);
  const monitoringRef = useRef(true);
  const audioRef = useRef(new Map());
  const canvasRef = useRef(new Map());
  const pendingRef = useRef(new Map());
  const phoneBoxesRef = useRef({});
  const viewRef = useRef(new Map());
  const signalRef = useRef({});
  const detectionTimesRef = useRef(new Map());

  settingsRef.current = settings;
  monitoringRef.current = monitoring;
  phoneBoxesRef.current = phoneBoxes;

  // ------------------------------------------------------------- bootstrap
  const [bootError, setBootError] = useState("");
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setBootError("");
    (async () => {
      try {
        await initBackend();
        const state = await api.state();
        if (cancelled) return;
        setSettings(state.settings);
        setCameras(state.cameras);
        setAlerts(state.alerts || []);
        setAudioFiles(state.audio.files || []);
      } catch (err) {
        if (cancelled) return;
        setBootError(err.message || "The backend is not reachable.");
      }
      try {
        const data = await api.webcams();
        if (!cancelled) setWebcams(data.devices || []);
      } catch {
        if (!cancelled) setWebcams([]);
      }
      try {
        const landmarker = await loadFaceLandmarker();
        if (!cancelled) landmarkerRef.current = landmarker;
      } catch (err) {
        if (!cancelled) setStatusMsg(`MediaPipe could not be loaded: ${err.message}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [retryCount]);

  // --------------------------------------------------------------- audio
  const playFile = useCallback(async (name) => {
    if (!name) return;
    let audio = audioRef.current.get(name);
    if (!audio) {
      audio = new Audio(await assetUrl(name));
      audioRef.current.set(name, audio);
    }
    audio.currentTime = 0;
    audio.play().catch(() => setStatusMsg("The browser blocked the sound - click anywhere on the page first."));
  }, []);

  const playAlert = useCallback(
    (event) => {
      if (!soundEnabled) return;
      playFile(event.audio);
    },
    [playFile, soundEnabled]
  );

  // Binary frames arrive as: 4 byte length + JSON header + JPEG data.
  const handleBinary = useCallback((buffer) => {
    try {
      const view = new DataView(buffer);
      const headerLength = view.getUint32(0, false);
      const header = new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength));
      const { cameraId } = JSON.parse(header);
      const blob = new Blob([new Uint8Array(buffer, 4 + headerLength)], { type: "image/jpeg" });
      pendingRef.current.set(cameraId, { cameraId, blob });
    } catch {
      /* ignore malformed frames */
    }
  }, []);

  const handleMessage = useCallback(
    (message) => {
      if (message.type === "binary") {
        handleBinary(message.buffer);
        return;
      }
      if (message.type === "hello") {
        setSettings(message.settings);
        setCameras(message.cameras || []);
        setAlerts(message.alerts || []);
        return;
      }
      if (message.type === "settings") {
        setSettings(message.settings);
        return;
      }
      if (message.type === "camera-status") {
        setCameras((prev) => upsertCamera(prev, message.camera));
        return;
      }
      if (message.type === "phone") {
        setPhoneBoxes((prev) => ({ ...prev, [message.cameraId]: message.boxes }));
        return;
      }
      if (message.type === "alert") {
        setAlerts((prev) => [message.event, ...prev].slice(0, 100));
        playAlert(message.event);
        return;
      }
    },
    [handleBinary, playAlert]
  );

  const socketRef = useWebSocket(handleMessage);

  // ------------------------------------------------------- browser camera
  const browserStreamsRef = useRef(new Map());

  const stopBrowserCamera = useCallback((cameraId) => {
    const entry = browserStreamsRef.current.get(cameraId);
    if (!entry) return;
    browserStreamsRef.current.delete(cameraId);
    clearInterval(entry.timer);
    if (entry.video) {
      entry.video.srcObject = null;
      entry.video.remove();
    }
  }, []);

  const startBrowserCamera = useCallback(
    async (cameraId) => {
      if (browserStreamsRef.current.has(cameraId)) return;
      if (!navigator.mediaDevices?.getUserMedia) {
        setStatusMsg("This browser does not allow camera access.");
        return;
      }
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (err) {
        setStatusMsg(`Camera access was denied: ${err.message}`);
        return;
      }
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      await video.play().catch(() => {});
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      const timer = setInterval(async () => {
        if (!video.videoWidth) return;
        const width = 640;
        const height = Math.round((video.videoHeight / video.videoWidth) * width);
        canvas.width = width;
        canvas.height = height;
        context.drawImage(video, 0, 0, width, height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.7));
        if (!blob) return;
        try {
          await api.pushFrame(cameraId, blob);
        } catch {
          /* the server may be restarting - the next frame retries */
        }
      }, 200);
      browserStreamsRef.current.set(cameraId, { stream, video, timer });
    },
    []
  );

  // ------------------------------------------------------------- cameras
  const connectCamera = async () => {
    try {
      const result = await api.addCamera({
        type: sourceType,
        value: sourceValue,
        name: sourceName,
        username,
        password,
      });
      setCameras((prev) => upsertCamera(prev, result.camera));
      setStatusMsg(`Camera "${result.camera.name}" connected.`);
      if (result.camera.type === "browser") {
        startBrowserCamera(result.camera.id);
      }
    } catch (err) {
      setStatusMsg(err.message);
    }
  };

  const disconnectCamera = async (id) => {
    stopBrowserCamera(id);
    await api.removeCamera(id);
    setCameras((prev) => prev.filter((c) => c.id !== id));
    setHasSignal((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });
    signalRef.current[id] = false;
    setPhoneBoxes((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });
    trackersRef.current.delete(id);
  };

  // ------------------------------------------------------ frame handling
  useEffect(() => {
    const timer = setInterval(() => {
      // Process the newest frame of every camera.
      for (const [cameraId, buffer] of Array.from(pendingRef.current.entries())) {
        pendingRef.current.delete(cameraId);

        createImageBitmap(buffer.blob)
        .then(async (bitmap) => {
          // Use an offscreen canvas per camera for decoding and drawing.
          let canvas = canvasRef.current.get(cameraId);
          if (!canvas) {
            canvas = document.createElement("canvas");
            canvasRef.current.set(cameraId, canvas);
          }
          const context = canvas.getContext("2d", { willReadFrequently: true });
          const width = 480;
          const height = Math.round((bitmap.height / bitmap.width) * width);
          canvas.width = width;
          canvas.height = height;
          context.drawImage(bitmap, 0, 0, width, height);
          const imageData = context.getImageData(0, 0, width, height);

          // Phone boxes are drawn onto the canvas.
          const boxes = phoneBoxesRef.current[cameraId] || [];
          const scaleX = width;
          const scaleY = height;
          context.lineWidth = 2;
          context.font = "14px Segoe UI";
          boxes.forEach((box) => {
            const x = box.x1 * scaleX;
            const y = box.y1 * scaleY;
            const w = (box.x2 - box.x1) * scaleX;
            const h = (box.y2 - box.y1) * scaleY;
            context.strokeStyle = "#f23c3c";
            context.strokeRect(x, y, w, h);
            context.fillStyle = "#f23c3c";
            context.fillText(`Mobile phone ${Math.round(box.confidence * 100)}%`, x, Math.max(14, y - 4));
          });

          // The canvas element in the camera card shows the picture.
          const view = viewRef.current.get(cameraId);
          if (view) {
            const viewContext = view.getContext("2d");
            if (view.width !== width || view.height !== height) {
              view.width = width;
              view.height = height;
            }
            viewContext.drawImage(canvas, 0, 0);
          }
          if (!signalRef.current[cameraId]) {
            signalRef.current[cameraId] = true;
            setHasSignal((prev) => ({ ...prev, [cameraId]: true }));
          }

          // Face / eye detection (MediaPipe, runs in the browser).
          if (!landmarkerRef.current || !monitoringRef.current) return;
          const detectionTimes = detectionTimesRef.current;
          const now = performance.now();
          const lastDetection = detectionTimes.get(cameraId) || 0;
          if (now - lastDetection < 150) return;
          detectionTimes.set(cameraId, now);

          let tracker = trackersRef.current.get(cameraId);
          if (!tracker) {
            tracker = new SleepTracker(settingsRef.current);
            trackersRef.current.set(cameraId, tracker);
          }
          const times = frameTimesRef.current;
          const last = times.get(cameraId) || 0;
          const timestamp = Math.max(last + 1, Math.round(performance.now()));
          times.set(cameraId, timestamp);
          const result = tracker.detect(landmarkerRef.current, imageData, timestamp);
          setEyeStatus((prev) => ({ ...prev, [cameraId]: result }));

          const socket = socketRef.current;
          if (socket && socket.readyState === 1) {
            socket.send(
              JSON.stringify({
                type: "eye-state",
                cameraId,
                cameraName: cameraId,
                closed: result.sleeping,
                ear: result.ear,
                blink: result.blink,
              })
            );
          }
        })
        .catch(() => {});
      }
    }, 120);
    return () => clearInterval(timer);
  }, [socketRef]);

  // ------------------------------------------------------------ settings
  const updateSetting = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const applySettings = async () => {
    try {
      const saved = await api.saveSettings(settings);
      setSettings(saved);
      trackersRef.current.forEach((tracker) => tracker.updateSettings(saved));
      setStatusMsg("Settings applied.");
    } catch (err) {
      setStatusMsg(err.message);
    }
  };

  const uploadAudio = async (kind, event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const result = await api.uploadAudio(file.name, file);
      setAudioFiles(result.files);
      updateSetting(kind, file.name);
      setAudioReady((prev) => ({ ...prev, [kind]: true }));
      setStatusMsg(`Audio file "${file.name}" uploaded.`);
    } catch (err) {
      setStatusMsg(err.message);
    }
  };

  const clearAlerts = async () => {
    await api.clearAlerts();
    setAlerts([]);
  };

  if (!settings) {
    return (
      <div className="loading">
        {bootError ? (
          <>
            <h2>Cannot reach the Study Tracker server</h2>
            <p className="muted">{bootError}</p>
            <p className="muted">
              Check that the backend is running and that the frontend knows its address
              (VITE_API_BASE_URL or public/backend.json).
            </p>
            <button className="btn primary" onClick={() => setRetryCount((n) => n + 1)}>
              Retry
            </button>
          </>
        ) : (
          "Loading STUDY TRACKER…"
        )}
      </div>
    );
  }

  const sleepActive = Object.values(eyeStatus).some((s) => s.sleeping);
  const phoneActive = Object.values(phoneBoxes).some((boxes) => boxes.length > 0);
  const anyDisconnected = cameras.some((c) => c.status !== "connected");

  return (
    <div className="app">
      <header className="topbar">
        <h1>STUDY TRACKER</h1>
        <span className="subtitle">AI powered study monitoring - runs fully offline</span>
        <div className="topbar-actions">
          <button
            className={monitoring ? "btn danger" : "btn primary"}
            onClick={() => setMonitoring((m) => !m)}
          >
            {monitoring ? "Stop Monitoring" : "Start Monitoring"}
          </button>
          <label className="toggle">
            <input type="checkbox" checked={soundEnabled} onChange={(e) => setSoundEnabled(e.target.checked)} />
            Sound {soundEnabled ? "on" : "off"}
          </label>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <section className="panel">
            <h2>CAMERA SOURCE</h2>
            <select value={sourceType} onChange={(e) => setSourceType(e.target.value)}>
              <option value="browser">Browser camera (this device)</option>
              <option value="webcam">Laptop / USB webcam (server)</option>
              <option value="rtsp">IP camera (RTSP)</option>
              <option value="file">Video file (test)</option>
            </select>

            {sourceType === "browser" ? (
              <p className="muted small">
                Uses the camera of this computer through the browser. Works with a hosted backend.
              </p>
            ) : (
              <>
                <label>
                  {sourceType === "webcam"
                    ? "Camera index or device name"
                    : sourceType === "rtsp"
                      ? "RTSP URL"
                      : "Video file path"}
                </label>
                <input
                  value={sourceValue}
                  onChange={(e) => setSourceValue(e.target.value)}
                  placeholder={sourceType === "rtsp" ? "rtsp://192.168.1.100:554/stream1" : "0"}
                />
                {sourceType === "webcam" &&
                  (webcams.length > 0 ? (
                    <select className="devices" onChange={(e) => setSourceValue(e.target.value)}>
                      <option value="">detected devices:</option>
                      {webcams.map((device) => (
                        <option key={device.name} value={device.index}>
                          {device.index}: {device.name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <p className="muted small">
                      No camera is attached to the server. Use "Browser camera" or an RTSP camera.
                    </p>
                  ))}
              </>
            )}

            <label>Camera name (optional)</label>
            <input value={sourceName} onChange={(e) => setSourceName(e.target.value)} />

            {sourceType === "rtsp" && (
              <>
                <label>RTSP user name</label>
                <input value={username} onChange={(e) => setUsername(e.target.value)} />
                <label>RTSP password</label>
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
              </>
            )}

            <button className="btn primary block" onClick={connectCamera}>
              Connect Camera
            </button>
          </section>

          <section className="panel">
            <h2>CONNECTED CAMERAS</h2>
            {cameras.length === 0 && <p className="muted">No camera connected yet.</p>}
            <ul className="camera-list">
              {cameras.map((camera) => (
                <li key={camera.id}>
                  <div>
                    <strong>{camera.name}</strong>
                    <span className={`badge ${camera.status}`}>{camera.status}</span>
                    <small>
                      {camera.type} · {camera.fps} fps
                    </small>
                    {camera.error && <small className="error">{camera.error}</small>}
                  </div>
                  <button className="btn small" onClick={() => disconnectCamera(camera.id)}>
                    Disconnect
                  </button>
                </li>
              ))}
            </ul>
          </section>

          <section className="panel">
            <h2>AUDIO ALERTS</h2>
            {["sleep", "phone"].map((kind) => (
              <div key={kind} className="audio-row">
                <label>{kind === "sleep" ? "Sleep alert file" : "Phone alert file"}</label>
                <input readOnly value={settings[kind === "sleep" ? "sleep_audio" : "phone_audio"]} />
                <div className="row">
                  <label className="file-button">
                    Upload
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => uploadAudio(kind === "sleep" ? "sleep_audio" : "phone_audio", e)}
                    />
                  </label>
                  <button className="btn small" onClick={() => playFile(settings[kind === "sleep" ? "sleep_audio" : "phone_audio"])}>
                    Test
                  </button>
                </div>
              </div>
            ))}
            {audioFiles.length > 0 && (
              <p className="muted small">Available in assets/: {audioFiles.map((f) => f.name).join(", ")}</p>
            )}
          </section>

          <section className="panel">
            <h2>DETECTION SETTINGS</h2>
            {[
              ["ear_closed_threshold", "Eye closed threshold (EAR)", "number"],
              ["ear_open_threshold", "Eye open threshold (EAR)", "number"],
              ["blink_threshold", "Blink score threshold", "number"],
              ["adaptive_ratio", "Adaptive EAR ratio", "number"],
              ["min_eye_closure_duration", "Min. eye closure (s)", "number"],
              ["sleep_alert_cooldown", "Sleep alert cooldown (s)", "number"],
              ["phone_confidence_threshold", "Phone confidence", "number"],
              ["min_phone_detection_duration", "Min. phone duration (s)", "number"],
              ["phone_alert_cooldown", "Phone alert cooldown (s)", "number"],
              ["phone_detection_interval", "Phone check interval (ms)", "number"],
            ].map(([key, label, type]) => (
              <div key={key} className="setting">
                <label>{label}</label>
                <input
                  type={type}
                  step="0.01"
                  value={settings[key]}
                  onChange={(e) => updateSetting(key, Number(e.target.value))}
                />
              </div>
            ))}
            <button className="btn primary block" onClick={applySettings}>
              Apply Settings
            </button>
          </section>
        </aside>

        <main className="content">
          <div className="statusbar">
            <StatusTile
              label="Monitoring"
              value={monitoring ? (anyDisconnected ? "camera issue" : "active") : "stopped"}
              color={monitoring ? (anyDisconnected ? LEVEL_COLOR.yellow : LEVEL_COLOR.green) : LEVEL_COLOR.yellow}
            />
            <StatusTile
              label="Sleep detection"
              value={sleepActive ? "SLEEPING" : "awake"}
              color={sleepActive ? LEVEL_COLOR.red : LEVEL_COLOR.green}
            />
            <StatusTile
              label="Phone detection"
              value={phoneActive ? "PHONE" : "clear"}
              color={phoneActive ? LEVEL_COLOR.red : LEVEL_COLOR.green}
            />
          </div>

          {statusMsg && <div className="notice">{statusMsg}</div>}

          <section className="videos">
            {cameras.length === 0 && (
              <div className="placeholder">
                No camera connected.
                <br />
                Use the controls on the left to connect a laptop webcam, an RTSP IP camera or a test video.
              </div>
            )}
            {cameras.map((camera) => (
              <div className="camera-card" key={camera.id}>
                <div className="camera-head">
                  <strong>{camera.name}</strong>
                  <span className={`badge ${camera.status}`}>{camera.status}</span>
                  <span className="metrics">
                    {eyeStatus[camera.id]?.faceFound
                      ? `EAR ${eyeStatus[camera.id].leftEar.toFixed(3)}/${eyeStatus[camera.id].rightEar.toFixed(3)} · blink ${eyeStatus[camera.id].blink.toFixed(2)} · ${
                          eyeStatus[camera.id].eyesClosed ? "CLOSED" : "open"
                        }`
                      : "no face detected"}
                  </span>
                </div>
                <div className="feed">
                  <canvas
                    className="feed-canvas"
                    ref={(element) => {
                      if (element) viewRef.current.set(camera.id, element);
                      else viewRef.current.delete(camera.id);
                    }}
                  />
                  {!hasSignal[camera.id] && <div className="no-signal">No signal</div>}
                  {eyeStatus[camera.id]?.sleeping && <div className="overlay red">WAKE UP! YOU ARE SLEEPING.</div>}
                  {phoneBoxes[camera.id]?.length > 0 && (
                    <div className="overlay red">WARNING! MOBILE PHONE DETECTED.</div>
                  )}
                </div>
              </div>
            ))}
          </section>

          <section className="panel history">
            <div className="history-head">
              <h2>ALERT HISTORY</h2>
              <button className="btn small" onClick={clearAlerts}>
                Clear history
              </button>
            </div>
            {alerts.length === 0 ? (
              <p className="muted">No alerts yet.</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Camera</th>
                    <th>Type</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.id}>
                      <td>{alert.timestamp}</td>
                      <td>{alert.cameraName || alert.cameraId}</td>
                      <td className={alert.type}>{alert.type.toUpperCase()}</td>
                      <td>{alert.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}

function StatusTile({ label, value, color }) {
  return (
    <div className="tile">
      <span className="tile-label">{label}</span>
      <span className="tile-value" style={{ color }}>
        {value}
      </span>
    </div>
  );
}
