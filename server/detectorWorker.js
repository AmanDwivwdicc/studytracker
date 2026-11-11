// Runs the YOLO phone detector in a separate thread.
// The ONNX inference needs ~0.4 s on this CPU - running it on the main thread
// would block the WebSocket server and make the video stream stutter.
import { parentPort } from "node:worker_threads";
import { PhoneDetector } from "./phoneDetector.js";

const detector = new PhoneDetector({ size: 320, confidence: 0.4 });
await detector.init();

parentPort.on("message", (msg) => {
  const { id, buffer, confidence, size } = msg || {};
  if (typeof confidence === "number") detector.setConfidence(confidence);
  if (typeof size === "number") detector.setSize(size);
  const jpegBuffer = Buffer.from(buffer);
  detector
    .detect(jpegBuffer)
    .then((boxes) => parentPort.postMessage({ id, boxes }))
    .catch(() => parentPort.postMessage({ id, boxes: [] }));
});
