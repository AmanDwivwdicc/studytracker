// Probe: does MediaPipe Face Landmarker + onnxruntime-node work in Node?
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as ort from "onnxruntime-node";
import fs from "node:fs";
import path from "node:path";

const root = "C:\\Users\\LENOVO\\study_tracker";

const vision = await FilesetResolver.forVisionTasks(
  path.join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm")
);
const landmarker = await FaceLandmarker.createFromOptions(vision, {
  baseOptions: { modelAssetPath: path.join(root, "models", "face_landmarker.task") },
  runningMode: "IMAGE",
  numFaces: 1,
  outputFaceBlendshapes: true,
});
console.log("FaceLandmarker: OK");

const session = await ort.InferenceSession.create(path.join(root, "models", "yolo11n.onnx"), {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
});
console.log("ORT session: OK", session.inputNames, session.outputNames);

const input = new ort.Tensor("float32", new Float32Array(3 * 320 * 320), [1, 3, 320, 320]);
const out = await session.run({ [session.inputNames[0]]: input });
console.log("ORT run: OK", out[session.outputNames[0]].dims);
