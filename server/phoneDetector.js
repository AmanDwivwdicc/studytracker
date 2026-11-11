// Mobile phone detection with YOLO (ONNX Runtime, Node).
//
// The COCO class "cell phone" has index 67.  The model is the exported
// YOLO11-nano ONNX file in the models folder - it runs on the CPU and needs
// no Python installation.
import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";
import jpeg from "jpeg-js";
import { MODELS_DIR } from "./settings.js";

export const CELL_PHONE_CLASS_ID = 67;

// Keep only the strongest box when several overlap (same phone detected twice).
function nonMaxSuppression(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const box of sorted) {
    const overlaps = kept.some((other) => {
      const x1 = Math.max(box.x1, other.x1);
      const y1 = Math.max(box.y1, other.y1);
      const x2 = Math.min(box.x2, other.x2);
      const y2 = Math.min(box.y2, other.y2);
      const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const areaA = (box.x2 - box.x1) * (box.y2 - box.y1);
      const areaB = (other.x2 - other.x1) * (other.y2 - other.y1);
      const union = areaA + areaB - intersection;
      return union > 0 ? intersection / union : 0;
    });
    if (!overlaps) kept.push(box);
  }
  return kept;
}

const MODEL_CANDIDATES = ["yolo11n.onnx", "yolov8n.onnx", "yolo11n.onnx.data"];

function findModel() {
  for (const name of MODEL_CANDIDATES) {
    const p = path.join(MODELS_DIR, name);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

export class PhoneDetector {
  constructor({ size = 320, confidence = 0.4 } = {}) {
    this.size = size;
    this.confidence = confidence;
    this.session = null;
    this.inputName = null;
    this.outputName = null;
    this.inputSize = size;
    this.available = false;
    this.error = "";
    this.modelPath = findModel();
  }

  async init() {
    if (this.session) return this.available;
    if (!this.modelPath) {
      this.error = `No YOLO model found in ${MODELS_DIR}. Expected yolo11n.onnx.`;
      return false;
    }
    try {
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ["cpu"],
        graphOptimizationLevel: "all",
        intraOpNumThreads: Math.max(2, Math.min(4, (await import("node:os")).cpus().length)),
      });
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];

      // The exported model has a fixed input size - read it from the graph.
      const meta = this.session.inputMetadata?.[0];
      const dims = meta?.dimensions;
      if (Array.isArray(dims) && dims.length === 4) {
        const h = dims[2];
        const w = dims[3];
        if (typeof h === "number" && typeof w === "number" && h > 0 && w > 0) {
          this.inputSize = h;
        }
      }
      this.available = true;
      this.error = "";
      console.log(`[phone-detector] ONNX model ready (${path.basename(this.modelPath)}, input ${this.inputSize}px)`);
    } catch (err) {
      this.error = err.message;
      console.warn("[phone-detector] could not load model:", err.message);
    }
    return this.available;
  }

  setSize(size) {
    this.size = Math.min(Math.max(Number(size) || 320, 160), 640);
  }

  setConfidence(value) {
    this.confidence = Math.min(Math.max(Number(value) || 0.4, 0.05), 0.99);
  }

  // Detect mobile phones inside a JPEG buffer.  Returns a list of boxes with
  // normalised coordinates (0..1) and the confidence.
  async detect(jpegBuffer) {
    if (!this.available) {
      await this.init();
      if (!this.available) return [];
    }
    let image;
    try {
      image = jpeg.decode(jpegBuffer, { useTArray: true, formatAsRGBA: true });
    } catch (err) {
      return [];
    }

    const size = this.inputSize;
    const tensorData = new Float32Array(3 * size * size);
    const { width, height, data } = image;

    // Letterbox the image into the square network input.
    const scale = Math.min(size / width, size / height);
    const newW = Math.round(width * scale);
    const newH = Math.round(height * scale);
    const padX = Math.floor((size - newW) / 2);
    const padY = Math.floor((size - newH) / 2);

    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        const src = ((y / scale) | 0) * width + ((x / scale) | 0);
        const si = src * 4;
        const di = y * size + x + padY * size + padX;
        tensorData[di] = data[si] / 255;
        tensorData[size * size + di] = data[si + 1] / 255;
        tensorData[2 * size * size + di] = data[si + 2] / 255;
      }
    }

    const input = new ort.Tensor("float32", tensorData, [1, 3, size, size]);
    const outputs = await this.session.run({ [this.inputName]: input });
    const tensor = outputs[this.outputName];
    const dims = tensor.dims;
    const values = tensor.data;
    // YOLO11 output layout: [1, 4 + 80, anchors]
    const numClasses = dims[1] - 4;
    const numAnchors = dims[2];

    const boxes = [];
    const candidates = [];
    for (let i = 0; i < numAnchors; i++) {
      let bestScore = 0;
      let bestClass = -1;
      // Rows 0..3 hold the box coordinates - the class scores start at row 4.
      for (let c = 0; c < numClasses; c++) {
        const score = values[(c + 4) * numAnchors + i];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestClass !== CELL_PHONE_CLASS_ID || bestScore < this.confidence) continue;

      const cx = values[0 * numAnchors + i];
      const cy = values[1 * numAnchors + i];
      const w = values[2 * numAnchors + i];
      const h = values[3 * numAnchors + i];

      // Convert the letterboxed centre box back to the original image.
      const x1 = (cx - w / 2 - padX) / scale;
      const y1 = (cy - h / 2 - padY) / scale;
      const x2 = (cx + w / 2 - padX) / scale;
      const y2 = (cy + h / 2 - padY) / scale;
      candidates.push({
        x1: Math.max(0, x1 / width),
        y1: Math.max(0, y1 / height),
        x2: Math.min(width, x2) / width,
        y2: Math.min(height, y2) / height,
        confidence: bestScore,
      });
    }
    return nonMaxSuppression(candidates, 0.45);
  }
}
