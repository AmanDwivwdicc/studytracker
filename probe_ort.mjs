// Probe: ONNX Runtime in Node with the exported YOLO model.
import * as ort from "onnxruntime-node";
import fs from "node:fs";
import jpeg from "jpeg-js";

const root = "C:\\Users\\LENOVO\\study_tracker";
const modelPath = `${root}\\models\\yolo11n.onnx`;
console.log("model exists:", fs.existsSync(modelPath));
console.log("external data:", fs.existsSync(`${root}\\models\\yolo11n.onnx.data`));

const session = await ort.InferenceSession.create(modelPath, {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
  intraOpNumThreads: 4,
});
console.log("inputs:", session.inputNames);
console.log("outputs:", session.outputNames);

const meta = session.inputMetadata?.[0];
console.log("input dims:", meta?.dimensions);

const size = 320;
const raw = fs.readFileSync("C:\\Users\\LENOVO\\AppData\\Local\\Temp\\opencode\\bus.jpg");
const img = jpeg.decode(raw, { useTArray: true });
console.log("jpeg decoded:", img.width, img.height);

const tensorData = new Float32Array(3 * size * size);
for (let y = 0; y < size; y++) {
  for (let x = 0; x < size; x++) {
    const si = (y * size + x) * 4;
    const di = y * size + x;
    tensorData[di] = img.data[si] / 255;
    tensorData[size * size + di] = img.data[si + 1] / 255;
    tensorData[2 * size * size + di] = img.data[si + 2] / 255;
  }
}
const input = new ort.Tensor("float32", tensorData, [1, 3, size, size]);

const t0 = Date.now();
const out = await session.run({ [session.inputNames[0]]: input });
console.log("run ms:", Date.now() - t0);
const tensor = out[session.outputNames[0]];
console.log("output dims:", tensor.dims);
const data = tensor.data;
const numPred = tensor.dims[2];
let phones = 0;
for (let i = 0; i < numPred; i++) {
  const base = i * 84;
  let best = 0;
  let bestIdx = -1;
  for (let c = 0; c < 80; c++) {
    const score = data[base + 4 + c];
    if (score > best) { best = score; bestIdx = c; }
  }
  if (bestIdx === 67 && best > 0.1) { phones++; console.log("cell phone score", best.toFixed(2)); }
}
console.log("phone predictions:", phones);
