// Face / eye detection in the browser with MediaPipe Face Landmarker (WASM).
//
// The landmarker gives us 468 face points; from those we calculate the Eye
// Aspect Ratio (EAR) for both eyes and read the "eyeBlink" blendshape,
// which is a well calibrated second signal for closed eyes.
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380];

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function eyeAspectRatio(eye) {
  const [p1, p2, p3, p4, p5, p6] = eye;
  const horizontal = distance(p1, p4);
  if (horizontal <= 1e-6) return 0;
  return (distance(p2, p6) + distance(p3, p5)) / (2 * horizontal);
}

let landmarkerPromise = null;

export function loadFaceLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks("/wasm");
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: "/models/face_landmarker.task" },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.5,
      });
    })();
  }
  return landmarkerPromise;
}

// Keeps the closed-eye state machine (hysteresis + duration) for one camera.
export class SleepTracker {
  constructor(settings) {
    this.settings = settings;
    this.closed = false;
    this.closedSince = 0;
    this.baseline = 0;
    this.lastResult = {
      faceFound: false,
      leftEar: 0,
      rightEar: 0,
      ear: 0,
      blink: 0,
      eyesClosed: false,
      sleeping: false,
    };
  }

  updateSettings(settings) {
    this.settings = settings;
  }

  reset() {
    this.closed = false;
    this.closedSince = 0;
  }

  // frame: ImageData-like object, timestampMs: strictly increasing number.
  detect(landmarker, frame, timestampMs) {
    const result = landmarker.detectForVideo(frame, timestampMs);
    const faces = result.faceLandmarks || [];
    if (!faces.length) {
      this.reset();
      this.lastResult = {
        faceFound: false,
        leftEar: 0,
        rightEar: 0,
        ear: 0,
        blink: 0,
        eyesClosed: false,
        sleeping: false,
      };
      return this.lastResult;
    }

    const landmarks = faces[0];
    const leftEar = eyeAspectRatio(LEFT_EYE_IDS.map((i) => landmarks[i]));
    const rightEar = eyeAspectRatio(RIGHT_EYE_IDS.map((i) => landmarks[i]));

    let blinkLeft = 0;
    let blinkRight = 0;
    const shapes = result.faceBlendshapes?.[0]?.categories || [];
    for (const shape of shapes) {
      if (shape.categoryName === "eyeBlinkLeft") blinkLeft = shape.score;
      if (shape.categoryName === "eyeBlinkRight") blinkRight = shape.score;
    }

    const measure = this.settings.require_both_eyes
      ? Math.max(leftEar, rightEar)
      : (leftEar + rightEar) / 2;
    const blink = this.settings.require_both_eyes
      ? Math.min(blinkLeft, blinkRight)
      : Math.max(blinkLeft, blinkRight);

    // Adaptive threshold based on the personal open-eye baseline.
    let closedThreshold = this.settings.ear_closed_threshold;
    let openThreshold = this.settings.ear_open_threshold;
    if (this.settings.use_adaptive_threshold && this.baseline > 0) {
      const adaptive = this.baseline * this.settings.adaptive_ratio;
      if (adaptive < closedThreshold) {
        closedThreshold = adaptive;
        openThreshold = Math.max(closedThreshold * 1.25, closedThreshold + 0.01);
      }
    }

    const blinkSaysClosed = blink >= this.settings.blink_threshold;
    const now = performance.now();

    if (!this.closed && (measure < closedThreshold || blinkSaysClosed)) {
      this.closed = true;
      this.closedSince = now;
    } else if (this.closed && measure > openThreshold && !blinkSaysClosed) {
      this.closed = false;
      this.closedSince = 0;
    }

    if (measure > this.settings.ear_closed_threshold) {
      this.baseline = Math.max(this.baseline * 0.995, measure);
    }

    const heldMs = this.closed ? now - this.closedSince : 0;
    const sleeping = this.closed && heldMs >= this.settings.min_eye_closure_duration * 1000;

    this.lastResult = {
      faceFound: true,
      leftEar,
      rightEar,
      ear: measure,
      blink: Math.max(blinkLeft, blinkRight),
      eyesClosed: this.closed,
      sleeping,
    };
    return this.lastResult;
  }
}
