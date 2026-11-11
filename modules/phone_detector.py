"""Mobile phone detection with a lightweight YOLO model.

The default backend is Ultralytics YOLO with the nano model (``yolo11n.pt``).
The weights are downloaded automatically on first use and stored in the
``models`` folder.  Only the COCO class **cell phone** (class index 67) is
reported - the other 79 classes are discarded.

An OpenCV DNN backend is included as a fallback: it can run an exported
YOLO ONNX file (``yolo11n.onnx``) without the Ultralytics dependency.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import MODELS_DIR

LOGGER = logging.getLogger(__name__)

# COCO index of the "cell phone" class (class 77 is "teddy bear").
CELL_PHONE_CLASS_ID = 67
CELL_PHONE_CLASS_NAME = "cell phone"

YOLO_NANO_WEIGHTS = MODELS_DIR / "yolo11n.pt"
YOLO_NANO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
ONNX_WEIGHTS = MODELS_DIR / "yolo11n.onnx"


@dataclass
class PhoneDetection:
    """One detected mobile phone."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def download_weights(url: str = YOLO_NANO_URL, target: Path = YOLO_NANO_WEIGHTS) -> bool:
    """Download the YOLO nano weights when they are not present yet."""
    target = Path(target)
    if target.exists() and target.stat().st_size > 1_000_000:
        return True
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading YOLO nano weights to %s", target)
        urllib.request.urlretrieve(url, target)
        LOGGER.info("Weights downloaded (%.1f MB)", target.stat().st_size / 1_048_576)
        return True
    except Exception as exc:
        LOGGER.warning("Could not download YOLO weights: %s", exc)
        return False


def _configure_torch_threads() -> None:
    """Give PyTorch more than the default two threads - it speeds up YOLO a lot."""
    try:
        import torch

        cpus = os.cpu_count() or 4
        torch.set_num_threads(max(2, min(8, cpus)))
    except Exception as exc:  # pragma: no cover - torch is optional
        LOGGER.debug("Could not configure torch threads: %s", exc)


class _UltralyticsBackend:
    """Runs YOLO through the Ultralytics Python package."""

    def __init__(self, weights: Path, confidence: float, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        _configure_torch_threads()
        weights = Path(weights)
        if not weights.exists():
            download_weights(target=weights)
        self._model = YOLO(str(weights))
        self._confidence = confidence
        self._imgsz = imgsz

    def warmup(self) -> None:
        """Run one inference so the first real frame is not delayed by the warm-up."""
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        try:
            self._model(blank, conf=self._confidence, classes=[CELL_PHONE_CLASS_ID],
                        imgsz=self._imgsz, verbose=False)
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Warm-up failed: %s", exc)

    def detect(self, frame: np.ndarray) -> List[PhoneDetection]:
        # Calling the model directly is noticeably faster than model.predict()
        # because it skips the extra bookkeeping of the prediction API.
        results = self._model(
            frame,
            conf=self._confidence,
            classes=[CELL_PHONE_CLASS_ID],
            imgsz=self._imgsz,
            verbose=False,
        )
        detections: List[PhoneDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
                continue
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                if int(cls) != CELL_PHONE_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                detections.append(
                    PhoneDetection(
                        x1=max(0, x1),
                        y1=max(0, y1),
                        x2=max(0, x2),
                        y2=max(0, y2),
                        confidence=float(conf),
                    )
                )
        return detections


class _OnnxBackend:
    """Runs an exported YOLO ONNX model through OpenCV DNN."""

    def __init__(self, weights: Path, confidence: float, imgsz: int = 640) -> None:
        self._net = cv2.dnn.readNetFromONNX(str(weights))
        self._confidence = confidence
        self._imgsz = imgsz
        self._classes = 80

    def warmup(self) -> None:
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        try:
            self.detect(blank)
        except Exception:  # pragma: no cover
            pass

    def detect(self, frame: np.ndarray) -> List[PhoneDetection]:
        height, width = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(frame, self._imgsz)
        self._net.setInput(blob)
        output = self._net.forward(self._net.getUnconnectedOutLayersNames())
        return self._decode(output[0], scale, pad_x, pad_y, width, height)

    def _letterbox(self, frame: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
        """Resize while keeping the aspect ratio (letterbox padding)."""
        height, width = frame.shape[:2]
        scale = min(size / height, size / width)
        new_w, new_h = int(width * scale), int(height * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (size - new_w) / 2
        pad_y = (size - new_h) / 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = int(pad_y), int(pad_x)
        canvas[top : top + new_h, left : left + new_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True, crop=False)
        return blob, scale, left, top

    def _decode(self, output: np.ndarray, scale: float, pad_x: int, pad_y: int,
                width: int, height: int) -> List[PhoneDetection]:
        """Decode YOLO output of shape (1, 84, N) into phone detections."""
        predictions = np.squeeze(output, axis=0).T  # (N, 84)
        boxes: List[List[int]] = []
        scores: List[float] = []
        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if class_id != CELL_PHONE_CLASS_ID or score < self._confidence:
                continue
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x1 = int((cx - w / 2 - pad_x) / scale)
            y1 = int((cy - h / 2 - pad_y) / scale)
            x2 = int((cx + w / 2 - pad_x) / scale)
            y2 = int((cy + h / 2 - pad_y) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self._confidence, 0.45)
        detections: List[PhoneDetection] = []
        for index in np.array(indices).flatten().tolist():
            x1, y1, x2, y2 = boxes[index]
            detections.append(PhoneDetection(x1, y1, x2, y2, scores[index]))
        return detections


class PhoneDetector:
    """Detects mobile phones in a frame."""

    def __init__(
        self,
        confidence: float = 0.5,
        backend: str = "auto",
        weights: Optional[Path] = None,
        imgsz: int = 640,
    ) -> None:
        self.confidence = float(confidence)
        self.backend = backend
        self.weights = Path(weights) if weights else YOLO_NANO_WEIGHTS
        self.imgsz = int(imgsz)

        self._backend = None
        self._lock = threading.Lock()
        self._error = ""

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def error(self) -> str:
        return self._error

    def set_confidence(self, confidence: float) -> None:
        with self._lock:
            self.confidence = float(confidence)
            if isinstance(self._backend, _UltralyticsBackend):
                self._backend._confidence = self.confidence
            elif isinstance(self._backend, _OnnxBackend):
                self._backend._confidence = self.confidence

    def set_imgsz(self, imgsz: int) -> None:
        """Change the inference resolution (smaller = faster)."""
        with self._lock:
            self.imgsz = int(imgsz)
            if isinstance(self._backend, _UltralyticsBackend):
                self._backend._imgsz = self.imgsz
            elif isinstance(self._backend, _OnnxBackend):
                self._backend._imgsz = self.imgsz

    def _ensure_backend(self) -> bool:
        if self._backend is not None:
            return True
        with self._lock:
            if self._backend is not None:
                return True
            if self.backend in ("auto", "ultralytics"):
                try:
                    self._backend = _UltralyticsBackend(self.weights, self.confidence, self.imgsz)
                    self._backend.warmup()
                    self._error = ""
                    LOGGER.info("Phone detector ready (Ultralytics YOLO, %s)", self.weights.name)
                    return True
                except Exception as exc:
                    LOGGER.warning("Ultralytics backend unavailable: %s", exc)
                    if self.backend == "ultralytics":
                        self._error = f"YOLO backend failed: {exc}"
                        return False
            if self.backend in ("auto", "onnx") and ONNX_WEIGHTS.exists():
                try:
                    self._backend = _OnnxBackend(ONNX_WEIGHTS, self.confidence, self.imgsz)
                    self._backend.warmup()
                    self._error = ""
                    LOGGER.info("Phone detector ready (OpenCV DNN, %s)", ONNX_WEIGHTS.name)
                    return True
                except Exception as exc:
                    LOGGER.warning("ONNX backend unavailable: %s", exc)
            self._error = (
                "No phone detection model available. Download yolo11n.pt into the "
                "models folder or export yolo11n.onnx."
            )
            return False

    def detect(self, frame: np.ndarray) -> List[PhoneDetection]:
        """Return all phones detected in the given BGR frame."""
        if frame is None or frame.size == 0:
            return []
        if not self._ensure_backend():
            return []
        with self._lock:
            try:
                return self._backend.detect(frame)
            except Exception as exc:
                LOGGER.warning("Phone detection failed: %s", exc)
                return []

    def close(self) -> None:
        with self._lock:
            self._backend = None
