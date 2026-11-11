"""Drowsiness / sleep detection based on MediaPipe Face Landmarker + EAR.

The Eye Aspect Ratio (EAR) is the vertical opening of an eye divided by its
horizontal width.  It stays around 0.25 - 0.35 for an open eye and drops
below ~0.20 when the eye is closed.  A short blink lasts only 100 - 200 ms,
while real drowsiness keeps the eyes closed for seconds - that duration is
what separates "blinking" from "sleeping".

The MediaPipe model file is downloaded automatically on first use and stored
in the ``models`` folder.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config.settings import MODELS_DIR

LOGGER = logging.getLogger(__name__)

FACE_LANDMARKS_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKS_PATH = MODELS_DIR / "face_landmarker.task"

# MediaPipe face mesh indices of the six points used for each eye.
# p1/p4 = outer/inner corner, p2/p6 and p3/p5 = upper/lower eyelid points.
LEFT_EYE_IDS: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDS: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# Landmark indices of the lips - used to confirm that a detected face is a
# real person and to measure the yawning mouth (not used for the alert yet).
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


@dataclass
class SleepResult:
    """Result of one sleep-detection pass."""

    face_found: bool = False
    left_ear: float = 0.0
    right_ear: float = 0.0
    ear: float = 0.0
    blink_left: float = 0.0
    blink_right: float = 0.0
    eyes_closed: bool = False
    sleeping: bool = False
    error: str = ""


@dataclass
class FaceObservation:
    """Landmarks and blendshape scores of one detected face."""

    landmarks: np.ndarray
    blink_left: float = 0.0
    blink_right: float = 0.0


def _distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def eye_aspect_ratio(eye: Sequence[Sequence[float]]) -> float:
    """Compute the Eye Aspect Ratio of six eye landmarks.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    The ratio is scale invariant, so the same thresholds work regardless of
    how close the student sits to the camera.
    """
    p1, p2, p3, p4, p5, p6 = eye
    vertical_1 = _distance(p2, p6)
    vertical_2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def download_model(url: str = FACE_LANDMARKS_URL, target: Path = FACE_LANDMARKS_PATH) -> bool:
    """Download the MediaPipe face landmarker model when it is missing."""
    target = Path(target)
    if target.exists() and target.stat().st_size > 1_000_000:
        return True
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading MediaPipe face landmarker model to %s", target)
        urllib.request.urlretrieve(url, target)
        LOGGER.info("Model download finished (%.1f MB)", target.stat().st_size / 1_048_576)
        return True
    except Exception as exc:  # network problems must not crash the app
        LOGGER.warning("Could not download the face landmarker model: %s", exc)
        return False


class _TasksBackend:
    """MediaPipe 1.x / Tasks API backend."""

    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._vision = vision
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        # MediaPipe 1.x exposes Image/ImageFormat on the top level package,
        # older releases keep them inside the vision module.
        self._image_factory = getattr(mp, "Image", None) or getattr(vision, "Image")
        image_format = getattr(mp, "ImageFormat", None) or getattr(vision, "ImageFormat")
        self._image_format = image_format.SRGB

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Optional[List[FaceObservation]]:
        image = self._image_factory(image_format=self._image_format, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None

        observations: List[FaceObservation] = []
        blendshapes = getattr(result, "face_blendshapes", None) or []
        for index, face in enumerate(result.face_landmarks):
            landmarks = np.array([(lm.x, lm.y) for lm in face], dtype=np.float32)
            blink_left = blink_right = 0.0
            if index < len(blendshapes):
                for shape in blendshapes[index]:
                    name = shape.category_name
                    if name == "eyeBlinkLeft":
                        blink_left = float(shape.score)
                    elif name == "eyeBlinkRight":
                        blink_right = float(shape.score)
            observations.append(FaceObservation(landmarks, blink_left, blink_right))
        return observations

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass


class _LegacyBackend:
    """Fallback for older MediaPipe releases that ship ``mp.solutions``."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> Optional[List[FaceObservation]]:
        result = self._face_mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None
        face = result.multi_face_landmarks[0]
        landmarks = np.array([(lm.x, lm.y) for lm in face.landmark], dtype=np.float32)
        return [FaceObservation(landmarks)]

    def close(self) -> None:
        try:
            self._face_mesh.close()
        except Exception:  # pragma: no cover
            pass


class SleepDetector:
    """Detects closed eyes and raises the sleep alert after a duration."""

    def __init__(
        self,
        closed_threshold: float = 0.21,
        open_threshold: float = 0.28,
        min_closure_duration: float = 2.5,
        require_both_eyes: bool = True,
        blink_threshold: float = 0.55,
        use_adaptive_threshold: bool = True,
        adaptive_ratio: float = 0.62,
        model_path: Optional[Path] = None,
    ) -> None:
        self.closed_threshold = float(closed_threshold)
        self.open_threshold = float(open_threshold)
        self.min_closure_duration = float(min_closure_duration)
        self.require_both_eyes = bool(require_both_eyes)
        # MediaPipe eyeBlink blendshape score above which an eye counts as closed.
        self.blink_threshold = float(blink_threshold)
        # Adaptive EAR threshold: closed when the EAR drops below
        # baseline * adaptive_ratio of the person's usual open-eye EAR.
        self.use_adaptive_threshold = bool(use_adaptive_threshold)
        self.adaptive_ratio = float(adaptive_ratio)
        self.model_path = Path(model_path) if model_path else FACE_LANDMARKS_PATH

        self._backend = None
        self._lock = threading.Lock()
        self._last_timestamp = 0
        self._closed = False
        self._closed_since: Optional[float] = None
        self._ear_baseline = 0.0
        self._error = ""

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def error(self) -> str:
        return self._error

    def set_thresholds(
        self,
        closed_threshold: float,
        open_threshold: float,
        min_closure_duration: float,
        require_both_eyes: Optional[bool] = None,
    ) -> None:
        with self._lock:
            self.closed_threshold = float(closed_threshold)
            self.open_threshold = float(open_threshold)
            self.min_closure_duration = float(min_closure_duration)
            if require_both_eyes is not None:
                self.require_both_eyes = bool(require_both_eyes)

    def _ensure_backend(self) -> bool:
        if self._backend is not None:
            return True
        with self._lock:
            if self._backend is not None:
                return True
            if not self.model_path.exists() and not download_model(target=self.model_path):
                self._error = "Face landmarker model could not be downloaded."
                return False
            try:
                self._backend = _TasksBackend(self.model_path)
                LOGGER.info("MediaPipe face landmarker ready (%s)", self.model_path)
            except Exception as exc:
                LOGGER.info("Tasks API unavailable (%s), trying legacy MediaPipe backend", exc)
                try:
                    self._backend = _LegacyBackend()
                except Exception as legacy_exc:
                    self._error = f"MediaPipe could not be initialised: {legacy_exc}"
                    LOGGER.error(self._error)
                    return False
            self._error = ""
            return True

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> SleepResult:
        """Run one detection pass on a BGR frame."""
        if frame is None or frame.size == 0:
            return SleepResult(error="Empty frame")
        if not self._ensure_backend():
            return SleepResult(error=self._error)

        with self._lock:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # The MediaPipe VIDEO mode requires strictly increasing timestamps.
            self._last_timestamp = max(self._last_timestamp + 1, int(time.monotonic() * 1000))
            timestamp = self._last_timestamp
            try:
                faces = self._backend.detect(rgb_frame, timestamp)
            except Exception as exc:
                LOGGER.warning("Face landmark detection failed: %s", exc)
                return SleepResult(error=str(exc))

            if not faces:
                self._reset_closed_state()
                return SleepResult(face_found=False)

            landmarks = faces[0].landmarks
            height, width = frame.shape[:2]
            pixels = [(float(lm[0]) * width, float(lm[1]) * height) for lm in landmarks]

            try:
                left_ear = eye_aspect_ratio([pixels[i] for i in LEFT_EYE_IDS])
                right_ear = eye_aspect_ratio([pixels[i] for i in RIGHT_EYE_IDS])
            except (IndexError, TypeError):
                self._reset_closed_state()
                return SleepResult(face_found=True, error="Incomplete face landmarks")

            blink_left = faces[0].blink_left
            blink_right = faces[0].blink_right

            # When both eyes must be closed the *higher* EAR of the two eyes is
            # used: both have to fall below the threshold before we react.
            measure = max(left_ear, right_ear) if self.require_both_eyes else (left_ear + right_ear) / 2.0
            now = time.monotonic()

            # MediaPipe's "eyeBlink" blendshape is a second, independent signal
            # for closed eyes.  It is well calibrated and survives difficult
            # lighting better than the geometric EAR alone.
            blink_measure = (
                min(blink_left, blink_right)
                if self.require_both_eyes
                else max(blink_left, blink_right)
            )
            blink_says_closed = blink_measure >= self.blink_threshold

            # Adaptive threshold: every face has a slightly different eye
            # opening.  The baseline of the open-eye EAR is tracked, so a
            # person with narrow eyes is not reported as sleeping all the time.
            # The adaptive value can only make the detector stricter, never
            # more sensitive than the value chosen by the user.
            effective_closed = self.closed_threshold
            effective_open = self.open_threshold
            if self.use_adaptive_threshold and self._ear_baseline > 0:
                adaptive = self._ear_baseline * self.adaptive_ratio
                if adaptive < effective_closed:
                    effective_closed = adaptive
                    effective_open = max(effective_closed * 1.25, effective_closed + 0.01)

            ear_says_closed = measure < effective_closed
            closed_now = ear_says_closed or blink_says_closed

            # Hysteresis avoids flickering between open and closed around the
            # threshold and therefore avoids false "sleeping" alerts.
            if not self._closed and closed_now:
                self._closed = True
                self._closed_since = now
            elif self._closed and measure > effective_open and not blink_says_closed:
                self._closed = False
                self._closed_since = None

            # Remember how open the eyes were - this is the baseline for the
            # adaptive threshold.  It decays slowly so it follows the student.
            if measure > self.closed_threshold:
                self._ear_baseline = max(self._ear_baseline * 0.995, measure)

            closed_for = 0.0 if self._closed_since is None else now - self._closed_since
            sleeping = self._closed and closed_for >= self.min_closure_duration

            return SleepResult(
                face_found=True,
                left_ear=left_ear,
                right_ear=right_ear,
                ear=measure,
                blink_left=blink_left,
                blink_right=blink_right,
                eyes_closed=self._closed,
                sleeping=sleeping,
            )

    def _reset_closed_state(self) -> None:
        self._closed = False
        self._closed_since = None

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
