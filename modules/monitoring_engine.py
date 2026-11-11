"""Monitoring engine: one processing thread per camera.

Each camera gets its own worker thread that

1. grabs the latest frame from its :class:`~modules.camera_manager.CameraStream`,
2. runs the sleep detector and the phone detector on it,
3. keeps track of how long a distraction has been present,
4. triggers audio/visual alerts through the :class:`~modules.alert_manager.AlertManager`,
5. stores an annotated frame and a status snapshot for the dashboard.

A dedicated detector instance is created per camera because the MediaPipe and
YOLO models are not safe to call from several threads at the same time.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from modules.alert_manager import PHONE_ALERT, SLEEP_ALERT, AlertManager
from modules.camera_manager import CameraStream, ConnectionState
from modules.phone_detector import PhoneDetection, PhoneDetector
from modules.sleep_detector import SleepDetector

LOGGER = logging.getLogger(__name__)

GREEN = (60, 200, 90)
YELLOW = (40, 210, 240)
RED = (60, 60, 235)

STATUS_GREEN = "green"
STATUS_YELLOW = "yellow"
STATUS_RED = "red"


@dataclass
class CameraStatus:
    """Snapshot of one camera, read by the dashboard."""

    source_id: str
    name: str
    connected: bool = False
    state: str = "idle"
    error: str = ""
    sleeping: bool = False
    phone_detected: bool = False
    phone_count: int = 0
    face_found: bool = False
    eye_aspect_ratio: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    blink_score: float = 0.0
    eyes_closed: bool = False
    fps: float = 0.0
    detect_fps: float = 0.0
    level: str = STATUS_YELLOW
    message: str = "Waiting for camera..."
    boxes: List[Tuple[int, int, int, int, float]] = field(default_factory=list)
    frame_time: float = 0.0


class CameraWorker:
    """Processes the frames of one camera in its own thread."""

    def __init__(
        self,
        stream: CameraStream,
        settings: Settings,
        alert_manager: AlertManager,
    ) -> None:
        self.stream = stream
        self.settings = settings
        self.alert_manager = alert_manager

        self._sleep_detector = SleepDetector(
            closed_threshold=settings.ear_closed_threshold,
            open_threshold=settings.ear_open_threshold,
            min_closure_duration=settings.min_eye_closure_duration,
            require_both_eyes=settings.require_both_eyes,
            blink_threshold=settings.blink_threshold,
            use_adaptive_threshold=settings.use_adaptive_threshold,
            adaptive_ratio=settings.adaptive_ratio,
        )
        self._phone_detector = PhoneDetector(
            confidence=settings.phone_confidence_threshold,
            imgsz=settings.detection_width,
        )

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._status = CameraStatus(
            source_id=stream.source.source_id,
            name=stream.source.name,
        )
        self._annotated_frame: Optional[np.ndarray] = None

        # Distraction timers.
        self._phone_since: Optional[float] = None
        self._phone_alert_active = False
        self._sleep_alert_active = False

        # Detection speed measurement (shown in the dashboard).
        self._detect_fps = 0.0
        self._detect_frames = 0
        self._detect_window_start = time.monotonic()

    # ------------------------------------------------------------------
    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    @property
    def annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._annotated_frame

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"worker-{self.stream.source.source_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._running = False
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._sleep_detector.close()
        self._phone_detector.close()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        frame_counter = 0
        last_sequence = -1
        while self._running:
            frame = self.stream.latest_frame
            sequence = self.stream.frame_sequence
            if frame is None or sequence == last_sequence:
                self._update_connection_status()
                time.sleep(0.03)
                continue

            last_sequence = sequence
            frame_counter += 1
            skip = max(self.settings.frame_skip, 0)
            if skip and frame_counter % (skip + 1) != 0:
                continue

            try:
                self._process(frame)
            except Exception as exc:  # never let one camera kill the thread
                LOGGER.exception("Processing error on %s: %s", self.stream.source.name, exc)
                self._set_error(f"Processing error: {exc}")
            time.sleep(0.005)

    # ------------------------------------------------------------------
    def _process(self, frame: np.ndarray) -> None:
        sleep_result = self._sleep_detector.detect(frame)
        detections = self._phone_detector.detect(frame)

        self._detect_frames += 1
        elapsed = time.monotonic() - self._detect_window_start
        if elapsed >= 1.0:
            self._detect_fps = self._detect_frames / elapsed
            self._detect_frames = 0
            self._detect_window_start = time.monotonic()

        now = time.monotonic()
        sleeping = sleep_result.sleeping
        phone_present = len(detections) > 0

        # The phone has to stay visible for the configured duration before the
        # alert is triggered - this removes one-frame false positives.
        if phone_present:
            if self._phone_since is None:
                self._phone_since = now
            phone_duration = now - self._phone_since
        else:
            self._phone_since = None
            phone_duration = 0.0
            self._phone_alert_active = False

        phone_alert = (
            phone_present
            and phone_duration >= self.settings.min_phone_detection_duration
            and not self._phone_alert_active
        )

        if sleeping and not self._sleep_alert_active:
            self.alert_manager.trigger(
                SLEEP_ALERT,
                message="WAKE UP! YOU ARE SLEEPING.",
                camera_id=self.stream.source.source_id,
                camera_name=self.stream.source.name,
            )
            self._sleep_alert_active = True
        elif not sleeping:
            self._sleep_alert_active = False

        if phone_alert:
            self.alert_manager.trigger(
                PHONE_ALERT,
                message="WARNING! MOBILE PHONE DETECTED.",
                camera_id=self.stream.source.source_id,
                camera_name=self.stream.source.name,
            )
            self._phone_alert_active = True

        annotated = self._annotate(frame, sleep_result, detections)
        self._publish(annotated, sleep_result, detections)

    # ------------------------------------------------------------------
    def _annotate(
        self,
        frame: np.ndarray,
        sleep_result,
        detections: List[PhoneDetection],
    ) -> np.ndarray:
        """Draw the bounding boxes and the warning banners onto the frame."""
        canvas = frame.copy()

        for detection in detections:
            cv2.rectangle(
                canvas,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (40, 60, 240),
                2,
            )
            label = f"Mobile phone {detection.confidence * 100:.0f}%"
            cv2.putText(
                canvas,
                label,
                (detection.x1, max(18, detection.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 60, 240),
                2,
                cv2.LINE_AA,
            )

        if sleep_result.sleeping:
            self._draw_banner(canvas, "WAKE UP! YOU ARE SLEEPING.", RED)
        elif detections:
            self._draw_banner(canvas, "WARNING! MOBILE PHONE DETECTED.", RED)
        else:
            self._draw_banner(canvas, "No distraction detected", GREEN)

        # Live measurement readout so the user can see what the app measures.
        self._draw_readout(canvas, sleep_result, detections)
        return canvas

    @staticmethod
    def _draw_readout(canvas: np.ndarray, sleep_result, detections: List[PhoneDetection]) -> None:
        height, width = canvas.shape[:2]
        lines = []
        if sleep_result.face_found:
            lines.append(
                f"EAR L {sleep_result.left_ear:.3f} | R {sleep_result.right_ear:.3f}"
                f" | eyes {'CLOSED' if sleep_result.eyes_closed else 'open'}"
            )
            lines.append(
                f"Blink score {max(sleep_result.blink_left, sleep_result.blink_right):.2f}"
            )
        else:
            lines.append("No face detected")
        lines.append(f"Phones: {len(detections)}")

        y = height - 12 - 22 * len(lines)
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 22

    @staticmethod
    def _draw_banner(canvas: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        height, width = canvas.shape[:2]
        scale = max(0.5, min(1.0, width / 900.0))
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2
        )
        y = text_h + baseline + 24
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        cv2.putText(
            canvas,
            text,
            (max(8, (width - text_w) // 2), text_h + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    def _publish(self, annotated, sleep_result, detections: List[PhoneDetection]) -> None:
        state = self.stream.state
        connected = state is ConnectionState.CONNECTED
        level = STATUS_GREEN
        message = "Monitoring active - no distraction"

        if not connected:
            level = STATUS_YELLOW
            message = self.stream.error or "Camera disconnected"
        elif sleep_result.sleeping:
            level = STATUS_RED
            message = "WAKE UP! YOU ARE SLEEPING."
        elif detections:
            level = STATUS_RED
            message = "WARNING! MOBILE PHONE DETECTED."
        elif sleep_result.error:
            level = STATUS_YELLOW
            message = sleep_result.error

        status = CameraStatus(
            source_id=self.stream.source.source_id,
            name=self.stream.source.name,
            connected=connected,
            state=state.value if hasattr(state, "value") else str(state),
            error=self.stream.error,
            sleeping=sleep_result.sleeping,
            phone_detected=bool(detections),
            phone_count=len(detections),
            face_found=sleep_result.face_found,
            eye_aspect_ratio=sleep_result.ear,
            left_ear=sleep_result.left_ear,
            right_ear=sleep_result.right_ear,
            blink_score=max(sleep_result.blink_left, sleep_result.blink_right),
            eyes_closed=sleep_result.eyes_closed,
            fps=self.stream.fps,
            detect_fps=self._detect_fps,
            level=level,
            message=message,
            boxes=[(d.x1, d.y1, d.x2, d.y2, d.confidence) for d in detections],
            frame_time=time.monotonic(),
        )
        with self._lock:
            self._status = status
            self._annotated_frame = annotated

    def _update_connection_status(self) -> None:
        state = self.stream.state
        connected = state is ConnectionState.CONNECTED
        status = self.status
        if status.connected == connected and status.state == state.value:
            return
        message = self.stream.error or ("Camera connected" if connected else "Camera disconnected")
        with self._lock:
            self._status = CameraStatus(
                source_id=status.source_id,
                name=status.name,
                connected=connected,
                state=state.value,
                error=self.stream.error,
                sleeping=False,
                phone_detected=False,
                phone_count=0,
                face_found=False,
                eye_aspect_ratio=0.0,
                left_ear=0.0,
                right_ear=0.0,
                blink_score=0.0,
                eyes_closed=False,
                fps=self.stream.fps,
                detect_fps=self._detect_fps,
                level=STATUS_GREEN if connected else STATUS_YELLOW,
                message=message,
                frame_time=time.monotonic(),
            )

    def _set_error(self, message: str) -> None:
        status = self.status
        with self._lock:
            self._status = CameraStatus(
                source_id=status.source_id,
                name=status.name,
                connected=status.connected,
                state=status.state,
                error=message,
                sleeping=False,
                phone_detected=False,
                phone_count=0,
                face_found=False,
                eye_aspect_ratio=0.0,
                left_ear=0.0,
                right_ear=0.0,
                blink_score=0.0,
                eyes_closed=False,
                fps=status.fps,
                detect_fps=self._detect_fps,
                level=STATUS_YELLOW,
                message=message,
                frame_time=time.monotonic(),
            )


class MonitoringEngine:
    """Owns one :class:`CameraWorker` per connected camera."""

    def __init__(self, settings: Settings, alert_manager: AlertManager) -> None:
        self.settings = settings
        self.alert_manager = alert_manager
        self._workers: Dict[str, CameraWorker] = {}
        self._lock = threading.Lock()
        self._monitoring = False

    @property
    def monitoring(self) -> bool:
        return self._monitoring

    def add_camera(self, stream: CameraStream) -> CameraWorker:
        """Register a camera and (re)start its worker when monitoring is active."""
        with self._lock:
            existing = self._workers.get(stream.source.source_id)
            if existing is not None:
                existing.stop()
            worker = CameraWorker(stream, self.settings, self.alert_manager)
            self._workers[stream.source.source_id] = worker
        if self._monitoring:
            worker.start()
        return worker

    def remove_camera(self, source_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(source_id, None)
        if worker is not None:
            worker.stop()

    def start(self) -> None:
        self._monitoring = True
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.start()

    def stop(self) -> None:
        self._monitoring = False
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.stop()

    def statuses(self) -> List[CameraStatus]:
        with self._lock:
            workers = list(self._workers.values())
        return [worker.status for worker in workers]

    def worker(self, source_id: str) -> Optional[CameraWorker]:
        with self._lock:
            return self._workers.get(source_id)

    def apply_settings(self, settings: Settings) -> None:
        """Push new settings into all running workers."""
        self.settings = settings
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.settings = settings
            worker._sleep_detector.set_thresholds(
                settings.ear_closed_threshold,
                settings.ear_open_threshold,
                settings.min_eye_closure_duration,
                settings.require_both_eyes,
            )
            worker._sleep_detector.blink_threshold = settings.blink_threshold
            worker._sleep_detector.use_adaptive_threshold = settings.use_adaptive_threshold
            worker._sleep_detector.adaptive_ratio = settings.adaptive_ratio
            worker._phone_detector.set_confidence(settings.phone_confidence_threshold)
        self.alert_manager.set_cooldown(SLEEP_ALERT, settings.sleep_alert_cooldown)
        self.alert_manager.set_cooldown(PHONE_ALERT, settings.phone_alert_cooldown)
