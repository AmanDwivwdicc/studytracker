"""Camera source manager.

Supports three kinds of sources:

* ``webcam``  - the built-in laptop camera or any USB camera (index 0, 1, 2 ...)
* ``rtsp``    - network cameras via ``rtsp://user:password@host:port/stream``
* ``file``    - local video files, handy for testing without a camera

Every source runs in its own capture thread.  The thread publishes the latest
decoded frame and automatically reconnects when a connection drops, so a
broken RTSP camera can never take down the rest of the application.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2

LOGGER = logging.getLogger(__name__)

# RTSP is far more reliable over TCP on home networks; the timeout prevents
# the capture thread from blocking forever when the camera is unplugged.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


class SourceType(str, Enum):
    """Kind of camera source."""

    WEBCAM = "webcam"
    RTSP = "rtsp"
    FILE = "file"


class ConnectionState(str, Enum):
    """Lifecycle of a camera connection."""

    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class CameraSource:
    """Description of one camera source (never contains hard-coded credentials)."""

    source_id: str
    name: str
    source_type: SourceType
    location: str
    username: str = ""
    password: str = ""

    def build_url(self) -> str:
        """Return the location OpenCV should open.

        For RTSP sources the user supplied credentials are injected into the
        URL when they are not already part of it.
        """
        if self.source_type is not SourceType.RTSP:
            return self.location

        parsed = urllib.parse.urlsplit(self.location)
        if not self.username or parsed.username:
            return self.location

        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        credentials = urllib.parse.quote(self.username, safe="")
        if self.password:
            credentials += ":" + urllib.parse.quote(self.password, safe="")
        return urllib.parse.urlunsplit(parsed._replace(netloc=f"{credentials}@{host}"))

    def masked_location(self) -> str:
        """Location without the password, safe to show in the GUI/logs."""
        if self.source_type is not SourceType.RTSP:
            return self.location
        parsed = urllib.parse.urlsplit(self.location)
        if not parsed.password:
            return self.location
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc += f":{parsed.port}"
        return urllib.parse.urlunsplit(parsed._replace(netloc=f"{parsed.username}:***@{netloc}"))

    def describe(self) -> str:
        if self.source_type is SourceType.WEBCAM:
            return f"Webcam #{self.location}"
        if self.source_type is SourceType.FILE:
            return f"File: {self.location}"
        return f"RTSP: {self.masked_location()}"


class CameraStream:
    """Background capture for a single source, publishing the latest frame."""

    def __init__(self, source: CameraSource, reconnect_delay: float = 3.0) -> None:
        self.source = source
        self.reconnect_delay = max(float(reconnect_delay), 0.5)

        self._lock = threading.Lock()
        self._frame = None
        self._frame_sequence = 0
        self._state = ConnectionState.IDLE
        self._error = ""
        self._fps = 0.0
        self._last_frame_time = 0.0

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def latest_frame(self):
        """The most recent decoded frame (or None)."""
        with self._lock:
            return self._frame

    @property
    def frame_sequence(self) -> int:
        with self._lock:
            return self._frame_sequence

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the capture thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self.source.source_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the capture thread and release the device."""
        self._running = False
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._release_capture()
        self._set_state(ConnectionState.DISCONNECTED)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set_state(self, state: ConnectionState, error: str = "") -> None:
        with self._lock:
            self._state = state
            self._error = error

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception as exc:  # pragma: no cover - driver specific
                LOGGER.debug("Releasing capture failed: %s", exc)

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open the configured source.  Returns None when it fails."""
        source = self.source
        if source.source_type is SourceType.FILE:
            path = Path(source.location)
            if not path.exists():
                self._set_state(ConnectionState.ERROR, f"Video file not found: {path}")
                return None
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                self._set_state(ConnectionState.ERROR, f"Cannot open video file: {path}")
                capture.release()
                return None
            return capture

        if source.source_type is SourceType.RTSP:
            url = source.build_url()
            LOGGER.info("Opening RTSP source %s", source.masked_location())
            # CAP_FFMPEG handles RTSP properly on Windows; DSHOW does not.
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(url)
            if not capture.isOpened():
                self._set_state(ConnectionState.ERROR, "Could not connect to the RTSP camera")
                capture.release()
                return None
            return capture

        # Local webcam / USB camera
        try:
            index = int(source.location)
        except (TypeError, ValueError):
            self._set_state(ConnectionState.ERROR, f"Invalid camera index: {source.location}")
            return None

        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            self._set_state(ConnectionState.ERROR, f"Cannot open camera index {index}")
            capture.release()
            return None
        return capture

    def _run(self) -> None:
        attempts = 0
        while self._running:
            attempts += 1
            self._set_state(
                ConnectionState.CONNECTING if attempts == 1 else ConnectionState.RECONNECTING
            )
            capture = self._open_capture()
            if capture is None:
                if not self._running:
                    break
                time.sleep(self.reconnect_delay)
                continue

            self._capture = capture
            self._set_state(ConnectionState.CONNECTED)
            attempts = 0
            LOGGER.info("Camera %s connected", self.source.describe())

            frame_counter = 0
            window_start = time.monotonic()
            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    if self.source.source_type is SourceType.FILE:
                        # Test videos are replayed in a loop instead of
                        # treating the end of the file as a lost connection.
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.01)
                        continue
                    LOGGER.warning("Camera %s lost the stream, reconnecting...", self.source.describe())
                    break
                frame_counter += 1
                with self._lock:
                    self._frame = frame
                    self._frame_sequence += 1
                elapsed = time.monotonic() - window_start
                if elapsed >= 1.0:
                    with self._lock:
                        self._fps = frame_counter / elapsed
                    frame_counter = 0
                    window_start = time.monotonic()

            self._release_capture()
            if self._running:
                self._set_state(ConnectionState.RECONNECTING, "Stream interrupted")
                time.sleep(self.reconnect_delay)

        self._set_state(ConnectionState.DISCONNECTED)


class CameraManager:
    """Creates and owns all configured camera streams."""

    def __init__(self, reconnect_delay: float = 3.0) -> None:
        self.reconnect_delay = reconnect_delay
        self._streams: Dict[str, CameraStream] = {}
        self._lock = threading.Lock()

    @staticmethod
    def create_source(
        source_type: SourceType,
        location: str,
        name: str = "",
        username: str = "",
        password: str = "",
    ) -> CameraSource:
        """Build a :class:`CameraSource` with a generated id and a friendly name."""
        location = str(location).strip()
        if source_type is SourceType.WEBCAM:
            default_name = f"Webcam {location}"
        elif source_type is SourceType.RTSP:
            default_name = "IP Camera"
        else:
            default_name = Path(location).name or "Video File"
        return CameraSource(
            source_id=uuid.uuid4().hex[:8],
            name=(name.strip() or default_name),
            source_type=source_type,
            location=location,
            username=username.strip(),
            password=password,
        )

    def add(self, source: CameraSource) -> CameraStream:
        stream = CameraStream(source, reconnect_delay=self.reconnect_delay)
        with self._lock:
            self._streams[source.source_id] = stream
        return stream

    def remove(self, source_id: str) -> None:
        with self._lock:
            stream = self._streams.pop(source_id, None)
        if stream is not None:
            stream.stop()

    def get(self, source_id: str) -> Optional[CameraStream]:
        with self._lock:
            return self._streams.get(source_id)

    @property
    def streams(self) -> List[CameraStream]:
        with self._lock:
            return list(self._streams.values())

    def start_all(self) -> None:
        for stream in self.streams:
            stream.start()

    def stop_all(self) -> None:
        for stream in self.streams:
            stream.stop()
