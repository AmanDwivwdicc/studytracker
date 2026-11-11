"""Offline tests for the individual Study Tracker modules.

Run with:

    python main.py --self-test

The tests do not need a camera, a network connection or the GUI.  Model
downloads are skipped automatically when there is no internet connection.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings, load_settings, save_settings
from modules.alert_manager import SLEEP_ALERT, AlertManager
from modules.camera_manager import CameraManager, SourceType
from modules.monitoring_engine import MonitoringEngine
from modules.phone_detector import PhoneDetector
from modules.sleep_detector import SleepDetector, eye_aspect_ratio


class TestResult:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, name: str) -> None:
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name: str, reason: str) -> None:
        self.failed += 1
        print(f"  [FAIL] {name}: {reason}")

    def skip(self, name: str, reason: str) -> None:
        self.skipped += 1
        print(f"  [SKIP] {name}: {reason}")


def test_settings(result: TestResult) -> None:
    """Settings must survive a save/load round trip."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "settings.json"
        settings = Settings(ear_closed_threshold=0.19, phone_confidence_threshold=0.62)
        if not save_settings(settings, path):
            result.fail("settings round trip", "save_settings returned False")
            return
        loaded = load_settings(path)
        if abs(loaded.ear_closed_threshold - 0.19) > 1e-6:
            result.fail("settings round trip", f"threshold was {loaded.ear_closed_threshold}")
            return
        if abs(loaded.phone_confidence_threshold - 0.62) > 1e-6:
            result.fail("settings round trip", "confidence was not restored")
            return
        result.ok("settings round trip")


def test_ear_math(result: TestResult) -> None:
    """A wide open eye must have a higher EAR than a closed eye."""
    open_eye = [
        (0, 0), (0, 0), (0, 1), (1, 0), (0, 1), (0, 1),
    ]
    # p1=(0,0) p2=(0,0) p3=(0,1) p4=(1,0) p5=(0,1) p6=(0,1)
    ear_open = eye_aspect_ratio(open_eye)
    closed_eye = [
        (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0),
    ]
    ear_closed = eye_aspect_ratio(closed_eye)
    if not (ear_open > ear_closed):
        result.fail("EAR calculation", f"open={ear_open:.3f} closed={ear_closed:.3f}")
        return
    if ear_closed != 0.0:
        result.fail("EAR calculation", "a fully closed eye must return 0")
        return
    result.ok("EAR calculation")


def test_camera_manager(result: TestResult) -> None:
    """RTSP URLs must receive the credentials and the mask the password."""
    manager = CameraManager()
    source = manager.create_source(
        SourceType.RTSP,
        "rtsp://192.168.1.100:554/stream1",
        name="Desk camera",
        username="student",
        password="secret",
    )
    url = source.build_url()
    if "student:secret@192.168.1.100:554" not in url:
        result.fail("RTSP credential injection", f"unexpected url: {url}")
        return
    if "secret" in source.masked_location():
        result.fail("RTSP credential masking", "password is visible in the masked location")
        return
    result.ok("RTSP credential injection")


def test_video_file_stream(result: TestResult) -> None:
    """A generated video file must be opened and publish frames."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "sample.avi"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)
        )
        if not writer.isOpened():
            result.skip("video file stream", "OpenCV could not create a test video")
            return
        for _ in range(10):
            frame = np.full((240, 320, 3), 30, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        manager = CameraManager(reconnect_delay=0.5)
        source = manager.create_source(SourceType.FILE, str(video_path), name="Test file")
        stream = manager.add(source)
        stream.start()
        deadline = time.time() + 8.0
        got_frame = False
        while time.time() < deadline:
            if stream.latest_frame is not None:
                got_frame = True
                break
            time.sleep(0.1)
        stream.stop()
        manager.stop_all()
        if not got_frame:
            result.fail("video file stream", "no frame was published")
            return
        result.ok("video file stream")


def test_sleep_detector(result: TestResult) -> None:
    """The detector must not crash on an empty frame and must report errors."""
    detector = SleepDetector(min_closure_duration=0.1)
    empty = np.zeros((240, 320, 3), dtype=np.uint8)
    detection = detector.detect(empty)
    if detection.face_found:
        result.fail("sleep detector on blank frame", "a face was reported on a blank frame")
        return
    if detection.error:
        result.skip("sleep detector on blank frame", detection.error)
        return
    result.ok("sleep detector on blank frame")


def test_phone_detector(result: TestResult) -> None:
    """The phone detector must return an empty list for an empty frame."""
    detector = PhoneDetector(confidence=0.5)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    if detections:
        result.fail("phone detector on blank frame", f"{len(detections)} false detections")
        return
    if not detector.available:
        result.skip("phone detector on blank frame", detector.error)
        return
    result.ok("phone detector on blank frame")


def test_alert_manager(result: TestResult) -> None:
    """Cooldowns must suppress repeated alerts, missing files must not crash."""
    manager = AlertManager(
        sleep_audio=Path("does_not_exist_sleep.mp3"),
        phone_audio=Path("does_not_exist_phone.mp3"),
        sleep_cooldown=30.0,
        phone_cooldown=30.0,
    )
    manager.start()
    try:
        missing = manager.missing_audio_files()
        if len(missing) != 2:
            result.fail("alert manager missing files", f"expected 2 missing files, got {missing}")
            return
        first = manager.trigger(SLEEP_ALERT, "WAKE UP! YOU ARE SLEEPING.")
        second = manager.trigger(SLEEP_ALERT, "WAKE UP! YOU ARE SLEEPING.")
        if first:
            result.fail("alert manager cooldown", "an alert with a missing file was reported as played")
            return
        if second:
            result.fail("alert manager cooldown", "the cooldown did not suppress the second alert")
            return
        if not manager.events:
            result.fail("alert manager cooldown", "no alert event was recorded")
            return
        result.ok("alert manager cooldown")
    finally:
        manager.stop()


def test_alert_with_real_audio(result: TestResult) -> None:
    """A generated WAV file must be playable and recorded in the history."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        wav_path = Path(tmp) / "test_alert.wav"
        sample_rate = 22050
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave_data = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
        import wave

        with wave.open(str(wav_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(wave_data.tobytes())

        manager = AlertManager(sleep_audio=wav_path, phone_audio=wav_path, sleep_cooldown=0.0)
        manager.start()
        try:
            played = manager.trigger(SLEEP_ALERT, "test", force=True)
            time.sleep(0.5)
            if not played:
                result.fail("alert playback", "the alert was not queued for playback")
                return
            if manager.last_error:
                result.fail("alert playback", manager.last_error)
                return
            result.ok("alert playback")
        finally:
            manager.stop()


def test_monitoring_engine(result: TestResult) -> None:
    """The engine must publish a status for a connected camera."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "engine.avi"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240)
        )
        if not writer.isOpened():
            result.skip("monitoring engine", "OpenCV could not create a test video")
            return
        for _ in range(15):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        settings = Settings()
        alert_manager = AlertManager(sleep_cooldown=1.0, phone_cooldown=1.0)
        alert_manager.start()
        manager = CameraManager(reconnect_delay=0.5)
        source = manager.create_source(SourceType.FILE, str(video_path), name="Engine test")
        stream = manager.add(source)
        stream.start()
        engine = MonitoringEngine(settings, alert_manager)
        worker = engine.add_camera(stream)
        engine.start()

        deadline = time.time() + 10.0
        status = None
        while time.time() < deadline:
            status = worker.status
            if status.connected:
                break
            time.sleep(0.2)

        engine.stop()
        manager.stop_all()
        alert_manager.stop()
        if status is None or not status.connected:
            result.fail("monitoring engine", "the camera never reached the connected state")
            return
        result.ok("monitoring engine")


def run_tests() -> int:
    """Run every test and return the process exit code."""
    print("Study Tracker - module self test")
    print("-" * 60)
    result = TestResult()
    tests = (
        ("settings", test_settings),
        ("EAR calculation", test_ear_math),
        ("camera manager", test_camera_manager),
        ("video file stream", test_video_file_stream),
        ("sleep detector", test_sleep_detector),
        ("phone detector", test_phone_detector),
        ("alert manager", test_alert_manager),
        ("alert playback", test_alert_with_real_audio),
        ("monitoring engine", test_monitoring_engine),
    )
    for name, test in tests:
        print(f"[{name}]")
        try:
            test(result)
        except Exception as exc:
            result.fail(name, f"unexpected exception: {exc}")
    print("-" * 60)
    print(f"passed: {result.passed} | failed: {result.failed} | skipped: {result.skipped}")
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_tests())
