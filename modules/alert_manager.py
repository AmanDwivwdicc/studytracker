"""Alert manager: plays the user's custom sounds without blocking the camera.

Design notes
------------
* Playback runs in its own worker thread, so the detection thread never waits
  for an audio file to finish.
* Every alert type has its own cooldown, so a phone that stays on the desk
  does not restart the sound every frame.
* ``pygame`` is used when available; on Windows the built-in ``winsound``
  module is used as a fallback (WAV files only).  If neither can play the
  file, the visual warning still works and the error is reported to the GUI.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

SLEEP_ALERT = "sleep"
PHONE_ALERT = "phone"

ALERT_LABELS = {SLEEP_ALERT: "Sleep", PHONE_ALERT: "Phone"}


@dataclass
class AlertEvent:
    """One entry of the alert history."""

    timestamp: str
    alert_type: str
    message: str
    camera_id: str = ""
    camera_name: str = ""
    played: bool = True
    detail: str = ""


class AudioPlayer:
    """Small wrapper around the available audio backends."""

    def __init__(self) -> None:
        self.backend = "none"
        self._pygame = None
        self._initialised = False
        self._lock = threading.Lock()

    def _initialise(self) -> bool:
        if self._initialised:
            return self.backend != "none"
        with self._lock:
            if self._initialised:
                return self.backend != "none"
            try:
                import pygame

                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                self._pygame = pygame
                self.backend = "pygame"
                self._initialised = True
                LOGGER.info("Audio backend: pygame")
                return True
            except Exception as exc:
                LOGGER.info("pygame unavailable (%s)", exc)
            if platform.system() == "Windows":
                try:
                    import winsound  # noqa: F401

                    self.backend = "winsound"
                    self._initialised = True
                    LOGGER.info("Audio backend: winsound (WAV only)")
                    return True
                except Exception as exc:
                    LOGGER.warning("winsound unavailable: %s", exc)
            self._initialised = True
            self.backend = "none"
            return False

    @staticmethod
    def _is_playable_with_winsound(path: Path) -> bool:
        return path.suffix.lower() == ".wav"

    def play(self, path: Path) -> str:
        """Play ``path``.  Returns an empty string on success or an error text."""
        path = Path(path)
        if not path.exists():
            return f"Audio file not found: {path}"
        if not self._initialise():
            return "No audio backend available (install pygame for MP3 support)."

        try:
            if self.backend == "pygame":
                self._pygame.mixer.music.load(str(path))
                self._pygame.mixer.music.play()
                return ""
            if self.backend == "winsound":
                if not self._is_playable_with_winsound(path):
                    return (
                        f"{path.name} cannot be played by the Windows fallback backend. "
                        "Install pygame (pip install pygame) to play MP3 files."
                    )
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                return ""
        except Exception as exc:
            return f"Could not play {path.name}: {exc}"
        return f"Unknown audio backend: {self.backend}"

    def stop(self) -> None:
        if self.backend == "pygame" and self._pygame is not None:
            try:
                self._pygame.mixer.music.stop()
                # Quitting the mixer releases the file handle of the loaded
                # sound, so the file can be deleted or replaced afterwards.
                self._pygame.mixer.quit()
                self._initialised = False
                self.backend = "none"
            except Exception:
                pass


class AlertManager:
    """Central place for audio alerts, cooldowns and the alert history."""

    def __init__(
        self,
        sleep_audio: Optional[Path] = None,
        phone_audio: Optional[Path] = None,
        sleep_cooldown: float = 30.0,
        phone_cooldown: float = 30.0,
        on_event: Optional[Callable[[AlertEvent], None]] = None,
    ) -> None:
        self.sleep_audio = Path(sleep_audio) if sleep_audio else None
        self.phone_audio = Path(phone_audio) if phone_audio else None
        self.sleep_cooldown = float(sleep_cooldown)
        self.phone_cooldown = float(phone_cooldown)
        self.on_event = on_event

        self._events: List[AlertEvent] = []
        self._lock = threading.Lock()
        self._last_played: Dict[str, float] = {}
        self._queue: List[tuple[str, Path]] = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._player = AudioPlayer()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error = ""

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="alert-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._player.stop()

    # ------------------------------------------------------------------
    @property
    def events(self) -> List[AlertEvent]:
        with self._lock:
            return list(self._events)

    @property
    def last_error(self) -> str:
        return self._last_error

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()

    def audio_path(self, alert_type: str) -> Optional[Path]:
        return self.sleep_audio if alert_type == SLEEP_ALERT else self.phone_audio

    def set_audio_path(self, alert_type: str, path: Optional[Path]) -> None:
        if alert_type == SLEEP_ALERT:
            self.sleep_audio = Path(path) if path else None
        else:
            self.phone_audio = Path(path) if path else None

    def set_cooldown(self, alert_type: str, seconds: float) -> None:
        if alert_type == SLEEP_ALERT:
            self.sleep_cooldown = float(seconds)
        else:
            self.phone_cooldown = float(seconds)

    def missing_audio_files(self) -> List[str]:
        """Names of the configured alert files that are not available."""
        missing = []
        for alert_type, path in ((SLEEP_ALERT, self.sleep_audio), (PHONE_ALERT, self.phone_audio)):
            if path is None or not Path(path).exists():
                missing.append(path.name if path else f"{alert_type}_alert.mp3")
        return missing

    # ------------------------------------------------------------------
    def trigger(
        self,
        alert_type: str,
        message: str = "",
        camera_id: str = "",
        camera_name: str = "",
        force: bool = False,
    ) -> bool:
        """Queue an alert.

        Returns True when the sound was actually played, False when the
        cooldown suppressed it or no audio file was configured.
        """
        cooldown = self.sleep_cooldown if alert_type == SLEEP_ALERT else self.phone_cooldown
        now = time.monotonic()
        with self._lock:
            last = self._last_played.get(alert_type, 0.0)
            if not force and (now - last) < cooldown:
                return False
            self._last_played[alert_type] = now

        path = self.audio_path(alert_type)
        played = False
        detail = ""
        if path is None:
            detail = f"No {alert_type} audio file configured."
        elif not Path(path).exists():
            detail = f"Audio file not found: {path}"
        else:
            with self._queue_lock:
                self._queue.append((alert_type, Path(path)))
                self._queue_event.set()
            played = True

        event = AlertEvent(
            timestamp=time.strftime("%H:%M:%S"),
            alert_type=alert_type,
            message=message or f"{ALERT_LABELS.get(alert_type, alert_type)} alert",
            camera_id=camera_id,
            camera_name=camera_name,
            played=played,
            detail=detail,
        )
        with self._lock:
            self._events.append(event)
            if len(self._events) > 500:
                del self._events[:-500]
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:  # a broken callback must not stop alerts
                LOGGER.warning("Alert callback failed: %s", exc)
        return played

    def _worker(self) -> None:
        """Audio worker thread - never blocks the detection threads."""
        while self._running:
            self._queue_event.wait(timeout=0.5)
            self._queue_event.clear()
            while True:
                with self._queue_lock:
                    if not self._queue:
                        break
                    _, path = self._queue.pop(0)
                error = self._player.play(path)
                if error:
                    self._last_error = error
                    LOGGER.warning(error)
                else:
                    self._last_error = ""
                time.sleep(0.05)  # tiny gap so two alerts do not overlap badly
