"""Central configuration for Study Tracker.

Every tunable value lives here so the dashboard, the detection modules and the
alert system all read from a single source of truth.  Settings can be stored
as JSON (``config/settings.json``) and are restored automatically on the next
start-up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

DEFAULT_SLEEP_AUDIO = ASSETS_DIR / "sleep_alert.mp3"
DEFAULT_PHONE_AUDIO = ASSETS_DIR / "phone_alert.mp3"


@dataclass
class Settings:
    """All user-tunable options of the application."""

    # --- Sleep / drowsiness detection -------------------------------------
    # Eye Aspect Ratio below which an eye is considered closed.
    ear_closed_threshold: float = 0.21
    # EAR above which a closed eye is considered open again (hysteresis).
    ear_open_threshold: float = 0.28
    # Seconds the eyes must stay closed before the sleep alert fires.
    min_eye_closure_duration: float = 2.5
    # Seconds before the very same alert type may play again.
    sleep_alert_cooldown: float = 30.0
    # Require *both* eyes to be closed (more robust, less false positives).
    require_both_eyes: bool = True
    # MediaPipe "eyeBlink" blendshape score above which an eye counts as closed.
    blink_threshold: float = 0.55
    # Adapt the EAR threshold to the personal open-eye baseline of the student.
    use_adaptive_threshold: bool = True
    # Fraction of the baseline EAR that still counts as "closed".
    adaptive_ratio: float = 0.62

    # --- Mobile phone detection -------------------------------------------
    # Minimum YOLO confidence for a phone detection.
    phone_confidence_threshold: float = 0.50
    # Seconds a phone must stay visible before the alert fires.
    min_phone_detection_duration: float = 1.5
    # Seconds before the phone alert may play again.
    phone_alert_cooldown: float = 30.0

    # --- Performance -------------------------------------------------------
    # Number of frames skipped between two detection runs (1 = every 2nd frame).
    frame_skip: int = 2
    # Width the frame is resized to before inference (smaller = faster).
    detection_width: int = 640
    # Seconds waited before a broken camera connection is retried.
    reconnect_delay: float = 3.0

    # --- Audio -------------------------------------------------------------
    sleep_audio_path: str = str(DEFAULT_SLEEP_AUDIO)
    phone_audio_path: str = str(DEFAULT_PHONE_AUDIO)

    # --- Camera ------------------------------------------------------------
    default_camera_index: int = 0

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Clamp obviously invalid values so the GUI never breaks the engine."""
        self.ear_closed_threshold = min(max(float(self.ear_closed_threshold), 0.05), 0.60)
        self.ear_open_threshold = min(max(float(self.ear_open_threshold), self.ear_closed_threshold + 0.01), 0.80)
        self.min_eye_closure_duration = max(float(self.min_eye_closure_duration), 0.1)
        self.sleep_alert_cooldown = max(float(self.sleep_alert_cooldown), 0.0)
        self.blink_threshold = min(max(float(self.blink_threshold), 0.1), 0.95)
        self.use_adaptive_threshold = bool(self.use_adaptive_threshold)
        self.adaptive_ratio = min(max(float(self.adaptive_ratio), 0.2), 0.95)
        self.phone_confidence_threshold = min(max(float(self.phone_confidence_threshold), 0.05), 0.99)
        self.min_phone_detection_duration = max(float(self.min_phone_detection_duration), 0.1)
        self.phone_alert_cooldown = max(float(self.phone_alert_cooldown), 0.0)
        self.frame_skip = int(min(max(int(self.frame_skip), 0), 20))
        self.detection_width = int(min(max(int(self.detection_width), 160), 1920))
        self.reconnect_delay = max(float(self.reconnect_delay), 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}
        settings = cls(**clean)
        settings.validate()
        return settings


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from disk, falling back to defaults when unavailable."""
    target = Path(path) if path else SETTINGS_FILE
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as handle:
                return Settings.from_dict(json.load(handle))
        except (OSError, ValueError) as exc:  # broken file should never stop the app
            LOGGER.warning("Could not read %s (%s) - using defaults.", target, exc)
    return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> bool:
    """Persist the settings as JSON.  Returns True on success."""
    target = Path(path) if path else SETTINGS_FILE
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(settings.to_dict(), handle, indent=2)
        return True
    except OSError as exc:
        LOGGER.warning("Could not save settings to %s: %s", target, exc)
        return False


def ensure_directories() -> None:
    """Create the folders that ship with the project (assets/models)."""
    for folder in (ASSETS_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
