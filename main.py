"""Study Tracker - entry point.

Usage:
    python main.py              # start the dashboard
    python main.py --self-test  # run the offline module tests
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure the project root is importable when started from another folder.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    ASSETS_DIR,
    MODELS_DIR,
    Settings,
    ensure_directories,
    load_settings,
    save_settings,
)
from modules.alert_manager import AlertManager  # noqa: E402
from modules.camera_manager import CameraManager  # noqa: E402
from modules.monitoring_engine import MonitoringEngine  # noqa: E402
from ui.dashboard import Dashboard  # noqa: E402

LOGGER = logging.getLogger("study_tracker")


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_application(settings: Settings):
    """Create every component and wire them together."""
    camera_manager = CameraManager(reconnect_delay=settings.reconnect_delay)
    alert_manager = AlertManager(
        sleep_audio=Path(settings.sleep_audio_path),
        phone_audio=Path(settings.phone_audio_path),
        sleep_cooldown=settings.sleep_alert_cooldown,
        phone_cooldown=settings.phone_alert_cooldown,
    )
    alert_manager.start()
    engine = MonitoringEngine(settings, alert_manager)
    return camera_manager, alert_manager, engine


def run_self_test() -> int:
    """Run the offline module tests (no camera and no GUI required)."""
    from tests.test_modules import run_tests

    return run_tests()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Study Tracker - AI study monitoring")
    parser.add_argument("--self-test", action="store_true",
                        help="run the offline module tests and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    if args.self_test:
        return run_self_test()

    ensure_directories()
    settings = load_settings()
    camera_manager, alert_manager, engine = build_application(settings)

    LOGGER.info("Study Tracker starting. Assets: %s | Models: %s", ASSETS_DIR, MODELS_DIR)
    missing = alert_manager.missing_audio_files()
    if missing:
        LOGGER.warning("Missing alert file(s): %s - add them to %s", ", ".join(missing), ASSETS_DIR)

    dashboard = Dashboard(settings, camera_manager, engine, alert_manager)
    try:
        dashboard.mainloop()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        engine.stop()
        camera_manager.stop_all()
        alert_manager.stop()
        save_settings(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
