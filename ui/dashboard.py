"""Tkinter dashboard for Study Tracker.

The GUI only reads data produced by the worker threads - it never performs
detection itself, so the interface always stays responsive.  Frames are
refreshed with ``after()`` callbacks on the Tk main thread.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from config.settings import ASSETS_DIR, Settings, save_settings
from modules.alert_manager import PHONE_ALERT, SLEEP_ALERT, AlertManager
from modules.camera_manager import CameraManager, SourceType
from modules.monitoring_engine import (
    STATUS_GREEN,
    STATUS_RED,
    STATUS_YELLOW,
    CameraStatus,
    MonitoringEngine,
)

LOGGER = logging.getLogger(__name__)

BG = "#101418"
PANEL = "#1a2027"
FG = "#e8eef5"
ACCENT = "#2f81f7"
GREEN = "#3cc85a"
YELLOW = "#f2c744"
RED = "#f23c3c"

LEVEL_COLORS = {STATUS_GREEN: GREEN, STATUS_YELLOW: YELLOW, STATUS_RED: RED}


class CameraPanel(ttk.Frame):
    """Shows the live feed and the status of a single camera."""

    def __init__(self, master, source_id: str, name: str) -> None:
        super().__init__(master, style="Camera.TFrame")
        self.source_id = source_id
        self.name = name
        self._photo: Optional[ImageTk.PhotoImage] = None

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x")
        self.title_label = tk.Label(
            header, text=name, bg=PANEL, fg=FG, font=("Segoe UI", 10, "bold"), anchor="w"
        )
        self.title_label.pack(side="left", padx=6, pady=3)
        self.badge = tk.Label(header, text="OFFLINE", bg=YELLOW, fg="#000000",
                              font=("Segoe UI", 8, "bold"), padx=6, pady=1)
        self.badge.pack(side="right", padx=6, pady=3)

        self.canvas = tk.Canvas(self, bg="#05070a", highlightthickness=1, height=240)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.canvas.create_text(
            160, 120, text="No signal", fill="#7c8794", font=("Segoe UI", 10)
        )

        self.status_var = tk.StringVar(value="Waiting...")
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel").pack(
            fill="x", padx=6, pady=(0, 4)
        )

    def set_title(self, name: str) -> None:
        self.name = name
        self.title_label.config(text=name)

    def update_frame(self, frame: Optional[np.ndarray], status: Optional[CameraStatus]) -> None:
        if status is not None:
            self.badge.config(text=status.level.upper(), bg=LEVEL_COLORS.get(status.level, YELLOW))
            self.status_var.set(self._status_text(status))
        if frame is None:
            return
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            canvas_width = max(self.canvas.winfo_width(), 320)
            canvas_height = max(self.canvas.winfo_height(), 200)
            pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.BILINEAR)
            self._photo = ImageTk.PhotoImage(pil_image)
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                image=self._photo,
                anchor="center",
            )
        except Exception as exc:  # pragma: no cover - GUI safety net
            LOGGER.debug("Could not render frame: %s", exc)

    @staticmethod
    def _status_text(status: CameraStatus) -> str:
        """Status line with the live measurements of this camera."""
        parts = [status.message]
        if status.face_found:
            parts.append(
                f"EAR {status.left_ear:.3f}/{status.right_ear:.3f}"
                f"  blink {status.blink_score:.2f}"
                f"  eyes {'CLOSED' if status.eyes_closed else 'open'}"
            )
        else:
            parts.append("no face detected")
        parts.append(f"cam {status.fps:.0f} fps / detect {status.detect_fps:.0f} fps")
        return "   |   ".join(parts)

    def clear(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="No signal",
            fill="#7c8794",
            font=("Segoe UI", 10),
        )


class Dashboard(tk.Tk):
    """The main application window."""

    def __init__(
        self,
        settings: Settings,
        camera_manager: CameraManager,
        engine: MonitoringEngine,
        alert_manager: AlertManager,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.camera_manager = camera_manager
        self.engine = engine
        self.alert_manager = alert_manager

        self.panels: Dict[str, CameraPanel] = {}
        self._shown_alerts = 0
        self._closing = False

        self.title("STUDY TRACKER")
        self.configure(bg=BG)
        self.geometry("1360x820")
        self.minsize(1080, 680)

        self._configure_styles()
        self._build_layout()
        self._apply_settings_to_widgets()
        self._refresh_audio_warning()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._refresh_loop)

    # ------------------------------------------------------------------
    # Styling / layout
    # ------------------------------------------------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Camera.TFrame", background=PANEL, relief="ridge", borderwidth=1)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 8))
        style.configure("Head.TLabel", background=BG, foreground=FG, font=("Segoe UI", 15, "bold"))
        style.configure("Title.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI", 20, "bold"))
        style.configure("Section.TLabel", background=PANEL, foreground=ACCENT,
                        font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background="#27313d", foreground=FG)
        style.map("TButton", background=[("active", "#34404f")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#1f6ae0")])
        style.configure("TEntry", fieldbackground="#0d1117", foreground=FG)
        style.configure("TCombobox", fieldbackground="#0d1117", foreground=FG)
        style.configure("Treeview", background="#0d1117", fieldbackground="#0d1117", foreground=FG)
        style.configure("Treeview.Heading", background="#27313d", foreground=FG)

    def _build_layout(self) -> None:
        # --- Top bar ---------------------------------------------------
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(top, text="STUDY TRACKER", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="AI powered study monitoring - runs fully offline",
                  style="TLabel").pack(side="left", padx=14)

        self.monitor_button = ttk.Button(top, text="Start Monitoring", style="Accent.TButton",
                                         command=self._start_monitoring)
        self.monitor_button.pack(side="right", padx=4)
        self.stop_button = ttk.Button(top, text="Stop Monitoring", command=self._stop_monitoring)
        self.stop_button.pack(side="right", padx=4)

        # --- Main body -------------------------------------------------
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=12, pady=6)
        body.columnconfigure(0, weight=0, minsize=330)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_control_panel(body)
        self._build_video_panel(body)

    def _build_control_panel(self, parent) -> None:
        controls = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        canvas = tk.Canvas(controls, bg=PANEL, highlightthickness=0, width=310)
        scrollbar = ttk.Scrollbar(controls, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Camera source --------------------------------------------
        ttk.Label(inner, text="CAMERA SOURCE", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.source_type = ttk.Combobox(
            inner,
            values=["Laptop / USB webcam", "IP camera (RTSP)", "Video file (test)"],
            state="readonly",
        )
        self.source_type.current(0)
        self.source_type.pack(fill="x", pady=(0, 6))
        self.source_type.bind("<<ComboboxSelected>>", lambda _e: self._source_type_changed())

        ttk.Label(inner, text="Camera index / RTSP URL / file path").pack(anchor="w")
        self.source_entry = ttk.Entry(inner)
        self.source_entry.insert(0, str(self.settings.default_camera_index))
        self.source_entry.pack(fill="x", pady=(0, 6))

        ttk.Label(inner, text="Camera name (optional)").pack(anchor="w")
        self.name_entry = ttk.Entry(inner)
        self.name_entry.pack(fill="x", pady=(0, 6))

        ttk.Label(inner, text="RTSP user name").pack(anchor="w")
        self.user_entry = ttk.Entry(inner)
        self.user_entry.pack(fill="x", pady=(0, 6))

        ttk.Label(inner, text="RTSP password").pack(anchor="w")
        self.password_entry = ttk.Entry(inner, show="*")
        self.password_entry.pack(fill="x", pady=(0, 8))

        ttk.Button(inner, text="Connect Camera", style="Accent.TButton",
                   command=self._connect_camera).pack(fill="x", pady=(0, 4))
        ttk.Button(inner, text="Disconnect Camera", command=self._disconnect_camera).pack(
            fill="x", pady=(0, 10)
        )

        ttk.Label(inner, text="CONNECTED CAMERAS", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.camera_list = tk.Listbox(inner, height=4, bg="#0d1117", fg=FG,
                                      selectbackground=ACCENT, highlightthickness=0)
        self.camera_list.pack(fill="x", pady=(0, 12))

        # --- Audio ------------------------------------------------------
        ttk.Label(inner, text="AUDIO ALERTS", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.audio_warning = ttk.Label(inner, text="", style="Status.TLabel", foreground=YELLOW,
                                       wraplength=300, justify="left")
        self.audio_warning.pack(fill="x", pady=(0, 6))

        ttk.Label(inner, text="Sleep alert file").pack(anchor="w")
        self.sleep_audio_entry = ttk.Entry(inner)
        self.sleep_audio_entry.pack(fill="x", pady=(2, 2))
        row = ttk.Frame(inner, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 8))
        ttk.Button(row, text="Browse", command=lambda: self._browse_audio(SLEEP_ALERT)).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(row, text="Test", command=lambda: self._test_audio(SLEEP_ALERT)).pack(
            side="left", fill="x", expand=True
        )

        ttk.Label(inner, text="Phone alert file").pack(anchor="w")
        self.phone_audio_entry = ttk.Entry(inner)
        self.phone_audio_entry.pack(fill="x", pady=(2, 2))
        row = ttk.Frame(inner, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 12))
        ttk.Button(row, text="Browse", command=lambda: self._browse_audio(PHONE_ALERT)).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(row, text="Test", command=lambda: self._test_audio(PHONE_ALERT)).pack(
            side="left", fill="x", expand=True
        )

        # --- Settings ---------------------------------------------------
        ttk.Label(inner, text="DETECTION SETTINGS", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.setting_vars: Dict[str, tk.StringVar] = {}
        self._add_setting(inner, "ear_closed_threshold", "Eye closed threshold (EAR)")
        self._add_setting(inner, "ear_open_threshold", "Eye open threshold (EAR)")
        self._add_setting(inner, "blink_threshold", "Blink score threshold (0-1)")
        self._add_setting(inner, "adaptive_ratio", "Adaptive EAR ratio (0-1)")
        self._add_setting(inner, "min_eye_closure_duration", "Min. eye closure (s)")
        self._add_setting(inner, "sleep_alert_cooldown", "Sleep alert cooldown (s)")
        self._add_setting(inner, "phone_confidence_threshold", "Phone confidence")
        self._add_setting(inner, "min_phone_detection_duration", "Min. phone duration (s)")
        self._add_setting(inner, "phone_alert_cooldown", "Phone alert cooldown (s)")
        self._add_setting(inner, "frame_skip", "Frame skip (0 = every frame)")
        ttk.Button(inner, text="Apply Settings", command=self._apply_settings).pack(fill="x", pady=(4, 12))

    def _add_setting(self, parent, key: str, label: str) -> None:
        ttk.Label(parent, text=label).pack(anchor="w")
        var = tk.StringVar()
        self.setting_vars[key] = var
        ttk.Entry(parent, textvariable=var).pack(fill="x", pady=(1, 5))

    def _build_video_panel(self, parent) -> None:
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Status strip
        status_frame = ttk.Frame(right)
        status_frame.grid(row=0, column=0, sticky="nsew")
        status_frame.rowconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)

        self.status_labels: Dict[str, tk.Label] = {}
        status_box = tk.Frame(status_frame, bg=PANEL)
        status_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for index, (key, caption) in enumerate(
            (("monitoring", "Monitoring"), ("sleep", "Sleep detection"), ("phone", "Phone detection"))
        ):
            label = tk.Label(status_box, text=f"{caption}: --", bg=PANEL, fg=YELLOW,
                             font=("Segoe UI", 10, "bold"), padx=14, pady=6)
            label.grid(row=0, column=index, sticky="ew")
            status_box.columnconfigure(index, weight=1)
            self.status_labels[key] = label

        self.video_frame = ttk.Frame(status_frame, style="Panel.TFrame")
        self.video_frame.grid(row=1, column=0, sticky="nsew")
        self.video_frame.columnconfigure(0, weight=1)
        self.video_frame.rowconfigure(0, weight=1)
        self.placeholder = tk.Label(
            self.video_frame,
            text="No camera connected.\n\nUse the controls on the left to connect a laptop webcam,\n"
                 "an RTSP IP camera or a test video file.",
            bg=PANEL,
            fg="#7c8794",
            font=("Segoe UI", 11),
        )
        self.placeholder.grid(row=0, column=0, sticky="nsew")

        # Alert history
        history = ttk.LabelFrame(right, text="ALERT HISTORY", padding=6)
        history.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        columns = ("time", "camera", "type", "message")
        self.alert_tree = ttk.Treeview(history, columns=columns, show="headings", height=6)
        self.alert_tree.heading("time", text="Time")
        self.alert_tree.heading("camera", text="Camera")
        self.alert_tree.heading("type", text="Type")
        self.alert_tree.heading("message", text="Message")
        self.alert_tree.column("time", width=90, stretch=False)
        self.alert_tree.column("camera", width=140, stretch=False)
        self.alert_tree.column("type", width=80, stretch=False)
        self.alert_tree.column("message", width=460)
        tree_scroll = ttk.Scrollbar(history, orient="vertical", command=self.alert_tree.yview)
        self.alert_tree.configure(yscrollcommand=tree_scroll.set)
        self.alert_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        ttk.Button(history, text="Clear history", command=self.alert_manager.clear_events).pack(
            anchor="e", pady=(4, 0)
        )

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------
    def _apply_settings_to_widgets(self) -> None:
        values = self.settings.to_dict()
        for key, var in self.setting_vars.items():
            var.set(str(values.get(key, "")))
        self.sleep_audio_entry.delete(0, "end")
        self.sleep_audio_entry.insert(0, self.settings.sleep_audio_path)
        self.phone_audio_entry.delete(0, "end")
        self.phone_audio_entry.insert(0, self.settings.phone_audio_path)

    def _collect_settings(self) -> Settings:
        data = self.settings.to_dict()
        for key, var in self.setting_vars.items():
            text = var.get().strip()
            if not text:
                continue
            try:
                data[key] = type(data[key])(text)
            except (TypeError, ValueError):
                messagebox.showwarning("Invalid value", f"Could not read the value for '{key}'.")
        settings = Settings.from_dict(data)
        settings.sleep_audio_path = self.sleep_audio_entry.get().strip()
        settings.phone_audio_path = self.phone_audio_entry.get().strip()
        return settings

    def _apply_settings(self) -> None:
        settings = self._collect_settings()
        self.settings = settings
        self.engine.apply_settings(settings)
        self.alert_manager.set_audio_path(SLEEP_ALERT, Path(settings.sleep_audio_path))
        self.alert_manager.set_audio_path(PHONE_ALERT, Path(settings.phone_audio_path))
        save_settings(settings)
        self._refresh_audio_warning()
        messagebox.showinfo("Settings", "Settings applied and saved.")

    # ------------------------------------------------------------------
    # Camera handling
    # ------------------------------------------------------------------
    def _source_type_changed(self) -> None:
        choice = self.source_type.current()
        if choice == 0:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, str(self.settings.default_camera_index))
            self.user_entry.config(state="disabled")
            self.password_entry.config(state="disabled")
        elif choice == 1:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, "rtsp://")
            self.user_entry.config(state="normal")
            self.password_entry.config(state="normal")
        else:
            self.source_entry.delete(0, "end")
            self.user_entry.config(state="disabled")
            self.password_entry.config(state="disabled")

    def _connect_camera(self) -> None:
        location = self.source_entry.get().strip()
        if not location:
            messagebox.showwarning("Missing input", "Enter a camera index, an RTSP URL or a video file.")
            return

        choice = self.source_type.current()
        if choice == 0:
            try:
                int(location)
            except ValueError:
                messagebox.showerror("Invalid camera", "The camera index must be a number (0, 1, 2 ...).")
                return
            source_type = SourceType.WEBCAM
            username = password = ""
        elif choice == 1:
            if not location.lower().startswith("rtsp://"):
                messagebox.showerror("Invalid URL", "An RTSP URL has to start with rtsp://")
                return
            source_type = SourceType.RTSP
            username = self.user_entry.get().strip()
            password = self.password_entry.get()
        else:
            path = Path(location)
            if not path.exists():
                messagebox.showerror("File not found", f"The file does not exist:\n{path}")
                return
            source_type = SourceType.FILE
            username = password = ""

        source = self.camera_manager.create_source(
            source_type,
            location,
            name=self.name_entry.get().strip(),
            username=username,
            password=password,
        )
        stream = self.camera_manager.add(source)
        stream.start()
        self.engine.add_camera(stream)
        self._create_panel(source.source_id, source.name)
        self.camera_list.insert("end", f"{source.name}  [{source.describe()}]")
        self.name_entry.delete(0, "end")
        self._refresh_status_labels()

    def _disconnect_camera(self) -> None:
        selection = self.camera_list.curselection()
        if not selection:
            messagebox.showinfo("No camera", "Select a camera in the list first.")
            return
        index = selection[0]
        source_id = self._source_id_for_entry(index)
        if source_id:
            self.engine.remove_camera(source_id)
            self.camera_manager.remove(source_id)
            panel = self.panels.pop(source_id, None)
            if panel is not None:
                panel.destroy()
        self.camera_list.delete(index)
        self._update_placeholder()

    def _source_id_for_entry(self, list_index: int) -> Optional[str]:
        streams = self.camera_manager.streams
        if 0 <= list_index < len(streams):
            return streams[list_index].source.source_id
        return None

    def _create_panel(self, source_id: str, name: str) -> None:
        panel = CameraPanel(self.video_frame, source_id, name)
        self.panels[source_id] = panel
        self._layout_panels()

    def _layout_panels(self) -> None:
        panels = list(self.panels.values())
        if not panels:
            return
        columns = 1 if len(panels) == 1 else 2
        for index, panel in enumerate(panels):
            panel.grid(row=index // columns, column=index % columns, sticky="nsew",
                       padx=4, pady=4)
            self.video_frame.columnconfigure(index % columns, weight=1)
            self.video_frame.rowconfigure(index // columns, weight=1)
        self._update_placeholder()

    def _update_placeholder(self) -> None:
        if self.panels:
            self.placeholder.grid_forget()
        else:
            self.placeholder.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def _browse_audio(self, alert_type: str) -> None:
        initial = str(ASSETS_DIR)
        path = filedialog.askopenfilename(
            title="Select alert sound",
            initialdir=initial,
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg"), ("All files", "*.*")],
        )
        if not path:
            return
        if alert_type == SLEEP_ALERT:
            self.sleep_audio_entry.delete(0, "end")
            self.sleep_audio_entry.insert(0, path)
        else:
            self.phone_audio_entry.delete(0, "end")
            self.phone_audio_entry.insert(0, path)
        self.alert_manager.set_audio_path(alert_type, Path(path))
        self._refresh_audio_warning()

    def _test_audio(self, alert_type: str) -> None:
        path = Path(self.alert_manager.audio_path(alert_type) or Path())
        if not path.exists():
            messagebox.showwarning(
                "Audio file missing",
                f"The alert file '{path.name}' was not found.\n\n"
                f"Place your sound file in:\n{ASSETS_DIR}\n\n"
                "or use Browse to select it manually.",
            )
            return
        self.alert_manager.trigger(
            alert_type,
            message="Audio test",
            camera_id="test",
            camera_name="Test",
            force=True,
        )

    def _refresh_audio_warning(self) -> None:
        missing = self.alert_manager.missing_audio_files()
        if missing:
            self.audio_warning.config(
                text=(
                    "Missing audio file(s): " + ", ".join(missing) + "\n"
                    f"Copy your sound files into {ASSETS_DIR} or use Browse."
                )
            )
        else:
            self.audio_warning.config(text="All alert sounds are ready.", foreground=GREEN)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def _start_monitoring(self) -> None:
        self.engine.start()
        self._refresh_status_labels()

    def _stop_monitoring(self) -> None:
        self.engine.stop()
        self._refresh_status_labels()

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------
    def _refresh_loop(self) -> None:
        if self._closing:
            return
        try:
            self._refresh_video()
            self._refresh_status_labels()
            self._refresh_alerts()
        except Exception as exc:  # pragma: no cover - GUI safety net
            LOGGER.debug("Refresh failed: %s", exc)
        self.after(100, self._refresh_loop)

    def _refresh_video(self) -> None:
        for source_id, panel in self.panels.items():
            worker = self.engine.worker(source_id)
            if worker is None:
                panel.clear()
                continue
            panel.update_frame(worker.annotated_frame, worker.status)

    def _refresh_status_labels(self) -> None:
        statuses = self.engine.statuses()
        if not statuses:
            self.status_labels["monitoring"].config(text="Monitoring: stopped", fg=YELLOW)
            self.status_labels["sleep"].config(text="Sleep detection: --", fg=YELLOW)
            self.status_labels["phone"].config(text="Phone detection: --", fg=YELLOW)
            return

        monitoring = self.engine.monitoring
        sleeping = any(status.sleeping for status in statuses)
        phone = any(status.phone_detected for status in statuses)
        disconnected = any(not status.connected for status in statuses)

        if monitoring and disconnected:
            self.status_labels["monitoring"].config(text="Monitoring: camera issue", fg=YELLOW)
        elif monitoring:
            self.status_labels["monitoring"].config(text="Monitoring: active", fg=GREEN)
        else:
            self.status_labels["monitoring"].config(text="Monitoring: stopped", fg=YELLOW)

        self.status_labels["sleep"].config(
            text="Sleep detection: SLEEPING" if sleeping else "Sleep detection: awake",
            fg=RED if sleeping else GREEN,
        )
        self.status_labels["phone"].config(
            text="Phone detection: PHONE" if phone else "Phone detection: clear",
            fg=RED if phone else GREEN,
        )

    def _refresh_alerts(self) -> None:
        events = self.alert_manager.events
        if len(events) < self._shown_alerts:
            self.alert_tree.delete(*self.alert_tree.get_children())
            self._shown_alerts = 0
        for event in events[self._shown_alerts:]:
            self.alert_tree.insert(
                "",
                "end",
                values=(event.timestamp, event.camera_name or "-", event.alert_type.upper(),
                        event.message),
            )
        if len(events) > self._shown_alerts:
            self._shown_alerts = len(events)
            children = self.alert_tree.get_children()
            if children:
                self.alert_tree.see(children[-1])

    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._closing = True
        try:
            self.engine.stop()
            self.camera_manager.stop_all()
            self.alert_manager.stop()
        finally:
            self.destroy()
