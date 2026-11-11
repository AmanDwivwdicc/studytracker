// Alert engine: keeps the alert history, applies the cooldowns and decides
// when a sleep / phone alert has to be fired.
//
// The audio itself is played by the browser (the alert event is pushed over
// the WebSocket), so a long sound never blocks the detection pipeline.
export const ALERT_SLEEP = "sleep";
export const ALERT_PHONE = "phone";

export class AlertEngine {
  constructor(settings) {
    this.settings = settings;
    this.events = [];
    this.lastFired = { sleep: 0, phone: 0 };
    this.phoneSince = 0;
    this.sleepSince = 0;
    this.phoneActive = false;
    this.sleepActive = false;
  }

  updateSettings(settings) {
    this.settings = settings;
  }

  cooldownFor(type) {
    return type === ALERT_SLEEP ? this.settings.sleep_alert_cooldown : this.settings.phone_alert_cooldown;
  }

  // Called when the browser reports that eyes are closed / open.
  handleEyeState({ cameraId, cameraName, closed }) {
    const now = Date.now();
    if (closed) {
      if (!this.sleepSince) this.sleepSince = now;
      const heldMs = now - this.sleepSince;
      if (heldMs >= this.settings.min_eye_closure_duration * 1000 && !this.sleepActive) {
        this.sleepActive = true;
        return this.fire(ALERT_SLEEP, {
          cameraId,
          cameraName,
          message: "WAKE UP! YOU ARE SLEEPING.",
        });
      }
    } else {
      this.sleepSince = 0;
      this.sleepActive = false;
    }
    return null;
  }

  // Called when the phone detector reports phones (or none).
  handlePhoneState({ cameraId, cameraName, count }) {
    const now = Date.now();
    if (count > 0) {
      if (!this.phoneSince) this.phoneSince = now;
      const heldMs = now - this.phoneSince;
      if (heldMs >= this.settings.min_phone_detection_duration * 1000 && !this.phoneActive) {
        this.phoneActive = true;
        return this.fire(ALERT_PHONE, {
          cameraId,
          cameraName,
          message: "WARNING! MOBILE PHONE DETECTED.",
        });
      }
    } else {
      this.phoneSince = 0;
      this.phoneActive = false;
    }
    return null;
  }

  fire(type, { cameraId = "", cameraName = "", message = "" }) {
    const now = Date.now();
    if (now - this.lastFired[type] < this.cooldownFor(type) * 1000) return null;
    this.lastFired[type] = now;

    const event = {
      id: `${now}_${type}`,
      timestamp: new Date(now).toLocaleTimeString("en-GB"),
      type,
      cameraId,
      cameraName,
      message: message || (type === ALERT_SLEEP ? "WAKE UP! YOU ARE SLEEPING." : "WARNING! MOBILE PHONE DETECTED."),
      audio: type === ALERT_SLEEP ? this.settings.sleep_audio : this.settings.phone_audio,
    };
    this.events.unshift(event);
    if (this.events.length > 300) this.events.length = 300;
    return event;
  }

  clear() {
    this.events = [];
  }
}
