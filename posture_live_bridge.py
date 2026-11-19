"""Live posture risk inference from a real Polar H10 accelerometer stream.

This is the live-hardware counterpart to posture_live_replay.py: readings
arrive from a browser's real Web Bluetooth connection to a Polar H10 (raw
Ax/Ay/Az in g), POSTed one at a time to this process.

Two live classification modes, chosen by POSTURE_LIVE_MODE env var:

  "dbr" (default) -- posture_dbr.py's calibrated decision-based rule. Each
      session starts with a two-phase calibration: the wearer holds a
      neutral pose (~1s), then a slight forward lean (~1s), giving the two
      reference vectors posture_dbr.calibrate_axes() needs to decompose
      Pitch (forward/back tilt -- the PRIMARY severity signal per the
      instructor's feedback) from Roll (side-to-side tilt -- secondary,
      cause-only) -- see posture_dbr.py's module docstring and
      PITCH_ROLL_FIX_PLAN.md for why a single-axis formula or a single
      omnidirectional tilt angle can't make that distinction. This mode
      also attributes *why* a reading is High (forward bending, twisting,
      overhead work, ...) so feedback can name a cause, which the trained
      ML model's binary/3-class label alone can't do.

  "ml" -- the original trained-model path (models/posture_risk_forecast_
      model.joblib), fed the exact same chest-strap-orientation Y_Pitch
      formula the model was trained on (computed here from the raw axes,
      since the client no longer pre-computes a single pitch value). Kept
      side by side with "dbr" so the two can be compared, the same way
      decision_based_rule.py exists alongside the ML pathway for EDA/temp.

Either way, everything downstream of "I have a classified reading" is
shared: same WINDOW_SIZE_READINGS window cut, same posture_feedback.py
streak-trigger state machine, same TTS call site -- a live prediction is
held to one pipeline, not two parallel re-implementations of it.

condition_with is fixed at 0 ("without") for "ml" mode: that feature encoded
an experimental condition from the recorded dataset that has no live
equivalent, so there's no principled live value for it -- 0 was the
majority condition in training.
"""
from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import checkin_bridge
from posture_dbr import (
    AxisFrame,
    GravitySample,
    HIGH,
    build_feedback_message,
    build_voice_message,
    calibrate_axes,
    classify_cause,
    classify_tilt_severity,
    compute_window_stats,
    escalate_for_static_duration,
    tilt_angle_deg,
)
from posture_feedback import INITIAL_STATE as FEEDBACK_INITIAL_STATE
from posture_feedback import advance_posture_feedback
from posture_ml.windows import READING_BAD_THRESHOLD_DEG, RISK_LEVELS, WINDOW_SIZE_READINGS
from posture_ml.windows import _featurize_window  # reused on purpose, "ml" mode only

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "posture_risk_forecast_model.joblib"
METADATA_PATH = ROOT / "models" / "posture_risk_forecast_model_metadata.json"

PARTICIPANT_ID = os.environ.get("POSTURE_DEMO_PARTICIPANT_ID", "LIVE-POLAR-01")
SPEAK_FEEDBACK = os.environ.get("POSTURE_SPEAK_FEEDBACK", "1").lower() not in ("0", "false")
LIVE_MODE = os.environ.get("POSTURE_LIVE_MODE", "dbr").lower()
# One window's worth of readings (~1s), held still, is enough to average out
# sensor noise for a neutral baseline without making the wearer wait long.
CALIBRATION_READINGS = int(os.environ.get("POSTURE_CALIBRATION_READINGS", str(WINDOW_SIZE_READINGS)))

HIGH_RISK_MESSAGE = (  # "ml" mode fallback only; "dbr" mode builds a cause-specific message instead
    "Posture alert. You've been in a high risk position for three seconds. "
    "Please sit up and adjust your posture."
)
RESOLVED_MESSAGE = "Posture back to a safe range. Nice recovery."


def _load_model_and_features():
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    import joblib
    model = joblib.load(MODEL_PATH)
    return model, meta["selected_model"]["features"]


def _average_sample(samples: list[GravitySample]) -> GravitySample:
    n = len(samples)
    return GravitySample(
        sum(s.ax for s in samples) / n,
        sum(s.ay for s in samples) / n,
        sum(s.az for s in samples) / n,
    )


def _legacy_chest_pitch_deg(sample: GravitySample) -> float:
    """Same atan2(ax, sqrt(ay^2+az^2)) formula the training CSV's Y_Pitch
    used. Kept only for LIVE_MODE=ml so switching modes doesn't change what
    the trained model sees -- the default "dbr" mode uses tilt_angle_deg
    instead (see posture_dbr.py for why a fixed-axis formula breaks on a
    lower-back mount).
    """
    return math.degrees(math.atan2(sample.ax, math.sqrt(sample.ay ** 2 + sample.az ** 2)))


class PostureLiveBridge:
    """Buffers (timestamp, Ax/Ay/Az) readings from a real Polar H10 into
    calibrated, windowed posture classifications and the shared feedback
    state machine used everywhere else in this project.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model, self._feature_columns = (None, None)
        if LIVE_MODE == "ml":
            self._model, self._feature_columns = _load_model_and_features()
        self._engine = None
        self._buffer: list[tuple[float, GravitySample]] = []
        self._calibration_buffer: list[GravitySample] = []
        self._neutral: Optional[GravitySample] = None
        self._forward_ref: Optional[GravitySample] = None
        self._frame: Optional[AxisFrame] = None
        self._feedback_state = FEEDBACK_INITIAL_STATE
        self._session_active = False
        self._static_cause: Optional[str] = None
        self._static_since_t: Optional[float] = None
        self.latest = {
            "participant_id": PARTICIPANT_ID,
            # disconnected -> idle -> calibrating -> calibrating_forward -> running
            "status": "disconnected",
            "polar_connected": False,
            "mode": LIVE_MODE,
            "updated_at": None,
            "source": (
                "posture_live_bridge.py -- LIVE readings from a real Polar H10 "
                "accelerometer stream (raw Ax/Ay/Az), classified by a "
                + ("calibrated decision-based rule (Pitch-primary/Roll-secondary "
                   "tilt decomposition + axis-pattern cause attribution)" if LIVE_MODE != "ml" else
                   "trained ML model")
                + ". Not replayed or scripted."
            ),
            "replay_segment": None,
            "angle_deg": None,
            "tilt_deg": None,
            "pitch_deg": None,
            "roll_deg": None,
            "cause": None,
            "window_features": None,
            "risk": None,
            "consecutive_high": 0,
            "feedback_active": False,
            "feedback_event": None,
            "feedback_text": None,
            "system_message": None,
        }

    def _speak(self, text: str) -> None:
        if not SPEAK_FEEDBACK:
            return
        if self._engine is None:
            from tts_utils import init_engine
            self._engine = init_engine()
        from tts_utils import speak
        speak(self._engine, text)

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self.latest["polar_connected"] = connected
            if not connected:
                self._reset_session_state()
                self.latest.update({
                    "status": "disconnected",
                    "replay_segment": None,
                    "angle_deg": None,
                    "tilt_deg": None,
                    "pitch_deg": None,
                    "roll_deg": None,
                    "cause": None,
                    "window_features": None,
                    "risk": None,
                    "consecutive_high": 0,
                    "feedback_active": False,
                    "feedback_event": None,
                    "feedback_text": None,
                    "system_message": None,
                })
            elif self.latest["status"] == "disconnected":
                self.latest["status"] = "idle"

    def _reset_session_state(self) -> None:
        self._session_active = False
        self._buffer.clear()
        self._calibration_buffer.clear()
        self._neutral = None
        self._forward_ref = None
        self._frame = None
        self._feedback_state = FEEDBACK_INITIAL_STATE
        self._static_cause = None
        self._static_since_t = None

    def start_session(self) -> None:
        """Also doubles as "Recalibrate": the dashboard's Restart button
        calls this same endpoint, so re-zeroing mid-session (if the strap
        shifted, or the wearer just wants a fresh baseline) never requires a
        full Bluetooth reconnect.
        """
        with self._lock:
            if not self.latest["polar_connected"]:
                return
            self._reset_session_state()
            self._session_active = True
            self.latest.update({
                "status": "calibrating",
                "consecutive_high": 0,
                "feedback_active": False,
                "feedback_event": None,
                "feedback_text": None,
                "system_message": None,
            })

    def add_reading(self, t: float, ax: float, ay: float, az: float) -> None:
        with self._lock:
            if not self._session_active:
                return
            sample = GravitySample(ax, ay, az)

            if self._neutral is None:
                # Phase 1: hold neutral.
                self._calibration_buffer.append(sample)
                self.latest["updated_at"] = datetime.now().isoformat(timespec="seconds")
                if len(self._calibration_buffer) < CALIBRATION_READINGS:
                    return
                self._neutral = _average_sample(self._calibration_buffer)
                self._calibration_buffer.clear()
                # "ml" mode has no phase 2 -- it's untouched, still on the
                # legacy fixed-axis formula (see module docstring).
                self.latest["status"] = "running" if LIVE_MODE == "ml" else "calibrating_forward"
                return

            if LIVE_MODE != "ml" and self._frame is None:
                # Phase 2 ("dbr" mode only): hold a slight forward lean, so
                # calibrate_axes() can isolate Pitch (primary severity
                # signal) from Roll (secondary, cause-only) -- see
                # posture_dbr.py's module docstring / PITCH_ROLL_FIX_PLAN.md.
                self._calibration_buffer.append(sample)
                self.latest["updated_at"] = datetime.now().isoformat(timespec="seconds")
                if len(self._calibration_buffer) < CALIBRATION_READINGS:
                    return
                self._forward_ref = _average_sample(self._calibration_buffer)
                self._calibration_buffer.clear()
                self._frame = calibrate_axes(self._neutral, self._forward_ref)
                self.latest["status"] = "running"
                return

            self._buffer.append((t, sample))
            if len(self._buffer) < WINDOW_SIZE_READINGS:
                return
            window_rows, self._buffer = self._buffer[:WINDOW_SIZE_READINGS], self._buffer[WINDOW_SIZE_READINGS:]
            self._process_window(window_rows)

    def _process_window(self, window_rows: list[tuple[float, GravitySample]]) -> None:
        timestamps = [t for t, _ in window_rows]
        samples = [s for _, s in window_rows]
        tilt_deg = tilt_angle_deg(samples[-1], self._neutral)
        pitch_deg_val = None
        roll_deg_val = None

        if LIVE_MODE == "ml":
            pred_label, window_features = self._predict_with_model(timestamps, samples)
            cause = None
            feedback_text = None
        else:
            stats = compute_window_stats(samples, self._neutral, self._frame)
            # Severity is now Pitch-only (primary signal, per instructor
            # feedback) -- abs() because classify_tilt_severity expects an
            # unsigned magnitude, same as the omnidirectional tilt it used
            # to take; a large lean-BACK should be just as severe as a
            # lean-forward of the same size. Roll never drives severity,
            # only classify_cause()'s side_bending attribution below.
            pred_label = classify_tilt_severity(abs(stats.max_pitch_deg))
            cause = classify_cause(stats)
            pred_label, cause = self._apply_static_escalation(timestamps[-1], pred_label, cause)
            feedback_text = build_feedback_message(pred_label, cause)
            pitch_deg_val = round(stats.max_pitch_deg, 1)
            roll_deg_val = round(stats.max_roll_deg, 1)
            window_features = {
                "mean_tilt_deg": round(stats.mean_tilt_deg, 2),
                "max_tilt_deg": round(stats.max_tilt_deg, 2),
                "mean_pitch_deg": round(stats.mean_pitch_deg, 2),
                "max_pitch_deg": round(stats.max_pitch_deg, 2),
                "mean_roll_deg": round(stats.mean_roll_deg, 2),
                "max_roll_deg": round(stats.max_roll_deg, 2),
                "std_ax": round(stats.std_dax, 4),
                "std_ay": round(stats.std_day, 4),
                "std_az": round(stats.std_daz, 4),
            }

        self._feedback_state, events = advance_posture_feedback(self._feedback_state, PARTICIPANT_ID, pred_label)
        system_message = self._handle_events(events, cause)
        self._consider_checkin(pred_label, window_features)

        self.latest.update({
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "replay_segment": "live",
            "angle_deg": round(tilt_deg, 1),
            "tilt_deg": round(tilt_deg, 1),
            "pitch_deg": pitch_deg_val,
            "roll_deg": roll_deg_val,
            "cause": cause,
            "window_features": window_features,
            "risk": {"label": pred_label, "code": RISK_LEVELS.index(pred_label)},
            "consecutive_high": self._feedback_state.consecutive_high,
            "feedback_active": self._feedback_state.feedback_active,
            "feedback_event": events[0].kind if events else None,
            "feedback_text": feedback_text,
        })
        if system_message is not None:
            self.latest["system_message"] = system_message

    def _consider_checkin(self, pred_label: str, window_features: dict) -> None:
        """Closed-loop step 5 for the POSTURE lane (checkin.py / checkin_bridge.py
        -- see their module docstrings for why posture only gets steps 5-8,
        not retraining). sample_time is wall-clock "now": unlike the
        biomarkers poll loop, there's no S3/network lag to account for here
        -- a window is classified essentially as soon as its readings arrive.
        """
        checkin_bridge.bridge.check_expiry()
        checkin_bridge.bridge.consider_sample(
            PARTICIPANT_ID, None, None, datetime.now(), window_features, posture_label=pred_label,
        )

    def _apply_static_escalation(self, t: float, severity: str, cause: str) -> tuple[str, str]:
        if cause != self._static_cause:
            self._static_cause = cause
            self._static_since_t = t
        unchanged_seconds = t - self._static_since_t
        return escalate_for_static_duration(severity, cause, unchanged_seconds)

    def _handle_events(self, events, cause: Optional[str]) -> Optional[dict]:
        system_message = None
        for event in events:
            if event.kind == "high_risk_feedback":
                text = build_voice_message(HIGH, cause) if LIVE_MODE != "ml" else HIGH_RISK_MESSAGE
                text = text or HIGH_RISK_MESSAGE
                system_message = {
                    "text": text, "kind": "high_risk_feedback",
                    "said_at": datetime.now().isoformat(timespec="seconds"),
                }
                self._speak(text)
            elif event.kind == "resolved":
                system_message = {
                    "text": RESOLVED_MESSAGE, "kind": "resolved",
                    "said_at": datetime.now().isoformat(timespec="seconds"),
                }
                self._speak(RESOLVED_MESSAGE)
        return system_message

    def _predict_with_model(self, timestamps: list[float], samples: list[GravitySample]) -> tuple[str, dict]:
        pitches = [_legacy_chest_pitch_deg(s) for s in samples]
        chunk = pd.DataFrame({"Timestamp": timestamps, "Y_Pitch": pitches})
        chunk["reading_bad"] = (chunk["Y_Pitch"] > READING_BAD_THRESHOLD_DEG).astype(int)
        features = _featurize_window(chunk)
        features["condition_with"] = 0
        row = pd.DataFrame([{c: features[c] for c in self._feature_columns}], columns=self._feature_columns)
        pred_code = int(self._model.predict(row)[0])
        window_features = {c: round(float(features[c]), 3) for c in self._feature_columns}
        return RISK_LEVELS[pred_code], window_features

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self.latest)

    def get_checkin_context(self) -> Optional[dict]:
        """Snapshot for the posture-side "How's your posture? Tell CHAP"
        self-report button -- mirrors LiveDashboardBridge.get_live_context()
        for the biomarkers pathway. Returns None before the session is
        running (no classified window yet, so no posture_label or features
        to attach the self-report to)."""
        with self._lock:
            if self.latest["status"] != "running" or self.latest["risk"] is None:
                return None
            return {
                "participant_id": PARTICIPANT_ID,
                "posture_label": self.latest["risk"]["label"],
                "features_snapshot": dict(self.latest["window_features"] or {}),
            }


bridge = PostureLiveBridge()
