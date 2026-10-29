"""Demo-safe "live" posture stream: replays real recorded posture sessions
through the trained risk-level model as if they were arriving live from a
wristband/IMU, for the 1-minute demo video.

Why replay real data instead of a live sensor: this project's live
Empatica/Polar pipeline streams EDA/temp/pulse, not the Y_Pitch angle this
posture pathway was trained on -- there's no live posture sensor wired up
yet. Rather than fabricate synthetic angle data, this replays three REAL
recorded sessions from posture_cleaned.csv back-to-back, at
~1 window/second, through the actual trained model
(models/posture_risk_forecast_model.joblib) and the actual feedback state
machine (posture_feedback.py) -- the exact same code path a live sensor feed
would run through. Swapping this script for a real posture sensor later only
means replacing _build_replay_windows()'s CSV read with a live queue; the
model, feature engineering, and feedback logic don't change.

Serves the same kind of JSON-over-HTTP /latest endpoint the other pollers in
this repo use (see mock_poller.py), on its own port so it doesn't collide
with the EDA/temp/pulse poller.

Demo arc (REPLAY_SEQUENCE below), picked and verified against the trained
model's actual predictions (see team2_posture_risk_deliverables/TECHNICAL_WRITEUP.md):
  P6 without (~28 windows, mostly low/medium) -- "good" posture
  P1 with    (~12 windows, pct>45deg=43%)     -- "bad" posture; the model
             predicts >=3 consecutive "high" windows here TWICE, each
             firing the feedback trigger
  P5 without (~19 windows, mostly low)        -- "good" posture again,
             resolving the feedback
~59 windows * 0.9s/window = ~53s per loop, then it repeats.

Usage:
    python train_posture_risk_model.py   # once, to produce the model file
    python posture_live_replay.py        # this script

The model and the replay data load immediately, but playback does NOT start
on its own -- it waits (serving an "idle" /latest) until something calls
POST /start, so you can get the dashboard up first and start the data flowing
on cue (e.g. right after you hit record). Call POST /restart at any point to
jump playback back to the beginning of the good->bad->good arc for another
take, without restarting this process. The dashboard's Posture tab has
buttons for both.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify

from posture_feedback import INITIAL_STATE as FEEDBACK_INITIAL_STATE
from posture_feedback import advance_posture_feedback
from posture_ml.windows import RISK_LEVELS, build_windows, load_readings
from tts_utils import init_engine, speak

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "posture_cleaned.csv"
MODEL_PATH = ROOT / "models" / "posture_risk_forecast_model.joblib"
METADATA_PATH = ROOT / "models" / "posture_risk_forecast_model_metadata.json"

PORT = int(os.environ.get("POSTURE_PORT", "7002"))
PARTICIPANT_ID = os.environ.get("POSTURE_DEMO_PARTICIPANT_ID", "DEMO-POSTURE-01")
WINDOW_SECONDS = float(os.environ.get("POSTURE_REPLAY_WINDOW_SECONDS", "0.9"))
SPEAK_FEEDBACK = os.environ.get("POSTURE_SPEAK_FEEDBACK", "1").lower() not in ("0", "false")

HIGH_RISK_MESSAGE = (
    "Posture alert. You've been in a high risk position for three seconds. "
    "Please sit up and adjust your posture."
)
RESOLVED_MESSAGE = "Posture back to a safe range. Nice recovery."

REPLAY_SEQUENCE = [(6, "without", 1), (1, "with", 1), (5, "without", 1)]

app = Flask(__name__)
_lock = threading.Lock()
_start_event = threading.Event()
_restart_requested = threading.Event()
latest_sample = {
    "participant_id": PARTICIPANT_ID,
    "status": "loading",  # "loading" -> "idle" -> "running"
    "updated_at": None,
    "source": (
        "posture_live_replay.py -- REPLAYED from real recorded sessions in "
        "posture_cleaned.csv, standing in for a live posture "
        "sensor feed. Model inference and feedback logic are real, not scripted."
    ),
    "replay_segment": None,
    "angle_deg": None,
    "window_features": None,
    "risk": None,
    "consecutive_high": 0,
    "feedback_active": False,
    "feedback_event": None,
    "system_message": None,
}


@app.after_request
def _add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/latest", methods=["GET"])
def latest():
    with _lock:
        return jsonify(latest_sample)


@app.route("/start", methods=["POST", "GET"])
def start():
    """Begin playback. No-op if already running (so clicking the dashboard's
    Start button twice can't do anything surprising)."""
    _start_event.set()
    with _lock:
        already_running = latest_sample["status"] == "running"
    return jsonify({"ok": True, "already_running": already_running})


@app.route("/restart", methods=["POST", "GET"])
def restart():
    """Jump playback back to the start of the good->bad->good arc (and reset
    the feedback streak) for another take, without restarting the process.
    Implies /start, so it also works before the first start."""
    _start_event.set()
    _restart_requested.set()
    return jsonify({"ok": True})


def _load_model_and_features():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} / {METADATA_PATH} not found -- run "
            "`python train_posture_risk_model.py` first to produce the trained model."
        )
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, meta["selected_model"]["features"], meta["selected_model"]["name"]


def _build_replay_windows() -> pd.DataFrame:
    readings = load_readings(CSV_PATH)
    frames = []
    for participant, condition, segment in REPLAY_SEQUENCE:
        subset = readings[
            (readings["Participant"] == participant)
            & (readings["Condition"] == condition)
            & (readings["Segment"] == segment)
        ]
        windows = build_windows(subset)
        windows["replay_label"] = f"P{participant} {condition}"
        frames.append(windows)
    return pd.concat(frames, ignore_index=True)


def replay_loop():
    model, feature_columns, model_name = _load_model_and_features()
    windows = _build_replay_windows()
    loop_seconds = len(windows) * WINDOW_SECONDS
    print(
        f"[posture_live_replay] model={model_name!r} features={feature_columns} "
        f"windows={len(windows)} ({loop_seconds:.0f}s/loop at {WINDOW_SECONDS}s/window)"
    )

    engine = init_engine() if SPEAK_FEEDBACK else None
    if not SPEAK_FEEDBACK:
        print("[posture_live_replay] TTS disabled (POSTURE_SPEAK_FEEDBACK=0)")

    with _lock:
        latest_sample["status"] = "idle"
    print(f"[posture_live_replay] ready -- waiting for POST /start (or GET http://127.0.0.1:{PORT}/start) to begin playback")
    _start_event.wait()
    _restart_requested.clear()  # a /restart called before the first /start just meant "start"
    with _lock:
        latest_sample["status"] = "running"
    print("[posture_live_replay] playback started")

    feedback_state = FEEDBACK_INITIAL_STATE

    while True:
        for _, row in windows.iterrows():
            if _restart_requested.is_set():
                _restart_requested.clear()
                print("[posture_live_replay] restart requested -- jumping back to the start of the loop")
                break  # falls through to the loop-end reset below, then re-enters `while True`

            # A one-row DataFrame (not a bare ndarray) so sklearn sees the same
            # column names/order it was fit on -- avoids a spurious "X does
            # not have valid feature names" warning on every tick.
            features = pd.DataFrame([row[feature_columns]], columns=feature_columns)
            pred_code = int(model.predict(features)[0])
            pred_label = RISK_LEVELS[pred_code]

            feedback_state, events = advance_posture_feedback(feedback_state, PARTICIPANT_ID, pred_label)

            system_message = None
            for event in events:
                if event.kind == "high_risk_feedback":
                    print(
                        "[posture_live_replay] *** 3 CONSECUTIVE HIGH-RISK PREDICTIONS *** "
                        f"segment={row['replay_label']} -- firing feedback"
                    )
                    system_message = {
                        "text": HIGH_RISK_MESSAGE,
                        "kind": "high_risk_feedback",
                        "said_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    if engine is not None:
                        speak(engine, HIGH_RISK_MESSAGE)
                elif event.kind == "resolved":
                    print(f"[posture_live_replay] posture risk resolved segment={row['replay_label']}")
                    system_message = {
                        "text": RESOLVED_MESSAGE,
                        "kind": "resolved",
                        "said_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    if engine is not None:
                        speak(engine, RESOLVED_MESSAGE)

            with _lock:
                latest_sample.update(
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "replay_segment": row["replay_label"],
                        "angle_deg": round(float(row["mean"]), 1),
                        "window_features": {col: round(float(row[col]), 3) for col in feature_columns},
                        "risk": {"label": pred_label, "code": pred_code},
                        "consecutive_high": feedback_state.consecutive_high,
                        "feedback_active": feedback_state.feedback_active,
                        "feedback_event": events[0].kind if events else None,
                    }
                )
                if system_message is not None:
                    latest_sample["system_message"] = system_message

            print(
                f"[posture_live_replay] seg={row['replay_label']:10s} angle={row['mean']:6.1f} "
                f"-> risk={pred_label:6s} streak={feedback_state.consecutive_high}"
            )
            time.sleep(WINDOW_SECONDS)

        # Loop: start a fresh "session" each pass (participant_id unchanged,
        # but feedback_state reset) so repeated demo takes behave identically
        # instead of carrying a streak across the loop boundary.
        feedback_state = FEEDBACK_INITIAL_STATE


if __name__ == "__main__":
    threading.Thread(target=replay_loop, daemon=True).start()
    print(f"[posture_live_replay] serving http://127.0.0.1:{PORT}/latest")
    app.run(host="0.0.0.0", port=PORT)
