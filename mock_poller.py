"""Demo-safe stand-in for avro_stream_poller_with_ml.py.

Serves the exact same /latest JSON shape, but from a scripted synthetic
scenario instead of S3 + the trained models -- no AWS credentials, no
network, no model compatibility to worry about. Use this if the real
poller can't reach S3 during a live demo/presentation.

It runs a looping scenario: low -> medium -> high (held long enough to
cross the escalation threshold twice) -> back to low. The simulated
clock advances faster than real time (SIM_MINUTES_PER_TICK per TICK_SECONDS)
so construction_stress_speech.py's 45-minute escalation logic -- which
tracks elapsed time from the data's own timestamps, not the wall clock --
fires within a short, demo-friendly window. Defaults: 22 simulated
minutes per 2s tick, ~50s per full loop. Each "high" episode is defined in
simulated minutes (SCENARIO_MINUTES below), not a fixed tick count, so
overriding SIM_MINUTES_PER_TICK can't accidentally shrink an episode below
the escalation threshold.

Run it exactly like the real poller:
    python mock_poller.py
and point speak_biomarker_labels.py / construction_stress_speech.py at it
the same way (BIOMARKER_API_URL defaults to http://localhost:7001/latest
for both, so no extra configuration is needed).
"""
import os
import random
import threading
import time
from datetime import datetime, timedelta

from flask import Flask, jsonify

from risk_score import RollingRiskScorer
from system_message_api import register_system_message_routes

SIM_MINUTES_PER_TICK = float(os.environ.get("SIM_MINUTES_PER_TICK", "22"))
TICK_SECONDS = float(os.environ.get("TICK_SECONDS", "2"))
PARTICIPANT_ID = os.environ.get("MOCK_PARTICIPANT_ID", "DEMO-P01")

# (stage label, how many SIMULATED MINUTES to hold it). Defined in minutes,
# not tick counts, on purpose: construction_stress_speech.py's escalation
# threshold (ESCALATION_MINUTES, default 45) is a minutes value, so if the
# scenario were pinned to a fixed tick count instead, overriding
# SIM_MINUTES_PER_TICK (e.g. to slow the demo down) would silently shrink
# every episode's real duration and the escalation message might never
# fire at all -- that's the bug this replaced. Tick counts are now derived
# from these minute targets at startup (see _ticks_for), so every episode
# reaches its intended duration regardless of SIM_MINUTES_PER_TICK.
#
# Three separate "high" episodes per loop, each demonstrating a different
# part of the TTS behavior. The risk score (risk_score.py) that drives
# round-1/escalation and the dashboard's threshold line is a direct,
# deterministic function of the label -- no rolling window, no ramp-up
# tick -- so it crosses the 78 cutoff the instant a "high" tick is served,
# exactly in step with what the chart draws:
#   Episode 1 (~85 min): round 1 ("please have some water"), then one
#     escalation reminder once 45 min elapsed, ending before the next one.
#   Episode 2 (~30 min): round 1 only -- ends well before the 45-minute
#     escalation threshold, so you only hear the water message.
#   Episode 3 (~200 min): round 1, then two escalation reminders
#     (at 45 and 90+ min elapsed), showing the repeat behavior with
#     headroom before a third would fire.
# Between episodes, stress drops back to low/medium so each new "high"
# fires round 1 again as a fresh episode (see stress_state.py).
SCENARIO_MINUTES = [
    ("low", 20),
    ("medium", 20),
    ("high", 85),     # episode 1: round 1 + 1 escalation
    ("medium", 20),
    ("low", 20),
    ("high", 30),      # episode 2: round 1 only
    ("low", 20),
    ("medium", 20),
    ("high", 200),     # episode 3: round 1 + 2 escalations
    ("low", 40),
]

STAGE_RANGES = {
    "low": {"eda": (0.3, 0.6), "temp": (33.0, 34.0), "pulse": (65, 75), "confidence": (0.75, 0.95)},
    "medium": {"eda": (0.8, 1.3), "temp": (33.5, 34.5), "pulse": (75, 90), "confidence": (0.6, 0.85)},
    "high": {"eda": (1.5, 2.5), "temp": (34.5, 35.5), "pulse": (90, 110), "confidence": (0.65, 0.9)},
}

app = Flask(__name__)
_lock = threading.Lock()
latest_sample = {
    "eda": None,
    "temp": None,
    "pulse_rate": None,
    "eda_time": None,
    "temp_time": None,
    "pulse_time": None,
    "participant_id": None,
    "source_prefix": "mock://demo-scenario",
    "updated_at": None,
    "observed_latency_seconds": 0.0,
    "note": "mock poller warming up",
    "predictions": None,
}


@app.route("/latest", methods=["GET"])
def latest():
    with _lock:
        return jsonify(latest_sample)


register_system_message_routes(app, latest_sample, lock=_lock)


def _build_scenario(sim_minutes_per_tick):
    """Convert SCENARIO_MINUTES into (label, tick_count) pairs for whatever
    SIM_MINUTES_PER_TICK is actually configured, so every episode always
    lasts its intended number of simulated minutes."""
    return [
        (label, max(1, round(minutes / sim_minutes_per_tick)))
        for label, minutes in SCENARIO_MINUTES
    ]


def _stage_at_tick(tick_index, scenario):
    remaining = tick_index
    for label, hold_ticks in scenario:
        if remaining < hold_ticks:
            return label
        remaining -= hold_ticks
    return None  # loop boundary; caller wraps the index first


def _round(value, ndigits=2):
    return round(value, ndigits)


def scenario_loop():
    scenario = _build_scenario(SIM_MINUTES_PER_TICK)
    total_ticks = sum(hold for _, hold in scenario)
    sim_time = datetime.now()
    tick_index = 0
    minute_index = -1
    previous_stage = None
    risk_scorer = RollingRiskScorer()

    print(
        f"Mock poller running: {SIM_MINUTES_PER_TICK} simulated minute(s) per "
        f"{TICK_SECONDS}s tick, {total_ticks} ticks per loop "
        f"(~{total_ticks * TICK_SECONDS:.0f}s real time per loop)."
    )
    print(f"[mock] scenario (label, ticks): {scenario}")

    while True:
        stage = _stage_at_tick(tick_index % total_ticks, scenario)
        ranges = STAGE_RANGES[stage]
        minute_index += 1

        if stage == "high" and previous_stage != "high":
            print(f"[mock] *** STRESS LEVEL EXCEEDED *** scenario entered HIGH stage at tick={tick_index}")
        elif stage != "high" and previous_stage == "high":
            print(f"[mock] scenario left HIGH stage at tick={tick_index} -- back to {stage}")
        previous_stage = stage

        sim_time += timedelta(minutes=SIM_MINUTES_PER_TICK)
        iso_time = sim_time.isoformat(timespec="seconds")

        eda_val = _round(random.uniform(*ranges["eda"]))
        temp_val = _round(random.uniform(*ranges["temp"]))
        pulse_val = _round(random.uniform(*ranges["pulse"]), 1)
        confidence = _round(random.uniform(*ranges["confidence"]))

        mental_confidence = _round(random.uniform(*ranges["confidence"]))
        risk = risk_scorer.score(
            frustration_label=stage,
            mental_label=stage,
            frustration_confidence=confidence,
            mental_confidence=mental_confidence,
        )

        predictions = {
            "session_minute_index": minute_index,
            "history_size": min(minute_index + 1, 10),
            "model_notes": [
                "mock_poller.py: synthetic scenario data for demo purposes only.",
                "Not derived from real EmbracePlus data or the trained models.",
            ],
            "frustration": {
                "label": stage,
                "confidence": confidence,
                "ready": True,
                "reason": None,
            },
            "mental_demand": {
                "label": stage,
                "confidence": mental_confidence,
                "ready": True,
                "reason": None,
            },
            "risk": risk,
        }

        with _lock:
            latest_sample.update(
                {
                    "eda": eda_val,
                    "temp": temp_val,
                    "pulse_rate": pulse_val,
                    "eda_time": iso_time,
                    "temp_time": iso_time,
                    "pulse_time": iso_time,
                    "participant_id": PARTICIPANT_ID,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "observed_latency_seconds": 0.0,
                    "note": f"mock scenario stage={stage}",
                    "predictions": predictions,
                }
            )

        print(f"[mock] tick={tick_index} stage={stage} sim_time={iso_time} eda={eda_val} pulse={pulse_val}")

        tick_index += 1
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    threading.Thread(target=scenario_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=7001)
