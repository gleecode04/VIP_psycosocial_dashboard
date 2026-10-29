"""Specialized speech system for the construction-worker-POV in-class
activity: two-tier verbal feedback driven by the live frustration/stress
label.

Round 1 -- the moment stress first reads "high":
    "Are you feeling tired? Please have some water."

Escalation -- stress has stayed "high" continuously for ESCALATION_MINUTES:
    "Please take a break. You've been stressed for {N} minutes."
    (repeats every ESCALATION_REPEAT_MINUTES after that, as long as the
    worker is still reading "high")

Why 45 minutes by default: NIOSH's heat-stress guidance recommends a
45-minutes-work / 15-minutes-rest cycle for heavy manual labor around a
95 F heat index (NIOSH Pub. 2016-106, "Criteria for a Recommended
Standard: Occupational Exposure to Heat and Hot Environments"; summarized
in OSHA's "Using the Heat Index: A Guide for Employers"). That is the
standard work/rest cadence this kind of system is meant to reinforce for a
construction crew, so we reuse 45 minutes as the escalation interval for a
continuously elevated stress reading. Override it with ESCALATION_MINUTES
for a classroom demo.

All the escalation/timing logic lives in stress_state.py as a pure, unit
tested function (see test_stress_state.py) -- this file is just the I/O
loop around it: poll /latest, feed the state machine, speak what it says
to speak.

Usage:
    python avro_stream_poller_with_ml.py        # terminal 1, real data
    # or, for a demo that doesn't depend on AWS / S3 / the trained models:
    python mock_poller.py                       # terminal 1, synthetic data

    python construction_stress_speech.py        # terminal 2
"""
import os
import time

import requests

from stress_state import INITIAL_STATE, advance_state, get_sample_time, get_stress_label
from tts_utils import init_engine, speak

API_URL = os.environ.get("BIOMARKER_API_URL", "http://localhost:7001/latest")
# Same host:port as API_URL, just a different path -- both
# avro_stream_poller_with_ml.py and mock_poller.py expose this (see
# system_message_api.py) so the dashboard can show the last thing the
# voice system said, as a fallback if a screen recording's audio capture
# fails or doesn't pick up system TTS output.
SYSTEM_MESSAGE_URL = API_URL.rsplit("/latest", 1)[0] + "/system-message"

# Faster than mock_poller.py's default 2s tick, so this never skips a tick
# and misses an escalation threshold crossing. Fine against the real
# poller too -- an extra local HTTP call per second costs nothing.
POLL_SECONDS = float(os.environ.get("SPEECH_POLL_SECONDS", "1"))

ESCALATION_MINUTES = float(os.environ.get("ESCALATION_MINUTES", "45"))
ESCALATION_REPEAT_MINUTES = float(os.environ.get("ESCALATION_REPEAT_MINUTES", str(ESCALATION_MINUTES)))

ROUND1_MESSAGE = "Are you feeling tired? Please have some water."


def escalation_message(elapsed_minutes):
    return f"Please take a break. You've been stressed for {round(elapsed_minutes)} minutes."


def fetch_sample():
    resp = requests.get(API_URL, timeout=5)
    resp.raise_for_status()
    return resp.json()


def report_system_message(text, kind="info"):
    """Best-effort: the dashboard banner is a nice-to-have fallback, not a
    dependency the speech loop should ever crash over."""
    try:
        requests.post(SYSTEM_MESSAGE_URL, json={"text": text, "kind": kind}, timeout=3)
    except Exception as e:
        print(f"[construction_stress_speech] could not report system message to dashboard: {e}")


def main():
    if ESCALATION_MINUTES <= 0 or ESCALATION_REPEAT_MINUTES <= 0:
        raise ValueError("ESCALATION_MINUTES and ESCALATION_REPEAT_MINUTES must both be > 0")

    engine = init_engine()
    state = INITIAL_STATE

    print(
        f"Watching {API_URL} every {POLL_SECONDS}s "
        f"(escalation at {ESCALATION_MINUTES} min, repeat every {ESCALATION_REPEAT_MINUTES} min)..."
    )

    while True:
        try:
            sample = fetch_sample()
        except Exception as e:
            print(f"[construction_stress_speech] fetch error: {e}")
            time.sleep(POLL_SECONDS)
            continue

        participant_id = sample.get("participant_id")
        stress_label = get_stress_label(sample)

        if participant_id and stress_label:
            sample_time = get_sample_time(sample)
            state, events = advance_state(
                state,
                participant_id,
                stress_label,
                sample_time,
                ESCALATION_MINUTES,
                ESCALATION_REPEAT_MINUTES,
            )
            for event in events:
                if event.kind == "round1":
                    print(f"[construction_stress_speech] *** STRESS LEVEL EXCEEDED *** "
                          f"participant={participant_id} -- starting a new high-stress episode.")
                    report_system_message(ROUND1_MESSAGE, kind="round1")
                    speak(engine, ROUND1_MESSAGE)
                elif event.kind == "escalation":
                    print(f"[construction_stress_speech] escalation reminder -- "
                          f"participant={participant_id} continuously high for {round(event.elapsed_minutes)} min.")
                    text = escalation_message(event.elapsed_minutes)
                    report_system_message(text, kind="escalation")
                    speak(engine, text)
                elif event.kind == "episode_ended":
                    text = (f"Stress episode ended (was high for {round(event.elapsed_minutes)} min). "
                            f"Next high reading starts a fresh episode.")
                    print(f"[construction_stress_speech] {text}")
                    report_system_message(text, kind="episode_ended")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
