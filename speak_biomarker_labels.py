"""Speaks the live Frustration / Mental Demand labels as they change.

This is a reimplementation built from the setup handout's description of
"speak_biomarker_labels.py" (Step 5-6): poll the poller's /latest endpoint
every few seconds, and announce a label only when it changes, e.g.
"Frustration is now High" or "Mental demand is now Pending".

The original file was distributed over Teams chat and was not available
here to copy directly -- if you have it, you can use it instead of this
one. Either way it only needs to agree with avro_stream_poller_with_ml.py's
/latest JSON shape (or mock_poller.py's, which matches it) to work.

Usage:
    python avro_stream_poller_with_ml.py   # terminal 1 (or mock_poller.py)
    python speak_biomarker_labels.py       # terminal 2
"""
import os
import time

import requests

from tts_utils import init_engine, speak

API_URL = os.environ.get("BIOMARKER_API_URL", "http://localhost:7001/latest")
POLL_SECONDS = float(os.environ.get("SPEECH_POLL_SECONDS", "1"))


def fetch_sample():
    resp = requests.get(API_URL, timeout=5)
    resp.raise_for_status()
    return resp.json()


def main():
    engine = init_engine()
    last_frustration = None
    last_mental_demand = None

    print(f"Listening to {API_URL} every {POLL_SECONDS}s...")

    while True:
        try:
            sample = fetch_sample()
        except Exception as e:
            print(f"[speak_biomarker_labels] fetch error: {e}")
            time.sleep(POLL_SECONDS)
            continue

        predictions = sample.get("predictions") or {}
        frustration = (predictions.get("frustration") or {}).get("label")
        mental_demand = (predictions.get("mental_demand") or {}).get("label")

        if frustration is not None and frustration != last_frustration:
            speak(engine, f"Frustration is now {frustration}")
            last_frustration = frustration

        if mental_demand is not None and mental_demand != last_mental_demand:
            speak(engine, f"Mental demand is now {mental_demand}")
            last_mental_demand = mental_demand

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
