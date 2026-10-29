"""Decision-Based Rule (DBR) classifier -- a PLACEHOLDER rule engine.

This is *not* derived from the instructor's "POLAR H10 PREDICTION - Decision
Based Rule" doc (decisionbasedrule/*.docx/pdf) -- that doc sketches the
concept (EDA + HRV + temperature -> emotional arousal/distress) and a Polar
H10 chest-strap pipeline we aren't wearing, but it doesn't specify numeric
cutoffs, and those cutoffs are highly individual (resting EDA varies a lot
person to person, sensor to sensor). So rather than invent an absolute
"EDA > X uS = high" threshold, this rule classifies relative to each
participant's own baseline:

    EDA ratio   = current EDA / mean EDA over the first few minutes
    temp signal = skin temp dropping vs. that same baseline (peripheral
                  vasoconstriction under sympathetic arousal)

    high   if EDA ratio crosses HIGH_EDA_RATIO, or crosses MEDIUM_EDA_RATIO
           *and* temp is dropping (both signals agree)
    medium if EDA ratio crosses MEDIUM_EDA_RATIO alone
    low    otherwise

Swap in the instructor's real thresholds here once they're defined -- this
exists so you can run it side by side with the ML terminal today and see
where the two agree/disagree.

Usage (reads the same /latest the ML poller publishes -- run that first):
    python avro_stream_poller_with_ml.py   # terminal 1, real data
    python decision_based_rule.py          # terminal 2, this rule engine
"""
import os
import time
from collections import deque

import requests

API_URL = os.environ.get("BIOMARKER_API_URL", "http://localhost:7001/latest")
POLL_SECONDS = float(os.environ.get("DBR_POLL_SECONDS", "5"))

BASELINE_SAMPLES = int(os.environ.get("DBR_BASELINE_SAMPLES", "5"))
HIGH_EDA_RATIO = float(os.environ.get("DBR_HIGH_EDA_RATIO", "1.5"))
MEDIUM_EDA_RATIO = float(os.environ.get("DBR_MEDIUM_EDA_RATIO", "1.2"))
TEMP_DROP_C = float(os.environ.get("DBR_TEMP_DROP_C", "0.3"))


def fetch_sample():
    resp = requests.get(API_URL, timeout=5)
    resp.raise_for_status()
    return resp.json()


def classify(eda, baseline_eda, temp, baseline_temp):
    if eda is None or baseline_eda in (None, 0):
        return "pending"

    ratio = eda / baseline_eda
    temp_dropping = (
        temp is not None and baseline_temp is not None and (baseline_temp - temp) >= TEMP_DROP_C
    )

    if ratio >= HIGH_EDA_RATIO or (ratio >= MEDIUM_EDA_RATIO and temp_dropping):
        return "high"
    if ratio >= MEDIUM_EDA_RATIO:
        return "medium"
    return "low"


def main():
    eda_history = deque(maxlen=BASELINE_SAMPLES)
    temp_history = deque(maxlen=BASELINE_SAMPLES)
    baseline_eda = None
    baseline_temp = None
    last_updated_at = None

    print(
        "[decision_based_rule] PLACEHOLDER rule engine -- thresholds are NOT "
        "from the instructor's spec (it has no numeric cutoffs yet). Classifies "
        f"on EDA-ratio-to-personal-baseline + skin-temp drop (baseline = first "
        f"{BASELINE_SAMPLES} readings for this participant)."
    )
    print(f"Watching {API_URL} every {POLL_SECONDS}s...")

    while True:
        try:
            sample = fetch_sample()
        except Exception as e:
            print(f"[decision_based_rule] fetch error: {e}")
            time.sleep(POLL_SECONDS)
            continue

        updated_at = sample.get("updated_at")
        if updated_at and updated_at == last_updated_at:
            time.sleep(POLL_SECONDS)
            continue
        last_updated_at = updated_at

        eda = sample.get("eda")
        temp = sample.get("temp")
        participant_id = sample.get("participant_id")

        if eda is not None:
            eda_history.append(eda)
        if temp is not None:
            temp_history.append(temp)

        if baseline_eda is None and len(eda_history) == BASELINE_SAMPLES:
            baseline_eda = sum(eda_history) / len(eda_history)
        if baseline_temp is None and len(temp_history) == BASELINE_SAMPLES:
            baseline_temp = sum(temp_history) / len(temp_history)

        if baseline_eda is None:
            print(
                f"[decision_based_rule] participant={participant_id} building baseline "
                f"({len(eda_history)}/{BASELINE_SAMPLES} samples)... eda={eda} temp={temp}"
            )
        else:
            label = classify(eda, baseline_eda, temp, baseline_temp)
            pulse = sample.get("pulse_rate")
            print(
                f"[DBR] {updated_at} participant={participant_id} "
                f"eda={eda} temp={temp} pulse={pulse} "
                f"-> prediction: {label.upper()} "
                f"(eda_baseline={baseline_eda:.3f}, temp_baseline={baseline_temp})"
            )

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
