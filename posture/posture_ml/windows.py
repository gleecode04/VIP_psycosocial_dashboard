"""Reading-level labeling and windowing for the posture CSV.

Pipeline stages (see "The Pathway - Posture Machine Learning"):
  1. load the cleaned readings
  2. label each reading Good/Bad/Critical from Y_Pitch:
       - collapsed to a binary reading_bad flag (Bad + Critical = 1), the
         pathway doc's own target, kept as-is for the validated binary model
       - also kept as a 3-level reading_risk code (0=low/Good, 1=medium/Bad,
         2=high/Critical) for the course-wide Low/Medium/High risk
         requirement every team's pathway must satisfy
  3. cut each (Participant, Condition, Segment) recording into non-overlapping
     windows of WINDOW_SIZE_READINGS readings
  4. compute per-window features, a window-level bad/good label (majority
     vote of reading_bad across the window), and a window-level risk level
     (window_risk = the *worst* reading_risk in the window -- see note below)
  5. build the forecasting targets:
       - Next_Posture_bad = next window's window_bad (binary)
       - Next_Posture_risk = next window's window_risk (low/medium/high)

Window size: the pathway doc specifies 2-second windows (PATHWAY_WINDOW_SIZE_READINGS,
20 readings at the ~10Hz sampling rate -- reproduces its 144-window count exactly). We
default to 1-second windows (WINDOW_SIZE_READINGS, 10 readings) instead, because
evaluation showed the 2-second forecast horizon throws away most of the predictable
signal: persistence-baseline balanced accuracy was 0.55 at 2s vs. 0.69 at 1s, and the
best LOPO-evaluated model went from 0.68 (gap 0.14) to 0.86 (gap 0.08) -- a smaller
overfitting gap *and* much higher accuracy, consistently across all 6 participants, not
just one lucky fold. Posture state at a 2-second horizon is close to uncorrelated with
its current state for several participants in this dataset; at 1 second it isn't.

Why window_risk takes the WORST reading, not a majority vote: window_bad above uses a
>50% majority vote (matches the pathway doc's own binary rule), but a majority vote
would dilute a brief, genuinely dangerous angle spike (a single >45 deg "Critical"
reading inside an otherwise-fine 1-second window) down to "low". For a worker-safety
signal that's the wrong failure mode -- a transient critical spike is exactly the kind
of event the risk level should flag, not average away. Max-severity aggregation also
evaluates far better for this 3-class target: a majority-vote window label puts every
"high" window inside just 2 of the 6 participants (useless for Leave-One-Participant-Out
testing on 4 of 6 folds), while max-severity spreads "high" windows across 4 of the 6
participants -- see train_posture_risk_model.py / the technical write-up for details.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

READING_BAD_THRESHOLD_DEG = 20.0
READING_CRITICAL_THRESHOLD_DEG = 45.0
PATHWAY_WINDOW_SIZE_READINGS = 20  # 2s, per the doc spec -- kept for comparison/regression tests
WINDOW_SIZE_READINGS = 10  # 1s -- the default used for the saved model (see module docstring)
WINDOW_BAD_FRACTION_THRESHOLD = 0.5

# Ordinal 3-level risk coding shared by reading_risk and window_risk: index
# into RISK_LEVELS with the int code to get the course-wide "low"/"medium"/
# "high" label string (every pathway -- posture, frustration, fatigue -- uses
# this same vocabulary; see posture_feedback.py and decision_based_rule.py).
RISK_LEVELS = ("low", "medium", "high")
RISK_LOW, RISK_MEDIUM, RISK_HIGH = 0, 1, 2

GROUP_COLUMNS = ["Participant", "Condition", "Segment"]
# Same grouping, but the lowercase names build_windows() emits per window row.
WINDOW_GROUP_COLUMNS = ["participant", "condition", "segment"]

# Feature columns the pathway doc specifies per window: mean, min, max,
# spread, slope, movement speed, condition.
PATHWAY_FEATURES = ["mean", "min", "max", "spread", "slope", "movement_speed", "condition_with"]

# A few extra, equally leakage-free engineered features tried alongside the
# pathway's minimal set (see model_specs.py for how these are combined).
EXTENDED_FEATURES = PATHWAY_FEATURES + ["std", "median", "bad_fraction_now"]


def reading_severity(angle: float) -> int:
    """DBR thresholds on a single raw angle -> RISK_LOW/MEDIUM/HIGH.

    <=20 deg Good (low), 20-45 deg Bad (medium), >45 deg Critical (high) --
    exactly the instructor's reference thresholds (see the pathway doc and
    load_readings' reading_bad, which collapses medium+high into one flag).
    """
    if angle > READING_CRITICAL_THRESHOLD_DEG:
        return RISK_HIGH
    if angle > READING_BAD_THRESHOLD_DEG:
        return RISK_MEDIUM
    return RISK_LOW


def load_readings(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned posture CSV and attach the per-reading bad flag."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(GROUP_COLUMNS + ["Timestamp"]).reset_index(drop=True)
    df["reading_bad"] = (df["Y_Pitch"] > READING_BAD_THRESHOLD_DEG).astype(int)
    df["reading_risk"] = df["Y_Pitch"].apply(reading_severity)
    return df


def _window_slope(timestamps: np.ndarray, angles: np.ndarray) -> float:
    """Degrees-per-second trend within the window (0 if readings are simultaneous)."""
    t_rel = timestamps - timestamps[0]
    if np.ptp(t_rel) == 0:
        return 0.0
    return float(np.polyfit(t_rel, angles, 1)[0])


def _window_movement_speed(timestamps: np.ndarray, angles: np.ndarray) -> float:
    """Mean absolute degrees-per-second change between consecutive readings."""
    dt = np.diff(timestamps)
    dy = np.diff(angles)
    valid = dt > 0
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(dy[valid]) / dt[valid]))


def _featurize_window(chunk: pd.DataFrame) -> dict:
    angles = chunk["Y_Pitch"].to_numpy()
    timestamps = chunk["Timestamp"].to_numpy()
    bad_fraction = float(chunk["reading_bad"].mean())
    # Computed from angles directly (not a precomputed reading_risk column)
    # so this also works on hand-built windows in tests that never called
    # load_readings().
    window_risk = max(reading_severity(a) for a in angles)
    return {
        "n_readings": len(chunk),
        "mean": float(angles.mean()),
        "min": float(angles.min()),
        "max": float(angles.max()),
        "spread": float(angles.max() - angles.min()),
        "std": float(angles.std(ddof=0)),
        "median": float(np.median(angles)),
        "slope": _window_slope(timestamps, angles),
        "movement_speed": _window_movement_speed(timestamps, angles),
        "bad_fraction_now": bad_fraction,
        "window_bad": int(bad_fraction > WINDOW_BAD_FRACTION_THRESHOLD),
        "window_risk": window_risk,
    }


def build_windows(readings: pd.DataFrame, window_size_readings: int = WINDOW_SIZE_READINGS) -> pd.DataFrame:
    """Cut labeled readings into non-overlapping windows with per-window features.

    Each (Participant, Condition, Segment) recording is windowed independently
    so a window never mixes readings from two different recordings. Any
    trailing readings that don't fill a full window are dropped. Pass
    window_size_readings=PATHWAY_WINDOW_SIZE_READINGS to reproduce the doc's
    144-window, 2-second-block count exactly.
    """
    rows = []
    for (participant, condition, segment), group in readings.groupby(GROUP_COLUMNS, sort=False):
        group = group.sort_values("Timestamp").reset_index(drop=True)
        n_windows = len(group) // window_size_readings
        for window_index in range(n_windows):
            start = window_index * window_size_readings
            chunk = group.iloc[start : start + window_size_readings]
            row = {
                "participant": participant,
                "condition": condition,
                "segment": segment,
                "window_index": window_index,
                "condition_with": int(condition == "with"),
            }
            row.update(_featurize_window(chunk))
            rows.append(row)

    windows = pd.DataFrame(rows)
    return windows.sort_values(["participant", "condition", "segment", "window_index"]).reset_index(
        drop=True
    )


def add_forecast_target(windows: pd.DataFrame) -> pd.DataFrame:
    """Add forecast targets for the next window, within the same recording:

      - Next_Posture_bad  = next window's window_bad  (binary, pathway doc's target)
      - Next_Posture_risk = next window's window_risk (low/medium/high, added
        only when the input already has a window_risk column -- callers that
        hand-build a windows frame without one, e.g. some unit tests, get the
        same binary-only behavior as before)

    The last window of every (participant, condition, segment) recording has
    no "next" window and is dropped -- it cannot supply a label.
    """
    windows = windows.copy()
    grouped = windows.groupby(WINDOW_GROUP_COLUMNS)
    windows["Next_Posture_bad"] = grouped["window_bad"].shift(-1)
    subset = ["Next_Posture_bad"]

    has_risk = "window_risk" in windows.columns
    if has_risk:
        windows["Next_Posture_risk"] = grouped["window_risk"].shift(-1)
        subset.append("Next_Posture_risk")

    labeled = windows.dropna(subset=subset).copy()
    labeled["Next_Posture_bad"] = labeled["Next_Posture_bad"].astype(int)
    if has_risk:
        labeled["Next_Posture_risk"] = labeled["Next_Posture_risk"].astype(int)
    return labeled.reset_index(drop=True)


def build_dataset(csv_path: str | Path, window_size_readings: int = WINDOW_SIZE_READINGS) -> pd.DataFrame:
    """Full pipeline: CSV -> labeled, windowed, forecast-target dataset."""
    readings = load_readings(csv_path)
    windows = build_windows(readings, window_size_readings=window_size_readings)
    return add_forecast_target(windows)
