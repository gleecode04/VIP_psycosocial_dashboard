#!/usr/bin/env python3
"""Closed-loop retraining -- Closed loop Guide.md steps 9 ("Retrain") and
10 ("Deploy safely").

Builds a CANDIDATE frustration/mental_demand model from the original study
windows (the same labeled CSVs train_biomarker_models.py was trained from --
see build_frustration_dataset()/build_mental_demand_dataset(), reused here
rather than reimplemented) plus every answered check-in collected under
live_output/ (checkin_bridge.py's storage), with each answer repeated
--weight times so it counts as that many study windows, exactly as the
guide's design table describes.

Evaluates the candidate two ways before touching anything live:

  1. Leave-one-WORKER-out over the check-in answers: for each worker who
     answered at least one question, retrain on everyone else's answers +
     all study data, then predict that worker's own answers. This is the
     "does it generalize to a worker it didn't memorize" check.
  2. Leave-one-PARTICIPANT-out over the ORIGINAL study data (P1, P2, ...):
     for each study participant, retrain on every other participant +
     ALL of the check-in data, then predict the held-out participant. This
     is the "did adding check-ins hurt the original study accuracy" check.

Both numbers are also computed for whatever model is CURRENTLY live in
models/, using the identical methodology (not the differently-computed
GroupKFold numbers in model_metadata.json), so the before/after comparison
is apples-to-apples.

--apply only switches the live model if the candidate is at least as good
on (1) and loses no more than --max-drop (default 0.02) balanced accuracy
on (2); --force bypasses both that gate and the --min answers-required gate.
The replaced model is backed up next to itself, never deleted; restart
avro_stream_poller_with_ml.py / mock_poller.py to pick up the new one (the
live biomarker_runtime.LiveBiomarkerPredictor loads it once at process
start, there's no hot-reload).

One honest limitation, stated plainly rather than hidden: a check-in answer
is a Yes/A little/No confirmation of a High call, which only ever yields a
two-way signal (high vs. not-high) -- there is no way to recover whether a
"No" really meant Low or Medium from that answer alone. Every check-in-
derived training row is therefore labeled "high" (confirmed) or "low"
(false alarm, the majority/default class), never "medium". See
_checkins_to_training_rows()'s docstring.

Usage:
    python retrain_from_checkins.py                  # report only, no changes
    python retrain_from_checkins.py --apply           # apply if the safety gate passes
    python retrain_from_checkins.py --apply --force   # apply regardless of the gate
    python retrain_from_checkins.py --lane frustration --weight 5 --min 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import checkin

# This script lives in dashboard/ (it's the closed loop's retrain step,
# reading the check-ins checkin_bridge.py stores under dashboard/live_output/)
# but the frustration/mental_demand models and their training code live in
# ../biomarkers/, which is otherwise a self-contained pathway folder. This
# is the one deliberate exception to that separation -- see the repo README.
ROOT = Path(__file__).resolve().parent
BIOMARKERS_DIR = ROOT.parent / "biomarkers"
if str(BIOMARKERS_DIR) not in sys.path:
    sys.path.insert(0, str(BIOMARKERS_DIR))

from train_biomarker_models import build_frustration_dataset, build_mental_demand_dataset

MODELS_DIR = BIOMARKERS_DIR / "models"
LIVE_OUTPUT_DIR = ROOT / "live_output"

DEFAULT_STUDY_DATA = os.environ.get(
    "CHECKIN_STUDY_DATA", str(Path.home() / "Desktop" / "Final Files to Use for the Script")
)

PARTICIPANT_RE = re.compile(r"^[Pp](\d+)")

LANES = {
    "frustration": {
        "model_file": "frustration_model.joblib",
        "metadata_key": "frustration",
        "build_study_df": build_frustration_dataset,
        "study_subdir": "Frustration",
        "answer_field": "stressed",
    },
    "mental_demand": {
        "model_file": "mental_demand_model.joblib",
        "metadata_key": "mental_demand",
        "build_study_df": build_mental_demand_dataset,
        "study_subdir": "Mental Demand",
        "answer_field": "mental",
    },
}


def participant_from_filename(filename: str) -> str:
    m = PARTICIPANT_RE.match(filename)
    return f"P{m.group(1)}" if m else filename


def make_model(lane: str) -> Pipeline:
    """Identical pipeline shape to train_biomarker_models.py -- a retrained
    model must stay a drop-in replacement for biomarker_runtime.py."""
    if lane == "frustration":
        estimator = LogisticRegression(max_iter=5000, class_weight="balanced")
    else:
        estimator = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _checkins_to_training_rows(lane: str, little_counts_as: str) -> pd.DataFrame:
    """Every answered (non-expired) check-in recorded under live_output/,
    turned into one training row for `lane`.

    Label mapping: derived answer True -> "high" (the worker confirmed the
    High call), False -> "low" (a false alarm -- "low" rather than "medium"
    because a boolean confirm/deny can't distinguish those two; "low" is
    both the majority class in the original data and the more conservative
    read of "this wasn't actually high"). Self-reports are included too --
    they're real worker-labeled moments even though no prediction triggered
    them, and the guide doesn't say to exclude them from training, only
    from the asked/answered agreement stats (see checkin_bridge.py).
    """
    answer_field = LANES[lane]["answer_field"]
    feature_keys = (
        ["eda_t_minus_1", "temp_t_minus_1", "eda_t_0", "temp_t_0", "eda_delta_t_0", "temp_delta_t_0", "minute_index"]
        if lane == "frustration" else ["eda", "temp", "minute_index"]
    )
    rows = []
    if not LIVE_OUTPUT_DIR.exists():
        return pd.DataFrame(rows, columns=[*feature_keys, "label", "worker", "question_id"])

    for participant_dir in sorted(LIVE_OUTPUT_DIR.iterdir()):
        json_path = participant_dir / "checkins.json"
        if not json_path.exists():
            continue
        try:
            records = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in records:
            raw_answer = record.get(f"{answer_field}_raw_answer")
            derived = checkin.derive_answer(raw_answer, little_counts_as=little_counts_as)
            if derived is None:
                continue
            features = record.get("features") or {}
            if any(features.get(k) is None for k in feature_keys):
                continue  # incomplete snapshot (e.g. first reading of a session) -- skip rather than impute a fabricated row
            row = {k: features[k] for k in feature_keys}
            row["label"] = "high" if derived else "low"
            row["worker"] = record["participant_id"]
            row["question_id"] = record["question_id"]
            rows.append(row)
    return pd.DataFrame(rows, columns=[*feature_keys, "label", "worker", "question_id"])


def _repeat(df: pd.DataFrame, weight: int) -> pd.DataFrame:
    return pd.concat([df] * weight, ignore_index=True) if weight > 1 else df.reset_index(drop=True)


def _fit(lane: str, study_X, study_y, checkin_X=None, checkin_y=None, weight: int = 1) -> Pipeline:
    X_parts = [study_X.reset_index(drop=True)]
    y_parts = [study_y.reset_index(drop=True)]
    if checkin_X is not None and len(checkin_X):
        X_parts.append(_repeat(checkin_X, weight))
        y_parts.append(_repeat(checkin_y.to_frame("label"), weight)["label"])
    model = make_model(lane)
    model.fit(pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True))
    return model


def evaluate_on_answers(lane: str, feature_cols: list[str], study_X, study_y, checkin_df: pd.DataFrame,
                         weight: int, baseline_model: Optional[Pipeline]) -> dict:
    """Step 9's check (1): leave-one-worker-out for the candidate; the
    already-trained baseline model is just scored directly (it never saw
    any check-in data, so no leakage risk to hold out against)."""
    if checkin_df.empty:
        return {"baseline": None, "candidate": None, "n_answers": 0, "n_workers": 0}

    workers = sorted(checkin_df["worker"].unique())
    y_true, y_pred_candidate = [], []
    for worker in workers:
        train_df = checkin_df[checkin_df["worker"] != worker]
        held_out = checkin_df[checkin_df["worker"] == worker]
        model = _fit(lane, study_X, study_y, train_df[feature_cols], train_df["label"], weight)
        y_true.extend(held_out["label"])
        y_pred_candidate.extend(model.predict(held_out[feature_cols]))

    result = {
        "candidate": round(balanced_accuracy_score(y_true, y_pred_candidate), 4),
        "n_answers": len(checkin_df), "n_workers": len(workers),
    }
    if baseline_model is not None:
        y_pred_baseline = baseline_model.predict(checkin_df[feature_cols])
        result["baseline"] = round(balanced_accuracy_score(checkin_df["label"], y_pred_baseline), 4)
    else:
        result["baseline"] = None
    return result


def evaluate_study_lopo(lane: str, feature_cols: list[str], study_df: pd.DataFrame, checkin_df: pd.DataFrame,
                         weight: int, with_checkins: bool) -> float:
    """Step 9's check (2): standard leave-one-participant-out on the
    original study data. `with_checkins` controls whether every training
    fold also gets the (weighted) check-in rows -- False reproduces the
    "baseline, study data alone" number; True is the candidate's number."""
    participants = study_df["file"].map(participant_from_filename)
    y_true, y_pred = [], []
    for p in sorted(participants.unique()):
        train_mask, test_mask = participants != p, participants == p
        checkin_X = checkin_df[feature_cols] if (with_checkins and not checkin_df.empty) else None
        checkin_y = checkin_df["label"] if (with_checkins and not checkin_df.empty) else None
        model = _fit(lane, study_df.loc[train_mask, feature_cols], study_df.loc[train_mask, "label"],
                     checkin_X, checkin_y, weight)
        y_true.extend(study_df.loc[test_mask, "label"])
        y_pred.extend(model.predict(study_df.loc[test_mask, feature_cols]))
    return round(balanced_accuracy_score(y_true, y_pred), 4)


def run_lane(lane: str, args) -> dict:
    cfg = LANES[lane]
    study_dir = Path(args.study_data) / cfg["study_subdir"]
    if not study_dir.exists():
        return {"lane": lane, "skipped": f"study data not found at {study_dir}"}

    study_df, used_files = cfg["build_study_df"](study_dir)
    feature_cols = [c for c in study_df.columns if c not in ("file", "label")]

    checkin_df = _checkins_to_training_rows(lane, args.little)
    n_high = (checkin_df["label"] == "high").sum() if not checkin_df.empty else 0
    n_low = (checkin_df["label"] == "low").sum() if not checkin_df.empty else 0
    has_both_classes = n_high > 0 and n_low > 0

    report = {
        "lane": lane, "n_study_sessions": len(used_files), "n_answers": len(checkin_df),
        "n_answers_high": int(n_high), "n_answers_low": int(n_low),
    }

    if not args.force and (len(checkin_df) < args.min or not has_both_classes):
        report["skipped"] = (
            f"only {len(checkin_df)} usable answer(s) (need >= {args.min}, with both a 'high' and a 'low'); "
            "retraining skipped -- pass --force to override."
        )
        return report

    model_path = MODELS_DIR / cfg["model_file"]
    baseline_model = joblib.load(model_path) if model_path.exists() else None

    answers_eval = evaluate_on_answers(lane, feature_cols, study_df[feature_cols], study_df["label"],
                                        checkin_df, args.weight, baseline_model)
    baseline_study_lopo = evaluate_study_lopo(lane, feature_cols, study_df, checkin_df, args.weight, with_checkins=False)
    candidate_study_lopo = evaluate_study_lopo(lane, feature_cols, study_df, checkin_df, args.weight, with_checkins=True)

    report.update({
        "answers_eval": answers_eval,
        "baseline_study_lopo_balanced_accuracy": baseline_study_lopo,
        "candidate_study_lopo_balanced_accuracy": candidate_study_lopo,
        "study_lopo_drop": round(baseline_study_lopo - candidate_study_lopo, 4),
    })

    answers_ok = answers_eval["baseline"] is None or answers_eval["candidate"] >= answers_eval["baseline"]
    study_ok = report["study_lopo_drop"] <= args.max_drop
    report["safe_to_deploy"] = bool(answers_ok and study_ok)
    report["verdict"] = (
        "SAFE to deploy" if report["safe_to_deploy"]
        else f"NOT safe (answers_ok={answers_ok}, study_ok={study_ok}, drop={report['study_lopo_drop']} > {args.max_drop})"
    )

    if args.apply and (report["safe_to_deploy"] or args.force):
        candidate = _fit(lane, study_df[feature_cols], study_df["label"], checkin_df[feature_cols],
                          checkin_df["label"], args.weight)
        if model_path.exists():
            backup_path = model_path.with_name(f"{model_path.stem}.{datetime.now():%Y%m%d%H%M%S}.bak.joblib")
            shutil.copy2(model_path, backup_path)
            report["backup"] = str(backup_path)
        joblib.dump(candidate, model_path)
        report["deployed"] = True
        report["note"] = "Restart avro_stream_poller_with_ml.py / mock_poller.py to load the new model."
    else:
        report["deployed"] = False

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-data", default=DEFAULT_STUDY_DATA,
                         help="Folder containing Frustration/ and Mental Demand/ subfolders of labeled CSVs.")
    parser.add_argument("--lane", choices=["frustration", "mental_demand", "both"], default="both")
    parser.add_argument("--little", choices=["yes", "no"], default=checkin.LITTLE_COUNTS_AS,
                         help="Whether 'A little' counts as a confirmed high reading.")
    parser.add_argument("--weight", type=int, default=checkin.ANSWER_WEIGHT,
                         help="How many study windows one answer counts as.")
    parser.add_argument("--min", type=int, default=checkin.MIN_ANSWERS_TO_RETRAIN,
                         help="Minimum answered check-ins (with both a 'high' and a 'low') before retraining.")
    parser.add_argument("--max-drop", type=float, default=checkin.MAX_BALANCED_ACCURACY_DROP,
                         help="Largest acceptable balanced-accuracy drop on the study LOPO check.")
    parser.add_argument("--apply", action="store_true", help="Deploy the candidate if it passes the safety gate.")
    parser.add_argument("--force", action="store_true",
                         help="Bypass the --min answers gate and the safety gate (use with care).")
    args = parser.parse_args()

    lanes = list(LANES) if args.lane == "both" else [args.lane]
    reports = [run_lane(lane, args) for lane in lanes]
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
