#!/usr/bin/env python3
"""Train and evaluate the Low/Medium/High posture RISK forecasting model.

Sibling script to train_posture_model.py, same pathway (clean -> label(DBR)
-> window -> featurize -> forecast target -> train candidates -> LOPO ->
keep the best -> refit on all data -> save), but for the 3-class risk level
target (Next_Posture_risk) the assignment requires every team's pathway to
predict, instead of train_posture_model.py's binary Next_Posture_bad.

Both targets come out of the same windowed dataset (posture_ml/windows.py)
and the binary model stays as-is -- this script doesn't replace it, it adds
the risk-level forecaster the feedback trigger (posture_feedback.py) and the
live demo (posture_live_replay.py) actually consume.

window_risk/Next_Posture_risk use MAX-severity aggregation within a window
(the worst reading wins), not the binary target's >50%-majority-vote rule --
see posture_ml/windows.py's module docstring for why, and this script's
printed report for how that changes which participants contribute "high"
examples to each Leave-One-Participant-Out fold.

Usage:
    python train_posture_risk_model.py \
        --csv "data/posture_cleaned.csv" \
        --outdir models
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib

# Expected and harmless: some LOPO folds hold out a participant with zero
# "high" examples in the *test* fold, so sklearn's precision/recall/f1 see a
# predicted class absent from y_true for that fold. zero_division=0 (passed
# throughout posture_ml/evaluate.py) already handles the resulting
# division-by-zero correctly; this only silences the duplicate warning text.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from posture_ml.evaluate import (
    EvalResult,
    best_result,
    majority_class_baseline,
    persistence_baseline,
    run_lopo,
    summarize,
)
from posture_ml.model_specs import FEATURE_SETS, MODEL_BUILDERS
from posture_ml.windows import RISK_LEVELS, WINDOW_SIZE_READINGS, build_dataset

TARGET_COLUMN = "Next_Posture_risk"
CURRENT_COLUMN = "window_risk"
LABELS = tuple(range(len(RISK_LEVELS)))  # (0, 1, 2) == (low, medium, high)


def evaluate_all(df) -> list[EvalResult]:
    results: list[EvalResult] = []

    results.append(
        summarize(
            "baseline_majority_class",
            "-",
            "majority_class",
            majority_class_baseline(df, target_column=TARGET_COLUMN),
            labels=LABELS,
        )
    )
    results.append(
        summarize(
            "baseline_persistence",
            "-",
            "persistence",
            persistence_baseline(df, target_column=TARGET_COLUMN, current_column=CURRENT_COLUMN),
            labels=LABELS,
        )
    )

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        for model_name, model_builder in MODEL_BUILDERS.items():
            folds = run_lopo(df, feature_columns, model_builder, target_column=TARGET_COLUMN)
            name = f"{feature_set_name}__{model_name}"
            results.append(summarize(name, feature_set_name, model_name, folds, labels=LABELS))

    return results


def _fmt_confusion(cm: list[list[int]]) -> str:
    header = "            " + "  ".join(f"pred_{lvl:<6s}" for lvl in RISK_LEVELS)
    lines = [header]
    for true_lvl, row in zip(RISK_LEVELS, cm):
        lines.append(f"true_{true_lvl:<6s} " + "  ".join(f"{v:10d}" for v in row))
    return "\n".join(lines)


def print_report(results: list[EvalResult], winner: EvalResult) -> None:
    header = f"{'model':32s} {'bal_acc':>8s} {'f1_macro':>9s} {'prec':>6s} {'recall':>6s} {'acc':>6s} {'gap':>6s}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.pooled_metrics["balanced_accuracy"], reverse=True):
        m = r.pooled_metrics
        gap_str = f"{r.generalization_gap:6.3f}" if r.generalization_gap is not None else "   n/a"
        marker = " <== selected" if r is winner else ""
        print(
            f"{r.name:32s} {m['balanced_accuracy']:8.3f} {m['f1']:9.3f} "
            f"{m['precision']:6.3f} {m['recall']:6.3f} {m['accuracy']:6.3f} {gap_str}{marker}"
        )
    print()
    print("bal_acc/f1/precision/recall are macro-averaged across low/medium/high (each class")
    print("weighted equally, regardless of how rare 'high' is in this dataset)")
    print("gap = mean LOPO train-fold balanced accuracy minus mean test-fold balanced accuracy")
    print()
    print(f"Selected model: {winner.name}")
    print("Pooled confusion matrix (rows=true, cols=pred):")
    print(_fmt_confusion(winner.pooled_metrics["confusion_matrix"]))
    print()
    print("Class distribution (share of true labels):")
    for lvl, frac in zip(RISK_LEVELS, winner.pooled_metrics["class_distribution"].values()):
        print(f"  {lvl}: {frac:.3f}")
    print("Per-held-out-participant balanced accuracy:")
    for participant, ba in sorted(winner.per_fold_balanced_accuracy.items()):
        print(f"  participant {participant}: {ba:.3f}")
    if winner.generalization_gap is not None:
        print(f"Train-vs-test generalization gap: {winner.generalization_gap:.3f}")


def build_metadata(
    df, results: list[EvalResult], winner: EvalResult, feature_columns: list[str], window_size_readings: int
) -> dict:
    window_seconds = window_size_readings / 10  # ~10Hz sampling
    return {
        "pathway": (
            "clean -> label(DBR, 3-level) -> window(max-severity) -> features -> "
            "Next_Posture_risk -> train -> LOPO test -> evaluate"
        ),
        "risk_levels": list(RISK_LEVELS),
        "window_aggregation": (
            "window_risk = the WORST reading_risk in the window (max aggregation), not a "
            "majority vote -- see posture_ml/windows.py module docstring for why."
        ),
        "dataset": {
            "window_size_readings": window_size_readings,
            "window_size_seconds": window_seconds,
            "n_windows_with_target": len(df),
            "participants": sorted(int(p) for p in df["participant"].unique()),
            "class_distribution": {
                lvl: float((df[TARGET_COLUMN] == code).mean()) for code, lvl in enumerate(RISK_LEVELS)
            },
        },
        "selected_model": {
            "name": winner.name,
            "feature_set": winner.feature_set,
            "model_name": winner.model_name,
            "features": feature_columns,
            "lopo_metrics": winner.pooled_metrics,
            "lopo_per_participant_balanced_accuracy": {
                str(k): v for k, v in winner.per_fold_balanced_accuracy.items()
            },
            "train_vs_test_generalization_gap": winner.generalization_gap,
        },
        "all_candidates": [
            {
                "name": r.name,
                "feature_set": r.feature_set,
                "model_name": r.model_name,
                "lopo_metrics": r.pooled_metrics,
                "lopo_per_participant_balanced_accuracy": {
                    str(k): v for k, v in r.per_fold_balanced_accuracy.items()
                },
                "train_vs_test_generalization_gap": r.generalization_gap,
            }
            for r in results
        ],
        "notes": [
            (
                f"Next_Posture_risk forecasts the NEXT {window_seconds:g}-second window's risk "
                "level from the CURRENT window's features; the last window of every "
                "(participant, condition, segment) recording has no next window and is dropped."
            ),
            "Evaluation is Leave-One-Participant-Out: every reported metric is on a participant the model never trained on.",
            (
                "'high' (Critical, >45 deg) is rare and concentrated in a few participants -- "
                "treat the per-class and per-participant breakdowns as directional, not "
                "production-grade precision, and re-evaluate as more participants are collected."
            ),
            (
                "Feedback is NOT fired directly off this model's per-tick output -- see "
                "posture_feedback.py: only 3 consecutive 'high' predictions trigger feedback, "
                "and any 'low' or 'medium' prediction resets the streak, per the assignment's "
                "course-wide requirement."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/posture_cleaned.csv", help="Path to the cleaned posture CSV")
    parser.add_argument("--outdir", default="models", help="Output directory for the trained model + metadata")
    parser.add_argument(
        "--window-readings",
        type=int,
        default=WINDOW_SIZE_READINGS,
        help="Readings per window (10 ~= 1s at ~10Hz sampling)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(csv_path, window_size_readings=args.window_readings)
    results = evaluate_all(df)
    winner = best_result([r for r in results if r.feature_set != "-"])

    print_report(results, winner)

    feature_columns = FEATURE_SETS[winner.feature_set]
    final_model = MODEL_BUILDERS[winner.model_name]()
    final_model.fit(df[feature_columns], df[TARGET_COLUMN])

    model_path = outdir / "posture_risk_forecast_model.joblib"
    joblib.dump(final_model, model_path)

    metadata = build_metadata(df, results, winner, feature_columns, args.window_readings)
    metadata_path = outdir / "posture_risk_forecast_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
