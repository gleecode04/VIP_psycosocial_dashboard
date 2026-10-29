"""Leave-One-Participant-Out evaluation for posture-forecasting models.

Grouping by participant (not by window) is the whole point: a model that
merely memorizes "participant 7 slouches a lot" would ace a random split but
is useless for a person the dashboard has never seen. LOPO is the harshest
honest test available with only 6 participants, and it is the test the
pathway doc itself specifies (step 7).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TARGET_COLUMN = "Next_Posture_bad"
GROUP_COLUMN = "participant"


@dataclass(frozen=True)
class FoldResult:
    held_out_participant: int
    n_test: int
    y_true: np.ndarray
    y_pred: np.ndarray
    train_balanced_accuracy: float | None = None


@dataclass(frozen=True)
class EvalResult:
    name: str
    feature_set: str
    model_name: str
    fold_results: list[FoldResult] = field(repr=False)
    pooled_metrics: dict = field(default_factory=dict)
    per_fold_balanced_accuracy: dict = field(default_factory=dict)
    generalization_gap: float | None = None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[int, ...] = (0, 1)) -> dict:
    """Pooled metrics. Binary targets (labels=(0, 1)) use scikit-learn's binary
    averaging, exactly as before. A 3+-class labels tuple (e.g. the
    low/medium/high risk target) switches f1/precision/recall to macro
    averaging -- each class weighted equally regardless of how rare it is --
    and reports a confusion matrix sized to match instead of a 2x2.
    """
    labels = list(labels)
    multiclass = len(labels) > 2
    average = "macro" if multiclass else "binary"
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average=average, zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": len(y_true),
    }
    if multiclass:
        metrics["class_distribution"] = {
            str(label): float(np.mean(y_true == label)) for label in labels
        }
    else:
        metrics["positive_rate"] = float(np.mean(y_true))
    return metrics


def _pool_folds(folds: list[FoldResult]) -> EvalResult:
    y_true = np.concatenate([f.y_true for f in folds])
    y_pred = np.concatenate([f.y_pred for f in folds])
    per_fold_ba = {
        f.held_out_participant: float(balanced_accuracy_score(f.y_true, f.y_pred)) if len(set(f.y_true)) > 1
        else float(np.mean(f.y_true == f.y_pred))
        for f in folds
    }
    return y_true, y_pred, per_fold_ba


def run_lopo(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_builder: Callable[[], object],
    target_column: str = TARGET_COLUMN,
) -> list[FoldResult]:
    """Fit on all-but-one participant, predict the held-out one, for every participant.

    target_column defaults to the binary Next_Posture_bad target; pass
    "Next_Posture_risk" to evaluate the low/medium/high forecaster instead --
    sklearn's classifiers handle 3-class targets natively, so nothing else
    about the fit/predict loop changes.
    """
    folds = []
    for participant in sorted(df[GROUP_COLUMN].unique()):
        train = df[df[GROUP_COLUMN] != participant]
        test = df[df[GROUP_COLUMN] == participant]
        if test.empty:
            continue

        model = model_builder()
        model.fit(train[feature_columns], train[target_column])
        y_pred = model.predict(test[feature_columns])
        train_pred = model.predict(train[feature_columns])

        folds.append(
            FoldResult(
                held_out_participant=int(participant),
                n_test=len(test),
                y_true=test[target_column].to_numpy(),
                y_pred=np.asarray(y_pred),
                train_balanced_accuracy=float(
                    balanced_accuracy_score(train[target_column], train_pred)
                ),
            )
        )
    return folds


def majority_class_baseline(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[FoldResult]:
    """Per fold: predict the training fold's majority class for every test row."""
    folds = []
    for participant in sorted(df[GROUP_COLUMN].unique()):
        train = df[df[GROUP_COLUMN] != participant]
        test = df[df[GROUP_COLUMN] == participant]
        if test.empty:
            continue
        majority = int(train[target_column].mode().iloc[0])
        y_pred = np.full(len(test), majority)
        folds.append(
            FoldResult(
                held_out_participant=int(participant),
                n_test=len(test),
                y_true=test[target_column].to_numpy(),
                y_pred=y_pred,
            )
        )
    return folds


def persistence_baseline(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN, current_column: str = "window_bad"
) -> list[FoldResult]:
    """Predict "next window = same as current window" -- needs no training.

    current_column defaults to window_bad (paired with Next_Posture_bad);
    pass current_column="window_risk" alongside target_column="Next_Posture_risk"
    for the 3-class persistence baseline.
    """
    folds = []
    for participant in sorted(df[GROUP_COLUMN].unique()):
        test = df[df[GROUP_COLUMN] == participant]
        if test.empty:
            continue
        folds.append(
            FoldResult(
                held_out_participant=int(participant),
                n_test=len(test),
                y_true=test[target_column].to_numpy(),
                y_pred=test[current_column].to_numpy(),
            )
        )
    return folds


def summarize(
    name: str,
    feature_set: str,
    model_name: str,
    folds: list[FoldResult],
    labels: tuple[int, ...] = (0, 1),
) -> EvalResult:
    y_true, y_pred, per_fold_ba = _pool_folds(folds)

    train_scores = [f.train_balanced_accuracy for f in folds if f.train_balanced_accuracy is not None]
    gap = None
    if train_scores:
        mean_train = float(np.mean(train_scores))
        mean_test = float(np.mean(list(per_fold_ba.values())))
        gap = mean_train - mean_test

    return EvalResult(
        name=name,
        feature_set=feature_set,
        model_name=model_name,
        fold_results=folds,
        pooled_metrics=_metrics(y_true, y_pred, labels=labels),
        per_fold_balanced_accuracy=per_fold_ba,
        generalization_gap=gap,
    )


MAX_ACCEPTABLE_GENERALIZATION_GAP = 0.25


def rank_key(result: EvalResult) -> tuple:
    """Higher pooled balanced accuracy first, then higher F1, then lower fold variance."""
    ba_values = list(result.per_fold_balanced_accuracy.values())
    stability = -float(np.std(ba_values)) if ba_values else 0.0
    return (result.pooled_metrics["balanced_accuracy"], result.pooled_metrics["f1"], stability)


def best_result(results: list[EvalResult], max_gap: float = MAX_ACCEPTABLE_GENERALIZATION_GAP) -> EvalResult:
    """Pick the top candidate by rank_key, first excluding high train/test gaps.

    A model that fits its training participants much better than it predicts
    held-out ones is overfitting even if its held-out score looks fine --
    exclude those before ranking on accuracy. Falls back to the full pool if
    every candidate happens to exceed the gap threshold.
    """
    within_gap = [r for r in results if r.generalization_gap is None or r.generalization_gap <= max_gap]
    candidates = within_gap or results
    return max(candidates, key=rank_key)
