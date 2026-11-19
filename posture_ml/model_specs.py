"""Feature sets and candidate model pipelines evaluated for posture forecasting.

Every entry is deliberately regularized / shallow: with 133 windows pooled
across 6 participants and a Leave-One-Participant-Out evaluation, a model
with much flexibility will memorize individual participants' posture habits
instead of learning a pattern that transfers to someone new.
"""
from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from posture_ml.windows import EXTENDED_FEATURES, PATHWAY_FEATURES

RANDOM_STATE = 42

FEATURE_SETS: dict[str, list[str]] = {
    "pathway": PATHWAY_FEATURES,
    "extended": EXTENDED_FEATURES,
}


def _build_logreg(c: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )


def _build_random_forest() -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def _build_gradient_boosting() -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def _build_svm() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )


# name -> zero-argument factory, so every CV fold gets a freshly initialized
# (unfit) pipeline instead of a shared, partially-fit one.
MODEL_BUILDERS: dict[str, Callable[[], Pipeline]] = {
    "logreg_c1.0": lambda: _build_logreg(1.0),
    "logreg_c0.3": lambda: _build_logreg(0.3),
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
    "svm_rbf": _build_svm,
}
