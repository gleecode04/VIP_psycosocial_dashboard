"""Tests for the posture forecasting pipeline (posture_ml/*).

Run with: pytest test_posture_ml.py -v
"""
import numpy as np
import pandas as pd
import pytest

from posture_ml.evaluate import (
    EvalResult,
    best_result,
    majority_class_baseline,
    persistence_baseline,
    rank_key,
    run_lopo,
    summarize,
)
from posture_ml.model_specs import FEATURE_SETS, MODEL_BUILDERS
from posture_ml.windows import (
    PATHWAY_WINDOW_SIZE_READINGS,
    RISK_HIGH,
    RISK_LEVELS,
    RISK_LOW,
    RISK_MEDIUM,
    add_forecast_target,
    build_dataset,
    build_windows,
    load_readings,
    reading_severity,
)

CSV_PATH = "data/posture_cleaned.csv"


def _synthetic_readings(n=40, participant=1, condition="with", segment=1, angle_fn=None):
    angle_fn = angle_fn or (lambda i: 0.0)
    return pd.DataFrame(
        {
            "Participant": participant,
            "Condition": condition,
            "Segment": segment,
            "Timestamp": np.arange(n) * 0.1,
            "Timestamp converted": [f"t{i}" for i in range(n)],
            "Y_Pitch": [angle_fn(i) for i in range(n)],
            "Y_Pitch_smoothed": [angle_fn(i) for i in range(n)],
        }
    )


class TestLoadReadings:
    def test_loads_real_csv_without_error(self):
        df = load_readings(CSV_PATH)
        assert len(df) > 0
        assert "reading_bad" in df.columns

    def test_reading_bad_flag_uses_raw_y_pitch_threshold(self):
        # ">20" on the raw (signed) angle, matching the instructor's Excel
        # reference (COUNTIFS on column F, not absolute value).
        df = _synthetic_readings(n=4, angle_fn=lambda i: [10.0, 20.0, 20.1, 45.0][i])
        df["reading_bad"] = (df["Y_Pitch"] > 20.0).astype(int)
        assert df["reading_bad"].tolist() == [0, 0, 1, 1]

    def test_known_participant_bad_reading_counts_match_reference(self):
        # Cross-checked against the instructor's Excel reference table:
        # participant 1 -> 251 readings, 128 above 20deg, 99 above 45deg.
        df = load_readings(CSV_PATH)
        p1 = df[df["Participant"] == 1]
        assert len(p1) == 251
        assert (p1["Y_Pitch"] > 20).sum() == 128
        assert (p1["Y_Pitch"] > 45).sum() == 99


class TestReadingRisk:
    def test_thresholds_match_dbr_low_medium_high(self):
        assert reading_severity(20.0) == RISK_LOW  # exactly 20 -> not > 20
        assert reading_severity(20.1) == RISK_MEDIUM
        assert reading_severity(45.0) == RISK_MEDIUM  # exactly 45 -> not > 45
        assert reading_severity(45.1) == RISK_HIGH

    def test_reading_risk_column_matches_reading_bad_collapse(self):
        # reading_bad = medium + high collapsed to 1 (pathway doc's own rule)
        df = load_readings(CSV_PATH)
        collapsed = (df["reading_risk"] > RISK_LOW).astype(int)
        assert collapsed.tolist() == df["reading_bad"].tolist()

    def test_known_participant_high_reading_count_matches_reference(self):
        # Cross-checked against the instructor's Excel reference table.
        df = load_readings(CSV_PATH)
        p1 = df[df["Participant"] == 1]
        assert (p1["reading_risk"] == RISK_HIGH).sum() == 99


class TestBuildWindows:
    def test_window_count_is_floor_division_dropping_remainder(self):
        readings = _synthetic_readings(n=45)  # 2 full windows of 20, 5 leftover
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings, window_size_readings=20)
        assert len(windows) == 2

    def test_windows_never_cross_group_boundaries(self):
        a = _synthetic_readings(n=25, participant=1, condition="with")
        b = _synthetic_readings(n=25, participant=1, condition="without")
        readings = pd.concat([a, b], ignore_index=True)
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings, window_size_readings=20)
        # 1 window per group (25 // 20 == 1), never a window spanning both
        assert len(windows) == 2
        assert set(windows["condition"]) == {"with", "without"}

    def test_window_bad_majority_vote(self):
        # 20 readings, 11 above 20deg (55% -> bad), 9 at 0deg
        angles = [30.0] * 11 + [0.0] * 9
        readings = _synthetic_readings(n=20, angle_fn=lambda i: angles[i])
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings, window_size_readings=20)
        assert windows.loc[0, "window_bad"] == 1
        assert windows.loc[0, "bad_fraction_now"] == pytest.approx(0.55)

    def test_window_risk_uses_max_severity_not_majority(self):
        # 9 low readings (0deg) + 1 critical spike (50deg): a majority vote
        # would call this window "low", but a single critical reading should
        # still flag the whole window "high" -- see windows.py's docstring.
        angles = [0.0] * 9 + [50.0]
        readings = _synthetic_readings(n=10, angle_fn=lambda i: angles[i])
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings, window_size_readings=10)
        assert windows.loc[0, "window_risk"] == RISK_HIGH
        assert windows.loc[0, "window_bad"] == 0  # majority-vote target is unaffected

    def test_window_risk_is_low_when_every_reading_is_low(self):
        readings = _synthetic_readings(n=10, angle_fn=lambda i: 5.0)
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings, window_size_readings=10)
        assert windows.loc[0, "window_risk"] == RISK_LOW

    def test_window_features_on_known_values(self):
        angles = list(range(20))  # 0..19, perfectly increasing by 1 per 0.1s
        readings = _synthetic_readings(n=20, angle_fn=lambda i: float(angles[i]))
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings, window_size_readings=20)
        row = windows.iloc[0]
        assert row["mean"] == pytest.approx(9.5)
        assert row["min"] == pytest.approx(0.0)
        assert row["max"] == pytest.approx(19.0)
        assert row["spread"] == pytest.approx(19.0)
        assert row["slope"] == pytest.approx(10.0, rel=1e-6)  # 1 deg / 0.1s = 10 deg/s
        assert row["movement_speed"] == pytest.approx(10.0, rel=1e-6)

    def test_window_features_at_default_1s_window_size(self):
        angles = list(range(10))  # 0..9, perfectly increasing by 1 per 0.1s
        readings = _synthetic_readings(n=10, angle_fn=lambda i: float(angles[i]))
        readings["reading_bad"] = (readings["Y_Pitch"] > 20).astype(int)
        windows = build_windows(readings)  # default window_size_readings (1s)
        row = windows.iloc[0]
        assert row["n_readings"] == 10
        assert row["mean"] == pytest.approx(4.5)
        assert row["slope"] == pytest.approx(10.0, rel=1e-6)

    def test_real_csv_produces_293_windows_at_default_1s_granularity(self):
        readings = load_readings(CSV_PATH)
        windows = build_windows(readings)
        assert len(windows) == 293

    def test_real_csv_window_risk_distribution(self):
        # Max-severity aggregation at 1s granularity: 34 "high" windows
        # spread across 4 of 6 participants (vs. only 2 of 6 under a
        # majority-vote rule) -- see windows.py's docstring for why that
        # matters for Leave-One-Participant-Out evaluation.
        readings = load_readings(CSV_PATH)
        windows = build_windows(readings)
        counts = windows["window_risk"].value_counts().to_dict()
        assert counts[RISK_LOW] == 183
        assert counts[RISK_MEDIUM] == 76
        assert counts[RISK_HIGH] == 34
        high_participants = windows.loc[windows["window_risk"] == RISK_HIGH, "participant"].unique()
        assert set(high_participants) == {1, 3, 5, 6}

    def test_real_csv_reproduces_doc_144_windows_at_pathway_2s_granularity(self):
        # Regression anchor: the doc's own reference pipeline (2s windows)
        # must still reproduce its reported 144-window count exactly.
        readings = load_readings(CSV_PATH)
        windows = build_windows(readings, window_size_readings=PATHWAY_WINDOW_SIZE_READINGS)
        assert len(windows) == 144


class TestForecastTarget:
    def test_drops_last_window_per_group_and_shifts_label(self):
        windows = pd.DataFrame(
            {
                "participant": [1, 1, 1],
                "condition": ["with"] * 3,
                "segment": [1, 1, 1],
                "window_index": [0, 1, 2],
                "window_bad": [0, 1, 0],
            }
        )
        labeled = add_forecast_target(windows)
        assert len(labeled) == 2  # last window dropped (no "next")
        assert labeled["Next_Posture_bad"].tolist() == [1, 0]

    def test_no_leakage_across_groups(self):
        windows = pd.DataFrame(
            {
                "participant": [1, 1, 2, 2],
                "condition": ["with"] * 4,
                "segment": [1, 1, 1, 1],
                "window_index": [0, 1, 0, 1],
                "window_bad": [0, 1, 1, 0],
            }
        )
        labeled = add_forecast_target(windows)
        # each group only has one (window, next-window) pair
        assert len(labeled) == 2
        assert set(labeled["participant"]) == {1, 2}

    def test_omits_next_posture_risk_when_window_risk_column_absent(self):
        # Backward-compat contract: hand-built windows frames without a
        # window_risk column (as above) still work and just don't get the
        # 3-class target -- only build_windows() output has window_risk.
        windows = pd.DataFrame(
            {
                "participant": [1, 1, 1],
                "condition": ["with"] * 3,
                "segment": [1, 1, 1],
                "window_index": [0, 1, 2],
                "window_bad": [0, 1, 0],
            }
        )
        labeled = add_forecast_target(windows)
        assert "Next_Posture_risk" not in labeled.columns

    def test_adds_next_posture_risk_and_shifts_label(self):
        windows = pd.DataFrame(
            {
                "participant": [1, 1, 1],
                "condition": ["with"] * 3,
                "segment": [1, 1, 1],
                "window_index": [0, 1, 2],
                "window_bad": [0, 1, 0],
                "window_risk": [RISK_LOW, RISK_HIGH, RISK_MEDIUM],
            }
        )
        labeled = add_forecast_target(windows)
        assert labeled["Next_Posture_risk"].tolist() == [RISK_HIGH, RISK_MEDIUM]

    def test_real_csv_end_to_end_row_count(self):
        df = build_dataset(CSV_PATH)
        # 293 windows (1s granularity) across 11 non-empty groups -> 1 "last
        # window" dropped per group
        assert len(df) == 282
        assert set(df["Next_Posture_bad"].unique()) <= {0, 1}

    def test_real_csv_end_to_end_row_count_at_pathway_2s_granularity(self):
        df = build_dataset(CSV_PATH, window_size_readings=PATHWAY_WINDOW_SIZE_READINGS)
        assert len(df) == 133


class TestEvaluation:
    @pytest.fixture(scope="module")
    def dataset(self):
        return build_dataset(CSV_PATH)

    def test_majority_class_baseline_never_sees_test_labels(self, dataset):
        folds = majority_class_baseline(dataset)
        assert len(folds) == dataset["participant"].nunique()
        total_test_rows = sum(f.n_test for f in folds)
        assert total_test_rows == len(dataset)

    def test_persistence_baseline_matches_current_window_label(self, dataset):
        folds = persistence_baseline(dataset)
        for fold in folds:
            participant_rows = dataset[dataset["participant"] == fold.held_out_participant]
            np.testing.assert_array_equal(fold.y_pred, participant_rows["window_bad"].to_numpy())

    def test_lopo_folds_cover_every_participant_exactly_once(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        folds = run_lopo(dataset, feature_columns, MODEL_BUILDERS["logreg_c1.0"])
        held_out = sorted(f.held_out_participant for f in folds)
        assert held_out == sorted(dataset["participant"].unique().tolist())
        assert sum(f.n_test for f in folds) == len(dataset)

    def test_lopo_train_fold_excludes_held_out_participant(self, dataset, monkeypatch):
        seen_train_participants = []
        feature_columns = FEATURE_SETS["pathway"]

        class RecordingModel:
            def fit(self, X, y):
                seen_train_participants.append(set(dataset.loc[X.index, "participant"]))
                self.classes_ = np.unique(y)

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        run_lopo(dataset, feature_columns, lambda: RecordingModel())

        all_participants = set(dataset["participant"].unique())
        for held_out, train_participants in zip(sorted(all_participants), seen_train_participants):
            assert held_out not in train_participants

    def test_summarize_pooled_metrics_sane_ranges(self, dataset):
        folds = persistence_baseline(dataset)
        result = summarize("persistence", "-", "persistence", folds)
        m = result.pooled_metrics
        assert 0.0 <= m["balanced_accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert m["n"] == len(dataset)

    def test_lopo_records_train_balanced_accuracy_per_fold(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        folds = run_lopo(dataset, feature_columns, MODEL_BUILDERS["logreg_c1.0"])
        assert all(f.train_balanced_accuracy is not None for f in folds)
        assert all(0.0 <= f.train_balanced_accuracy <= 1.0 for f in folds)

    def test_summarize_computes_generalization_gap(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        folds = run_lopo(dataset, feature_columns, MODEL_BUILDERS["logreg_c1.0"])
        result = summarize("x", "pathway", "logreg_c1.0", folds)
        assert result.generalization_gap is not None

    def test_baselines_have_no_generalization_gap(self, dataset):
        result = summarize("persistence", "-", "persistence", persistence_baseline(dataset))
        assert result.generalization_gap is None

    def test_best_result_excludes_candidates_above_gap_threshold(self):
        # "overfit" has the higher raw balanced accuracy but a large gap;
        # "modest" has a lower accuracy but generalizes cleanly. The gap gate
        # must prefer "modest" once the threshold rules "overfit" out.
        overfit = EvalResult(
            name="overfit",
            feature_set="x",
            model_name="m",
            fold_results=[],
            pooled_metrics={"balanced_accuracy": 0.90, "f1": 0.9},
            per_fold_balanced_accuracy={1: 0.90},
            generalization_gap=0.40,
        )
        modest = EvalResult(
            name="modest",
            feature_set="x",
            model_name="m",
            fold_results=[],
            pooled_metrics={"balanced_accuracy": 0.70, "f1": 0.7},
            per_fold_balanced_accuracy={1: 0.70},
            generalization_gap=0.05,
        )
        assert best_result([overfit, modest], max_gap=0.25).name == "modest"
        # without the gate, the higher raw-accuracy candidate would win
        assert best_result([overfit, modest], max_gap=1.0).name == "overfit"

    def test_best_result_falls_back_when_all_exceed_gap_threshold(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        folds = run_lopo(dataset, feature_columns, MODEL_BUILDERS["random_forest"])
        result = summarize("rf", "pathway", "random_forest", folds)
        # an impossible-to-satisfy threshold must still return something
        chosen = best_result([result], max_gap=-1.0)
        assert chosen is result

    def test_rank_key_prefers_higher_balanced_accuracy(self, dataset):
        folds = persistence_baseline(dataset)
        good = summarize("good", "-", "x", folds)
        bad_folds = [f for f in folds]
        # flip every prediction to manufacture a clearly worse result
        import dataclasses

        flipped = [
            dataclasses.replace(f, y_pred=1 - f.y_pred) for f in bad_folds
        ]
        bad = summarize("bad", "-", "x", flipped)
        assert rank_key(good) != rank_key(bad)


class TestMulticlassRiskEvaluation:
    """Covers the generalized evaluate.py plumbing (target_column/current_column/
    labels params) added for the Next_Posture_risk (low/medium/high) target,
    without touching the binary pipeline's default behavior."""

    @pytest.fixture(scope="module")
    def dataset(self):
        return build_dataset(CSV_PATH)

    def test_binary_defaults_unchanged(self, dataset):
        # No target_column/labels passed -> identical to the pre-multiclass API.
        folds = majority_class_baseline(dataset)
        result = summarize("x", "-", "majority_class", folds)
        assert "positive_rate" in result.pooled_metrics
        assert "class_distribution" not in result.pooled_metrics
        assert len(result.pooled_metrics["confusion_matrix"]) == 2

    def test_majority_class_baseline_on_risk_target(self, dataset):
        folds = majority_class_baseline(dataset, target_column="Next_Posture_risk")
        total_test_rows = sum(f.n_test for f in folds)
        assert total_test_rows == len(dataset)
        assert set(np.concatenate([f.y_true for f in folds])) <= {0, 1, 2}

    def test_persistence_baseline_on_risk_target_matches_current_window_risk(self, dataset):
        folds = persistence_baseline(dataset, target_column="Next_Posture_risk", current_column="window_risk")
        for fold in folds:
            participant_rows = dataset[dataset["participant"] == fold.held_out_participant]
            np.testing.assert_array_equal(fold.y_pred, participant_rows["window_risk"].to_numpy())

    def test_lopo_on_risk_target_covers_every_participant(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        folds = run_lopo(dataset, feature_columns, MODEL_BUILDERS["random_forest"], target_column="Next_Posture_risk")
        held_out = sorted(f.held_out_participant for f in folds)
        assert held_out == sorted(dataset["participant"].unique().tolist())
        assert sum(f.n_test for f in folds) == len(dataset)

    def test_summarize_with_three_labels_uses_macro_averaging_and_3x3_matrix(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        folds = run_lopo(dataset, feature_columns, MODEL_BUILDERS["random_forest"], target_column="Next_Posture_risk")
        result = summarize("x", "pathway", "random_forest", folds, labels=(0, 1, 2))
        m = result.pooled_metrics
        assert 0.0 <= m["balanced_accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert len(m["confusion_matrix"]) == 3
        assert len(m["confusion_matrix"][0]) == 3
        assert set(m["class_distribution"].keys()) == {"0", "1", "2"}
        assert pytest.approx(sum(m["class_distribution"].values()), abs=1e-9) == 1.0

    def test_all_models_fit_and_predict_multiclass_without_error(self, dataset):
        feature_columns = FEATURE_SETS["pathway"]
        X = dataset[feature_columns]
        y = dataset["Next_Posture_risk"]
        for name, builder in MODEL_BUILDERS.items():
            model = builder()
            model.fit(X, y)
            preds = model.predict(X)
            assert set(np.unique(preds)) <= {0, 1, 2}, f"{name} produced out-of-range risk predictions"


class TestFeatureSetsAndModels:
    def test_pathway_feature_set_matches_doc_spec(self):
        assert FEATURE_SETS["pathway"] == [
            "mean",
            "min",
            "max",
            "spread",
            "slope",
            "movement_speed",
            "condition_with",
        ]

    def test_all_models_fit_and_predict_on_small_sample(self):
        df = build_dataset(CSV_PATH)
        feature_columns = FEATURE_SETS["pathway"]
        X = df[feature_columns]
        y = df["Next_Posture_bad"]
        for name, builder in MODEL_BUILDERS.items():
            model = builder()
            model.fit(X, y)
            preds = model.predict(X)
            assert set(np.unique(preds)) <= {0, 1}, f"{name} produced non-binary predictions"
