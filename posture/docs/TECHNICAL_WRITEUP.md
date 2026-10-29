# Team 2 — Good/Bad Posture Prediction & Feedback System

**Team:** Daniel, Isaac, Deborah
**Pathway owned:** Posture (good/bad posture prediction and feedback). Stress is excluded per the assignment; Frustration/Fatigue belong to Teams 1 and 3.

This folder is the complete documentation package for this deliverable:

| File | What it is |
|---|---|
| `TECHNICAL_WRITEUP.md` | This file. |
| `participant_summary_table.md` | Reproduces the instructor's Excel reference table (*Posture Machine Learning.pdf*, Steps 2–8) programmatically, as a sanity check on the input data. |
| `risk_model_comparison.png` | LOPO balanced-accuracy bar chart, every candidate model, risk-level target. |
| `risk_confusion_and_participants.png` | Confusion matrix + per-participant accuracy, selected risk model. |
| `make_summary_table.py` / `make_charts.py` | The scripts that generated the two files above, for reproducibility. |

---

## 1. Input data

`data/posture_cleaned.csv` — 2,969 rows, one row per sensor reading at roughly 10 Hz, already cleaned (blanks removed, labels fixed, a duplicate participant-4 recording dropped, sorted, gaps split, smoothed) per the instructor's pathway doc.

| Column | Meaning |
|---|---|
| `Participant` | 1, 3, 4, 5, 6, 7 (6 participants total; no participant 2) |
| `Condition` | `with` / `without` (an experimental condition, e.g. exoskeleton) |
| `Segment` | recording segment within a participant/condition |
| `Timestamp` | seconds within that recording |
| `Y_Pitch`, `Y_Pitch_smoothed` | forward-lean angle in degrees (the only signal this pathway predicts from) |

`participant_summary_table.md` in this folder reproduces the instructor's own per-participant COUNTIF/COUNTIFS summary table from this exact CSV, integer-for-integer — confirmed before any further work, so everything downstream is known to start from the right data.

---

## 2. Preprocessing & windowing

Reuses the already-working binary pipeline (`posture_ml/windows.py`), extended (not replaced) to also produce a 3-level risk label. Steps:

1. **Label each reading** against the instructor's DBR thresholds on raw `Y_Pitch`:
   `≤20° → Low (Good)`, `20–45° → Medium (Bad)`, `>45° → High (Critical)`.
2. **Window**: cut each `(Participant, Condition, Segment)` recording into **non-overlapping 1-second windows** (10 readings each). The pathway doc specifies 2-second windows; we default to 1 second instead — see §3 for why, and `posture_ml/windows.py`'s module docstring for the full before/after comparison. The 2-second pipeline still exists (`--window-readings 20`) and is covered by regression tests.
3. **Per-window features** (7 "pathway" features the doc specifies, 3 extra engineered ones tried alongside): `mean, min, max, spread, slope (deg/s), movement_speed (deg/s), condition_with`, plus `std, median, bad_fraction_now`.
4. **Window-level labels** — two different aggregation rules, used for two different targets:
   - `window_bad` (binary): **majority vote** — `>50%` of readings in the window are Bad/Critical. This is the pathway doc's own rule.
   - `window_risk` (Low/Medium/High): **the worst reading in the window wins** (max-severity), *not* a majority vote. See §4 for why this matters.
5. **Forecast targets** — both are "what will the *next* window look like," built by shifting the current window's label back one position within its own recording: `Next_Posture_bad` (binary) and `Next_Posture_risk` (Low/Medium/High). The last window of every recording has no "next" window and is dropped (293 windows → 282/287 with a target, depending on which column you condition on).

---

## 3. Prediction model

Two forecasters exist side by side, trained on the same windowed dataset, evaluated the same way:

| | `train_posture_model.py` (existing) | `train_posture_risk_model.py` (new, this deliverable) |
|---|---|---|
| Target | `Next_Posture_bad` (binary) | `Next_Posture_risk` (Low/Medium/High) — **the required output for this assignment** |
| Candidates | logistic regression (×2 C), random forest, gradient boosting, SVM-RBF, × 2 feature sets, + majority-class/persistence baselines (12 total) | identical candidate grid (12 total) |
| Evaluation | **Leave-One-Participant-Out** (train on 5 people, test on the 6th, ×6) | same |
| Selection rule | highest pooled balanced accuracy, after excluding any candidate whose train/test generalization gap exceeds 0.25 (a model that just memorizes its training participants can't win on a technicality) | same rule |
| Selected | Random Forest, pathway's 7-feature set | Logistic Regression (C=1.0), extended 10-feature set |
| Result | **0.856 balanced accuracy**, gap 0.075 | **0.679 balanced accuracy**, gap 0.140 |
| Saved to | `models/posture_forecast_model.joblib` | `models/posture_risk_forecast_model.joblib` |

Why keep both instead of just the 3-class one: the binary model is the validated, higher-confidence signal (it only has to separate 2 classes, and "high"/Critical readings are rare in this dataset — see §6); the risk-level model is what the assignment's Low/Medium/High requirement and the feedback trigger actually consume. Both come from the exact same windowing/feature code, so there's no duplicated logic, only a duplicated *training script* (deliberately — see `train_posture_risk_model.py`'s own docstring for why it wasn't worth risking a shared-code refactor of the already-validated binary script for this deliverable).

Software: Python, pandas, NumPy, scikit-learn (matches the doc's "Software to use" section).

---

## 4. Risk classification (Low / Medium / High)

Every reading is classified against the instructor's own DBR thresholds — no separate rule needed, since the pathway doc already specifies them:

```
Y_Pitch ≤ 20°   -> Low    (Good)
20° < Y_Pitch ≤ 45° -> Medium (Bad)
Y_Pitch > 45°   -> High   (Critical)
```

The only new design decision is **how a 1-second *window* of 10 readings gets one risk label**. We use **max-severity aggregation** — the window's risk is the worst reading inside it — instead of the majority-vote rule the binary target uses. Reasoning:

- **Safety**: a single >45° spike inside an otherwise-fine second is exactly the kind of transient risk event a posture-safety system should flag, not average away.
- **It also evaluates better**: under majority-vote aggregation, "High" windows exist in only **2 of 6 participants** (useless for 4 of 6 Leave-One-Participant-Out folds — the model would never see a single "High" training example for those folds). Under max-severity aggregation, "High" windows are spread across **4 of 6 participants**, which is what makes the LOPO-validated risk model above possible at all.

---

## 5. Feedback logic

Implemented in `posture_feedback.py` as a pure, dependency-free state machine (same pattern as the existing `stress_state.py`), unit-tested independently of any model, poller, or TTS engine (`test_posture_feedback.py`, 15 tests).

Assignment rule, implemented exactly:

- **Never trigger feedback on Medium.** A Medium prediction resets the High streak, the same as Low — it's a streak-breaker, not an escalation.
- **Trigger feedback only after 3 consecutive High predictions.** The trigger fires **once**, the moment the streak reaches 3 (not on every subsequent High tick) — otherwise a 10-second-long High episode would fire 8 extra redundant alerts.
- **Reset on Low or Medium.** Any non-High prediction breaks the streak back to 0.
- (Added, beyond the literal spec, matching the existing `stress_state.py` pattern): dropping back to Low/Medium **after** feedback has fired emits a `resolved` event, so a UI can clear a "high risk" banner. This never fires feedback a second time on its own — it's purely informational.

---

## 6. Testing results

### 6.1 Risk-level model (Low/Medium/High) — the required output

LOPO = Leave-One-Participant-Out: every number below is on a participant the model never trained on. Metrics are macro-averaged across the 3 classes (each class weighted equally, regardless of how rare "High" is).

| Metric | Value |
|---|---|
| Balanced accuracy (macro, LOPO) | **0.679** |
| F1 (macro) | 0.667 |
| Precision (macro) | 0.659 |
| Recall (macro) | 0.679 |
| Overall accuracy | 0.720 |
| Train/test generalization gap | 0.140 (under the 0.25 exclusion threshold) |
| vs. majority-class baseline | 0.333 (chance for 3 balanced classes) |
| vs. "next = same as now" baseline | 0.657 |

**Pooled confusion matrix** (rows = true, columns = predicted, LOPO-pooled across all 6 participants):

| | pred Low | pred Medium | pred High |
|---|---|---|---|
| **true Low** | 139 | 35 | 5 |
| **true Medium** | 20 | 43 | 8 |
| **true High** | 4 | 7 | 21 |

Class balance in the data: Low 63.5%, Medium 25.2%, High 11.3% — High is rare, which is also where the model is weakest (21/32 = 66% of true-High windows correctly caught; most of the misses land in Medium, not Low, i.e. the model rarely mistakes a dangerous window for a completely safe one).

**Per-participant balanced accuracy** (each one entirely held out): P1 0.72, P3 0.49, P4 0.58, P5 0.66, P6 0.49, **P7 0.83**. See `risk_confusion_and_participants.png`.

**Full model comparison** (all 12 candidates) is in `risk_model_comparison.png` and `models/posture_risk_forecast_model_metadata.json`.

### 6.2 Binary model (for comparison — not the assignment's required output, but the stronger/more validated signal)

| Metric | Value |
|---|---|
| Balanced accuracy (LOPO) | 0.856 |
| F1 | 0.766 |
| Train/test generalization gap | 0.075 |

Full detail in `results/REPORT.md`, `results/model_comparison.png`, `results/confusion_and_participants.png`.

### 6.3 Code tests

| File | Tests | Covers |
|---|---|---|
| `test_posture_ml.py` | 41 | reading/window labeling (both binary and 3-class), no train/test leakage, LOPO fold coverage, gap-gating, multiclass metric plumbing |
| `test_posture_feedback.py` | 15 | the 3-consecutive-High trigger, Medium-never-triggers, streak reset, resolve events, participant isolation |
| `test_stress_state.py` | (pre-existing, Team 1's pattern this reuses) | n/a |

All 58 posture-specific tests pass (`pytest test_posture_ml.py test_posture_feedback.py -v`).

---

## 7. Pathway diagram

```
 1. CLEAN
    data/posture_cleaned.csv -- already cleaned (2,969 readings)
                       |
 2. LABEL (DBR thresholds on raw Y_Pitch)
    <=20 deg Low(Good) . 20-45 deg Medium(Bad) . >45 deg High(Critical)
                       |
 3. WINDOW  (non-overlapping, 1s / 10 readings per window)
    window_bad   = majority vote (>50% Bad+Critical)      [binary target]
    window_risk  = WORST reading in the window (max)       [3-class target]
                       |
 4. FEATURES per window
    mean, min, max, spread, slope, movement_speed, condition_with
    (+ std, median, bad_fraction_now for the "extended" feature set)
                       |
 5. FORECAST TARGET (shift window label back 1 within its own recording)
    Next_Posture_bad   (binary)        Next_Posture_risk (Low/Medium/High)
                       |                              |
 6. TRAIN  (12 candidates: 2 feature sets x 5 models + 2 baselines, each target)
                       |                              |
 7. TEST  Leave-One-Participant-Out  (train on 5, test on the 6th, x6)
                       |                              |
 8. SELECT  best balanced accuracy, excluding gap > 0.25
    -> Random Forest, 0.856 bal.acc.    -> LogReg C=1.0, 0.679 bal.acc.
       (models/posture_forecast_         (models/posture_risk_forecast_
        model.joblib)                     model.joblib)
                                                       |
                                     9. FEEDBACK (posture_feedback.py)
                                        3 consecutive "High" predictions in a
                                        row -> fire feedback once.
                                        Any Low/Medium prediction -> reset
                                        the streak. Medium never fires
                                        feedback on its own.
                                                       |
                                    10. LIVE DEMO (posture_live_replay.py)
                                        Replays a real recorded session
                                        through steps 4-9 live, serves
                                        /latest, speaks via Kokoro TTS when
                                        feedback fires, feeds the dashboard's
                                        new "Posture" tab.
```

---

## 8. Live demo integration (for the 1-minute video)

Two new pieces turn the trained model into something demoable, without needing a live posture sensor:

- **`posture_live_replay.py`** — a small Flask service (`http://localhost:7002/latest`) that replays three *real* recorded sessions back-to-back — P6-without (good), P1-with (bad, 43% of its readings are >45°), P5-without (good again) — at ~1 window/second, through the actual trained risk model and the actual `posture_feedback.py` state machine (not a scripted/fake sequence). Verified against the trained model's real output: the 3-consecutive-High trigger fires **twice** during the "bad" segment (~27s and ~35s into the loop) and resolves on return to "good" posture — a full good→bad→good arc in **~53 seconds**. When feedback fires, it speaks aloud via the existing Kokoro TTS pipeline (`tts_utils.py`, same engine Team-1's stress system uses) and posts a banner message, exactly like `construction_stress_speech.py` does for the frustration pathway.
- **Dashboard "Posture" tab** — added to `psychosocial_dashboard.html` as a new top-level tab (alongside the existing Frustration/Mental Demand view, untouched) that polls `posture_live_replay.py` directly and renders: the live angle trace with Low/Medium/High background bands, the current prediction, the consecutive-High streak meter, and the feedback banner.

To run the demo: `python train_posture_risk_model.py` (once) → `python posture_live_replay.py` → `python serve_dashboard.py` → open the dashboard, click the "Posture (Team 2)" tab.

---

## 9. Limitations

- **Only 6 participants**, 15–95 seconds of recording each (282–287 windows total). All metrics above are directional, not production-grade precision — re-evaluate as more participants/sessions are collected.
- **"High" (Critical, >45°) is rare** (11.3% of windows) and concentrated in 4 of 6 participants; the risk model's weakest class is exactly this one, and 2 of 6 participants (P4, P7) have zero Critical readings at all, so they contribute no "High" training signal whatsoever. The binary model sidesteps this by collapsing Medium+High together, which is part of why it scores higher — that tradeoff (coarser output, higher confidence) is the main limitation of requiring a 3-class output on a dataset this size.
- **The live demo replays real recorded data, not a live sensor.** This project's live biomarker pipeline streams EDA/temperature/pulse (wristband), not the forward-lean angle this pathway was trained on — there is no live posture sensor wired up yet. `posture_live_replay.py` is built so a real sensor feed could be substituted later without touching the model, feature engineering, or feedback logic (see that file's module docstring).
- **1-second windows** (not the doc's 2-second spec) were chosen because evaluation showed the 2-second horizon discards most of the forecastable signal (full comparison in `posture_ml/windows.py`'s docstring) — a deliberate, evaluated deviation from the reference doc, not an oversight.
- **Feedback resolution granularity**: the trigger operates on single consecutive predictions, so its exact timing (e.g. "3 seconds") depends on the 1-second window size; a different window size would need a different streak threshold to represent the same ~3 real seconds.
