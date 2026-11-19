# VIP Psychosocial Dashboard

A manager-facing dashboard for construction-site psychosocial risk, built on wristband biomarker data (EDA, temperature, pulse) and posture-angle data, with ML-driven Low/Medium/High risk predictions and spoken (TTS) feedback.

> **Looking for the posture ML write-up?** The full technical write-up (input data, preprocessing, model, risk classification, feedback logic, performance tables, limitations) is at [`posture/docs/TECHNICAL_WRITEUP.md`](posture/docs/TECHNICAL_WRITEUP.md). Performance charts and the reproduced instructor reference table are alongside it in [`posture/docs/`](posture/docs/).

## Layout

```
dashboard/     the web dashboard + its Flask server
biomarkers/    Frustration / Mental Demand pathway (EDA + temp + pulse -> stress ML)
posture/       Good/Bad Posture pathway (forward-lean angle -> posture ML)
docs/          project-wide setup notes
requirements.txt, .env.example   (root, shared across all three)
```

Each of `dashboard/`, `biomarkers/`, and `posture/` is self-contained: its own code, its own `models/` (trained artifacts), its own data. They share one Python virtual environment (`requirements.txt` at the repo root). Two pieces are deliberate exceptions, because they're cross-cutting by nature rather than belonging to one pathway: the live Polar H10 posture bridge (`dashboard/posture_live_bridge.py`) runs in-process inside the dashboard server but reuses `posture/`'s feedback state machine, windowing, and trained models; and the closed-loop retrain script (`dashboard/retrain_from_checkins.py`) reads check-in answers stored under `dashboard/` but retrains `biomarkers/`'s models. Both resolve the other folder's path explicitly rather than assuming a shared working directory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in AWS credentials only if you're using the real live poller (biomarkers/avro_stream_poller_with_ml.py)
```

TTS (used by both the biomarkers and posture pathways) needs the `espeak-ng` system package:
```bash
brew install espeak-ng          # macOS
sudo apt-get install espeak-ng  # Linux
```
The actual voice model (Kokoro-82M, ~330 MB) downloads automatically from Hugging Face the first time any script calls `init_engine()` — it is **not** checked into this repo.

## Run it

**1. Start a data source** (pick one, in its own terminal):
```bash
cd biomarkers && python3 avro_stream_poller_with_ml.py   # real wristband data via S3 (needs .env)
# or, with no AWS setup needed:
cd biomarkers && python3 mock_poller.py                  # scripted demo data
# and/or, for the posture pathway:
cd posture && python3 posture_live_replay.py              # replays a real recorded posture session
```

**2. Start the dashboard server** (second terminal):
```bash
cd dashboard && python3 serve_dashboard.py
```
Opens `http://127.0.0.1:8000/psychosocial_dashboard.html` — Frustration/Mental Demand and Posture are separate tabs. The Posture tab's **Connect Polar H10** button talks to a real Polar H10 over Web Bluetooth directly from the browser (Chrome/Edge, localhost or HTTPS) — no separate posture process needed for that path; `posture/posture_live_replay.py` above is a separate, independent way to demo the posture pathway from a recorded session instead.

**3. (optional) Start a speech script** for spoken feedback on the Frustration pathway:
```bash
cd biomarkers && python3 construction_stress_speech.py
```
The posture pathway speaks on its own, directly from `posture_live_replay.py`, when its feedback trigger fires.

Full setup/demo details: [`docs/TTS_SETUP.md`](docs/TTS_SETUP.md).

## Folders in detail

### `dashboard/`
- `psychosocial_dashboard.html` — the frontend (vanilla JS, no build step, three tabs: Frustration/Mental Demand, Posture, Research)
- `serve_dashboard.py` — static file server + `/api/dashboard-data`, `/api/posture/*`, `/api/checkin/*` routes; run this
- `live_dashboard_backend.py` — polls the biomarkers poller's `/latest`, merges it into dashboard data, and feeds each tick into the closed-loop check-in gate
- `system_message_api.py` — shared `/system-message` endpoint so a TTS banner shows on screen
- `data/dashboard_data.json` — baseline historical session dataset the dashboard renders
- `assets/` — Polar H10 EXG sensor panel (`polar-panel.js`) and the live posture accelerometer panel (`posture-panel.js`), both pre-bundled, no build step

**Live posture pathway (Pitch-primary, Roll-secondary):**
- `posture_dbr.py` — pure decision logic: a two-vector calibration (neutral pose + a forward lean) decomposes each reading into signed Pitch (forward/back tilt — the primary severity signal) and Roll (side-to-side tilt — secondary, used only to help attribute *cause*, e.g. side bending), replacing an earlier single omnidirectional tilt angle that couldn't tell the two apart
- `posture_live_bridge.py` — the I/O layer around it: buffers real Polar H10 accelerometer readings POSTed from `assets/posture-panel.js`'s Web Bluetooth connection, runs the two-phase calibration, and classifies each ~1s window
- `test_posture_dbr.py` — unit tests for the pitch/roll decomposition and cause attribution

**Closed-loop check-ins:** the Frustration/Mental Demand tab asks the worker a quick Yes/A little/No "Quick check" whenever a High prediction fires (rate-limited, auto-expiring), and offers a **How do you feel? Tell CHAP** button to self-report any time. The Research tab shows how often each prediction source was confirmed.
- `checkin.py` — pure ask/expire timing rules (when to ask, how long a question stays open)
- `checkin_bridge.py` — storage (`live_output/<participant>/checkins.{csv,json}`) and the agreement stats the Research tab reads
- `retrain_from_checkins.py` — builds a candidate `biomarkers/` model from the answered check-ins and only deploys it if it's at least as good on held-out workers' answers and doesn't meaningfully hurt the original study accuracy (see the script's own docstring for the full safety gate)
- `test_checkin.py` — unit tests for the ask/expire rules

### `biomarkers/`
Frustration / Mental Demand pathway: EDA + temperature + pulse -> trained classifiers -> Low/Medium/High -> stress-escalation speech.
- `avro_stream_poller_with_ml.py` / `mock_poller.py` — data sources (real S3 stream vs. scripted demo), both serve the same `/latest` JSON shape on port 7001
- `biomarker_runtime.py` — loads `models/*.joblib` and runs predictions
- `risk_score.py` — turns categorical labels into a single `overall` risk score + band
- `stress_state.py` + `construction_stress_speech.py` — pure escalation state machine + the TTS loop around it (round-1 / escalation reminders)
- `decision_based_rule.py` — rule-based (non-ML) classifier for comparison
- `speak_biomarker_labels.py` — simpler "announce on label change" TTS script
- `train_biomarker_models.py` — training script (needs the original raw labeled dataset, not included here)
- `test_stress_state.py` — 12 unit tests for the escalation logic
- `models/` — trained `frustration_model.joblib`, `mental_demand_model.joblib`

### `posture/`
Good/Bad Posture pathway: forward-lean angle -> windowed features -> trained classifiers -> Low/Medium/High -> 3-consecutive-High feedback trigger (with TTS).
- `posture_ml/` — labeling, windowing, feature engineering, Leave-One-Participant-Out evaluation
- `train_posture_model.py` — trains the binary (good/bad) forecaster
- `train_posture_risk_model.py` — trains the Low/Medium/High risk-level forecaster
- `posture_feedback.py` — pure state machine: fires feedback after 3 consecutive "High" predictions, never on Medium, resets on Low/Medium
- `posture_live_replay.py` — replays a real recorded session through the trained model + feedback logic + TTS, serving `/latest` on port 7002, for live demoing without a live posture sensor
- `data/posture_cleaned.csv` — the cleaned input dataset (6 participants, ~10 Hz forward-lean angle)
- `models/` — trained forecasters + their LOPO evaluation metadata
- `results/` — binary-model evaluation report + charts
- `docs/` — full technical write-up, performance tables/charts, and the reproduced instructor reference table
- `test_posture_ml.py`, `test_posture_feedback.py` — 56 unit tests

### `docs/`
- `TTS_SETUP.md` — full TTS setup + demo walkthrough (applies to both pathways)
- `REALTIME_RESEARCH.md` — research notes on real-time vs. polling options for the wristband data

## Not in this repo

A few things were deliberately left out of this public copy:
- **Raw per-session biomarker CSVs** (`bio/` locally) — real recorded sensor dumps, not needed for the code to run (the live poller pulls from S3 directly) and not something to publish.
- **Instructional reference material** (assignment PDFs/docx, an Excel decision-rule worksheet) — course handouts, not part of the running system.
- **Planning/design docs** for the closed-loop and Pitch/Roll features (e.g. a step-by-step implementation guide, a before/after fix writeup) — useful while building the feature, not needed to run it; the code's own docstrings and this README cover what matters for running it.
- **The Kokoro TTS model weights** (~120 MB locally) — auto-downloaded from Hugging Face on first use; checking in a binary model file isn't necessary.
- A couple of dead/non-portable one-off scripts that depended on a local absolute path from how the historical dataset was originally built.
