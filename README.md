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

Each of `dashboard/`, `biomarkers/`, and `posture/` is self-contained: its own code, its own `models/` (trained artifacts), its own data. They share one Python virtual environment (`requirements.txt` at the repo root).

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
Opens `http://127.0.0.1:8000/psychosocial_dashboard.html` — Frustration/Mental Demand and Posture are separate tabs.

**3. (optional) Start a speech script** for spoken feedback on the Frustration pathway:
```bash
cd biomarkers && python3 construction_stress_speech.py
```
The posture pathway speaks on its own, directly from `posture_live_replay.py`, when its feedback trigger fires.

Full setup/demo details: [`docs/TTS_SETUP.md`](docs/TTS_SETUP.md).

## Folders in detail

### `dashboard/`
- `psychosocial_dashboard.html` — the frontend (vanilla JS, no build step, two tabs)
- `serve_dashboard.py` — static file server + `/api/dashboard-data` bridge; run this
- `live_dashboard_backend.py` — polls the biomarkers poller's `/latest` and merges it into dashboard data
- `system_message_api.py` — shared `/system-message` endpoint so a TTS banner shows on screen
- `data/dashboard_data.json` — baseline historical session dataset the dashboard renders
- `assets/`, `sensor/` — Polar H10 EXG sensor panel (Bluetooth), pre-bundled + source

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
- **The Kokoro TTS model weights** (~120 MB locally) — auto-downloaded from Hugging Face on first use; checking in a binary model file isn't necessary.
- A couple of dead/non-portable one-off scripts that depended on a local absolute path from how the historical dataset was originally built.
