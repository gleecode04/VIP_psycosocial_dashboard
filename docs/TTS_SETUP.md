# Text-to-Speech Setup (Construction Worker Stress Demo)

This covers the in-class activity: a speech system that reacts to the live
Frustration/stress label, plus a demo-safe fallback that doesn't depend on
AWS/S3 or the trained models.

**Note on `speak_biomarker_labels.py`:** the handout says to download the
original from Teams chat. That file wasn't available in this environment,
so `speak_biomarker_labels.py` here is a reimplementation built from the
handout's described behavior (Step 5-6: announce a label only when it
changes). If your team has the real file, you can use it instead — it only
needs to read the same `/latest` JSON shape.

## Files

| File | Role |
|---|---|
| `avro_stream_poller_with_ml.py` | Real poller: S3 -> biomarkers -> ML -> `/latest` on port 7001 |
| `mock_poller.py` | **Demo fallback.** Same `/latest` shape, synthetic scripted data, no AWS needed |
| `tts_utils.py` | Shared TTS engine init (Kokoro-82M local neural voice, American female "Bella") + `speak()` |
| `stress_state.py` | Pure escalation state machine — no I/O, independently unit tested |
| `test_stress_state.py` | 12 unit tests for the escalation logic |
| `speak_biomarker_labels.py` | Base behavior: announce Frustration/Mental Demand on change |
| `construction_stress_speech.py` | **The in-class activity deliverable** — round 1 + escalation messages |

## Step 1 — Project folder

Everything above should sit in one folder, alongside `biomarker_runtime.py` and `models/`.

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

This now includes `kokoro`, `soundfile`, `requests`, and `numpy` in addition to the
original list (`boto3`, `pandas`, `flask`, `joblib`, `scikit-learn`,
`python-dotenv`).

## Step 3 — espeak-ng and audio player

`tts_utils.py` synthesizes speech locally with **Kokoro-82M** (Apache 2.0,
free, no API key, works offline). The model (~330 MB) downloads from
Hugging Face automatically on first run. Kokoro needs the `espeak-ng`
system package and a native command-line player:

- **macOS:** `brew install espeak-ng`; `afplay` ships with the OS
- **Linux:** `sudo apt-get install espeak-ng alsa-utils`
- **Windows:** install espeak-ng; PowerShell's `Media.SoundPlayer` is built in

## Step 4 — Start a data source (terminal 1)

**Option A — real data (needs working AWS/S3 access, see `.env`):**
```bash
python avro_stream_poller_with_ml.py
```

**Option B — demo fallback, no AWS needed, use this if S3 access breaks before the presentation:**
```bash
python mock_poller.py
```
It serves identical `/latest` JSON on the same port (7001), so nothing
downstream needs to change. It loops through a scripted scenario (low ->
medium -> high, held long enough to trigger round 1 and two escalation
reminders, then back to low) with a simulated clock that runs faster than
real time. By default each tick represents **22 simulated minutes** and
real ticks are **2 seconds apart**, so a full loop is about 24 seconds.
Tune it with:

```bash
SIM_MINUTES_PER_TICK=22 TICK_SECONDS=2 python mock_poller.py
```

Because each tick jumps 22 simulated minutes at once, the escalation
message will say **66 minutes** (not 45) the first time, and **132**
the second — it always speaks the real elapsed time from the data's own
timestamps, it never rounds down to the configured threshold. Use a
smaller `SIM_MINUTES_PER_TICK` (e.g. `15`) if you want the first
escalation to land closer to exactly 45.

`construction_stress_speech.py` and `speak_biomarker_labels.py` poll
`/latest` every **1 second** by default — faster than the mock's 2-second
tick — so they never skip a tick. `speak()` blocks until each message
finishes before the loop continues, so messages can never overlap or cut
each other off no matter how the poll/tick intervals are tuned.

Either way, confirm it's serving data:
```bash
curl http://localhost:7001/latest
```

## Step 5 — Start a speech script (terminal 2)

**Base behavior** (announces label changes):
```bash
python speak_biomarker_labels.py
```

**The in-class activity** (round 1 + escalation):
```bash
python construction_stress_speech.py
```

You should hear, in order:
1. On the first "high" reading: *"Are you feeling tired? Please have some water."*
2. If "high" persists continuously for 45 minutes (of data time, not wall-clock time — see below): *"Please take a break. You've been stressed for 45 minutes."* — repeating every 45 minutes after that as long as it's still "high".

## Why 45 minutes, and why elapsed time isn't wall-clock time

**The 45-minute default is backed by NIOSH's heat-stress work/rest
guidance**, which recommends a **45-minutes-work / 15-minutes-rest** cycle
for heavy manual labor around a 95 F heat index (NIOSH Publication
2016-106, *Criteria for a Recommended Standard: Occupational Exposure to
Heat and Hot Environments*; summarized in OSHA's *Using the Heat Index: A
Guide for Employers*). That's the standard cadence this kind of system is
meant to reinforce for a construction crew, so the escalation interval
reuses it. Change it for a demo:

```bash
ESCALATION_MINUTES=1 ESCALATION_REPEAT_MINUTES=1 python construction_stress_speech.py
```

**Elapsed time is computed from each sample's own timestamp
(`eda_time`), not wall-clock time.** That's deliberate: it makes the
"you've been stressed for N minutes" number reflect the participant's
actual data timeline, and it's exactly what lets `mock_poller.py` compress
45 real minutes into under a minute of demo time by advancing its own
simulated clock faster than real time — the speech script doesn't need to
know or care that it's talking to synthetic, time-compressed data.

## Stress signal used

`construction_stress_speech.py` treats the **Frustration** label as the
stress proxy (falling back to Mental Demand only during the first minute
of a session, before the Frustration model has two samples to work with —
see `biomarker_runtime.py`). "High" is a categorical label from the
trained classifier (`low` / `medium` / `high`), not a numeric threshold.

## Voice selection

`tts_utils.py` defaults to `af_bella` (American female "Bella"). Pass
`name_hint="heart"`, `"nicole"`, `"sarah"`, `"sky"`, `"michael"`, or `"adam"`
to `init_engine()` for alternatives, or any full Kokoro voice id. Override
playback volume for automated testing with `TTS_VOLUME=0` (speech is still
synthesized, just not played).

## Running the tests

```bash
python test_stress_state.py
```

12 tests cover: entering/leaving "high", the exact escalation threshold,
repeat timing, participant switches, and out-of-order timestamps. These
test the pure logic only — no network, no audio, no Flask — so they run in
well under a second.

## On-screen fallback for recordings

`construction_stress_speech.py` posts every message it speaks (and when a
stress episode ends) to a `/system-message` endpoint on whichever poller
is running (`system_message_api.py`, used by both
`avro_stream_poller_with_ml.py` and `mock_poller.py`). The dashboard shows
it as a banner above the worker panel, with the exact text and the time it
was said — so if a screen recording's audio capture fails, the message is
still visible on screen. It updates on the dashboard's normal 5-second
poll cycle, same as everything else.

Capturing the actual TTS audio in a screen recording needs one extra step
on macOS, since the built-in recorder (Cmd+Shift+5) only captures the
microphone, not system/app audio:
- **OBS Studio** (free): its "Application Audio Capture" source (macOS 13+)
  can capture just the Python process's audio directly, no extra driver.
- **BlackHole** (free virtual audio driver) + a Multi-Output Device in
  Audio MIDI Setup, then record with QuickTime using BlackHole as the
  input. More setup steps than OBS.

## If something breaks right before the presentation

Switch terminal 1 from `avro_stream_poller_with_ml.py` to `mock_poller.py`
— everything downstream (the dashboard, both speech scripts) keeps working
unchanged, since the JSON shape is identical.
