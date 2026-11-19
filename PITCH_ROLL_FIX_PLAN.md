# Pitch-primary / Roll-secondary severity fix — implementation plan

**Status:** implemented (posture_dbr.py: AxisFrame/calibrate_axes/pitch_deg/roll_deg,
WindowStats, classify_cause's side_bending branch; posture_live_bridge.py: two-phase
calibration state machine, pitch-driven severity; psychosocial_dashboard.html:
calibration copy, status branch, Pitch-primary/Roll-secondary metrics; tests added
to test_posture_dbr.py). Verified: unit tests pass, and a live-bridge smoke test
confirms a pure side lean no longer escalates severity (pitch~0 -> LOW) while a
pure forward lean does (pitch=50 -> HIGH), with roll correctly attributing
side_bending as the cause either way.

## 1. The issue (verified against current code)

Mentor's note: Pitch is the primary signal for detecting slouch/forward-lean (the
dangerous direction); Roll is a secondary signal for unbalanced/asymmetric sitting.
Our live DBR path doesn't currently compute Pitch at all.

Confirmed in `posture_dbr.py`:
- `tilt_angle_deg()` (line 101) returns a single **omnidirectional** angle between
  the live gravity vector and the calibrated neutral vector — no forward/back vs.
  left/right distinction.
- `classify_tilt_severity()` (line 119), which assigns Good/Medium/High, runs
  **only** on that omnidirectional magnitude (`stats.max_tilt_deg`, used at line 247
  in `posture_live_bridge.py`).
- The only directional signal that exists is inside `classify_cause()` (line 165),
  which compares `mean_day` vs `mean_daz` as a rough heuristic to *label* a cause
  (`side_bending` vs `forward_bending`) — downstream of severity, never feeding
  back into it.

Net effect: a pure sideways lean and a pure forward slouch of the same magnitude
get identical severity today. The sheet's Pitch thresholds (0–20 Good / >20 Bad /
>45 Critical — already encoded as `GOOD_MAX_DEG`/`BAD_MAX_DEG`) are being applied
to the wrong (blended) quantity.

### Why we can't just reuse the old single-axis formula
The original `Y_Pitch = atan2(ax, sqrt(ay²+az²))` formula assumes a fixed mapping
from one raw accelerometer axis to "forward lean." That assumption is exactly what
broke when the strap moved from chest to lower back (see `posture_dbr.py` module
docstring) — neutral read ~-45°, full bend read ~-2°. A mount can be rotated or
flipped relative to the body in ways we can't predict, so we can't hardcode which
raw axis is "forward" — same reason we moved to calibrated-neutral `tilt_angle_deg`
in the first place. Isolating true Pitch from true Roll needs **one more piece of
calibration data** than we currently capture (today we only ever capture one
reference vector: neutral).

## 2. The fix: two-vector calibration, then decompose

### Calibration (extends the existing single-step calibration)
1. **Phase 1 (existing, unchanged):** wearer holds neutral for `CALIBRATION_READINGS`
   samples → average → `v_neutral` (already `self._neutral` in
   `posture_live_bridge.py`).
2. **Phase 2 (new):** immediately after, prompt "Now lean forward slightly and
   hold" for the same number of readings → average → `v_forward_ref`.

This gives us a second reference vector that defines the *sagittal plane* (the
forward/back bending plane) for this specific wearer and this specific mount
orientation — no assumption about which raw axis is "forward" required.

### Geometry
- Lateral (roll) axis: `n̂ = normalize(cross(v_neutral, v_forward_ref))`. This is,
  by construction, the axis a forward lean rotates *about* — i.e. the left-right
  body axis, regardless of how the sensor happens to be mounted.
- **Pitch**: project the live sample onto the sagittal plane spanned by
  `v_neutral` and `v_forward_ref`; signed angle of that projection from
  `v_neutral` within the plane. Positive = toward the forward reference
  (slouching forward), negative = leaning back.
- **Roll**: the angular deviation component *perpendicular* to the sagittal plane
  (i.e. the part of the tilt that's along `n̂`) — captures side-to-side lean that
  pitch doesn't explain.

This is still fully mount-orientation-agnostic (works no matter how the strap
sits), it just needs two reference poses instead of one.

### New functions in `posture_dbr.py`
- `calibrate_axes(v_neutral, v_forward_ref) -> AxisFrame` — returns `n̂` (roll axis)
  and the sagittal-plane basis, computed once per session/recalibration.
- `pitch_deg(sample, neutral, frame) -> float` (signed)
- `roll_deg(sample, neutral, frame) -> float` (signed)
- Keep `tilt_angle_deg()` as-is — still useful as an omnidirectional diagnostic
  value for display, just no longer the thing severity is based on.

### Severity (primary signal — Pitch only)
- `classify_tilt_severity()` changes input from `stats.max_tilt_deg` to
  `abs(stats.max_pitch_deg)`. Thresholds (`GOOD_MAX_DEG`/`BAD_MAX_DEG`) unchanged —
  they already match the sheet's Pitch row, they were just wired to the wrong
  input.

### Roll (secondary signal — cause attribution only, never drives severity up)
- `classify_cause()`'s `side_bending` branch switches from the
  `mean_day`/`mean_daz` heuristic to a direct check on `roll_deg` magnitude
  (cleaner, no more guessing which raw axis is lateral).
- Roll is surfaced in `window_features`/UI as a secondary stat (e.g.
  "Lean: 8° to the right") but **does not** change the Good/Bad/Critical level —
  matches "Lean Angle/Roll (Secondary Sensor)" framing. A future refinement could
  give Roll its own informational-only severity label if the sheet ever specifies
  Roll thresholds, but nothing in the current spec sheet asks for that.

### `WindowStats` dataclass changes
Add `mean_pitch_deg`, `max_pitch_deg`, `mean_roll_deg`, `max_roll_deg`. Keep
`mean_tilt_deg`/`max_tilt_deg` (still useful, now informational-only).

### `posture_live_bridge.py` changes
- `_calibration_buffer` logic (currently single-phase, `add_reading()` around
  line 220) splits into two phases: after neutral calibration completes, don't go
  straight to `"running"` — go to a new status `"calibrating_forward"`, collect
  another `CALIBRATION_READINGS` samples, average into `v_forward_ref`, compute
  the axis frame, *then* go to `"running"`.
- `start_session()` / recalibrate flow resets both calibration phases.
- `_process_window()` computes `pitch_deg`/`roll_deg` for the window instead of
  (or alongside) `tilt_deg`; `latest` dict gains `pitch_deg`, `roll_deg` keys.
- `LIVE_MODE == "ml"` path is untouched — it intentionally still uses the legacy
  chest-pitch formula for compatibility with the trained model; out of scope here.

### `psychosocial_dashboard.html` changes
- New calibration instruction text for phase 2: "Now lean forward slightly and
  hold..." (mirrors the existing "hold neutral" copy for phase 1).
- `renderPostureStatus()` gets a `'calibrating_forward'` branch.
- `renderPostureMetrics()` / `renderPostureGuidance()` show Pitch as the primary
  stat (where tilt is shown today) and Roll as a secondary stat line.

### Tests (`test_posture_dbr.py`)
Add cases for:
- Neutral sample → `pitch_deg ≈ 0`, `roll_deg ≈ 0`.
- Pure forward lean (sample along `v_forward_ref` direction) → `pitch_deg > 0`,
  `roll_deg ≈ 0`.
- Pure side lean (sample rotated about `n̂` only) → `roll_deg > 0`,
  `pitch_deg ≈ 0`.
- Combined forward+side lean → both nonzero, pitch magnitude still drives
  severity via `classify_tilt_severity`, not the combined magnitude.
- `classify_cause`'s `side_bending` branch triggers off `roll_deg`, not the old
  `mean_day`/`mean_daz` ratio.

## 3. Fallback option (not recommended, documented for completeness)
If a two-phase calibration gesture is judged too much UX friction: pick whichever
raw axis has the *smallest* magnitude in `v_neutral` as a heuristic lateral axis
(reasoning: the dominant neutral-baseline axis is probably the spine-vertical
axis, so the other two span the tilt plane) and arbitrarily assign one of the
remaining two as "pitch axis," one as "roll axis." This is weaker — it's a
heuristic, not a measured reference — and reintroduces some of the same
mount-dependent guessing we moved away from. Only fall back to this if the
two-reference-vector calibration proves impractical in practice.

## 4. Out of scope
- Yaw: not in the sheet's actionable thresholds, and not measurable from a single
  accelerometer anyway (needs gyro/magnetometer fusion) — not touched.
- `LIVE_MODE=ml`: untouched, still uses the legacy fixed-axis formula on purpose
  for model compatibility.

## 5. File-by-file change list (for quick execution later)
1. `posture_dbr.py` — add `AxisFrame`, `calibrate_axes()`, `pitch_deg()`,
   `roll_deg()`; update `WindowStats`, `compute_window_stats()`,
   `classify_tilt_severity()` call site, `classify_cause()`'s side_bending branch.
2. `posture_live_bridge.py` — two-phase calibration state machine, `latest` dict
   keys, `_process_window()` wiring.
3. `psychosocial_dashboard.html` — calibration copy, status branch, metrics/
   guidance display.
4. `test_posture_dbr.py` — new pitch/roll decomposition tests.
