"""Decision-Based Rule (DBR) engine for live Polar H10 posture monitoring.

Pure decision logic, no I/O -- mirrors posture_feedback.py's "no I/O on
purpose" design so this is independently unit-testable without a Bluetooth
connection, a browser, or a TTS engine. posture_live_bridge.py is the I/O
loop around it (buffers readings, calls these functions, speaks the result).

Thresholds and the cause taxonomy come from the instructor's "DBR Posture" /
"Feedback System Posture" spec sheet (Pitch/Roll/Yaw thresholds, Good/Bad/
Critical posture levels, and a Cause column: Forward Bending, Twisting/
Rotation, Overhead Work, Asymmetric Lifting, Static Posture).

Why tilt-from-neutral instead of a fixed-axis pitch formula
-------------------------------------------------------------
The original single-axis formula, pitch = atan2(ax, sqrt(ay^2+az^2)), reads
"forward lean" off one specific accelerometer axis. That only works if the
sensor is always mounted the same way relative to the body (e.g. flat
against the chest, with a known axis pointing "up"). Strapped to the lower
back instead, the device's local axes are rotated relative to that
assumption -- which is exactly why a neutral standing posture was reading
~-45 degrees, and a full forward bend was reading ~-2 degrees: the axis the
formula reads as "forward tilt" isn't the axis that actually tracks forward
tilt for this mount point, so the output isn't even monotonic with real
bending.

tilt_angle_deg() fixes this by calibrating a neutral gravity-vector baseline
at the start of each session (the wearer holds still in a neutral pose for
~1s) and then reporting the angle *between* the live gravity vector and that
calibrated neutral vector. That's mount-orientation-agnostic: it is 0
degrees at neutral by construction, and grows monotonically the further the
body tilts away from neutral in *any* direction, regardless of which raw
axis that motion happens to show up on.

Pitch vs. Roll (see PITCH_ROLL_FIX_PLAN.md)
--------------------------------------------
Instructor feedback: Pitch (forward/back lean) is the primary severity
signal; Roll (side-to-side lean) is secondary and should inform *cause*,
not severity. tilt_angle_deg() alone can't tell those apart -- it's a single
omnidirectional magnitude, so a pure sideways lean and a pure forward
slouch of the same size look identical to it.

Splitting the single calibrated-neutral vector into true pitch and roll
needs one more reference vector than tilt_angle_deg() uses: a forward-lean
pose captured right after neutral (see calibrate_axes()). That second
vector pins down the sagittal (forward/back) plane for this specific
wearer and mount orientation, with no assumption about which raw axis is
"forward" -- the same mount-agnostic reasoning that motivated tilt_angle_deg
in the first place, just with two reference poses instead of one.
pitch_deg()/roll_deg() then decompose any later sample against that frame.
tilt_angle_deg() itself is unchanged and kept as an informational,
omnidirectional value -- severity is now driven by pitch alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

LOW, MEDIUM, HIGH = "low", "medium", "high"

# Severity thresholds on tilt-from-neutral angle, degrees -- the DBR sheet's
# Pitch row: Good 0-20, Bad >20, Critical >45.
GOOD_MAX_DEG = 20.0
BAD_MAX_DEG = 45.0

CAUSE_NEUTRAL = "neutral"
CAUSE_FORWARD_BENDING = "forward_bending"
CAUSE_TWISTING = "twisting"
CAUSE_SIDE_BENDING = "side_bending"
CAUSE_OVERHEAD = "overhead_work"
CAUSE_ASYMMETRIC = "asymmetric_lifting"
CAUSE_STATIC = "static_posture"
CAUSE_EXCESSIVE = "excessive_activity"
CAUSE_UNKNOWN = "posture_deviation"

# Cause-pattern thresholds, in g (accelerometer units). Tuned for slow,
# deliberate work movements, not athletic motion -- see classify_cause()
# docstring for the reasoning behind each branch and its caveats.
OSCILLATION_MIN_SIGN_CHANGES = 2
STATIC_STD_G = 0.03
HIGH_STD_G = 0.15
UNEVEN_MAX_STD_SPREAD_G = 0.02
LATERAL_DOMINANT_RATIO = 1.4

# Static-posture duration escalation (DBR sheet's Duration row: Static
# Posture Bad >5min, Critical >10min) -- independent of lean angle, since
# staying still is itself an MSD risk factor even in an otherwise-fine pose.
STATIC_BAD_SECONDS = 5 * 60
STATIC_CRITICAL_SECONDS = 10 * 60

_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2}


@dataclass(frozen=True)
class GravitySample:
    """One accelerometer reading, in g, however the device happens to be mounted."""
    ax: float
    ay: float
    az: float


@dataclass(frozen=True)
class AxisFrame:
    """Orthonormal frame built from two calibration poses (see
    calibrate_axes()), used to decompose any later sample into signed pitch
    (forward/back tilt) and roll (side-to-side tilt) -- see the module
    docstring for why two reference vectors are needed instead of one.

    up: the calibrated neutral direction (unit vector); pitch/roll are both
        0 here by construction.
    forward: unit vector orthogonal to `up`, pointing toward the forward-
        lean reference pose within the sagittal (forward/back) plane.
    lateral: unit vector orthogonal to both `up` and `forward` -- the axis a
        forward lean rotates *about* (the body's left-right axis).
    """
    up: Tuple[float, float, float]
    forward: Tuple[float, float, float]
    lateral: Tuple[float, float, float]


@dataclass(frozen=True)
class WindowStats:
    mean_tilt_deg: float
    max_tilt_deg: float
    mean_pitch_deg: float
    max_pitch_deg: float
    mean_roll_deg: float
    max_roll_deg: float
    mean_dax: float
    mean_day: float
    mean_daz: float
    std_dax: float
    std_day: float
    std_daz: float
    sign_changes_dax: int
    sign_changes_day: int


def _vector_norm(v: GravitySample) -> float:
    return math.sqrt(v.ax * v.ax + v.ay * v.ay + v.az * v.az)


def _as_tuple(v: GravitySample) -> Tuple[float, float, float]:
    return (v.ax, v.ay, v.az)


def _normalize_tuple(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0:
        raise ValueError("cannot normalize a zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def calibrate_axes(v_neutral: GravitySample, v_forward_ref: GravitySample) -> AxisFrame:
    """Build the pitch/roll axis frame from the two calibration poses: the
    existing neutral hold, plus a new "lean forward slightly and hold"
    phase captured right after it.

    `lateral` (the roll axis) is, by construction, the axis a forward lean
    rotates *about* -- i.e. the body's left-right axis, regardless of how
    the sensor happens to be mounted (no hardcoded raw axis involved).
    `forward` completes a right-handed orthonormal frame with `up` and
    `lateral`, giving a sagittal-plane direction to measure pitch against.
    """
    up = _normalize_tuple(_as_tuple(v_neutral))
    lateral_raw = _cross(up, _as_tuple(v_forward_ref))
    lateral_norm = math.sqrt(sum(c * c for c in lateral_raw))
    if lateral_norm < 1e-6:
        # Degenerate calibration -- the forward-lean pose came back ~parallel
        # to neutral (e.g. the wearer didn't actually lean). Fall back to an
        # arbitrary lateral axis orthogonal to `up` rather than crashing;
        # pitch/roll stay well-defined, just without a validated forward
        # direction from this calibration.
        arbitrary = (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)
        lateral_raw = _cross(up, arbitrary)
        lateral_norm = math.sqrt(sum(c * c for c in lateral_raw))
    lateral = tuple(c / lateral_norm for c in lateral_raw)
    forward = _cross(lateral, up)  # unit already: cross of two orthonormal unit vectors
    return AxisFrame(up=up, forward=forward, lateral=lateral)


def pitch_deg(sample: GravitySample, frame: AxisFrame) -> float:
    """Signed forward/back tilt -- rotation about the frame's lateral axis.
    0 at neutral; positive toward the forward-lean reference pose
    (slouching forward), negative leaning back. This is now the PRIMARY
    severity signal (see classify_tilt_severity's call site)."""
    norm = _vector_norm(sample)
    if norm == 0:
        return 0.0
    v = (sample.ax / norm, sample.ay / norm, sample.az / norm)
    return math.degrees(math.atan2(_dot(v, frame.forward), _dot(v, frame.up)))


def roll_deg(sample: GravitySample, frame: AxisFrame) -> float:
    """Signed side-to-side tilt -- rotation about the frame's forward axis.
    0 at neutral. SECONDARY signal only: used to attribute *cause*
    (classify_cause's side_bending branch), never to drive severity."""
    norm = _vector_norm(sample)
    if norm == 0:
        return 0.0
    v = (sample.ax / norm, sample.ay / norm, sample.az / norm)
    return math.degrees(math.atan2(_dot(v, frame.lateral), _dot(v, frame.up)))


def tilt_angle_deg(sample: GravitySample, neutral: GravitySample) -> float:
    """Angle between the live gravity vector and the calibrated neutral one.

    0 at neutral by construction; grows with any bending direction. See the
    module docstring for why this replaces a fixed-axis pitch formula.
    """
    sample_norm, neutral_norm = _vector_norm(sample), _vector_norm(neutral)
    if sample_norm == 0 or neutral_norm == 0:
        return 0.0
    dot = sample.ax * neutral.ax + sample.ay * neutral.ay + sample.az * neutral.az
    cos_angle = max(-1.0, min(1.0, dot / (sample_norm * neutral_norm)))
    return math.degrees(math.acos(cos_angle))


def axis_delta(sample: GravitySample, neutral: GravitySample) -> GravitySample:
    return GravitySample(sample.ax - neutral.ax, sample.ay - neutral.ay, sample.az - neutral.az)


def classify_tilt_severity(tilt_deg: float) -> str:
    if tilt_deg <= GOOD_MAX_DEG:
        return LOW
    if tilt_deg <= BAD_MAX_DEG:
        return MEDIUM
    return HIGH


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _sign_changes(values: Sequence[float]) -> int:
    changes = 0
    for a, b in zip(values, values[1:]):
        if a == 0 or b == 0:
            continue
        if (a > 0) != (b > 0):
            changes += 1
    return changes


def compute_window_stats(window: Sequence[GravitySample], neutral: GravitySample, frame: AxisFrame) -> WindowStats:
    """Per-axis deviation-from-neutral stats for one ~1s window of readings,
    plus the window's pitch/roll decomposition against `frame` (see
    calibrate_axes()).
    """
    deltas = [axis_delta(s, neutral) for s in window]
    tilts = [tilt_angle_deg(s, neutral) for s in window]
    pitches = [pitch_deg(s, frame) for s in window]
    rolls = [roll_deg(s, frame) for s in window]
    daxs = [d.ax for d in deltas]
    days = [d.ay for d in deltas]
    dazs = [d.az for d in deltas]
    return WindowStats(
        mean_tilt_deg=sum(tilts) / len(tilts),
        max_tilt_deg=max(tilts),
        # max_* keeps sign and picks the largest-MAGNITUDE reading in the
        # window (not just the largest positive one), the same way
        # max_tilt_deg already behaves (tilt is unsigned, so "max" and "max
        # magnitude" were always the same thing there).
        mean_pitch_deg=sum(pitches) / len(pitches),
        max_pitch_deg=max(pitches, key=abs),
        mean_roll_deg=sum(rolls) / len(rolls),
        max_roll_deg=max(rolls, key=abs),
        mean_dax=sum(daxs) / len(daxs),
        mean_day=sum(days) / len(days),
        mean_daz=sum(dazs) / len(dazs),
        std_dax=_std(daxs),
        std_day=_std(days),
        std_daz=_std(dazs),
        sign_changes_dax=_sign_changes(daxs),
        sign_changes_day=_sign_changes(days),
    )


def classify_cause(stats: WindowStats) -> str:
    """Attribute a non-neutral window to one of the DBR's named causes.

    Mirrors the spec sheet's Ax/Ay/Az interpretation table: Forward Bending
    is a sustained mean shift (a held posture, low *within-window*
    variation); Twisting is an oscillating sign flip across two axes;
    Overhead Work is high *variation* (not a sustained shift) concentrated
    on one axis, the signature of a moving arm/shoulder coupling into the
    trunk sensor; Asymmetric Lifting is variation spread diffusely across
    all three axes with no single dominant direction.

    Static Posture is deliberately NOT one of this function's outputs: a
    single ~1s window held steady is just normal controlled bending (e.g.
    reaching, lifting), not yet the sustained-minutes pattern the sheet's
    Static Posture row describes. That's a duration concept spanning many
    windows, so it's handled separately by escalate_for_static_duration()
    rather than guessed from one window's low variance here -- otherwise
    every calm, deliberate bend would get mislabeled "static" immediately.

    Caveat: this is a single lower-back accelerometer, not a full-body
    motion-capture rig. Overhead Work and Asymmetric Lifting in particular
    are proxies inferred from how the trunk responds to arm/load work, not a
    direct measurement of arm height or load symmetry -- treat them as
    best-effort, not validated classifications.
    """
    if stats.max_tilt_deg <= GOOD_MAX_DEG:
        return CAUSE_NEUTRAL

    if (stats.sign_changes_dax >= OSCILLATION_MIN_SIGN_CHANGES
            and stats.sign_changes_day >= OSCILLATION_MIN_SIGN_CHANGES):
        return CAUSE_TWISTING

    stds = (stats.std_dax, stats.std_day, stats.std_daz)
    total_std = sum(stds)

    if stats.std_day >= max(stds) and stats.std_day > HIGH_STD_G:
        return CAUSE_OVERHEAD

    # Only call it "diffuse/uneven" once there's actual movement to be
    # diffuse about -- a dead-steady held lean has a ~0 spread too, but
    # that's forward_bending (below), not asymmetric.
    if total_std > STATIC_STD_G and (max(stds) - min(stds)) <= UNEVEN_MAX_STD_SPREAD_G:
        return CAUSE_ASYMMETRIC

    # Roll (secondary signal) vs. Pitch (primary signal), directly -- no more
    # guessing which raw delta axis is "lateral" from mean_day/mean_daz.
    if abs(stats.mean_roll_deg) > abs(stats.mean_pitch_deg) * LATERAL_DOMINANT_RATIO:
        return CAUSE_SIDE_BENDING

    if total_std > HIGH_STD_G:
        return CAUSE_EXCESSIVE

    return CAUSE_FORWARD_BENDING


def escalate_for_static_duration(base_severity: str, cause: str, unchanged_seconds: float) -> Tuple[str, str]:
    """Apply the Duration row's escalation: holding the *same* cause for a
    long time is itself a risk, independent of lean angle. Only escalates
    (never downgrades), and only relabels the cause to static_posture when
    duration is actually what's driving the escalation -- an active bad
    angle (e.g. sustained forward bending) keeps its own cause even if it's
    also been held a while, since that's the more actionable reason.
    """
    if unchanged_seconds >= STATIC_CRITICAL_SECONDS:
        duration_severity = HIGH
    elif unchanged_seconds >= STATIC_BAD_SECONDS:
        duration_severity = MEDIUM
    else:
        return base_severity, cause

    if _RANK[duration_severity] > _RANK[base_severity]:
        return duration_severity, CAUSE_STATIC
    if duration_severity == base_severity and cause == CAUSE_NEUTRAL:
        return duration_severity, CAUSE_STATIC
    return base_severity, cause


_CAUSE_LABELS = {
    CAUSE_NEUTRAL: "neutral position",
    CAUSE_FORWARD_BENDING: "forward bending",
    CAUSE_TWISTING: "twisting or rotation",
    CAUSE_SIDE_BENDING: "side bending",
    CAUSE_OVERHEAD: "overhead or raised-arm work",
    CAUSE_ASYMMETRIC: "uneven, asymmetric loading",
    CAUSE_STATIC: "staying in the same position too long",
    CAUSE_EXCESSIVE: "a lot of fast movement",
    CAUSE_UNKNOWN: "a non-neutral posture",
}

# (severity, cause) -> the spec sheet's cause-specific feedback line. Only
# the 5 causes the sheet gives dedicated copy for have entries; anything
# else falls back to _GENERIC_FEEDBACK's per-level text.
_CAUSE_FEEDBACK = {
    (MEDIUM, CAUSE_FORWARD_BENDING): "Straighten your back and lift your head slightly.",
    (MEDIUM, CAUSE_TWISTING): "Reposition your feet before continuing.",
    (MEDIUM, CAUSE_OVERHEAD): "Lower your arms briefly if safe.",
    (MEDIUM, CAUSE_ASYMMETRIC): "Distribute weight evenly and adjust your stance.",
    (MEDIUM, CAUSE_STATIC): "Change position and stretch briefly.",
    (HIGH, CAUSE_FORWARD_BENDING): "Unsafe posture detected. Stand upright and reposition.",
    (HIGH, CAUSE_TWISTING): "Stop and reposition before continuing.",
    (HIGH, CAUSE_OVERHEAD): "Take a short arm recovery break if safe.",
    (HIGH, CAUSE_ASYMMETRIC): "Realign your body before lifting.",
    (HIGH, CAUSE_STATIC): "Move and stretch before continuing.",
}

_GENERIC_FEEDBACK = {
    LOW: None,
    MEDIUM: "Adjust your posture and maintain a neutral working position.",
    HIGH: "Unsafe posture detected. Correct posture immediately if safe.",
}


def build_feedback_message(severity: str, cause: str) -> Optional[str]:
    """The spec sheet's feedback line alone (no cause clause) -- meant for
    continuous UI display at any severity. Good/low posture gets no
    feedback, matching the sheet's Posture Level 1 row.
    """
    if severity == LOW:
        return None
    return _CAUSE_FEEDBACK.get((severity, cause)) or _GENERIC_FEEDBACK[severity]


def build_voice_message(severity: str, cause: str) -> Optional[str]:
    """Spoken version: names WHY (the cause) before the action line, e.g.
    "High risk posture detected due to forward bending. Unsafe posture
    detected. Stand upright and reposition." Only meaningful for HIGH:
    that's the only level posture_feedback.py's streak trigger ever speaks
    (Medium never triggers audible feedback, by course-wide spec).
    """
    feedback = build_feedback_message(severity, cause)
    if feedback is None:
        return None
    cause_label = _CAUSE_LABELS.get(cause, _CAUSE_LABELS[CAUSE_UNKNOWN])
    severity_word = "High risk" if severity == HIGH else "Elevated"
    return f"{severity_word} posture detected due to {cause_label}. {feedback}"
