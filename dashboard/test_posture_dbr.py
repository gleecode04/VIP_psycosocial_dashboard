"""Tests for the calibrated DBR posture engine (posture_dbr.py).

Run directly (python test_posture_dbr.py) or with pytest, matching
test_posture_feedback.py's style.
"""
import math

import pytest

from posture_dbr import (
    CAUSE_ASYMMETRIC,
    CAUSE_FORWARD_BENDING,
    CAUSE_NEUTRAL,
    CAUSE_OVERHEAD,
    CAUSE_SIDE_BENDING,
    CAUSE_STATIC,
    CAUSE_TWISTING,
    GOOD_MAX_DEG,
    GravitySample,
    HIGH,
    LOW,
    MEDIUM,
    STATIC_BAD_SECONDS,
    STATIC_CRITICAL_SECONDS,
    axis_delta,
    build_feedback_message,
    build_voice_message,
    calibrate_axes,
    classify_cause,
    classify_tilt_severity,
    compute_window_stats,
    escalate_for_static_duration,
    pitch_deg,
    roll_deg,
    tilt_angle_deg,
)

NEUTRAL = GravitySample(0.0, 0.0, 1.0)  # device "up" axis aligned with gravity at calibration
# A slight forward lean along the device's +x axis, captured right after
# neutral -- gives calibrate_axes() a sagittal plane to work with. Matches
# the existing tests' convention below of using ax for forward-ish bending
# and ay for lateral/side bending.
FORWARD_REF = GravitySample(math.sin(math.radians(15)), 0.0, math.cos(math.radians(15)))
FRAME = calibrate_axes(NEUTRAL, FORWARD_REF)


def _window(samples):
    return list(samples)


# --- tilt_angle_deg -----------------------------------------------------

def test_tilt_is_zero_at_neutral_regardless_of_mount_orientation():
    # The whole point of calibrating against a vector-angle instead of a
    # fixed-axis formula: neutral reads 0 no matter which raw axis gravity
    # happens to sit on for this mount point.
    weird_mount_neutral = GravitySample(0.6, -0.7, 0.39)
    assert tilt_angle_deg(weird_mount_neutral, weird_mount_neutral) == pytest.approx(0.0, abs=1e-9)


def test_tilt_is_90_for_a_right_angle_bend():
    bent = GravitySample(1.0, 0.0, 0.0)
    assert tilt_angle_deg(bent, NEUTRAL) == pytest.approx(90.0, abs=1e-6)


def test_tilt_grows_monotonically_with_forward_lean():
    angles_deg = [0, 10, 30, 45, 60, 90]
    samples = [GravitySample(math.sin(math.radians(a)), 0.0, math.cos(math.radians(a))) for a in angles_deg]
    tilts = [tilt_angle_deg(s, NEUTRAL) for s in samples]
    assert tilts == sorted(tilts)
    for expected, actual in zip(angles_deg, tilts):
        assert actual == pytest.approx(expected, abs=1e-6)


def test_axis_delta_is_zero_at_neutral():
    delta = axis_delta(NEUTRAL, NEUTRAL)
    assert (delta.ax, delta.ay, delta.az) == (0.0, 0.0, 0.0)


# --- calibrate_axes / pitch_deg / roll_deg --------------------------------
# Mentor feedback: Pitch (forward/back lean) should be the PRIMARY severity
# signal; Roll (side-to-side lean) is secondary, cause-only. These tests
# confirm the two-vector calibration actually separates them, instead of
# both being blended into one omnidirectional tilt angle.

def test_pitch_and_roll_are_zero_at_neutral():
    assert pitch_deg(NEUTRAL, FRAME) == pytest.approx(0.0, abs=1e-6)
    assert roll_deg(NEUTRAL, FRAME) == pytest.approx(0.0, abs=1e-6)


def test_pure_forward_lean_is_pitch_only():
    # Along the sagittal plane (ay = 0), same direction as the forward
    # calibration reference -- pitch should read the lean angle, roll ~0.
    forward_lean = GravitySample(math.sin(math.radians(40)), 0.0, math.cos(math.radians(40)))
    assert pitch_deg(forward_lean, FRAME) == pytest.approx(40.0, abs=1e-6)
    assert roll_deg(forward_lean, FRAME) == pytest.approx(0.0, abs=1e-6)


def test_pure_side_lean_is_roll_only():
    # Rotated purely about the forward axis (ax = 0) -- pure side-to-side
    # lean, no forward/back component.
    side_lean = GravitySample(0.0, math.sin(math.radians(30)), math.cos(math.radians(30)))
    assert roll_deg(side_lean, FRAME) == pytest.approx(30.0, abs=1e-6)
    assert pitch_deg(side_lean, FRAME) == pytest.approx(0.0, abs=1e-6)


def test_combined_lean_has_both_nonzero_but_severity_is_pitch_only():
    combined = GravitySample(0.5, 0.5, math.sqrt(1 - 0.5 ** 2 - 0.5 ** 2))
    pitch, roll = pitch_deg(combined, FRAME), roll_deg(combined, FRAME)
    assert pitch > 0
    assert roll > 0

    window = [combined] * 10
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    # Severity comes from pitch's magnitude alone, which is smaller than the
    # blended omnidirectional tilt angle -- exactly the distinction a single
    # tilt_angle_deg() could never make.
    assert stats.max_tilt_deg > abs(stats.max_pitch_deg)
    assert classify_tilt_severity(abs(stats.max_pitch_deg)) == classify_tilt_severity(abs(pitch))


def test_calibrate_axes_handles_degenerate_forward_reference():
    # Forward-lean calibration pose came back ~identical to neutral (e.g.
    # the wearer didn't actually lean) -- must not crash, and neutral still
    # reads as neutral.
    frame = calibrate_axes(NEUTRAL, GravitySample(0.0, 0.0, 1.0001))
    assert pitch_deg(NEUTRAL, frame) == pytest.approx(0.0, abs=1e-3)
    assert roll_deg(NEUTRAL, frame) == pytest.approx(0.0, abs=1e-3)


# --- classify_tilt_severity ----------------------------------------------

def test_severity_boundaries():
    assert classify_tilt_severity(0.0) == LOW
    assert classify_tilt_severity(20.0) == LOW
    assert classify_tilt_severity(20.1) == MEDIUM
    assert classify_tilt_severity(45.0) == MEDIUM
    assert classify_tilt_severity(45.1) == HIGH


# --- classify_cause -------------------------------------------------------

def _tilted(ax, ay, az):
    return GravitySample(ax, ay, az)


def test_neutral_window_has_neutral_cause():
    window = [NEUTRAL] * 10
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert classify_cause(stats) == CAUSE_NEUTRAL


def test_sustained_forward_lean_is_forward_bending():
    # A 50-degree-ish lean held steady (low within-window variation).
    lean = GravitySample(0.76, 0.0, 0.64)
    window = [lean] * 10
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert stats.max_tilt_deg > 45
    assert classify_cause(stats) == CAUSE_FORWARD_BENDING


def test_oscillating_ax_and_ay_is_twisting():
    window = []
    for i in range(10):
        sign = 1 if i % 2 == 0 else -1
        window.append(GravitySample(0.5 * sign, 0.5 * sign, 0.7))
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert classify_cause(stats) == CAUSE_TWISTING


def test_a_single_steady_window_is_not_static_cause():
    # Static Posture is a multi-minute DURATION concept (see
    # escalate_for_static_duration), not something one steady ~1s window
    # should claim -- otherwise every calm, deliberate bend would get
    # mislabeled "static" immediately. A single held window should still
    # resolve to a dominant-axis cause like forward_bending.
    held = GravitySample(0.4, 0.0, 0.9)
    window = [held] * 10
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert stats.max_tilt_deg > GOOD_MAX_DEG
    assert classify_cause(stats) != CAUSE_STATIC


def test_high_ay_variation_is_overhead_work():
    window = []
    for i in range(10):
        wobble = 0.3 if i % 2 == 0 else -0.3
        window.append(GravitySample(0.2, wobble, 0.85))
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert classify_cause(stats) == CAUSE_OVERHEAD


def test_diffuse_variation_with_no_dominant_axis_is_asymmetric():
    window = []
    for i in range(10):
        jitter = 0.05 if i % 2 == 0 else -0.05
        window.append(GravitySample(0.3 + jitter, 0.3 + jitter, 0.8 + jitter))
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert classify_cause(stats) == CAUSE_ASYMMETRIC


def test_sustained_side_lean_is_side_bending_via_roll_deg():
    # Held steady, dominant roll with ~no pitch -- classify_cause's
    # side_bending branch must come from roll_deg/pitch_deg directly now,
    # not the old mean_day/mean_daz raw-delta ratio.
    side_lean = GravitySample(0.0, math.sin(math.radians(30)), math.cos(math.radians(30)))
    window = [side_lean] * 10
    stats = compute_window_stats(window, NEUTRAL, FRAME)
    assert stats.max_tilt_deg > GOOD_MAX_DEG
    assert abs(stats.mean_pitch_deg) < 1.0
    assert abs(stats.mean_roll_deg) > GOOD_MAX_DEG
    assert classify_cause(stats) == CAUSE_SIDE_BENDING


# --- escalate_for_static_duration -----------------------------------------

def test_no_escalation_below_duration_threshold():
    severity, cause = escalate_for_static_duration(LOW, CAUSE_NEUTRAL, STATIC_BAD_SECONDS - 1)
    assert (severity, cause) == (LOW, CAUSE_NEUTRAL)


def test_escalates_neutral_hold_to_medium_after_five_minutes():
    severity, cause = escalate_for_static_duration(LOW, CAUSE_NEUTRAL, STATIC_BAD_SECONDS)
    assert severity == MEDIUM
    assert cause == CAUSE_STATIC


def test_escalates_to_high_after_ten_minutes():
    severity, cause = escalate_for_static_duration(LOW, CAUSE_NEUTRAL, STATIC_CRITICAL_SECONDS)
    assert severity == HIGH
    assert cause == CAUSE_STATIC


def test_active_bad_angle_keeps_its_own_cause_even_if_also_sustained():
    severity, cause = escalate_for_static_duration(HIGH, CAUSE_FORWARD_BENDING, STATIC_CRITICAL_SECONDS)
    assert severity == HIGH
    assert cause == CAUSE_FORWARD_BENDING  # not overwritten to static_posture


def test_never_downgrades():
    severity, cause = escalate_for_static_duration(HIGH, CAUSE_TWISTING, 0)
    assert severity == HIGH
    assert cause == CAUSE_TWISTING


# --- feedback / voice messages --------------------------------------------

def test_low_severity_has_no_feedback():
    assert build_feedback_message(LOW, CAUSE_NEUTRAL) is None
    assert build_voice_message(LOW, CAUSE_NEUTRAL) is None


def test_medium_forward_bending_uses_cause_specific_text():
    assert build_feedback_message(MEDIUM, CAUSE_FORWARD_BENDING) == (
        "Straighten your back and lift your head slightly."
    )


def test_high_twisting_uses_cause_specific_text():
    assert build_feedback_message(HIGH, CAUSE_TWISTING) == "Stop and reposition before continuing."


def test_unknown_cause_falls_back_to_generic_text():
    assert build_feedback_message(HIGH, "some_future_cause") == (
        "Unsafe posture detected. Correct posture immediately if safe."
    )


def test_voice_message_names_the_cause():
    msg = build_voice_message(HIGH, CAUSE_FORWARD_BENDING)
    assert "forward bending" in msg
    assert msg.endswith("Unsafe posture detected. Stand upright and reposition.")


def test_voice_message_none_when_no_feedback():
    assert build_voice_message(LOW, CAUSE_NEUTRAL) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
