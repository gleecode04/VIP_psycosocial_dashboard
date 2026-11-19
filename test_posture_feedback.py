"""Tests for the posture-risk feedback trigger state machine.

Run directly (python test_posture_feedback.py) or with pytest -- both work
because every test is a plain test_* function with no fixtures, matching
test_stress_state.py's style.
"""
import pytest

from posture_feedback import INITIAL_STATE, HIGH_STREAK_THRESHOLD, advance_posture_feedback


def _advance_all(state, participant_id, labels):
    all_events = []
    for label in labels:
        state, events = advance_posture_feedback(state, participant_id, label)
        all_events.extend(events)
    return state, all_events


def test_low_produces_no_events_and_no_streak():
    state, events = advance_posture_feedback(INITIAL_STATE, "P1", "low")
    assert events == []
    assert state.consecutive_high == 0


def test_medium_produces_no_events_and_no_streak():
    state, events = advance_posture_feedback(INITIAL_STATE, "P1", "medium")
    assert events == []
    assert state.consecutive_high == 0


def test_medium_never_triggers_feedback_even_after_many_in_a_row():
    state = INITIAL_STATE
    for _ in range(10):
        state, events = advance_posture_feedback(state, "P1", "medium")
        assert events == []
    assert state.feedback_active is False


def test_one_and_two_highs_produce_no_event():
    state, events = _advance_all(INITIAL_STATE, "P1", ["high", "high"])
    assert events == []
    assert state.consecutive_high == 2
    assert state.feedback_active is False


def test_three_consecutive_highs_trigger_feedback_exactly_once():
    assert HIGH_STREAK_THRESHOLD == 3
    state, events = _advance_all(INITIAL_STATE, "P1", ["high", "high", "high"])
    assert [e.kind for e in events] == ["high_risk_feedback"]
    assert events[0].consecutive_high == 3
    assert state.feedback_active is True


def test_staying_high_after_trigger_does_not_refire():
    state, _ = _advance_all(INITIAL_STATE, "P1", ["high", "high", "high"])
    state, events = advance_posture_feedback(state, "P1", "high")
    assert events == []
    state, events = advance_posture_feedback(state, "P1", "high")
    assert events == []


def test_medium_resets_the_streak_before_three_highs():
    state, events = _advance_all(INITIAL_STATE, "P1", ["high", "high", "medium", "high", "high"])
    # the "medium" reset the streak, so two highs afterward is not yet 3
    assert events == []
    assert state.consecutive_high == 2


def test_low_resets_the_streak_before_three_highs():
    state, events = _advance_all(INITIAL_STATE, "P1", ["high", "high", "low", "high"])
    assert events == []
    assert state.consecutive_high == 1


def test_streak_continues_across_more_than_three_highs():
    state, events = _advance_all(INITIAL_STATE, "P1", ["high"] * 5)
    assert [e.kind for e in events] == ["high_risk_feedback"]
    assert state.consecutive_high == 5


def test_dropping_to_low_after_feedback_fires_resolved_and_resets():
    state, _ = _advance_all(INITIAL_STATE, "P1", ["high", "high", "high"])
    state, events = advance_posture_feedback(state, "P1", "low")
    assert [e.kind for e in events] == ["resolved"]
    assert state.consecutive_high == 0
    assert state.feedback_active is False


def test_dropping_to_medium_after_feedback_fires_resolved_and_resets():
    state, _ = _advance_all(INITIAL_STATE, "P1", ["high", "high", "high"])
    state, events = advance_posture_feedback(state, "P1", "medium")
    assert [e.kind for e in events] == ["resolved"]


def test_a_new_three_high_streak_after_reset_fires_again():
    state, _ = _advance_all(INITIAL_STATE, "P1", ["high", "high", "high"])
    state, _ = advance_posture_feedback(state, "P1", "low")
    state, events = _advance_all(state, "P1", ["high", "high", "high"])
    assert [e.kind for e in events] == ["high_risk_feedback"]


def test_participant_change_resets_streak_with_no_event():
    state, _ = _advance_all(INITIAL_STATE, "P1", ["high", "high"])
    state, events = advance_posture_feedback(state, "P2", "high")
    assert events == []
    assert state.consecutive_high == 1
    assert state.participant_id == "P2"


def test_participant_change_mid_active_feedback_resets_with_no_resolved_event():
    state, _ = _advance_all(INITIAL_STATE, "P1", ["high", "high", "high"])
    state, events = advance_posture_feedback(state, "P2", "low")
    assert events == []
    assert state.feedback_active is False


def test_invalid_label_raises():
    with pytest.raises(ValueError):
        advance_posture_feedback(INITIAL_STATE, "P1", "critical")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
