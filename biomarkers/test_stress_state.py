"""Tests for the stress escalation state machine.

Run directly (python test_stress_state.py) or with pytest -- both work
because every test is a plain test_* function with no fixtures.
"""
from datetime import datetime, timedelta

from stress_state import INITIAL_STATE, advance_state

ESCALATION_MIN = 45
REPEAT_MIN = 45


def _t(minutes_from_zero):
    return datetime(2026, 1, 1, 8, 0, 0) + timedelta(minutes=minutes_from_zero)


def test_low_stress_produces_no_events():
    state, events = advance_state(INITIAL_STATE, "P1", "low", _t(0), ESCALATION_MIN, REPEAT_MIN)
    assert events == []
    assert state.high_since is None


def test_medium_stress_produces_no_events():
    state, events = advance_state(INITIAL_STATE, "P1", "medium", _t(0), ESCALATION_MIN, REPEAT_MIN)
    assert events == []


def test_entering_high_fires_round1_once():
    state, events = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)
    assert [e.kind for e in events] == ["round1"]
    assert state.high_since == _t(0)

    # staying high the next tick, still below threshold: no repeat of round1
    state, events = advance_state(state, "P1", "high", _t(5), ESCALATION_MIN, REPEAT_MIN)
    assert events == []


def test_escalation_fires_exactly_at_threshold_with_correct_elapsed_minutes():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P1", "high", _t(44), ESCALATION_MIN, REPEAT_MIN)
    assert events == [], "should not fire 1 minute before threshold"

    state, events = advance_state(state, "P1", "high", _t(45), ESCALATION_MIN, REPEAT_MIN)
    assert len(events) == 1
    assert events[0].kind == "escalation"
    assert events[0].elapsed_minutes == 45


def test_escalation_does_not_repeat_until_next_interval():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)
    state, _ = advance_state(state, "P1", "high", _t(45), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P1", "high", _t(60), ESCALATION_MIN, REPEAT_MIN)
    assert events == [], "should not re-fire before the repeat interval elapses"


def test_escalation_repeats_after_repeat_interval():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)
    state, _ = advance_state(state, "P1", "high", _t(45), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P1", "high", _t(89), ESCALATION_MIN, REPEAT_MIN)
    assert events == []

    state, events = advance_state(state, "P1", "high", _t(90), ESCALATION_MIN, REPEAT_MIN)
    assert [e.kind for e in events] == ["escalation"]
    assert events[0].elapsed_minutes == 90


def test_custom_repeat_interval_shorter_than_escalation_threshold():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), 45, 15)
    state, _ = advance_state(state, "P1", "high", _t(45), 45, 15)

    state, events = advance_state(state, "P1", "high", _t(60), 45, 15)
    assert [e.kind for e in events] == ["escalation"]


def test_dropping_below_high_fires_episode_ended_and_resets_round1_trigger():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P1", "medium", _t(10), ESCALATION_MIN, REPEAT_MIN)
    assert [e.kind for e in events] == ["episode_ended"]
    assert events[0].elapsed_minutes == 10
    assert state.high_since is None

    state, events = advance_state(state, "P1", "high", _t(12), ESCALATION_MIN, REPEAT_MIN)
    assert [e.kind for e in events] == ["round1"]


def test_staying_low_after_drop_stays_quiet():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)
    state, _ = advance_state(state, "P1", "low", _t(5), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P1", "low", _t(10), ESCALATION_MIN, REPEAT_MIN)
    assert events == []


def test_participant_switch_resets_state_and_fires_round1():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P2", "high", _t(1), ESCALATION_MIN, REPEAT_MIN)
    assert [e.kind for e in events] == ["round1"]
    assert state.participant_id == "P2"
    assert state.high_since == _t(1)


def test_participant_switch_while_not_high_just_resets_quietly():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(0), ESCALATION_MIN, REPEAT_MIN)

    state, events = advance_state(state, "P2", "low", _t(1), ESCALATION_MIN, REPEAT_MIN)
    assert events == []
    assert state.participant_id == "P2"
    assert state.high_since is None


def test_time_going_backward_does_not_crash_or_escalate_early():
    state, _ = advance_state(INITIAL_STATE, "P1", "high", _t(50), ESCALATION_MIN, REPEAT_MIN)

    # a sample arriving "earlier" than high_since (e.g. out-of-order poll)
    state, events = advance_state(state, "P1", "high", _t(10), ESCALATION_MIN, REPEAT_MIN)
    assert events == []


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
