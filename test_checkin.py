"""Tests for the closed-loop check-in ask/expire state machine.

Run directly (python test_checkin.py) or with pytest -- both work because
every test is a plain test_* function with no fixtures (matches
test_stress_state.py / test_posture_dbr.py's convention).
"""
from datetime import datetime, timedelta

from checkin import (
    INITIAL_STATE,
    MENTAL_LOAD,
    POSTURE,
    STRESS,
    advance_checkin,
    derive_answer,
    lane_agreement,
)

MIN_GAP = 15
MAX_AGE = 30
EXPIRE = 10


def _t(minutes_from_zero):
    return datetime(2026, 1, 1, 8, 0, 0) + timedelta(minutes=minutes_from_zero)


def _advance(state, participant_id, stress_label, mental_label, sample_minute, now_minute):
    return advance_checkin(state, participant_id, stress_label, mental_label, _t(sample_minute), now=_t(now_minute))


# -- deciding whether to ask -------------------------------------------------

def test_low_and_medium_never_ask():
    state, events = _advance(INITIAL_STATE, "P1", "low", "low", 0, 0)
    assert events == []
    state, events = _advance(state, "P1", "medium", "medium", 1, 1)
    assert events == []


def test_high_stress_opens_a_question():
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, 0)
    assert [e.kind for e in events] == ["ask"]
    assert events[0].question.trigger_source == STRESS
    assert state.pending is not None


def test_high_mental_load_opens_a_question():
    state, events = _advance(INITIAL_STATE, "P1", "low", "high", 0, 0)
    assert [e.kind for e in events] == ["ask"]
    assert events[0].question.trigger_source == MENTAL_LOAD


def test_both_high_at_once_prefers_stress():
    state, events = _advance(INITIAL_STATE, "P1", "high", "high", 0, 0)
    assert events[0].question.trigger_source == STRESS


def test_only_one_question_open_at_a_time():
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, 0)
    assert len(events) == 1
    first_id = events[0].question.question_id

    # Still high a minute later, well within EXPIRE_MIN and MIN_GAP_MIN:
    # no second question while one is already open.
    state, events = _advance(state, "P1", "high", "low", 1, 1)
    assert events == []
    assert state.pending.question_id == first_id


def test_minimum_gap_between_questions_is_enforced():
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, 0)
    first_id = events[0].question.question_id

    # Answer/clear it immediately by expiring it, then try again too soon.
    state, _ = _advance(state, "P1", "high", "low", 0, MIN_GAP + EXPIRE - 1)
    # right after expiry, still inside the 15-min gap window from last_asked_at (t=0)
    state2, events2 = _advance(state, "P1", "high", "low", 0, MIN_GAP - 1)
    assert events2 == []


def test_question_expires_after_expire_minutes_with_no_answer():
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, 0)
    question = events[0].question

    state, events = _advance(state, "P1", "high", "low", 0, EXPIRE - 1)
    assert events == [], "should not expire 1 minute early"
    assert state.pending is not None

    state, events = _advance(state, "P1", "high", "low", 0, EXPIRE)
    assert [e.kind for e in events] == ["expired"]
    assert events[0].question.question_id == question.question_id
    assert state.pending is None


def test_new_question_can_open_right_after_gap_elapses_post_expiry():
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, 0)

    # Expires at t=EXPIRE, freeing the slot; MIN_GAP (15) > EXPIRE (10) here,
    # so the gap from the original ask time is still the binding constraint.
    state, events = _advance(state, "P1", "high", "low", MIN_GAP, MIN_GAP)
    assert [e.kind for e in events] == ["expired", "ask"]


def test_too_old_a_moment_is_not_asked_about():
    # sample_time is MAX_AGE+1 minutes in the past relative to "now"
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, MAX_AGE + 1)
    assert events == []
    assert state.pending is None


def test_moment_exactly_at_max_age_is_still_askable():
    state, events = _advance(INITIAL_STATE, "P1", "high", "low", 0, MAX_AGE)
    assert [e.kind for e in events] == ["ask"]


def test_high_posture_opens_a_question():
    # Posture has no stress/mental_load reading at all (it's a separate
    # bridge/participant stream) -- both are None, only posture_label is set.
    state, events = advance_checkin(
        INITIAL_STATE, "LIVE-POLAR-01", None, None, _t(0), now=_t(0), posture_label="high",
    )
    assert [e.kind for e in events] == ["ask"]
    assert events[0].question.trigger_source == POSTURE
    assert events[0].question.posture_label_at_ask == "high"


def test_posture_is_lowest_priority_when_multiple_lanes_trigger():
    # Same tick, all three ask-worthy -- stress still wins (guide's own
    # ordering), posture is checked last since it's not in the guide at all.
    state, events = advance_checkin(
        INITIAL_STATE, "P1", "high", "high", _t(0), now=_t(0), posture_label="high",
    )
    assert events[0].question.trigger_source == STRESS

    state, events = advance_checkin(
        INITIAL_STATE, "P1", "low", "high", _t(0), now=_t(0), posture_label="high",
    )
    assert events[0].question.trigger_source == MENTAL_LOAD


def test_participant_switch_resets_tracking():
    state, _ = _advance(INITIAL_STATE, "P1", "high", "low", 0, 0)
    state, events = _advance(state, "P2", "low", "low", 1, 1)
    assert events == []
    assert state.participant_id == "P2"
    assert state.pending is None

    # P2 should be immediately askable (no inherited last_asked_at from P1)
    state, events = _advance(state, "P2", "high", "low", 1, 1)
    assert [e.kind for e in events] == ["ask"]


# -- deriving an answer -------------------------------------------------------

def test_derive_answer_yes_no():
    assert derive_answer("Yes") is True
    assert derive_answer("No") is False
    assert derive_answer(None) is None
    assert derive_answer("") is None
    assert derive_answer("garbage") is None


def test_derive_answer_a_little_follows_setting():
    assert derive_answer("A little", little_counts_as="yes") is True
    assert derive_answer("A little", little_counts_as="no") is False


# -- agreement -----------------------------------------------------------------

def test_lane_agreement_confirmed_vs_false_alarm():
    assert lane_agreement(STRESS, "high", True) is True
    assert lane_agreement(STRESS, "high", False) is False


def test_lane_agreement_none_when_no_answer():
    assert lane_agreement(STRESS, "high", None) is None


def test_lane_agreement_for_untriggered_lane_still_computable():
    # mental_load was "medium" (not ask-worthy) but we still got an answer
    # for it (both questions are always asked on the one card); the worker
    # saying "no" there agrees with the prediction not having been high.
    assert lane_agreement(MENTAL_LOAD, "medium", False) is True
    assert lane_agreement(MENTAL_LOAD, "medium", True) is False


def test_posture_lane_agreement():
    assert lane_agreement(POSTURE, "high", True) is True
    assert lane_agreement(POSTURE, "high", False) is False
    assert lane_agreement(POSTURE, "low", False) is True  # "no, wasn't bad posture" confirms a low call


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
