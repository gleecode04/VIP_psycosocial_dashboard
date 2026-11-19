"""Closed-loop worker check-in logic -- implements steps 5 ("Decide whether
to ask") and part of step 6/7 of Closed loop Guide.md.

No I/O here on purpose, mirroring stress_state.py and posture_feedback.py:
this keeps the ask/expire timing rules independently unit-testable (see
test_checkin.py) without a running poller, a Flask route, or a CSV file.
checkin_bridge.py is the I/O layer around it -- it owns storage, the
feature-snapshot attached to each question, and the HTTP-facing API that
serve_dashboard.py exposes.

Settings below are deliberately plain module constants (not a class), so a
grader/instructor can change them in one place, exactly as the guide's
"Design choices and settings" table describes ("change them in checkin.py").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Tuple

STRESS = "stress"
MENTAL_LOAD = "mental_load"

# Which categorical labels count as "ask-worthy" per lane. The guide's table
# says "mental load High or Very High" -- but the trained mental_demand
# model (models/model_metadata.json) only ever emits low/medium/high, there
# is no "Very High" level anywhere in this codebase. ASK_LOAD therefore
# gates on the closest existing equivalent ("high") rather than inventing a
# fourth class that nothing produces.
ASK_STRESS = frozenset(os.environ.get("CHECKIN_ASK_STRESS", "high").split(","))
ASK_LOAD = frozenset(os.environ.get("CHECKIN_ASK_LOAD", "high").split(","))

MIN_GAP_MIN = float(os.environ.get("CHECKIN_MIN_GAP_MIN", "15"))
MAX_AGE_MIN = float(os.environ.get("CHECKIN_MAX_AGE_MIN", "30"))
EXPIRE_MIN = float(os.environ.get("CHECKIN_EXPIRE_MIN", "10"))

# "A little" counts as stressed/working-hard by default -- the guide's
# table lists this as a --little yes|no choice; "yes" is the more
# conservative default for a safety-facing system (treats the ambiguous
# middle answer as a signal worth keeping, not noise to discard).
LITTLE_COUNTS_AS = os.environ.get("CHECKIN_LITTLE_COUNTS_AS", "yes").strip().lower()
if LITTLE_COUNTS_AS not in ("yes", "no"):
    raise ValueError(f"CHECKIN_LITTLE_COUNTS_AS must be 'yes' or 'no', got {LITTLE_COUNTS_AS!r}")

ANSWER_WEIGHT = int(os.environ.get("CHECKIN_ANSWER_WEIGHT", "3"))
MIN_ANSWERS_TO_RETRAIN = int(os.environ.get("CHECKIN_MIN_ANSWERS", "20"))
MAX_BALANCED_ACCURACY_DROP = float(os.environ.get("CHECKIN_MAX_ACC_DROP", "0.02"))


@dataclass(frozen=True)
class PendingQuestion:
    """One open "Quick check" card. Carries just enough to time it out and
    label why it was asked -- the sensor-feature snapshot used for later
    retraining lives in checkin_bridge.py, not here, so this stays plain
    old timing/label data with no opinion about what a "feature" is.
    """
    question_id: str
    participant_id: str
    trigger_source: str  # STRESS or MENTAL_LOAD -- whichever crossed its ask threshold
    stress_label_at_ask: Optional[str]
    mental_label_at_ask: Optional[str]
    moment_time: datetime
    asked_at: datetime


@dataclass(frozen=True)
class CheckinState:
    participant_id: Optional[str] = None
    pending: Optional[PendingQuestion] = None
    last_asked_at: Optional[datetime] = None


@dataclass(frozen=True)
class CheckinEvent:
    kind: str  # "ask" or "expired"
    question: PendingQuestion


INITIAL_STATE = CheckinState()


def _trigger(stress_label: Optional[str], mental_label: Optional[str]) -> Optional[str]:
    """Which lane (if any) crossed its ask threshold this tick. Stress is
    checked first when both cross in the same tick, matching the guide's
    own listing order ("Stress High; mental load High or Very High") --
    only one question card is ever open at a time, so a tie has to pick one.
    """
    if stress_label in ASK_STRESS:
        return STRESS
    if mental_label in ASK_LOAD:
        return MENTAL_LOAD
    return None


def advance_checkin(
    state: CheckinState,
    participant_id: str,
    stress_label: Optional[str],
    mental_label: Optional[str],
    sample_time: datetime,
    now: Optional[datetime] = None,
) -> Tuple[CheckinState, List[CheckinEvent]]:
    """One pure transition for a single new prediction tick.

    - A participant change resets tracking, with no event (mirrors
      stress_state.py / posture_feedback.py).
    - A stale open question (older than EXPIRE_MIN) is expired first, which
      frees the "one question open at a time" slot in the same tick.
    - A new question is only opened if: no question is currently open, at
      least MIN_GAP_MIN has passed since the last one was asked, the
      triggering sample is no older than MAX_AGE_MIN (don't ask about a
      moment nobody will remember), and a lane actually crossed its
      threshold this tick.
    """
    now = now or datetime.now()

    if participant_id != state.participant_id:
        state = CheckinState(participant_id=participant_id)

    events: List[CheckinEvent] = []

    if state.pending is not None:
        age_min = (now - state.pending.asked_at).total_seconds() / 60.0
        if age_min >= EXPIRE_MIN:
            events.append(CheckinEvent("expired", state.pending))
            state = replace(state, pending=None)

    if state.pending is not None:
        return state, events

    if state.last_asked_at is not None:
        gap_min = (now - state.last_asked_at).total_seconds() / 60.0
        if gap_min < MIN_GAP_MIN:
            return state, events

    moment_age_min = (now - sample_time).total_seconds() / 60.0
    if moment_age_min > MAX_AGE_MIN:
        return state, events

    source = _trigger(stress_label, mental_label)
    if source is None:
        return state, events

    question = PendingQuestion(
        question_id=f"{participant_id}:{int(now.timestamp() * 1000)}",
        participant_id=participant_id,
        trigger_source=source,
        stress_label_at_ask=stress_label,
        mental_label_at_ask=mental_label,
        moment_time=sample_time,
        asked_at=now,
    )
    state = replace(state, pending=question, last_asked_at=now)
    events.append(CheckinEvent("ask", question))
    return state, events


def derive_answer(raw_answer: Optional[str], little_counts_as: str = LITTLE_COUNTS_AS) -> Optional[bool]:
    """Maps a raw "Yes" / "A little" / "No" answer to a boolean label per
    the --little setting. None means unanswered/unrecognized (e.g. the
    question expired with no answer at all).
    """
    text = (raw_answer or "").strip().lower()
    if text in ("yes", "y"):
        return True
    if text in ("a little", "a_little", "little"):
        return little_counts_as == "yes"
    if text in ("no", "n"):
        return False
    return None


def lane_agreement(lane: str, label_at_ask: Optional[str], derived_answer: Optional[bool]) -> Optional[bool]:
    """Did the worker's answer confirm this lane's prediction at ask time?
    True = confirmed, False = the worker says this was a false alarm,
    None = no usable answer (expired, or the lane's question wasn't
    answered) -- not counted either way in agreement stats.
    """
    if derived_answer is None:
        return None
    ask_set = ASK_STRESS if lane == STRESS else ASK_LOAD
    predicted_ask_worthy = label_at_ask in ask_set
    return derived_answer == predicted_ask_worthy
