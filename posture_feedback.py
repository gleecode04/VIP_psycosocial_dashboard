"""Pure state machine for the posture-risk feedback trigger.

Assignment-wide requirement (every team, every pathway):
  - Predict Low, Medium, and High risk levels.
  - Never trigger feedback for Medium-level risk.
  - Trigger High-risk feedback only after three consecutive High
    predictions; reset the streak if a Low or Medium prediction occurs.

No I/O here on purpose, mirroring stress_state.py: this keeps the trigger
rule independently unit-testable without a running poller, a trained model,
or a TTS engine. posture_live_replay.py is the I/O loop around it.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
VALID_LABELS = (LOW, MEDIUM, HIGH)

HIGH_STREAK_THRESHOLD = 3


@dataclass(frozen=True)
class PostureFeedbackState:
    participant_id: Optional[str] = None
    consecutive_high: int = 0
    feedback_active: bool = False


@dataclass(frozen=True)
class PostureFeedbackEvent:
    kind: str  # "high_risk_feedback" or "resolved"
    consecutive_high: int


INITIAL_STATE = PostureFeedbackState()


def advance_posture_feedback(
    state: PostureFeedbackState,
    participant_id: str,
    risk_label: str,
) -> Tuple[PostureFeedbackState, List[PostureFeedbackEvent]]:
    """One pure transition on a single new Low/Medium/High prediction.

    - Medium NEVER triggers feedback (assignment requirement) and resets the
      High streak exactly like Low does. A Medium reading isn't "safe", but
      it also isn't the sustained-High pattern this feedback exists to
      catch, so -- per spec -- it's a streak-breaker, not an escalation.
    - High increments a streak counter; feedback fires once, the moment the
      streak reaches HIGH_STREAK_THRESHOLD (3), not on every subsequent High
      tick. This mirrors stress_state.py's fire-once "round1" pattern, so a
      sustained-high episode produces one feedback event instead of a flood
      of repeats.
    - Dropping back to Low or Medium after feedback was fired re-arms the
      trigger and emits a "resolved" event (so a caller -- e.g. the live
      dashboard -- can clear a "high risk" banner); the next 3-High streak
      then fires again from zero.
    - A participant change resets all tracking, with no event.
    """
    if risk_label not in VALID_LABELS:
        raise ValueError(f"risk_label must be one of {VALID_LABELS}, got {risk_label!r}")

    if participant_id != state.participant_id:
        state = PostureFeedbackState(participant_id=participant_id)

    if risk_label != HIGH:
        if state.consecutive_high == 0 and not state.feedback_active:
            return state, []
        events = [PostureFeedbackEvent("resolved", 0)] if state.feedback_active else []
        return PostureFeedbackState(participant_id=participant_id), events

    streak = state.consecutive_high + 1
    if streak >= HIGH_STREAK_THRESHOLD and not state.feedback_active:
        new_state = PostureFeedbackState(
            participant_id=participant_id, consecutive_high=streak, feedback_active=True
        )
        return new_state, [PostureFeedbackEvent("high_risk_feedback", streak)]

    new_state = PostureFeedbackState(
        participant_id=participant_id, consecutive_high=streak, feedback_active=state.feedback_active
    )
    return new_state, []
