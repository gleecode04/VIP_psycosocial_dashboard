"""Pure state machine for the construction-worker stress escalation speech
system.

No I/O here on purpose: no Flask, no requests, no pyttsx3. That keeps the
escalation logic independently testable (see test_stress_state.py) without
a running poller, a network call, or an audio engine.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Tuple

from risk_score import band_for

HIGH_LABEL = "high"


@dataclass(frozen=True)
class StressState:
    participant_id: Optional[str] = None
    high_since: Optional[datetime] = None
    next_escalation_minutes: float = 0.0


@dataclass(frozen=True)
class StressEvent:
    kind: str  # "round1" or "escalation"
    elapsed_minutes: float


INITIAL_STATE = StressState()


def get_stress_label(sample: dict) -> Optional[str]:
    """The overall risk score (risk_score.py) is preferred when the poller
    provides one, because that's the exact same number the dashboard's
    trend-chart threshold line is drawn against -- using it here instead of
    the raw categorical ML label keeps the TTS trigger and the on-screen
    threshold crossing in sync (they used to be computed independently on
    different schedules, which made the spoken message appear to fire a
    tick ahead of the graph).

    Falls back to the categorical label (frustration, then mental demand as
    a first-minute fallback before the frustration model has two samples to
    work with -- see biomarker_runtime.py) when no risk score is present.
    """
    predictions = sample.get("predictions") or {}

    risk = predictions.get("risk") or {}
    if risk.get("overall") is not None:
        return band_for(risk["overall"])

    frustration = predictions.get("frustration") or {}
    if frustration.get("ready") and frustration.get("label"):
        return frustration["label"]

    mental_demand = predictions.get("mental_demand") or {}
    if mental_demand.get("ready") and mental_demand.get("label"):
        return mental_demand["label"]

    return None


def get_sample_time(sample: dict) -> datetime:
    """Use the biomarker's own timestamp, not wall-clock time, so an
    "you've been stressed for N minutes" announcement reflects the
    participant's actual timeline. This also lets a demo/mock poller
    fast-forward its simulated clock to compress a 45-minute escalation
    into a short live demo.
    """
    for key in ("eda_time", "temp_time", "pulse_time", "updated_at"):
        value = sample.get(key)
        if not value:
            continue
        try:
            text = value.replace("Z", "+00:00")
            return datetime.fromisoformat(text).replace(tzinfo=None)
        except (ValueError, TypeError):
            continue
    return datetime.now()


def advance_state(
    state: StressState,
    participant_id: str,
    stress_label: str,
    sample_time: datetime,
    escalation_minutes: float,
    repeat_minutes: float,
) -> Tuple[StressState, List[StressEvent]]:
    """One pure transition. Returns the next state and zero or more events
    to speak for this tick.

    - Entering "high" from anything else fires a single "round1" event.
    - Staying continuously "high" fires an "escalation" event once the
      elapsed time (by sample_time, not wall clock) crosses
      escalation_minutes, then again every repeat_minutes after that.
    - Dropping out of "high" resets tracking, so the next entry into "high"
      fires "round1" again.
    - A participant change resets tracking.
    """
    if participant_id != state.participant_id:
        state = StressState(participant_id=participant_id)

    if stress_label != HIGH_LABEL:
        if state.high_since is None:
            return state, []
        elapsed_minutes = (sample_time - state.high_since).total_seconds() / 60.0
        return StressState(participant_id=participant_id), [StressEvent("episode_ended", elapsed_minutes)]

    if state.high_since is None:
        new_state = StressState(
            participant_id=participant_id,
            high_since=sample_time,
            next_escalation_minutes=escalation_minutes,
        )
        return new_state, [StressEvent("round1", 0.0)]

    elapsed_minutes = (sample_time - state.high_since).total_seconds() / 60.0
    if elapsed_minutes >= state.next_escalation_minutes:
        # Schedule the next reminder repeat_minutes from *now*, not from the
        # original threshold, so a long gap between polls/ticks can't queue
        # up a burst of back-to-back catch-up announcements.
        new_state = replace(state, next_escalation_minutes=elapsed_minutes + repeat_minutes)
        return new_state, [StressEvent("escalation", elapsed_minutes)]

    return state, []
