"""I/O layer around checkin.py -- implements steps 6 ("Ask"), 7 ("Store")
and 8 ("Measure agreement") of Closed loop Guide.md.

Mirrors posture_live_bridge.py's role relative to posture_dbr.py: the pure
module decides *when* to ask and *whether an answer agrees*; this module
owns the open question's feature snapshot, persistence (checkins.csv +
checkins.json per participant, under live_output/), and the small
aggregate stats the Research view reads.

live_dashboard_backend.py calls consider_sample()/check_expiry() on every
poll tick (it already computes the frustration/mental_demand labels this
needs); posture_live_bridge.py calls the same consider_sample()/
check_expiry() for its own posture_label (see checkin.py's module docstring
for why posture only gets steps 5-8, not retraining); serve_dashboard.py
exposes the rest over HTTP.
"""
from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import checkin

ROOT = Path(__file__).resolve().parent
LIVE_OUTPUT_DIR = ROOT / "live_output"

CSV_FIELDS = [
    "question_id", "participant_id", "trigger_source",
    "stress_label_at_ask", "mental_label_at_ask", "posture_label_at_ask",
    "moment_time", "asked_at", "answered_at",
    "stressed_raw_answer", "mental_raw_answer", "posture_raw_answer", "free_text",
    "stressed_derived", "mental_derived", "posture_derived",
    "stressed_agreed", "mental_agreed", "posture_agreed",
    "is_self_report",
]


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(timespec="seconds") if value is not None else None


class CheckinBridge:
    def __init__(self, output_dir: Path = LIVE_OUTPUT_DIR):
        self._lock = threading.Lock()
        self._output_dir = output_dir
        self._state: dict[str, checkin.CheckinState] = {}
        self._pending_features: dict[str, dict[str, Any]] = {}  # question_id -> feature snapshot

    # -- step 5/6: deciding + opening a question ---------------------------

    def consider_sample(
        self,
        participant_id: str,
        stress_label: Optional[str],
        mental_label: Optional[str],
        sample_time: datetime,
        features_snapshot: dict[str, Any],
        posture_label: Optional[str] = None,
    ) -> Optional[dict]:
        """Called once per new prediction tick. Returns the newly opened
        question (as a dict, ready for the dashboard) if one was asked,
        else None. Also persists any question that expired this tick.

        posture_label is optional and defaults to None -- the biomarkers
        poll loop (live_dashboard_backend.py) never has one to pass; the
        posture bridge (posture_live_bridge.py) passes its own risk label
        here and leaves stress_label/mental_label None.
        """
        with self._lock:
            state = self._state.get(participant_id, checkin.INITIAL_STATE)
            state, events = checkin.advance_checkin(
                state, participant_id, stress_label, mental_label, sample_time, posture_label=posture_label,
            )
            self._state[participant_id] = state

            opened = None
            for event in events:
                if event.kind == "ask":
                    self._pending_features[event.question.question_id] = features_snapshot
                    opened = self._question_to_dict(event.question)
                elif event.kind == "expired":
                    self._persist_resolved(event.question, answered_at=None, stressed_raw=None,
                                            mental_raw=None, posture_raw=None, free_text=None, is_self_report=False)
                    self._pending_features.pop(event.question.question_id, None)
            return opened

    def check_expiry(self, now: Optional[datetime] = None) -> None:
        """Call on every poll tick (even when no new sample arrived) so a
        10-minute expiry fires promptly instead of waiting for the next
        changed reading."""
        with self._lock:
            for participant_id, state in list(self._state.items()):
                if state.pending is None:
                    continue
                # Re-run the same transition with the *same* labels/moment
                # it was asked with -- only the expiry clock matters here;
                # advance_checkin() is a no-op for asking again because a
                # question is already (still) open.
                q = state.pending
                state, events = checkin.advance_checkin(
                    state, participant_id, q.stress_label_at_ask, q.mental_label_at_ask, q.moment_time, now=now,
                    posture_label=q.posture_label_at_ask,
                )
                self._state[participant_id] = state
                for event in events:
                    if event.kind == "expired":
                        self._persist_resolved(event.question, answered_at=None, stressed_raw=None,
                                                mental_raw=None, posture_raw=None, free_text=None, is_self_report=False)
                        self._pending_features.pop(event.question.question_id, None)

    # -- dashboard reads -----------------------------------------------------

    def get_pending(self, participant_id: str) -> Optional[dict]:
        with self._lock:
            state = self._state.get(participant_id)
            if state is None or state.pending is None:
                return None
            return self._question_to_dict(state.pending)

    def _question_to_dict(self, question: checkin.PendingQuestion) -> dict:
        return {
            "question_id": question.question_id,
            "participant_id": question.participant_id,
            "trigger_source": question.trigger_source,
            "stress_label_at_ask": question.stress_label_at_ask,
            "mental_label_at_ask": question.mental_label_at_ask,
            "posture_label_at_ask": question.posture_label_at_ask,
            "moment_time": _iso(question.moment_time),
            "asked_at": _iso(question.asked_at),
        }

    # -- step 6/7: answering + storing --------------------------------------

    def answer(self, question_id: str, stressed_answer: Optional[str], mental_answer: Optional[str],
               free_text: Optional[str], posture_answer: Optional[str] = None) -> Optional[dict]:
        """Resolves the open question matching question_id, if any is still
        open and matches. Returns the stored record, or None if the
        question is unknown (already expired/answered, or never existed --
        e.g. the dashboard was showing a stale card)."""
        with self._lock:
            for participant_id, state in list(self._state.items()):
                if state.pending is not None and state.pending.question_id == question_id:
                    question = state.pending
                    self._state[participant_id] = checkin.CheckinState(
                        participant_id=participant_id, last_asked_at=state.last_asked_at,
                    )
                    record = self._persist_resolved(
                        question, answered_at=datetime.now(), stressed_raw=stressed_answer,
                        mental_raw=mental_answer, posture_raw=posture_answer, free_text=free_text,
                        is_self_report=False,
                    )
                    self._pending_features.pop(question_id, None)
                    return record
            return None

    def self_report(self, participant_id: str, stressed_answer: Optional[str], mental_answer: Optional[str],
                     free_text: Optional[str], stress_label_now: Optional[str], mental_label_now: Optional[str],
                     features_snapshot: dict[str, Any], posture_answer: Optional[str] = None,
                     posture_label_now: Optional[str] = None) -> dict:
        """The "How do you feel? Tell CHAP" button -- always recorded,
        never gated by the ask-rate limits (it's the worker reaching out,
        not CHAP interrupting). Not counted in asked/answered agreement
        stats (see Closed loop Guide.md's closing note: self-reports fill
        the gap where CHAP never asked, they aren't a check on a
        prediction CHAP made)."""
        now = datetime.now()
        question = checkin.PendingQuestion(
            question_id=f"{participant_id}:self:{int(now.timestamp() * 1000)}",
            participant_id=participant_id,
            trigger_source="self_report",
            stress_label_at_ask=stress_label_now,
            mental_label_at_ask=mental_label_now,
            moment_time=now,
            asked_at=now,
            posture_label_at_ask=posture_label_now,
        )
        with self._lock:
            self._pending_features[question.question_id] = features_snapshot
            record = self._persist_resolved(
                question, answered_at=now, stressed_raw=stressed_answer,
                mental_raw=mental_answer, posture_raw=posture_answer, free_text=free_text, is_self_report=True,
            )
            self._pending_features.pop(question.question_id, None)
        return record

    def _persist_resolved(self, question: checkin.PendingQuestion, answered_at: Optional[datetime],
                           stressed_raw: Optional[str], mental_raw: Optional[str],
                           free_text: Optional[str], is_self_report: bool,
                           posture_raw: Optional[str] = None) -> dict:
        stressed_derived = checkin.derive_answer(stressed_raw)
        mental_derived = checkin.derive_answer(mental_raw)
        posture_derived = checkin.derive_answer(posture_raw)
        stressed_agreed = None if is_self_report else checkin.lane_agreement(
            checkin.STRESS, question.stress_label_at_ask, stressed_derived)
        mental_agreed = None if is_self_report else checkin.lane_agreement(
            checkin.MENTAL_LOAD, question.mental_label_at_ask, mental_derived)
        posture_agreed = None if is_self_report else checkin.lane_agreement(
            checkin.POSTURE, question.posture_label_at_ask, posture_derived)

        record = {
            "question_id": question.question_id,
            "participant_id": question.participant_id,
            "trigger_source": question.trigger_source,
            "stress_label_at_ask": question.stress_label_at_ask,
            "mental_label_at_ask": question.mental_label_at_ask,
            "posture_label_at_ask": question.posture_label_at_ask,
            "moment_time": _iso(question.moment_time),
            "asked_at": _iso(question.asked_at),
            "answered_at": _iso(answered_at),
            "stressed_raw_answer": stressed_raw,
            "mental_raw_answer": mental_raw,
            "posture_raw_answer": posture_raw,
            "free_text": free_text,
            "stressed_derived": stressed_derived,
            "mental_derived": mental_derived,
            "posture_derived": posture_derived,
            "stressed_agreed": stressed_agreed,
            "mental_agreed": mental_agreed,
            "posture_agreed": posture_agreed,
            "is_self_report": is_self_report,
        }
        features = self._pending_features.get(question.question_id, {})
        self._write_record(question.participant_id, record, features)
        return record

    def _write_record(self, participant_id: str, record: dict, features: dict) -> None:
        participant_dir = self._output_dir / participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)

        csv_path = participant_dir / "checkins.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: record.get(k) for k in CSV_FIELDS})

        json_path = participant_dir / "checkins.json"
        existing: list[dict] = []
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append({**record, "features": features})
        json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- step 8: measuring agreement ------------------------------------------

    def get_stats(self) -> dict:
        """Per-source asked/answered/agreement counts across every
        participant folder under live_output/ -- the Research view's
        "Check-ins" summary. Reads from disk (not in-memory state) so this
        reflects everything ever recorded, including past process runs."""
        per_source = {
            checkin.STRESS: {"asked": 0, "answered": 0, "confirmed": 0},
            checkin.MENTAL_LOAD: {"asked": 0, "answered": 0, "confirmed": 0},
            checkin.POSTURE: {"asked": 0, "answered": 0, "confirmed": 0},
        }
        _AGREED_FIELD = {
            checkin.STRESS: "stressed_agreed",
            checkin.MENTAL_LOAD: "mental_agreed",
            checkin.POSTURE: "posture_agreed",
        }
        self_report_count = 0
        rows: list[dict] = []

        if self._output_dir.exists():
            for participant_dir in sorted(self._output_dir.iterdir()):
                json_path = participant_dir / "checkins.json"
                if not json_path.exists():
                    continue
                try:
                    records = json.loads(json_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                for record in records:
                    # Feature vectors stay on disk for retrain_from_checkins.py;
                    # the dashboard-facing stats payload doesn't need them
                    # (and they can contain NaN, which isn't valid JSON).
                    rows.append({k: v for k, v in record.items() if k != "features"})
                    if record.get("is_self_report"):
                        self_report_count += 1
                        continue
                    source = record.get("trigger_source")
                    if source not in per_source:
                        continue
                    per_source[source]["asked"] += 1
                    agreed = record.get(_AGREED_FIELD[source])
                    if agreed is not None:
                        per_source[source]["answered"] += 1
                        if agreed:
                            per_source[source]["confirmed"] += 1

        for source, counts in per_source.items():
            counts["agreement_rate"] = (
                round(counts["confirmed"] / counts["answered"], 3) if counts["answered"] else None
            )

        rows.sort(key=lambda r: r.get("asked_at") or "", reverse=True)
        return {"by_source": per_source, "self_report_count": self_report_count, "history": rows[:200]}


bridge = CheckinBridge()
