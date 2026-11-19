"""Shared stress/fatigue/cognitive-load scoring.

This used to live only inside live_dashboard_backend.py, computed there
from a *rolling, min/max-normalized* window of recently-polled samples.
Two problems with that:

1. It made the dashboard's trend-chart risk line cross the 78 "High"
   threshold a tick or two *after* construction_stress_speech.py had
   already spoken the round-1 message (which triggered instantly off the
   categorical ML label) -- the on-screen system-message banner looked
   like it was running ahead of the graph.
2. An earlier attempt to fix that by feeding the rolling-window score back
   into the TTS trigger made things worse: the weighted formula often
   never reliably climbed past 78 at all, so round-1 stopped firing
   consistently.

The fix here is to make the score a direct, deterministic function of the
categorical "high"/"medium"/"low" label instead of a rolling window -- no
ramp-up tick, and the chart and the TTS trigger are reading the exact same
number by construction, so they cannot disagree. Confidence is used only
to add a little visual variation within each band, never enough to cross
a band boundary.
"""

RISK_THRESHOLD = 78  # "High" band cutoff -- keep in sync with psychosocial_dashboard.html's STRESS_THRESHOLD
CAUTION_THRESHOLD = 60  # "Caution"/medium band cutoff

# (low, high) overall-score range per label, each strictly inside its
# band_for() bucket (low < 60 <= medium < 78 <= high) with margin so no
# amount of confidence jitter -- or convex combination of stress/fatigue/
# cognitive below, which are each independently in-range -- can cross a
# boundary by accident.
_BAND_RANGES = {
    "low": (10.0, 55.0),
    "medium": (61.0, 76.0),
    "high": (80.0, 98.0),
    "pending": (20.0, 40.0),
}


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def _score_for_label(label, confidence):
    lo, hi = _BAND_RANGES.get(label, _BAND_RANGES["pending"])
    confidence = _clamp(confidence if confidence is not None else 0.5, 0.0, 1.0)
    return lo + confidence * (hi - lo)


class RollingRiskScorer:
    """Kept as a class (rather than a bare function) so pollers can hold one
    instance per data source without the call site changing if this ever
    grows session-relative state again."""

    def score(self, frustration_label, mental_label, frustration_confidence=None, mental_confidence=None):
        stress = _score_for_label(frustration_label, frustration_confidence)
        cognitive = _score_for_label(mental_label, mental_confidence)
        # Fatigue has no dedicated ML label; approximate it from whichever
        # of the two readings is worse, so a long, unbroken "high" episode
        # still reads as more fatiguing than a brand-new one.
        fatigue = max(stress, cognitive) * 0.9 + min(stress, cognitive) * 0.1

        overall = _clamp(0.50 * stress + 0.18 * fatigue + 0.32 * cognitive)
        return {
            "stress": round(stress, 1),
            "fatigue": round(fatigue, 1),
            "cognitive": round(cognitive, 1),
            "overall": round(overall, 1),
        }


def band_for(overall):
    if overall >= RISK_THRESHOLD:
        return "high"
    if overall >= CAUTION_THRESHOLD:
        return "medium"
    return "low"
