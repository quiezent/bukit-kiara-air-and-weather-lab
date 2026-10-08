"""Pure shared labels for current observations and arrival endpoint outcomes.

There are no model fits, provider/database reads, clock rewrites, or numeric
forecast adjustments here. Forecast clock/reference availability belongs to
the adapters. Arrival percentages remain the learner's original fractions.
"""
from collections.abc import Mapping
import math


CURRENT_DEFAULT = "Latest sensor reading"
_ARRIVAL_FIELDS = ("probabilityFall20", "probabilityFall40", "probabilityRise20",
                   "probabilityRise40", "probabilityWithin20")


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def current_display_text(reading, analysis, now):
    """Mirror the web's observation label using the actual server sensor age."""
    if not isinstance(reading, Mapping) or not isinstance(analysis, Mapping):
        return CURRENT_DEFAULT
    air = analysis.get("airWindow")
    observed = air.get("observedMovement") if isinstance(air, Mapping) else None
    if not isinstance(observed, Mapping):
        return CURRENT_DEFAULT
    label, stamp = observed.get("eventLabel"), reading.get("epoch")
    through = observed.get("observedThroughEpoch")
    if through is None:
        current = analysis.get("current")
        through = current.get("epoch") if isinstance(current, Mapping) else None
    if (not isinstance(label, str) or not label.strip()
            or not _finite_number(stamp) or stamp <= 0
            or not _finite_number(through) or through <= 0
            or not _finite_number(now)):
        return CURRENT_DEFAULT
    sensor_age = now - stamp
    if through > stamp or stamp - through > 600 or not 0 <= sensor_age <= 720:
        return CURRENT_DEFAULT
    return "Observed: " + label


def arrival_outcome(source):
    """Summarize the native disjoint fall/rise/within20 endpoint probabilities.

The five native bins produce three exhaustive headline groups: fall20 is the
two fall bins, rise20 is the two rise bins, and within20 is the center bin.
This formatter never reads or substitutes first-crossing probabilities.
"""
    empty = {"display_text": None, "outcome": None, "outcome_probability": None}
    if not isinstance(source, Mapping) or source.get("available") is not True:
        return empty
    values = [source.get(name) for name in _ARRIVAL_FIELDS]
    if any(not _finite_number(value) or not 0 <= value <= 1 for value in values):
        return empty
    fall20, fall40, rise20, rise40, within20 = values
    if (fall40 > fall20 + 1e-12 or rise40 > rise20 + 1e-12
            or abs(fall20 + rise20 + within20 - 1) > 1e-8):
        return empty
    outcomes = ((fall20, "fall20", "Fall on arrival"),
                (rise20, "rise20", "Rise on arrival"),
                (within20, "within20", "No change on arrival"))
    largest = max(item[0] for item in outcomes)
    leaders = [item for item in outcomes if item[0] == largest]
    if len(leaders) != 1:
        return {"display_text": "Arrival outcome uncertain", "outcome": None,
                "outcome_probability": None}
    probability, outcome, text = leaders[0]
    return {"display_text": text, "outcome": outcome,
            "outcome_probability": probability}
