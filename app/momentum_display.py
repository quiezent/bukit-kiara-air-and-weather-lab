"""Shared presentation of published first-crossing probabilities.

This pure formatter does not fit a model, change a probability or qualify a
forecast. The adapters retain responsibility for sensor/issue freshness.
"""
from collections.abc import Mapping
from decimal import Decimal, ROUND_HALF_UP
import math


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def display_text(event):
    """Return the model's momentum label, or None for unavailable/invalid data."""
    if not isinstance(event, Mapping) or event.get("available") is not True:
        return None
    probabilities = [event.get(key) for key in
                     ("probabilityRise", "probabilityDrop", "probabilityNoCrossing")]
    if any(not _finite_number(value) or not 0 <= value <= 1 for value in probabilities):
        return None
    if abs(sum(probabilities) - 1) > 1e-8:
        return None
    if "referencePm" in event:
        reference = event["referencePm"]
        if not _finite_number(reference) or reference < 0:
            return None
    if "changeThresholdUgM3" in event and event["changeThresholdUgM3"] != 20:
        return None
    if ("directionDecision" in event
            and event["directionDecision"] != "three_class_argmax_ties_unresolved"):
        return None
    largest = max(probabilities)
    tied = probabilities.count(largest) != 1
    winner = None if tied else probabilities.index(largest)
    direction = "rise" if winner == 0 else "drop" if winner == 1 else "unresolved"
    declared = event.get("diagnosticDirection")
    if declared is None:
        declared = event.get("direction")
    if declared is not None and declared != direction:
        return None
    if tied:
        return "Momentum outcomes tied"
    # Match f(p * 100, 1): toFixed rounds the exact binary float, resolving
    # exact half steps upward. Decimal.from_float retains that binary value.
    percentage = Decimal.from_float(float(probabilities[winner] * 100)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP)
    formatted = format(percentage, ".1f").removesuffix(".0")
    if winner == 2:
        return f"No ≥20 µg/m³ change: {formatted}%"
    return f"{formatted} % ≥20 µg/m³ {'rise' if winner == 0 else 'fall'}"


def attach_display_text(event):
    """Copy an event for web/Coach presentation, leaving its source untouched."""
    if not isinstance(event, Mapping):
        return {"available": False, "displayText": None}
    return {**event, "displayText": display_text(event)}
