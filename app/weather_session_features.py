"""Issue-vintage rain, transport and mixing features for longer PM sessions.

This module is read-only and has no effect on the near-term feature builders.
The latest weather fetch at or before the actual issue is selected by the
shared validated archive. Precipitation is a preceding-hour total; gusts are
preceding-hour maxima. Temperature, humidity, winds and mixing height are
instantaneous hourly samples. Missing channels remain None, never dry/calm.
Rain probability is a separate predictor, not a PM washout probability.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from copy import deepcopy
import json
import math
import threading
from types import SimpleNamespace

import rain_weather_features as rain

VERSION = "weather_session_transport_rain_v1"
HOUR = 3600.0
EPSILON = 1e-6
FEATURE_NAMES = (
    "weatherAgeHours", "priorRainMm", "priorExposureMm", "duringRainMm",
    "duringAccumulationMeanMm", "priorRainMaxHourlyMm", "duringRainMaxHourlyMm",
    "prior3hRainMm", "issue3hRainMm",
    "priorRainProbMean", "duringRainProbMean", "priorAmountWetHours",
    "duringAmountWetHours", "firstRainLeadHours", "lastRainEndLeadHours",
    "hoursSinceRainEndAtStart", "targetTempDeltaC", "targetTempMinDeltaC",
    "targetRhDeltaPct", "targetRhMaxDeltaPct",
    *(f"wind{height}{suffix}" for height in ("10", "925") for suffix in (
        "UMeanKmh", "VMeanKmh", "UDeltaKmh", "VDeltaKmh", "SpeedMeanKmh",
        "SpeedDeltaKmh", "DirectionChangeDeg")),
    "gustHourlyEnvelopeMaxKmh", "gustHourlyEnvelopeDeltaKmh",
    "boundaryLayerMeanM", "boundaryLayerDeltaM", "ventilationMeanM2S",
    "ventilationDeltaM2S",
)
_OPTIONAL_UNITS = {
    "wind_gusts_10m": {"km/h"}, "wind_speed_180m": {"km/h"},
    "boundary_layer_height": {"m"},
}
_LOCK = threading.RLock()
_PARSED = OrderedDict()
_DESCRIPTIONS = OrderedDict()


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _parsed(run):
    key = (run.fetched_epoch, run.payload_hash)
    with _LOCK:
        if key in _PARSED:
            _PARSED.move_to_end(key)
            return _PARSED[key]
    data = json.loads(run.payload_json)
    units = data.get("hourlyUnits") or data.get("hourly_units") or {}
    points = {float(row["epoch"]): row for row in data.get("hourly", [])}
    values = dict(run.values)
    missing = []
    for field, allowed in _OPTIONAL_UNITS.items():
        usable = units.get(field) in allowed
        if not usable:
            missing.append(field)
        column = []
        for epoch in run.epochs:
            number = _finite(points[epoch].get(field)) if usable else None
            column.append(number if number is not None and number >= 0 else None)
        values[field] = tuple(column)
    for height, suffix in (("10", "10m"), ("925", "925hPa")):
        speed, direction = values["wind_speed_" + suffix], values["wind_direction_" + suffix]
        values["u" + height] = tuple(
            -s * math.sin(math.radians(d)) if s is not None and d is not None else None
            for s, d in zip(speed, direction))
        values["v" + height] = tuple(
            -s * math.cos(math.radians(d)) if s is not None and d is not None else None
            for s, d in zip(speed, direction))
    result = SimpleNamespace(epochs=run.epochs, values=values, missing_units=tuple(missing))
    with _LOCK:
        _PARSED[key] = result
        while len(_PARSED) > 4096:
            _PARSED.popitem(last=False)
    return result


def _blocks(run, start, end):
    return [(i, max(start, run.epochs[i] - HOUR), min(end, run.epochs[i]))
            for i in range(bisect_right(run.epochs, start),
                           bisect_left(run.epochs, end + HOUR))
            if min(end, run.epochs[i]) > max(start, run.epochs[i] - HOUR)]


def _hourly_stats(run, field, start, end):
    if end == start:
        return {"total": 0.0, "max": None, "wetHours": 0.0, "blocks": [],
                "accumulationMean": 0.0}
    blocks = _blocks(run, start, end)
    complete = sum(b-a for _, a, b in blocks) >= end-start-EPSILON
    complete &= all(run.values[field][i] is not None for i, _, _ in blocks)
    if not complete:
        return None
    total = sum(run.values[field][i] * (b-a) / HOUR for i, a, b in blocks)
    # Integral over the target of the rain accumulated since target start.
    # Hourly amount is spread uniformly within its hour because onset is not
    # resolved by an hourly forecast; each block contributes its later exposure.
    exposure = sum(run.values[field][i] / HOUR *
                   ((end-a)**2 - (end-b)**2) / 2 for i, a, b in blocks) / (end-start)
    return {"total": total, "max": max(run.values[field][i] for i, _, _ in blocks),
            "wetHours": sum((b-a)/HOUR for i, a, b in blocks if run.values[field][i] >= .1),
            "blocks": blocks, "accumulationMean": exposure}


def _sample(run, field, timestamp):
    right = bisect_left(run.epochs, timestamp)
    if right < len(run.epochs) and run.epochs[right] == timestamp:
        return run.values[field][right]
    left = right - 1
    if left < 0 or right >= len(run.epochs) or run.epochs[right]-run.epochs[left] > HOUR+EPSILON:
        return None
    a, b = run.values[field][left], run.values[field][right]
    if a is None or b is None:
        return None
    fraction = (timestamp-run.epochs[left]) / (run.epochs[right]-run.epochs[left])
    return a + fraction * (b-a)


def _path(run, fields, start, end):
    times = [start, *run.epochs[bisect_right(run.epochs, start):bisect_left(run.epochs, end)]]
    if end > start:
        times.append(end)
    series = [[_sample(run, field, t) for t in times] for field in fields]
    if any(value is None for column in series for value in column):
        return None
    if any(b-a > HOUR+EPSILON for a, b in zip(times, times[1:])):
        return None
    return times, series


def _stats(run, field, start, end):
    path = _path(run, (field,), start, end)
    if path is None:
        return None
    times, (values,) = path
    mean = (sum((b-a)*(x+y)/2 for a, b, x, y in
                zip(times, times[1:], values, values[1:])) / (end-start)
            if end > start else values[0])
    return mean, min(values), max(values)


def _ventilation(run, start, end):
    path = _path(run, ("boundary_layer_height", "wind_speed_180m"), start, end)
    if path is None:
        return None
    times, (height, speed) = path
    if start == end:
        return height[0] * speed[0] / 3.6
    # Exact integral of the product of two linearly interpolated channels.
    integral = sum((b-a) * (2*ha*sa + ha*sb + hb*sa + 2*hb*sb) / 6
                   for a, b, ha, hb, sa, sb in
                   zip(times, times[1:], height, height[1:], speed, speed[1:]))
    return integral / (end-start) / 3.6


def describe_window(runs, issue_epoch, start_epoch, end_epoch):
    """Return JSON-safe causal weather features for an exact session target."""
    issue, start, end = (_finite(value) for value in (issue_epoch, start_epoch, end_epoch))
    if any(value is None for value in (issue, start, end)) or start < issue or end < start:
        raise ValueError("require finite issue_epoch <= start_epoch <= end_epoch")
    run = rain.latest_run(runs, issue)
    key = (VERSION, runs.revision, runs.cutoff_epoch,
           None if run is None else (run.fetched_epoch, run.payload_hash), issue, start, end)
    with _LOCK:
        if key in _DESCRIPTIONS:
            _DESCRIPTIONS.move_to_end(key)
            return deepcopy(_DESCRIPTIONS[key])
    base = rain.describe_window(runs, issue, start, end)
    values = {field: None for field in FEATURE_NAMES}
    metadata = {**base["metadata"], "version": VERSION,
                "rainTimingBasis": "forecast_amount_at_least_0.1_mm_per_hour_not_probability",
                "accumulationSemantics": "mean_during_session_accumulated_rain_from_session_start",
                "windSemantics": "meteorological_from_direction_to_east_north_transport_components",
                "gustSemantics": "preceding_hour_max_envelope_for_hours_intersecting_session",
                "ventilationSemantics": "mixing_height_times_180m_wind_speed_m2_per_second_proxy",
                "prior3hType": "modelled_from_selected_issue_vintage_not_observed"}
    result = {"available": bool(base["available"]), "fetchedEpoch": base["fetchedEpoch"],
              "featureValues": values, "metadata": metadata}
    if run is not None and not base["metadata"]["errors"]:
        parsed = _parsed(run)
        values["weatherAgeHours"] = (issue-run.fetched_epoch) / HOUR
        for name in ("priorRainMm", "duringRainMm", "priorRainProbMean", "duringRainProbMean",
                     "priorAmountWetHours", "duringAmountWetHours", "targetTempDeltaC",
                     "targetTempMinDeltaC", "targetRhDeltaPct"):
            values[name] = base["featureValues"].get(name)
        values["priorExposureMm"] = values["priorRainMm"]
        prior = _hourly_stats(parsed, "precipitation", issue, start)
        during = _hourly_stats(parsed, "precipitation", start, end)
        before = _hourly_stats(parsed, "precipitation", start-3*HOUR, start)
        recent = _hourly_stats(parsed, "precipitation", issue-3*HOUR, issue)
        values["duringAccumulationMeanMm"] = during["accumulationMean"] if during else None
        values["priorRainMaxHourlyMm"] = prior["max"] if prior else None
        values["duringRainMaxHourlyMm"] = during["max"] if during else None
        values["prior3hRainMm"] = before["total"] if before else None
        values["issue3hRainMm"] = recent["total"] if recent else None
        if prior is not None and during is not None:
            rainy = [(i, a, b) for i, a, b in prior["blocks"]+during["blocks"]
                     if parsed.values["precipitation"][i] >= .1]
            if rainy:
                values["firstRainLeadHours"] = (rainy[0][1]-issue) / HOUR
                values["lastRainEndLeadHours"] = (rainy[-1][2]-issue) / HOUR
            prior_rain = [(i, a, b) for i, a, b in prior["blocks"]
                          if parsed.values["precipitation"][i] >= .1]
            if prior_rain:
                values["hoursSinceRainEndAtStart"] = (start-prior_rain[-1][2]) / HOUR
        rh = _stats(parsed, "relative_humidity_2m", start, end)
        rh_now = _sample(parsed, "relative_humidity_2m", issue)
        if rh is not None and rh_now is not None:
            values["targetRhMaxDeltaPct"] = rh[2]-rh_now
        for height, suffix in (("10", "10m"), ("925", "925hPa")):
            means, current = {}, {}
            for axis in ("u", "v"):
                stats = _stats(parsed, axis+height, start, end)
                now = _sample(parsed, axis+height, issue)
                means[axis] = stats[0] if stats else None
                current[axis] = now
                values[f"wind{height}{axis.upper()}MeanKmh"] = means[axis]
                values[f"wind{height}{axis.upper()}DeltaKmh"] = means[axis]-now if means[axis] is not None and now is not None else None
            speed = _stats(parsed, "wind_speed_"+suffix, start, end)
            speed_now = _sample(parsed, "wind_speed_"+suffix, issue)
            values[f"wind{height}SpeedMeanKmh"] = speed[0] if speed else None
            values[f"wind{height}SpeedDeltaKmh"] = speed[0]-speed_now if speed and speed_now is not None else None
            if all(v is not None for v in (*means.values(), *current.values())):
                if math.hypot(*means.values()) >= 1 and math.hypot(*current.values()) >= 1:
                    cross = current["u"]*means["v"]-current["v"]*means["u"]
                    dot = current["u"]*means["u"]+current["v"]*means["v"]
                    values[f"wind{height}DirectionChangeDeg"] = math.degrees(math.atan2(cross, dot))
        gust = _hourly_stats(parsed, "wind_gusts_10m", start, end)
        gust_now = _hourly_stats(parsed, "wind_gusts_10m", issue-HOUR, issue)
        values["gustHourlyEnvelopeMaxKmh"] = gust["max"] if gust else None
        if gust and gust_now and gust["max"] is not None and gust_now["max"] is not None:
            values["gustHourlyEnvelopeDeltaKmh"] = gust["max"]-gust_now["max"]
        mixing = _stats(parsed, "boundary_layer_height", start, end)
        mixing_now = _sample(parsed, "boundary_layer_height", issue)
        values["boundaryLayerMeanM"] = mixing[0] if mixing else None
        if mixing and mixing_now is not None:
            values["boundaryLayerDeltaM"] = mixing[0]-mixing_now
        vent = _ventilation(parsed, start, end)
        vent_now = _ventilation(parsed, issue, issue)
        values["ventilationMeanM2S"] = vent
        values["ventilationDeltaM2S"] = vent-vent_now if vent is not None and vent_now is not None else None
        metadata["missingOptionalUnits"] = list(dict.fromkeys(
            base["metadata"].get("missingOptionalUnits", []) + list(parsed.missing_units)))
    with _LOCK:
        _DESCRIPTIONS[key] = deepcopy(result)
        while len(_DESCRIPTIONS) > 16384:
            _DESCRIPTIONS.popitem(last=False)
    return result


def describe_windows(runs, windows):
    return [describe_window(runs, issue, start, end) for issue, start, end in windows]
