"""Pure, per-field coverage contracts for published hourly weather windows.

Instantaneous hourly samples represent their following hourly block. Rain,
probability and gust endpoints describe the preceding hour. Only overlapping
blocks contribute; missing, conflicting or invalid values never become zeros.
"""
from __future__ import annotations

import copy
import math

VERSION = "hourly_weather_full_window_coverage_v1"
HOUR = 3600
REQUIRED_FIELDS = (
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "wind_speed_10m", "wind_gusts_10m", "precipitation", "precipitation_probability",
)
INSTANT_FIELDS = (
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover", "pressure_msl",
    "wind_speed_180m", "wind_direction_180m", "wind_direction_925hPa",
)
INTERVAL_FIELDS = ("precipitation", "precipitation_probability", "showers", "wind_gusts_10m")
_PERCENT = {"relative_humidity_2m", "cloud_cover", "precipitation_probability"}
_NONNEGATIVE = {"precipitation", "showers", "wind_speed_10m", "wind_speed_180m", "wind_gusts_10m"}
_DIRECTIONS = {"wind_direction_10m", "wind_direction_180m", "wind_direction_925hPa"}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _value(point, field):
    value = _number(point.get(field))
    if value is None or (field in _PERCENT and not 0 <= value <= 100):
        return None
    if field in _NONNEGATIVE and value < 0:
        return None
    if field in _DIRECTIONS and not 0 <= value <= 360:
        return None
    return value


def sanitize_for_legacy_summary(payload):
    """Copy values safe for legacy display aggregation; retain original for proof.

    Missing, malformed, boolean, nonfinite and out-of-domain weather values
    become null. Invalid hourly clocks are omitted. Coverage must still be
    checked against the original payload, so this presentation copy cannot
    silently repair conflicting duplicate samples or missing support.
    """
    original = payload if isinstance(payload, dict) else {}
    result = copy.deepcopy(original)
    result["hourly"] = []
    points = original.get("hourly")
    for point in points if isinstance(points, (list, tuple)) else ():
        if not isinstance(point, dict):
            continue
        stamp = _number(point.get("epoch"))
        if stamp is None or stamp <= 0 or not stamp.is_integer() or int(stamp) % HOUR:
            continue
        safe = copy.deepcopy(point)
        safe["epoch"] = int(stamp)
        for field in INSTANT_FIELDS + INTERVAL_FIELDS:
            if field in safe:
                safe[field] = _value(point, field)
        result["hourly"].append(safe)
    return result


def _clock(start, end):
    start, end = _number(start), _number(end)
    if start is None or end is None or start <= 0 or end <= start:
        return None
    return start, end


def _points(payload):
    points = {}
    for point in (payload or {}).get("hourly", ()):
        if not isinstance(point, dict):
            continue
        stamp = _number(point.get("epoch"))
        if stamp is not None and stamp > 0 and stamp.is_integer() and int(stamp) % HOUR == 0:
            points.setdefault(int(stamp), []).append(point)
    return points


def _parts(payload, start, end):
    clock = _clock(start, end)
    if clock is None:
        return {}, {}
    start, end = clock
    points = _points(payload)
    parts, fields = {}, {}
    for field in INSTANT_FIELDS + INTERVAL_FIELDS:
        values, missing, covered = [], [], 0.0
        for block in range(math.floor(start / HOUR) * HOUR, math.ceil(end / HOUR) * HOUR, HOUR):
            overlap = max(0.0, min(end, block + HOUR) - max(start, block))
            stamp = block + HOUR if field in INTERVAL_FIELDS else block
            readings = [_value(point, field) for point in points.get(stamp, ())]
            valid = bool(readings and all(value is not None for value in readings)
                         and all(value == readings[0] for value in readings))
            if valid:
                values.append((readings[0], overlap / HOUR))
                covered += overlap
            else:
                missing.append(stamp)
        fields[field] = {
            "complete": not missing,
            "requiredHours": (end - start) / HOUR,
            "coveredHours": covered / HOUR,
            "missingHours": max(0.0, (end - start - covered) / HOUR),
            "missingOrInvalidEpochs": missing,
            "semantics": "preceding_hour_interval" if field in INTERVAL_FIELDS else "following_hour_sample_block",
        }
        parts[field] = values if not missing else []
    return parts, fields


def window_coverage(payload, start, end):
    """Require complete overlapping hourly support independently for each field."""
    _, fields = _parts(payload, start, end)
    missing = [field for field in REQUIRED_FIELDS if not (fields.get(field) or {}).get("complete")]
    return {"version": VERSION, "startEpoch": start, "endEpoch": end,
            "complete": bool(fields and not missing), "requiredFields": list(REQUIRED_FIELDS),
            "missingRequiredFields": missing, "fields": fields}


def coverage_verified(summary, start, end):
    """Legacy aggregate counts cannot establish per-field input completeness."""
    coverage = (summary or {}).get("weatherCoverage") or {}
    if not isinstance(coverage, dict) or not isinstance(coverage.get("fields"), dict):
        return False
    return bool(coverage.get("version") == VERSION and coverage.get("complete") is True
                and coverage.get("startEpoch") == start and coverage.get("endEpoch") == end
                and all((coverage.get("fields", {}).get(field) or {}).get("complete") is True
                        for field in REQUIRED_FIELDS))


def compact_coverage(summary):
    """Keep device projection small; preserve full detail in the dashboard API."""
    coverage = (summary or {}).get("weatherCoverage")
    if not isinstance(coverage, dict):
        return None
    fields = coverage.get("fields") or {}
    return {"version": coverage.get("version"), "complete": coverage.get("complete") is True,
            "missing_required_fields": coverage.get("missingRequiredFields") or [],
            "missing_fields": [field for field, value in fields.items()
                               if not isinstance(value, dict) or value.get("complete") is not True]}


def instantaneous_points(payload, start, end):
    """Select all hourly sample blocks overlapping the exact requested window."""
    clock = _clock(start, end)
    if clock is None:
        return []
    start, end = clock
    return [point for point in (payload or {}).get("hourly", ())
            if isinstance(point, dict) and _number(point.get("epoch")) is not None
            and point["epoch"] < end and point["epoch"] + HOUR > start]


def _stat(parts, field, kind):
    values = parts[field]
    if not values:
        return None
    if kind == "max":
        return max(value for value, _ in values)
    if kind == "min":
        return min(value for value, _ in values)
    if kind == "sum":
        return sum(value * weight for value, weight in values)
    if kind == "direction":
        sine = sum(math.sin(math.radians(value)) * weight for value, weight in values)
        cosine = sum(math.cos(math.radians(value)) * weight for value, weight in values)
        return math.degrees(math.atan2(sine, cosine)) % 360 if math.hypot(sine, cosine) > 1e-10 else None
    return sum(value * weight for value, weight in values) / sum(weight for _, weight in values)


def _compass(value):
    return ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")[int((value + 11.25) // 22.5) % 16] if value is not None else None


def guard_summary(summary, payload, start, end):
    """Retain presentation metadata; replace numeric summaries with proven values.

    Fractional edge blocks are duration weighted. Every field with incomplete
    support is null, even when other fields are complete. Overall availability
    requires all fields used by the compact ride-weather display.
    """
    result = copy.deepcopy(summary or {})
    parts, fields = _parts(payload, start, end)
    coverage = window_coverage(payload, start, end)
    result.update(startEpoch=start, endEpoch=end, weatherCoverage=coverage,
                  available=bool(coverage["complete"]))
    result["coverageReason"] = None if coverage["complete"] else "incomplete_hourly_weather_window"
    mappings = (
        ("apparentTemperatureMax", "apparent_temperature", "max", 1),
        ("temperatureMax", "temperature_2m", "max", 1),
        ("relativeHumidityMean", "relative_humidity_2m", "mean", 0),
        ("precipitationProbabilityMax", "precipitation_probability", "max", 0),
        ("precipitationMm", "precipitation", "sum", 1),
        ("showersMm", "showers", "sum", 1),
        ("windSpeed10mMean", "wind_speed_10m", "mean", 1),
        ("windGust10mMax", "wind_gusts_10m", "max", 1),
        ("windDirection10m", "wind_direction_10m", "direction", 0),
        ("cloudCoverMean", "cloud_cover", "mean", 0),
        ("cloudCoverMax", "cloud_cover", "max", 0),
        ("pressureMslMin", "pressure_msl", "min", 1),
        ("windSpeed180mMean", "wind_speed_180m", "mean", 1),
        ("windDirection180m", "wind_direction_180m", "direction", 0),
        ("windDirection925hPa", "wind_direction_925hPa", "direction", 0),
    )
    for destination, field, kind, digits in mappings:
        value = _stat(parts, field, kind) if fields else None
        result[destination] = round(value, digits) if value is not None else None
    for height in ("10m", "180m"):
        result[f"windDirection{height}Compass"] = _compass(result[f"windDirection{height}"])
    if not (fields.get("wind_direction_925hPa") or {}).get("complete"):
        result["sourceAlignment925"] = None
    rain, amount = result["precipitationProbabilityMax"], result["precipitationMm"]
    result["rainLabel"] = ("Rain forecast unavailable" if rain is None or amount is None
                           else "Higher modeled rain chance" if rain >= 60 or amount >= 1.0
                           else "Some modeled rain chance" if rain >= 30 else "Lower modeled rain chance")
    cloud = result["cloudCoverMean"]
    result["skyLabel"] = ("Sky forecast unavailable" if cloud is None else "Mostly clear" if cloud <= 20
                          else "Partly cloudy" if cloud <= 50 else "Mostly cloudy" if cloud <= 80 else "Overcast")
    wind = result["windSpeed180mMean"]
    supported = bool(result.get("ventilationUsed"))
    result["ventilationState"] = "unknown" if wind is None else "strong" if wind > 14 else "moderate" if wind >= 10 else "weak"
    result["ventilationLabel"] = ("Airflow forecast unavailable" if wind is None
        else ("Stronger issued-wind dispersion signal" if supported else "Strong modeled flow aloft · clearing not assured") if wind > 14
        else ("Moderate issued-wind dispersion signal" if supported else "Moderate modeled flow aloft") if wind >= 10
        else "Weak modeled flow aloft")
    if wind is None:
        result["ventilationUsed"] = False
    return result
