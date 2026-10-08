"""Bounded observed sensor history for the read-only RLCD API.

The caller supplies database rows. This helper performs no I/O and never fills
missing observations from weather forecasts. Values are rounded only for the
small graph wire format; sample timestamps always remain actual timestamps.
"""
import math

WINDOW_S = 6 * 3600
MAX_POINTS = 96
BIN_COUNT = (MAX_POINTS - 2) // 2
GAP_AFTER_S = 600
FRESH_S = 420
SOURCE = "AirGradient TTDI 86311"
COLUMNS = ["epoch", "pm25_ugm3", "temperature_c", "heat_index_c"]
UNITS = ["UTC s", "ug/m3", "C", "C"]


def _get(row, name):
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return None


def _stamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 < value < 10**12 or not math.isfinite(value) or value != int(value):
        return None
    return int(value)


def _accepted_number(value, low, high):
    """Reject malformed/out-of-range values; never clamp an observation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not low <= value <= high or not math.isfinite(value):
        return None
    return value


def _number(value, low, high):
    value = _accepted_number(value, low, high)
    if value is None:
        return None
    value = round(value, 1)
    return int(value) if value == int(value) else value


def _select(points, start, max_points=MAX_POINTS):
    if isinstance(max_points, bool) or not isinstance(max_points, int) or not 4 <= max_points <= MAX_POINTS:
        raise ValueError("max_points must be an integer from 4 to 96")
    if len(points) <= max_points:
        return points, "actual_rows"
    bin_count = (max_points - 2) // 2
    bins = [[] for _ in range(bin_count)]
    for point in points:
        index = min(bin_count - 1,
                    (point[0] - start) * bin_count // WINDOW_S)
        bins[index].append(point)
    chosen = {points[0][0]: points[0], points[-1][0]: points[-1]}
    for bucket in bins:
        if not bucket:
            continue
        measured = [p for p in bucket if p[1] is not None]
        if measured:
            low = min(measured, key=lambda p: (p[1], p[0]))
            high = max(measured, key=lambda p: (p[1], p[0]))
        else:
            low, high = bucket[0], bucket[-1]
        chosen[low[0]], chosen[high[0]] = low, high
    return sorted(chosen.values(), key=lambda p: p[0]), \
        f"{bin_count}_time_bins_pm25_minmax_plus_endpoints"


def build_history(rows, now, *, max_points=MAX_POINTS):
    """Project rows with epoch/pm02/atmp/heatindex over [now-6h, now].

    ``available`` describes historical observations, so stale history remains
    present with ``fresh=false``. ``observed_epoch`` is the newest row containing
    any accepted graph value, not necessarily the latest value of every field.
    The caller must keep each field's nulls when rendering.

    Downsampling retains actual PM min/max rows in 47 equal time bins and the
    first/last source rows, at most 96 total. Temperature/HI extrema or isolated
    field-null rows between selected points can be omitted at high density.
    ``gaps`` records actual collection gaps, before downsampling. Rendering
    must break lines across these intervals, including when no point is kept
    at a gap boundary. Also break across field nulls. Sparse selected point
    spacing alone is not a collection gap: downsampling can widen spacing.

    ``pm25_summary`` is the arithmetic sample mean/min/max of accepted original
    PM observations in the exact inclusive window, before graph rounding or
    thinning. Summary numbers retain original precision; consumers may round
    their display. Missing PM values do not contribute to its sample count.
    """
    now = _stamp(now)
    if now is None:
        raise ValueError("now must be a positive integral UTC epoch")
    start = now - WINDOW_S
    unique = {}
    original_pm = {}
    for row in rows or ():
        stamp = _stamp(_get(row, "epoch"))
        if stamp is None or stamp < start or stamp > now:
            continue
        point = [stamp,
                 _number(_get(row, "pm02"), 0, 100000),
                 _number(_get(row, "atmp"), -100, 100),
                 _number(_get(row, "heatindex"), -100, 250)]
        # The production epoch column is unique. For malformed duplicate
        # input retain one actual row, never merge values across observations.
        if stamp not in unique:
            unique[stamp] = point
            original_pm[stamp] = _accepted_number(_get(row, "pm02"), 0, 100000)
    source_points = sorted(unique.values(), key=lambda p: p[0])
    observed = next((p[0] for p in reversed(source_points)
                     if any(v is not None for v in p[1:])), None)
    age = now - observed if observed is not None else None
    gaps = [[left[0], right[0]]
            for left, right in zip(source_points, source_points[1:])
            if right[0] - left[0] > GAP_AFTER_S]
    points, method = _select(source_points, start, max_points)
    pm_values = [value for value in original_pm.values() if value is not None]
    pm_summary = {
        "available": bool(pm_values),
        "average_ugm3": math.fsum(pm_values) / len(pm_values) if pm_values else None,
        "lowest_ugm3": min(pm_values) if pm_values else None,
        "highest_ugm3": max(pm_values) if pm_values else None,
        "sample_count": len(pm_values),
    }
    return {
        "available": observed is not None,
        "fresh": age is not None and age <= FRESH_S,
        "source": SOURCE,
        "start_epoch": start,
        "end_epoch": now,
        "observed_epoch": observed,
        "age_s": age,
        "columns": list(COLUMNS),
        "units": list(UNITS),
        "precision_dp": 1,
        "method": method,
        "source_rows": len(source_points),
        "gap_after_s": GAP_AFTER_S,
        "gap_kind": "source_collection_before_downsampling",
        "gaps": gaps,
        "points": points,
        "pm25_summary": pm_summary,
    }
