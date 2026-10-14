"""Causal regional wind context for experimental TTDI session models.

Only genuine ``regional_air_paths_v1`` issued snapshots are eligible. A query
uses the latest fetch known at its actual issue, never later stitched winds.
The 30-point GFS field is coarse regional context: modeled Sumatra contacts
and travel times are not detected smoke, upwind PM measurements, or proof of
a source. No network requests, archive writes, or forecast selection occur.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading

import numpy as np

import haze_transport as transport

VERSION = "causal_regional_transport_features_v1"
MAX_AGE_SECONDS = 6 * 3600
SOURCE_BEARING_DEG = 184.0
SOURCE_DISTANCE_KM = 294.0
MIN_PATH_HOURS = 12.0
FEATURE_NAMES = (
    "regional925UMeanKmh", "regional925VMeanKmh", "regional925SpeedDeltaKmh",
    "regional850UMeanKmh", "regional850VMeanKmh", "regional850SpeedDeltaKmh",
    "regional925SourceAlignment", "regional925SourceComponentKmh",
    "regional925SourceTravelHoursApprox", "regional925DirectionChangeDeg",
    "regionalVerticalShearKmh", "regional925SumatraPathFraction",
    "regional925TransitMinHours", "regional925PathCoverageHours",
    "regionalHeightAgreementFraction",
)
_LOCK = threading.RLock()
_GRIDS = OrderedDict()
_PATHS = OrderedDict()
_RESULTS = OrderedDict()


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class RegionalRun:
    fetched_epoch: float
    model_version: str
    payload_blob: bytes
    payload_hash: str


@dataclass(frozen=True)
class RegionalArchive:
    runs: tuple[RegionalRun, ...]
    fetched_epochs: tuple[float, ...]
    cutoff_epoch: float
    revision: str
    database_path: str
    archive_error: str | None = None


def load_runs(db_path, cutoff_epoch, lookback_days=30):
    """Read bounded compressed snapshots without inflating the entire archive.

    Thirty days cover this model's 14-day fit plus warmup. For a longer replay,
    explicitly request its history duration; as-of selection stays unchanged.
    """
    cutoff = _finite(cutoff_epoch)
    if cutoff is None or _finite(lookback_days) is None or lookback_days <= 0:
        raise ValueError("finite cutoff and positive lookback_days required")
    path = Path(db_path).resolve()
    error = None
    try:
        with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=20)) as conn:
            raw = conn.execute(
                "SELECT fetched_epoch,model_version,payload FROM regional_wind_runs "
                "WHERE fetched_epoch<=? AND fetched_epoch>=? ORDER BY fetched_epoch",
                (cutoff, cutoff-float(lookback_days)*86400-MAX_AGE_SECONDS)).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
        raw, error = [], "missing_regional_archive"
    runs, digest = [], hashlib.blake2b(digest_size=16)
    for fetched, model_version, payload in raw:
        fetched = _finite(fetched)
        if fetched is None:
            continue
        blob = bytes(payload) if isinstance(payload, (bytes, bytearray, memoryview)) else str(payload).encode()
        checksum = hashlib.blake2b(blob, digest_size=16).hexdigest()
        digest.update(repr((fetched, model_version, checksum)).encode())
        runs.append(RegionalRun(fetched, str(model_version), blob, checksum))
    return RegionalArchive(tuple(runs), tuple(r.fetched_epoch for r in runs),
                           cutoff, digest.hexdigest(), str(path), error)


def latest_run(archive, issue_epoch):
    issue = _finite(issue_epoch)
    if issue is None:
        raise ValueError("finite issue_epoch required")
    index = bisect_right(archive.fetched_epochs, issue)-1
    return archive.runs[index] if index >= 0 else None


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _grid(run):
    key = (run.fetched_epoch, run.model_version, run.payload_hash)
    with _LOCK:
        if key in _GRIDS:
            _GRIDS.move_to_end(key)
            return _GRIDS[key]
    errors, data, grid = [], {}, None
    try:
        data = json.loads(gzip.decompress(run.payload_blob))
        if not isinstance(data, dict):
            raise ValueError("regional payload is not an object")
        if run.model_version != transport.VERSION or data.get("version") != transport.VERSION:
            errors.append("unsupported_regional_version")
        if data.get("source") != "Open-Meteo / NOAA GFS":
            errors.append("invalid_regional_source")
        if data.get("dataRole") != "issued_regional_wind_snapshot":
            errors.append("not_issued_regional_snapshot")
        if _finite(data.get("fetchedEpoch")) != run.fetched_epoch:
            errors.append("payload_fetch_timestamp_mismatch")
        times = np.asarray(data.get("times", []), dtype=float)
        if (len(times) < 2 or not np.isfinite(times).all() or
                np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 3600+1e-6)):
            errors.append("invalid_regional_time_axis")
        latitudes, longitudes = data.get("latitudes", []), data.get("longitudes", [])
        points = data.get("points", [])
        if len(latitudes) < 2 or len(longitudes) < 2 or len(points) != len(latitudes)*len(longitudes):
            errors.append("invalid_regional_grid_shape")
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                raise ValueError("regional point is not an object")
            if longitudes and latitudes and index < len(latitudes)*len(longitudes):
                lat = latitudes[index//len(longitudes)]
                lon = longitudes[index % len(longitudes)]
                if _finite(point.get("latitude")) != _finite(lat) or _finite(point.get("longitude")) != _finite(lon):
                    errors.append("regional_points_not_row_major")
            for field in transport.FIELDS:
                if not isinstance(point.get(field), list) or len(point[field]) != len(times):
                    errors.append("missing_regional_field:"+field)
        # v1 snapshots retain the collector's checked km/h winds, but do not
        # retain a raw hourly-units object. Respect it if future payloads do.
        units = data.get("hourlyUnits") or data.get("hourly_units")
        if units is not None:
            if not isinstance(units, dict):
                errors.append("invalid_regional_units")
            else:
                for level in (925, 850):
                    if units.get(f"wind_speed_{level}hPa") != "km/h":
                        errors.append("invalid_regional_wind_unit")
                    if units.get(f"wind_direction_{level}hPa") not in ("°", "degree", "degrees"):
                        errors.append("invalid_regional_direction_unit")
        if not errors:
            # Missing or invalid individual cells stay NaN rather than calm.
            for point in points:
                for level in (925, 850):
                    speed_key, direction_key = f"wind_speed_{level}hPa", f"wind_direction_{level}hPa"
                    for index, (speed, direction) in enumerate(zip(point[speed_key], point[direction_key])):
                        speed, direction = _finite(speed), _finite(direction)
                        if speed is None or speed < 0 or direction is None or not 0 <= direction <= 360:
                            point[speed_key][index] = None
                            point[direction_key][index] = None
                point["geopotential_height_925hPa"] = [
                    _finite(value) for value in point["geopotential_height_925hPa"]]
            grid = transport.WindGrid(data)
            if (not np.isfinite(grid.lats).all() or not np.isfinite(grid.lons).all()
                    or np.any(np.diff(grid.lats) <= 0) or np.any(np.diff(grid.lons) <= 0)
                    or np.any(np.abs(grid.lats) > 90) or np.any(np.abs(grid.lons) > 180)):
                grid = None
                errors.append("invalid_regional_coordinate_axes")
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, EOFError) as exc:
        errors.append("malformed_regional_snapshot")
    result = (grid, data, tuple(dict.fromkeys(errors)))
    with _LOCK:
        return _remember(_GRIDS, key, result, 128)


def _wind_stats(grid, level, issue, start, end):
    times = np.unique(np.r_[start, grid.times[(grid.times > start) & (grid.times < end)], end])
    lat, lon = transport.TARGET
    u = [grid.value(f"u{level}", lat, lon, t) for t in times]
    v = [grid.value(f"v{level}", lat, lon, t) for t in times]
    now = [grid.value(f"{axis}{level}", lat, lon, issue) for axis in ("u", "v")]
    if any(value is None for value in (*u, *v, *now)):
        return None
    mean = lambda values: float(np.trapz(values, times)/(end-start)) if end > start else float(values[0])
    return {"u": mean(u), "v": mean(v), "speed": mean(np.hypot(u, v)),
            "now_u": now[0], "now_v": now[1], "now_speed": math.hypot(*now)}


def _vector(grid, level, lat, lon, epoch):
    """Interpolate both WindGrid components with one shared set of brackets."""
    brackets = []
    for axis, value in ((grid.lats, lat), (grid.lons, lon), (grid.times, epoch)):
        if value < axis[0] or value > axis[-1]:
            return None
        index = min(max(bisect_right(axis, value)-1, 0), len(axis)-2)
        brackets.append((index, (value-axis[index])/(axis[index+1]-axis[index])))
    (i, a), (j, b), (k, c) = brackets
    field_u, field_v = grid.fields[f"u{level}"], grid.fields[f"v{level}"]
    u = v = 0.0
    for di, wa in ((0, 1-a), (1, a)):
        for dj, wb in ((0, 1-b), (1, b)):
            for dk, wc in ((0, 1-c), (1, c)):
                weight = wa*wb*wc
                if weight <= 1e-12:
                    continue
                x, y = field_u[i+di, j+dj, k+dk], field_v[i+di, j+dj, k+dk]
                if not math.isfinite(x) or not math.isfinite(y):
                    return None
                u += weight*x
                v += weight*y
    return u, v


def _compact_path(grid, arrival, level):
    """Same midpoint back integration as WindGrid.trace, without vertices."""
    lat, lon = transport.TARGET
    epoch, hits, step = float(arrival), {}, 0.5
    age, reason = 0.0, "48_hour_limit"
    for index in range(97):
        age = index*step
        for key, (_, polygon) in transport.SECTORS.items():
            if key not in hits and transport.inside(lat, lon, polygon):
                hits[key] = age
        if age >= 48:
            break
        current = _vector(grid, level, lat, lon, epoch)
        if current is None:
            reason = "grid_or_time_limit"
            break
        u, v = current
        midlat = lat-v*step/2/111.2
        midlon = lon-u*step/2/(111.2*math.cos(math.radians(lat)))
        midpoint = _vector(grid, level, midlat, midlon, epoch-step*1800)
        if midpoint is None:
            reason = "grid_or_time_limit"
            break
        mu, mv = midpoint
        lat -= mv*step/111.2
        lon -= mu*step/(111.2*math.cos(math.radians(midlat)))
        epoch -= step*3600
    return {"arrivalEpoch": int(arrival), "levelHpa": level,
            "sectorContacts": hits, "tracedHours": age, "stopReason": reason}


def _trace(run, grid, arrival, level):
    key = (run.fetched_epoch, run.payload_hash, float(arrival), level)
    with _LOCK:
        if key in _PATHS:
            _PATHS.move_to_end(key)
            return _PATHS[key]
    result = _compact_path(grid, arrival, level)
    with _LOCK:
        return _remember(_PATHS, key, result, 16384)


def describe_window(archive, issue_epoch, start_epoch, end_epoch):
    """Return at most 15 as-issued regional context features; missing is None."""
    issue, start, end = (_finite(v) for v in (issue_epoch, start_epoch, end_epoch))
    if any(v is None for v in (issue, start, end)) or start < issue or end < start:
        raise ValueError("require finite issue_epoch <= start_epoch <= end_epoch")
    run = latest_run(archive, issue)
    key = (VERSION, archive.revision, archive.cutoff_epoch,
           None if run is None else (run.fetched_epoch, run.payload_hash), issue, start, end)
    with _LOCK:
        if key in _RESULTS:
            _RESULTS.move_to_end(key)
            return deepcopy(_RESULTS[key])
    values = {name: None for name in FEATURE_NAMES}
    errors = []
    if archive.archive_error:
        errors.append(archive.archive_error)
    if issue > archive.cutoff_epoch:
        errors.append("issue_after_regional_archive_cutoff")
    if run is None:
        errors.append("no_regional_run_at_issue")
    elif issue-run.fetched_epoch > MAX_AGE_SECONDS:
        errors.append("stale_regional_run")
    metadata = {
        "version": VERSION, "forecastIssuedEpoch": issue,
        "windowStartEpoch": start, "windowEndEpoch": end,
        "fetchedEpoch": run.fetched_epoch if run else None,
        "payloadHash": run.payload_hash if run else None,
        "selectedBy": "latest_fetched_at_or_before_actual_issue",
        "maxAgeSeconds": MAX_AGE_SECONDS, "errors": errors,
        "sourceAssumption": "Pekanbaru / central Sumatra; not detected emissions",
        "sourceBearingDegrees": SOURCE_BEARING_DEG,
        "sourceDistanceKmApprox": SOURCE_DISTANCE_KM,
        "vectorUnits": "km/h", "unitsProvenance": "regional_air_paths_v1_collector_wind_unit_check",
        "windConvention": "meteorological_from_direction_converted_to_transport_U_V_before_interpolation",
        "sourceTravelTimeMeaning": "distance_divided_by_target_source_aligned_component_not_plume_arrival",
        "pathFractionMeaning": "contacts_over_three_partially_covered_start_mid_end_backpaths_not_smoke_probability",
        "missingValues": "null_not_zero_imputed", "sourceAttributionEstablished": False,
        "upwindPmAvailable": False,
    }
    result = {"available": False, "fetchedEpoch": metadata["fetchedEpoch"],
              "featureValues": values, "metadata": metadata}
    if not errors:
        grid, data, parse_errors = _grid(run)
        errors.extend(parse_errors)
        metadata.update(source=data.get("source"), dataRole=data.get("dataRole"),
                        archiveModelVersion=run.model_version)
        if grid is not None and not errors:
            metadata.update(gridPointCount=len(data["points"]), gridTimeCount=len(grid.times),
                            gridStartEpoch=float(grid.times[0]), gridEndEpoch=float(grid.times[-1]))
            stats = {level: _wind_stats(grid, level, issue, start, end) for level in (925, 850)}
            for level, wind in stats.items():
                if wind is not None:
                    values[f"regional{level}UMeanKmh"] = wind["u"]
                    values[f"regional{level}VMeanKmh"] = wind["v"]
                    values[f"regional{level}SpeedDeltaKmh"] = wind["speed"]-wind["now_speed"]
            wind = stats[925]
            result["available"] = wind is not None
            if wind is None:
                errors.append("missing_925_target_or_issue_wind")
            else:
                magnitude = math.hypot(wind["u"], wind["v"])
                # Wind FROM the southern assumed source transports northward.
                bearing = math.radians(SOURCE_BEARING_DEG)
                component = wind["u"]*(-math.sin(bearing)) + wind["v"]*(-math.cos(bearing))
                values["regional925SourceComponentKmh"] = component
                if magnitude >= 1:
                    values["regional925SourceAlignment"] = component/magnitude
                if component >= 1:
                    values["regional925SourceTravelHoursApprox"] = SOURCE_DISTANCE_KM/component
                if magnitude >= 1 and wind["now_speed"] >= 1:
                    cross = wind["now_u"]*wind["v"]-wind["now_v"]*wind["u"]
                    dot = wind["now_u"]*wind["u"]+wind["now_v"]*wind["v"]
                    values["regional925DirectionChangeDeg"] = math.degrees(math.atan2(cross, dot))
                if stats[850] is not None:
                    values["regionalVerticalShearKmh"] = math.hypot(
                        wind["u"]-stats[850]["u"], wind["v"]-stats[850]["v"])
                arrivals = sorted(set((start, (start+end)/2, end)))
                paths = {level: [_trace(run, grid, t, level) for t in arrivals] for level in (925, 850)}
                primary = paths[925]
                coverage = min(p["tracedHours"] for p in primary)
                values["regional925PathCoverageHours"] = float(coverage)
                sufficient = all(p["tracedHours"] >= MIN_PATH_HOURS for p in primary)
                contacts = [bool(p["sectorContacts"]) for p in primary]
                if sufficient:
                    values["regional925SumatraPathFraction"] = sum(contacts)/len(primary)
                    transit = [min(p["sectorContacts"].values()) for p in primary if p["sectorContacts"]]
                    values["regional925TransitMinHours"] = min(transit) if transit else None
                    if all(p["tracedHours"] >= MIN_PATH_HOURS for p in paths[850]):
                        secondary = [bool(p["sectorContacts"]) for p in paths[850]]
                        values["regionalHeightAgreementFraction"] = sum(a == b for a, b in zip(contacts, secondary))/len(primary)
                metadata.update(paths={str(level): paths[level] for level in paths},
                                minimumPathHoursRequired=MIN_PATH_HOURS,
                                complete48HourPaths=all(p["tracedHours"] >= 48 for p in primary),
                                pathCoverageSufficient=sufficient)
    with _LOCK:
        _remember(_RESULTS, key, deepcopy(result), 16384)
    return result


def describe_windows(archive, windows):
    return [describe_window(archive, issue, start, end) for issue, start, end in windows]
