"""Shared, read-only, issue-time weather features for TTDI rain experiments.

This standalone research module does not fetch weather or write to the archive.
The feature definition is fixed independently of PM outcomes (version below).

Public contract
---------------
``load_runs(db_path, issue_epoch) -> RunArchive`` reads all archived weather
payloads fetched <= the exact floating-point cutoff. RunArchive is an immutable
Sequence[WeatherRun], with ``fetched_epochs``, ``cutoff_epoch`` and a content
``revision``. Load once at the final replay cutoff, then reuse for all origins.
``latest_run(runs, issue_epoch)`` selects the latest fetch <= the *actual issue*,
never a rounded sensor-feature anchor. The selected WeatherRun exposes
``fetched_epoch``, ``payload_hash``, ``epochs``, ``values`` and ``payload_json``.
``selected_payload`` returns a fresh dict of that same chosen payload for UI use.
``describe_window(runs, issue_epoch, start_epoch, end_epoch)`` returns a JSON-safe
dict: ``available: bool``, ``fetchedEpoch: float|None``, ``featureValues`` with
the fixed ``FEATURE_NAMES`` (each float|None), ``metadata``, and interval
descriptions ``prior``, ``during``, ``context3h``. ``describe_windows`` is the
batch equivalent for an iterable of (issue, start, end) triples.

Rain intervals are (issue,start], (start,end], and (issue-3h,issue]. A point
arrival uses start == end: prior includes the entire lead, during is empty.
Each hourly precipitation amount belongs to (row.epoch-3600,row.epoch], and is
prorated by interval overlap. Probability uses those same piecewise constant
hour blocks, with a duration-weighted mean and max: it is never interpolated or
summed into an event probability. HighProbHours uses probability >=60 percent;
AmountWetHours uses hourly amount >=0.1 mm; WetSignalHours uses either signal.
An empty interval has amount/durations 0 but undefined probability max/mean.

The three interval prefixes prior/during/context3h have suffixes RainMm,
RainProbMax, RainProbMean, HighProbHours, AmountWetHours, WetSignalHours,
FirstWetLeadHours, LastWetEndLeadHours. All rain/probability features require
complete coverage of their own channel. Signal features require both channels
to be covered. Missing values are None, never zero-imputed. Interval timing is
the start/end of the overlapping hourly signal blocks relative to actual issue,
not a precise subhour rain-onset forecast. Empty/no-signal timing is None.

Global features include weatherAgeHours, priorHours, duringHours,
firstWetLeadHours (first signal in issue..targetEnd), expectedWetEndLeadHours
(end of the last signal episode intersecting that range, only if a following
covered dry hour establishes cessation), and hoursUntilExpectedWetEnd (relative
to targetStart). Cessation may be before the target and thus have negative
hoursUntilExpectedWetEnd. Metadata distinguishes no signal from unknown timing.

Temperature/RH/wind are instantaneous samples and may be interpolated only
between complete adjacent hourly points, never across missing hours. The target
features compare the time-weighted target mean (or point) with model conditions
at issue. TargetTempMinDeltaC uses target minimum. Wet features compare the
minimum temperature, maximum RH, and largest absolute signed 10m wind-speed
change over signal intervals in issue..targetEnd with model conditions at issue.
These optional channels have independent missingness. All context3h values are
from the selected forecast vintage and explicitly modelled_not_observed; they
are never measured rainfall. Archive truncation may leave context3h incomplete.

Global availability requires a validated source/location/units, age <=2h and
complete rain/probability coverage for prior and during. Missing context3h or
optional meteorology alone does not invalidate otherwise covered rain. The
latest malformed run is reported unavailable rather than silently replaced by
an older vintage. Cache identities include float fetch timestamps, payload
content and exact issue/cutoff; same-count payload corrections invalidate them.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
from types import MappingProxyType
from typing import Mapping

VERSION = "ttdi_rain_weather_v1"
SOURCE = "Open-Meteo Best Match"
MAX_AGE_SECONDS = 7200.0
HOUR = 3600.0
EPSILON = 1e-6
INTERVAL_SUFFIXES = (
    "RainMm", "RainProbMax", "RainProbMean", "HighProbHours",
    "AmountWetHours", "WetSignalHours", "FirstWetLeadHours",
    "LastWetEndLeadHours",
)
GLOBAL_FEATURE_NAMES = (
    "weatherAgeHours", "priorHours", "duringHours", "firstWetLeadHours",
    "expectedWetEndLeadHours", "hoursUntilExpectedWetEnd", "targetTempDeltaC",
    "targetTempMinDeltaC", "targetRhDeltaPct", "targetWind10DeltaKmh",
    "targetWind925MeanKmh", "wetTempMinDeltaC", "wetRhMaxDeltaPct",
    "wetWind10DeltaKmh",
)
FEATURE_NAMES = GLOBAL_FEATURE_NAMES + tuple(
    prefix + suffix for prefix in ("prior", "during", "context3h")
    for suffix in INTERVAL_SUFFIXES
)
FIELDS = (
    "precipitation", "precipitation_probability", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "wind_speed_925hPa", "wind_direction_925hPa",
)
_UNITS = {
    "precipitation": {"mm"}, "precipitation_probability": {"%"},
    "temperature_2m": {"°C", "C", "celsius"},
    "relative_humidity_2m": {"%"},
    "wind_speed_10m": {"km/h"}, "wind_speed_925hPa": {"km/h"},
    "wind_direction_10m": {"°", "degree", "degrees"},
    "wind_direction_925hPa": {"°", "degree", "degrees"},
}
_LOCK = threading.RLock()
_PARSED_CACHE: OrderedDict = OrderedDict()
_ARCHIVE_CACHE: OrderedDict = OrderedDict()
_DESCRIPTION_CACHE: OrderedDict = OrderedDict()


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _epoch(value, name):
    number = _finite(value)
    if number is None:
        raise ValueError(f"{name} must be a finite epoch")
    return number


def _remember(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


@dataclass(frozen=True)
class WeatherRun:
    fetched_epoch: float
    payload_hash: str
    payload_json: str
    source: str
    epochs: tuple[float, ...]
    values: Mapping[str, tuple[float | None, ...]]
    validation_errors: tuple[str, ...]
    missing_optional_units: tuple[str, ...]
    latitude: float | None
    longitude: float | None


@dataclass(frozen=True)
class RunArchive(Sequence[WeatherRun]):
    runs: tuple[WeatherRun, ...]
    fetched_epochs: tuple[float, ...]
    cutoff_epoch: float
    revision: str
    database_path: str

    def __getitem__(self, index):
        return self.runs[index]

    def __len__(self):
        return len(self.runs)

    def __iter__(self) -> Iterator[WeatherRun]:
        return iter(self.runs)


def _parse_run(fetched, archive_source, payload):
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=20).hexdigest()
    key = (fetched.hex(), archive_source, digest)
    with _LOCK:
        if key in _PARSED_CACHE:
            _PARSED_CACHE.move_to_end(key)
            return _PARSED_CACHE[key]
    errors = []
    try:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("payload is not an object")
    except (ValueError, TypeError):
        data = {}
        errors.append("malformed_payload")
    source = str(data.get("source", ""))
    if source != SOURCE or archive_source != SOURCE:
        errors.append("invalid_source")
    if "fetchedEpoch" in data and _finite(data["fetchedEpoch"]) != fetched:
        errors.append("payload_fetch_timestamp_mismatch")
    latitude = _finite(data.get("latitude"))
    longitude = _finite(data.get("longitude"))
    if (latitude is None or longitude is None or
            abs(latitude - 3.1411) > .3 or abs(longitude - 101.6275) > .3):
        errors.append("invalid_coordinates")
    units = data.get("hourlyUnits") or data.get("hourly_units") or {}
    if not isinstance(units, dict):
        units = {}
    missing_optional = []
    for field, allowed in _UNITS.items():
        if field not in units and field not in ("precipitation", "precipitation_probability"):
            missing_optional.append(field)
        elif units.get(field) not in allowed:
            errors.append("invalid_unit:" + field)
    hourly = data.get("hourly", [])
    if not isinstance(hourly, list):
        hourly = []
        errors.append("invalid_hourly_rows")
    points = {}
    for point in hourly:
        if not isinstance(point, dict):
            errors.append("invalid_hourly_row")
            continue
        timestamp = _finite(point.get("epoch"))
        if timestamp is None:
            errors.append("invalid_hourly_epoch")
            continue
        if timestamp in points:
            errors.append("duplicate_hourly_epoch")
        points[timestamp] = point
    epochs = tuple(sorted(points))
    if not epochs:
        errors.append("empty_hourly_rows")
    if any(b - a < HOUR - EPSILON for a, b in zip(epochs, epochs[1:])):
        errors.append("overlapping_hourly_intervals")
    values = {}
    for field in FIELDS:
        column = []
        for timestamp in epochs:
            value = _finite(points[timestamp].get(field))
            if field in missing_optional:
                value = None
            if value is not None:
                if field in ("precipitation", "wind_speed_10m", "wind_speed_925hPa") and value < 0:
                    value = None
                if field in ("precipitation_probability", "relative_humidity_2m") and not 0 <= value <= 100:
                    value = None
                if field.startswith("wind_direction") and not 0 <= value <= 360:
                    value = None
            column.append(value)
        values[field] = tuple(column)
    run = WeatherRun(fetched, digest, payload, source, epochs,
                     MappingProxyType(values), tuple(dict.fromkeys(errors)),
                     tuple(missing_optional), latitude, longitude)
    with _LOCK:
        return _remember(_PARSED_CACHE, key, run, 4096)


def load_runs(db_path, issue_epoch) -> RunArchive:
    """Read-only archive snapshot; caches actual content, not row counts/mtime.

    For replay performance call this once at the last permissible issue and
    pass the returned archive to every historical describe_window invocation.
    Repeated loads reread/hash source bytes so a same-count correction is seen.
    No live SQLite connection survives this function.
    """
    cutoff = _epoch(issue_epoch, "issue_epoch")
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=20)) as conn:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            "SELECT fetched_epoch,source,payload FROM weather_forecast_runs "
            "WHERE fetched_epoch<=? ORDER BY fetched_epoch,payload", (cutoff,)
        ).fetchall()
    digest = hashlib.blake2b(digest_size=20)
    normalized = []
    for fetched, source, payload in rows:
        fetched = _finite(fetched)
        if fetched is None or fetched > cutoff:
            continue
        source = str(source)
        payload = str(payload)
        for part in (fetched.hex(), source, payload):
            encoded = part.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        normalized.append((fetched, source, payload))
    revision = digest.hexdigest()
    key = (str(path), cutoff.hex(), revision)
    with _LOCK:
        if key in _ARCHIVE_CACHE:
            _ARCHIVE_CACHE.move_to_end(key)
            return _ARCHIVE_CACHE[key]
    runs = tuple(_parse_run(*row) for row in normalized)
    archive = RunArchive(runs, tuple(run.fetched_epoch for run in runs), cutoff,
                         revision, str(path))
    with _LOCK:
        return _remember(_ARCHIVE_CACHE, key, archive, 8)


def latest_run(runs: RunArchive, issue_epoch) -> WeatherRun | None:
    issue = _epoch(issue_epoch, "issue_epoch")
    index = bisect_right(runs.fetched_epochs, issue) - 1
    return runs[index] if index >= 0 else None


def selected_payload(runs: RunArchive, issue_epoch) -> dict | None:
    """Fresh payload dict from the exact selected vintage; check describe first."""
    run = latest_run(runs, issue_epoch)
    if run is None:
        return None
    try:
        result = json.loads(run.payload_json)
    except ValueError:
        return None
    return result if isinstance(result, dict) else None


def _blocks(run, start, end):
    if end <= start:
        return []
    left = bisect_right(run.epochs, start)
    right = bisect_left(run.epochs, end + HOUR)
    return [(i, max(start, run.epochs[i] - HOUR), min(end, run.epochs[i]))
            for i in range(left, right)
            if min(end, run.epochs[i]) > max(start, run.epochs[i] - HOUR)]


def _coverage(covered, duration):
    missing = max(0.0, duration - covered)
    return {"complete": missing <= EPSILON, "coveredHours": covered / HOUR,
            "requiredHours": duration / HOUR, "missingHours": missing / HOUR}


def _is_wet(amount, probability):
    return amount >= .1 or probability >= 60.0


def _interval(run, issue, start, end):
    duration = end - start
    blocks = _blocks(run, start, end)
    amount = run.values["precipitation"]
    probability = run.values["precipitation_probability"]
    amount_blocks = [(i, a, b) for i, a, b in blocks if amount[i] is not None]
    prob_blocks = [(i, a, b) for i, a, b in blocks if probability[i] is not None]
    signal_blocks = [(i, a, b) for i, a, b in blocks
                     if amount[i] is not None and probability[i] is not None]
    wet_blocks = [(i, a, b) for i, a, b in signal_blocks
                  if _is_wet(amount[i], probability[i])]
    coverage = {key: _coverage(sum(b - a for _, a, b in records), duration)
                for key, records in (("amount", amount_blocks),
                                     ("probability", prob_blocks),
                                     ("signal", signal_blocks))}
    ac = coverage["amount"]["complete"]
    pc = coverage["probability"]["complete"]
    sc = coverage["signal"]["complete"]
    values = {
        "RainMm": sum(amount[i] * (b - a) / HOUR for i, a, b in amount_blocks) if ac else None,
        "RainProbMax": max((probability[i] for i, _, _ in prob_blocks), default=None) if pc else None,
        "RainProbMean": (sum(probability[i] * (b-a) for i, a, b in prob_blocks) / duration) if pc and duration else None,
        "HighProbHours": sum((b-a) / HOUR for i, a, b in prob_blocks if probability[i] >= 60) if pc else None,
        "AmountWetHours": sum((b-a) / HOUR for i, a, b in amount_blocks if amount[i] >= .1) if ac else None,
        "WetSignalHours": sum((b-a) / HOUR for _, a, b in wet_blocks) if sc else None,
        "FirstWetLeadHours": (wet_blocks[0][1]-issue) / HOUR if sc and wet_blocks else None,
        "LastWetEndLeadHours": (wet_blocks[-1][2]-issue) / HOUR if sc and wet_blocks else None,
    }
    values = {name: float(value) if value is not None else None
              for name, value in values.items()}
    return {"available": ac and pc, "startEpoch": start, "endEpoch": end,
            "durationHours": duration / HOUR, "coverage": coverage,
            "featureValues": values, "modelContext": "modelled_not_observed"}


def _sample(run, field, timestamp):
    values = run.values[field]
    right = bisect_left(run.epochs, timestamp)
    if right < len(run.epochs) and run.epochs[right] == timestamp:
        return values[right]
    left = right - 1
    if left < 0 or right >= len(run.epochs):
        return None
    if run.epochs[right] - run.epochs[left] > HOUR + EPSILON:
        return None
    a, b = values[left], values[right]
    if a is None or b is None:
        return None
    weight = (timestamp-run.epochs[left]) / (run.epochs[right]-run.epochs[left])
    return a + weight * (b-a)


def _path_stats(run, field, start, end):
    times = (start,) + tuple(run.epochs[bisect_right(run.epochs, start):
                                     bisect_left(run.epochs, end)])
    if end > start:
        times += (end,)
    values = tuple(_sample(run, field, t) for t in times)
    if any(v is None for v in values) or any(b-a > HOUR+EPSILON for a, b in zip(times, times[1:])):
        return None
    average = (sum((b-a)*(va+vb)/2 for a, b, va, vb in
                   zip(times, times[1:], values, values[1:])) / (end-start)
               if end > start else values[0])
    return average, min(values), max(values)


def _meteorology(run, issue, start, end, wet_blocks):
    values = {}
    missing = []
    for field, name in (("temperature_2m", "targetTempDeltaC"),
                        ("relative_humidity_2m", "targetRhDeltaPct"),
                        ("wind_speed_10m", "targetWind10DeltaKmh"),
                        ("wind_speed_925hPa", "targetWind925MeanKmh")):
        stats = _path_stats(run, field, start, end)
        baseline = _sample(run, field, issue)
        usable = stats is not None and (baseline is not None or field == "wind_speed_925hPa")
        values[name] = (stats[0] if field == "wind_speed_925hPa" else stats[0]-baseline) if usable else None
        if field == "temperature_2m":
            values["targetTempMinDeltaC"] = stats[1]-baseline if usable else None
        if not usable:
            missing.append(field)
    for field, name, mode in (("temperature_2m", "wetTempMinDeltaC", "min"),
                              ("relative_humidity_2m", "wetRhMaxDeltaPct", "max"),
                              ("wind_speed_10m", "wetWind10DeltaKmh", "change")):
        baseline = _sample(run, field, issue)
        paths = [_path_stats(run, field, a, b) for _, a, b in wet_blocks]
        if baseline is None or not paths or any(path is None for path in paths):
            values[name] = None
        elif mode == "min":
            values[name] = min(path[1] for path in paths)-baseline
        elif mode == "max":
            values[name] = max(path[2] for path in paths)-baseline
        else:
            deltas = [value-baseline for path in paths for value in path[1:]]
            values[name] = max(deltas, key=abs)
    return values, missing


def _wet_timing(run, issue, end, complete):
    if not complete:
        return [], {"status": "missing_rain_coverage"}, None
    amount = run.values["precipitation"]
    probability = run.values["precipitation_probability"]
    blocks = [(i, a, b) for i, a, b in _blocks(run, issue, end)
              if _is_wet(amount[i], probability[i])]
    if not blocks:
        return [], {"status": "no_wet_signal"}, None
    last = blocks[-1][0]
    cessation = None
    status = "cessation_right_censored"
    for index in range(last + 1, len(run.epochs)):
        if run.epochs[index]-run.epochs[index-1] > HOUR+EPSILON:
            status = "cessation_missing_hour"
            break
        a, p = amount[index], probability[index]
        if a is None or p is None:
            status = "cessation_missing_channel"
            break
        if not _is_wet(a, p):
            cessation = run.epochs[index-1]
            status = "cessation_known"
            break
    first_index = blocks[0][0]
    return blocks, {"status": status,
                    "firstWetSignalIntervalStartEpoch": run.epochs[first_index]-HOUR,
                    "firstWetSignalIntervalEndEpoch": run.epochs[first_index],
                    "expectedWetEndEpoch": cessation,
                    "onsetResolution": "hourly_signal_interval_not_exact_onset"}, cessation


def describe_window(runs: RunArchive, issue_epoch, start_epoch, end_epoch) -> dict:
    """JSON-safe features for one actual issue and point/window target."""
    issue = _epoch(issue_epoch, "issue_epoch")
    start = _epoch(start_epoch, "start_epoch")
    end = _epoch(end_epoch, "end_epoch")
    if start < issue or end < start:
        raise ValueError("require issue_epoch <= start_epoch <= end_epoch")
    run = latest_run(runs, issue)
    source_key = None if run is None else (run.fetched_epoch.hex(), run.payload_hash, run.validation_errors)
    key = (VERSION, source_key, issue.hex(), start.hex(), end.hex(), issue > runs.cutoff_epoch)
    with _LOCK:
        if key in _DESCRIPTION_CACHE:
            _DESCRIPTION_CACHE.move_to_end(key)
            return deepcopy(_DESCRIPTION_CACHE[key])
    features = {name: None for name in FEATURE_NAMES}
    features.update(priorHours=(start-issue)/HOUR, duringHours=(end-start)/HOUR)
    errors = []
    if issue > runs.cutoff_epoch:
        errors.append("issue_after_archive_cutoff")
    if run is None:
        errors.append("no_run_at_issue")
    else:
        features["weatherAgeHours"] = (issue-run.fetched_epoch)/HOUR
        errors.extend(run.validation_errors)
        if issue-run.fetched_epoch > MAX_AGE_SECONDS:
            errors.append("stale_weather_run")
    metadata = {
        "version": VERSION, "forecastIssuedEpoch": issue,
        "windowStartEpoch": start, "windowEndEpoch": end,
        "source": run.source if run else None,
        "fetchedEpoch": run.fetched_epoch if run else None,
        "payloadHash": run.payload_hash if run else None,
        "selectedBy": "latest_fetched_at_or_before_actual_issue",
        "maxWeatherAgeSeconds": MAX_AGE_SECONDS,
        "precipitationSemantics": "preceding_hour_amount_overlap_weighted",
        "probabilitySemantics": "preceding_hour_probability_duration_weighted_not_event_union",
        "context3hType": "modelled_not_observed",
        "missingValues": "null_not_zero_imputed",
        "errors": errors,
    }
    result = {"available": False, "fetchedEpoch": run.fetched_epoch if run else None,
              "featureValues": features, "metadata": metadata,
              "prior": None, "during": None, "context3h": None}
    if not errors:
        for prefix, left, right in (("prior", issue, start), ("during", start, end),
                                    ("context3h", issue-3*HOUR, issue)):
            interval = _interval(run, issue, left, right)
            result[prefix] = interval
            features.update({prefix+name: value for name, value in interval["featureValues"].items()})
        result["available"] = result["prior"]["available"] and result["during"]["available"]
        if not result["available"]:
            errors.append("incomplete_target_rain_coverage")
        wet_blocks, wet_timing, cessation = _wet_timing(run, issue, end, result["available"])
        metadata["wetTiming"] = wet_timing
        metadata["context3hAvailable"] = result["context3h"]["available"]
        metadata["missingOptionalUnits"] = list(run.missing_optional_units)
        if wet_blocks:
            features["firstWetLeadHours"] = (wet_blocks[0][1]-issue)/HOUR
        if cessation is not None:
            features["expectedWetEndLeadHours"] = (cessation-issue)/HOUR
            features["hoursUntilExpectedWetEnd"] = (cessation-start)/HOUR
        meteo, missing_meteo = _meteorology(run, issue, start, end, wet_blocks)
        features.update(meteo)
        metadata["missingTargetMeteorology"] = missing_meteo
    with _LOCK:
        _remember(_DESCRIPTION_CACHE, key, deepcopy(result), 16384)
    return result


def describe_windows(runs: RunArchive, windows: Iterable[tuple[float, float, float]]) -> list[dict]:
    """Batch feature pass using a shared parsed archive and bisection selection."""
    return [describe_window(runs, issue, start, end) for issue, start, end in windows]


def clear_caches():
    """Release in-memory caches; no source files or database state are changed."""
    with _LOCK:
        _PARSED_CACHE.clear()
        _ARCHIVE_CACHE.clear()
        _DESCRIPTION_CACHE.clear()
