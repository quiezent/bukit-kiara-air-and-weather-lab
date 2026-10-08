"""Immutable issued CAMS spatial PM inputs for future evaluated candidates.

Only the explicit collector functions fetch or write. Archive readers and
diagnostics never fetch, mutate an archive, or select a published forecast.
The nine cells are one coarse modeled field, not nine independent stations.
Hourly API interpolation does not improve CAMS Global's three-hour source
resolution. Fetch time establishes availability, not the model run's origin.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import closing
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zlib


VERSION = "regional_cams_pm_grid_v1"
FEATURE_VERSION = "regional_cams_spatial_diagnostic_v1"
TARGET = (3.1411106257487, 101.62749852676)
GRID_SPACING_DEGREES = .4
MAX_AGE_SECONDS = 7200
POLL_SECONDS = 3600
FORECAST_HOURS = 72
HTTP_TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"
DOCUMENTATION_URL = "https://open-meteo.com/en/docs/air-quality-api"
SOURCE = "Open-Meteo / CAMS Global"
DATA_ROLE = "issued_regional_modeled_pm_snapshot"
FIELDS = ("pm2_5", "pm10")
PM_UNITS = frozenset(("μg/m³", "µg/m³", "ug/m3"))
COORDINATE_TOLERANCE = .21001
GRID_TOLERANCE = .0001


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _epoch(value):
    number = _finite(value)
    if number is None or number <= 0 or number != int(number):
        raise ValueError("positive integer UTC epoch required")
    return int(number)


def _encode(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("regional PM response exceeds size limit")
    return encoded


def requested_grid(center=TARGET):
    """Return nine row-major requests, on a snapped 0.4-degree lattice."""
    lat, lon = (_finite(item) for item in center)
    if lat is None or lon is None or not -89 <= lat <= 89 or not -179 <= lon <= 179:
        raise ValueError("finite nonpolar coordinates required")
    lat = round(lat / GRID_SPACING_DEGREES) * GRID_SPACING_DEGREES
    lon = round(lon / GRID_SPACING_DEGREES) * GRID_SPACING_DEGREES
    return tuple((round(lat + i * GRID_SPACING_DEGREES, 7),
                  round(lon + j * GRID_SPACING_DEGREES, 7))
                 for i in (-1, 0, 1) for j in (-1, 0, 1))


def request_url(locations=None):
    locations = requested_grid() if locations is None else locations
    params = {
        "latitude": ",".join(str(p[0]) for p in locations),
        "longitude": ",".join(str(p[1]) for p in locations),
        "hourly": ",".join(FIELDS), "domains": "cams_global",
        "cell_selection": "nearest", "timezone": "GMT",
        "timeformat": "unixtime", "forecast_hours": FORECAST_HOURS,
    }
    return ENDPOINT + "?" + urlencode(params)


def source_metadata():
    return {
        "source": SOURCE, "documentationUrl": DOCUMENTATION_URL,
        "sourceDataRole": DATA_ROLE, "modelDomain": "cams_global",
        "sourceGridSpacingDegrees": GRID_SPACING_DEGREES,
        "sourceSpatialResolutionKmApprox": 45,
        "sourceTemporalResolutionHours": 3, "sourceUpdateHours": 12,
        "apiTemporalResolutionHours": 1,
        "apiHourlyValuesRole": "interpolated_model_output_not_independent_hourly_measurements",
        "modelRunEpoch": None,
        "modelRunEpochAvailability": "not_supplied_by_this_API_response",
        "concentrationUnits": "µg/m³", "concentrationTimeRole": "instantaneous",
        "measurement": False, "independentStationCount": 0,
        "appliedToPrimaryForecast": False, "prospectivelyValidated": False,
    }


def _check_time_axis(values):
    if not isinstance(values, list) or not 3 <= len(values) <= FORECAST_HOURS + 1:
        raise ValueError("regional PM time axis needs 3 to 73 hourly epochs")
    epochs = tuple(_epoch(value) for value in values)
    if any(epoch % 3600 for epoch in epochs) or any(
            right - left != 3600 for left, right in zip(epochs, epochs[1:])):
        raise ValueError("regional PM epochs must be strictly increasing UTC hours")
    return epochs


def build_snapshot(raw_response, fetched_epoch, locations=None):
    """Validate one raw multi-coordinate response and retain its provenance."""
    fetched = _epoch(fetched_epoch)
    requested = requested_grid() if locations is None else tuple(tuple(p) for p in locations)
    if len(requested) != 9 or len(set(requested)) != 9:
        raise ValueError("nine unique requested coordinates required")
    if not isinstance(raw_response, list) or len(raw_response) != 9:
        raise ValueError("regional PM response must contain nine locations")
    _encode(raw_response)
    times, returned = None, []
    for index, raw in enumerate(raw_response):
        if not isinstance(raw, dict) or raw.get("error"):
            raise ValueError("invalid regional PM location response")
        lat, lon = (_finite(raw.get(field)) for field in ("latitude", "longitude"))
        if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("invalid returned model coordinates")
        wanted_lat, wanted_lon = requested[index]
        if (_finite(wanted_lat) is None or _finite(wanted_lon) is None
                or abs(lat - wanted_lat) > COORDINATE_TOLERANCE
                or abs(lon - wanted_lon) > COORDINATE_TOLERANCE):
            raise ValueError("returned location does not match its request position")
        if raw.get("utc_offset_seconds", 0) != 0:
            raise ValueError("regional PM epochs must use GMT")
        hourly, units = raw.get("hourly"), raw.get("hourly_units")
        if not isinstance(hourly, dict) or not isinstance(units, dict):
            raise ValueError("missing regional PM hourly data or units")
        if units.get("time") != "unixtime":
            raise ValueError("regional PM time units must be unixtime")
        point_times = _check_time_axis(hourly.get("time"))
        if times is not None and times != point_times:
            raise ValueError("regional PM locations have different time axes")
        times = point_times
        for field in FIELDS:
            if units.get(field) not in PM_UNITS:
                raise ValueError("invalid or missing regional PM units: " + field)
            values = hourly.get(field)
            if not isinstance(values, list) or len(values) != len(times):
                raise ValueError("invalid or missing regional PM field: " + field)
            if any(value is not None and (_finite(value) is None or float(value) < 0)
                   for value in values):
                raise ValueError("invalid regional PM concentration: " + field)
        returned.append((lat, lon))
    # Coordinates can be offset from our requested lattice. Interpolation uses
    # the actual returned cell centres and requires a regular complete grid.
    rounded = {(round(lat, 5), round(lon, 5)) for lat, lon in returned}
    if len(rounded) != 9:
        raise ValueError("duplicate returned model cells")
    lats = tuple(returned[row * 3][0] for row in range(3))
    lons = tuple(returned[column][1] for column in range(3))
    if any(abs(b - a - GRID_SPACING_DEGREES) > GRID_TOLERANCE
           for axis in (lats, lons) for a, b in zip(axis, axis[1:])):
        raise ValueError("returned model cells are not a regular ordered 0.4-degree grid")
    for index, (lat, lon) in enumerate(returned):
        if (abs(lat - lats[index // 3]) > GRID_TOLERANCE
                or abs(lon - lons[index % 3]) > GRID_TOLERANCE):
            raise ValueError("returned model cells are not row-major")
    snapshot = {
        "version": VERSION, "fetchedEpoch": fetched,
        "metadata": source_metadata(), "requestUrl": request_url(requested),
        "requestForecastHours": FORECAST_HOURS, "requestHistoryHours": 0,
        "requestedLocations": [list(point) for point in requested],
        "returnedLocations": [list(point) for point in returned],
        "returnedLatitudes": list(lats), "returnedLongitudes": list(lons),
        "times": list(times), "rawResponse": raw_response,
    }
    # Detach the immutable archive snapshot from a caller's mutable response.
    return json.loads(_encode(snapshot))


def _validated_snapshot(payload):
    if not isinstance(payload, dict):
        raise ValueError("regional PM snapshot must be an object")
    rebuilt = build_snapshot(payload.get("rawResponse"), payload.get("fetchedEpoch"),
                             payload.get("requestedLocations"))
    if _encode(rebuilt) != _encode(payload):
        raise ValueError("regional PM snapshot provenance or version mismatch")
    return rebuilt


def init_archive(db_path):
    """Explicit archive writer; does not run on import or reader access."""
    with closing(sqlite3.connect(db_path, timeout=20)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS regional_pm_runs("
                     "fetched_epoch INTEGER PRIMARY KEY, model_version TEXT NOT NULL, "
                     "payload_sha256 TEXT NOT NULL, payload BLOB NOT NULL, "
                     "recorded_epoch INTEGER NOT NULL)")


def save_snapshot(db_path, payload, recorded_epoch=None, clock=time.time):
    """Insert an issued snapshot once; conflicts can never replace old inputs."""
    snapshot = _validated_snapshot(payload)
    encoded = _encode(snapshot)
    checksum = hashlib.sha256(encoded).hexdigest()
    blob = gzip.compress(encoded, compresslevel=6, mtime=0)
    # Archive receipt is distinct from network completion. A delayed manual
    # import must not masquerade as an input to earlier service forecasts.
    with closing(sqlite3.connect(db_path, timeout=20)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT model_version,payload_sha256 FROM regional_pm_runs "
                                "WHERE fetched_epoch=?", (snapshot["fetchedEpoch"],)).fetchone()
        if existing:
            if existing != (VERSION, checksum):
                raise ValueError("regional PM timestamp already contains a different immutable snapshot")
            return False
        # Sample receipt only after the writer lock: waiting for a busy DB
        # must not make this input appear available before archive insertion.
        recorded = math.ceil(clock()) if recorded_epoch is None else math.ceil(_finite(recorded_epoch) or 0)
        if recorded <= 0:
            raise ValueError("positive archive recording epoch required")
        conn.execute("INSERT INTO regional_pm_runs VALUES(?,?,?,?,?)",
                     (snapshot["fetchedEpoch"], VERSION, checksum, blob, recorded))
        conn.commit()
    return True


@dataclass(frozen=True)
class RegionalPMRun:
    fetched_epoch: int
    model_version: str
    payload_hash: str
    payload_json: str
    payload_blob: bytes
    recorded_epoch: int

    @property
    def available_epoch(self):
        return max(self.fetched_epoch, self.recorded_epoch)


@dataclass(frozen=True)
class RegionalPMArchive:
    runs: tuple[RegionalPMRun, ...]
    fetched_epochs: tuple[int, ...]
    cutoff_epoch: float
    revision: str
    database_path: str
    archive_error: str | None = None
    invalid_run_count: int = 0


def _uncompress(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) > MAX_RESPONSE_BYTES:
        raise ValueError("invalid regional PM compressed payload")
    with gzip.GzipFile(fileobj=io.BytesIO(bytes(blob)), mode="rb") as stream:
        encoded = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("regional PM decompressed payload exceeds size limit")
    return encoded


def load_runs(db_path, cutoff_epoch, lookback_days=30):
    """Load a bounded immutable archive, using a read-only SQLite connection."""
    cutoff, days = _finite(cutoff_epoch), _finite(lookback_days)
    if cutoff is None or days is None or days <= 0:
        raise ValueError("finite cutoff and positive lookback required")
    path = Path(db_path).resolve()
    raw, error = [], None
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=20)) as conn:
            raw = conn.execute("SELECT fetched_epoch,model_version,payload_sha256,payload,recorded_epoch "
                               "FROM regional_pm_runs WHERE fetched_epoch<=? AND recorded_epoch<=? "
                               "AND fetched_epoch>=? ORDER BY fetched_epoch",
                               (cutoff, cutoff, cutoff - days * 86400 - MAX_AGE_SECONDS)).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc) or not path.exists():
            error = "missing_regional_pm_archive"
        else:
            raise
    runs, invalid, digest = [], 0, hashlib.sha256()
    for fetched, version, checksum, blob, recorded in raw:
        try:
            if version != VERSION:
                raise ValueError("unsupported regional PM archive version")
            encoded = _uncompress(blob)
            if hashlib.sha256(encoded).hexdigest() != checksum:
                raise ValueError("regional PM checksum mismatch")
            snapshot = _validated_snapshot(json.loads(encoded))
            if snapshot["fetchedEpoch"] != fetched:
                raise ValueError("regional PM fetch timestamp mismatch")
            recorded = _epoch(recorded)
            runs.append(RegionalPMRun(int(fetched), version, checksum,
                                      encoded.decode("utf-8"), bytes(blob), recorded))
            digest.update(repr((fetched, recorded, version, checksum)).encode())
        except (ValueError, TypeError, OSError, EOFError, UnicodeError, zlib.error):
            invalid += 1
    return RegionalPMArchive(tuple(runs), tuple(run.fetched_epoch for run in runs),
                             cutoff, digest.hexdigest(), str(path), error, invalid)


def latest_run(archive, issue_epoch):
    """Latest validated fetch actually available at issue, no older than 2h."""
    issue = _finite(issue_epoch)
    if issue is None:
        raise ValueError("finite issue epoch required")
    # An older loaded archive cannot establish which fetch was latest later.
    if issue > archive.cutoff_epoch:
        return None
    index = bisect_right(archive.fetched_epochs, issue) - 1
    while index >= 0 and archive.runs[index].available_epoch > issue:
        index -= 1
    run = archive.runs[index] if index >= 0 else None
    return run if run is not None and issue - run.fetched_epoch <= MAX_AGE_SECONDS else None


def decoded_payload(run):
    return json.loads(run.payload_json)


def _bracket(axis, value):
    if value < axis[0] - 1e-7 or value > axis[-1] + 1e-7:
        return None
    value = min(axis[-1], max(axis[0], value))
    index = min(len(axis) - 2, max(0, bisect_right(axis, value) - 1))
    return index, (value - axis[index]) / (axis[index + 1] - axis[index])


def interpolate(run, epoch, latitude, longitude, field="pm2_5"):
    """Bilinear actual-cell PM and linear time interpolation, without filling."""
    epoch, latitude, longitude = (_finite(item) for item in (epoch, latitude, longitude))
    if any(item is None for item in (epoch, latitude, longitude)) or field not in FIELDS:
        return None
    payload = decoded_payload(run)
    brackets = [_bracket(payload[key], value) for key, value in
                (("times", epoch), ("returnedLatitudes", latitude),
                 ("returnedLongitudes", longitude))]
    if any(item is None for item in brackets):
        return None
    (ti, tw), (yi, yw), (xi, xw) = brackets
    total = 0.
    for t, wt in ((ti, 1 - tw), (ti + 1, tw)):
        for y, wy in ((yi, 1 - yw), (yi + 1, yw)):
            for x, wx in ((xi, 1 - xw), (xi + 1, xw)):
                weight = wt * wy * wx
                if weight <= 1e-12:
                    continue
                value = payload["rawResponse"][y * 3 + x]["hourly"][field][t]
                if value is None:
                    return None
                total += weight * float(value)
    return total


def fetch_snapshot(centre_latlon=TARGET, fetcher=None, clock=time.time):
    """Explicit bounded live request. Injectable fetch_json-compatible client."""
    locations = requested_grid(centre_latlon)
    url = request_url(locations)
    if fetcher is None:
        request = Request(url, headers={"User-Agent": "TTDIAirDashboard/regional-pm-v1"})
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            encoded = response.read(MAX_RESPONSE_BYTES + 1)
        if len(encoded) > MAX_RESPONSE_BYTES:
            raise ValueError("regional PM response exceeds size limit")
        raw = json.loads(encoded)
    else:
        raw = fetcher(url, timeout=HTTP_TIMEOUT_SECONDS)
    # Round availability upwards: fractional response completion must never be
    # made available to an issue fractionally before the body was received.
    return build_snapshot(raw, math.ceil(clock()), locations)


def poll_once(db_path, centre_latlon=TARGET, fetcher=None, clock=time.time, force=False):
    """Collector-only fetch/write, deduplicated using persisted fetch time."""
    init_archive(db_path)
    # Collector scheduling may use this conservative rounded clock. Forecast
    # readers still select against their own exact, unrounded issue time.
    now = math.ceil(clock())
    archived = latest_run(load_runs(db_path, now, lookback_days=1), now)
    if not force and archived is not None and now - archived.fetched_epoch < POLL_SECONDS:
        return {"status": "recent_snapshot", "inserted": False,
                "fetchedEpoch": archived.fetched_epoch, "payloadSha256": archived.payload_hash}
    snapshot = fetch_snapshot(centre_latlon, fetcher, clock)
    inserted = save_snapshot(db_path, snapshot, clock=clock)
    return {"status": "archived" if inserted else "duplicate", "inserted": inserted,
            "fetchedEpoch": snapshot["fetchedEpoch"],
            "payloadSha256": hashlib.sha256(_encode(snapshot)).hexdigest()}


def collector(db_path, centre_latlon=TARGET, stop_event=None):
    """Independent hourly collector; restart checks the persisted archive."""
    while stop_event is None or not stop_event.is_set():
        delay = POLL_SECONDS
        try:
            result = poll_once(db_path, centre_latlon)
            age = max(0., time.time() - result["fetchedEpoch"])
            delay = max(30., POLL_SECONDS - age)
            print("[Regional PM]", result["status"], result["fetchedEpoch"], flush=True)
        except Exception as error:
            delay = 300
            print("[Regional PM] ERROR:", error, flush=True)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            wait = min(5., max(0., deadline - time.monotonic()))
            if stop_event is not None:
                if stop_event.wait(wait):
                    return
            else:
                time.sleep(wait)


def _wind_vectors(weather_wind, issue):
    if not isinstance(weather_wind, dict):
        return None
    fetched = _finite(weather_wind.get("fetchedEpoch"))
    if fetched is None or not 0 <= issue - fetched <= MAX_AGE_SECONDS:
        return None
    if weather_wind.get("source") not in ("Open-Meteo Best Match", "Open-Meteo"):
        return None
    units = weather_wind.get("hourlyUnits") or weather_wind.get("hourly_units") or {}
    if units.get("wind_speed_10m") != "km/h" or units.get("wind_direction_10m") not in ("°", "degree", "degrees"):
        return None
    hourly = weather_wind.get("hourly")
    if isinstance(hourly, dict):
        times = hourly.get("time", [])
        speeds, directions = hourly.get("wind_speed_10m", []), hourly.get("wind_direction_10m", [])
        if len(times) != len(speeds) or len(times) != len(directions):
            return None
    elif isinstance(hourly, list):
        times = [row.get("epoch") for row in hourly if isinstance(row, dict)]
        speeds = [row.get("wind_speed_10m") for row in hourly if isinstance(row, dict)]
        directions = [row.get("wind_direction_10m") for row in hourly if isinstance(row, dict)]
    else:
        return None
    # Weather archives may contain more than 73 hours; validate their axis here.
    numbers = tuple(_finite(value) for value in times)
    if len(numbers) < 3 or any(value is None or value != int(value) for value in numbers):
        return None
    if any(b - a != 3600 for a, b in zip(numbers, numbers[1:])):
        return None
    vectors = []
    for speed, direction in zip(speeds, directions):
        speed, direction = _finite(speed), _finite(direction)
        if speed is None or direction is None or speed < 0 or not 0 <= direction <= 360:
            vectors.append(None)
        else:
            radians = math.radians(direction)
            vectors.append((-speed * math.sin(radians), -speed * math.cos(radians)))
    return numbers, tuple(vectors), fetched


def _wind_at(times, vectors, epoch):
    bracket = _bracket(times, epoch)
    if bracket is None:
        return None
    index, weight = bracket
    result = [0., 0.]
    for position, factor in ((index, 1 - weight), (index + 1, weight)):
        if factor <= 1e-12:
            continue
        if vectors[position] is None:
            return None
        for axis in (0, 1):
            result[axis] += factor * vectors[position][axis]
    return tuple(result)


def describe(archive, weather_wind, issue_epoch, start_epoch, end_epoch, origin=TARGET):
    """Coarse upwind displacement/gradient diagnostic, never a PM forecast.

    Integrate issued local 10m vectors to the target midpoint, then sample the
    modeled PM field at issue at that upwind point. This straight displacement
    proxy has no dispersion, mixing, deposition or spatial wind evolution.
    CAMS already models transport; the proxy must not be added to CAMS PM.
    """
    issue, start, end = (_finite(item) for item in (issue_epoch, start_epoch, end_epoch))
    base = {"available": False, "featureVersion": FEATURE_VERSION,
            "appliedToPrimaryForecast": False, "metadata": source_metadata()}
    if any(item is None for item in (issue, start, end)) or not issue <= start <= end:
        return {**base, "reason": "invalid_diagnostic_clock"}
    run = latest_run(archive, issue)
    if run is None:
        return {**base, "reason": "missing_or_stale_as_issued_regional_pm"}
    winds = _wind_vectors(weather_wind, issue)
    if winds is None:
        return {**base, "reason": "missing_or_stale_as_issued_10m_wind"}
    wind_lat, wind_lon = (_finite(weather_wind.get(key)) for key in ("latitude", "longitude"))
    if (wind_lat is None or wind_lon is None or abs(wind_lat - origin[0]) > .3
            or abs(wind_lon - origin[1]) > .3):
        return {**base, "reason": "10m_wind_coordinates_do_not_match_origin"}
    times, vectors, wind_fetched = winds
    target = (start + end) / 2
    breaks = [issue] + [epoch for epoch in times if issue < epoch < target] + [target]
    east, north = 0., 0.
    for left, right in zip(breaks, breaks[1:]):
        a, b = _wind_at(times, vectors, left), _wind_at(times, vectors, right)
        if a is None or b is None:
            return {**base, "reason": "incomplete_issued_10m_wind_path"}
        for axis in (0, 1):
            displacement = (a[axis] + b[axis]) * .5 * (right - left) / 3600
            if axis == 0:
                east += displacement
            else:
                north += displacement
    lat, lon = origin
    upwind_lat = lat - north / 111.32
    upwind_lon = lon - east / (111.32 * math.cos(math.radians(lat)))
    local = interpolate(run, issue, lat, lon)
    upwind = interpolate(run, issue, upwind_lat, upwind_lon)
    target_local = interpolate(run, target, lat, lon)
    if local is None or upwind is None or target_local is None:
        return {**base, "reason": "upwind_point_outside_grid_or_missing_concentration"}
    return {**base, "available": True, "fetchedEpoch": run.fetched_epoch,
            "recordedEpoch": run.recorded_epoch, "availableEpoch": run.available_epoch,
            "payloadSha256": run.payload_hash, "windFetchedEpoch": wind_fetched,
            "issueEpoch": issue, "targetRepresentativeEpoch": target,
            "featureValues": {"modeledLocalPm25AtIssue": local,
                              "modeledUpwindPm25AtIssue": upwind,
                              "modeledUpwindMinusLocalPm25": upwind - local,
                              "modeledLocalPm25AtTarget": target_local,
                              "proxyDisplacementEastKm": east,
                              "proxyDisplacementNorthKm": north,
                              "proxyDistanceKm": math.hypot(east, north),
                              "proxyTravelHours": (target - issue) / 3600},
            "upwindPoint": {"latitude": upwind_lat, "longitude": upwind_lon},
            "proxyRole": "issued_10m_wind_displacement_in_coarse_modeled_PM_field",
            "limitations": "No identified plume or source; local uniform-wind displacement, not a resolved air trajectory or measured arrival time."}


def describe_context(db_path, issue_epoch, origin=TARGET, weather_wind=None):
    """Small read-only spatial input status for the dashboard/API."""
    issue = _finite(issue_epoch)
    if issue is None:
        raise ValueError("finite issue epoch required")
    archive = load_runs(db_path, issue, lookback_days=1)
    run = latest_run(archive, issue)
    base = {"available": False, "modelVersion": VERSION,
            "featureVersion": FEATURE_VERSION, "sourceIssueEpoch": issue,
            "appliedToPrimaryForecast": False, "metadata": source_metadata(),
            "invalidArchivedRuns": archive.invalid_run_count}
    if run is None:
        return {**base, "reason": archive.archive_error or "missing_or_stale_as_issued_regional_pm"}
    payload = decoded_payload(run)
    lat, lon = origin
    lats, lons = payload["returnedLatitudes"], payload["returnedLongitudes"]
    local = interpolate(run, issue, lat, lon)
    cardinal = {"west": interpolate(run, issue, lat, lons[0]),
                "east": interpolate(run, issue, lat, lons[-1]),
                "south": interpolate(run, issue, lats[0], lon),
                "north": interpolate(run, issue, lats[-1], lon)}
    if local is None or any(value is None for value in cardinal.values()):
        return {**base, "fetchedEpoch": run.fetched_epoch,
                "payloadSha256": run.payload_hash,
                "reason": "missing_issue_time_spatial_concentration"}
    east_distance = (lons[-1] - lons[0]) * 111.32 * math.cos(math.radians(lat))
    north_distance = (lats[-1] - lats[0]) * 111.32
    transport = describe(archive, weather_wind, issue, issue + 5400, issue + 5400, origin)
    return {**base, "available": True, "fetchedEpoch": run.fetched_epoch,
            "recordedEpoch": run.recorded_epoch, "availableEpoch": run.available_epoch,
            "ageSeconds": issue - run.fetched_epoch, "payloadSha256": run.payload_hash,
            "requestedCellCount": 9, "uniqueReturnedModelCellCount": 9,
            "requestedLocations": payload["requestedLocations"],
            "returnedLocations": payload["returnedLocations"],
            "timeCoverage": {"startEpoch": payload["times"][0], "endEpoch": payload["times"][-1]},
            "featureValues": {"modeledPm25AtOrigin": local,
                              "modeledGradientEastUgM3PerKm": (cardinal["east"] - cardinal["west"]) / east_distance,
                              "modeledGradientNorthUgM3PerKm": (cardinal["north"] - cardinal["south"]) / north_distance},
            "concentrationRole": "modeled_spatial_field_not_station_measurements_or_local_forecast",
            "transportProxy": transport}
