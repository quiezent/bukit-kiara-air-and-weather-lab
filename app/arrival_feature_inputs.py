"""Shared arrival-model inputs: fresh sensors and confirmed issued weather.

No targets, learner fitting, imputation, fetching, numerical forecast selection,
or archive writes occur here. Neighbor HTTP/request records are diagnostic only:
the current archive has no observation insertion/commit-confirmation clock.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

import fresh_sensor_features as sensor
import observation_provenance as receipts

VERSION = "arrival_fresh_sensor_confirmed_issued_weather_v1"
SENSOR_FEATURE_COLUMNS = sensor.FEATURE_COLUMNS
WEATHER_LEADS_MINUTES = (0, 30, 60, 90)
WEATHER_VALUE_FIELDS = (
    "temperature_2m", "relative_humidity_2m", "precipitation_probability",
    "precipitation", "wind_speed_10m", "wind_gusts_10m", "pressure_msl",
    "boundary_layer_height", "cape", "wind_speed_180m", "wind_speed_925hPa",
)
WEATHER_DIRECTION_FIELDS = (
    "wind_direction_10m", "wind_direction_180m", "wind_direction_925hPa",
)
WEATHER_POINT_FIELDS = (*WEATHER_VALUE_FIELDS,
                        *(name + suffix for name in WEATHER_DIRECTION_FIELDS for suffix in ("_sin", "_cos")))
WEATHER_REVISION_FIELDS = ("temperature_2m", "relative_humidity_2m",
                           "precipitation_probability", "wind_speed_10m")
WEATHER_REVISION_MINUTES = (30, 60)
WEATHER_FEATURE_COLUMNS = (
    *(f"issued_weather_{field}_{lead}min" for lead in WEATHER_LEADS_MINUTES for field in WEATHER_POINT_FIELDS),
    "issued_weather_fetch_age_minutes",
    *(f"issued_weather_arrival_revision_{field}_{age}min" for age in WEATHER_REVISION_MINUTES
      for field in WEATHER_REVISION_FIELDS),
)
FEATURE_COLUMNS = (*SENSOR_FEATURE_COLUMNS, *WEATHER_FEATURE_COLUMNS)
MAX_WEATHER_AGE_SECONDS = 120 * 60
NEIGHBOR_MAX_AGE_SECONDS = 90 * 60
NEIGHBOR_STATIONS = (
    ("mont_kiara", 99667152), ("setapak", 109266655), ("klcc", 111145052),
    ("setia_eco_park", 1087), ("enviro_exceltech", 98982064), ("cyberjaya", 52426946),
)
NEIGHBOR_IDS = dict(NEIGHBOR_STATIONS)
NEIGHBOR_POLICY = "excluded_no_observation_commit_confirmation"


def _number(value):
    if isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _issue(value):
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise ValueError("Issue timestamp requires a timezone")
        value = value.timestamp()
    value = _number(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Issue epoch must be positive and finite")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _decode_payload(value):
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, bytes):
        if value.startswith(b"\x1f\x8b"):
            value = gzip.decompress(value)
        value = value.decode("utf-8")
    return json.loads(value)


def _confirmed_weather_clock(row):
    provenance = row.get("_provenance") or row.get("provenance") or {}
    sources = (row, provenance)
    clocks = {}
    for canonical, names in {
        "received": ("receivedEpoch", "received_epoch"),
        "persisted": ("persistedEpoch", "persisted_epoch", "recordedEpoch", "recorded_epoch"),
        "confirmed": ("confirmedEpoch", "confirmed_epoch"),
        "available": ("availableEpoch", "available_epoch", "confirmed_available_epoch"),
    }.items():
        values = [_number(source.get(name)) for source in sources for name in names if name in source]
        finite = [value for value in values if math.isfinite(value) and value > 0]
        if not finite or len(finite) != len(values):
            return None
        clocks[canonical] = max(finite)
    # Confirmation is a post-first-commit proof, not a substituted fetch clock.
    if clocks["confirmed"] < max(clocks["received"], clocks["persisted"]):
        return None
    return max(clocks.values())


@dataclass(frozen=True)
class WeatherRun:
    fetched: float
    available: float
    receipt_id: int
    revision_id: str | None
    payload_sha256: str
    epochs: np.ndarray
    values: np.ndarray
    intervals: np.ndarray


@dataclass(frozen=True)
class PreparedInputs:
    sensorRows: tuple
    sensor: sensor.PreparedSensorRows
    weather: tuple[WeatherRun, ...]
    weather_fetched: np.ndarray
    neighbor_rows: tuple
    neighbor_ends: np.ndarray
    neighbor_metadata: tuple
    archive_metadata: dict


def prepare_inputs(sensor_rows, source_weather_rows=(), neighbor_rows=(), neighbor_metadata=(), *, archive_metadata=None):
    """Prepare detached ordered snapshots once; each issue rechecks visibility."""
    if isinstance(sensor_rows, PreparedInputs):
        if source_weather_rows or neighbor_rows or neighbor_metadata:
            raise ValueError("Cannot add rows to an already prepared snapshot")
        return sensor_rows
    detached = tuple(deepcopy(dict(row)) for row in sensor_rows)
    weather = []
    rejected = 0
    for original in source_weather_rows:
        row = dict(original)
        try:
            available = _confirmed_weather_clock(row)
            payload = _decode_payload(row.get("payload", row.get("payload_blob")))
            fetched = _number(row.get("fetched_epoch", row.get("source_fetched_epoch", payload.get("fetchedEpoch"))))
            if (available is None or not math.isfinite(fetched) or fetched <= 0
                    or available < fetched
                    or _number(payload.get("fetchedEpoch")) != fetched):
                rejected += 1
                continue
            points = {}
            for point in payload.get("hourly", []):
                valid = _number(point.get("epoch"))
                if math.isfinite(valid) and valid > 0:
                    points[valid] = point
            if not points:
                rejected += 1
                continue
            epochs = np.asarray(sorted(points), dtype=float)
            matrix = np.full((len(epochs), len(WEATHER_POINT_FIELDS)), np.nan)
            intervals = np.full(len(epochs), np.nan)
            for index, epoch in enumerate(epochs):
                point = points[epoch]
                matrix[index, :len(WEATHER_VALUE_FIELDS)] = [_number(point.get(name)) for name in WEATHER_VALUE_FIELDS]
                intervals[index] = _number(point.get("precipitation"))
                for direction_index, field in enumerate(WEATHER_DIRECTION_FIELDS):
                    direction = _number(point.get(field))
                    if math.isfinite(direction):
                        radians = math.radians(direction % 360)
                        matrix[index, len(WEATHER_VALUE_FIELDS)+2*direction_index:len(WEATHER_VALUE_FIELDS)+2*direction_index+2] = (
                            math.sin(radians), math.cos(radians))
            for array in (epochs, matrix, intervals):
                array.setflags(write=False)
            provenance = row.get("_provenance") or row.get("provenance") or {}
            serialized = row.get("payload",row.get("payload_blob"))
            if isinstance(serialized,bytes):
                serialized=gzip.decompress(serialized) if serialized.startswith(b"\x1f\x8b") else serialized
                serialized=serialized.decode("utf-8")
            if not isinstance(serialized,str):
                serialized=json.dumps(payload,sort_keys=True,separators=(",", ":"),allow_nan=False)
            checksum=hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            if provenance.get("payloadSha256") is not None and provenance["payloadSha256"]!=checksum:
                rejected+=1
                continue
            weather.append(WeatherRun(fetched, available, int(provenance.get("receiptId") or row.get("receipt_id") or 0),
                                      provenance.get("revisionId") or row.get("revision_id"), checksum,
                                      epochs, matrix, intervals))
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError, json.JSONDecodeError, OSError):
            rejected += 1
    weather.sort(key=lambda row: (row.fetched, row.available, row.receipt_id))
    fetched = np.asarray([row.fetched for row in weather], dtype=float)
    fetched.setflags(write=False)
    metadata = deepcopy(archive_metadata or {})
    metadata.update(weatherConfirmedPreparedRows=len(weather), weatherUnknownOrInvalidRowsExcluded=rejected,
                    neighborPolicy=NEIGHBOR_POLICY)
    neighbors=[]
    for original in neighbor_rows:
        row=deepcopy(dict(original))
        observed=_number(row.get("end_epoch",_number(row.get("bucket_start_epoch"))+_number(row.get("bucket_seconds",0))))
        if math.isfinite(observed):
            row["end_epoch"]=observed
            neighbors.append(row)
    neighbors.sort(key=lambda row:row["end_epoch"])
    neighbor_ends=np.asarray([row["end_epoch"] for row in neighbors],dtype=float)
    neighbor_ends.setflags(write=False)
    return PreparedInputs(detached, sensor.prepare_rows(detached), tuple(weather), fetched,
                          tuple(neighbors),neighbor_ends,
                          tuple(deepcopy(dict(row)) for row in neighbor_metadata), metadata)


def source_sensor_rows(prepared):
    """Detached original observations for independently constructed responses."""
    return [deepcopy(row) for row in prepare_inputs(prepared).sensorRows]


def _weather_asof(prepared, issue):
    position = int(np.searchsorted(prepared.weather_fetched, issue, side="right")) - 1
    while position >= 0:
        run = prepared.weather[position]
        if issue - run.fetched > MAX_WEATHER_AGE_SECONDS:
            return None
        if run.available <= issue:
            return run
        position -= 1
    return None


def _weather_values(run, valid_epoch):
    """Bounded interpolation; precipitation keeps its source hourly amount."""
    result = np.full(len(WEATHER_POINT_FIELDS), np.nan)
    if run is None or valid_epoch < run.epochs[0] or valid_epoch > run.epochs[-1]:
        return result
    index = int(np.searchsorted(run.epochs, valid_epoch, side="left"))
    if index < len(run.epochs) and run.epochs[index] == valid_epoch:
        return run.values[index].copy()
    if index == 0 or index >= len(run.epochs):
        return result
    lower, upper = index-1, index
    if run.epochs[upper]-run.epochs[lower] > 3600:
        return result  # Missing hourly record is not filled across a gap.
    fraction = (valid_epoch-run.epochs[lower])/(run.epochs[upper]-run.epochs[lower])
    pair = run.values[[lower, upper]]
    complete = np.isfinite(pair).all(axis=0)
    result[complete] = pair[0, complete]*(1-fraction)+pair[1, complete]*fraction
    # Hourly precipitation is an interval amount, not an interpolated rate.
    # Use the amount at the right-labelled source hour containing valid_epoch.
    result[WEATHER_VALUE_FIELDS.index("precipitation")] = run.intervals[upper]
    return result


def _weather_features(prepared, issue):
    result = dict.fromkeys(WEATHER_FEATURE_COLUMNS, math.nan)
    run = _weather_asof(prepared, issue)
    if run is None:
        return result, None
    for lead in WEATHER_LEADS_MINUTES:
        values = _weather_values(run, issue+lead*60)
        result.update(zip((f"issued_weather_{name}_{lead}min" for name in WEATHER_POINT_FIELDS), values.tolist()))
    result["issued_weather_fetch_age_minutes"] = (issue-run.fetched)/60
    arrival = _weather_values(run, issue+5400)
    for age in WEATHER_REVISION_MINUTES:
        previous = _weather_asof(prepared, issue-age*60)
        previous_values = _weather_values(previous, issue+5400)
        for field in WEATHER_REVISION_FIELDS:
            index = WEATHER_POINT_FIELDS.index(field)
            result[f"issued_weather_arrival_revision_{field}_{age}min"] = float(arrival[index]-previous_values[index])
    return result, run


def _neighbor_diagnostics(prepared, issue):
    """Request-clock sensitivity only; deliberately outside FEATURE_COLUMNS."""
    selected = {}
    lower=int(np.searchsorted(prepared.neighbor_ends,issue-NEIGHBOR_MAX_AGE_SECONDS,side="left"))
    upper=int(np.searchsorted(prepared.neighbor_ends,issue,side="right"))
    for row in prepared.neighbor_rows[lower:upper]:
        try:
            location = int(row.get("location_id"))
        except (TypeError, ValueError, OverflowError):
            continue
        if location not in NEIGHBOR_IDS.values() or row.get("valid", 1) != 1 or row.get("measure", "pm25") != "pm25":
            continue
        observed = _number(row.get("end_epoch", _number(row.get("bucket_start_epoch"))+_number(row.get("bucket_seconds", 0))))
        received, recorded = _number(row.get("received_epoch")), _number(row.get("recorded_epoch"))
        value = _number(row.get("pm25"))
        if (not all(math.isfinite(v) for v in (observed, received, recorded, value)) or value < 0
                or observed > issue or max(received, recorded) > issue or issue-observed > NEIGHBOR_MAX_AGE_SECONDS):
            continue
        key = (observed, max(received, recorded), int(row.get("request_id") or 0))
        if location not in selected or key > selected[location][0]:
            selected[location] = (key, value)
    return {"policy":NEIGHBOR_POLICY,"usedAsLearnerFeatures":False,
            "observationCommitConfirmationAvailable":False,
            "requestClockEligibleStationCount":len(selected),
            "requestClockEligibleStations":{name:{"pm25":selected[location][1],
                "observedEpoch":selected[location][0][0],"requestRecordedOrReceivedEpoch":selected[location][0][1]}
                for name,location in NEIGHBOR_STATIONS if location in selected}}


def _metadata(prepared, issue, sensor_metadata, weather_features, run):
    weather_count = sum(math.isfinite(value) for value in weather_features.values())
    lineage = {"sensorInputValuesSha256":sensor_metadata["inputValuesSha256"],
               "weatherPayloadSha256":run.payload_sha256 if run else None,
               "weatherReceiptId":run.receipt_id if run else None,
               "weatherAvailableEpoch":run.available if run else None,
               "forecastIssuedEpoch":issue}
    return {**sensor_metadata,"sensorFeatureVersion":sensor_metadata["featureVersion"],
            "featureVersion":VERSION,"featureCount":len(FEATURE_COLUMNS),
            "weatherFeaturesAvailable":weather_count>0,"weatherNonMissingFeatureCount":weather_count,
            "weatherAvailabilityPolicy":"all_received_persisted_confirmed_available_clocks_at_or_before_issue",
            "weatherForecastValidTimesMayFollowIssue":True,
            "weatherMaxAgeSeconds":MAX_WEATHER_AGE_SECONDS,
            "weatherFetchedEpoch":run.fetched if run else None,
            "weatherAvailableEpoch":run.available if run else None,
            "weatherReceiptId":run.receipt_id if run else None,
            "weatherRevisionId":run.revision_id if run else None,
            "weatherPayloadSha256":run.payload_sha256 if run else None,
            "weatherMissingReason":None if run else "no_confirmed_weather_vintage_visible_at_issue",
            "weatherTrainingSupportNotEstablishedByQueryPresence":True,
            "neighborDiagnostics":_neighbor_diagnostics(prepared,issue),
            "combinedInputLineageSha256":_digest(lineage),"archiveMetadata":deepcopy(prepared.archive_metadata)}


def issue_features(prepared, issue_epoch):
    prepared, issue = prepare_inputs(prepared), _issue(issue_epoch)
    result = sensor.issue_features(prepared.sensor, issue)
    weather, _ = _weather_features(prepared, issue)
    result.update(weather)
    return {name:float(result[name]) for name in FEATURE_COLUMNS}


def feature_metadata(prepared, issue_epoch):
    prepared, issue = prepare_inputs(prepared), _issue(issue_epoch)
    weather, run = _weather_features(prepared, issue)
    return _metadata(prepared, issue, sensor.feature_metadata(prepared.sensor, issue), weather, run)


def issue_feature_frame(prepared, issue_epochs):
    prepared = prepare_inputs(prepared)
    issues = [_issue(value) for value in issue_epochs]
    if len(issues) != len(set(issues)):
        raise ValueError("Issue epochs must be unique")
    sensor_frame = sensor.issue_feature_frame(prepared.sensor, issues)
    return augment_sensor_frame(prepared,sensor_frame)


def augment_sensor_frame(prepared,sensor_frame):
    """Reuse a fresh_sensor_features frame without recomputing sequence bins.

    The caller must pass the same retained sensor snapshot to prepare_inputs.
    This checks each cached sensor lineage hash before appending external data.
    """
    prepared=prepare_inputs(prepared)
    if tuple(sensor_frame.columns)!=SENSOR_FEATURE_COLUMNS:
        raise ValueError("Cached sensor feature columns differ")
    issues=[_issue(value) for value in sensor_frame.index]
    sensor_provenance=sensor_frame.attrs.get("featureProvenance",[])
    if len(sensor_provenance)!=len(issues):
        raise ValueError("Cached sensor provenance is missing")
    weather_records, provenance = [], []
    for issue, sensor_metadata in zip(issues,sensor_provenance):
        expected=sensor.feature_metadata(prepared.sensor,issue)
        if sensor_metadata.get("inputValuesSha256")!=expected["inputValuesSha256"]:
            raise ValueError("Cached sensor frame comes from a different snapshot")
        weather, run = _weather_features(prepared,issue)
        weather_records.append(weather)
        provenance.append(_metadata(prepared,issue,sensor_metadata,weather,run))
    weather_frame = pd.DataFrame(weather_records,index=sensor_frame.index,columns=WEATHER_FEATURE_COLUMNS,dtype=float)
    frame = pd.concat((sensor_frame,weather_frame),axis=1).loc[:,FEATURE_COLUMNS]
    frame.attrs.update(featureVersion=VERSION,featureProvenance=provenance,imputationApplied=False,
                       featureIdentity=feature_identity())
    return frame


def feature_identity():
    return {"version":VERSION,"featureColumns":list(FEATURE_COLUMNS),"sensorVersion":sensor.VERSION,
            "sensorSourceSha256":hashlib.sha256(Path(sensor.__file__).read_bytes()).hexdigest(),
            "sourceSha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "weatherLeadsMinutes":list(WEATHER_LEADS_MINUTES),"weatherMaxAgeSeconds":MAX_WEATHER_AGE_SECONDS,
            "weatherPointInterpolation":"bounded complete adjacent hourly records; directions as sin/cos",
            "weatherPrecipitation":"source hourly interval amount from right-labelled containing hour",
            "weatherRevisionComparison":"same absolute +90 arrival time using vintage visible 30/60 minutes earlier",
            "weatherReceiptPolicy":"received,persisted,confirmed,available<=issue; no legacy fetch-only rows",
            "neighborPolicy":NEIGHBOR_POLICY,"imputationApplied":False}


def _table_present(connection, table):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone() is not None


def load_inputs(db_path,start_epoch,end_epoch,neighbor_db_path=None,*,sensor_rows=None,
                sensor_receipt_watermark=None,weather_receipt_watermark=None):
    """Load all bounded visible revisions for reuse across issue-time queries.

    Legacy sensor measurements retain their explicit unknown-receipt caveat.
    Legacy weather runs are counted but never supplied as numerical features.
    Optional original issue watermarks bound the retained archive further.
    """
    start,end=_issue(start_epoch),_issue(end_epoch)
    if end<start:
        raise ValueError("Archive end must not precede start")
    path=Path(db_path).resolve()
    metadata={"archiveReadOnly":True,"weatherLegacyFetchClockUsed":False,
              "sensorReceiptWatermark":sensor_receipt_watermark,"weatherReceiptWatermark":weather_receipt_watermark}
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro",uri=True,timeout=30)) as connection:
        connection.row_factory=sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        weather=receipts.compatible_source_runs(connection,"weather_forecast_runs",end,
            max(1,start-MAX_WEATHER_AGE_SECONDS-3600),strict=True,include_revisions=True)
        if weather_receipt_watermark is not None:
            weather=[row for row in weather if row["_provenance"]["receiptId"]<=int(weather_receipt_watermark)]
        if _table_present(connection,"weather_forecast_runs"):
            metadata["legacyWeatherRunsInRange"]=int(connection.execute(
                "SELECT count(*) FROM weather_forecast_runs WHERE fetched_epoch>=? AND fetched_epoch<=?",
                (max(1,start-MAX_WEATHER_AGE_SECONDS-3600),end)).fetchone()[0])
        if sensor_rows is None:
            rows=[]
            lower=max(1,start-sensor.LOOKBACK_MINUTES*60)
            tracked=_table_present(connection,"sensor_observation_receipts") and _table_present(connection,"sensor_observation_visibility")
            if tracked:
                watermark=9223372036854775807 if sensor_receipt_watermark is None else int(sensor_receipt_watermark)
                for receipt in connection.execute(
                    "SELECT r.*,v.payload_json,s.confirmed_epoch FROM sensor_observation_receipts r "
                    "JOIN sensor_observation_revisions v ON v.revision_id=r.revision_id "
                    "JOIN sensor_observation_visibility s ON s.receipt_id=r.receipt_id "
                    "WHERE r.observed_epoch>=? AND r.observed_epoch<=? AND r.available_epoch<=? "
                    "AND s.confirmed_epoch<=? AND r.receipt_id<=? ORDER BY r.observed_epoch,r.available_epoch",
                    (lower,end,end,end,watermark)):
                    source=json.loads(receipt["payload_json"])
                    source["_provenance"]={"receiptKnown":True,"availableEpoch":receipt["available_epoch"],
                        "receivedEpoch":receipt["received_epoch"],"persistedEpoch":receipt["persisted_epoch"],
                        "confirmedEpoch":receipt["confirmed_epoch"],"receiptId":receipt["receipt_id"]}
                    rows.append(source)
            query="SELECT r.epoch,r.pm02,r.atmp,r.rhum FROM readings r WHERE r.epoch>=? AND r.epoch<=?"
            if tracked:
                query+=" AND NOT EXISTS(SELECT 1 FROM sensor_observation_receipts t WHERE t.observed_epoch=r.epoch)"
            for row in connection.execute(query+" ORDER BY r.epoch",(lower,end)):
                source=dict(row)
                source["_provenance"]={"receiptKnown":False,"availableEpoch":None,"receiptPolicy":"legacy_receipt_unknown"}
                rows.append(source)
        else:
            rows=[dict(row) for row in sensor_rows]
    neighbors=[]
    neighbor_metadata=[]
    if neighbor_db_path is not None and Path(neighbor_db_path).exists():
        with closing(sqlite3.connect(Path(neighbor_db_path).resolve().as_uri()+"?mode=ro",uri=True,timeout=30)) as connection:
            connection.row_factory=sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            neighbors=[dict(row) for row in connection.execute(
                "SELECT o.*,r.received_epoch,r.recorded_epoch FROM observations o "
                "JOIN requests r ON r.request_id=o.request_id WHERE o.bucket_start_epoch+o.bucket_seconds>=? "
                "AND o.bucket_start_epoch+o.bucket_seconds<=? AND r.received_epoch<=? AND r.recorded_epoch<=?",
                (max(1,start-NEIGHBOR_MAX_AGE_SECONDS),end,end,end))]
    return prepare_inputs(rows,weather,neighbors,neighbor_metadata,archive_metadata=metadata)
