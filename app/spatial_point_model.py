"""Live spatial (neighbour) HGB point model and its ride-mean <= 70 probability (v25).

Research and decision: research/models/RESULTS.md (2026-10-14). The model code is vendored
byte for byte, so every research check applies unchanged:

* spatial_final_models.py == research/models/final/models.py
* spatial_live_features.py == research/models/final/spatial_live.py

This adapter only reads the inputs with their receipt clocks, builds the research feature
rows, refits once per Malaysia day and publishes the literal outputs.

Heads
* +90 arrival and the +90..210 ride mean/min/max: P1_no_weather, four HistGradientBoosting
  absolute-error regressors (models.fit_point / predict_point). Inputs: 28 TTDI sensor columns,
  4 TTDI mean-reversion controls, 5 time columns and 44 neighbour columns. No weather or CAMS
  model inputs.
* P(ride mean <= 70): E1_spatial_residual (models.fit_le70 / predict_le70), the point model's
  ride mean plus the empirical out-of-fold residual distribution of the same training window.

Inputs at issue t, each selected by its own visibility clock
* TTDI readings: receipt clock max(available, confirmed). Untracked legacy readings (before the
  receipt ledger) are visible at observation + 60 s, the research harness rule.
* Neighbour polls from the archive the service already writes every 300 s
  (research/external_airgradient/neighbors.sqlite3): receipt clock MAX(received, recorded).
* The newest visible Open-Meteo Best Match run (age <= 2 h): 10 m and 925 hPa wind at
  +0/30/60/90 min. It only weights the neighbours by upwind alignment.

Training (nightly, like the other models): cutoff = Malaysia midnight of the issue day, 28-day
lookback, 330-minute embargo, labels complete before the cutoff, 15-minute origins. Feature
rows of past origins never change once their inputs are in, so they are cached incrementally
in private/ and only new origins are computed at each refit.

Literal outputs: no gates, selection, clipping or persistence substitution. An unavailable
input (no fresh TTDI reference, no model for the issue day, an unreadable neighbour archive,
no neighbour station with a bucket within 90 minutes, no issued wind within 2 hours) makes
every head unavailable with a reason. A single offline station is a NaN input, as in training.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import sqlite3
import threading
import time
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import fresh_numeric_model_v4 as numeric_targets
import fresh_sensor_features as sensor
from forecast_clock import point_weights, window_weights
from forecast_payload import loads as load_payload
import spatial_final_models as M
import spatial_live_features as SL

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private"
ARCHIVE_PATH = ROOT / "research" / "external_airgradient" / "neighbors.sqlite3"
MODEL_VERSION = "spatial_hgb_point_p1_no_weather_v25_20261014"
LE70_MODEL_VERSION = "spatial_residual_le70_e1_v25_20261014"
ADAPTER_VERSION = "spatial_point_live_adapter_v25"
FEATURE_VERSION = "spatial_point_issue_features_v25"
ARTIFACT_NAME = "spatial-point-model-v25.pkl"
STATUS_NAME = "spatial-point-model-v25-status.json"
FEATURE_CACHE_NAME = "spatial-point-features-v25.pkl"
LE70_ISSUE_NAME = "spatial-le70-issued.jsonl"
METHOD = "Spatial neighbour gradient-boosted model"
# Byte-identical copies of the research modules (checked by test_spatial_point_model).
VENDORED_SOURCES = {
    "spatial_final_models.py": ("research/models/final/models.py",
                                "7f5f9476664970183cd0cd9dbbe7112b5ca54ec6f9d4aad3a677eb60b1c19b4b"),
    "spatial_live_features.py": ("research/models/final/spatial_live.py",
                                 "25426c747a38c5b0a19abef91bda42ed6f802759b0734ee212d9328278fbb726"),
}

LEGACY_SENSOR_VISIBILITY_SECONDS = 60      # untracked legacy readings: observation + 60 s (harness)
LEGACY_SOURCE_VISIBILITY_SECONDS = 5       # untracked legacy weather runs: fetch + 5 s (harness)
# The provider history backfill was received 2026-10-05 19:30 MYT; its buckets carry the
# research's assumed clock (bucket end + 300 s). Any later history request uses its real receipt.
HISTORY_ASSUMED_VISIBLE_RECEIVED_BY = 1791216000   # 2026-10-06 00:00 MYT
WEATHER_SOURCE = "Open-Meteo Best Match"
WEATHER_MAX_AGE_SECONDS = 2 * 3600
WEATHER_LOOKBACK_SECONDS = 6 * 3600
WIND_LEADS_MINUTES = (0, 30, 60, 90)
SPATIAL_WINDOW_SECONDS = 3 * 3600
SENSOR_WINDOW_SECONDS = 4 * 3600
MINIMUM_TRAINING_ORIGINS = 240
TRAINING_LOOKBACK_DAYS = M.LOOKBACK_DAYS        # 28 days; models.training_rows applies the same bound
ARRIVAL_SECONDS = 5400
RIDE_END_SECONDS = 12600
CUTOFF_UG_M3 = float(M.PLANNING_CUTOFF)
FIT_CPU_THREADS = 1
MYT = timezone(timedelta(hours=8))

INPUT_COLUMNS = tuple(M.POINT_VARIANTS[M.FINAL_POINT_VARIANT])
SEQUENCE_COLUMNS = tuple("s_" + c for c in sensor.SEQUENCE_FEATURE_COLUMNS if c.startswith("seq_pm25_"))
WIND_COLUMNS = tuple(f"w_{level}_{part}_{lead}" for level in ("wind10", "wind925")
                     for part in ("u", "v") for lead in WIND_LEADS_MINUTES)
SPATIAL_COLUMNS = tuple(M.SP_STATION + M.SP_NET + M.SP_WIND)
CACHE_COLUMNS = tuple(dict.fromkeys([*M.SENSOR28, *SEQUENCE_COLUMNS, *M.TIME, *WIND_COLUMNS,
                                     *SPATIAL_COLUMNS, "m_fresh_available"]))
TARGET_COLUMNS = ("y_arrival", "y_ride_mean", "y_ride_min", "y_ride_max", "y_arrival_complete", "y_ride_complete")
IDENTITY_FILES = ("spatial_point_model.py", "spatial_final_models.py", "spatial_live_features.py",
                  "fresh_sensor_features.py", "fresh_numeric_model_v4.py", "forecast_clock.py",
                  "collection_quality.py", "forecast_payload.py")
FEATURE_IDENTITY_FILES = ("spatial_point_model.py", "spatial_final_models.py", "spatial_live_features.py",
                          "fresh_sensor_features.py", "forecast_payload.py")

POINT_PERFORMANCE_EVIDENCE = {
    "state": "development_only", "prospectivelyValidated": False, "modelVersion": MODEL_VERSION,
    "evidenceSource": "research/models/RESULTS.md (2026-10-14)",
    "holdoutDates": ["2026-10-06", "2026-10-13"],
    "selection": "variant chosen on the 2026-09-10..10-05 tune split; the variant list was written after "
                 "reading candidate results that included holdout days",
    "arrivalCaseCount": 2248, "arrivalMaeUgM3": 9.55, "arrivalPersistenceMaeUgM3": 10.64,
    "arrivalPreviousRidgeMaeUgM3": 11.08,
    "rideCaseCount": 2199, "trailMeanMaeUgM3": 12.88, "trailMeanPersistenceMaeUgM3": 14.09,
    "trailMeanPreviousRidgeMaeUgM3": 14.06,
    "issuedClockArrivalCaseCount": 1181, "issuedClockArrivalMaeUgM3": 8.64,
    "issuedClockArrivalIssuedRidgeMaeUgM3": 10.15, "issuedClockArrivalPersistenceMaeUgM3": 9.65,
    "overlappingIssues": True,
    "summary": ("8-day holdout (Oct 6-13): +90-minute average error 9.55 µg/m³ against 10.64 for the "
                "sensor reference and 11.08 for the previous Ridge; ride-mean error 12.88 against 14.09 and 14.06. "
                "Uses neighbouring AirGradient stations. Development evidence only; it seldom calls ≥20 changes, "
                "so advance warning still comes from the arrival-change model."),
}
LE70_PERFORMANCE_EVIDENCE = {
    "state": "development_only", "prospectivelyValidated": False, "modelVersion": LE70_MODEL_VERSION,
    "evidenceSource": "research/models/RESULTS.md (2026-10-14)",
    "holdoutDates": ["2026-10-06", "2026-10-13"], "holdoutCaseCount": 2194,
    "brierScore": 0.0540, "persistenceBrierScore": 0.0825, "previousCyclingLogisticBrierScore": 0.0953,
    "issuedClockCaseCount": 1612, "issuedClockBrierScore": 0.0667,
    "issuedClockPreviousCyclingLogisticBrierScore": 0.1037,
    "selectionUsedHoldout": True,
    "summary": ("Holdout Oct 6-13: Brier 0.054 against 0.095 for the previous cycling logistic and 0.083 for a "
                "current-level call. The choice of this variant over a simpler one used the holdout, so it is "
                "not independent confirmation; it puts less probability than the logistic on windows that open."),
}

_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_LOADED = {}
_IDENTITY = {}
_RECORDED = OrderedDict()
_ISSUE_PATHS_READ = set()


# ----------------------------------------------------------------------------- small helpers
def midnight_epoch(issue_epoch):
    local = datetime.fromtimestamp(int(issue_epoch), MYT)
    return int(local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def forecast_clock(issue_epoch):
    """The production +90 / +90..210 target clock (same targets as the replaced Ridge v4)."""
    return numeric_targets.forecast_clock(issue_epoch)


def _finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha_text(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def _signature(names):
    return tuple((name, (ROOT / name).stat().st_mtime_ns, (ROOT / name).stat().st_size) for name in names)


def model_identity():
    """Identity of the recipe; any source change refits once at the next start."""
    signature = _signature(IDENTITY_FILES)
    with _LOCK:
        if _IDENTITY.get("model", (None,))[0] == signature:
            return _IDENTITY["model"][1]
    identity = {"modelVersion": MODEL_VERSION, "le70ModelVersion": LE70_MODEL_VERSION,
                "adapterVersion": ADAPTER_VERSION, "featureVersion": FEATURE_VERSION,
                "pointVariant": M.FINAL_POINT_VARIANT, "le70Variant": M.FINAL_LE70_VARIANT,
                "inputColumnCount": len(INPUT_COLUMNS), "inputColumnsSha256": _sha_text(list(INPUT_COLUMNS)),
                "hgbParameters": dict(M.HGB_PARAMS),
                "trainingLookbackDays": M.LOOKBACK_DAYS, "embargoMinutes": M.EMBARGO_MINUTES,
                "trainingOriginSeconds": M.ORIGIN_SECONDS, "minimumTrainingOrigins": MINIMUM_TRAINING_ORIGINS,
                "legacySensorVisibilitySeconds": LEGACY_SENSOR_VISIBILITY_SECONDS,
                "legacySourceVisibilitySeconds": LEGACY_SOURCE_VISIBILITY_SECONDS,
                "historyAssumedVisibleReceivedBy": HISTORY_ASSUMED_VISIBLE_RECEIVED_BY,
                "weatherSource": WEATHER_SOURCE, "weatherMaxAgeSeconds": WEATHER_MAX_AGE_SECONDS,
                "trainingCutoff": "issue_day_midnight_Asia_Kuala_Lumpur",
                "targetDefinition": "fresh_numeric_model_v4.targets_for_issues (production +90 and +90..210 targets)",
                "numericalSelectionApplied": False, "persistenceFallback": False,
                "sourceSha256": {name: _file_sha(name) for name in IDENTITY_FILES}}
    identity["identitySha256"] = _sha_text(identity)
    with _LOCK:
        _IDENTITY["model"] = (signature, identity)
    return identity


def feature_identity():
    """Cached training rows stay valid while the feature code is unchanged."""
    signature = _signature(FEATURE_IDENTITY_FILES)
    with _LOCK:
        if _IDENTITY.get("feature", (None,))[0] == signature:
            return _IDENTITY["feature"][1]
    identity = _sha_text({"featureVersion": FEATURE_VERSION, "columns": list(CACHE_COLUMNS),
                          "legacySensorVisibilitySeconds": LEGACY_SENSOR_VISIBILITY_SECONDS,
                          "legacySourceVisibilitySeconds": LEGACY_SOURCE_VISIBILITY_SECONDS,
                          "historyAssumedVisibleReceivedBy": HISTORY_ASSUMED_VISIBLE_RECEIVED_BY,
                          "weatherSource": WEATHER_SOURCE, "weatherMaxAgeSeconds": WEATHER_MAX_AGE_SECONDS,
                          "sourceSha256": {name: _file_sha(name) for name in FEATURE_IDENTITY_FILES}})
    with _LOCK:
        _IDENTITY["feature"] = (signature, identity)
    return identity


def _connect_ro(path, timeout=30):
    """Read-only connection; never immutable=1 (the live DB is in WAL mode)."""
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _atomic_write(path, data):
    """Write <name>.pending, then replace: a reader sees the old or the new file, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(data)
    for attempt in range(5):
        try:
            pending.replace(path)
            return path
        except PermissionError:
            # Windows refuses the swap while a reader holds the old file open for a moment.
            if attempt == 4:
                raise
            time.sleep(0.2)
    return path


# ----------------------------------------------------------------------------- input readers
SENSOR_COLUMNS = ("epoch", "pm02", "atmp", "rhum", "visible_epoch")


def _sensor_frame(records):
    frame = pd.DataFrame(records, columns=list(SENSOR_COLUMNS))
    frame["epoch"] = frame["epoch"].astype(np.int64)
    for column in ("pm02", "atmp", "rhum", "visible_epoch"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return frame.sort_values("epoch", kind="mergesort").reset_index(drop=True)


def read_sensor_table(conn, start_epoch, end_epoch):
    """TTDI readings with their visibility clock (training view; the caller owns the transaction).

    One row per observation: the readings value, visible at the first confirmed receipt
    max(available, confirmed). A tracked observation without a confirmed receipt is never
    visible; an untracked legacy reading is visible at observation + 60 s.
    """
    start, end = int(start_epoch), int(end_epoch)
    readings = conn.execute("SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch",
                            (start, end)).fetchall()
    confirmed, tracked = {}, set()
    if {"sensor_observation_receipts", "sensor_observation_visibility"} <= _tables(conn):
        confirmed = {int(e): float(a) for e, a in conn.execute(
            """SELECT r.observed_epoch,MIN(MAX(r.available_epoch,v.confirmed_epoch))
               FROM sensor_observation_receipts r JOIN sensor_observation_visibility v ON v.receipt_id=r.receipt_id
               WHERE r.observed_epoch>=? AND r.observed_epoch<=? GROUP BY r.observed_epoch""", (start, end))}
        tracked = {int(r[0]) for r in conn.execute(
            "SELECT DISTINCT observed_epoch FROM sensor_observation_receipts WHERE observed_epoch>=? AND observed_epoch<=?",
            (start, end))}
    records = []
    for epoch, pm02, atmp, rhum in readings:
        epoch = int(epoch)
        visible = confirmed.get(epoch, math.inf if epoch in tracked else epoch + LEGACY_SENSOR_VISIBILITY_SECONDS)
        records.append((epoch, pm02, atmp, rhum, visible))
    return _sensor_frame(records)


def live_sensor_table(rows, issue_epoch, lookback_seconds=SENSOR_WINDOW_SECONDS):
    """The issue's materialised sensor rows (observation_provenance.compatible_sensor_rows).

    Receipt-tracked rows carry availableEpoch = max(available, confirmed); untracked legacy rows
    use observation + 60 s. Only rows visible by the issue are kept, one per observation.
    """
    issue = int(issue_epoch)
    best = {}
    for original in rows:
        row = original if isinstance(original, dict) else dict(original)
        epoch = _finite(row.get("epoch"))
        if epoch is None or epoch > issue or epoch < issue - lookback_seconds:
            continue
        provenance = row.get("_provenance") or {}
        if provenance.get("receiptKnown") is True:
            available = _finite(provenance.get("availableEpoch"))
            visible = available if available is not None else math.inf
        else:
            visible = epoch + LEGACY_SENSOR_VISIBILITY_SECONDS
        if visible > issue:
            continue
        key = int(epoch)
        if key not in best or visible >= best[key][4]:
            best[key] = (key, row.get("pm02"), row.get("atmp"), row.get("rhum"), visible)
    return _sensor_frame(list(best.values()))


def _sensor_records(table):
    """fresh_sensor_features rows whose availability is the visibility clock."""
    return [{"epoch": int(e), "pm02": p, "atmp": t, "rhum": h,
             "_provenance": {"receiptKnown": True, "availableEpoch": float(v)}}
            for e, p, t, h, v in zip(table.epoch.to_numpy(), table.pm02.to_numpy(), table.atmp.to_numpy(),
                                     table.rhum.to_numpy(), table.visible_epoch.to_numpy())]


def label_rows(conn, start_epoch, end_epoch):
    """Plain readings for the production completed targets (latest stored values)."""
    return [{"epoch": int(e), "pm02": p} for e, p in conn.execute(
        "SELECT epoch,pm02 FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch", (int(start_epoch), int(end_epoch)))]


class WindRun:
    """One issued Open-Meteo run as hourly toward-vector winds (harness weather_group semantics)."""

    def __init__(self, fetched, visible, payload):
        self.fetched, self.visible = int(fetched), float(visible)
        hourly = payload.get("hourly", []) if isinstance(payload, dict) else []
        points = {int(p["epoch"]): p for p in hourly if isinstance(p, dict) and p.get("epoch") is not None}
        self.epochs = np.array(sorted(points), dtype=float)

        def column(name):
            out = np.full(len(self.epochs), np.nan)
            for i, epoch in enumerate(self.epochs):
                value = points[int(epoch)].get(name)
                try:
                    out[i] = float(value) if value is not None else np.nan
                except (TypeError, ValueError):
                    pass
            return out

        self.values = {}
        for short, speed_name, direction_name in (("wind10", "wind_speed_10m", "wind_direction_10m"),
                                                  ("wind925", "wind_speed_925hPa", "wind_direction_925hPa")):
            speed = column(speed_name)
            angle = np.deg2rad(column(direction_name) % 360)
            self.values[short + "_u"] = -speed * np.sin(angle)
            self.values[short + "_v"] = -speed * np.cos(angle)

    def at(self, name, times):
        """Bounded linear interpolation between hourly points at most 1 h apart (harness _Run.at)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        out = np.full(len(times), np.nan)
        x, y = self.epochs, self.values[name]
        if len(x) < 2:
            return out
        hi = np.searchsorted(x, times, side="left")
        hi_c = np.minimum(hi, len(x) - 1)
        exact = (hi < len(x)) & (x[hi_c] == times)
        lo_c = np.maximum(hi - 1, 0)
        inside = (hi > 0) & (hi < len(x)) & ~exact & ((x[hi_c] - x[lo_c]) <= 3600)
        span = np.where(inside, x[hi_c] - x[lo_c], 1.0)
        frac = (times - x[lo_c]) / span
        value = y[lo_c] + (y[hi_c] - y[lo_c]) * frac
        out[inside] = value[inside]
        out[exact] = y[hi_c][exact]
        return out

    def features(self, issue_epoch):
        times = int(issue_epoch) + np.asarray(WIND_LEADS_MINUTES, dtype=float) * 60
        out = {}
        for level in ("wind10", "wind925"):
            for part in ("u", "v"):
                values = self.at(f"{level}_{part}", times)
                out.update({f"w_{level}_{part}_{lead}": float(v) for lead, v in zip(WIND_LEADS_MINUTES, values)})
        return out


def read_weather_runs(conn, start_fetched_epoch, end_fetched_epoch):
    """[(fetched, visible)] Best Match runs sorted by (visible, fetched); payloads load lazily."""
    if "weather_forecast_runs" not in _tables(conn):
        return []
    lo, hi = int(start_fetched_epoch), int(end_fetched_epoch)
    runs = conn.execute("SELECT fetched_epoch,source FROM weather_forecast_runs WHERE fetched_epoch>=? AND fetched_epoch<=?",
                        (lo, hi)).fetchall()
    confirmed, tracked = {}, set()
    if {"source_forecast_receipts", "source_forecast_visibility"} <= _tables(conn):
        confirmed = {int(f): float(a) for f, a in conn.execute(
            """SELECT r.source_fetched_epoch,MIN(MAX(r.available_epoch,v.confirmed_epoch))
               FROM source_forecast_receipts r JOIN source_forecast_visibility v ON v.receipt_id=r.receipt_id
               WHERE r.source_table='weather_forecast_runs' AND r.source_fetched_epoch>=? AND r.source_fetched_epoch<=?
               GROUP BY r.source_fetched_epoch""", (lo, hi))}
        tracked = {int(r[0]) for r in conn.execute(
            """SELECT DISTINCT source_fetched_epoch FROM source_forecast_receipts
               WHERE source_table='weather_forecast_runs' AND source_fetched_epoch>=? AND source_fetched_epoch<=?""", (lo, hi))}
    out = []
    for fetched, source in runs:
        if source != WEATHER_SOURCE:
            continue
        fetched = int(fetched)
        visible = confirmed.get(fetched, math.inf if fetched in tracked else fetched + LEGACY_SOURCE_VISIBILITY_SECONDS)
        out.append((fetched, visible))
    out.sort(key=lambda run: (run[1], run[0]))
    return out


def select_weather_run(runs, issue_epoch, visible=None):
    """Newest visible run by the issue; unavailable when it is older than 2 h (harness rule)."""
    visible = [clock for _, clock in runs] if visible is None else visible
    position = bisect_right(visible, int(issue_epoch)) - 1
    if position < 0:
        return None
    fetched, visible = runs[position]
    return (fetched, visible) if int(issue_epoch) - fetched <= WEATHER_MAX_AGE_SECONDS else None


def load_wind_run(conn, fetched, visible):
    row = conn.execute("SELECT payload FROM weather_forecast_runs WHERE fetched_epoch=?", (int(fetched),)).fetchone()
    if row is None:
        raise ValueError("selected_weather_run_missing")
    return WindRun(fetched, visible, load_payload(row[0]))


POLL_COLUMNS = ("location_id", "kind", "end_epoch", "pm25", "receipt_epoch")


def read_neighbour_polls(archive_path, end_after_epoch, end_until_epoch):
    """Valid neighbour PM2.5 with end in (after, until], one row per station and end clock.

    Same table as the research loader: the value of the first request that returned it and
    that request's receipt clock MAX(received, recorded). A history row received after the
    October 5 backfill is treated by its real receipt (as a 'current' observation).
    """
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError("neighbour_archive_missing")
    after, until = int(end_after_epoch), int(end_until_epoch)
    stations = tuple(int(location) for location in SL.STATIONS)
    with closing(_connect_ro(path, timeout=10)) as conn:
        rows = conn.execute(
            f"""SELECT o.location_id,o.kind,o.bucket_start_epoch+o.bucket_seconds,o.pm25,
                       MAX(r.received_epoch,r.recorded_epoch),o.request_id
                FROM observations o JOIN requests r ON r.request_id=o.request_id
                WHERE o.valid=1 AND o.measure='pm25' AND o.location_id IN ({",".join("?" * len(stations))})
                  AND o.bucket_start_epoch>? AND o.bucket_start_epoch<=?""",
            (*stations, after - 900, until)).fetchall()
    frame = pd.DataFrame(rows, columns=[*POLL_COLUMNS, "request_id"])
    frame = frame.loc[(frame.end_epoch > after) & (frame.end_epoch <= until)]
    if frame.empty:
        return pd.DataFrame({"location_id": pd.Series(dtype=np.int64), "kind": pd.Series(dtype=object),
                             "end_epoch": pd.Series(dtype=np.int64), "pm25": pd.Series(dtype=float),
                             "receipt_epoch": pd.Series(dtype=float)})
    frame = frame.sort_values(["receipt_epoch", "request_id"], kind="mergesort")
    frame = (frame.groupby(["location_id", "end_epoch"], as_index=False)
             .agg(pm25=("pm25", "first"), receipt_epoch=("receipt_epoch", "min"), kind=("kind", "first")))
    late = (frame.kind == "history") & (frame.receipt_epoch > HISTORY_ASSUMED_VISIBLE_RECEIVED_BY)
    frame.loc[late, "kind"] = "current"
    frame["location_id"] = frame.location_id.astype(np.int64)
    frame["end_epoch"] = frame.end_epoch.astype(np.int64)
    frame["pm25"] = frame.pm25.astype(float)
    frame["receipt_epoch"] = frame.receipt_epoch.astype(float)
    return (frame[list(POLL_COLUMNS)].sort_values(["end_epoch", "location_id"], kind="mergesort")
            .reset_index(drop=True))


# ----------------------------------------------------------------------------- feature rows
def _assemble(issue, sensor_values, sensor_columns, metadata, ttdi, polls, wind):
    """One research feature row: s_*, m_fresh_available, t_*, w_wind*, sp_* (P1's 44)."""
    row = {"s_" + name: float(value) for name, value in zip(sensor_columns, sensor_values)}
    row["m_fresh_available"] = bool(metadata.get("available"))
    row.update(M.time_features(int(issue)))
    row.update(wind if wind is not None else dict.fromkeys(WIND_COLUMNS, float("nan")))
    spatial = SL.spatial_features(int(issue), SL.NeighbourBuckets(polls), SL.TTDIBuckets(ttdi), row)
    row.update({name: float(spatial[name]) for name in SPATIAL_COLUMNS})
    return row


def feature_rows(issues, sensor_table, polls, wind_by_issue):
    """Feature rows for sorted issues from one sensor table, one poll table and their wind.

    Each issue sees only records visible by it: fresh_sensor_features selects readings by
    availability, NeighbourBuckets/TTDIBuckets by bucket visibility, and wind_by_issue holds
    the run selected for the issue. Slices span the 3-hour spatial window.
    """
    issues = [int(issue) for issue in issues]
    prepared = sensor.prepare_rows(_sensor_records(sensor_table))
    frame = sensor.issue_feature_frame(prepared, issues)
    provenance = frame.attrs.get("featureProvenance", [])
    values = frame.to_numpy(dtype=float)
    columns = list(frame.columns)
    t_epoch = sensor_table.epoch.to_numpy(np.int64)
    p_end = polls.end_epoch.to_numpy(np.int64)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for index, issue in enumerate(issues):
            lo = issue - SPATIAL_WINDOW_SECONDS
            ttdi = sensor_table.iloc[np.searchsorted(t_epoch, lo, "right"):np.searchsorted(t_epoch, issue, "right")]
            nearby = polls.iloc[np.searchsorted(p_end, lo, "right"):np.searchsorted(p_end, issue, "right")]
            rows.append(_assemble(issue, values[index], columns, provenance[index], ttdi, nearby,
                                  wind_by_issue.get(issue)))
    return rows, provenance


def _frame_from_rows(rows, issues):
    frame = pd.DataFrame(rows, index=pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch"),
                         columns=list(CACHE_COLUMNS))
    frame["m_fresh_available"] = frame["m_fresh_available"].astype(bool)
    for column in CACHE_COLUMNS:
        if column != "m_fresh_available":
            frame[column] = frame[column].astype(float)
    return frame


def training_origins(cutoff_epoch):
    """15-minute origins in (cutoff - 28 d, cutoff - 330 min), the research training rows."""
    cutoff = int(cutoff_epoch)
    lower = cutoff - TRAINING_LOOKBACK_DAYS * 86400
    upper = cutoff - M.EMBARGO_MINUTES * 60
    first = (lower // M.ORIGIN_SECONDS + 1) * M.ORIGIN_SECONDS
    return np.arange(first, upper, M.ORIGIN_SECONDS, dtype=np.int64)


def compute_training_features(db_path, origins, *, archive_path=None):
    """Full-view feature rows for past origins (every input of an origin <= cutoff - 330 min is final)."""
    origins = sorted(int(origin) for origin in origins)
    if not origins:
        return _frame_from_rows([], [])
    first, last = origins[0], origins[-1]
    # Archive first: an unreadable archive fails before the heavier DB reads (the worker retries).
    polls = read_neighbour_polls(archive_path or ARCHIVE_PATH, first - SPATIAL_WINDOW_SECONDS, last)
    with closing(_connect_ro(db_path)) as conn:
        conn.execute("BEGIN")
        table = read_sensor_table(conn, first - SENSOR_WINDOW_SECONDS, last)
        runs = read_weather_runs(conn, first - WEATHER_LOOKBACK_SECONDS, last)
        clocks = [clock for _, clock in runs]
        chosen = {origin: select_weather_run(runs, origin, clocks) for origin in origins}
        loaded = {}
        for run in set(filter(None, chosen.values())):
            loaded[run] = load_wind_run(conn, *run)
    wind ={origin: loaded[run].features(origin) for origin, run in chosen.items() if run is not None}
    rows, _ = feature_rows(origins, table, polls, wind)
    return _frame_from_rows(rows, origins)


def training_targets(rows, origins):
    """Production targets (fresh_numeric_model_v4) with the arrival target on its own buckets."""
    origins = np.asarray(origins, dtype=np.int64)
    series = numeric_targets.observed_series(rows)
    joint, complete = numeric_targets.targets_for_issues(series, origins)
    buckets = {int(t): float(v) for t, v in zip(series.index.asi8 // 10**9, series.to_numpy(float)) if np.isfinite(v)}
    arrival, arrival_complete = [], []
    for issue in origins:
        anchor = int(issue // 900 * 900)
        weights = point_weights(int(issue - anchor + ARRIVAL_SECONDS))
        labels = [anchor + offset * 900 for offset in weights]
        arrival_complete.append(max(labels))
        arrival.append(sum(buckets[label] * w for label, w in zip(labels, weights.values()))
                       if all(label in buckets for label in labels) else np.nan)
    frame = pd.DataFrame({"y_arrival": np.asarray(arrival, float),
                          "y_ride_mean": joint["trailMeanPoint"].to_numpy(float),
                          "y_ride_min": joint["trailMinimumPoint"].to_numpy(float),
                          "y_ride_max": joint["trailMaximumPoint"].to_numpy(float),
                          "y_arrival_complete": np.asarray(arrival_complete, np.int64),
                          "y_ride_complete": np.asarray(complete, np.int64)},
                         index=pd.Index(origins, name="issue_epoch"))
    return frame


# ----------------------------------------------------------------------------- feature cache
def _paths(model_dir=None):
    directory = Path(model_dir) if model_dir is not None else PRIVATE
    return directory / ARTIFACT_NAME, directory / STATUS_NAME, directory / FEATURE_CACHE_NAME


def _load_feature_cache(model_dir=None):
    _, _, path = _paths(model_dir)
    identity = feature_identity()
    try:
        value = pickle.loads(path.read_bytes())
        if (isinstance(value, dict) and value.get("identity") == identity
                and isinstance(value.get("frame"), pd.DataFrame) and list(value["frame"].columns) == list(CACHE_COLUMNS)):
            return value["frame"]
    except (OSError, EOFError, ValueError, TypeError, AttributeError, ImportError, pickle.UnpicklingError):
        pass
    return _frame_from_rows([], [])


def training_features(db_path, cutoff_epoch, *, archive_path=None, model_dir=None):
    """Feature rows for every training origin, computing only origins absent from the cache."""
    origins = training_origins(cutoff_epoch)
    cached = _load_feature_cache(model_dir)
    missing = [int(o) for o in origins if int(o) not in cached.index]
    started = time.perf_counter()
    if missing:
        fresh = compute_training_features(db_path, missing, archive_path=archive_path)
        keep = cached.loc[cached.index >= int(cutoff_epoch) - (TRAINING_LOOKBACK_DAYS + 1) * 86400]
        cached = pd.concat([keep, fresh]).sort_index() if len(keep) else fresh
        cached = cached.loc[~cached.index.duplicated(keep="last")]
        _, _, path = _paths(model_dir)
        _atomic_write(path, pickle.dumps({"identity": feature_identity(), "featureVersion": FEATURE_VERSION,
                                          "frame": cached}, protocol=pickle.HIGHEST_PROTOCOL))
    frame = cached.reindex(origins)
    frame["m_fresh_available"] = frame["m_fresh_available"].fillna(False).astype(bool)
    info = {"origins": int(len(origins)), "cachedOrigins": int(len(origins) - len(missing)),
            "computedOrigins": int(len(missing)), "computeSeconds": round(time.perf_counter() - started, 3)}
    return frame, info


def training_frame(db_path, cutoff_epoch, *, archive_path=None, model_dir=None):
    cutoff = int(cutoff_epoch)
    features, info = training_features(db_path, cutoff, archive_path=archive_path, model_dir=model_dir)
    with closing(_connect_ro(db_path)) as conn:
        conn.execute("BEGIN")
        rows = label_rows(conn, cutoff - TRAINING_LOOKBACK_DAYS * 86400 - SENSOR_WINDOW_SECONDS, cutoff)
    targets = training_targets(rows, features.index.to_numpy(np.int64))
    return features.join(targets), info


# ----------------------------------------------------------------------------- fitting
def fit(history, cutoff_epoch):
    """Fit the point heads and E1 at the cutoff; refuses thin training support."""
    cutoff = int(cutoff_epoch)
    base = M.training_rows(history, cutoff, origin_seconds=M.ORIGIN_SECONDS)
    counts = {head: int((base & np.isfinite(history[target].to_numpy(float))).sum())
              for head, target in M.POINT_HEADS.items()}
    if min(counts.values()) < MINIMUM_TRAINING_ORIGINS:
        return None, None, {"available": False, "reason": "insufficient_completed_training_origins",
                            "trainingOrigins": counts}
    with threadpool_limits(limits=FIT_CPU_THREADS), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        point = M.fit_point(history, cutoff)
        le70 = M.fit_le70(history, cutoff, point=point)
    if le70.z is None or not len(le70.z):
        return point, None, {"available": True, "reason": "available", "trainingOrigins": dict(point.n_train),
                             "le70Available": False, "le70Reason": "no_out_of_fold_residuals"}
    return point, le70, {"available": True, "reason": "available", "trainingOrigins": dict(point.n_train),
                         "le70Available": True, "le70ResidualCount": int(le70.n_train)}


def _training_window(history, cutoff):
    used = history.loc[M.training_rows(history, cutoff, origin_seconds=M.ORIGIN_SECONDS)]
    if used.empty:
        return {}
    return {"earliestTrainingOriginEpoch": int(used.index.min()), "latestTrainingOriginEpoch": int(used.index.max()),
            "latestTrainingOutcomeCompleteEpoch": int(max(used.y_arrival_complete.max(), used.y_ride_complete.max())),
            "trainingDateCount": int(len({datetime.fromtimestamp(int(i), MYT).date() for i in used.index})),
            "trainingDataHash": _frame_hash(used)}


def _frame_hash(frame):
    digest = hashlib.sha256(json.dumps(list(frame.columns)).encode())
    digest.update(np.asarray(frame.index, dtype="<i8").tobytes())
    values = frame.drop(columns=["m_fresh_available"], errors="ignore").to_numpy(dtype="<f8").copy()
    values[~np.isfinite(values)] = np.nan
    digest.update(values.tobytes())
    return digest.hexdigest()


def load_artifact(model_dir=None):
    path, _, _ = _paths(model_dir)
    try:
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        return None
    identity = model_identity()
    key = str(path.resolve())
    with _LOCK:
        cached = _LOADED.get(key)
        if cached and cached[0] == stamp and cached[1].get("identity") == identity:
            return cached[1]
    try:
        value = pickle.loads(path.read_bytes())
    except (OSError, EOFError, ValueError, TypeError, AttributeError, ImportError, pickle.UnpicklingError):
        return None
    if (not isinstance(value, dict) or value.get("identity") != identity or value.get("modelVersion") != MODEL_VERSION
            or value.get("adapterVersion") != ADAPTER_VERSION):
        return None
    with _LOCK:
        _LOADED[key] = (stamp, value)
    return value


def refresh_model(db_path, issue_epoch=None, *, archive_path=None, model_dir=None):
    """Background-worker entry: one fit per Malaysia day (and after any recipe change)."""
    issue = int(time.time()) if issue_epoch is None else int(issue_epoch)
    cutoff = midnight_epoch(issue)
    artifact_path, status_path, _ = _paths(model_dir)
    with _FIT_LOCK:
        prior = load_artifact(model_dir)
        if (prior is not None and prior["trainingCutoffEpoch"] == cutoff
                and prior["metadata"].get("reason") != "background_fit_unavailable"):
            return dict(prior["metadata"], cacheReloaded=True)
        started = time.perf_counter()
        identity = model_identity()
        metadata = {"modelVersion": MODEL_VERSION, "le70ModelVersion": LE70_MODEL_VERSION,
                    "adapterVersion": ADAPTER_VERSION, "trainingCutoffEpoch": cutoff,
                    "modelIdentity": identity, "fitCpuThreads": FIT_CPU_THREADS,
                    "pointVariant": M.FINAL_POINT_VARIANT, "le70Variant": M.FINAL_LE70_VARIANT,
                    "numericalSelectionApplied": False, "persistenceFallback": False,
                    "calibrated": False, "prospectivelyValidated": False}
        point = le70 = None
        try:
            history, cache = training_frame(db_path, cutoff, archive_path=archive_path, model_dir=model_dir)
            metadata["featureCache"] = cache
            metadata.update(_training_window(history, cutoff))
            point, le70, support = fit(history, cutoff)
            metadata.update(support)
        except Exception as error:
            metadata.update(available=False, reason="background_fit_unavailable",
                            errorType=type(error).__name__, error=str(error)[:300])
            point = le70 = None
        metadata["fittedAtEpoch"] = int(time.time())
        metadata["fitElapsedSeconds"] = round(time.perf_counter() - started, 3)
        value = {"modelVersion": MODEL_VERSION, "le70ModelVersion": LE70_MODEL_VERSION,
                 "adapterVersion": ADAPTER_VERSION, "identity": identity, "trainingCutoffEpoch": cutoff,
                 "metadata": metadata, "point": point, "le70": le70}
        _atomic_write(artifact_path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        _atomic_write(status_path, (json.dumps(_json_value(metadata), indent=2, allow_nan=False) + "\n").encode("utf-8"))
        with _LOCK:
            _LOADED[str(artifact_path.resolve())] = ((artifact_path.stat().st_mtime_ns, artifact_path.stat().st_size), value)
        return dict(metadata, cacheReloaded=False)


# ----------------------------------------------------------------------------- inference
def _le70_policy():
    return {"variant": M.FINAL_LE70_VARIANT, "anchor": "point_model_ride_mean_prediction",
            "residual": "log((y + 5) / (anchor + 5)) out of fold over three day blocks of the 28-day window",
            "planningCutoffUgM3": CUTOFF_UG_M3, "targetComparison": "mean_at_or_below",
            "target": "exact_two_hour_ride_window_mean", "initialLeadSeconds": ARRIVAL_SECONDS,
            "durationSeconds": RIDE_END_SECONDS - ARRIVAL_SECONDS,
            "trainingLookbackDays": M.LOOKBACK_DAYS, "trainingCutoff": "issue_day_midnight_Asia_Kuala_Lumpur",
            "calibrated": False, "cutoffMeaning": "user_selected_planning_cutoff"}


def _target_clock(issue):
    start, end = issue + ARRIVAL_SECONDS, issue + RIDE_END_SECONDS
    anchor = issue // 900 * 900
    weights = window_weights(start - anchor, end - anchor)
    return {"featureOriginEpoch": anchor, "windowStartEpoch": start, "windowEndEpoch": end,
            "targetMeanDefinition": "overlap_duration_weighted_complete_15_minute_sensor_medians",
            "bucketWeights": [{"bucketEndEpoch": int(anchor + offset * 900), "weight": float(weight)}
                              for offset, weight in weights.items()],
            "observedOutcomeCompleteAfterEpoch": int(anchor + max(weights) * 900)}


def _le70_base(issue, reason=None):
    start, end = issue + ARRIVAL_SECONDS, issue + RIDE_END_SECONDS
    return {"available": False, "modelVersion": LE70_MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
            "anchorModelVersion": MODEL_VERSION, "forecastIssuedEpoch": issue, "startEpoch": start,
            "endEpoch": end, "mode": "initial90", "referencePm": None, "cutoffUgM3": CUTOFF_UG_M3,
            "chanceMeanAtOrBelowCutoff": None, "experimental": True, "calibrated": False,
            "selectedPolicy": _le70_policy(), "target": "exact_two_hour_ride_window_mean",
            "cutoffMeaning": "user_selected_planning_cutoff", "planId": f"cycling70-{start}-{end}",
            "planIssuedEpoch": issue, "plannedRecheckEpoch": issue + 1800, "remainingLeadMinutes": 90.0,
            "targetClock": _target_clock(issue), "reason": reason,
            "unavailableReasons": [reason] if reason else [],
            "performanceEvidence": dict(LE70_PERFORMANCE_EVIDENCE)}


def _base(issue, allow_reconstruction=False):
    return {"available": False, "modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
            "forecastIssuedEpoch": issue, "forecastClock": forecast_clock(issue), "reason": None,
            "method": METHOD, "numericalSelectionApplied": False, "persistenceFallback": False,
            "rawOutputArithmeticApplied": False, "calibrated": False, "prospectivelyValidated": False,
            "estimatorOwnsInverseTargetTransform": True, "reconstructed": bool(allow_reconstruction),
            "forecastRole": "counterfactual_reconstruction" if allow_reconstruction else "live_cached_model_output",
            "arrivalPoint": None, "trailMeanPoint": None, "trailMinimumPoint": None, "trailMaximumPoint": None,
            "rawOutputUgM3": None, "freshReferencePm25": None, "inputLineage": {},
            "performanceEvidence": dict(POINT_PERFORMANCE_EVIDENCE)}


def _unavailable(result, reason, **extra):
    le70 = _le70_base(result["forecastIssuedEpoch"], reason)
    le70["referencePm"] = _finite(result.get("freshReferencePm25"))
    return {**result, **extra, "available": False, "reason": reason, "le70": le70}


def _artifact_problem(artifact, issue, allow_reconstruction):
    if artifact is None:
        return "spatial_point_model_asset_unavailable"
    metadata = artifact.get("metadata") or {}
    if artifact.get("trainingCutoffEpoch") != midnight_epoch(issue):
        return "spatial_point_model_not_fitted_for_issue_day"
    if not metadata.get("available") or artifact.get("point") is None:
        return metadata.get("reason") or "spatial_point_model_training_unavailable"
    fitted = _finite(metadata.get("fittedAtEpoch"))
    if fitted is None:
        return "model_fit_timestamp_unavailable"
    if fitted > issue and not allow_reconstruction:
        return "model_not_fitted_at_issue"
    if (metadata.get("latestTrainingOutcomeCompleteEpoch") or issue + 1) > issue:
        return "training_outcomes_after_issue"
    return None


def live_issue_features(db_path, rows, issue_epoch, *, archive_path=None):
    """Feature row for one live issue from inputs visible by it.

    Returns {"features", "lineage", "reason", "errorType"}; features is None with a reason when
    a required input is unavailable or stale.
    """
    issue = int(issue_epoch)
    table = live_sensor_table(rows, issue)
    query = sensor.issue_feature_frame(sensor.prepare_rows(_sensor_records(table)), [issue])
    lineage = dict(query.attrs["featureProvenance"][0])
    out = {"features": None, "lineage": lineage, "reason": None, "errorType": None}
    if not lineage.get("available"):
        return {**out, "reason": lineage.get("reason") or "fresh_sensor_features_unavailable"}
    with closing(_connect_ro(db_path)) as conn:
        conn.execute("BEGIN")
        selected = select_weather_run(read_weather_runs(conn, issue - WEATHER_LOOKBACK_SECONDS, issue), issue)
        wind_run = load_wind_run(conn, *selected) if selected is not None else None
    lineage["issuedWindInput"] = (
        {"source": WEATHER_SOURCE, "fetchedEpoch": wind_run.fetched, "visibleEpoch": wind_run.visible,
         "ageSeconds": issue - wind_run.fetched, "maxAgeSeconds": WEATHER_MAX_AGE_SECONDS,
         "fields": ["wind_speed_10m", "wind_direction_10m", "wind_speed_925hPa", "wind_direction_925hPa"],
         "role": "upwind_weighting_of_neighbour_inputs_only"} if wind_run is not None else None)
    if wind_run is None:
        return {**out, "reason": "issued_wind_unavailable_or_older_than_2h"}
    try:
        polls = read_neighbour_polls(archive_path or ARCHIVE_PATH, issue - SPATIAL_WINDOW_SECONDS, issue)
    except (OSError, sqlite3.Error) as error:
        return {**out, "reason": "neighbour_archive_unavailable", "errorType": type(error).__name__}
    polls = polls.loc[polls.receipt_epoch <= issue].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        features = _assemble(issue, query.to_numpy(dtype=float)[0], list(query.columns), lineage, table,
                             polls, wind_run.features(issue))
    lineage["neighbourInputs"] = {
        "archive": "research/external_airgradient/neighbors.sqlite3", "pollsReceivedByIssue": int(len(polls)),
        "latestReceiptEpoch": _finite(polls.receipt_epoch.max()) if len(polls) else None,
        "stationsWithBucketWithin90Minutes": int(features["sp_count"]),
        "stationAgeMinutes": {name: _finite(features.get(f"sp_{name}_age")) for name, _, _ in SL.STATIONS.values()},
        "visibility": "max(newest poll receipt, bucket end + 300 s)"}
    if not features["sp_count"] > 0:
        return {**out, "reason": "neighbour_inputs_unavailable_or_stale"}
    return {**out, "features": features}


def predict(db_path, rows, issue_epoch, *, archive_path=None, model_dir=None, artifact=None,
            allow_reconstruction=False):
    """Read-only inference for one issue: literal point heads plus the E1 probability ('le70')."""
    issue = int(issue_epoch)
    result = _base(issue, allow_reconstruction)
    try:
        live = live_issue_features(db_path, rows, issue, archive_path=archive_path)
        lineage = live["lineage"]
        result.update(inputLineage=lineage, freshReferencePm25=lineage.get("freshReferencePm25"),
                      featureSourceMaxEpoch=lineage.get("featureSourceMaxEpoch"))
        if live["features"] is None:
            extra = {"errorType": live["errorType"]} if live["errorType"] else {}
            return _unavailable(result, live["reason"], **extra)
        features = live["features"]
        artifact = load_artifact(model_dir) if artifact is None else artifact
        problem = _artifact_problem(artifact, issue, allow_reconstruction)
        if problem:
            return _unavailable(result, problem)
        metadata = artifact["metadata"]
        result.update(training=dict(metadata), modelIdentity=artifact.get("identity"),
                      trainingCutoffEpoch=artifact["trainingCutoffEpoch"])
        model_inputs = M._frame_from_dict(features, list(INPUT_COLUMNS))[list(INPUT_COLUMNS)].to_numpy(float)[0]
        values = [float(v) if math.isfinite(v) else None for v in model_inputs]
        with threadpool_limits(limits=FIT_CPU_THREADS), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            point = M.predict_point(artifact["point"], features)
            chance = M.predict_le70(artifact["le70"], features) if artifact.get("le70") is not None else float("nan")
        raw = {"arrivalPoint": point["pred_arrival"], "trailMeanPoint": point["pred_ride_mean"],
               "trailMinimumPoint": point["pred_ride_min"], "trailMaximumPoint": point["pred_ride_max"]}
        result.update(featureColumns=list(INPUT_COLUMNS), featureValues=values,
                      featureValuesSha256=_sha_text(values), queryFeaturesHash=_sha_text(values))
        if not all(math.isfinite(v) for v in raw.values()):
            return _unavailable(result, "spatial_point_model_output_invalid")
        result.update(raw, available=True, reason=None, rawOutputUgM3=dict(raw),
                      rangeOrdered=raw["trailMinimumPoint"] <= raw["trailMaximumPoint"],
                      outputsPublishedWithoutArithmetic=True)
        le70 = _le70_base(issue)
        le70.update(referencePm=_finite(result["freshReferencePm25"]), trainingCutoffEpoch=artifact["trainingCutoffEpoch"],
                    fittedAtEpoch=metadata.get("fittedAtEpoch"), modelIdentity=artifact.get("identity"),
                    rideMeanAnchorUgM3=raw["trailMeanPoint"], featureColumns=list(INPUT_COLUMNS),
                    featureValuesSha256=result["featureValuesSha256"],
                    residualTrainingCases=int(getattr(artifact.get("le70"), "n_train", 0) or 0))
        if artifact.get("le70") is None or not (math.isfinite(chance) and 0 <= chance <= 1):
            reason = metadata.get("le70Reason") or "le70_model_output_unavailable"
            le70.update(reason=reason, unavailableReasons=[reason])
        else:
            le70.update(available=True, reason="available", unavailableReasons=[], chanceMeanAtOrBelowCutoff=float(chance))
        result["le70"] = le70
        return result
    except Exception as error:
        return _unavailable(result, "spatial_point_inference_unavailable", errorType=type(error).__name__)


def cycling_window_output(forecast, issue_epoch):
    """The published cyclingWindowForecast object (E1), or an unavailable one with a reason."""
    issue = int(issue_epoch)
    le70 = (forecast or {}).get("le70") if isinstance(forecast, dict) else None
    if not isinstance(le70, dict) or le70.get("forecastIssuedEpoch") != issue:
        return _le70_base(issue, "spatial_point_forecast_unavailable")
    return dict(le70)


def record_issued_le70(result, *, output_path=None, recorded_epoch=None):
    """Append a genuinely published P(ride mean <= 70) once; call only after real publication."""
    recorded = int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue = int(result["forecastIssuedEpoch"])
    if not 0 <= recorded - issue <= 120:
        raise ValueError("Only a fresh actual publication may be recorded as originally issued")
    path = Path(output_path) if output_path is not None else PRIVATE / LE70_ISSUE_NAME

    def identity(item):
        return (str(path.resolve()), int(item["forecastIssuedEpoch"]), item.get("planId"), item.get("mode"),
                item.get("trainingCutoffEpoch"), item.get("featureValuesSha256"), item.get("reason"))

    key = identity(result)
    record = {**_json_value(result), "recordedEpoch": recorded, "publishedEpoch": recorded,
              "isOriginallyIssued": True, "predictionRole": "genuinely_issued_live_spatial_le70_prediction"}
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path.resolve()) not in _ISSUE_PATHS_READ:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2_000_000))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()
                for line in tail:
                    try:
                        _RECORDED[identity(json.loads(line))] = None
                    except (ValueError, KeyError, TypeError):
                        continue
            _ISSUE_PATHS_READ.add(str(path.resolve()))
        if key in _RECORDED:
            return False
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        _RECORDED[key] = None
        while len(_RECORDED) > 4096:
            _RECORDED.popitem(last=False)
    return True
