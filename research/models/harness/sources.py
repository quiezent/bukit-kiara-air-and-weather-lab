"""Read-only loaders for the snapshot DB and neighbour archive, with visibility clocks.

Every returned record carries ``visible_epoch``: the earliest time the value
could have been used by a forecast issued then.

* Sensor rows with a receipt (since 2026-10-08 13:12): max(available, confirmed)
  of the first receipt (production's per-revision availability).
* Older sensor rows: observation epoch + LEGACY_SENSOR_VISIBILITY_LAG_SECONDS.
* Weather / CAMS runs: receipt max(available, confirmed) when present, else
  fetched_epoch + LEGACY_SOURCE_VISIBILITY_LAG_SECONDS.
* Regional PM grid: recorded_epoch. Regional wind: fetched_epoch + lag.
* Subang METAR-style reports: first_fetched_epoch.
* Neighbours: receipt clock of the first request that returned the value
  (strict) and, separately, bucket end + 30 min (reconstructed assumption).
"""
from __future__ import annotations

from contextlib import closing
from functools import lru_cache
import gzip
import json
import sqlite3

import numpy as np
import pandas as pd

from . import config


def connect(path=None):
    path = config.SNAPSHOT_DB if path is None else path
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _decode(blob):
    if isinstance(blob, (bytes, bytearray, memoryview)):
        raw = bytes(blob)
        if raw.startswith(b"\x1f\x8b"):
            raw = gzip.decompress(raw)
        blob = raw.decode("utf-8")
    return json.loads(blob)


@lru_cache(maxsize=1)
def sensor_table() -> pd.DataFrame:
    """All readings with receipt knowledge and the harness visibility clock."""
    with closing(connect()) as conn:
        frame = pd.read_sql("SELECT epoch,pm02,atmp,rhum FROM readings ORDER BY epoch", conn)
        receipts = pd.read_sql(
            "SELECT r.observed_epoch AS epoch, r.receipt_id, r.received_epoch, r.available_epoch, "
            "v.confirmed_epoch FROM sensor_observation_receipts r "
            "JOIN sensor_observation_visibility v ON v.receipt_id=r.receipt_id ORDER BY r.receipt_id", conn)
    receipts["receipt_available"] = np.maximum(receipts.available_epoch, receipts.confirmed_epoch)
    first = receipts.groupby("epoch").agg(receipt_available=("receipt_available", "min"),
                                          received_epoch=("received_epoch", "min"))
    frame = frame.merge(first, left_on="epoch", right_index=True, how="left")
    frame["receipt_known"] = frame.receipt_available.notna()
    frame["visible_epoch"] = np.where(frame.receipt_known, frame.receipt_available,
                                      frame.epoch + config.LEGACY_SENSOR_VISIBILITY_LAG_SECONDS)
    for column in ("pm02", "atmp", "rhum"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def harness_rows(frame=None):
    """Production-format rows whose availability is the harness visibility clock."""
    frame = sensor_table() if frame is None else frame
    return [{"epoch": int(e), "pm02": p, "atmp": t, "rhum": h,
             "_provenance": {"receiptKnown": True, "availableEpoch": float(v)}}
            for e, p, t, h, v in zip(frame.epoch, frame.pm02, frame.atmp, frame.rhum, frame.visible_epoch)]


def production_receipt_rows(frame=None):
    """Production Ridge/logistic semantics: receipts known, legacy rows unknown (lag 0)."""
    frame = sensor_table() if frame is None else frame
    rows = []
    for e, p, t, h, known, available in zip(frame.epoch, frame.pm02, frame.atmp, frame.rhum,
                                            frame.receipt_known, frame.receipt_available):
        provenance = ({"receiptKnown": True, "availableEpoch": float(available)} if known
                      else {"receiptKnown": False, "availableEpoch": None})
        rows.append({"epoch": int(e), "pm02": p, "atmp": t, "rhum": h, "_provenance": provenance})
    return rows


def plain_rows(frame=None, visible_by=None):
    """Plain readings dicts (production first-crossing / session semantics).

    ``visible_by`` restricts to rows whose harness visibility <= that epoch.
    """
    frame = sensor_table() if frame is None else frame
    if visible_by is not None:
        frame = frame.loc[frame.visible_epoch <= visible_by]
    return [{"epoch": int(e), "pm02": p, "atmp": t, "rhum": h}
            for e, p, t, h in zip(frame.epoch, frame.pm02, frame.atmp, frame.rhum)]


def _source_receipts(conn, table):
    rows = conn.execute(
        "SELECT r.source_fetched_epoch, MIN(MAX(r.available_epoch, v.confirmed_epoch)) "
        "FROM source_forecast_receipts r JOIN source_forecast_visibility v ON v.receipt_id=r.receipt_id "
        "WHERE r.source_table=? GROUP BY r.source_fetched_epoch", (table,)).fetchall()
    return {int(f): float(a) for f, a in rows}


@lru_cache(maxsize=2)
def forecast_runs(table: str) -> tuple:
    """Issued Open-Meteo weather or CAMS runs, ordered by visibility clock.

    Returns tuples (fetched_epoch, visible_epoch, receipt_known, payload_dict).
    CAMS keeps only production-valid sources (Open-Meteo / CAMS Global,
    cams_anchor_v1/v2), as window_pm_predictor._raw_runs does.
    """
    if table not in ("weather_forecast_runs", "air_quality_forecast_runs"):
        raise ValueError(table)
    with closing(connect()) as conn:
        receipts = _source_receipts(conn, table)
        columns = "fetched_epoch,source,payload" + (",model_version" if table == "air_quality_forecast_runs" else "")
        records = conn.execute(f"SELECT {columns} FROM {table} ORDER BY fetched_epoch").fetchall()
    runs = []
    for record in records:
        fetched, source = int(record[0]), record[1]
        if table == "air_quality_forecast_runs" and (
                source != "Open-Meteo / CAMS Global" or record[3] not in ("cams_anchor_v1", "cams_anchor_v2")):
            continue
        if table == "weather_forecast_runs" and source != "Open-Meteo Best Match":
            continue
        payload = _decode(record[2])
        known = fetched in receipts
        visible = receipts[fetched] if known else fetched + config.LEGACY_SOURCE_VISIBILITY_LAG_SECONDS
        runs.append((fetched, float(visible), known, payload))
    runs.sort(key=lambda run: (run[1], run[0]))
    return tuple(runs)


def hourly_matrix(payload, fields):
    """Sorted hourly valid epochs and an (n, len(fields)) float matrix."""
    points = {int(p["epoch"]): p for p in payload.get("hourly", []) if p.get("epoch") is not None}
    epochs = np.array(sorted(points), dtype=float)
    matrix = np.full((len(epochs), len(fields)), np.nan)
    for i, epoch in enumerate(epochs):
        point = points[int(epoch)]
        for j, field in enumerate(fields):
            value = point.get(field)
            try:
                matrix[i, j] = float(value) if value is not None else np.nan
            except (TypeError, ValueError):
                pass
    return epochs, matrix


@lru_cache(maxsize=1)
def regional_pm_runs() -> tuple:
    """CAMS 3x3 regional grid (since 2026-10-01). (fetched, visible, times, lat, lon, pm25[time, point])."""
    with closing(connect()) as conn:
        records = conn.execute("SELECT fetched_epoch,recorded_epoch,payload FROM regional_pm_runs "
                               "ORDER BY fetched_epoch").fetchall()
    runs = []
    for fetched, recorded, blob in records:
        payload = _decode(blob)
        times = np.asarray(payload.get("times") or [], dtype=float)
        raw = payload.get("rawResponse") or []
        if not len(times) or not raw:
            continue
        lats, lons, series = [], [], []
        for location in raw:
            hourly = location.get("hourly") or {}
            values = hourly.get("pm2_5")
            if values is None or len(values) != len(times):
                continue
            lats.append(float(location.get("latitude")))
            lons.append(float(location.get("longitude")))
            series.append([np.nan if v is None else float(v) for v in values])
        if not series:
            continue
        visible = float(recorded) if recorded is not None else fetched + config.LEGACY_SOURCE_VISIBILITY_LAG_SECONDS
        runs.append((int(fetched), visible, times, np.array(lats), np.array(lons), np.array(series).T))
    runs.sort(key=lambda run: run[1])
    return tuple(runs)


@lru_cache(maxsize=1)
def regional_wind_runs() -> tuple:
    """GFS 925/850 hPa winds on a 6x5 grid (since 2026-09-05)."""
    with closing(connect()) as conn:
        records = conn.execute("SELECT fetched_epoch,payload FROM regional_wind_runs ORDER BY fetched_epoch").fetchall()
    runs = []
    for fetched, blob in records:
        payload = _decode(blob)
        times = np.asarray(payload.get("times") or [], dtype=float)
        points = payload.get("points") or []
        if not len(times) or not points:
            continue
        fields = ("wind_speed_925hPa", "wind_direction_925hPa", "wind_speed_850hPa", "wind_direction_850hPa")
        data = np.full((len(points), len(times), len(fields)), np.nan)
        lats, lons = [], []
        for i, point in enumerate(points):
            lats.append(float(point.get("latitude")))
            lons.append(float(point.get("longitude")))
            for j, field in enumerate(fields):
                values = point.get(field) or []
                if len(values) == len(times):
                    data[i, :, j] = [np.nan if v is None else float(v) for v in values]
        visible = int(payload.get("fetchedEpoch") or fetched) + config.LEGACY_SOURCE_VISIBILITY_LAG_SECONDS
        runs.append((int(fetched), float(visible), times, np.array(lats), np.array(lons), data))
    runs.sort(key=lambda run: run[1])
    return tuple(runs)


@lru_cache(maxsize=1)
def subang_table() -> pd.DataFrame:
    with closing(connect()) as conn:
        frame = pd.read_sql("SELECT report_epoch,atmp,rhum,wind_speed_kmh,wind_direction,dewpoint,visibility_km,"
                            "present_weather,precipitation_mm,precipitation_period_hours,first_fetched_epoch "
                            "FROM subang_observations ORDER BY first_fetched_epoch,report_epoch", conn)
    frame["visible_epoch"] = frame.first_fetched_epoch.astype(float)
    return frame


@lru_cache(maxsize=1)
def neighbor_table() -> pd.DataFrame:
    """Valid neighbour PM2.5 by station and observation end, with both clocks."""
    with closing(sqlite3.connect(f"file:{config.NEIGHBOR_DB.as_posix()}?mode=ro", uri=True)) as conn:
        frame = pd.read_sql(
            "SELECT o.location_id, o.kind, o.bucket_start_epoch + o.bucket_seconds AS end_epoch, o.pm25, "
            "MAX(r.received_epoch, r.recorded_epoch) AS receipt_epoch FROM observations o "
            "JOIN requests r ON r.request_id=o.request_id WHERE o.valid=1 AND o.measure='pm25'", conn)
        names = dict(conn.execute("SELECT location_id,name FROM locations").fetchall())
    frame = (frame.sort_values("receipt_epoch")
             .groupby(["location_id", "end_epoch"], as_index=False)
             .agg(pm25=("pm25", "first"), receipt_epoch=("receipt_epoch", "min"), kind=("kind", "first")))
    frame["reconstructed_visible"] = np.minimum(
        frame.receipt_epoch, frame.end_epoch + config.NEIGHBOR_RECONSTRUCTED_LAG_SECONDS)
    frame["name"] = frame.location_id.map(names)
    return frame.sort_values(["location_id", "end_epoch"]).reset_index(drop=True)
