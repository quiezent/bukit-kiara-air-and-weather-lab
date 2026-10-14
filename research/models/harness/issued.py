"""Genuinely issued production outputs (read only).

Sources, all read only:
* dashboard_forecast_issues: gzip ledger of every published payload since
  2026-09-07 (decoded with forecast_payload.loads semantics).
* server/private/*-issued*.jsonl: append-only records of the logistic,
  first-crossing HGB and cycling-window outputs (only lines recorded before
  the snapshot are used; the live files are never written).
* prospective_validation_issued: per-family case ledger since 2026-10-08 13:12.
"""
from __future__ import annotations

from contextlib import closing
import json

import numpy as np
import pandas as pd

from . import config, sources

LEDGER_PATH = config.OUT / "issued_ledger.parquet"
INCUMBENT_VERSIONS = {
    "ridge": "fresh_sequence_ridge_delta_numeric_v4_receipt_training_20261009",
    "arrival_logistic": "receipt_visible_logistic_arrival_change20_40_v1",
    "first_crossing": "fresh_sensor_hgb_direct_first20_90min_v2",
    "cycling70": "ttdi_fixed_ride_window70_live_v1",
    "morning": "morning_hgb_basic31_model_direct_v3",
    "afternoon": "afternoon_patchtst_original_model_direct_v3",
}


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _row(issued, forecast_epoch, sensor_epoch, build, payload):
    air = payload.get("airWindow") or {}
    fresh = payload.get("freshSensorForecast") or {}
    arrival, trail = air.get("arrival") or {}, air.get("trail") or {}
    change = air.get("arrivalChangeForecast") or {}
    first = air.get("firstCrossingEventForecast") or {}
    cycling = air.get("cyclingWindowForecast") or {}
    windows = payload.get("windows") or {}
    record = {
        "published_epoch": int(issued), "issue_epoch": int(payload.get("forecastIssuedEpoch") or forecast_epoch),
        "sensor_epoch": sensor_epoch, "build": build,
        "ref": _num(fresh.get("freshReferencePm25", arrival.get("baselinePoint"))),
        "arrival_model": arrival.get("modelVersion"), "arrival_point": _num(arrival.get("point")) if arrival.get("available") else np.nan,
        "ride_model": trail.get("modelVersion"), "ride_mean_point": _num(trail.get("point")) if trail.get("available") else np.nan,
        "fresh_model": fresh.get("modelVersion"),
        "ride_min_point": _num(fresh.get("trailMinimumPoint")) if fresh.get("available") else np.nan,
        "ride_max_point": _num(fresh.get("trailMaximumPoint")) if fresh.get("available") else np.nan,
        "ac_model": change.get("modelVersion"), "ac_available": bool(change.get("available")),
        "fc_model": first.get("modelVersion"), "fc_available": bool(first.get("available")),
        "fc_p_drop": _num(first.get("probabilityDrop")), "fc_p_none": _num(first.get("probabilityNoCrossing")),
        "fc_p_rise": _num(first.get("probabilityRise")),
        "cy_model": cycling.get("modelVersion"), "cy_available": bool(cycling.get("available")),
        "cy_p_le70": _num(cycling.get("chanceMeanAtOrBelowCutoff")),
        "cy_start": _num(cycling.get("startEpoch")), "cy_end": _num(cycling.get("endEpoch")),
    }
    probabilities = change.get("classProbabilities") if change.get("available") else None
    for i, name in enumerate(("p_fall40", "p_fall20to40", "p_within20", "p_rise20to40", "p_rise40")):
        record["ac_" + name] = _num(probabilities[i]) if probabilities else np.nan
    for key in ("morning", "afternoon"):
        window = windows.get(key) or {}
        forecast = window.get("particleForecast") or {}
        record[f"{key}_model"] = forecast.get("modelVersion")
        record[f"{key}_point"] = _num(forecast.get("point", forecast.get("mean"))) if forecast.get("available") else np.nan
        record[f"{key}_baseline"] = _num(forecast.get("baselinePoint"))
        record[f"{key}_start"] = _num(window.get("startEpoch"))
        record[f"{key}_end"] = _num(window.get("endEpoch"))
    return record


def ledger(refresh=False) -> pd.DataFrame:
    """Decode every published payload once; cached as parquet."""
    if LEDGER_PATH.exists() and not refresh:
        return pd.read_parquet(LEDGER_PATH)
    records = []
    with closing(sources.connect()) as conn:
        cursor = conn.execute("SELECT issued_epoch,forecast_epoch,sensor_epoch,dashboard_build,payload "
                              "FROM dashboard_forecast_issues ORDER BY issued_epoch,issue_id")
        for issued, forecast_epoch, sensor_epoch, build, blob in cursor:
            try:
                records.append(_row(issued, forecast_epoch, sensor_epoch, build, sources._decode(blob)))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    frame = pd.DataFrame(records)
    frame["ac_p_fall20"] = frame.ac_p_fall40 + frame.ac_p_fall20to40
    frame["ac_p_rise20"] = frame.ac_p_rise40 + frame.ac_p_rise20to40
    frame.to_parquet(LEDGER_PATH)
    return frame


def jsonl(name: str, fields=None) -> pd.DataFrame:
    """Read an append-only issued JSONL (lines recorded before the snapshot only)."""
    path = config.ISSUED_JSONL_DIR / name
    records = []
    with path.open("rb") as stream:                         # read only
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue                                   # a partially written tail line
            if int(record.get("recordedEpoch", 0)) > config.SNAPSHOT_TAKEN_EPOCH:
                break
            if fields is not None:
                record = {k: record.get(k) for k in fields}
            records.append(record)
    return pd.DataFrame(records)


def jsonl_tables() -> dict:
    common = ["forecastIssuedEpoch", "recordedEpoch", "modelVersion", "available", "reason",
              "referencePm", "trainingCutoffEpoch"]
    return {
        "arrival_logistic": jsonl("arrival-logistic-change-issued-v1.jsonl", common + ["classProbabilities"]),
        "first_crossing": jsonl("fresh-event-issued-v2.jsonl",
                                common + ["probabilityDrop", "probabilityNoCrossing", "probabilityRise"]),
        "cycling70": jsonl("cycling-window70-issued.jsonl",
                           common + ["chanceMeanAtOrBelowCutoff", "startEpoch", "endEpoch", "mode"]),
    }


def prospective_cases() -> pd.DataFrame:
    with closing(sources.connect()) as conn:
        rows = conn.execute("SELECT issue_epoch,recorded_epoch,target_family,head,cohort,start_epoch,end_epoch,"
                            "candidate_version,actual_model,payload FROM prospective_validation_issued").fetchall()
    records = []
    for issue, recorded, family, head, cohort, start, end, candidate, model, blob in rows:
        payload = sources._decode(blob)
        records.append({"issue_epoch": issue, "recorded_epoch": recorded, "family": family, "head": head,
                        "cohort": cohort, "start_epoch": start, "end_epoch": end, "candidate": candidate,
                        "model": model, "point": payload.get("point"), "baseline": payload.get("baselinePoint"),
                        "available": payload.get("available"), "probability": payload.get("probability")})
    return pd.DataFrame(records)


def slot_sample(frame: pd.DataFrame, start=None, end=None, column="issue_epoch") -> pd.DataFrame:
    """First issued forecast in each 5-min grid slot (exact issue clock retained)."""
    frame = frame.sort_values([column, "published_epoch"] if "published_epoch" in frame else [column])
    frame = frame.drop_duplicates(subset=[column], keep="first")
    if start is not None:
        frame = frame.loc[frame[column] >= start]
    if end is not None:
        frame = frame.loc[frame[column] < end]
    slot = frame[column] // config.GRID_SECONDS
    return frame.loc[~slot.duplicated(keep="first")].reset_index(drop=True)


def canonical_sample(frame: pd.DataFrame, tolerance=120) -> pd.DataFrame:
    """First publication within +120 s of each canonical clock (prospective_validation rule)."""
    local = pd.to_datetime(frame.issue_epoch, unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
    out = []
    for clock in config.CANONICAL_CLOCKS:
        hour, minute = map(int, clock.split(":"))
        tick = local.dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)
        offset = (local - tick).dt.total_seconds()
        part = frame.loc[(offset >= 0) & (offset <= tolerance)].copy()
        part["canonical"] = clock
        part["canonical_tick"] = (tick.loc[part.index].astype("int64") // 10**9).to_numpy()
        out.append(part.sort_values("issue_epoch").drop_duplicates("canonical_tick"))
    return pd.concat(out).sort_values("issue_epoch").reset_index(drop=True)
