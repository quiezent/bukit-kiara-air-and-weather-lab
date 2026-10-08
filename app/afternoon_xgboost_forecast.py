"""Explicitly selected experimental Afternoon mean; daily background CPU fit.

This production seam is separate from frozen research helpers and the Morning
model. HTTP prediction never fits a model. Missing inputs leave routing to the
previous forecast, and this version publishes no new uncertainty interval.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from types import MappingProxyType

import numpy as np
import pandas as pd

import afternoon_direction_features as causal
import rain_weather_features as rain
import weather_session_features as richer
import window_pm_predictor as legacy
from forecast_freshness import raw_arrays, VERSION as FRESHNESS_VERSION
import session_prediction_policy as selection
import observation_provenance as provenance
from forecast_payload import decode_payload_text

LEGACY_MODEL_VERSION = "afternoon_xgboost_basic31_mean_v1"
MODEL_VERSION = "afternoon_xgboost_basic31_qualified_mean_v2"
FEATURE_VERSION = "guarded_half_hour_zero_offset_basic31_session_v1"
TRAINING_NUMERICAL_POLICY = "daily_midnight_14d_guarded_basic31_xgb_absolute_mean_v1"
NUMERICAL_POLICY = "daily_midnight_14d_guarded_basic31_xgb_raw_canonical_issued_gate_v2"
SELECTION_POLICY = selection.VERSION
LIVE_INPUT_POLICY = "current_collected_snapshot_watermark_v1"
FEATURE_COLUMNS = tuple(causal.FEATURES)
LEAD_MINUTES = (105, 120, 240, 420, 750, 1080, 1380)
PARAMETERS = {
    "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
    "min_child_weight": 20.0, "reg_lambda": 50.0, "reg_alpha": 1.0,
    "gamma": 1.0, "max_bin": 64, "subsample": 1.0,
    "colsample_bytree": 1.0, "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
    "objective": "reg:absoluteerror", "eval_metric": "mae", "tree_method": "hist",
    "grow_policy": "depthwise", "booster": "gbtree", "random_state": 1749,
    "n_jobs": 1, "verbosity": 1, "validate_parameters": True,
}
_LOCK = threading.RLock()
_FITS = OrderedDict()
_STATUS = {}
_RETRY_SECONDS = 300


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def training_cutoff(issue):
    return int(pd.Timestamp(int(issue), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur").normalize().timestamp())


def _key(db_path, cutoff):
    return str(Path(db_path).resolve()), int(cutoff)


def _source(db_path, cutoff, rows=None, lookback_days=15):
    """Bounded original archive snapshot; no backfill/research inputs or writes."""
    cutoff = int(cutoff)
    start = cutoff - lookback_days * 86400
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=20)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        if rows is None:
            rows = provenance.compatible_sensor_rows(conn, cutoff, start_epoch=start, end_epoch=cutoff)
        else:
            rows = [dict(r) for r in rows if _number(dict(r).get("epoch")) is not None
                    and start <= float(dict(r)["epoch"]) <= cutoff]
        rows = sorted(rows, key=lambda r: float(r["epoch"]))
        publications = list(conn.execute(
            "SELECT issued_epoch,sensor_epoch FROM dashboard_forecast_issues "
            "WHERE issued_epoch>=? AND issued_epoch<=? ORDER BY issued_epoch,issue_id", (start, cutoff)))
        wr = provenance.compatible_source_runs(conn, "weather_forecast_runs", cutoff,
                                               start_fetched_epoch=start-7200, include_revisions=True)
        cr = provenance.compatible_source_runs(conn, "air_quality_forecast_runs", cutoff,
                                               start_fetched_epoch=start-7200, include_revisions=True)
        cr = [r for r in cr if r.get("model_version") in ("cams_anchor_v1", "cams_anchor_v2")]
    weather_hash = hashlib.sha256()
    weather_runs = []
    for record in wr:
        fetched, source, payload = record["fetched_epoch"], record["source"], decode_payload_text(record["payload"])
        receipt = record["_provenance"]
        weather_hash.update(repr((fetched, source)).encode())
        weather_hash.update(payload.encode())
        weather_hash.update(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
        weather_runs.append(replace(rain._parse_run(float(fetched), str(source), payload),
                                    receipt_provenance=MappingProxyType(dict(receipt))))
    weather = rain.RunArchive(tuple(weather_runs), tuple(r.fetched_epoch for r in weather_runs),
                              cutoff, weather_hash.hexdigest(), str(path))
    cams_hash = hashlib.sha256()
    cams_raw = legacy.SourceRuns()
    for record in cr:
        fetched, source, payload = record["fetched_epoch"], record["source"], decode_payload_text(record["payload"])
        receipt = record["_provenance"]
        cams_hash.update(repr((fetched, source)).encode())
        cams_hash.update(payload.encode())
        cams_hash.update(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
        if source == "Open-Meteo / CAMS Global":
            cams_raw.append((fetched, payload))
            cams_raw.receipt_provenance.append(dict(receipt))
    cams = legacy._read_runs(path, "air_quality_forecast_runs", cutoff,
                            raw_snapshot=(cams_raw, cams_hash.hexdigest()))
    # Scalar feature helpers must not coerce nested receipt metadata into a
    # sensor channel. Preserve immutable receipt evidence alongside the values.
    sensor_rows = [{k: dict(r).get(k) for k in ("epoch", "pm02", "atmp", "rhum")} for r in rows]
    frame = causal.strict_frame(sensor_rows, cutoff)
    raw = pd.DataFrame(sensor_rows)
    if len(raw):
        raw = raw.sort_values("epoch")
        raw.index = pd.to_datetime(raw.epoch, unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
        source_max = raw.epoch.resample("15min", label="right", closed="right").max().rolling(12, min_periods=1).max()
        available = [(_number(dict(r).get("_provenance", {}).get("availableEpoch"))
                      if dict(r).get("_provenance", {}).get("receiptKnown") is True else np.nan) for r in rows]
        available = pd.Series(available, index=raw.index, dtype=float)
        source_available = available.resample("15min", label="right", closed="right").max().rolling(12, min_periods=1).max()
    else:
        source_max = pd.Series(dtype=float)
        source_available = pd.Series(dtype=float)
    epochs, values = raw_arrays(sensor_rows, cutoff)
    receipt_by_epoch = {float(r["epoch"]): dict(r).get("_provenance") or {} for r in rows}
    raw_available = np.array([(_number(receipt_by_epoch.get(float(e), {}).get("availableEpoch"))
                               if receipt_by_epoch.get(float(e), {}).get("receiptKnown") is True else np.nan) for e in epochs], dtype=float)
    return {"rows": sensor_rows, "frame": frame, "weather": weather, "cams": cams, "cutoff": cutoff,
            "rawEpochs": epochs, "rawPm": values, "sourceMax": source_max,
            "rawAvailableEpochs": raw_available, "sourceAvailableMax": source_available,
            "sensorReceiptEvidence": receipt_by_epoch,
            "receiptPolicy": "immutable_available_revision_per_issue_unknown_legacy_receipts_explicit",
            "sensorReceiptKnownCount": sum(r.get("receiptKnown") is True for r in receipt_by_epoch.values()),
            "sensorReceiptUnknownCount": sum(r.get("receiptKnown") is not True for r in receipt_by_epoch.values()),
            "pubEpochs": np.array([r[0] for r in publications], dtype=np.int64),
            "pubMarks": np.array([r[1] if r[1] is not None else np.nan for r in publications], dtype=float),
            "weatherRevision": weather_hash.hexdigest(), "camsRevision": cams_hash.hexdigest()}


def _reference_at_issue(source, issue, watermark):
    """Known later receipts cannot enter a historical five-minute reference."""
    lo = np.searchsorted(source["rawEpochs"], issue-300, side="right")
    hi = np.searchsorted(source["rawEpochs"], min(issue, watermark), side="right")
    if hi <= lo:
        return np.nan, np.nan, 0
    available = source.get("rawAvailableEpochs", np.full(len(source["rawEpochs"]), np.nan))[lo:hi]
    eligible = ~np.isfinite(available) | (available <= issue)
    epochs, values = source["rawEpochs"][lo:hi][eligible], source["rawPm"][lo:hi][eligible]
    return (float(np.median(values)), float(epochs[-1]), len(values)) if len(values) else (np.nan, np.nan, 0)


def _feature_receipts_available(source, origins, issues):
    """Withhold touched late revisions instead of replaying revised values early."""
    maximum = source.get("sourceAvailableMax", pd.Series(dtype=float)).reindex(origins).to_numpy(float)
    return ~np.isfinite(maximum) | (maximum <= issues)


def _guarded_part(source, origins, lead_minutes, lag_seconds=0):
    """Match the evaluated watermark-guarded cohort, including fresh recomputation."""
    d = causal.design(source["frame"], source["rows"], source["cams"], source["weather"],
                      source["cutoff"], lead_minutes, 120, lag_seconds=lag_seconds, only_origins=origins)
    issues = np.asarray(d["issues"], dtype=np.int64)
    x = d["x"].copy()
    pub = source["pubEpochs"]
    ix = np.searchsorted(pub, issues, side="right") - 1
    marks, publications = np.full(len(ix), np.nan), np.full(len(ix), np.nan)
    found = ix >= 0
    marks[found] = source["pubMarks"][ix[found]]
    publications[found] = pub[ix[found]]
    fresh, stamps, counts = np.full(len(ix), np.nan), np.full(len(ix), np.nan), np.zeros(len(ix), dtype=int)
    for i, (issue, mark) in enumerate(zip(issues, marks)):
        if not np.isfinite(mark) or mark > issue:
            continue
        fresh[i], stamps[i], counts[i] = _reference_at_issue(source, issue, mark)
    x["fresh"], x["freshOffset"] = fresh, fresh - x.closedLevel
    descriptions = richer.describe_windows(source["weather"], zip(issues, d["start"], d["end"]))
    weather_fetch = np.array([_number(r.get("fetchedEpoch")) or np.nan for r in descriptions], dtype=float)
    maximum = source["sourceMax"].reindex(origins).to_numpy(float)
    pubage = issues - publications
    age = issues - stamps
    closed = x.closedLevel.to_numpy(float)
    valid = (np.isfinite(fresh) & (fresh >= 0) & np.isfinite(closed) &
             _feature_receipts_available(source, origins, issues) &
             (closed >= 0) & np.isfinite(marks) & (marks <= publications) &
             (publications <= issues) & np.isfinite(maximum) & (maximum <= marks) &
             np.isfinite(stamps) & (stamps <= marks) & (stamps <= issues) &
             (age >= 0) & (age <= 240) & (pubage >= 0) & (pubage <= 120) &
             np.isfinite(weather_fetch) & (weather_fetch <= issues) & (issues - weather_fetch <= 7200) &
             np.array([bool(r["available"]) for r in descriptions]))
    return {**d, "x": x.reindex(columns=FEATURE_COLUMNS), "valid": valid,
            "watermarks": marks, "publicationEpochs": publications, "publicationAges": pubage,
            "freshEpochs": stamps, "freshCounts": counts, "featureMaxEpochs": maximum,
            "weatherFetched": weather_fetch, "weatherDescriptions": descriptions}


def _live_part(source, origins, lead_minutes, lag_seconds, live_snapshot):
    """Use the actual collected live snapshot; training keeps its archive guard.

    A snapshot receipt clock is distinct from the previous forecast publication.
    The caller reads the rows before freezing the live issue clock.
    """
    q = _guarded_part(source, origins, lead_minutes, lag_seconds)
    issues = np.asarray(q["issues"], dtype=np.int64)
    receipt = _number((live_snapshot or {}).get("receivedEpoch"))
    watermark = _number((live_snapshot or {}).get("sensorWatermarkEpoch"))
    maximum_known = max((float(r["epoch"]) for r in source["rows"]), default=None)
    proof_valid = bool(receipt is not None and watermark is not None and maximum_known is not None
                       and watermark == maximum_known and watermark <= receipt
                       and source["cutoff"] == int(issues.max())
                       and np.all(receipt <= issues))
    fresh, stamps, counts = np.full(len(issues), np.nan), np.full(len(issues), np.nan), np.zeros(len(issues), dtype=int)
    if proof_valid:
        for i, issue in enumerate(issues):
            fresh[i], stamps[i], counts[i] = _reference_at_issue(source, issue, watermark)
    x = q["x"].copy()
    x["fresh"], x["freshOffset"] = fresh, fresh - x.closedLevel
    reference_age = issues - stamps
    closed = x.closedLevel.to_numpy(float)
    maximum = q["featureMaxEpochs"]
    weather_fetch = q["weatherFetched"]
    valid = (proof_valid & np.isfinite(fresh) & (fresh >= 0) & np.isfinite(closed) & (closed >= 0)
             & _feature_receipts_available(source, origins, issues)
             & np.isfinite(maximum) & (maximum <= (watermark if watermark is not None else np.nan))
             & np.isfinite(stamps) & (stamps <= issues)
             & (reference_age >= 0) & (reference_age <= 240)
             & np.isfinite(weather_fetch) & (weather_fetch <= issues) & (issues - weather_fetch <= 7200)
             & np.array([bool(r["available"]) for r in q["weatherDescriptions"]]))
    return {**q, "x": x.reindex(columns=FEATURE_COLUMNS), "valid": valid,
            "watermarks": np.full(len(issues), watermark if watermark is not None else np.nan),
            "freshEpochs": stamps, "freshCounts": counts, "liveSnapshotValid": proof_valid,
            "inputSnapshotReceivedEpochs": np.full(len(issues), receipt if receipt is not None else np.nan),
            "inputAvailabilityPolicy": LIVE_INPUT_POLICY}


def build_training_dataset(db_path, cutoff):
    source = _source(db_path, cutoff)
    frame = source["frame"]
    origins = frame.index[(frame.index.minute.isin([0, 30])) &
                          (frame.index.asi8 // 10**9 >= cutoff - 14 * 86400)]
    parts = [_guarded_part(source, origins, lead) for lead in LEAD_MINUTES]
    x = pd.concat([p["x"] for p in parts], ignore_index=True)
    fields = ("labels", "issues", "complete", "start", "end", "valid", "watermarks",
              "publicationEpochs", "publicationAges", "freshEpochs", "featureMaxEpochs", "weatherFetched")
    data = {name: np.concatenate([p[name] for p in parts]) for name in fields}
    data.update(x=x, source=source)
    mask = (data["valid"] & np.isfinite(data["labels"]) & (data["labels"] >= 0) &
            (data["complete"] < cutoff) & (data["issues"] < cutoff) &
            (data["issues"] >= cutoff - 14 * 86400))
    data["eligible"] = mask
    return data


def fit_now(db_path, cutoff):
    """Explicit background/preflight entry; never called inline by prediction."""
    import xgboost
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    data = build_training_dataset(db_path, int(cutoff))
    valid = data["eligible"]
    days = pd.to_datetime(data["start"][valid], unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur").date
    if valid.sum() < 120 or len(set(days)) < 10:
        raise RuntimeError("insufficient_guarded_complete_training_history")
    x = data["x"].loc[valid].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    y = data["labels"][valid].astype(np.float32)
    estimator = xgboost.XGBRegressor(**PARAMETERS, device="cpu")
    with threadpool_limits(limits=1):
        estimator.fit(x, y, verbose=False)
    digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    fitted = {"estimator": estimator, "trainingCount": int(valid.sum()),
              "trainingDistinctTargetDays": len(set(days)), "trainingCutoffEpoch": int(cutoff),
              "trainingLatestTargetEndEpoch": int(data["complete"][valid].max()),
              "trainingFeatureLabelSha256": digest,
              "trainingWeatherRevision": data["source"]["weatherRevision"],
              "trainingCamsRevision": data["source"]["camsRevision"],
              "trainingReceiptPolicy": data["source"]["receiptPolicy"],
              "trainingSensorReceiptKnownCount": data["source"]["sensorReceiptKnownCount"],
              "trainingSensorReceiptUnknownCount": data["source"]["sensorReceiptUnknownCount"],
              "fittedAtEpoch": time.time(), "fitSeconds": round(time.perf_counter() - started, 3)}
    with _LOCK:
        _FITS[_key(db_path, cutoff)] = fitted
        while len(_FITS) > 4:
            _FITS.popitem(last=False)
    return fitted


def ensure_background_fit(db_path, issue_epoch):
    key = _key(db_path, training_cutoff(issue_epoch))
    with _LOCK:
        if key in _FITS:
            return {"state": "ready", "trainingCutoffEpoch": key[1]}
        previous = _STATUS.get(key, {})
        if previous.get("state") == "fitting" or time.time() - previous.get("attemptEpoch", 0) < _RETRY_SECONDS:
            return dict(previous)
        _STATUS[key] = {"state": "fitting", "attemptEpoch": int(time.time()), "trainingCutoffEpoch": key[1]}
        def work():
            try:
                fit_now(db_path, key[1])
                state = {"state": "ready", "attemptEpoch": int(time.time()), "trainingCutoffEpoch": key[1]}
            except Exception as error:
                state = {"state": "unavailable", "attemptEpoch": int(time.time()),
                         "trainingCutoffEpoch": key[1], "errorType": type(error).__name__, "reason": str(error)}
            with _LOCK:
                _STATUS[key] = state
        threading.Thread(target=work, name="TTDI-Afternoon-XGB-Fit", daemon=True).start()
        return dict(_STATUS[key])


def predict_windows(db_path, rows, windows, issue_epoch, *, live_snapshot=None):
    issue = int(issue_epoch)
    if "afternoon" not in windows:
        return {}
    value = windows["afternoon"]
    start, end = _number(value.get("startEpoch")), _number(value.get("endEpoch"))
    scope = selection.candidate_scope(issue, start, end, "afternoon")
    if not scope.get("rawCandidatePermitted"):
        return {"afternoon": selection.baseline_session_result(
            rows, issue, start, end, "afternoon", scope["blockedReasons"][0], MODEL_VERSION)}
    state = ensure_background_fit(db_path, issue)
    with _LOCK:
        fitted = _FITS.get(_key(db_path, training_cutoff(issue)))
    failure = None if fitted else "afternoon_xgboost_background_fit_pending_or_unavailable"
    if fitted and fitted["fittedAtEpoch"] > issue:
        failure = "afternoon_xgboost_fit_completed_after_query_issue"
    origin = issue // 900 * 900
    if (start is None or end is None or end - start != 7200 or start <= issue or
            (start - origin) % 900 or not 105 <= (start - issue) / 60 <= 1380):
        failure = "unsupported_session_alignment_or_horizon"
    if failure:
        fallback = selection.baseline_session_result(rows, issue, start, end, "afternoon", failure, MODEL_VERSION)
        fallback["weatherSessionModel"] = {"applied": False, "reason": failure, "fitStatus": state}
        return {"afternoon": fallback}
    source = _source(db_path, issue, rows=rows, lookback_days=1)
    stamp = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    if stamp not in source["frame"].index:
        failure = "missing_closed_sensor_origin"
    else:
        if live_snapshot is None:
            q = _guarded_part(source, pd.DatetimeIndex([stamp]), (start - issue) / 60, issue - origin)
        else:
            q = _live_part(source, pd.DatetimeIndex([stamp]), (start - issue) / 60, issue - origin, live_snapshot)
        if not q["valid"][0]:
            failure = ("invalid_current_collected_sensor_snapshot" if live_snapshot is not None
                       and not q["liveSnapshotValid"] else "missing_stale_or_unpublished_sensor_or_weather_inputs")
    if failure:
        fallback = selection.baseline_session_result(rows, issue, start, end, "afternoon", failure, MODEL_VERSION)
        fallback["weatherSessionModel"] = {"applied": False, "reason": failure, "fitStatus": state}
        return {"afternoon": fallback}
    from threadpoolctl import threadpool_limits
    x = q["x"].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    with threadpool_limits(limits=1):
        point = _number(fitted["estimator"].predict(x)[0])
    if point is None:
        return {"afternoon": selection.baseline_session_result(
            rows, issue, start, end, "afternoon", "nonfinite_xgboost_prediction", MODEL_VERSION)}
    point, anchor = max(0, point), float(q["x"].fresh.iloc[0])
    target_local = pd.Timestamp(int(start), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    issue_local = pd.Timestamp(issue, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    lead_scope = "same_day_afternoon" if target_local.date() == issue_local.date() else "next_day_afternoon"
    issue_clock = issue_local.strftime("%H:%M:%S")
    target_clock = target_local.strftime("%H:%M:%S")
    evaluated_clock = bool(target_clock == "14:00:00" and (
        (lead_scope == "same_day_afternoon" and issue_clock == "12:00:00") or
        (lead_scope == "next_day_afternoon" and issue_clock == "18:05:00")))
    evaluated_origin = bool(evaluated_clock and lead_scope == "next_day_afternoon")
    actual_lead = (start - issue) / 60
    feature_domain = {"actualLeadMinutes": round(actual_lead, 3), "trainingMinimumLeadMinutes": 105,
                      "trainingMaximumLeadMinutes": 1380,
                      "trainingLeadRange": [105, 1380],
                      "queryLeadWithinTrainingRange": bool(105 <= actual_lead <= 1380),
                      "withinFittedLeadRange": bool(105 <= actual_lead <= 1380),
                       "boundaryCaveat": "Actual leads outside 105–1380 minutes are withheld; the target is never moved to accommodate the model."}
    development_scope = {"strongest": "same_day_12_00_issue_14_00_to_16_00_target",
                         "weaker": "previous_day_18_05_issue_next_14_00_to_16_00_target",
                         "queryMatchesEvaluatedClock": evaluated_clock,
                         "queryMatchesEvaluatedInputOrigin": evaluated_origin,
                         "applicability": "experimental_outside_tested_clocks_or_origin_alignment" if not evaluated_origin else "matches_fixed_clock_development_setup_not_prospective_validation",
                         "sameDayOriginCaveat": "Fixed noon research used the 11:45 closed origin; live queries use the latest closed quarter-hour origin."}
    description = q["weatherDescriptions"][0]
    model = {"applied": True, "experimental": True, "prospectivelyValidated": False,
             "featureVersion": FEATURE_VERSION, "numericalPolicy": NUMERICAL_POLICY,
              "trainingNumericalPolicy": TRAINING_NUMERICAL_POLICY,
              "legacyModelVersion": LEGACY_MODEL_VERSION,
              "originalTrainingRecipeUnchanged": True,
             "learner": "100-tree shallow XGBoost absolute session-mean regression",
             "featureColumns": list(FEATURE_COLUMNS), "parameters": dict(PARAMETERS),
             **{k: v for k, v in fitted.items() if k != "estimator"},
             "trainingPolicy": "guarded_half_hour_zero_offset_seven_leads_daily_midnight_14d_completed_labels",
              "weatherAsOfPolicy": "immutable_receipt_revision_available_at_actual_issue_age_at_most_two_hours_legacy_receipts_unknown",
             "weatherFeatures": description["featureValues"], "weatherMetadata": description["metadata"],
             "usedWeatherFeatureValues": {c: _number(q["x"].iloc[0][c]) for c in FEATURE_COLUMNS
                                          if c in causal.WEATHER or c.startswith("cams")},
             "weatherFeaturesRole": "basic31 rain_weather_and_CAMS_inputs; richer_channels_are_diagnostic_only",
             "probabilityRole": "weather_rain_probability_is_not_PM_washout_probability"}
    model.update(developmentScope=development_scope, featureDomain=feature_domain)
    result = {"afternoon": {
        "available": True, "modelVersion": MODEL_VERSION, "source": "experimental_afternoon_xgboost",
        "pointRole": "experimental_session_mean", "forecastState": "experimental_weather_session_mean",
        "mean": round(point, 1), "prediction": round(point, 1), "sensorAnchor": round(anchor, 1),
        "closedSensorAnchor": round(float(q["x"].closedLevel.iloc[0]), 1),
        "sensorReferenceEpoch": int(q["freshEpochs"][0]), "sensorReferenceCount": int(q["freshCounts"][0]),
        "sensorWatermarkEpoch": int(q["watermarks"][0]), "featureSourceMaxEpoch": int(q["featureMaxEpochs"][0]),
        "publicationEpoch": int(q["publicationEpochs"][0]) if np.isfinite(q["publicationEpochs"][0]) else None,
        "publicationAgeSeconds": int(q["publicationAges"][0]) if np.isfinite(q["publicationAges"][0]) else None,
        "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median_capped_by_published_sensor_watermark",
        "originEpoch": origin, "forecastedAtEpoch": issue, "startEpoch": int(start), "endEpoch": int(end),
        "leadHours": round((start - origin) / 3600, 2), "remainingLeadHours": round((start - issue) / 3600, 2),
        "target": "overlap_duration_weighted_mean_of_complete_15_minute_raw_sensor_medians",
        "method": "Experimental XGBoost · exact 2-hour Afternoon mean",
        "rawRangeLow": None, "rawRangeHigh": None, "calibrated": False, "experimental": True,
        "rangeRole": "unavailable_new_selected_model_issued_accuracy_collecting",
        "uncertaintyMethod": "no_new_model_interval_published", "confidence": "low",
        "usedForDecision": True, "usedForComparison": False,
        "modelSelection": {"version": SELECTION_POLICY, "selectedPolicy": "user_selected_experimental_candidate",
                           "selectedModelVersion": MODEL_VERSION, "candidateModelVersion": MODEL_VERSION,
                           "selectionReason": "explicit_user_request_deploy_afternoon_xgboost",
                           "selectionReasonText": "Experimental Afternoon XGBoost selected at your request; issued accuracy collecting.",
                           "appliedToPrimaryForecast": True, "skillGatePassed": False,
                           "gatesAppliedToSelection": False, "prospectivelyValidated": False,
                           "evidenceScope": lead_scope,
                           "developmentScope": development_scope, "featureDomain": feature_domain,
                           "evidenceCaveat": "Development evidence was strongest for fixed same-day 12:00 issue to 14:00-16:00; next-day 18:05 issue evidence was weaker. Arbitrary live issue clocks/windows are not an exact replay of those checks.",
                           "forecastIssuedEpoch": issue, "targetStartEpoch": int(start), "targetEndEpoch": int(end)},
        "candidateForecast": {"available": True, "modelVersion": MODEL_VERSION, "mean": round(point, 1), "prediction": round(point, 1),
                              "numericalPolicyIdentifier": NUMERICAL_POLICY, "forecastedAtEpoch": issue,
                              "startEpoch": int(start), "endEpoch": int(end), "rawRangeLow": None, "rawRangeHigh": None},
        "freshnessAdjustment": {"applied": False, "appliedToSelectedPoint": False, "version": FRESHNESS_VERSION,
                                "referenceUsedAsModelInput": True, "referenceEpoch": int(q["freshEpochs"][0]),
                                "referenceCount": int(q["freshCounts"][0])},
        "weatherSessionModel": model,
        "modelEvidence": {"available": False, "count": 0, "distinctDays": 0, "supportSufficient": False,
                          "skillGatePassed": False, "calibrated": False, "prospectivelyValidated": False,
                          "candidateModelVersion": MODEL_VERSION, "validationMode": "new_model_issued_accuracy_collecting",
                          "scope": lead_scope, "retrospectiveStrongestScope": "same_day_fixed_12_00_issue_14_00_to_16_00_target",
                          "developmentScope": development_scope, "featureDomain": feature_domain,
                          "nextDayEvidence": "weaker_development_evidence_not_a_demonstrated_replacement_gain"},
        "trainingCount": fitted["trainingCount"], "weatherFetchedEpoch": int(q["weatherFetched"][0]),
        "camsFetchedEpoch": q["audit"][0].get("camsFetchedEpoch"),
        "camsAvailable": _number(q["x"].camsTarget.iloc[0]) is not None, "weatherAvailable": True,
        "rainContext": rain.describe_window(source["weather"], issue, int(start), int(end)),
        "weatherSourceClock": {"asOfEpoch": issue, "fetchedEpoch": int(q["weatherFetched"][0]),
                               "availabilityRule": "fetched_epoch_at_or_before_actual_issue", "maximumAgeSeconds": 7200},
        "weights": {}, "components": {}, "componentsRole": "direct_absolute_session_mean_regression",
        "sourceCaveat": "Experimental, single TTDI sensor; rain may miss the site and new plume regimes can exceed learned errors.",
    }}
    if live_snapshot is not None:
        result["afternoon"].update(
            inputAvailabilityPolicy=LIVE_INPUT_POLICY,
            inputSnapshotReceivedEpoch=int(q["inputSnapshotReceivedEpochs"][0]),
            persistenceAnchorRole="trailing_5_minute_raw_sensor_median_capped_by_current_collected_snapshot",
        )
        result["afternoon"]["weatherSessionModel"]["liveInputAvailabilityPolicy"] = LIVE_INPUT_POLICY
    return {"afternoon": selection.apply_session_prediction_policy(result["afternoon"], "afternoon")}


def record_issue(db_path, result, recorded_epoch):
    """Append only this new live model version; never rewrite archived forecasts."""
    recorded_epoch = int(recorded_epoch)
    records = []
    for name, value in result.items():
        candidate = value.get("candidateForecast") or {}
        if (name != "afternoon" or not value.get("available")
                or candidate.get("modelVersion") != MODEL_VERSION or not candidate.get("available")):
            continue
        # Archive the genuinely computed raw v2 candidate separately from the
        # operational persistence point, preserving old versions and rows.
        raw_point = _number(candidate.get("mean"))
        if raw_point is None:
            continue
        value = {**value, "operationalMean": value.get("mean"),
                 "publishedModelVersion": value.get("modelVersion"),
                 "modelVersion": MODEL_VERSION, "mean": raw_point, "prediction": raw_point,
                 "archiveRole": "raw_candidate_before_operational_qualification"}
        try:
            issue, origin, start, end = (int(value[k]) for k in ("forecastedAtEpoch", "originEpoch", "startEpoch", "endEpoch"))
            m = value["weatherSessionModel"]
            if value.get("inputAvailabilityPolicy") == LIVE_INPUT_POLICY:
                availability_epoch = value["inputSnapshotReceivedEpoch"]
                source_guard = value["featureSourceMaxEpoch"] <= value["sensorWatermarkEpoch"] <= availability_epoch <= issue
            else:
                availability_epoch = value["publicationEpoch"]
                source_guard = (value["featureSourceMaxEpoch"] <= value["sensorWatermarkEpoch"] <= availability_epoch <= issue
                                and 0 <= issue - availability_epoch <= 120)
            valid = ((m.get("candidateComputed") is True or m.get("applied") is True)
                     and m["featureVersion"] == FEATURE_VERSION and
                     m["trainingCutoffEpoch"] == training_cutoff(issue) and m["trainingLatestTargetEndEpoch"] < m["trainingCutoffEpoch"] and
                     m["fittedAtEpoch"] <= issue and 0 <= recorded_epoch - issue <= 120 and recorded_epoch < start and
                     0 <= issue - origin < 900 and end - start == 7200 and
                     _number(value["mean"]) is not None and _number(value["sensorAnchor"]) is not None and
                      selection.candidate_scope(issue, start, end, "afternoon").get("rawCandidatePermitted") is True and
                     source_guard and
                     value["sensorReferenceEpoch"] <= value["sensorWatermarkEpoch"] and
                     0 <= issue - value["sensorReferenceEpoch"] <= 240 and
                     0 <= issue - float(value["weatherFetchedEpoch"]) <= 7200)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            continue
        payload = {**value, "recordedIssuedEpoch": recorded_epoch, "isOriginallyIssued": True}
        records.append((MODEL_VERSION, origin, start, end, recorded_epoch, name,
                        value["mean"], value["sensorAnchor"], json.dumps(payload, separators=(",", ":"), allow_nan=False)))
    if not records:
        return 0
    with closing(sqlite3.connect(str(db_path), timeout=20)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS window_pm_forecast_issues(
            model_version TEXT NOT NULL, origin_epoch INTEGER NOT NULL, start_epoch INTEGER NOT NULL,
            end_epoch INTEGER NOT NULL, issued_epoch INTEGER NOT NULL, window_key TEXT NOT NULL,
            predicted_mean REAL NOT NULL, anchor_pm25 REAL NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(model_version,origin_epoch,start_epoch,end_epoch))""")
        count = 0
        for record in records:
            count += conn.execute("INSERT OR IGNORE INTO window_pm_forecast_issues "
                                  "(model_version,origin_epoch,start_epoch,end_epoch,issued_epoch,window_key,predicted_mean,anchor_pm25,payload) "
                                  "VALUES(?,?,?,?,?,?,?,?,?)", record).rowcount
    return count
