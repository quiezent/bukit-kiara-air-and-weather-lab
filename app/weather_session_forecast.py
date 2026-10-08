"""Publish weather-responsive two-hour session forecasts and frozen evidence.

This module is isolated from the +90 minute and +90-to-210 minute models.
Replay scores are never substituted for forecasts actually issued in the past.
"""
from __future__ import annotations

from contextlib import closing
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np
import pandas as pd

import window_pm_predictor as legacy
import afternoon_direction_features as causal
import rain_weather_features as rain
import weather_session_features as weather_features
import observation_provenance
import session_model_output as direct
from forecast_freshness import raw_arrays, VERSION as FRESHNESS_VERSION

TRAINING_MODEL_VERSION = "weather_session_change_v1"
MODEL_VERSION = "morning_hgb_basic31_model_direct_v3"
NUMERICAL_POLICY = "original_hgb_basic31_daily_14d_direct_mean_receipt_inputs_no_selector_v3"
MIN_MATCHED_DAYS = 14
ISSUE_LAG_TOLERANCE_SECONDS = 60
_FIT_CACHE = OrderedDict()
_DAILY_CACHE = OrderedDict()
_REFRESH_STATUS = {}
_LOCK = threading.RLock()


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _read_connection(db_path):
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=20)


def training_cutoff(issue_epoch):
    """The same local-midnight, completed-label cutoff used in blocked replay."""
    return int(pd.Timestamp(issue_epoch, unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur").normalize().timestamp())


def _valid_training_clock(weather_model, issue_epoch):
    """Audit the declared daily freeze, including genuinely issued records."""
    try:
        cutoff = int(weather_model["trainingCutoffEpoch"])
        latest = int(weather_model["trainingLatestTargetEndEpoch"])
        return cutoff == training_cutoff(issue_epoch) and latest < cutoff
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _trained_model(db_path, rows, cutoff):
    import weather_session_model as learner

    past = [dict(row) for row in rows if cutoff-16*86400 <= float(row["epoch"]) <= cutoff]
    digest = hashlib.blake2b(digest_size=16)
    digest.update(json.dumps(past, sort_keys=True, separators=(",", ":"), default=str).encode())
    revisions = tuple(legacy._raw_runs(db_path, table, cutoff)[1]
                      for table in ("air_quality_forecast_runs", "weather_forecast_runs"))
    key = (str(db_path), TRAINING_MODEL_VERSION, cutoff, digest.hexdigest(), revisions)
    with _LOCK:
        if key in _FIT_CACHE:
            _FIT_CACHE.move_to_end(key)
            return _FIT_CACHE[key]
    # Dataset reconstruction and fitting must not hold the inference cache lock.
    snapshot = learner.build_dataset(db_path, past, cutoff)
    valid = np.asarray(snapshot["valid"], dtype=bool)
    labels = np.asarray(snapshot["labels"], dtype=float)
    complete = np.asarray(snapshot["complete"], dtype=np.int64)
    issues = np.asarray(snapshot["issues"], dtype=np.int64)
    mask = (valid & np.isfinite(labels) & (complete < cutoff)
            & (issues < cutoff) & (issues >= cutoff-14*86400))
    target_days = pd.to_datetime(complete[mask], unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur").date
    if mask.sum() < 120 or len(set(target_days)) < 10:
        return None
    estimator = learner.fit_model(snapshot["x"].loc[mask], labels[mask])
    if estimator is None:
        return None
    fitted = {
        "estimator": estimator, "trainingCount": int(mask.sum()),
        "trainingDistinctTargetDays": len(set(target_days)),
        "trainingLatestTargetEndEpoch": int(complete[mask].max()),
        "trainingCutoffEpoch": cutoff, "fittedAtEpoch": time.time(),
        "trainingSensorInputSha256": digest.hexdigest(),
        "trainingExternalInputRevisions": list(revisions),
    }
    with _LOCK:
        if key in _FIT_CACHE:
            return _FIT_CACHE[key]
        _FIT_CACHE[key] = fitted
        while len(_FIT_CACHE) > 2:
            _FIT_CACHE.popitem(last=False)
        return fitted


def _daily_key(db_path, cutoff):
    return str(Path(db_path).resolve()).casefold(), TRAINING_MODEL_VERSION, int(cutoff)


def _cached_model(db_path, issue):
    """Constant-time inference lookup; no reads, reconstruction or fitting."""
    key = _daily_key(db_path, training_cutoff(issue))
    with _LOCK:
        fitted = _DAILY_CACHE.get(key)
        if (fitted is None or not _valid_training_clock(fitted, issue)
                or fitted.get("fittedAtEpoch", float("inf")) > issue):
            return None
        return fitted


def refresh_model(db_path, issue_epoch, *, rows=None):
    """Background-worker seam: prepare one immutable daily inference artifact.

    The dashboard calls this from its diagnostics worker. Prediction never
    starts this work. A missing/new-day artifact returns unavailable while a
    refresh is running; no old-day model is relabelled as today's model.
    """
    issue = int(issue_epoch)
    cutoff = training_cutoff(issue)
    key = _daily_key(db_path, cutoff)
    with _LOCK:
        if key in _DAILY_CACHE:
            return dict(_REFRESH_STATUS.get(key, {"state": "ready", "trainingCutoffEpoch": cutoff}))
        if _REFRESH_STATUS.get(key, {}).get("state") == "fitting":
            return dict(_REFRESH_STATUS[key])
        _REFRESH_STATUS[key] = {"state": "fitting", "trainingCutoffEpoch": cutoff,
                                "dailyCacheIdentity": list(key)}
    try:
        if rows is None:
            with closing(_read_connection(db_path)) as conn:
                conn.execute("PRAGMA query_only=ON")
                conn.execute("BEGIN")
                rows = observation_provenance.compatible_sensor_rows(
                    conn, cutoff, start_epoch=cutoff-16*86400, end_epoch=cutoff)
        fitted = _trained_model(db_path, rows, cutoff)
        if fitted is None:
            raise ValueError("insufficient_complete_weather_training_history")
        if not _valid_training_clock(fitted, issue):
            raise ValueError("invalid_daily_training_clock")
        fitted = {**fitted, "fittedAtEpoch": fitted.get("fittedAtEpoch", time.time()),
                  "dailyCacheIdentity": list(key), "backgroundPrepared": True}
        with _LOCK:
            _DAILY_CACHE[key] = fitted
            while len(_DAILY_CACHE) > 2:
                _DAILY_CACHE.popitem(last=False)
            _REFRESH_STATUS[key] = {"state": "ready", "trainingCutoffEpoch": cutoff,
                                    "dailyCacheIdentity": list(key), "fittedAtEpoch": fitted["fittedAtEpoch"]}
    except Exception as error:
        with _LOCK:
            _REFRESH_STATUS[key] = {"state": "unavailable", "trainingCutoffEpoch": cutoff,
                                    "dailyCacheIdentity": list(key),
                                    "reason": f"{type(error).__name__}: {error}"}
    with _LOCK:
        return dict(_REFRESH_STATUS[key])


def _cached_only_unavailable(db_path, rows, windows, issue, reason):
    key = _daily_key(db_path, training_cutoff(issue))
    with _LOCK:
        status = dict(_REFRESH_STATUS.get(key, {"state": "background_cache_pending",
                                                "trainingCutoffEpoch": key[-1]}))
    result = {}
    for name, window in windows.items():
        result[name] = direct.unavailable_model_output(
            issue, window.get("startEpoch"), window.get("endEpoch"), MODEL_VERSION, reason, name)
        result[name]["weatherSessionModel"] = {"applied": False, "candidateComputed": False,
                                               "modelVersion": MODEL_VERSION, "reason": reason,
                                               "cachedOnly": True, "fitStatus": status}
    return result


def predict_windows(db_path, rows, windows, issue_epoch, *, allow_fit=True):
    """Return the fixed Morning HGB output; missing inputs remain unavailable.

    Fit once per local day, then refresh the query with the latest closed sensor
    features, five-minute reference and actual issued weather vintage. Never
    read a stored research pickle or outcome report as a live feature.
    """
    started = time.perf_counter()
    issue = int(issue_epoch)
    windows = {name: window for name, window in windows.items() if name == "morning"}
    rows = [dict(row) for row in rows if _number(dict(row).get("epoch")) is not None
            and float(dict(row)["epoch"]) <= issue]
    fitted = None
    if not allow_fit:
        fitted = _cached_model(db_path, issue)
        if fitted is None:
            return _cached_only_unavailable(db_path, rows, windows, issue,
                                          "current_daily_weather_background_artifact_not_ready")
    import weather_session_model as learner
    frame = causal.strict_frame(rows, issue)
    epochs, _ = raw_arrays(rows, issue)
    reason = None
    if frame.empty or not len(epochs) or issue-epochs[-1] > 600:
        reason = "stale_or_missing_sensor_data"
    elif not np.isfinite(frame.pm02.iloc[-1]) or not 0 <= issue-int(frame.index[-1].timestamp()) < 900:
        reason = "stale_or_missing_closed_sensor_bucket"
    if allow_fit:
        fitted = _trained_model(db_path, rows, training_cutoff(issue)) if reason is None else None
    if fitted is None and reason is None:
        reason = "insufficient_complete_weather_training_history"
    archive = rain.load_runs(db_path, issue) if reason is None else None
    snapshot = {
        "rows": rows, "frame": frame, "cutoff": issue,
        "weather": archive,
        "cams": legacy._read_runs(db_path, "air_quality_forecast_runs", issue) if reason is None else [],
    }
    result, failed = {}, {}
    for name, window in windows.items():
        failure = reason
        start, end = (_number(window.get(field)) for field in ("startEpoch", "endEpoch"))
        if failure is None:
            origin = int(frame.index[-1].timestamp())
            failure = direct.target_shape_reason(issue, start, end)
        if failure is None:
            start, end = int(start), int(end)
            context = weather_features.describe_window(archive, issue, start, end)
            if not context["available"]:
                failure = "missing_stale_or_incomplete_issued_weather"
        if failure is None:
            query = direct.query_features(snapshot, issue, start, end)
            anchor = _number(query["fresh"]) if query is not None else None
            if query is None or not query.attrs.get("valid", False) or anchor is None:
                failure = "missing_five_minute_sensor_reference"
        if failure is not None:
            failed[name] = failure
            continue
        mean = _number(learner.predict_model(fitted["estimator"], query))
        if mean is None:
            failed[name] = "nonfinite_weather_model_prediction"
            continue
        try:
            evidence, errors = issued_evidence(db_path, frame, name, start, end, origin, issue, learner.FEATURE_VERSION)
            interval = np.quantile(errors, [.1, .9]) if evidence["supportSufficient"] else None
        except Exception as error:
            # Scoring is descriptive only. It must never erase an already
            # computed fixed-model output when an archive/query is unavailable.
            evidence = {"available": False, "count": 0, "distinctDays": 0,
                        "supportSufficient": False, "skillGatePassed": False,
                        "performanceOnly": True, "appliedToNumber": False,
                        "reason": f"{type(error).__name__}: {error}"}
            interval = None
        lo, hi = ([round(max(0., mean+value), 1) for value in interval]
                  if interval is not None else [None, None])
        selected_weather = rain.latest_run(archive, issue)
        rain_context = rain.describe_window(archive, issue, start, end)
        values = context["featureValues"]
        fresh_stamps = epochs[(epochs > issue-300) & (epochs <= issue)]
        weather_model = {
            "applied": True, "experimental": True,
            "featureVersion": learner.FEATURE_VERSION,
            "weatherDiagnosticFeatureVersion": weather_features.VERSION,
            "numericalPolicy": NUMERICAL_POLICY,
            "trainingNumericalPolicy": learner.NUMERICAL_POLICY,
            "trainingModelVersion": TRAINING_MODEL_VERSION,
            "directQueryVersion": direct.QUERY_VERSION,
            "directQueryAdapterSha256": hashlib.sha256(Path(direct.__file__).read_bytes()).hexdigest(),
            "domainAppliedToNumber": False, "statisticalSelectorApplied": False,
            "learner": learner.LEARNER_NAME,
            "featureColumns": list(learner.FEATURES),
            "parameters": learner.PARAMETERS,
            **{key: value for key, value in fitted.items() if key != "estimator"},
            "trainingPolicy": "daily_refit_only_targets_end_strictly_before_local_midnight_of_issue_day",
            "weatherAsOfPolicy": "latest_validated_fetch_at_or_before_actual_issue_maximum_age_two_hours",
            "weatherFeatures": values,
            "weatherFeaturesRole": "includes_diagnostic_only_channels_not_all_used_in_selected_model",
            "usedWeatherFeatureValues": {field: _number(query[field]) for field in learner.FEATURES
                                         if field in causal.WEATHER or field.startswith("cams")},
            "weatherMetadata": context["metadata"],
            "prospectivelyValidated": False,
            "probabilityRole": "forecast_rain_probability_is_not_PM_washout_probability",
        }
        result[name] = {
            "available": True, "modelVersion": MODEL_VERSION,
            "source": "experimental_weather_session_change_model",
            "pointRole": "experimental_model_output",
            "forecastState": "model_output",
            "mean": mean, "prediction": mean, "numericalPolicyIdentifier": NUMERICAL_POLICY,
            "method": "Learned session change · rain timing and weather · exact 2-hour session",
            "sensorAnchor": anchor, "baselinePoint": anchor,
            "closedSensorAnchor": float(frame.pm02.iloc[-1]),
            "sensorReferenceEpoch": int(fresh_stamps[-1]), "sensorReferenceCount": len(fresh_stamps),
            "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median",
            "originEpoch": origin, "forecastedAtEpoch": issue,
            "startEpoch": start, "endEpoch": end,
            "leadHours": round((start-origin)/3600, 2),
            "remainingLeadHours": round((start-issue)/3600, 2),
            "target": "mean_of_eight_closed_15_minute_sensor_medians",
            "rawRangeLow": lo, "rawRangeHigh": hi,
            "rangeRole": "as_issued_same_clock_q10_q90_session_mean_error_span" if interval is not None else "unavailable_insufficient_matched_as_issued_outcomes",
            "uncertaintyMethod": "as_issued_same_session_target_and_issue_clock_residuals",
            "calibrated": False, "experimental": True, "confidence": "low",
            "usedForDecision": False, "usedForComparison": False,
            "modelSelection": {},
            "candidateForecast": {
                "modelVersion": MODEL_VERSION, "numericalPolicyIdentifier": NUMERICAL_POLICY,
                "mean": mean, "prediction": mean,
                "forecastedAtEpoch": issue, "startEpoch": start, "endEpoch": end,
                "rawRangeLow": lo, "rawRangeHigh": hi,
            },
            "freshnessAdjustment": {
                "applied": False, "appliedToSelectedPoint": False,
                "version": FRESHNESS_VERSION, "referenceUsedAsModelInput": True,
                "referenceEpoch": int(fresh_stamps[-1]), "referenceCount": len(fresh_stamps),
            },
            "weatherSessionModel": weather_model,
            "modelEvidence": evidence,
            "performanceEvidence": {**evidence, "appliedToNumber": False},
            "trainingCount": fitted["trainingCount"],
            "weatherFetchedEpoch": selected_weather.fetched_epoch,
            "camsFetchedEpoch": query.attrs["audit"].get("camsFetchedEpoch"),
            "camsAvailable": _number(query.get("camsTarget")) is not None,
            "weatherAvailable": True, "rainContext": rain_context,
            "weatherSourceClock": {"asOfEpoch": issue, "fetchedEpoch": selected_weather.fetched_epoch,
                                   "availabilityRule": "fetched_epoch_at_or_before_actual_issue", "maximumAgeSeconds": 7200},
            "weights": {}, "components": {},
            "componentsRole": "direct_learned_delta_from_recent_sensor_reference_no_legacy_expert_mixture",
            "sourceCaveat": "Single TTDI sensor; forecast rain may miss the site, and new plumes can exceed learned changes.",
            "computeSeconds": round(time.perf_counter()-started, 3),
        }
        result[name] = direct.publish_direct_output(result[name], name)
    if failed:
        fallback_windows = {name: windows[name] for name in failed}
        for name, failure in failed.items():
            unavailable = _cached_only_unavailable(db_path, rows, {name: fallback_windows[name]}, issue, failure)[name]
            result[name] = {**unavailable,
                            "weatherSessionModel": {"applied": False, "reason": failure,
                                                    "modelVersion": MODEL_VERSION}}
    return result


def issued_evidence(db_path, frame, name, start, end, origin, issue, feature_version):
    """Score new-model frozen issues at the same lead, issue and target clock.

    Keep one pair per target date. Neither repeated deliveries nor forecasts
    from previous model versions increase the independent sample count.
    """
    try:
        with closing(_read_connection(db_path)) as conn:
            archived = conn.execute(
                "SELECT origin_epoch,start_epoch,end_epoch,issued_epoch,"
                "predicted_mean,anchor_pm25,payload FROM window_pm_forecast_issues "
                "WHERE model_version=? AND window_key=? AND end_epoch<? "
                "ORDER BY issued_epoch",
                (MODEL_VERSION, name, issue),
            ).fetchall()
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error):
            raise
        archived = []
    pairs = {}
    excluded = missing = 0
    lag = issue-origin
    for old_origin, old_start, old_end, recorded, prediction, anchor, raw in archived:
        if (old_start-old_origin != start-origin or old_end-old_start != end-start
                or (old_start+28800) % 86400 != (start+28800) % 86400
                or (old_origin+28800) % 86400 != (origin+28800) % 86400):
            continue
        try:
            saved = json.loads(raw)
            actual_issue = int(saved["forecastedAtEpoch"])
            weather_model = saved["weatherSessionModel"]
            numbers = [prediction, anchor, saved["mean"], saved["sensorAnchor"]]
            valid = (
                all(_number(value) is not None for value in numbers)
                and saved["modelVersion"] == MODEL_VERSION
                and weather_model["featureVersion"] == feature_version
                and weather_model["applied"] is True
                and int(saved["originEpoch"]) == old_origin
                and int(saved["startEpoch"]) == old_start
                and int(saved["endEpoch"]) == old_end
                and abs(saved["mean"]-prediction) < 1e-6
                and abs(saved["sensorAnchor"]-anchor) < 1e-6
                and 0 <= actual_issue-old_origin < 900
                and abs(actual_issue-old_origin-lag) <= ISSUE_LAG_TOLERANCE_SECONDS
                and actual_issue <= recorded < old_start
                and recorded-actual_issue <= 120
                and _valid_training_clock(weather_model, actual_issue)
                and 0 <= actual_issue-float(saved["weatherFetchedEpoch"]) <= 7200
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            valid = False
        if not valid:
            excluded += 1
            continue
        actual = legacy._session_actual(frame, old_start)
        if actual is None:
            missing += 1
            continue
        date = str(pd.Timestamp(old_start, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur").date())
        pairs.setdefault(date, (old_end, float(prediction), float(anchor), actual))
    rows = list(pairs.values())
    errors = np.array([actual-point for _, point, _, actual in rows], dtype=float)
    baseline = np.array([actual-anchor for _, _, anchor, actual in rows], dtype=float)
    count = len(rows)
    mae = float(np.abs(errors).mean()) if count else None
    base_mae = float(np.abs(baseline).mean()) if count else None
    p90 = float(np.quantile(np.abs(errors), .9)) if count else None
    base_p90 = float(np.quantile(np.abs(baseline), .9)) if count else None
    support = count >= MIN_MATCHED_DAYS
    skill = bool(support and mae <= .9*base_mae and base_mae-mae >= 2 and p90 <= base_p90)
    recent = [row for row in rows if row[0] >= issue-3*86400]
    return {
        "available": bool(count), "count": count, "distinctDays": count,
        "matchedIssuedCount": count, "missingCompletedTargetCount": missing,
        "excludedPolicyOrClockCount": excluded,
        "mae": round(mae, 2) if count else None,
        "persistenceMae": round(base_mae, 2) if count else None,
        "p90AbsoluteError": round(p90, 2) if count else None,
        "persistenceP90AbsoluteError": round(base_p90, 2) if count else None,
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 2) if count else None,
        "meanErrorPredictionMinusActual": round(float(-errors.mean()), 2) if count else None,
        "validationMode": "as_issued_same_session_target_lead_and_local_issue_clock",
        "evidenceRole": "frozen_weather_session_model_versus_paired_same_issue_persistence",
        "candidateModelVersion": MODEL_VERSION,
        "issueLagToleranceSeconds": ISSUE_LAG_TOLERANCE_SECONDS,
        "minimumForRange": MIN_MATCHED_DAYS, "supportSufficient": support,
        "skillGatePassed": skill, "calibrated": False, "prospectivelyValidated": False,
        "independentOrigins": True, "asIssuedBeforeOutcome": True,
        "incompleteTargetsScored": False,
        "recentCompleted72Hours": {
            "count": len(recent),
            "mae": round(float(np.mean([abs(actual-point) for _, point, _, actual in recent])), 2) if recent else None,
            "persistenceMae": round(float(np.mean([abs(actual-anchor) for _, _, anchor, actual in recent])), 2) if recent else None,
        },
        "limitation": "One TTDI sensor; matched issued days are limited and sudden local plumes remain uncertain.",
    }, errors


def record_issue(db_path, result, recorded_epoch):
    """Append genuinely live new-version forecasts without rewriting old issues."""
    with closing(sqlite3.connect(str(db_path), timeout=20)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS window_pm_forecast_issues(
            model_version TEXT NOT NULL, origin_epoch INTEGER NOT NULL,
            start_epoch INTEGER NOT NULL, end_epoch INTEGER NOT NULL,
            issued_epoch INTEGER NOT NULL, window_key TEXT NOT NULL,
            predicted_mean REAL NOT NULL, anchor_pm25 REAL NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY(model_version,origin_epoch,start_epoch,end_epoch))""")
        count = 0
        for name, value in result.items():
            if name not in ("morning", "afternoon") or not value.get("available"):
                continue
            if value.get("modelVersion") != MODEL_VERSION:
                continue
            try:
                issue = int(value["forecastedAtEpoch"])
                origin, start, end = (int(value[field]) for field in ("originEpoch", "startEpoch", "endEpoch"))
                weather_model = value["weatherSessionModel"]
                valid = (0 <= recorded_epoch-issue <= 120 and recorded_epoch < start
                         and 0 <= issue-origin < 900 and end-start == 7200
                         and _number(value["mean"]) is not None
                         and _number(value["sensorAnchor"]) is not None
                         and weather_model["applied"] is True
                         and _valid_training_clock(weather_model, issue)
                         and 0 <= issue-float(value["weatherFetchedEpoch"]) <= 7200)
            except (ValueError, TypeError, KeyError, OverflowError):
                valid = False
            if not valid:
                continue
            payload = copy.deepcopy(value)
            payload["recordedIssuedEpoch"] = int(recorded_epoch)
            cursor = conn.execute(
                "INSERT OR IGNORE INTO window_pm_forecast_issues "
                "(model_version,origin_epoch,start_epoch,end_epoch,issued_epoch,window_key,"
                "predicted_mean,anchor_pm25,payload) VALUES(?,?,?,?,?,?,?,?,?)",
                (MODEL_VERSION, origin, start, end, int(recorded_epoch), name,
                 value["mean"], value["sensorAnchor"], json.dumps(payload, separators=(",", ":"), allow_nan=False)),
            )
            count += cursor.rowcount
        return count
