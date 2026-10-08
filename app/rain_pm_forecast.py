"""Experimental rain-aware PM change model with read-only prediction.

Predeclared before outcome inspection: one half-strength Ridge delta model,
alpha 50, seven-day training window, hourly origins and at least 48 outcomes.
Separate exact +90-minute point and +90..210-minute mean responses learn the
size and persistence of an association. Rain is never constrained to lower PM.
The comparator is the same issue's five-minute sensor median, with closed
bucket fallback. Rain forecasts describe a model grid, not a local rain gauge.
The optional record_issue writer stores separate unpromoted diagnostics only;
prediction and study functions never invoke it or change primary forecasts.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import closing
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from forecast_clock import ForecastClock, target_series
from forecast_freshness import raw_arrays, references
from window_pm_predictor import _frame, _features
import rain_weather_features as weather

MODEL_VERSION = "rain_nearterm_ridge_v1"
TZ = "Asia/Kuala_Lumpur"
TRAIN_DAYS = 7
MIN_TRAIN = 48
RIDGE_ALPHA = 50.0
DELTA_SHRINK = 0.5
LOCAL_COLUMNS = ("level", "delta30", "delta60", "mean3Offset",
                 "tempDelta30", "rhDelta30", "sinTarget", "cosTarget",
                 "freshOffset")
RAIN_COMMON = ("priorRainMm", "priorRainProbMax",
               "priorWetSignalHours", "firstWetLeadHours",
               "hoursUntilExpectedWetEnd", "targetTempMinDeltaC",
               "targetWind10DeltaKmh")
RAIN_DURING = ("duringRainMm", "duringRainProbMax", "duringWetSignalHours")
PARAMETERS = {"trainDays": TRAIN_DAYS, "minimumTrainingOrigins": MIN_TRAIN,
              "trainingOriginSpacingMinutes": 60, "ridgeAlpha": RIDGE_ALPHA,
              "deltaShrink": DELTA_SHRINK, "featureScale": "training_median_IQR_clip_4",
              "rainSignConstraint": False, "parameterSearch": False}
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (ValueError, TypeError, OverflowError):
        return np.nan


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _columns(horizon):
    rain = RAIN_COMMON + (RAIN_DURING if horizon == "mean" else ())
    interactions = ("levelPriorRain", "levelPriorProbability")
    if horizon == "mean":
        interactions += ("levelDuringRain", "levelDuringProbability")
    return LOCAL_COLUMNS + rain + interactions


def _fit_delta(x, y, q):
    """Training-only covariate imputation; no response or gap imputation."""
    finite = np.isfinite(x)
    med = np.array([np.median(x[finite[:, j], j]) if finite[:, j].any() else 0.
                    for j in range(x.shape[1])])
    quartiles = np.array([np.quantile(x[finite[:, j], j], [.25, .75])
                          if finite[:, j].any() else [0., 1.]
                          for j in range(x.shape[1])])
    scale = quartiles[:, 1] - quartiles[:, 0]
    scale = np.where(scale > 1e-5, scale, 1.)
    z = np.clip((np.where(finite, x, med) - med) / scale, -4., 4.)
    query = np.clip((np.where(np.isfinite(q), q, med) - med) / scale, -4., 4.)
    z = np.column_stack([np.ones(len(z)), z])
    query = np.r_[1., query]
    penalty = np.eye(z.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return float(query @ beta)


def _feature_rows(frame, rows, runs, lag_seconds, as_of_epoch, horizon,
                  include_live=False):
    origins = frame.index[frame.index.minute == 0]
    if include_live:
        origins = origins.union(frame.index[-1:]).sort_values()
    epochs = (origins.asi8 // 10**9).astype(np.int64)
    issued = epochs + lag_seconds
    ref, ref_epochs, ref_counts = references(rows, epochs, lag_seconds, as_of_epoch)
    f = _features(frame, 90 + lag_seconds / 60. + (0 if horizon == "mean" else -60))
    f = f.reindex(origins).copy()
    closed = f.level.to_numpy(float)
    f["freshOffset"] = np.where(np.isfinite(ref), ref - closed, 0.)
    f["level"] = np.where(np.isfinite(ref), ref, closed)
    descriptors = []
    for i, issue in enumerate(issued):
        start = int(issue + 90 * 60)
        end = int(start + (120 * 60 if horizon == "mean" else 0))
        context = weather.describe_window(runs, int(issue), start, end)
        descriptors.append(context)
        values = context.get("featureValues") or {}
        for col in RAIN_COMMON + RAIN_DURING:
            f.loc[origins[i], col] = _number(values.get(col))
        # No forecast wet interval means onset/end are beyond the modeled
        # horizon. The sentinel is explicit and fixed, not a fabricated event.
        wet_hours = [_number(values.get(k)) for k in
                     ("priorWetSignalHours", "duringWetSignalHours")]
        no_wet_signal = all(np.isfinite(v) for v in wet_hours) and sum(wet_hours) == 0
        for col in ("firstWetLeadHours", "hoursUntilExpectedWetEnd"):
            if not np.isfinite(f.loc[origins[i], col]) and context.get("available") and no_wet_signal:
                f.loc[origins[i], col] = 4.5
    f["levelPriorRain"] = f.level * f.priorRainMm
    f["levelPriorProbability"] = f.level * f.priorRainProbMax / 100.
    f["levelDuringRain"] = f.level * f.duringRainMm
    f["levelDuringProbability"] = f.level * f.duringRainProbMax / 100.
    valid = np.isfinite(closed) & np.isfinite(f.level.to_numpy(float))
    valid &= np.array([bool(c.get("available")) for c in descriptors])
    # Same fresh-reference eligibility on both live and historical origins.
    valid &= np.where(ref_counts > 0, issued - ref_epochs <= 600, True)
    return (origins, f[list(_columns(horizon))].to_numpy(float),
            f.level.to_numpy(float), descriptors, valid, ref_epochs, ref_counts)


def replay(frame, rows, runs, lag_seconds=0, as_of_epoch=None, include_live=False):
    """Walk forward at each historical issue's identical within-bucket lag."""
    if frame.empty:
        return {"arrival": [], "mean": []}
    if not 0 <= lag_seconds < 900:
        raise ValueError("Issue offset must be within its closed 15-minute bucket")
    if as_of_epoch is None:
        as_of_epoch = int(frame.index[-1].timestamp()) + lag_seconds
    frame = frame.loc[frame.index <= pd.Timestamp(as_of_epoch, unit="s", tz="UTC")]
    anchor = int(frame.index[-1].timestamp())
    clock = ForecastClock(anchor + lag_seconds, anchor)
    targets = target_series(frame.pm02, clock)
    output = {}
    with threadpool_limits(limits=1):
        for horizon in ("arrival", "mean"):
            origins, x, base, contexts, valid, fresh_epoch, fresh_count = _feature_rows(
                frame, rows, runs, lag_seconds, as_of_epoch, horizon, include_live)
            epochs = (origins.asi8 // 10**9).astype(np.int64)
            issues = epochs + lag_seconds
            completion = (math.ceil(clock.arrival_lead_seconds / 900) * 900
                          if horizon == "arrival" else clock.outcome_complete_lead_seconds)
            complete = epochs + completion
            actual = targets[horizon].reindex(origins).to_numpy(float)
            delta = actual - base
            columns = _columns(horizon)
            rain_high = x[:, columns.index("priorRainProbMax")] >= 60
            rain_amount = x[:, columns.index("priorRainMm")].copy()
            if horizon == "mean":
                rain_high |= x[:, columns.index("duringRainProbMax")] >= 60
                rain_amount += x[:, columns.index("duringRainMm")]
            wet_fit = rain_high | (rain_amount >= .1)
            hourly_training = np.asarray(origins.minute == 0)
            records = []
            for i, origin in enumerate(origins):
                if not valid[i] or issues[i] > as_of_epoch:
                    continue
                train = valid & hourly_training & np.isfinite(delta)
                train &= (complete < issues[i]) & (issues >= issues[i] - TRAIN_DAYS * 86400)
                if train.sum() < MIN_TRAIN:
                    continue
                change = _fit_delta(x[train], delta[train], x[i])
                issue = int(issues[i])
                context = contexts[i]
                probability = [_number(context.get("featureValues", {}).get(k))
                               for k in ("priorRainProbMax", "duringRainProbMax")]
                max_probability = max((p for p in probability if np.isfinite(p)), default=np.nan)
                rain_amounts = [_number(context.get("featureValues", {}).get(k))
                                for k in ("priorRainMm", "duringRainMm")]
                wet_forecast = bool(max_probability >= 60 or
                                    sum(p for p in rain_amounts if np.isfinite(p)) >= .1)
                actual_value = (float(actual[i]) if np.isfinite(actual[i]) and
                                complete[i] < as_of_epoch else None)
                records.append({
                    "originEpoch": int(epochs[i]), "issuedEpoch": issue,
                    "completeAfterEpoch": int(complete[i]), "day": str(origin.date()),
                    "actual": actual_value, "persistence": float(base[i]),
                    "prediction": max(0., float(base[i] + DELTA_SHRINK * change)),
                    "rawLearnedDelta": change, "rainProbabilityMax": float(max_probability),
                    "highRain": bool(np.isfinite(max_probability) and max_probability >= 60),
                    "wetForecast": wet_forecast,
                    "weatherFetchedEpoch": context.get("fetchedEpoch"),
                    "freshReferenceEpoch": int(fresh_epoch[i]) if fresh_count[i] else None,
                    "freshReferenceCount": int(fresh_count[i]),
                    "featureAnchorEpoch": int(epochs[i]),
                    "rainContext": context,
                    "trainingCount": int(train.sum()),
                    "trainingWetForecastCount": int((train & wet_fit).sum()),
                    "trainingHighRainCount": int((train & rain_high).sum()),
                    "trainingFirstIssueEpoch": int(issues[train].min()),
                    "trainingLastIssueEpoch": int(issues[train].max()),
                    "trainingLastOutcomeCompleteEpoch": int(complete[train].max()),
                })
            output[horizon] = records
    return output


def metric(records):
    if not records:
        return {"count": 0, "days": 0}
    actual = np.array([r["actual"] for r in records])
    base = np.array([r["persistence"] for r in records])
    pred = np.array([r["prediction"] for r in records])
    error, base_error = pred - actual, base - actual
    clearing = (base - actual >= 15) & (base - actual >= .2 * base)
    predicts_clear = (base - pred >= 15) & (base - pred >= .2 * base)
    return {"count": len(records), "days": len({r["day"] for r in records}),
            "mae": float(np.mean(np.abs(error))),
            "persistenceMae": float(np.mean(np.abs(base_error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "persistenceRmse": float(np.sqrt(np.mean(base_error ** 2))),
            "p90AbsoluteError": float(np.quantile(np.abs(error), .9)),
            "persistenceP90AbsoluteError": float(np.quantile(np.abs(base_error), .9)),
            "biasPredictionMinusActual": float(error.mean()),
            "meanActualChange": float(np.mean(actual - base)),
            "meanPredictedChange": float(np.mean(pred - base)),
            "clearingCount": int(clearing.sum()),
            "predictedClearingCount": int(predicts_clear.sum()),
            "trueClearingPredictedCount": int((predicts_clear & clearing).sum()),
            "falseClearingCount": int((predicts_clear & ~clearing).sum()),
            "falseClearingWithNoDropCount": int((predicts_clear & (actual >= base)).sum()),
            "increaseCount": int((actual >= base).sum())}


def evidence(records, cutoff, start_issue=None, end_issue=None):
    scored = [r for r in records if r["actual"] is not None and
              r["completeAfterEpoch"] < cutoff and
              (start_issue is None or r["issuedEpoch"] >= start_issue) and
              (end_issue is None or r["issuedEpoch"] < end_issue)]
    daily = {d: metric([r for r in scored if r["day"] == d])
             for d in sorted({r["day"] for r in scored})}
    high = [r for r in scored if r["highRain"]]
    wet = [r for r in scored if r.get("wetForecast", r["highRain"])]
    clearing = lambda r: r["persistence"] - r["actual"] >= max(15., .2 * r["persistence"])
    nonoverlap, previous_end = [], -1
    for r in sorted(scored, key=lambda r: r["issuedEpoch"]):
        # Separation starts at issue and ends when all response buckets close.
        # The fixed greedy cohort does not condition on error or rain outcome.
        if r["issuedEpoch"] > previous_end:
            nonoverlap.append(r)
            previous_end = r["completeAfterEpoch"]
    result = {"overall": metric(scored), "daily": daily,
              "dayBalancedMae": float(np.mean([m["mae"] for m in daily.values()])) if daily else None,
              "dayBalancedPersistenceMae": float(np.mean([m["persistenceMae"] for m in daily.values()])) if daily else None,
              "highRain": metric(high),
              "wetForecast": metric(wet),
              "highRainClearing": metric([r for r in high if clearing(r)]),
              "highRainNoClearing": metric([r for r in high if not clearing(r)]),
              "highRainIncrease": metric([r for r in high if r["actual"] >= r["persistence"]]),
              "nonoverlapping": metric(nonoverlap),
              "nonoverlappingHighRain": metric([r for r in nonoverlap if r["highRain"]]),
              "nonoverlappingHighRainClearing": metric([r for r in nonoverlap if r["highRain"] and clearing(r)]),
              "nonoverlappingHighRainNoClearing": metric([r for r in nonoverlap if r["highRain"] and not clearing(r)]),
              "nonoverlappingWetForecast": metric([r for r in nonoverlap if r.get("wetForecast", r["highRain"])]),
              "wetPolicyVsFreshPersistence": metric([
                  {**r, "prediction": r["prediction"] if r.get("wetForecast", r["highRain"]) else r["persistence"]}
                  for r in scored]),
              "nonoverlappingIssueEpochs": [r["issuedEpoch"] for r in nonoverlap],
              "incompleteTargetsScored": False,
              "prospectivelyValidated": False,
              "highRainDefinition": "forecast probability maximum >=60% between issue and response end",
              "clearingDefinition": "actual decline >=15 ug/m3 AND >=20% of same-issue persistence",
              "nonoverlapDefinition": "greedy chronological issue after previous selected outcome completion"}
    result["wetPolicyDefinition"] = "candidate on complete rain forecasts with maximum probability >=60% OR prior+during amount >=0.1 mm; comparator fresh persistence otherwise"
    policy = [{**r, "prediction": r["prediction"] if r.get("wetForecast", r["highRain"]) else r["persistence"]}
              for r in scored]
    result["cohortDayBalanced"] = {}
    for name, subset in (("all", scored), ("wetPolicy", policy), ("wetForecast", wet),
                         ("highRain", high), ("highRainClearing", [r for r in high if clearing(r)]),
                         ("highRainNoClearing", [r for r in high if not clearing(r)])):
        days = {r["day"] for r in subset}
        metrics = [metric([r for r in subset if r["day"] == day]) for day in days]
        result["cohortDayBalanced"][name] = {
            "days": len(days),
            "mae": float(np.mean([m["mae"] for m in metrics])) if metrics else None,
            "persistenceMae": float(np.mean([m["persistenceMae"] for m in metrics])) if metrics else None}
    return result


def _evidence_live(records, issue, wet_forecast):
    known = [r for r in records if r["actual"] is not None and
             r["completeAfterEpoch"] < issue and r["issuedEpoch"] >= issue - 7 * 86400 and
             bool(r.get("wetForecast", False)) == wet_forecast]
    full = evidence(known, issue)
    summary = {k: full[k] for k in ("overall", "highRain", "highRainClearing",
                                    "highRainNoClearing", "highRainIncrease",
                                    "nonoverlapping", "cohortDayBalanced")}
    errors = np.array([r["actual"] - r["prediction"] for r in known])
    quantiles = np.quantile(errors, [.1, .9]) if len(errors) >= 12 else None
    cohort = "wet_forecast" if wet_forecast else "dry_forecast"
    summary.update({"calibrated": False, "residualCount": len(errors),
                    "count": len(errors), "distinctDays": len({r["day"] for r in known}),
                    "independentOriginCount": summary["nonoverlapping"]["count"],
                    "residualCohort": cohort,
                    "residualPolicy": f"identical_rain_ridge_{cohort}_issues_same_issue_offset_seven_completed_days",
                    "residualLatestOutcomeCompleteEpoch": max((r["completeAfterEpoch"] for r in known), default=None),
                    "limitation": "Experimental retrospective association; overlapping outcomes and few rainy days limit evidence. Rain does not guarantee PM clearing."})
    return summary, quantiles


def predict_nearterm(db_path, rows, issue_epoch):
    """Read-only exact issue+90 point and issue+90..210 mean candidate seam."""
    started = time.perf_counter()
    unavailable = lambda reason: {"available": False, "modelVersion": MODEL_VERSION,
                                  "experimental": True, "reason": reason}
    number = _number(issue_epoch)
    if not np.isfinite(number):
        return unavailable("invalid_issue_epoch")
    if number != int(number):
        return unavailable("issue_epoch_requires_integer_seconds")
    issue = int(number)
    rows = [dict(r) for r in rows if np.isfinite(_number(dict(r).get("epoch"))) and
            _number(dict(r).get("epoch")) <= issue]
    frame = _frame(rows, issue)
    if frame.empty:
        return unavailable("missing_sensor_data")
    epoch_array, value_array = raw_arrays(rows, issue)
    if not len(epoch_array) or issue - epoch_array[-1] > 600:
        return unavailable("stale_latest_sensor_reading")
    anchor = int(frame.index[-1].timestamp())
    lag = issue - anchor
    if lag < 0 or lag >= 900 or not np.isfinite(frame.pm02.iloc[-1]):
        return unavailable("stale_or_missing_closed_sensor_bucket")
    runs = weather.load_runs(db_path, issue)
    live_context = weather.describe_window(runs, issue, issue + 5400, issue + 12600)
    if not live_context.get("available"):
        return {**unavailable("missing_or_stale_issued_weather"), "rainContext": live_context}
    digest = hashlib.blake2b(pd.util.hash_pandas_object(frame, index=True).values.tobytes(), digest_size=16)
    digest.update(epoch_array.tobytes())
    digest.update(value_array.tobytes())
    # Reader content revision must include past same-count payload corrections.
    revision = getattr(runs, "revision", None)
    if revision is None:
        revision = hashlib.blake2b(repr(runs).encode("utf-8"), digest_size=16).hexdigest()
    key = (str(db_path), MODEL_VERSION, issue, digest.hexdigest(), str(revision))
    with _LOCK:
        if key in _CACHE:
            return copy.deepcopy(_CACHE[key])
    records = replay(frame, rows, runs, lag, issue, include_live=True)
    clock = ForecastClock(issue, anchor)
    output = {"available": True, "modelVersion": MODEL_VERSION, "experimental": True,
              "prospectivelyValidated": False, "parameters": PARAMETERS,
              "forecastClock": clock.metadata(), "featureAnchorEpoch": anchor,
              "weatherFetchedEpoch": live_context.get("fetchedEpoch"), "rainContext": live_context}
    for horizon, target_key in (("arrival", "arrival"), ("mean", "trail")):
        live = next((r for r in reversed(records[horizon]) if r["issuedEpoch"] == issue), None)
        if live is None:
            output[target_key] = unavailable("insufficient_completed_weather_training")
            continue
        validation, quantiles = _evidence_live(records[horizon], issue, live["wetForecast"])
        point = live["prediction"]
        closed_reference = float(frame.pm02.iloc[-1])
        fresh_applied = live["freshReferenceCount"] > 0
        freshness = {"applied": fresh_applied, "version": "five_minute_sensor_refresh_v1",
                     "closedReferencePm25UgM3": closed_reference,
                     "amountUgM3": live["persistence"] - closed_reference,
                     "referenceEpoch": live["freshReferenceEpoch"] or anchor,
                     "referenceCount": live["freshReferenceCount"],
                     "windowSeconds": 300, "featureAnchorEpoch": anchor,
                     "policy": "same_issue_five_minute_reference_rain_delta_fit",
                     "prospectivelyValidated": False}
        output[target_key] = {
            "available": True, "modelVersion": MODEL_VERSION, "experimental": True,
            "point": point, "baselinePoint": live["persistence"],
            "wetForecast": live["wetForecast"],
            "rawRangeLow": max(0., point + float(quantiles[0])) if quantiles is not None else None,
            "rawRangeHigh": max(0., point + float(quantiles[1])) if quantiles is not None else None,
            "rangeMeaning": "q10_q90_past_same_wet_dry_cohort_forecast_residuals_not_rain_probability_or_trail_min_max",
            "rangePolicyVersion": "same_wet_dry_cohort_residual_v2",
            "forecastClock": clock.metadata(), "featureAnchorEpoch": anchor,
            "modelFeatureAnchor": closed_reference,
            "persistenceAnchorEpoch": live["freshReferenceEpoch"] or anchor,
            "persistenceAnchorRole": ("trailing_5_minute_raw_sensor_median" if fresh_applied
                                      else "latest_closed_15_minute_bucket_median"),
            "freshnessAdjustment": freshness,
            "sourceSnapshots": {"weatherFetchedEpoch": live["weatherFetchedEpoch"],
                                "sensorReferenceEpoch": live["freshReferenceEpoch"]},
            "freshReferenceCount": live["freshReferenceCount"],
            "trainingCount": live["trainingCount"],
            "trainingWetForecastCount": live["trainingWetForecastCount"],
            "trainingHighRainCount": live["trainingHighRainCount"],
            "trainingLastOutcomeCompleteEpoch": live["trainingLastOutcomeCompleteEpoch"],
            "rainContext": live["rainContext"], "validation": validation,
            "rawLearnedDelta": live["rawLearnedDelta"], "scored": False,
        }
    output["mean"] = copy.deepcopy(output["trail"])
    output["available"] = any(output[k].get("available") for k in ("arrival", "trail"))
    output["computeSeconds"] = time.perf_counter() - started
    output["appliedToPrimary"] = False
    output = _clean(output)
    with _LOCK:
        _CACHE[key] = copy.deepcopy(output)
        while len(_CACHE) > 6:
            _CACHE.popitem(last=False)
    return output


def record_issue(db_path, result, recorded_epoch):
    """Archive optional diagnostics separately; immutable per issue/horizon.

    This explicit writer is never called by prediction or study code. It owns
    only rain_pm_forecast_issues and cannot replace a primary forecast record.
    Actual issue and recording times are both retained so retrospective replay
    cannot masquerade as a forecast recorded before its response began.
    """
    recorded = _number(recorded_epoch)
    if not np.isfinite(recorded):
        raise ValueError("recorded_epoch must be finite")
    if not result.get("available"):
        return {"recordedCount": 0, "reason": "candidate_unavailable"}
    inserts = []
    for horizon in ("arrival", "trail"):
        item = result.get(horizon) or {}
        if not item.get("available"):
            continue
        clock = item.get("forecastClock") or {}
        issue = _number(clock.get("forecastIssuedEpoch"))
        start = _number(clock.get("arrivalTargetEpoch"))
        end = start if horizon == "arrival" else _number(clock.get("windowEndEpoch"))
        anchor = _number(clock.get("featureAnchorEpoch"))
        fetched = _number((item.get("sourceSnapshots") or {}).get("weatherFetchedEpoch"))
        point, baseline = _number(item.get("point")), _number(item.get("baselinePoint"))
        if not all(np.isfinite(v) for v in (issue, start, end, anchor, fetched, point, baseline)):
            raise ValueError("Cannot archive incomplete candidate provenance")
        if (recorded < issue or anchor > issue or fetched > issue or issue - fetched > 7200
                or start != issue + 5400 or end != (start if horizon == "arrival" else issue + 12600)):
            raise ValueError("Cannot archive mismatched candidate issue/target provenance")
        version = item.get("modelVersion")
        if version != result.get("modelVersion") or version != MODEL_VERSION:
            raise ValueError("Candidate model versions must agree")
        payload = {**copy.deepcopy(item), "appliedToPrimary": False,
                   "recordedEpoch": recorded, "recordedBeforeTarget": recorded < start,
                   "diagnosticRole": "experimental_unpromoted_rain_candidate",
                   "computeSeconds": result.get("computeSeconds")}
        issue_id = f"{version}:{int(issue)}:{horizon}"
        inserts.append((issue_id, version, int(issue), horizon, int(start), int(end),
                        int(anchor), fetched, item.get("persistenceAnchorEpoch"), point,
                        baseline, int(bool(item.get("wetForecast"))), recorded,
                        json.dumps(_clean(payload), sort_keys=True, allow_nan=False)))
    with closing(sqlite3.connect(str(db_path), timeout=20)) as connection:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS rain_pm_forecast_issues (
                issue_id TEXT PRIMARY KEY, model_version TEXT NOT NULL,
                issued_epoch INTEGER NOT NULL, horizon TEXT NOT NULL,
                start_epoch INTEGER NOT NULL, end_epoch INTEGER NOT NULL,
                feature_anchor_epoch INTEGER NOT NULL, weather_fetched_epoch REAL NOT NULL,
                sensor_reference_epoch REAL, point REAL NOT NULL, baseline_point REAL NOT NULL,
                wet_forecast INTEGER NOT NULL, recorded_epoch REAL NOT NULL,
                payload TEXT NOT NULL, applied_to_primary INTEGER NOT NULL DEFAULT 0
                    CHECK(applied_to_primary=0),
                UNIQUE(model_version,issued_epoch,horizon))""")
            before = connection.total_changes
            connection.executemany("""INSERT OR IGNORE INTO rain_pm_forecast_issues
                (issue_id,model_version,issued_epoch,horizon,start_epoch,end_epoch,
                 feature_anchor_epoch,weather_fetched_epoch,sensor_reference_epoch,
                 point,baseline_point,wet_forecast,recorded_epoch,payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", inserts)
            count = connection.total_changes - before
    return {"recordedCount": count, "appliedToPrimary": False,
            "table": "rain_pm_forecast_issues"}


def run_study(db_path):
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(r) for r in connection.execute("SELECT epoch,pm02,atmp,rhum FROM readings ORDER BY epoch")]
    asof = int(max(r["epoch"] for r in rows))
    frame = _frame(rows, asof)
    runs = weather.load_runs(db_path, asof)
    holdout = int(pd.Timestamp("2026-09-09", tz=TZ).timestamp())
    sep10 = int(pd.Timestamp("2026-09-10", tz=TZ).timestamp())
    result = {"modelVersion": MODEL_VERSION, "parameters": PARAMETERS,
              "sourcePath": str(db_path), "sourceAsOfEpoch": asof,
              "sensorRows": len(rows), "featureColumns": {h: _columns(h) for h in ("arrival", "mean")},
              "selection": "One candidate fixed before metrics; Sep9 is a separate completed-outcome evaluation, not parameter selection",
              "holdoutEpoch": holdout, "offsets": {}}
    for lag in (0, 300, asof % 900):
        if str(lag) in result["offsets"]:
            continue
        start = time.perf_counter()
        records = replay(frame, rows, runs, lag, asof)
        block = {}
        for horizon, items in records.items():
            block[horizon] = {"preSep9": evidence(items, holdout, end_issue=holdout),
                              "completedSep9": evidence(items, min(asof, sep10),
                                                        start_issue=holdout, end_issue=sep10),
                              "allCompleted": evidence(items, asof), "records": items}
        block["computeSeconds"] = time.perf_counter() - start
        result["offsets"][str(lag)] = block
    examples = ["2026-09-09 07:00", "2026-09-09 12:00", "2026-09-09 12:30",
                "2026-09-09 13:00", "2026-09-09 13:30", "2026-09-09 14:00"]
    result["sep9Examples"] = {}
    for stamp in examples:
        epoch = int(pd.Timestamp(stamp, tz=TZ).timestamp())
        result["sep9Examples"][stamp] = predict_nearterm(db_path, rows, epoch)
    result["snapshotLive"] = predict_nearterm(db_path, rows, asof)
    return _clean(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = run_study(args.db_path)
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for offset, block in report["offsets"].items():
        for horizon in ("arrival", "mean"):
            print(offset, horizon, "preSep9", json.dumps(block[horizon]["preSep9"]["overall"]))
            print(offset, horizon, "completedSep9", json.dumps(block[horizon]["completedSep9"]["overall"]))
    print("Saved", args.report)
