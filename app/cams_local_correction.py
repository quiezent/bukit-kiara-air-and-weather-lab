"""Experimental TTDI = issued regional CAMS + a learned, evolving local bias.

An exact-session prediction seam, not an emissions or air-parcel model. The
regional contribution already includes CAMS atmospheric transport. Every
historical feature uses the run available at that origin, and the training
target is local truth minus that SAME run's future regional concentration.
No telemetry collection, networking, coaching rules or additional fire penalty.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
import copy
import hashlib
import json
import math
import sqlite3
import threading
import time

import numpy as np
import pandas as pd

import window_pm_predictor as support

MODEL_VERSION = "cams_local_residual_change_trial_v2_coverage"
MODEL_VARIANT = "residual_ridge"
RESIDUAL_COMPONENTS = ("constant_residual", "residual_relaxation", "residual_ridge")
COLS = ("currentResidual", "camsDelta", "camsLevel", "level", "delta30", "delta60",
        "delta120", "mean3Offset", "mean24Offset", "sd60", "tempDelta30", "rhDelta30",
        "sinTarget", "cosTarget")
TRAIN_DAYS = 7
MIN_TRAIN = 48
MIN_BIAS = 12
MAX_SOURCE_AGE_SECONDS = 7200
MAX_CACHE_SIZE = 6
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def _regional_features(origins, lead_minutes, runs):
    """Match the closed sensor median to its centre, not a future model hour."""
    epochs = np.array([r[0] for r in runs], dtype=float)
    records = []
    for origin in origins:
        epoch = int(origin.timestamp())
        n = int(np.searchsorted(epochs, epoch, side="right")) - 1
        usable = n >= 0 and epoch - epochs[n] <= MAX_SOURCE_AGE_SECONDS
        run = runs[n][1] if usable else None
        centres = epoch + lead_minutes * 60 + np.arange(8) * 900 + 450
        pm = support._sample(run, np.r_[epoch - 450, centres], "pm2_5", max_gap_seconds=10800)
        complete = bool(np.isfinite(pm).all() and np.all(pm >= 0))
        records.append({"regionalCurrent": float(pm[0]) if complete else np.nan,
                        "regionalMean": float(np.mean(pm[1:])) if complete else np.nan,
                        "camsFetchedEpoch": int(epochs[n]) if complete else np.nan})
    return pd.DataFrame(records, index=origins)


def _ridge(x, y, q):
    usable = np.isfinite(q) & (np.isfinite(x).mean(axis=0) >= .6)
    x, q = x[:, usable], q[usable]
    if x.shape[1] < 5:
        return None
    median = np.nanmedian(x, axis=0)
    scale = np.nanquantile(x, .75, axis=0) - np.nanquantile(x, .25, axis=0)
    scale = np.where(scale > 1e-5, scale, 1.)
    x = np.clip((np.where(np.isfinite(x), x, median) - median) / scale, -4, 4)
    q = np.clip((q - median) / scale, -4, 4)
    x, q = np.column_stack([np.ones(len(x)), x]), np.r_[1., q]
    penalty = np.eye(x.shape[1]) * 50.
    penalty[0, 0] = 0
    try:
        coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        return None
    estimate = float(q @ coefficient)
    # Restrict an extrapolated correction-change to the range encountered in
    # completed training outcomes. This does not constrain measured raw PM.
    return float(np.clip(estimate, np.min(y), np.max(y)))


def replay(frame, lead_minutes, cams_runs, include_live=True, include_diagnostics=True):
    """Half-hourly prequential replay of six explicitly regional-first variants.

    Future windows comprise exactly eight complete 15-minute medians. Missing
    CAMS coverage excludes an origin; it must not become a persistence success.
    Replay predictions are retained even where later truth is unavailable so
    a missing future outcome cannot change the prediction issued in the past.
    """
    if frame.empty:
        return []
    features = support._features(frame, lead_minutes)
    origins = features.index[features.index.minute.isin([0, 30])]
    if include_live:
        origins = origins.union(frame.index[-1:]).sort_values()
    f = features.loc[origins].join(_regional_features(origins, lead_minutes, cams_runs))
    f["currentResidual"] = f.level - f.regionalCurrent
    f["camsDelta"] = f.regionalMean - f.regionalCurrent
    f["camsLevel"] = f.regionalCurrent
    target = support._targets(frame, lead_minutes).reindex(origins)
    ends = origins + pd.Timedelta(minutes=lead_minutes + 120)
    # Residual truth is tied to the origin's frozen regional future forecast.
    future_residual = (target - f.regionalMean).to_numpy(float)
    change = future_residual - f.currentResidual.to_numpy(float)
    x = f[list(COLS)].to_numpy(float)
    records = []
    for i, origin in enumerate(origins):
        if not np.isfinite(f.level.iloc[i]) or not np.isfinite(f.regionalMean.iloc[i]):
            continue
        epoch = int(origin.timestamp())
        train = (ends <= origin) & (origins >= origin - pd.Timedelta(days=TRAIN_DAYS)) & np.isfinite(change)
        recent = train & (ends >= origin - pd.Timedelta(hours=36))
        train_count = int(train.sum())
        r0 = float(f.currentResidual.iloc[i])
        regional = float(f.regionalMean.iloc[i])
        current = float(f.level.iloc[i])
        # Prior, completed forecasts supply a time-local estimate of the model
        # offset. Sample age is measured from target completion, not issue time.
        recent_n = int(recent.sum())
        bias = r0
        if include_diagnostics and recent_n >= MIN_BIAS:
            age_hours = np.array((origin - ends[recent]).total_seconds()) / 3600
            w = np.exp(-math.log(2) * age_hours / 12)
            bias = float(np.average(future_residual[recent], weights=w))
        relaxation = 1 - math.exp(-(lead_minutes + 60) / (12 * 60))
        relaxed = r0 + relaxation * (bias - r0)
        step = _ridge(x[train], change[train], x[i]) if train_count >= MIN_TRAIN else None
        # Continuous support shrinkage towards no residual evolution.
        support_weight = train_count / (train_count + MIN_TRAIN) if step is not None else 0.
        ridge_correction = r0 + support_weight * (step or 0.)
        learned_level = _ridge(x[train], future_residual[train], x[i]) if include_diagnostics and train_count >= MIN_TRAIN else None
        corrections = {"constant_residual": r0, "residual_relaxation": relaxed,
                       "residual_ridge": ridge_correction,
                       "residual_level_ridge": learned_level if learned_level is not None else r0,
                       "residual_full_ridge": r0 + (step or 0.)}
        candidates = {name: max(0., regional + value) for name, value in corrections.items()}
        known = [r for r in records if r["targetEndEpoch"] <= epoch and
                 r["targetEndEpoch"] >= epoch - 3 * 86400 and r["actual"] is not None] if include_diagnostics else []
        if len(known) >= MIN_BIAS:
            errors = np.array([np.mean([abs(r[c] - r["actual"]) for r in known])
                               for c in RESIDUAL_COMPONENTS])
            weights = 1 / (errors + 3.) ** 2
        else:
            weights = np.array([.5, .25, .25])
        weights /= weights.sum()
        adaptive_correction = float(sum(w * corrections[c] for w, c in zip(weights, RESIDUAL_COMPONENTS)))
        corrections["residual_adaptive"] = adaptive_correction
        candidates["residual_adaptive"] = max(0., regional + adaptive_correction)
        raw_correction = corrections[MODEL_VARIANT]
        raw_sum = regional + raw_correction
        record = {"originEpoch": epoch, "targetEndEpoch": int(ends[i].timestamp()),
                  "leadMinutes": int(lead_minutes), "actual": float(target.iloc[i]) if np.isfinite(target.iloc[i]) else None,
                  "prediction": max(0., raw_sum), "persistence": current, "raw_cams": regional,
                  "regionalMean": regional, "regionalCurrent": float(f.regionalCurrent.iloc[i]),
                  "currentResidual": r0, "learnedLocalCorrection": raw_correction,
                  "unclippedPrediction": raw_sum, "nonnegativeAdjustment": max(0., -raw_sum),
                  "effectiveLocalCorrection": max(0., raw_sum) - regional,
                  "recentCompletedBias": bias, "recentBiasCount": recent_n,
                  "trainingCount": train_count, "weightScoredCount": len(known),
                  "weights": dict(zip(RESIDUAL_COMPONENTS, weights.tolist())),
                  "corrections": corrections, "ridgeSupportWeight": support_weight,
                  "camsFetchedEpoch": int(f.camsFetchedEpoch.iloc[i]), **candidates}
        records.append(record)
    return records


def _archive_revision(db_path, origin_epoch):
    """Payload-content identity includes same-time source corrections."""
    digest = hashlib.blake2b(digest_size=16)
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)) as conn:
        for row in conn.execute("""SELECT fetched_epoch,model_version,payload FROM air_quality_forecast_runs
                WHERE fetched_epoch<=? AND model_version IN ('cams_anchor_v1','cams_anchor_v2') ORDER BY fetched_epoch""",
                                (origin_epoch,)):
            digest.update(str(row[0]).encode())
            digest.update(str(row[1]).encode())
            digest.update(row[2].encode())
    return digest.hexdigest()


def _evidence(records, epoch):
    scored = [r for r in records if r["actual"] is not None and r["targetEndEpoch"] <= epoch
              and r["trainingCount"] >= MIN_TRAIN]
    if not scored:
        return {"count": 0, "distinctDays": 0, "calibrated": False}, None
    error = np.array([r["actual"] - r["prediction"] for r in scored])
    baseline = np.array([r["actual"] - r["persistence"] for r in scored])
    constant = np.array([r["actual"] - r["constant_residual"] for r in scored])
    dates = {pd.Timestamp(r["originEpoch"], unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur").date()
             for r in scored}
    evidence = {"count": len(scored), "distinctDays": len(dates),
                "mae": round(float(np.abs(error).mean()), 2),
                "persistenceMae": round(float(np.abs(baseline).mean()), 2),
                "constantResidualMae": round(float(np.abs(constant).mean()), 2),
                "rmse": round(float(np.sqrt((error ** 2).mean())), 2),
                "validationMode": "issued_cams_exact_session_prequential_replay",
                "originSpacingMinutes": 30, "independentOrigins": False,
                "calibrated": False, "prospectivelyValidated": False,
                "limitation": "Exploratory selection on overlapping historical outcomes; tested alternatives were worse than the existing primary model overall. Not prospective validation."}
    return evidence, np.quantile(error, [.1, .9])


def predict_windows(db_path, rows, windows, issue_epoch):
    """Predict exact 120-minute windows with explicit CAMS + local correction."""
    clock = time.perf_counter()
    frame = support._frame(rows, issue_epoch)
    unavailable = lambda reason: {"available": False, "modelVersion": MODEL_VERSION,
                                  "reason": reason, "fallbackRequired": True}
    if frame.empty:
        return {key: unavailable("missing_sensor_data") for key in windows}
    latest_raw = max((float(dict(r).get("epoch", 0)) for r in rows
                      if dict(r).get("epoch") is not None and float(dict(r)["epoch"]) <= issue_epoch), default=0)
    if issue_epoch - latest_raw > 900:
        return {key: unavailable("stale_latest_sensor_reading") for key in windows}
    origin_epoch = int(frame.index[-1].timestamp())
    if not np.isfinite(frame.pm02.iloc[-1]) or issue_epoch - origin_epoch > 1800:
        return {key: unavailable("stale_or_missing_closed_sensor_bucket") for key in windows}
    digest = hashlib.blake2b(pd.util.hash_pandas_object(frame, index=True).values.tobytes(), digest_size=12).hexdigest()
    window_key = tuple((k, w.get("startEpoch"), w.get("endEpoch")) for k, w in sorted(windows.items()))
    revision = _archive_revision(db_path, origin_epoch)
    key = (str(db_path), MODEL_VERSION, MODEL_VARIANT, digest, window_key, revision)
    with _LOCK:
        # A later live call can share this closed frame and target with an
        # earlier replay. Reuse earlier issued trials without relabelling them,
        # but never expose a future issue to a historical request.
        cached = _CACHE.get(key)
        if cached is not None and all(
                value.get("forecastedAtEpoch") is None
                or value["forecastedAtEpoch"] <= issue_epoch
                for value in cached.values()):
            _CACHE.move_to_end(key)
            result = copy.deepcopy(cached)
            for value in result.values():
                value["computeSeconds"] = round(time.perf_counter() - clock, 3)
            return result
        runs = support._read_runs(db_path, "air_quality_forecast_runs", origin_epoch)
        result = {}
        for name, window in windows.items():
            start, end = window.get("startEpoch"), window.get("endEpoch")
            if start is None or end is None or end - start != 7200 or (start - origin_epoch) % 900:
                result[name] = unavailable("unsupported_session_alignment_or_duration")
                continue
            lead = (start - origin_epoch) // 60
            if not 90 <= lead <= support.MAX_LEAD_MINUTES:
                result[name] = unavailable("outside_90_min_to_24_hour_model_horizon")
                continue
            current_regional = _regional_features(frame.index[-1:], int(lead), runs).iloc[0]
            if not np.isfinite(current_regional.regionalMean):
                result[name] = unavailable("missing_stale_or_incomplete_issued_cams_forecast")
                continue
            records = replay(frame, int(lead), runs, include_diagnostics=False)
            if not records or records[-1]["originEpoch"] != origin_epoch:
                result[name] = unavailable("missing_regional_sensor_overlap")
                continue
            r = records[-1]
            if r["trainingCount"] < MIN_TRAIN:
                result[name] = unavailable("insufficient_complete_cams_residual_training_history")
                continue
            evidence, interval = _evidence(records, origin_epoch)
            mean = round(r["prediction"], 1)
            regional = round(r["regionalMean"], 1)
            # Effective correction and nonnegative adjustment preserve the
            # displayed one-decimal decomposition exactly.
            adjustment = round(r["nonnegativeAdjustment"], 1)
            correction = round(mean - regional - adjustment, 1)
            result[name] = {"available": True, "modelVersion": MODEL_VERSION,
                "source": "cams_regional_plus_local_correction", "pointRole": "experimental_window_mean",
                "mean": mean, "prediction": mean, "regionalMean": regional,
                "localCorrection": correction, "effectiveLocalCorrection": round(mean - regional, 1),
                "unclippedPrediction": round(r["unclippedPrediction"], 1), "nonnegativeAdjustment": adjustment,
                "decomposition": {"regionalMean": r["regionalMean"], "learnedLocalCorrection": r["learnedLocalCorrection"],
                                  "unclippedPrediction": r["unclippedPrediction"], "nonnegativeAdjustment": r["nonnegativeAdjustment"],
                                  "prediction": r["prediction"]},
                "regionalCorrection": {"regionalMeanPm25UgM3": regional,
                    "learnedCorrectionPm25UgM3": correction,
                    "unboundedMeanPm25UgM3": round(r["unclippedPrediction"], 1),
                    "nonnegativeFloorApplied": r["nonnegativeAdjustment"] > 0,
                    "currentRegionalPm25UgM3": round(r["regionalCurrent"], 1),
                    "currentLocalCorrectionPm25UgM3": round(r["currentResidual"], 1),
                    "method": "issued_cams_plus_shrunk_ridge_residual_change",
                    "ridgeSupportWeight": round(r["ridgeSupportWeight"], 4)},
                "currentRegional": round(r["regionalCurrent"], 1), "currentLocalResidual": round(r["currentResidual"], 1),
                "correctionChange": round(r["learnedLocalCorrection"] - r["currentResidual"], 1),
                "sensorAnchor": round(r["persistence"], 1), "anchorRole": "closed_15_minute_median_reference_not_forecast",
                "originEpoch": origin_epoch, "forecastedAtEpoch": int(issue_epoch), "startEpoch": int(start), "endEpoch": int(end),
                "leadHours": round(lead / 60, 2), "target": "mean_of_eight_closed_15_minute_sensor_medians",
                "rawRangeLow": round(max(0., r["prediction"] + interval[0]), 1) if interval is not None else None,
                "rawRangeHigh": round(max(0., r["prediction"] + interval[1]), 1) if interval is not None else None,
                "rangeRole": "empirical_q10_q90_session_mean_error_span", "calibrated": False, "confidence": "low",
                "experimental": True, "usedForDecision": False, "usedForComparison": False,
                "appliedToPrimaryForecast": False, "automaticPromotionEnabled": False,
                "trainingCount": r["trainingCount"], "adaptiveWeightScoredCount": 0,
                "components": {MODEL_VARIANT: round(r[MODEL_VARIANT], 2)},
                "weights": {MODEL_VARIANT: 1.0},
                "modelEvidence": evidence,
                "camsFetchedEpoch": r["camsFetchedEpoch"], "weatherFetchedEpoch": None,
                "camsAvailable": True, "weatherAvailable": False,
                "sourceCaveat": "Trial only: CAMS plus learned TTDI residual change did not beat the existing forecast overall; not applied to headline forecasts."}
        elapsed = round(time.perf_counter() - clock, 3)
        for value in result.values():
            value["computeSeconds"] = elapsed
        _CACHE[key] = copy.deepcopy(result)
        while len(_CACHE) > MAX_CACHE_SIZE:
            _CACHE.popitem(last=False)
        return result


def record_issue(db_path, result, recorded_epoch):
    """Preserve the first genuinely issued forecast; never use during replay."""
    with closing(sqlite3.connect(str(db_path), timeout=20)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS window_pm_forecast_issues(
            model_version TEXT NOT NULL, origin_epoch INTEGER NOT NULL,
            start_epoch INTEGER NOT NULL, end_epoch INTEGER NOT NULL,
            issued_epoch INTEGER NOT NULL, window_key TEXT NOT NULL,
            predicted_mean REAL NOT NULL, anchor_pm25 REAL NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(model_version,origin_epoch,start_epoch,end_epoch))""")
        count = 0
        for name, value in result.items():
            if not value.get("available") or value.get("modelVersion") != MODEL_VERSION:
                continue
            if int(recorded_epoch) < int(value["originEpoch"]) or int(recorded_epoch) >= int(value["startEpoch"]):
                continue
            payload = copy.deepcopy(value)
            payload["recordedIssuedEpoch"] = int(recorded_epoch)
            cursor = conn.execute("""INSERT OR IGNORE INTO window_pm_forecast_issues
                (model_version,origin_epoch,start_epoch,end_epoch,issued_epoch,window_key,predicted_mean,anchor_pm25,payload)
                VALUES(?,?,?,?,?,?,?,?,?)""", (MODEL_VERSION, value["originEpoch"], value["startEpoch"], value["endEpoch"],
                int(recorded_epoch), name, value["mean"], value["sensorAnchor"], json.dumps(payload, separators=(",", ":"))))
            count += cursor.rowcount
        return count
