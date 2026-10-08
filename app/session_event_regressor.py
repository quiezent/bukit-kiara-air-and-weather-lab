"""Experimental afternoon session-delta model with narrow issue-time scope.

The only application clocks are 07:00--07:05 -> 14:00--16:00 and
12:30--12:35 -> 14:15--16:15 Malaysia time.  The point is a fresh sensor
reference plus a learned target change.  Historical labels are used only
after their last qualifying bucket has closed at the actual issue time.
Weather and CAMS features are selected from vintages fetched by each issue.
This module makes no archive writes or network requests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import afternoon_direction_features as features
from forecast_freshness import VERSION as FRESHNESS_VERSION

MODEL_VERSION = "experimental_afternoon_delta_hgb_v1"
SOURCE = "experimental_afternoon_delta_regressor"
MIN_TRAINING_ROWS = 120
LOOKBACK_SECONDS = 14 * 86400
LOCAL = timezone(timedelta(hours=8))
UNCERTAINTY = "not_estimated_for_experimental_session_delta_model"
PARAMETERS = {
    "max_iter": 80,
    "max_depth": 2,
    "max_leaf_nodes": 4,
    "min_samples_leaf": 20,
    "l2_regularization": 50.0,
    "learning_rate": 0.05,
    "random_state": 1749,
    "early_stopping": False,
}
_LOGGER = logging.getLogger(__name__)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _scope(issue, window):
    """Return the exact tested target and clock label, else None."""
    issue_number = _number(issue)
    if (not isinstance(window, dict) or issue_number is None
            or issue_number != int(issue_number)):
        return None
    issue = int(issue_number)
    local = datetime.fromtimestamp(issue, LOCAL)
    seconds = local.hour * 3600 + local.minute * 60 + local.second
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if 7 * 3600 <= seconds < 7 * 3600 + 300:
        name, start = "07_afternoon", day + timedelta(hours=14)
    elif 12 * 3600 + 30 * 60 <= seconds < 12 * 3600 + 35 * 60:
        name, start = "1230_afternoon", day + timedelta(hours=14, minutes=15)
    else:
        return None
    begin, end = int(start.timestamp()), int((start + timedelta(hours=2)).timestamp())
    if (_number(window.get("startEpoch")), _number(window.get("endEpoch"))) != (begin, end):
        return None
    return name, begin, end


def _numeric(frame):
    values = frame.reindex(columns=features.FEATURES).apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    values[~np.isfinite(values)] = np.nan
    return values


def fit_delta(train_x, train_y, query_x):
    """Fixed regression head used in the read-only day-by-day replay."""
    if len(train_x) < MIN_TRAINING_ROWS:
        return None
    query = query_x.to_frame().T if isinstance(query_x, pd.Series) else query_x
    if len(query) != 1:
        raise ValueError("query must contain one row")
    y = np.asarray(train_y, dtype=float).reshape(-1)
    if len(y) != len(train_x) or not np.isfinite(y).all():
        raise ValueError("training labels must be finite and align with features")
    x = _numeric(train_x)
    test = _numeric(query)
    if not np.isfinite(x[:, 0]).all() or not np.isfinite(test[0, 0]):
        return None
    change = y - x[:, 0]
    estimator = make_pipeline(SimpleImputer(add_indicator=True),
                              HistGradientBoostingRegressor(**PARAMETERS))
    with threadpool_limits(limits=1):
        estimator.fit(x, change)
        result = float(estimator.predict(test)[0])
    return result if math.isfinite(result) else None


def predict_scoped(db_path, rows, issue, window, old=None):
    """Return a standalone replacement record, or None when ineligible.

    A caller must provide raw readings available at issue. The feature loader
    ignores later timestamps and uses at most 15 days for rolling warmup.
    """
    scope = _scope(issue, window)
    if scope is None:
        return None
    name, start, end = scope
    issue = int(issue)
    old = old if isinstance(old, dict) else {}
    rows, frame, cams, weather = features.load(db_path, issue, rows)
    if frame.empty:
        return None
    origin = issue // 900 * 900
    origin_stamp = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    if origin_stamp not in frame.index or not np.isfinite(frame.loc[origin_stamp, "pm02"]):
        return None
    lag = issue - origin
    # Both historical and live windows must retain the exact local target
    # clocks, even when the live issue is seconds after the quarter-hour.
    training = features.design(frame, rows, cams, weather, issue,
                               (start - origin) / 60, 120,
                               lag_seconds=0, origin_minute=origin_stamp.minute)
    query = features.design(frame, rows, cams, weather, issue,
                            (start - issue) / 60, 120,
                            lag_seconds=lag,
                            only_origins=pd.DatetimeIndex([origin_stamp]))
    if not bool(query["valid"][0]):
        return None
    usable = (training["valid"] & np.isfinite(training["labels"])
              & (training["complete"] <= issue)
              & (training["issues"] < issue)
              & (training["issues"] >= issue - LOOKBACK_SECONDS))
    if int(usable.sum()) < MIN_TRAINING_ROWS:
        return None
    delta = fit_delta(training["x"].loc[usable], training["labels"][usable],
                      query["x"].iloc[0])
    if delta is None:
        return None
    row = query["x"].iloc[0]
    fresh = float(row["fresh"])
    point = max(0.0, fresh + delta)
    applied_delta = point - fresh
    audit = query["audit"][0]
    prior = _number(old.get("mean"))
    if prior is None:
        prior = _number(old.get("prediction"))
    return {
        "available": True, "experimental": True,
        "prospectivelyValidated": False, "modelVersion": MODEL_VERSION,
        "source": SOURCE, "method": "80-iteration gradient-boosted session-delta regression",
        "forecastState": "experimental_afternoon_session_mean",
        "pointRole": "experimental_session_mean", "mean": point, "prediction": point,
        "startEpoch": start, "endEpoch": end,
        "originEpoch": origin, "forecastedAtEpoch": issue, "forecastIssuedEpoch": issue,
        "leadHours": (start - issue) / 3600, "remainingLeadHours": (start - issue) / 3600,
        "durationHours": 2.0,
        "target": "mean_of_eight_closed_15_minute_sensor_medians",
        "sensorAnchor": fresh, "closedSensorAnchor": float(row["closedLevel"]),
        "sensorReferenceEpoch": int(query["stamps"][0]),
        "sensorReferenceCount": int(query["counts"][0]),
        "anchorRole": "fresh_five_minute_sensor_median_reference_not_forecast",
        "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median",
        "freshnessAdjustment": {
            "applied": False, "referenceUsedByModel": True,
            "version": FRESHNESS_VERSION, "referenceDifference": fresh - float(row["closedLevel"]),
            "windowSeconds": 300, "referenceEpoch": int(query["stamps"][0]),
            "referenceCount": int(query["counts"][0]), "lagSeconds": lag,
            "pointPolicy": "experimental_session_delta_prediction_from_fresh_reference",
        },
        "rawRangeLow": None, "rawRangeHigh": None,
        "rangeRole": UNCERTAINTY, "uncertaintyMethod": UNCERTAINTY,
        "calibrated": False, "confidence": "low",
        "usedForDecision": True, "usedForComparison": False,
        "components": {"freshReference": fresh, "learnedDelta": applied_delta},
        "componentsRole": "fresh_reference_plus_learned_session_delta",
        "trainingCount": int(usable.sum()),
        "modelEvidence": {
            "validationMode": "retrospective_exploratory_daily_replay_not_prospective",
            "trainingLatestCompleteEpoch": int(training["complete"][usable].max()),
            "trainingEarliestIssueEpoch": int(training["issues"][usable].min()),
            "trainingLatestIssueEpoch": int(training["issues"][usable].max()),
            "lookbackSeconds": LOOKBACK_SECONDS,
            "featureVersion": features.FEATURE_VERSION,
            "historicalOriginMinute": origin_stamp.minute,
            "historicalIssueLagSeconds": 0, "queryIssueLagSeconds": lag,
            "reviewedIssueClock": name,
            "reviewedDates": "2026-08-26_to_2026-09-29",
            "targetCoveragePolicy": "all_intersected_15_minute_buckets_complete_and_gap_qualified",
        },
        "sessionDeltaModel": {
            "applied": True, "changeFromFreshUgM3": applied_delta,
            "directionAtFiveUgM3": ("rising" if applied_delta >= 5 else
                                      "falling" if applied_delta <= -5 else "steady"),
            "parameters": dict(PARAMETERS),
        },
        "incumbentComparator": {
            "role": "diagnostic_only", "appliedToPrimaryForecast": False,
            "modelVersion": old.get("modelVersion"), "meanPm25UgM3": prior,
            "forecastedAtEpoch": old.get("forecastedAtEpoch"),
            "startEpoch": old.get("startEpoch"), "endEpoch": old.get("endEpoch"),
        },
        "camsFetchedEpoch": audit["camsFetchedEpoch"],
        "weatherFetchedEpoch": audit["weatherFetchedEpoch"],
        "camsAvailable": bool(np.isfinite(row["camsCurrent"]) and np.isfinite(row["camsTarget"])),
        "weatherAvailable": bool(np.isfinite(row[list(features.WEATHER)].to_numpy(float)).any()),
        "weatherSourceClock": {
            "asOfEpoch": issue, "fetchedEpoch": audit["weatherFetchedEpoch"],
            "availabilityRule": "fetched_epoch_at_or_before_actual_issue",
            "maximumAgeSeconds": 7200,
        },
        "sourceCaveat": "Exploratory historical comparison selected after seeing outcomes; no calibrated interval or prospective accuracy claim.",
    }


def apply_experimental_sessions(db_path, rows, windows, issue, estimates):
    """Replace only the exact eligible afternoon point; retain all others."""
    result = dict(estimates)
    if not isinstance(windows, dict) or "afternoon" not in windows:
        return result
    if _scope(issue, windows["afternoon"]) is None:
        return result
    try:
        replacement = predict_scoped(db_path, rows, issue, windows["afternoon"],
                                     estimates.get("afternoon"))
    except Exception as error:
        _LOGGER.warning("Experimental session fallback at issue %s: %s", issue, error)
        return result
    if replacement is not None:
        result["afternoon"] = replacement
    return result
