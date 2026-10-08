"""Read-only shadow evaluation of the afternoon direction candidate.

Eligible 07:00--08:00 issues target that day's 14:00--16:00 session. Historical
issues use the same morning scope and exact afternoon target clock. The
candidate never replaces the selected session estimate; its diagnostics can
be archived alongside the issue for paired prospective evaluation.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import logging
import math
import time

import numpy as np
import pandas as pd

import afternoon_direction_features as features
import afternoon_direction_model as frozen_model
from forecast_freshness import references, VERSION as FRESHNESS_VERSION


MODEL_VERSION = "afternoon_direction_hgb_v1"
SOURCE = "experimental_afternoon_direction_model"
METHOD = "Experimental afternoon direction model · fixed 40-iteration classifier"
FORECAST_STATE = "experimental_afternoon_direction_mean"
UNCERTAINTY_METHOD = "not_estimated_for_experimental_direction_model"
KL = timezone(timedelta(hours=8))
_LOGGER = logging.getLogger(__name__)
SCOPE = {
    "timezone": "Asia/Kuala_Lumpur",
    "issueStartLocal": "07:00:00",
    "issueEndLocal": "08:00:00",
    "targetStartLocal": "14:00:00",
    "targetEndLocal": "16:00:00",
    "sameDayOnly": True,
    "historicalIssueScope": "07:00:00_through_08:00:00",
    "historicalOriginMinutes": [0, 15, 30, 45],
    "previouslyReviewedIssueLocal": "07:30:00",
}
RETROSPECTIVE_REVIEW = {
    "status": "previous_review_superseded_by_clock_alignment_fix",
    "previousReviewNotApplicableToAlignedCandidate": True,
    "prospectivelyValidated": False,
    "limitations": "The earlier review used historical issue and target clocks unlike the live scope. The aligned candidate remains shadow only.",
}


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _metadata_copy(value):
    """Copy diagnostics as strict JSON data, never changing forecast inputs."""
    if isinstance(value, dict):
        return {str(key): _metadata_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_copy(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return copy.deepcopy(value)


def _with_shadow(value, issue, start, end, attempt):
    """Keep the selected estimator and its provenance intact."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    diagnostic = {
        "applied": False,
        "usedForDecision": False,
        "appliedToPrimaryForecast": False,
        "status": "shadow_only",
        "modelVersion": MODEL_VERSION,
        "forecastIssuedEpoch": issue,
        "targetStartEpoch": start,
        "targetEndEpoch": end,
        "scope": dict(SCOPE),
        "prospectivelyValidated": False,
        "attempt": _metadata_copy(attempt),
    }
    # A caller may already have selected a direction record. Do not erase
    # that record or its provenance while attaching a separate trial.
    key = "directionShadow" if (result.get("directionalModel") or {}).get("applied") else "directionalModel"
    result[key] = diagnostic
    return result


def _applicable_windows(windows, issue):
    local = datetime.fromtimestamp(issue, KL)
    clock = (local.hour, local.minute, local.second, local.microsecond)
    if not (7, 0, 0, 0) <= clock <= (8, 0, 0, 0):
        return {}
    start = int(local.replace(hour=14, minute=0, second=0, microsecond=0).timestamp())
    end = int(local.replace(hour=16, minute=0, second=0, microsecond=0).timestamp())
    return {key: (start, end) for key, window in windows.items()
            if key == "afternoon" and isinstance(window, dict) and _number(window.get("startEpoch")) == start
            and _number(window.get("endEpoch")) == end}


def _same_day_afternoon(windows, issue):
    """Identify the one target whose estimator-selection policy we annotate."""
    window = windows.get("afternoon")
    if not isinstance(window, dict):
        return None
    local = datetime.fromtimestamp(issue, KL)
    start = int(local.replace(hour=14, minute=0, second=0, microsecond=0).timestamp())
    end = int(local.replace(hour=16, minute=0, second=0, microsecond=0).timestamp())
    if _number(window.get("startEpoch")) != start or _number(window.get("endEpoch")) != end:
        return None
    return start, end


def _training_and_query(db_path, rows, issue, start, end, *, preparation_diagnostics=None):
    def unavailable(reason, **support):
        if preparation_diagnostics is not None:
            preparation_diagnostics.update(state="not_fitted", reason=reason, **support)
        return None

    rows, frame, cams, weather = features.load(db_path, issue, rows)
    if frame.empty:
        return unavailable("required_sensor_history_unavailable")
    origin_epoch = int(issue // 900 * 900)
    origin = pd.Timestamp(origin_epoch, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    if origin not in frame.index or not np.isfinite(frame.loc[origin, "pm02"]):
        return unavailable("qualified_closed_reference_unavailable")
    lag = issue - origin_epoch
    lead_minutes = (start - issue) / 60.0
    duration_minutes = (end - start) / 60.0
    target_clock = (14, 0, 16, 0)
    # At each historical quarter-hour, issue with the same lag into the
    # current bucket as the live call. The 08:00 origin is eligible only when
    # the actual issue is exactly at the bucket boundary.
    historical_origins = frame.index[
        ((frame.index.hour == 7) & frame.index.minute.isin((0, 15, 30, 45)))
        | ((frame.index.hour == 8) & (frame.index.minute == 0) & (lag == 0))]
    training = features.design(frame, rows, cams, weather, issue, lead_minutes,
                               duration_minutes, lag_seconds=lag,
                               only_origins=historical_origins,
                               target_clock=target_clock)
    query = features.design(frame, rows, cams, weather, issue, lead_minutes,
                            duration_minutes, lag_seconds=lag,
                            only_origins=pd.DatetimeIndex([origin]),
                            target_clock=target_clock)
    if not bool(query["valid"][0]):
        return unavailable("required_query_reference_unavailable")
    usable = (training["valid"] & np.isfinite(training["labels"])
              & (training["complete"] <= issue) & (training["issues"] < issue)
              & (training["issues"] >= issue - 14 * 86400))
    if int(usable.sum()) < frozen_model.MIN_TRAINING_ROWS:
        return unavailable("insufficient_aligned_training_rows", trainingCount=int(usable.sum()),
                           trainingDistinctDays=int(training["origins"][usable].normalize().nunique()),
                           minimumTrainingCount=frozen_model.MIN_TRAINING_ROWS,
                           historicalIssueScope=SCOPE["historicalIssueScope"],
                           targetClock="same_day_14_to_16")
    changes = training["labels"][usable] - training["x"].loc[usable, "fresh"].to_numpy(dtype=float)
    threshold = frozen_model.EVENT_CHANGE_THRESHOLD
    directions = np.where(changes >= threshold, 1, np.where(changes <= -threshold, -1, 0))
    if len(np.unique(directions)) < 2:
        return unavailable("insufficient_training_classes", trainingCount=int(usable.sum()),
                           trainingClassCounts={str(c): int(np.sum(directions == c)) for c in (-1, 0, 1)})
    return training, query, usable, origin_epoch, lag


def _replacement(old, training, query, usable, issue, origin, lag, start, end, elapsed,
                 *, decision_audit=None):
    query_row = query["x"].iloc[0]
    fresh = float(query_row["fresh"])
    details = frozen_model.predict_details(
        training["x"].loc[usable].copy(), training["labels"][usable].copy(), query_row.copy(),
        training["issues"][usable].copy(), issue, "same_day_afternoon_14_16")
    if decision_audit is not None:
        decision_audit.update(state="evaluated", reason=details.get("decisionReason"),
                              diagnostics=_metadata_copy(details))
    point = _number(details.get("prediction"))
    if point is None or point < 0 or details.get("decisionReason") == "fit_or_prediction_failure":
        if decision_audit is not None:
            decision_audit["state"] = "failed"
        return None
    change = point - fresh
    direction = "steady" if abs(change) < 1e-9 else "rising" if change > 0 else "falling"
    audit = query["audit"][0]
    closed = float(query_row["closedLevel"])
    reference_epoch = int(query["stamps"][0])
    reference_count = int(query["counts"][0])
    comparator = _number(old.get("mean"))
    if comparator is None:
        comparator = _number(old.get("prediction"))
    review = dict(RETROSPECTIVE_REVIEW)
    evidence = {
        "validationMode": "retrospective_exploratory_review_not_prospective",
        "prospectivelyValidated": False,
        "retrospectiveReview": review,
        "trainingCount": int(usable.sum()),
        "trainingLatestCompleteEpoch": int(training["complete"][usable].max()),
        "trainingEarliestIssueEpoch": int(training["issues"][usable].min()),
        "trainingLatestIssueEpoch": int(training["issues"][usable].max()),
        "historicalIssueScope": SCOPE["historicalIssueScope"],
        "historicalOriginMinutes": SCOPE["historicalOriginMinutes"],
        "trainingIssueLagSeconds": lag,
        "queryIssueLagSeconds": lag,
        "historicalTargetClock": "same_day_14_to_16",
        "trainingDistinctDays": int(training["origins"][usable].normalize().nunique()),
        "featureVersion": features.FEATURE_VERSION,
        "targetCoveragePolicy": "all_intersected_15_minute_buckets_complete_and_gap_qualified",
    }
    # Construct a new estimator record: old intervals, calibration and old
    # performance evidence do not describe this different point estimator.
    return {
        "available": True, "modelVersion": MODEL_VERSION, "source": SOURCE,
        "method": METHOD, "forecastState": FORECAST_STATE,
        "pointRole": "experimental_session_mean", "mean": point, "prediction": point,
        "sensorAnchor": fresh, "closedSensorAnchor": closed,
        "sensorReferenceEpoch": reference_epoch, "sensorReferenceCount": reference_count,
        "anchorRole": "fresh_five_minute_sensor_median_reference_not_forecast",
        "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median",
        "freshnessAdjustment": {
            "applied": False, "referenceUsedByModel": True, "version": FRESHNESS_VERSION,
            "referenceDifference": fresh - closed, "windowSeconds": 300,
            "closedReferencePm25UgM3": closed, "featureAnchorEpoch": origin,
            "referenceEpoch": reference_epoch, "referenceCount": reference_count,
            "lagSeconds": lag, "pointPolicy": "experimental_direction_prediction_from_fresh_reference",
            "prospectivelyValidated": False,
        },
        "originEpoch": origin, "forecastedAtEpoch": issue, "forecastIssuedEpoch": issue,
        "startEpoch": start, "endEpoch": end,
        "leadHours": (start - issue) / 3600.0,
        "remainingLeadHours": (start - issue) / 3600.0, "durationHours": 2.0,
        "target": "mean_of_eight_closed_15_minute_sensor_medians",
        "rawRangeLow": None, "rawRangeHigh": None,
        "rangeRole": "not_estimated_for_experimental_direction_model",
        "uncertaintyMethod": UNCERTAINTY_METHOD, "calibrated": False,
        "confidence": "low", "experimental": True, "prospectivelyValidated": False,
        "usedForDecision": True, "usedForComparison": False,
        "trainingCount": int(usable.sum()), "modelEvidence": evidence,
        "directionalModel": {
            "applied": True, "direction": direction, "changeFromFreshUgM3": change,
            "frozenModelName": frozen_model.MODEL_NAME, "scope": dict(SCOPE),
            "experimental": True, "prospectivelyValidated": False,
            "retrospectiveReview": dict(review),
            "diagnostics": _metadata_copy(details),
        },
        "incumbentComparator": {
            "role": "diagnostic_only", "appliedToPrimaryForecast": False,
            "modelVersion": old.get("modelVersion"), "meanPm25UgM3": comparator,
            "forecastedAtEpoch": old.get("forecastedAtEpoch"),
            "startEpoch": old.get("startEpoch"), "endEpoch": old.get("endEpoch"),
        },
        "components": {"freshReference": fresh, "learnedDirectionChange": change},
        "weights": {}, "componentsRole": "fresh_reference_plus_learned_direction_class_median_change",
        "camsFetchedEpoch": audit["camsFetchedEpoch"],
        "weatherFetchedEpoch": audit["weatherFetchedEpoch"],
        "camsAvailable": bool(np.isfinite(query_row["camsCurrent"]) and np.isfinite(query_row["camsTarget"])),
        "weatherAvailable": bool(np.isfinite(query_row[list(features.WEATHER)].to_numpy(dtype=float)).any()),
        "weatherSourceClock": {
            "asOfEpoch": issue, "sensorFeatureOriginEpoch": origin,
            "fetchedEpoch": audit["weatherFetchedEpoch"], "issueLagSeconds": lag,
            "availabilityRule": "fetched_epoch_at_or_before_actual_issue", "maximumAgeSeconds": 7200,
        },
        "sourceCaveat": "Experimental direction classifier; retrospective review used known historical outcomes and failed the latest reviewed day. No calibrated interval or prospective accuracy claim.",
        "computeSeconds": round(elapsed, 4),
    }


def apply_experimental_afternoon(db_path, rows, windows, issue, estimates) -> dict:
    """Attach an as-issued shadow attempt; never change the selected point."""
    result = dict(estimates)
    number = _number(issue)
    if number is None or number != int(number):
        return result
    issue = int(number)
    target = None
    try:
        target = _same_day_afternoon(windows, issue)
        if target is None:
            return result
        start, end = target
        def retain(reason, attempt=None):
            if "afternoon" in estimates:
                result["afternoon"] = _with_shadow(
                    estimates["afternoon"], issue, start, end,
                    attempt or {"state": "not_evaluated", "reason": reason})
            return result
        eligible = _applicable_windows(windows, issue)
        if not eligible:
            return result
        origin = issue // 900 * 900
        lag = issue - origin
        recent, _, counts = references(rows, [origin], lag, issue)
        if not len(recent) or not np.isfinite(recent[0]) or counts[0] <= 0:
            return retain("fresh_sensor_reference_unavailable",
                          {"state": "not_fitted", "reason": "fresh_sensor_reference_unavailable"})
        started = time.perf_counter()
        preparation = {}
        prepared = _training_and_query(db_path, rows, issue, start, end,
                                       preparation_diagnostics=preparation)
        if prepared is None:
            reason = preparation.get("reason", "training_or_query_unavailable")
            return retain(reason, preparation or {"state": "not_fitted", "reason": reason})
        training, query, usable, origin, lag = prepared
        decision = {}
        value = _replacement(estimates.get("afternoon") or {}, training, query, usable,
                             issue, origin, lag, start, end, 0.0, decision_audit=decision)
        if value is None:
            reason = ("fit_or_prediction_failure" if decision.get("reason") == "fit_or_prediction_failure"
                      else "invalid_experimental_prediction")
            return retain(reason, decision)
        decision.update(
            state="evaluated", reason="shadow_candidate_available",
            candidateMeanPm25UgM3=value["mean"],
            incumbentMeanPm25UgM3=(estimates.get("afternoon") or {}).get("mean"),
            modelEvidence=value["modelEvidence"],
            computeSeconds=round(time.perf_counter() - started, 4),
        )
        result["afternoon"] = _with_shadow(
            estimates["afternoon"], issue, start, end, decision)
        return result
    except Exception as error:
        _LOGGER.warning("Experimental afternoon fallback at issue %s: %s", issue, error)
        result = dict(estimates)
        if target is not None and "afternoon" in estimates:
            result["afternoon"] = _with_shadow(
                estimates["afternoon"], issue, *target,
                {"state": "failed", "reason": "experimental_error", "errorType": type(error).__name__})
        return result
