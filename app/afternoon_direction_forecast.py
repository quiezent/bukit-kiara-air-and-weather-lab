"""Scoped, read-only experimental afternoon direction forecast.

Only morning issues from 07:00:00 through 08:00:00 Asia/Kuala_Lumpur may replace
the same day's exact 14:00--16:00 session. Historical origins remain hourly
at :30; the live query uses its actual issue and exact target clocks. All
training targets, including their last intersected bucket, must be complete.
No issue archives are written here. The caller owns publication/archiving.
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
SELECTION_METADATA_VERSION = "afternoon_scope_selection_v1"
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
    "historicalOriginMinute": 30,
    "reviewedIssueLocal": "07:30:00",
}
RETROSPECTIVE_REVIEW = {
    "status": "exploratory_research_user_accepted_experimental",
    "reviewedDays": 17,
    "modelMaeUgM3": 22.216,
    "comparatorMaeUgM3": 25.841,
    "comparator": "incumbent_current_model",
    "comparisonTarget": "same_day_14_to_16_mean_issued_at_07_30",
    "latestReviewedDay": "2026-09-20",
    "latestReviewedDayImproved": False,
    "latestDayFailureAcceptedAsExperimental": True,
    "prospectivelyValidated": False,
    "limitations": "Exploratory reused historical outcomes; latest reviewed day failed; no prospective accuracy claim.",
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


def _with_selection(value, issue, start, end, reason, attempt=None):
    """Attach issue-bound provenance without changing a forecast or its clocks.

    Scheduled transitions are policy metadata, never an invented observation
    of a previous issued forecast. There is no delivery-time clock or history
    cache here; a cached record keeps its original forecast issue and model.
    """
    if not isinstance(value, dict):
        return value
    result = dict(value)
    local = datetime.fromtimestamp(issue, KL)
    scope_start = int(local.replace(hour=7, minute=0, second=0, microsecond=0).timestamp())
    scope_end = int(local.replace(hour=8, minute=0, second=0, microsecond=0).timestamp())
    ordinary_start = scope_end + 1  # The public issue API accepts whole seconds.
    experimental_policy = "experimental_morning_direction"
    ordinary_policy = "ordinary_session"
    selected_policy = experimental_policy if value.get("modelVersion") == MODEL_VERSION else ordinary_policy
    forecast_issue = _number(value.get("forecastIssuedEpoch"))
    if forecast_issue is None:
        forecast_issue = _number(value.get("forecastedAtEpoch"))
    issue_state = ("unknown" if forecast_issue is None else "same_issue" if forecast_issue == issue
                   else "earlier_issue" if forecast_issue < issue else "future_issue")
    retained = issue_state == "earlier_issue"
    policy_reason = reason
    if issue_state == "future_issue":
        reason = "supplied_forecast_has_future_issue"
    elif selected_policy == experimental_policy and reason != "inside_approved_morning_scope":
        reason = "retained_previously_issued_experimental_forecast" if retained else "supplied_experimental_forecast_not_reissued"
    messages = {
        "inside_approved_morning_scope": "Experimental morning direction model selected for this issue.",
        "morning_scope_not_started": "Ordinary session model selected before the experimental morning window.",
        "morning_scope_ended": "Ordinary session model selected after the experimental morning window ended.",
        "fresh_sensor_reference_unavailable": "Existing session forecast retained because a fresh sensor reference is unavailable.",
        "training_or_query_unavailable": "Existing session forecast retained because the experimental training or query is unavailable.",
        "required_sensor_history_unavailable": "Existing session forecast retained because sensor history is unavailable.",
        "qualified_closed_reference_unavailable": "Existing session forecast retained because the required closed sensor reference is unavailable.",
        "required_query_reference_unavailable": "Existing session forecast retained because the experimental query reference is unavailable.",
        "insufficient_training_rows": "Existing session forecast retained because too few completed training targets are available.",
        "insufficient_training_classes": "Existing session forecast retained because fewer than two training direction classes are available.",
        "fit_or_prediction_failure": "Existing session forecast retained because the experimental model fit or prediction failed.",
        "invalid_experimental_prediction": "Existing session forecast retained because the experimental prediction is invalid.",
        "experimental_error": "Existing session forecast retained because the experimental calculation failed.",
        "retained_previously_issued_experimental_forecast": "Earlier experimental forecast retained with its original issue time; no new forecast was issued here.",
        "supplied_experimental_forecast_not_reissued": "Supplied experimental forecast preserved; it was not recomputed for this selection.",
        "supplied_forecast_has_future_issue": "Supplied forecast is dated after this selection issue; it is not a valid as-of forecast for this issue.",
    }
    entry = {"atEpoch": scope_start, "fromPolicy": ordinary_policy, "toPolicy": experimental_policy,
             "basis": "scheduled_scope_boundary", "priorForecastObserved": False}
    exit_transition = {"atEpoch": ordinary_start, "fromPolicy": experimental_policy, "toPolicy": ordinary_policy,
                       "lastExperimentalIssueEpoch": scope_end,
                       "basis": "scheduled_scope_boundary", "priorForecastObserved": False}
    previous = None if issue < scope_start else entry if issue <= scope_end else exit_transition
    following = entry if issue < scope_start else exit_transition if issue <= scope_end else None
    result["modelSelection"] = {
        "version": SELECTION_METADATA_VERSION,
        "numericalPolicyIdentifier": value.get("modelVersion"),
        "selectedModelVersion": value.get("modelVersion"), "selectedPolicy": selected_policy,
        "selectionReason": reason, "selectionReasonText": messages[reason],
        "policyEligibilityReason": policy_reason, "selectionIssueEpoch": issue,
        "forecastIssuedEpoch": forecast_issue, "forecastIssueState": issue_state,
        "retainedIssuedForecast": retained, "targetStartEpoch": start, "targetEndEpoch": end,
        "scope": {**SCOPE, "issueStartEpoch": scope_start, "lastExperimentalIssueEpoch": scope_end,
                  "ordinaryPolicyStartsEpoch": ordinary_start, "issueEndInclusive": True},
        "previousScheduledTransition": previous, "nextScheduledTransition": following,
        "handoff": {"lastExperimentalIssueEpoch": scope_end, "ordinaryPolicyStartsEpoch": ordinary_start,
                    "mayChangePointWithoutNewObservations": True,
                    "interpretation": "A point change at this boundary can reflect the selected estimator; it does not by itself indicate a physical PM2.5 change."},
    }
    if attempt is not None:
        result["modelSelection"]["experimentalAttempt"] = _metadata_copy(attempt)
    return result


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
    training = features.design(frame, rows, cams, weather, issue, lead_minutes,
                               duration_minutes, lag_seconds=0, origin_minute=30)
    query = features.design(frame, rows, cams, weather, issue, lead_minutes,
                            duration_minutes, lag_seconds=lag,
                            only_origins=pd.DatetimeIndex([origin]))
    if not bool(query["valid"][0]):
        return unavailable("required_query_reference_unavailable")
    usable = (training["valid"] & np.isfinite(training["labels"])
              & (training["complete"] <= issue) & (training["issues"] < issue)
              & (training["issues"] >= issue - 14 * 86400))
    if int(usable.sum()) < frozen_model.MIN_TRAINING_ROWS:
        return unavailable("insufficient_training_rows", trainingCount=int(usable.sum()),
                           minimumTrainingCount=frozen_model.MIN_TRAINING_ROWS)
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
        training["issues"][usable].copy(), issue, "session390")
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
        "historicalOriginMinute": 30,
        "trainingIssueLagSeconds": 0,
        "queryIssueLagSeconds": lag,
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
    """Replace only eligible early-afternoon records; failures retain estimates.

    Morning and other targets retain their original objects. The exact named
    same-day afternoon record receives additive issue-bound selection metadata
    on all paths. Its point and original issue/target clocks are preserved on
    fallbacks. No input is mutated. Finite recent PM and the expected closed
    bucket are both required for the experimental calculation.
    """
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
                result["afternoon"] = _with_selection(estimates["afternoon"], issue, start, end, reason, attempt)
            return result
        eligible = _applicable_windows(windows, issue)
        if not eligible:
            local = datetime.fromtimestamp(issue, KL)
            return retain("morning_scope_not_started" if local.hour < 7 else "morning_scope_ended",
                          {"state": "not_evaluated", "reason": "outside_approved_morning_scope"})
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
        value["computeSeconds"] = round(time.perf_counter() - started, 4)
        result["afternoon"] = _with_selection(value, issue, start, end, "inside_approved_morning_scope")
        return result
    except Exception as error:
        _LOGGER.warning("Experimental afternoon fallback at issue %s: %s", issue, error)
        result = dict(estimates)
        if target is not None and "afternoon" in estimates:
            result["afternoon"] = _with_selection(
                estimates["afternoon"], issue, *target, "experimental_error",
                {"state": "failed", "reason": "experimental_error", "errorType": type(error).__name__})
        return result
