"""Publish existing near-model numbers without numerical model selection.

This module does no I/O, fitting, forecast arithmetic, thresholding or model
promotion. The fixed mapping is the original local analogue +90-minute output
and original local Ridge +90..210-minute mean output. Structural checks prevent
an output from being attached to a different issue or target.
The recent-sensor persistence value is retained only as comparator metadata.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real

from forecast_clock import TARGET_VERSION


VERSION = "direct_near_raw_model_outputs_v1"
_HEADS = {
    "arrival": {
        "sourceField": "arrivalPoint", "deltaField": "arrivalDelta",
        "targetName": "point90", "modelVersion": "local_bounded_analogue_raw_point90_v1",
        "trainingField": "trainingOriginCount", "metricField": "rawAnalogueArrivalMae",
        "baselineMetricField": "arrivalPersistenceMae",
        "method": "Original bounded local analogue output for the exact +90-minute target",
    },
    "trail": {
        "sourceField": "trailMeanPoint", "deltaField": "trailMeanDelta",
        "targetName": "mean90_210", "modelVersion": "local_robust_ridge_alpha100_raw_mean90_210_v1",
        "trainingField": "trailMeanRidgeTrainingOriginCount", "metricField": "rawRidgeTrailMeanMae",
        "baselineMetricField": "trailMeanPersistenceMae",
        "method": "Original robust Ridge output for the exact +90..210-minute mean",
    },
}
_BASELINE_UNCERTAINTY_FIELDS = (
    "rangeLow", "rangeHigh", "rawRangeLow", "rawRangeHigh", "rawUpper90", "upper90",
    "decisionUpper", "upperTargetCoverage", "finiteSampleRankCoverage", "finiteSampleUpper",
    "finiteSampleRank", "rangeRole", "rawRangeRole", "upperQuantileRole", "upperMeanRole",
    "peakUpperRole", "rawPeakUpperRole", "confidence", "calibrationState",
    "peak", "peakUpper", "peakRangeLow", "peakRangeHigh", "projectedPeak",
    "decisionPeakUpper", "empiricalPeakUpper", "empiricalPeakQuantileRole",
    "rawUpperMean90", "upperMean", "decisionUpperMean", "rawPeakUpper90",
    "rawPeakRangeLow", "rawPeakRangeHigh", "rawPeak", "rawProjectedPeak", "upperPeak",
    "upperPeak90", "upperPeakRole", "peakRole", "peakQuantileRole", "meanQuantileRole",
    "meanSkillEligible", "peakSkillEligible", "deploymentSkill", "qualification", "qualificationPolicy",
)
_NUMERIC_UNCERTAINTY_FIELDS = (
    "rangeLow", "rangeHigh", "rawRangeLow", "rawRangeHigh", "rawUpper90", "upper90",
    "decisionUpper", "peak", "peakUpper", "peakRangeLow", "peakRangeHigh",
    "projectedPeak", "decisionPeakUpper", "empiricalPeakUpper",
    "rawUpperMean90", "upperMean", "decisionUpperMean", "rawPeakUpper90",
    "rawPeakRangeLow", "rawPeakRangeHigh", "rawPeak", "rawProjectedPeak", "upperPeak", "upperPeak90",
)
_UNCERTAINTY_ROLE_FIELDS = (
    "rangeRole", "rawRangeRole", "upperQuantileRole", "upperMeanRole", "peakUpperRole",
    "rawPeakUpperRole", "empiricalPeakQuantileRole", "upperPeakRole", "peakRole",
    "peakQuantileRole", "meanQuantileRole",
)


def _finite_number(value):
    # Numeric strings and booleans are malformed model outputs, not readings.
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _epoch(value):
    number = _finite_number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _clock_reason(clock, issue):
    if not isinstance(clock, dict):
        return "model_target_clock_unavailable"
    if clock.get("targetVersion") != TARGET_VERSION:
        return "model_target_definition_mismatch"
    if _epoch(clock.get("forecastIssuedEpoch")) != issue:
        return "model_issue_clock_mismatch"
    anchor = _epoch(clock.get("featureAnchorEpoch"))
    if anchor is None or anchor % 900 or anchor > issue:
        return "model_feature_anchor_invalid"
    expected = {"arrivalTargetEpoch": issue + 5400,
                "windowStartEpoch": issue + 5400, "windowEndEpoch": issue + 12600}
    if any(_epoch(clock.get(key)) != value for key, value in expected.items()):
        return "model_target_clock_mismatch"
    age = clock.get("featureAnchorAgeSeconds")
    if age is not None and _finite_number(age) != issue - anchor:
        return "model_feature_anchor_age_mismatch"
    return None


def _performance_evidence(source, spec):
    backtest = source.get("backtest")
    backtest = backtest if isinstance(backtest, dict) else {}
    return {
        "role": "exposed_historical_development_reconstruction",
        "prospectivelyValidated": False, "calibrated": False,
        "rawModelMaeUgM3": _finite_number(backtest.get(spec["metricField"])),
        "pairedClosedAnchorPersistenceMaeUgM3": _finite_number(backtest.get(spec["baselineMetricField"])),
        "comparisonReference": "original_closed_15_minute_model_anchor",
        "testOriginCount": deepcopy(backtest.get("testOriginCount")),
        "distinctDays": deepcopy(backtest.get("distinctDays")),
        "independentOriginCount": deepcopy(backtest.get("independentOriginCount")),
        "evidenceDoesNotSelectOrReplaceModelOutput": True,
    }


def attach_near_model_outputs(air_window, shadow_forecast, issue_epoch):
    """Copy the existing raw near numbers into their exact forecast fields.

    The raw model value is published regardless of the baseline value, model
    skill, event probabilities or promotion flags. Invalid/missing outputs are
    explicitly unavailable and never replaced by a persistence forecast.
    Freshness/reference metadata continues to identify the baseline used by the
    separate event models; the numeric output retains its own closed anchor.
    """
    result = deepcopy(air_window) if isinstance(air_window, dict) else {}
    source = shadow_forecast if isinstance(shadow_forecast, dict) else {}
    issue = _epoch(issue_epoch)
    source_clock = source.get("forecastClock")
    parent_clock = result.get("forecastClock")
    shared_reason = "invalid_issue_epoch" if issue is None else _clock_reason(source_clock, issue)
    if shared_reason is None and parent_clock is not None:
        shared_reason = _clock_reason(parent_clock, issue)
    if shared_reason is None and source.get("available") is False:
        shared_reason = "raw_model_output_unavailable"
    if shared_reason is None and source.get("enabled") is False:
        shared_reason = "raw_model_output_disabled"
    if shared_reason is None:
        source_anchor = _epoch(source.get("featureAnchorEpoch"))
        if source_anchor != _epoch(source_clock.get("featureAnchorEpoch")):
            shared_reason = "raw_model_feature_anchor_mismatch"
        elif _finite_number(source.get("featureAnchor")) is None or source["featureAnchor"] < 0:
            shared_reason = "raw_model_feature_reference_invalid"
    if shared_reason is None and source.get("status") in (
            "collecting", "sensor_stale", "incomplete_closed_anchor", "stale_closed_anchor", "unavailable"):
        shared_reason = "raw_model_output_unavailable"

    for name, spec in _HEADS.items():
        original = result.get(name)
        original = original if isinstance(original, dict) else {}
        head = deepcopy(original)
        baseline = _finite_number(original.get("baselinePoint"))
        # Legacy persistence-only input may omit baselinePoint. Its point is
        # still a comparator only, never a replacement model forecast number.
        if baseline is None and original.get("pointRole") == "persistence_anchor":
            baseline = _finite_number(original.get("point"))
        if baseline is not None and baseline < 0:
            baseline = None
        head["baselinePoint"] = baseline
        head["baselineReference"] = {
            "available": baseline is not None, "point": baseline,
            "role": "persistence_comparator_only",
            "referenceEpoch": deepcopy(original.get("persistenceAnchorEpoch")),
            "referenceRole": deepcopy(original.get("persistenceAnchorRole")),
            "freshnessAdjustment": deepcopy(original.get("freshnessAdjustment")),
        }
        head["baselineUncertainty"] = {
            key: deepcopy(original[key]) for key in _BASELINE_UNCERTAINTY_FIELDS if key in original
        }
        # Persistence-error bands are not residual intervals of this raw model.
        for field in _NUMERIC_UNCERTAINTY_FIELDS:
            head[field] = None
        for field in _UNCERTAINTY_ROLE_FIELDS:
            head[field] = "raw_model_interval_unavailable"
        for field in ("upperTargetCoverage", "finiteSampleRankCoverage"):
            head[field] = None
        for field in ("finiteSampleUpper", "finiteSampleRank", "meanSkillEligible", "peakSkillEligible"):
            head[field] = False
        head["calibrationState"] = "not_established"
        # Legacy promotion tags describe the superseded persistence/overlay
        # contract and cannot qualify this raw-model publication.
        for field in ("deploymentSkill", "qualification", "qualificationPolicy"):
            head.pop(field, None)
        point = _finite_number(source.get(spec["sourceField"]))
        reason = shared_reason
        if reason is None and point is None:
            reason = "raw_model_number_unavailable_or_invalid"
        head.update({
            "available": reason is None, "point": point if reason is None else None,
            "pointRole": "experimental_model_output", "forecastState": "model_output",
            "pointApproximate": True, "peakApproximate": False,
            "modelVersion": spec["modelVersion"], "numericalPolicyIdentifier": VERSION,
            "method": spec["method"], "experimental": True,
            "qualified": False, "validated": False, "prospectivelyValidated": False,
            "calibrated": False, "usedForDecision": False, "usedForComparison": False,
            "confidence": "Predictive accuracy unproven",
            "rangeRole": "raw_model_interval_unavailable",
            "uncertaintyMethod": "No residual interval fitted for the published raw model output",
            "reason": reason,
            "headline": ("+90-minute model output" if name == "arrival"
                         else "+90..210-minute mean model output") if reason is None
                        else "Model forecast unavailable",
            "modelFeatureAnchor": deepcopy(source.get("featureAnchor")),
            "modelFeatureAnchorEpoch": deepcopy(source.get("featureAnchorEpoch")),
            "performanceEvidence": _performance_evidence(source, spec),
            "modelOutput": {
                "version": VERSION, "targetName": spec["targetName"],
                "sourcePath": "shadowForecast." + spec["sourceField"],
                "rawOutputUgM3": point, "outputPublishedWithoutArithmetic": reason is None,
                "sourceModelVersion": deepcopy(source.get("modelVersion") or source.get("prospectiveModelVersion")),
                "rawDeltaDiagnosticUgM3": _finite_number(source.get(spec["deltaField"])),
                "featureAnchorPm25UgM3": deepcopy(source.get("featureAnchor")),
                "featureAnchorEpoch": deepcopy(source.get("featureAnchorEpoch")),
                "trainingOriginCount": deepcopy(source.get(spec["trainingField"])),
                "featureCount": deepcopy(source.get("featureCount")),
                "sourceForecastClock": deepcopy(source_clock),
                "eligibilityDoesNotSelectNumericalOutput": True,
                "baselineDoesNotReplaceNumericalOutput": True,
            },
        })
        target_start = (_epoch(source_clock.get("arrivalTargetEpoch"))
                        if isinstance(source_clock, dict) else None)
        target_end = (target_start if name == "arrival" else
                      _epoch(source_clock.get("windowEndEpoch"))
                      if isinstance(source_clock, dict) else None)
        head["candidateForecast"] = {
            "available": reason is None, "mean": point if reason is None else None,
            "point": point if reason is None else None, "prediction": point if reason is None else None,
            "modelVersion": spec["modelVersion"], "numericalPolicyIdentifier": VERSION,
            "forecastedAtEpoch": _epoch(source_clock.get("forecastIssuedEpoch")) if isinstance(source_clock, dict) else None,
            "forecastIssuedEpoch": _epoch(source_clock.get("forecastIssuedEpoch")) if isinstance(source_clock, dict) else None,
            "startEpoch": target_start, "endEpoch": target_end,
            "pointRole": "original_issued_raw_model_output", "sourcePath": "shadowForecast." + spec["sourceField"],
            "modelFeatureAnchor": deepcopy(source.get("featureAnchor")),
            "featureAnchorEpoch": deepcopy(source.get("featureAnchorEpoch")),
            "reason": reason, "experimental": True, "prospectivelyValidated": False,
        }
        if reason is None:
            head["forecastClock"] = deepcopy(source_clock)
            head["forecastIssuedEpoch"] = issue
            if name == "arrival":
                head["expectedEpoch"] = source_clock["arrivalTargetEpoch"]
            else:
                head["startEpoch"] = source_clock["windowStartEpoch"]
                head["endEpoch"] = source_clock["windowEndEpoch"]
        result[name] = head
    result["modelOutputPublication"] = {
        "version": VERSION, "fixedMapping": {s["targetName"]: "shadowForecast." + s["sourceField"] for s in _HEADS.values()},
        "forecastIssueEpoch": issue, "numericalPerformanceSelection": False,
        "persistenceFallback": False, "eventThresholdSubstitution": False,
    }
    return result


def attach_ride_extrema_output(air_window, shadow_forecast, issue_epoch):
    """Publish paired learned ride minimum/maximum without changing any mean.

    Extrema are a separate forecast target, never a mean confidence interval.
    Only output presence and the issue/target association are checked. Model
    performance, baseline, and mean cannot alter or select the two numbers.
    """
    result = deepcopy(air_window)
    trail = result.setdefault("trail", {})
    source = (shadow_forecast or {}).get("trailExtrema") or {}
    issue = _epoch(issue_epoch)
    reason = "invalid_issue_epoch" if issue is None else _clock_reason(source.get("forecastClock"), issue)
    if reason is None:
        reason = _clock_reason(result.get("forecastClock"), issue)
    low = _finite_number(source.get("low"))
    high = _finite_number(source.get("high"))
    if reason is None and source.get("available") is not True:
        reason = source.get("reason") or "ride_extrema_model_unavailable"
    if reason is None and (low is None or high is None):
        reason = "ride_extrema_output_missing_or_malformed"
    output = deepcopy(source)
    output.update({"available": reason is None,
                   "low": low if reason is None else None,
                   "high": high if reason is None else None,
                   "reason": reason, "role": "predicted_window_minimum_maximum",
                   "outputPublishedWithoutArithmetic": reason is None,
                   "numericalSelection": False, "baselineSubstituted": False,
                   "meanAppliedToRange": False, "isMeanPredictionInterval": False})
    trail["rideExtrema"] = output
    publication = result.setdefault("modelOutputPublication", {})
    publication["rideExtremaSource"] = "shadowForecast.trailExtrema"
    return result
