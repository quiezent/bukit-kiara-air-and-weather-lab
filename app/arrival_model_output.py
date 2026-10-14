"""Copy the fixed dedicated arrival learner's physical output literally."""
from copy import deepcopy
import math

from forecast_clock import TARGET_VERSION
import model_output_contracts

VERSION = "direct_dedicated_arrival_model_output_v1"


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def attach_numeric_output(air_window, source, issue_epoch):
    """Publish one predetermined model, with no numeric selection or fallback."""
    result = deepcopy(air_window) if isinstance(air_window, dict) else {}
    source = source if isinstance(source, dict) else {}
    issue = int(issue_epoch)
    clock = source.get("forecastClock") or {}
    parent = result.get("forecastClock") or {}
    value = _number(source.get("arrivalPoint"))
    reason = None
    if source.get("available") is not True:
        reason = source.get("reason") or "arrival_model_unavailable"
    elif (clock.get("targetVersion") != TARGET_VERSION
          or clock.get("forecastIssuedEpoch") != issue
          or clock.get("featureAnchorEpoch") != issue
          or clock.get("featureAnchorAgeSeconds") != 0
          or clock.get("arrivalTargetEpoch") != issue + 5400
          or parent.get("targetVersion") != TARGET_VERSION
          or parent.get("forecastIssuedEpoch") != issue
          or parent.get("arrivalTargetEpoch") != issue + 5400):
        reason = "arrival_model_target_clock_mismatch"
    elif value is None:
        reason = "arrival_model_number_invalid"
    head = deepcopy(result.get("arrival") or {})
    for key in ("modelSelection", "deploymentSkill", "qualification", "qualificationPolicy", "candidateForecast"):
        head.pop(key, None)
    for key in model_output_contracts._NUMERIC_UNCERTAINTY_FIELDS:
        head[key] = None
    for key in model_output_contracts._UNCERTAINTY_ROLE_FIELDS:
        head[key] = "raw_model_interval_unavailable"
    for key in ("finiteSampleUpper", "finiteSampleRank", "meanSkillEligible", "peakSkillEligible"):
        head[key] = False
    for key in ("finiteSampleRankCoverage", "upperTargetCoverage"):
        head[key] = None
    reference = _number(source.get("freshReferencePm25"))
    head.update(
        available=reason is None, point=value if reason is None else None,
        reason=reason, baselinePoint=reference, pointRole="raw_model_output",
        forecastState="model_output", modelVersion=source.get("modelVersion"),
        numericalPolicyIdentifier=VERSION, forecastIssuedEpoch=issue,
        expectedEpoch=issue + 5400, startEpoch=issue + 5400, endEpoch=issue + 5400,
        forecastClock=deepcopy(clock), modelFeatureAnchor=reference,
        modelFeatureAnchorEpoch=issue, pointApproximate=True,
        headline="+90-minute forecast", method="Dedicated learned arrival model",
        qualified=False, validated=False, prospectivelyValidated=False,
        calibrated=False, calibrationState="not_established", usedForDecision=False,
        usedForComparison=False, confidence="Predictive accuracy check in progress",
        persistenceAnchorRole="recent_five_minute_sensor_median_comparator",
        persistenceAnchorEpoch=(source.get("inputLineage") or {}).get("freshReferenceEpoch"),
        performanceEvidence=deepcopy(source.get("performanceEvidence") or {}),
        modelOutput={
            "version": VERSION, "targetName": "point90",
            "sourcePath": "arrivalModelForecast.arrivalPoint", "rawOutputUgM3": value,
            "outputPublishedWithoutArithmetic": reason is None,
            "sourceModelVersion": source.get("modelVersion"), "featureAnchorEpoch": issue,
            "sourceForecastClock": deepcopy(clock),
            "modelIdentity": deepcopy(source.get("modelIdentity")),
            "training": deepcopy(source.get("training")),
            "inputLineage": deepcopy(source.get("inputLineage")),
            "eligibilityDoesNotSelectNumericalOutput": True,
            "baselineDoesNotReplaceNumericalOutput": True,
        },
    )
    result["arrival"] = head
    return result
