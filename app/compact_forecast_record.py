"""Versioned evidence projection for FUTURE dashboard publications only.

This is a forecast ledger, not a saved copy of dashboard presentation HTML/API.
Legacy payloads must be compressed losslessly, never passed through this filter.

Field paths are retained for these consumers:
* forecast_backtest.extract_dashboard: exact clocks, models, points, baselines,
  range endpoints and legacy reference fallbacks for both near/session heads.
* near_term_hazard._clock_and_features: clock plus ten local input features.
* daily/*_issued_targets.provenance and session_longterm replay/issued audit:
  reference clocks, input policy, training cutoffs, weather source clocks/hashes,
  usedWeatherFeatureValues (including the CAMS-used test), fit completion.
* first20/cycling scoring: keep their complete small issued prediction objects.
  Frozen scorers validate nested modelIdentity/selectedPolicy equality and hashes;
  replacing these with a bare hash would silently invalidate genuine evidence.

Only repeated explanatory text, historical accuracy reports, model diagnostics
and full weather series are omitted. Weather source hashes/epochs resolve to
the separately retained original weather archive. No rounding, time changes,
probability recomputation, publication claims or database writes occur here.
"""
from __future__ import annotations
from copy import deepcopy

VERSION = "dashboard_forecast_ledger_v6_arrival_change_outputs"

NEAR_FIELDS = (
    "available", "point", "baselinePoint", "rangeLow", "rangeHigh", "rawRangeLow", "rawRangeHigh",
    "peak", "peakUpper", "peakRangeLow", "peakRangeHigh", "modelVersion", "method", "pointRole",
    "forecastState", "persistenceAnchorRole", "persistenceAnchorEpoch", "forecastIssuedEpoch",
    "forecastClock", "freshnessAdjustment", "source", "uncertaintyMethod", "modelSelection",
    "numericalPolicyIdentifier", "candidateForecast", "modelOutput", "performanceEvidence",
    "baselineReference", "baselineUncertainty", "qualified", "validated", "usedForDecision",
    "rideExtrema",
)
PARTICLE_FIELDS = (
    "available", "point", "mean", "baselinePoint", "sensorAnchor", "rangeLow", "rangeHigh",
    "rawRangeLow", "rawRangeHigh", "modelVersion", "pointRole", "pointApproximate", "forecastState",
    "source", "persistenceAnchorRole", "persistenceAnchorEpoch", "freshnessAdjustment",
    "inputAvailabilityPolicy", "inputSnapshotReceivedEpoch", "sensorWatermarkEpoch",
    "featureSourceMaxEpoch", "sensorReferenceEpoch", "publicationEpoch", "forecastIssuedEpoch",
    "forecastedAtEpoch", "originEpoch", "weatherSourceClock", "sourceSnapshots",
    "dbSnapshotOpenedEpoch", "forecastGeneratedEpoch",
    "candidateForecast", "qualificationPolicy", "uncertaintyMethod", "validated", "usedForDecision", "usedForComparison",
    "numericalPolicyIdentifier", "modelOutputPolicy", "performanceEvidence",
)
SELECTION_FIELDS = (
    "version", "selectedPolicy", "selectedModelVersion", "candidateModelVersion", "selectionReason",
    "appliedToPrimaryForecast", "skillGatePassed", "gatesAppliedToSelection", "prospectivelyValidated",
    "forecastIssuedEpoch", "targetStartEpoch", "targetEndEpoch", "evidenceScope",
)
MODEL_FIELDS = (
    "applied", "experimental", "prospectivelyValidated", "featureVersion", "weatherDiagnosticFeatureVersion",
    "numericalPolicy", "featureColumns", "parameters", "trainingCount", "trainingDistinctTargetDays",
    "trainingCutoffEpoch", "trainingLatestTargetEndEpoch", "trainingFeatureLabelSha256",
    "trainingWeatherRevision", "trainingCamsRevision", "fittedAtEpoch", "fitSeconds",
    "usedWeatherFeatureValues", "liveInputAvailabilityPolicy",
    "checkpointSha256", "trainingMetadataSha256", "dailyPolicySha256",
    "rawMeanUgM3", "deadbandUgM3", "deadbandApplied", "eventThresholdUgM3", "numericDeadbandUgM3", "legacyModelVersion", "queryTensorSha256",
    "materializedSnapshotSha256", "forecastGeneratedEpoch", "sequenceSourceMaxEpoch",
    "pastPmValidSteps", "tensorShapes", "developmentScope", "queryMatchesEvaluatedClock",
    "featureDomain", "movingWindowSkillVerified",
    "liveRuntimeRevision", "liveRuntimeRevisionSha256", "inputRevisionInvalidatesPreviousIssuedSkill",
)
WEATHER_META_FIELDS = (
    "version", "forecastIssuedEpoch", "windowStartEpoch", "windowEndEpoch", "source", "fetchedEpoch",
    "payloadHash", "maxWeatherAgeSeconds", "errors", "missingOptionalUnits", "missingTargetMeteorology",
)
WEATHER_WINDOW_FIELDS = (
    "available", "startEpoch", "endEpoch", "sourceFetchedEpoch", "forecastIssuedEpoch", "modeledSession", "weatherCoverage",
    "usedForParticleForecast", "pmInputSourceAligned", "precipitationMm", "precipitationProbabilityMax",
    "windSpeedMean", "windDirectionMean", "windGustsMax", "temperatureMean", "relativeHumidityMean",
)
REGIONAL_FIELDS = (
    "available", "modelVersion", "featureVersion", "sourceIssueEpoch", "appliedToPrimaryForecast",
    "invalidArchivedRuns", "fetchedEpoch", "recordedEpoch", "availableEpoch", "ageSeconds",
    "payloadSha256", "requestedCellCount", "uniqueReturnedModelCellCount", "featureValues",
)


def _pick(value, fields):
    """Preserve absent versus explicit null values; never mutate the input."""
    if not isinstance(value, dict):
        return deepcopy(value)
    return {key: deepcopy(value[key]) for key in fields if key in value}


def _child(result, original, key, fields):
    if key in original:
        result[key] = _pick(original[key], fields)


def _particle(value):
    if not isinstance(value, dict):
        return deepcopy(value)
    result = _pick(value, PARTICLE_FIELDS)
    _child(result, value, "modelSelection", SELECTION_FIELDS)
    for name in ("weatherSessionModel", "sessionDeltaModel", "directionalModel", "patchtstModel"):
        if name not in value:
            continue
        result[name] = _pick(value[name], MODEL_FIELDS)
        if isinstance(value[name], dict) and isinstance(result[name], dict):
            _child(result[name], value[name], "weatherMetadata", WEATHER_META_FIELDS)
    if "rainContext" in value:
        result["rainContext"] = _pick(value["rainContext"], ("available", "fetchedEpoch"))
        if isinstance(value["rainContext"], dict) and isinstance(result["rainContext"], dict):
            _child(result["rainContext"], value["rainContext"], "metadata", WEATHER_META_FIELDS)
    if "experimentalXgboostFallback" in value:
        result["experimentalXgboostFallback"] = _pick(value["experimentalXgboostFallback"],
            ("available", "reason", "modelVersion", "applied"))
    if "experimentalPatchtstFallback" in value:
        result["experimentalPatchtstFallback"] = _pick(value["experimentalPatchtstFallback"],
            ("reason", "modelVersion", "applied", "selectedModelVersion", "fitStatus"))
    return result


def compact_dashboard_record(result, build=None):
    """Project a current result/old-shaped payload into a future issue ledger.

    The writer owns actual publication timestamps, outer SHA256 and codec. This
    function deliberately does not invent these or change a model selection.
    """
    if not isinstance(result, dict):
        raise TypeError("dashboard result must be a dictionary")
    record = _pick(result, ("forecastIssuedEpoch", "current", "dashboardBuild", "inputProvenance", "validationStatus", "freshSensorForecast", "arrivalModelForecast"))
    record["archiveRecordVersion"] = VERSION
    if build is not None:
        record["dashboardBuild"] = build
    if "forecastPolicy" in result:
        policy = result["forecastPolicy"]
        # Identity/selection flags are scalars. The nested review prose and full
        # historical decision reports are not inputs to a forecast scorer.
        record["forecastPolicy"] = ({k: deepcopy(v) for k, v in policy.items()
                                     if not isinstance(v, (dict, list))}
                                    if isinstance(policy, dict) else deepcopy(policy))
        if isinstance(policy, dict):
            _child(record["forecastPolicy"], policy, "windowForecastModels", tuple((policy.get("windowForecastModels") or {}).keys()))
    if "dataCoverage" in result:
        record["dataCoverage"] = _pick(result["dataCoverage"],
            ("gapThresholdSeconds", "gapCount", "latestObservationEpoch", "latestObservationAgeSeconds", "missingDataFilled"))
    if "airWindow" in result:
        air = result["airWindow"]
        if not isinstance(air, dict):
            record["airWindow"] = deepcopy(air)
        else:
            trimmed = _pick(air, (
                "available", "current15", "currentPartial15", "currentFast", "forecastClock", "forecastPolicy",
                "closedBucketEndEpoch", "analysisBucketEndEpoch", "analysisBucketMinutes", "change30", "change60",
                "fastSlowGap", "pm10ChangePct30", "state", "sustainedImprovement", "fastRise", "minimumSinceEvent",
                "firstCrossingEventForecast", "cyclingWindowForecast", "shortHorizonChange", "experimentalRapidChangeRisk",
                "arrivalChangeForecast",
            ))
            _child(trimmed, air, "particleMix", ("fineShare", "coarseParticles"))
            for name in ("arrival", "trail"):
                _child(trimmed, air, name, NEAR_FIELDS)
                if isinstance(air.get(name), dict) and isinstance(trimmed.get(name), dict):
                    _child(trimmed[name], air[name], "modelSelection", SELECTION_FIELDS)
            _child(trimmed, air, "regionalPmInputs", REGIONAL_FIELDS)
            record["airWindow"] = trimmed
    if "windows" in result:
        windows = result["windows"]
        if not isinstance(windows, dict):
            record["windows"] = deepcopy(windows)
        else:
            record["windows"] = {}
            for name, value in windows.items():
                if not isinstance(value, dict):
                    record["windows"][name] = deepcopy(value)
                    continue
                window = _pick(value, ("startEpoch", "endEpoch", "forecastIssuedEpoch", "modeledSession"))
                if "particleForecast" in value:
                    window["particleForecast"] = _particle(value["particleForecast"])
                _child(window, value, "weatherForecast", WEATHER_WINDOW_FIELDS)
                record["windows"][name] = window
    return record


# Descriptive alias for callers that already use the archive's naming pattern.
compact_forecast_record = compact_dashboard_record
