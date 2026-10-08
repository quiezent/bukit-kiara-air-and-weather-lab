"""Publish the fixed fresh-sequence model heads without selecting numbers."""
from copy import deepcopy
import math
import model_output_contracts
from forecast_clock import TARGET_VERSION

VERSION = 'direct_fresh_sensor_model_outputs_v1'
_HEADS = {'arrival': ('arrivalPoint', 'point90'), 'trail': ('trailMeanPoint', 'mean90_210')}
_OLD_UNCERTAINTY = ('rangeLow', 'rangeHigh', 'rawRangeLow', 'rawRangeHigh', 'rawUpper90',
                    'upper90', 'decisionUpper', 'peak', 'peakUpper', 'peakRangeLow',
                    'peakRangeHigh', 'projectedPeak', 'decisionPeakUpper')

def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None

def attach_outputs(air_window, source, issue_epoch):
    result = deepcopy(air_window) if isinstance(air_window, dict) else {}
    source = source if isinstance(source, dict) else {}
    issue = int(issue_epoch)
    clock = source.get('forecastClock')
    clock = clock if isinstance(clock, dict) else {}
    parent_clock = result.get('forecastClock')
    reason = None
    if source.get('available') is not True:
        reason = source.get('reason') or 'fresh_model_unavailable'
    elif (clock.get('targetVersion') != TARGET_VERSION
          or clock.get('featureAnchorEpoch') != issue
          or clock.get('featureAnchorAgeSeconds') != 0
          or clock.get('forecastIssuedEpoch') != issue
          or clock.get('arrivalTargetEpoch') != issue + 5400
          or clock.get('windowStartEpoch') != issue + 5400
          or clock.get('windowEndEpoch') != issue + 12600):
        reason = 'fresh_model_target_clock_mismatch'
    if reason is None and isinstance(parent_clock, dict) and (
            parent_clock.get('forecastIssuedEpoch') != issue
            or parent_clock.get('arrivalTargetEpoch') != issue + 5400
            or parent_clock.get('windowStartEpoch') != issue + 5400
            or parent_clock.get('windowEndEpoch') != issue + 12600):
        reason = 'fresh_model_parent_clock_mismatch'
    if reason is None:
        result['forecastClock'] = deepcopy(clock)
    for name, (field, target) in _HEADS.items():
        head = deepcopy(result.get(name) or {})
        for key in ('modelSelection', 'deploymentSkill', 'qualification', 'qualificationPolicy',
                    'candidateForecast', 'rideExtrema'):
            head.pop(key, None)
        for key in model_output_contracts._NUMERIC_UNCERTAINTY_FIELDS:
            head[key] = None
        for key in model_output_contracts._UNCERTAINTY_ROLE_FIELDS:
            head[key] = 'raw_model_interval_unavailable'
        for key in ('finiteSampleUpper', 'finiteSampleRank', 'meanSkillEligible', 'peakSkillEligible'):
            head[key] = False
        for key in ('upperTargetCoverage', 'finiteSampleRankCoverage'):
            head[key] = None
        head['calibrationState'] = 'not_established'
        head['modelEvidence'] = {}
        head['rainModel'] = {}
        head['peakApproximate'] = False
        head['headline'] = '+90-minute forecast' if name == 'arrival' else '+90..210-minute mean forecast'
        fresh_reference = _number(source.get('freshReferencePm25'))
        if fresh_reference is not None:
            head['baselinePoint'] = fresh_reference
            head['persistenceAnchorRole'] = 'recent_five_minute_sensor_median_comparator'
            head['persistenceAnchorEpoch'] = (source.get('inputLineage') or {}).get('freshReferenceEpoch')
        value = _number(source.get(field))
        head_reason = reason or ('fresh_model_number_invalid' if value is None else None)
        head.update(available=head_reason is None, point=value if head_reason is None else None,
                    pointRole='raw_model_output', forecastState='model_output',
                    pointApproximate=True, experimental=True, modelVersion=source.get('modelVersion'),
                    numericalPolicyIdentifier=VERSION, forecastIssuedEpoch=issue,
                    expectedEpoch=issue + 5400, forecastClock=deepcopy(clock),
                    startEpoch=issue + 5400,
                    endEpoch=issue + (5400 if name == 'arrival' else 12600),
                    qualified=False, validated=False, prospectivelyValidated=False,
                    calibrated=False, usedForDecision=False, usedForComparison=False,
                    confidence='Predictive accuracy check in progress', reason=head_reason,
                    method='Fresh sensor sequence joint learned model',
                    performanceEvidence=deepcopy(source.get('performanceEvidence') or {}),
                    modelFeatureAnchor=source.get('freshReferencePm25'),
                    modelFeatureAnchorEpoch=issue,
                    uncertaintyMethod='Residual interval unavailable for this model',
                    rawRangeRole='raw_model_interval_unavailable',
                    modelOutput={'version': VERSION, 'targetName': target,
                        'sourcePath': 'freshSensorForecast.' + field,
                        'rawOutputUgM3': value,
                        'outputPublishedWithoutArithmetic': head_reason is None,
                        'sourceModelVersion': source.get('modelVersion'),
                        'featureAnchorEpoch': issue,
                        'sourceForecastClock': deepcopy(clock),
                        'modelIdentity': deepcopy(source.get('modelIdentity')),
                        'training': deepcopy(source.get('training')),
                        'inputLineage': deepcopy(source.get('inputLineage')),
                        'eligibilityDoesNotSelectNumericalOutput': True,
                        'baselineDoesNotReplaceNumericalOutput': True})
        result[name] = head
    low, high = (_number(source.get(field)) for field in ('trailMinimumPoint','trailMaximumPoint'))
    extrema_reason = reason or ('fresh_model_extrema_invalid' if low is None or high is None else None)
    result['trail']['rideExtrema'] = {
        'available': extrema_reason is None,
        'low': low if extrema_reason is None else None,
        'high': high if extrema_reason is None else None,
        'rawLow': low, 'rawHigh': high, 'reason': extrema_reason,
        'modelVersion': source.get('modelVersion'), 'targetName': 'extrema90_210',
        'forecastClock': deepcopy(clock), 'targetResolutionMinutes': 15,
        'interpretation': 'predicted_lowest_highest_bucket_medians_not_uncertainty_interval',
        'crossing': low > high if low is not None and high is not None else None,
        'outputsPublishedWithoutArithmetic': extrema_reason is None,
        'numericalSelection': False, 'persistenceFallback': False,
        'outputSorting': False, 'outputWidening': False, 'meanRecentering': False,
        'inputLineage': deepcopy(source.get('inputLineage')),
        'training': deepcopy(source.get('training')),
        'modelIdentity': deepcopy(source.get('modelIdentity')),
        'prospectivelyValidated': False, 'calibrated': False,
    }
    return result
