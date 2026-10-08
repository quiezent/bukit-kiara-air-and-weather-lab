"""Keep exact session baselines separate from unqualified learned estimates.

This module never fits, loads outcomes, writes archives, or changes a target.
The caller supplies any genuinely issued evidence; absent evidence cannot
promote a learned point. The 20 ug/m3 event threshold is not a numeric deadband.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import math
import statistics

VERSION = "canonical_session_issued_skill_gate_v1"
BASELINE_MODEL_VERSION = "fresh_session_persistence_qualification_v1"
KL = timezone(timedelta(hours=8))
MINIMUM_ISSUED_DAYS = 14
MINIMUM_RELATIVE_MAE_GAIN = 0.10
MINIMUM_ABSOLUTE_MAE_GAIN = 2.0
MINIMUM_POSITIVE_DAY_FRACTION = 0.70


def _number(value):
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def candidate_scope(issue, start, end, window_key="afternoon"):
    """Audit the actual target and lead, without adjusting either clock.

    Raw estimates are permitted only for canonical exact targets within the
    pooled training lead range. Operational promotion additionally requires a
    qualified issued cohort. An arbitrary issue clock is never relabelled as
    the fixed historical evaluation clock.
    """
    result = {"policyVersion": VERSION, "windowKey": window_key,
              "canonicalTarget": False, "trainedLead": False,
              "queryMatchesEvaluatedClock": False, "movingWindowSkillVerified": False,
              "trainingLeadRangeMinutes": [105, 1380], "blockedReasons": []}
    try:
        issue, start, end = int(issue), int(start), int(end)
        issued, target = (datetime.fromtimestamp(t, KL) for t in (issue, start))
    except (TypeError, ValueError, OverflowError, OSError):
        result["blockedReasons"] = ["invalid_exact_session_target"]
        return result
    clock = {"morning": 9, "afternoon": 14}.get(window_key)
    target_date_valid = target.date() in (issued.date(), issued.date() + timedelta(days=1))
    canonical = bool(clock is not None and end-start == 7200 and target_date_valid
                     and target.hour == clock and target.minute == 0 and target.second == 0
                     and start % 900 == 0 and end % 900 == 0)
    lead = (start-issue)/60
    scope = "same_day_" + window_key if target.date() == issued.date() else "next_day_" + window_key
    # Exact historical clocks remain descriptive, not retrospective qualification.
    expected = ((12, 0, 0) if target.date() == issued.date() else (18, 5, 0)) if window_key == "afternoon" else None
    evaluated = canonical and expected is not None and (issued.hour, issued.minute, issued.second) == expected
    result.update(canonicalTarget=canonical, trainedLead=105 <= lead <= 1380,
                  actualLeadMinutes=lead, evidenceScope=scope,
                  forecastIssuedEpoch=issue, targetStartEpoch=start, targetEndEpoch=end,
                  queryMatchesEvaluatedClock=evaluated)
    if not canonical:
        result["blockedReasons"].append("outside_canonical_exact_session_target")
    if not 105 <= lead <= 1380:
        result["blockedReasons"].append("outside_trained_105min_to_23hour_actual_lead")
    result["rawCandidatePermitted"] = not result["blockedReasons"]
    return result


def _qualification(evidence, version, numerical_policy, scope):
    """Only matched, independent, originally issued outcomes can pass.

    Training rows, development replay metrics, other versions and fabricated
    success booleans are not evidence of this forecast's operational skill.
    """
    evidence = evidence or {}
    checks = {}
    tolerance = _number(evidence.get("issueClockToleranceSeconds"))
    issue_clock = _number(evidence.get("matchedIssueLocalSeconds"))
    target_clock = _number(evidence.get("matchedTargetStartLocalSeconds"))
    duration = _number(evidence.get("matchedDurationSeconds"))
    lead = _number(evidence.get("matchedActualLeadSeconds"))
    query_issue = scope.get("forecastIssuedEpoch")
    query_start, query_end = scope.get("targetStartEpoch"), scope.get("targetEndEpoch")
    clocks = bool(all(v is not None for v in (tolerance, issue_clock, target_clock, duration, lead,
                                             query_issue, query_start, query_end))
                  and 0 <= tolerance <= 60 and tolerance == int(tolerance)
                  and 0 <= issue_clock < 86400 and 0 <= target_clock < 86400
                  and abs((query_issue+28800)%86400-issue_clock) <= tolerance
                  and (query_start+28800)%86400 == target_clock
                  and query_end-query_start == duration
                  and abs(query_start-query_issue-lead) <= tolerance)
    checks["actual_issue_target_and_lead_matched"] = clocks
    provenance = bool(isinstance(numerical_policy, str) and numerical_policy
                      and evidence.get("asIssuedBeforeOutcome") is True
                      and evidence.get("independentOrigins") is True
                      and evidence.get("incompleteTargetsScored") is False
                      and evidence.get("candidateModelVersion") == version
                      and evidence.get("numericalPolicyIdentifier") == numerical_policy
                      and evidence.get("scope") == scope.get("evidenceScope")
                      and evidence.get("targetClockMatched") is True
                      and evidence.get("issueClockMatched") is True
                      and evidence.get("validationMode") == "as_issued_same_session_target_lead_and_local_issue_clock")
    checks["genuine_matched_issued_provenance"] = provenance
    count, days = (_number(evidence.get(k)) for k in ("count", "distinctDays"))
    checks["independent_issued_days"] = bool(count is not None and days is not None
                                           and count == int(count) and days == int(days)
                                           and count >= days >= MINIMUM_ISSUED_DAYS)
    mae, baseline = (_number(evidence.get(k)) for k in ("mae", "persistenceMae"))
    checks["mae_gain"] = bool(mae is not None and baseline is not None and mae >= 0 and baseline > 0
                               and mae <= baseline*(1-MINIMUM_RELATIVE_MAE_GAIN)
                               and baseline-mae >= MINIMUM_ABSOLUTE_MAE_GAIN)
    for check, model_key, base_key in (("rmse_not_worse", "rmse", "persistenceRmse"),
                                       ("p90_not_worse", "p90AbsoluteError", "persistenceP90AbsoluteError")):
        candidate, reference = (_number(evidence.get(k)) for k in (model_key, base_key))
        checks[check] = bool(candidate is not None and reference is not None
                            and 0 <= candidate <= reference)
    positive = _number(evidence.get("positiveDayFraction"))
    checks["consistent_daily_gain"] = bool(positive is not None and MINIMUM_POSITIVE_DAY_FRACTION <= positive <= 1)
    return all(checks.values()), checks


def baseline_session_result(rows, issue, start, end, window_key="afternoon", reason=None,
                            candidate_model_version=None):
    """Return a fresh same-issue baseline for the caller's unchanged exact card."""
    issue = int(issue)
    scope = candidate_scope(issue, start, end, window_key)
    if _number(start) is None or _number(end) is None or int(end)-int(start) != 7200 or int(start) <= issue:
        return {"available": False, "reason": "invalid_future_exact_session_target",
                "modelVersion": BASELINE_MODEL_VERSION, "qualificationPolicy": scope}
    observations = []
    for source in rows:
        row = dict(source)
        epoch, value = _number(row.get("epoch")), _number(row.get("pm02"))
        if epoch is not None and value is not None and value >= 0 and issue-300 < epoch <= issue:
            observations.append((epoch, value))
    if not observations or issue-max(e for e, _ in observations) > 240:
        return {"available": False, "reason": "stale_or_missing_fresh_sensor_reference",
                "modelVersion": BASELINE_MODEL_VERSION,
                "candidateForecast": {"available": False, "modelVersion": candidate_model_version,
                                      "reason": reason or "candidate_unavailable"},
                "qualificationPolicy": scope}
    anchor = float(statistics.median(v for _, v in observations))
    reference = int(max(e for e, _ in observations))
    value = {"available": True, "modelVersion": candidate_model_version or BASELINE_MODEL_VERSION,
             "sensorAnchor": anchor, "sensorReferenceEpoch": reference,
             "sensorReferenceCount": len(observations), "mean": anchor, "prediction": anchor,
             "forecastedAtEpoch": issue, "originEpoch": issue//900*900,
             "startEpoch": int(start), "endEpoch": int(end),
             "candidateForecast": {"available": False, "modelVersion": candidate_model_version,
                                   "reason": reason or (scope["blockedReasons"][0] if scope["blockedReasons"] else "candidate_unavailable")},
             "modelEvidence": {}, "qualificationPolicy": scope,
             "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median",
             "remainingLeadHours": (int(start)-issue)/3600,
             "target": "overlap_duration_weighted_mean_of_complete_15_minute_raw_sensor_medians_over_exact_card"}
    return apply_session_prediction_policy(value, window_key)


def apply_session_prediction_policy(value, window_key="afternoon", evidence=None):
    """Publish qualified learned mean or persistence, retaining the raw candidate."""
    result = copy.deepcopy(value)
    if not result.get("available"):
        return result
    scope = candidate_scope(result.get("forecastedAtEpoch"), result.get("startEpoch"), result.get("endEpoch"), window_key)
    candidate = dict(result.get("candidateForecast") or {})
    candidate_version = candidate.get("modelVersion") or result.get("modelVersion")
    raw = _number(candidate.get("mean", result.get("mean")))
    anchor = _number(result.get("sensorAnchor"))
    issue = _number(result.get("forecastedAtEpoch"))
    reference = _number(result.get("sensorReferenceEpoch"))
    reference_count = _number(result.get("sensorReferenceCount"))
    watermark = _number(result.get("sensorWatermarkEpoch"))
    valid_reference = bool(issue is not None and reference is not None and reference_count is not None
                           and 0 <= issue-reference <= 240 and reference_count >= 1
                           and reference_count == int(reference_count)
                           and (watermark is None or reference <= watermark <= issue))
    if anchor is None or anchor < 0 or not valid_reference:
        return {"available": False, "modelVersion": BASELINE_MODEL_VERSION,
                "reason": "missing_stale_or_future_same_issue_persistence_reference"}
    candidate_clocks_match = bool(all(candidate.get(k) is not None and candidate.get(k) == result.get(k)
                                      for k in ("forecastedAtEpoch", "startEpoch", "endEpoch")))
    permitted = bool(scope.get("rawCandidatePermitted") and candidate.get("available", True)
                     and raw is not None and raw >= 0 and candidate_clocks_match)
    candidate.update(available=permitted, modelVersion=candidate_version,
                     appliedToPrimaryForecast=False, prospectivelyValidated=False,
                     pointRole="raw_unqualified_session_mean", numericDeadbandApplied=False,
                     eventThresholdUgM3=20, eventThresholdAppliedToPoint=False,
                     scopeQualification=scope)
    if permitted:
        candidate.update(mean=raw, prediction=raw, reason=None)
    else:
        candidate.update(mean=None, prediction=None,
                         reason=candidate.get("reason") or (scope["blockedReasons"][0] if scope["blockedReasons"] else
                                "candidate_issue_or_target_clock_mismatch" if not candidate_clocks_match else "invalid_raw_candidate"))
    evidence = evidence if evidence is not None else result.get("modelEvidence") or {}
    numerical_policy = candidate.get("numericalPolicyIdentifier")
    qualified, checks = _qualification(evidence, candidate_version, numerical_policy, scope)
    qualified = bool(permitted and qualified)
    candidate["appliedToPrimaryForecast"] = qualified
    candidate["prospectivelyValidated"] = qualified
    selected = raw if qualified else anchor
    selected_version = candidate_version if qualified else BASELINE_MODEL_VERSION
    result.update(mean=selected, prediction=selected, modelVersion=selected_version,
                  pointRole="qualified_session_mean" if qualified else "persistence_anchor",
                  forecastState="qualified_learned_session_mean" if qualified else "same_issue_persistence",
                  source="qualified_session_model" if qualified else "recent_sensor_persistence",
                  method="Qualified learned exact-session mean" if qualified else "Same-issue sensor persistence · learned estimate unqualified",
                  rawRangeLow=None, rawRangeHigh=None,
                  rangeRole="unavailable_issued_coverage_not_qualified",
                  uncertaintyMethod="no_calibrated_interval_published",
                  usedForDecision=True, usedForComparison=False, candidateForecast=candidate,
                  qualificationPolicy={**scope, "gateChecks": checks, "qualified": qualified},
                  sourceCaveat="The raw learned estimate is experimental; operational qualification requires matched originally issued outcomes.",
                  modelSelection={"version": VERSION,
                                  "selectedPolicy": "qualified_issued_candidate" if qualified else "same_issue_persistence",
                                  "selectedModelVersion": selected_version, "candidateModelVersion": candidate_version,
                                  "candidateAvailable": permitted, "appliedToPrimaryForecast": qualified,
                                  "skillGatePassed": qualified, "gatesAppliedToSelection": True,
                                  "prospectivelyValidated": qualified,
                                  "selectionReason": "matched_issued_skill_gate_passed" if qualified else "genuine_issued_qualification_not_passed",
                                  "evidenceScope": scope.get("evidenceScope"), "gateChecks": checks,
                                  "forecastIssuedEpoch": result.get("forecastedAtEpoch"),
                                  "targetStartEpoch": result.get("startEpoch"), "targetEndEpoch": result.get("endEpoch")})
    for key in ("patchtstModel", "weatherSessionModel"):
        if result.get(key):
            result[key].update(applied=qualified, candidateComputed=permitted,
                               prospectivelyValidated=qualified, operationalQualificationPassed=qualified)
    return result
