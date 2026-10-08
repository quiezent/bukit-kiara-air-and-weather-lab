"""Fixed model output contracts and exact query features; no point selector.

Evidence and training-domain comparisons describe a returned number. They
never replace, shrink, rebase, round or suppress a finite model output.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd

import session_prediction_policy as legacy_policy

VERSION = "fixed_session_model_output_v3"
QUERY_VERSION = "original_exact_features_without_training_lead_selector_v3"


def _number(value):
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def target_shape_reason(issue, start, end):
    values = [_number(value) for value in (issue, start, end)]
    if any(value is None or value != int(value) for value in values):
        return "invalid_integer_session_clocks"
    issue, start, end = map(int, values)
    if start <= issue or end-start != 7200:
        return "require_future_exact_two_hour_target"
    return None


def domain_metadata(issue, start, end, window_key):
    descriptive = legacy_policy.candidate_scope(issue, start, end, window_key)
    warnings = descriptive.pop("blockedReasons", [])
    descriptive.pop("rawCandidatePermitted", None)
    descriptive.update(policyVersion=VERSION, extrapolation=not descriptive.get("trainedLead", False),
                       domainWarnings=warnings, domainAppliedToNumber=False,
                       qualificationAppliedToNumber=False)
    return descriptive


def unavailable_model_output(issue, start, end, model_version, reason, window_key):
    issued = _number(issue)
    issued = int(issued) if issued is not None and issued == int(issued) else None
    return {"available": False, "mean": None, "prediction": None,
            "modelVersion": model_version, "forecastedAtEpoch": issued,
            "startEpoch": start, "endEpoch": end, "reason": reason,
            "pointRole": "experimental_model_output", "forecastState": "model_output_unavailable",
            "candidateForecast": {"available": False, "mean": None, "prediction": None,
                                  "modelVersion": model_version, "reason": reason,
                                  "forecastedAtEpoch": issued, "startEpoch": start, "endEpoch": end},
            "modelSelection": {}, "usedForDecision": False,
            "modelOutputPolicy": {"version": VERSION, "fixedModel": True,
                                  "numericalSelectionApplied": False, "baselineSubstituted": False,
                                  "alternateModelUsed": False},
            "modelDomain": domain_metadata(issue, start, end, window_key)}


def publish_direct_output(value, window_key):
    """Return literal fixed-model mean; evidence cannot modify this number."""
    result = copy.deepcopy(value)
    issue, start, end = (result.get(key) for key in ("forecastedAtEpoch", "startEpoch", "endEpoch"))
    reason = target_shape_reason(issue, start, end)
    candidate_input = result.get("candidateForecast") or {}
    if any(candidate_input.get(key) is not None and candidate_input[key] != result.get(key)
           for key in ("forecastedAtEpoch", "startEpoch", "endEpoch")):
        reason = "candidate_metadata_clocks_do_not_match_model_output"
    raw = _number(result.get("mean"))
    if not result.get("available") or reason or raw is None:
        return unavailable_model_output(issue, start, end, result.get("modelVersion"),
                                        reason or result.get("reason") or "invalid_model_output", window_key)
    result.update(mean=raw, prediction=raw, pointRole="experimental_model_output", forecastState="model_output",
                  modelSelection={}, modelDomain=domain_metadata(issue, start, end, window_key),
                  modelOutputPolicy={"version": VERSION, "fixedModel": True,
                                     "numericalSelectionApplied": False, "baselineSubstituted": False,
                                     "alternateModelUsed": False, "domainAppliedToNumber": False},
                  usedForDecision=False, prospectivelyValidated=False)
    candidate = dict(result.get("candidateForecast") or {})
    candidate.update(available=True, mean=raw, prediction=raw, modelVersion=result["modelVersion"],
                     pointRole="experimental_model_output", forecastedAtEpoch=int(issue),
                     startEpoch=int(start), endEpoch=int(end), appliedToPrimaryForecast=True,
                     prospectivelyValidated=False, numericDeadbandApplied=False,
                     numericalPolicyIdentifier=result.get("numericalPolicyIdentifier") or candidate.get("numericalPolicyIdentifier"),
                     modelDomain=result["modelDomain"])
    result["candidateForecast"] = candidate
    for key in ("qualificationPolicy", "operationalMean"):
        result.pop(key, None)
    return result


def query_features(source, issue, start, end):
    """Reuse original causal/enrichment math for any supplied future 2h target.

    The old 90-minute/24-hour training-domain guard is descriptive metadata.
    Missing closed features, current reference or covered issued weather still
    makes the model input genuinely unavailable.
    """
    import weather_session_model as learner
    reason = target_shape_reason(issue, start, end)
    if reason:
        raise ValueError(reason)
    issue, start, end = int(issue), int(start), int(end)
    origin = issue//900*900
    frame = source["frame"]
    if frame.empty:
        return None
    stamp = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    if stamp not in frame.index:
        return None
    data = learner.causal.design(frame, source["rows"], source["cams"], source["weather"], issue,
        (start-issue)/60, 120, lag_seconds=issue-origin, only_origins=pd.DatetimeIndex([stamp]))
    data = learner._enrich(data, source["weather"])
    fetched = data["audit"][0]["weatherFetchedEpoch"]
    if fetched is not None and fetched > issue:
        raise ValueError("future_weather_vintage_in_query")
    query = data["x"].iloc[0].copy()
    query.attrs.update(valid=bool(data["valid"][0]), audit=data["audit"][0],
                       weatherDescription=data["weatherDescriptions"][0], issueEpoch=issue,
                       startEpoch=start, endEpoch=end, originEpoch=origin,
                       referenceEpoch=int(data["stamps"][0]), referenceCount=int(data["counts"][0]),
                       directQueryVersion=QUERY_VERSION)
    return query


def patch_query(adapter, snapshot, start, end):
    """Original Patch tensors/audit, with direct rather than lead-gated context."""
    issue, mark = int(snapshot["issueEpoch"]), int(snapshot["sensorWatermarkEpoch"])
    reason = target_shape_reason(issue, start, end)
    if reason:
        raise ValueError(reason)
    start, end = int(start), int(end)
    if any(row["epoch"] > mark for row in snapshot["sensor"]) or mark > issue:
        raise ValueError("future_or_unpublished_sensor_input")
    source = adapter.prepare.Sources(snapshot)
    raw = [(fetched, payload) for fetched, _, payload in snapshot["cams"]]
    digest = hashlib.sha256(json.dumps(raw).encode()).hexdigest()
    cams = adapter.regional._read_runs(adapter.ROOT / "bukit_kiara_air_history.db", "air_quality_forecast_runs", issue,
                                       raw_snapshot=(raw, digest))
    context_source = {"frame": source.frame, "rows": snapshot["sensor"],
                      "weather": source.archive, "cams": cams, "cutoff": issue}
    context = query_features(context_source, issue, start, end)
    if context is None or not context.attrs["valid"]:
        raise RuntimeError("current_exact_causal_context_unavailable")
    fetched = context.attrs["audit"]["weatherFetchedEpoch"]
    past, past_grid, maximum, count = source.past(issue, mark)
    future, future_grid, weather_hash, units = source.future(issue, start, end, fetched)
    reference, origin = int(context.attrs["referenceEpoch"]), int(context.attrs["originEpoch"])
    basic = source.bucket_max.loc[source.bucket_max.index <= pd.Timestamp(origin, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")].iloc[-12:]
    basic_max = float(basic.max())
    if count < 48 or not maximum <= mark <= issue or not basic_max <= mark:
        raise RuntimeError("insufficient_covered_past_history_or_source_exceeds_watermark")
    if not 0 <= issue-reference <= 240 or reference > mark:
        raise RuntimeError("fresh_context_reference_stale_or_unavailable")
    features = context.reindex(adapter.prepare.CONTEXT).to_numpy(dtype=np.float32)
    bundle = {"past": past[None, ...], "future": future[None, ...], "context": features[None, ...],
              "fresh": np.array([float(context["fresh"])]), "valid": np.array([True]),
              "past_epochs": past_grid[None, ...], "future_epochs": future_grid[None, ...]}
    overlap = float(future[:, 13].sum())
    audit = {"issueEpoch": issue, "startEpoch": start, "endEpoch": end,
             "sensorWatermarkEpoch": mark, "sensorReferenceEpoch": reference,
             "sensorReferenceCount": context.attrs["referenceCount"], "fresh": float(context["fresh"]),
             "closedLevel": float(context["closedLevel"]), "featureSourceMaxEpoch": basic_max,
             "sequenceSourceMaxEpoch": maximum, "pastPmValidSteps": count,
             "weatherFetchedEpoch": fetched, "weatherPayloadSha256": weather_hash,
             "camsFetchedEpoch": context.attrs["audit"]["camsFetchedEpoch"],
             "tensorShapes": {key: list(bundle[key].shape) for key in ("past", "future", "context")},
             "contextFeatures": dict(zip(adapter.prepare.CONTEXT, features.astype(float))),
             "weatherUnits": units, "directQueryVersion": QUERY_VERSION,
             "futureTensorMaximumEpoch": int(future_grid[-1]),
             "futureTensorTargetOverlapHours": overlap,
             "futureTensorContainsFullTarget": math.isclose(overlap, (end-start)/3600, abs_tol=1e-6),
             "futureTensorBeyondTargetCoverageHours": max(0., (end-start)/3600-overlap),
             "targetContextWeatherComplete": bool(context.attrs["weatherDescription"]["available"]),
             "futureOutcomeLoaded": False}
    return bundle, audit
