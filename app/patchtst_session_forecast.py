"""Fixed Afternoon PatchTST output; no statistical or persistence selector.

Reuse the original frozen query adapter, recipe and daily CPU artifact. The
independent prospective worker prepares the runtime; requests never fit or
initialize Torch. Original artifacts remain intact. The old numeric deadband
is removed in this new wrapper identity; 20 ug/m3 remains an event threshold.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
from forecast_payload import decode_payload_text
import session_model_output as direct
import observation_provenance as provenance

LEGACY_MODEL_VERSION = "afternoon_patchtst_original_deadband20_mean_v1"
LEGACY_NUMERICAL_POLICY = "original_patchtst_daily_14d_30epoch_delta20_fixed20deadband_v1"
MODEL_VERSION = "afternoon_patchtst_original_model_direct_v3"
NUMERICAL_POLICY = "original_patchtst_daily_14d_30epoch_direct_mean_receipt_inputs_no_selector_v3"
LIVE_INPUT_POLICY = "current_collected_snapshot_receipt_revision_v2"
ROOT = Path(__file__).resolve().parent
FORWARD = ROOT / "research/sequence_followup_20261008/forward"
KL = timezone(timedelta(hours=8))
_LOCK = threading.RLock()
_READY = None
_PINNED = {}
_STATUS = {"state": "background_cache_pending"}


def training_cutoff(issue):
    return int(datetime.fromtimestamp(int(issue), KL).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp())


def scope_reason(issue, start, end):
    """Reject malformed target shape only; training lead is metadata."""
    return direct.target_shape_reason(issue, start, end)


def read_issued_sources(conn, cutoff):
    """Read newest available immutable receipt vintage in the sensor transaction.

    Only the already declared two-hour source eligibility range is loaded.
    Unknown legacy receipts remain explicitly unknown, never backdated.
    """
    cutoff = int(cutoff)
    first = max(0, cutoff-7200)
    weather_rows = provenance.compatible_source_runs(
        conn, "weather_forecast_runs", cutoff, start_fetched_epoch=first)
    cams_rows = provenance.compatible_source_runs(
        conn, "air_quality_forecast_runs", cutoff, start_fetched_epoch=first)
    cams_rows = [r for r in cams_rows if r.get("model_version") in ("cams_anchor_v1", "cams_anchor_v2")
                 and r.get("source") == "Open-Meteo / CAMS Global"]
    weather_rows, cams_rows = weather_rows[-1:], cams_rows[-1:]
    def tuples(rows):
        return [(int(r["fetched_epoch"]), str(r["source"]), decode_payload_text(r["payload"])) for r in rows]
    return {"weather": tuples(weather_rows), "cams": tuples(cams_rows),
            "sourceProvenance": {"weather": [dict(r["_provenance"]) for r in weather_rows],
                                 "cams": [dict(r["_provenance"]) for r in cams_rows]}}


def materialize_snapshot(rows, sources, issue, opened, received):
    mark = int(rows[-1]["epoch"])
    sensor = [{k: dict(r).get(k) for k in ("epoch", "pm02", "atmp", "rhum")}
              for r in rows if mark - 25 * 3600 <= int(r["epoch"]) <= mark]
    return {**sources, "sensor": sensor, "cutoffEpoch": int(issue), "issueEpoch": int(issue),
            "sensorWatermarkEpoch": mark, "dbSnapshotOpenedEpoch": float(opened),
            "inputSnapshotReceivedEpoch": float(received), "clockPrecision": "subsecond_receipt_integer_issue",
            "readOnly": True, "futureObservedSensorRowsLoaded": 0,
            # The original query adapter carries these research-only audit keys.
            # Current collected rows do not claim an earlier API publication.
            "apiSnapshotReceivedEpoch": None, "apiPublishedForecastIssueEpoch": None}


def prepare_runtime(cache=None, issue=None):
    """Background-only verified daily load; never fit here or in prediction."""
    global _READY, _STATUS
    issue = int(time.time()) if issue is None else int(issue)
    cutoff = training_cutoff(issue)
    try:
        if cache is None:
            if str(FORWARD) not in sys.path:
                sys.path.insert(0, str(FORWARD))
            import daily_cache as cache
        # Real original modules retain their old verifier. A separate strict
        # inference revision admits only the explicitly preserved and hashed
        # receipt readers; test adapters keep their supplied verification seam.
        if getattr(cache, "__file__", None):
            import patchtst_live_runtime
            cache = patchtst_live_runtime.wrap_cache(cache)
            patchtst_live_runtime.adopt_completed_artifact(cache, cutoff, issue)
        policy = cache.verify_policy()
        path = cache.CACHE / "models" / f"{cutoff}_patchtst.pt"
        info_path = path.with_suffix(".json")
        fingerprint = (path.stat().st_mtime_ns, path.stat().st_size,
                       info_path.stat().st_mtime_ns, info_path.stat().st_size)
        with _LOCK:
            if cutoff in _PINNED and _PINNED[cutoff] != fingerprint:
                raise RuntimeError("completed_daily_artifact_changed_after_initialization")
            if _READY and _READY["cutoff"] == cutoff and _READY["fingerprint"] == fingerprint:
                return dict(_STATUS)
        fitted = json.loads(info_path.read_text(encoding="utf-8"))
        learner = fitted.get("learnerMetadata") or {}
        if (fitted["trainingCutoffEpoch"] != cutoff
                or fitted["maximumTrainingCompleteEpoch"] >= cutoff
                or fitted["trainingRows"] < 120 or fitted["distinctTrainingTargetDates"] < 10
                or fitted["fittedAtEpoch"] > issue or learner.get("variant") != "patchtst"
                or learner.get("epochs") != 30 or learner.get("normalizerTrainingOnly") is not True
                or fitted["dailyPolicySha256"] != cache.original.sha(cache.HERE / "DAILY_POLICY.json")
                or cache.original.sha(path) != fitted["checkpointSha256"]):
            raise RuntimeError("daily_artifact_metadata_or_hash_invalid")
        model, cached_info = cache._cached(cutoff)
        if cached_info != fitted:
            raise RuntimeError("daily_cached_metadata_differs")
        ready = {"cutoff": cutoff, "model": model, "fitted": fitted, "adapter": cache.original,
                 "policy": policy, "fingerprint": fingerprint,
                 "metadataSha256": cache.original.sha(info_path)}
        with _LOCK:
            _PINNED[cutoff] = fingerprint
            _READY = ready
            _STATUS = {"state": "ready", "trainingCutoffEpoch": cutoff,
                       "checkpointSha256": fitted["checkpointSha256"],
                       "liveRuntimeRevision": policy.get("liveRuntimeRevision"),
                       "liveRuntimeRevisionSha256": policy.get("liveRuntimeRevisionSha256")}
    except Exception as error:
        with _LOCK:
            _READY = None
            _STATUS = {"state": "unavailable", "trainingCutoffEpoch": cutoff,
                       "reason": f"{type(error).__name__}: {error}"}
    return dict(_STATUS)


def _validate_snapshot(snapshot, issue):
    if int(snapshot["issueEpoch"]) != issue or int(snapshot["cutoffEpoch"]) != issue:
        raise ValueError("snapshot_issue_clock_mismatch")
    mark = int(snapshot["sensorWatermarkEpoch"])
    opened, receipt = (float(snapshot[k]) for k in ("dbSnapshotOpenedEpoch", "inputSnapshotReceivedEpoch"))
    if not mark <= opened <= receipt <= issue or not 0 <= issue - receipt <= 120:
        raise ValueError("invalid_or_stale_snapshot_receipt_clocks")
    if not 0 <= issue - mark <= 240:
        raise ValueError("stale_or_future_sensor_watermark")
    if snapshot.get("readOnly") is not True or snapshot.get("futureObservedSensorRowsLoaded") != 0:
        raise ValueError("require_read_only_inputs_without_future_observations")
    rows = snapshot["sensor"]
    if not rows or max(int(r["epoch"]) for r in rows) != mark or any(int(r["epoch"]) > mark for r in rows):
        raise ValueError("future_or_missing_watermark_sensor_input")
    if not snapshot["weather"]:
        raise ValueError("issued_weather_unavailable")
    if any(float(f) > opened for f, _, _ in snapshot["weather"] + snapshot["cams"]):
        raise ValueError("external_source_after_snapshot_open")
    if not 0 <= issue - float(snapshot["weather"][-1][0]) <= 7200:
        raise ValueError("stale_or_future_issued_weather")


def predict_windows(rows, windows, issue_epoch, *, snapshot, weather_fetched_epoch=None):
    """Fast original CPU inference on the exact published card and snapshot."""
    issue = int(issue_epoch)
    target = windows.get("afternoon") or {}
    start, end = target.get("startEpoch"), target.get("endEpoch")
    reason = scope_reason(issue, start, end)
    if reason:
        return {"afternoon": direct.unavailable_model_output(
            issue, start, end, MODEL_VERSION, reason, "afternoon")}
    try:
        _validate_snapshot(snapshot, issue)
        with _LOCK:
            ready = _READY
            if ready is None or ready["cutoff"] != training_cutoff(issue):
                raise RuntimeError("current_daily_background_artifact_not_ready")
            fitted, adapter = ready["fitted"], ready["adapter"]
            if fitted["fittedAtEpoch"] > issue:
                raise RuntimeError("daily_artifact_not_ready_at_issue")
            bundle, audit = direct.patch_query(adapter, snapshot, int(start), int(end))
            if weather_fetched_epoch is None or float(weather_fetched_epoch) != audit["weatherFetchedEpoch"]:
                raise ValueError("issued_weather_does_not_match_exact_card")
            raw = float(ready["model"].predict(bundle, np.array([0], dtype=np.int64))[0])
        generated = time.time()
        if not math.isfinite(raw) or not issue <= generated < start or generated - issue > 120:
            raise ValueError("invalid_or_overdue_generated_forecast")
        if generated - audit["sensorReferenceEpoch"] > 240 or generated - audit["weatherFetchedEpoch"] > 7200:
            raise ValueError("source_became_stale_during_generation")
        fresh = float(audit["fresh"])
        point = raw
        tensor_hash = hashlib.sha256(b"".join(np.ascontiguousarray(bundle[k]).tobytes()
                                             for k in ("past", "future", "context", "fresh"))).hexdigest()
        source_hash = hashlib.sha256(json.dumps(adapter.safe(snapshot), sort_keys=True,
                                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        lead_scope = ("same_day_afternoon" if datetime.fromtimestamp(start, KL).date()
                      == datetime.fromtimestamp(issue, KL).date() else "next_day_afternoon")
        canonical_clock = (datetime.fromtimestamp(start, KL).strftime("%H:%M:%S") == "14:00:00"
                           and datetime.fromtimestamp(issue, KL).strftime("%H:%M:%S")
                           == ("12:00:00" if lead_scope == "same_day_afternoon" else "18:05:00"))
        model = {"applied": True, "experimental": True, "prospectivelyValidated": False,
                 "featureVersion": adapter.prepare.VERSION, "numericalPolicy": NUMERICAL_POLICY,
                 "trainingCutoffEpoch": ready["cutoff"], "trainingCount": fitted["trainingRows"],
                 "trainingDistinctTargetDays": fitted["distinctTrainingTargetDates"],
                 "trainingLatestTargetEndEpoch": fitted["maximumTrainingCompleteEpoch"],
                 "trainingFeatureLabelSha256": fitted["trainingTensorLabelSha256"],
                 "fittedAtEpoch": fitted["fittedAtEpoch"], "checkpointSha256": fitted["checkpointSha256"],
                 "trainingMetadataSha256": ready["metadataSha256"],
                 "dailyPolicySha256": fitted["dailyPolicySha256"],
                 "rawMeanUgM3": raw, "deadbandUgM3": None, "deadbandApplied": False,
                 "eventThresholdUgM3": 20, "eventThresholdAppliedToPoint": False,
                 "legacyModelVersion": LEGACY_MODEL_VERSION,
                 "legacyNumericalPolicy": LEGACY_NUMERICAL_POLICY,
                 "originalTrainingArtifactUnchanged": True,
                 "modelOutputPolicyVersion": direct.VERSION,
                 "queryTensorSha256": tensor_hash, "materializedSnapshotSha256": source_hash,
                 "forecastGeneratedEpoch": generated, "sequenceSourceMaxEpoch": audit["sequenceSourceMaxEpoch"],
                 "pastPmValidSteps": audit["pastPmValidSteps"], "tensorShapes": audit["tensorShapes"],
                 "liveInputAvailabilityPolicy": LIVE_INPUT_POLICY,
                 "liveRuntimeRevision": ready["policy"].get("liveRuntimeRevision"),
                 "liveRuntimeRevisionSha256": ready["policy"].get("liveRuntimeRevisionSha256"),
                 "inputRevisionInvalidatesPreviousIssuedSkill": True,
                 "developmentScope": lead_scope + "_fixed_clock_previously_examined_dates",
                 "queryMatchesEvaluatedClock": canonical_clock,
                 "featureDomain": {"actualLeadMinutes": (start - issue) / 60, "trainingLeadRange": [105, 1380]},
                 "domainAppliedToNumber": False, "statisticalSelectorApplied": False,
                 "directQueryVersion": direct.QUERY_VERSION,
                 "directQueryAdapterSha256": hashlib.sha256(Path(direct.__file__).read_bytes()).hexdigest(),
                 "futureTensorMaximumEpoch": audit.get("futureTensorMaximumEpoch"),
                 "futureTensorTargetOverlapHours": audit.get("futureTensorTargetOverlapHours"),
                 "futureTensorContainsFullTarget": audit.get("futureTensorContainsFullTarget"),
                 "futureTensorBeyondTargetCoverageHours": audit.get("futureTensorBeyondTargetCoverageHours"),
                 "targetContextWeatherComplete": audit.get("targetContextWeatherComplete"),
                 "movingWindowSkillVerified": False}
        result = {
            "available": True, "modelVersion": MODEL_VERSION, "source": "experimental_afternoon_patchtst",
            "mean": point, "prediction": point, "sensorAnchor": fresh, "baselinePoint": fresh,
            "numericalPolicyIdentifier": NUMERICAL_POLICY,
            "closedSensorAnchor": audit["closedLevel"], "sensorReferenceEpoch": audit["sensorReferenceEpoch"],
            "sensorReferenceCount": audit["sensorReferenceCount"], "sensorWatermarkEpoch": audit["sensorWatermarkEpoch"],
            "featureSourceMaxEpoch": audit["featureSourceMaxEpoch"], "originEpoch": int(issue // 900 * 900),
            "forecastedAtEpoch": issue, "forecastGeneratedEpoch": generated,
            "startEpoch": int(start), "endEpoch": int(end), "remainingLeadHours": (start - issue) / 3600,
            "leadHours": (start - issue) / 3600, "inputAvailabilityPolicy": LIVE_INPUT_POLICY,
            "inputSnapshotReceivedEpoch": snapshot["inputSnapshotReceivedEpoch"],
            "dbSnapshotOpenedEpoch": snapshot["dbSnapshotOpenedEpoch"],
            "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median_capped_by_current_collected_snapshot",
            "forecastState": "experimental_patchtst_session_mean", "pointRole": "experimental_session_mean",
            "target": "overlap_duration_weighted_mean_of_complete_15_minute_raw_sensor_medians_over_exact_card",
            "method": "Experimental PatchTST · exact 2-hour Afternoon mean",
            "patchtstModel": model, "weatherSessionModel": {"applied": False},
            "weatherAvailable": True, "weatherFetchedEpoch": audit["weatherFetchedEpoch"],
            "camsFetchedEpoch": audit["camsFetchedEpoch"],
            "camsAvailable": math.isfinite(float(audit["contextFeatures"].get("camsTarget", float("nan")))),
            "rainContext": adapter.prepare.rain.describe_window(
                adapter.prepare.Sources(snapshot).archive, issue, int(start), int(end)),
            "weatherSourceClock": {"asOfEpoch": issue, "fetchedEpoch": audit["weatherFetchedEpoch"],
                                   "payloadHash": audit["weatherPayloadSha256"], "payloadHashAlgorithm": "blake2b_160",
                                   "availabilityRule": "fetched_at_or_before_materialized_snapshot_open", "maximumAgeSeconds": 7200},
            "freshnessAdjustment": {"applied": False, "referenceUsedAsModelInput": True,
                                    "referenceEpoch": audit["sensorReferenceEpoch"], "referenceCount": audit["sensorReferenceCount"]},
            "modelSelection": {},
            "candidateForecast": {"available": True, "modelVersion": MODEL_VERSION, "mean": raw, "prediction": raw,
                                  "pointRole": "raw_patchtst_mean_without_numeric_deadband",
                                  "appliedToPrimaryForecast": False,
                                  "numericalPolicyIdentifier": NUMERICAL_POLICY, "forecastedAtEpoch": issue,
                                  "startEpoch": int(start), "endEpoch": int(end)},
            "rawRangeLow": None, "rawRangeHigh": None, "uncertaintyMethod": "no_new_model_interval_published",
            "modelEvidence": {"available": False, "count": 0, "distinctDays": 0, "prospectivelyValidated": False,
                              "validationMode": "new_model_issued_accuracy_collecting", "scope": lead_scope},
            "usedForDecision": False, "usedForComparison": False,
            "weights": {}, "components": {}, "componentsRole": "original_patchtst_raw_mean_without_numeric_deadband",
            "sourceCaveat": "Experimental PatchTST; issued accuracy is collecting.",
        }
        return {"afternoon": direct.publish_direct_output(result, "afternoon")}
    except Exception as error:
        unavailable = direct.unavailable_model_output(
            issue, start, end, MODEL_VERSION, f"{type(error).__name__}: {error}", "afternoon")
        unavailable["patchtstModel"] = {
            "modelVersion": MODEL_VERSION, "applied": False,
            "reason": f"{type(error).__name__}: {error}", "fitStatus": dict(_STATUS)}
        return {"afternoon": unavailable}


def apply_experimental_afternoon(rows, windows, issue, estimates, *, snapshot, weather_fetched_epoch=None):
    """Assign the fixed Patch output; never choose an alternate point/model."""
    candidate = predict_windows(rows, windows, issue, snapshot=snapshot,
                                weather_fetched_epoch=weather_fetched_epoch)["afternoon"]
    result = dict(estimates)
    result["afternoon"] = candidate
    return result
