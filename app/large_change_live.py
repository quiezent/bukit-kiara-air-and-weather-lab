"""Cached, separately versioned >=20 first-crossing and magnitude inference.

Only ``refresh_model`` trains, from the service background worker.  Prediction
does a bounded read of the already-available measurement/weather snapshot and
never trains.  The selected policy starts pending so staging this module cannot
activate a candidate before its chronological evaluation has been reviewed.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
import time

import numpy as np
import pandas as pd

import first_crossing_live as source
import large_change_hazard as hazard
import large_change_paths as paths
from event_qualification import (apply_event_qualification, qualification_metadata,
                                 publication_metadata)

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private"
ADAPTER_VERSION = "ttdi_large20_live_adapter_v2_qualified_diagnostics"
MODEL_VERSION = "ttdi_large20_first_crossing_and_magnitude_v1"
ARTIFACT_NAME = "large-change20-model.pkl"
STATUS_NAME = "large-change20-model-status.json"
ISSUE_NAME = "large-change20-issued.jsonl"

# Root sets these only after reviewing the fixed historical candidate trial.
# Changing either is part of the model identity and invalidates existing caches.
SELECTED_CANDIDATE = "dedicated20"
MAGNITUDE_METHOD = "training_direction_median"
SELECTION_EVIDENCE = "explicit_user_requested_20_threshold_20261006"
THRESHOLD = 20.0
_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_MODELS = {}
_RECORDED = OrderedDict()
_ISSUE_PATHS_READ = set()


def selected_policy():
    return {"probabilityCandidate": SELECTED_CANDIDATE,
            "conditionalMagnitudeMethod": MAGNITUDE_METHOD,
            "selectionEvidence": SELECTION_EVIDENCE,
            "changeThresholdUgM3": THRESHOLD, "thresholdComparison": "at_least",
            "target": "first_sampled_crossing_within90_minutes",
            "sampleLeadsMinutes": list(hazard.LEADS),
            "directionDecision": "three_class_argmax_ties_unresolved",
            "directionTieTolerance": 1e-12 if SELECTED_CANDIDATE == "joint_path_forest" else 0.0,
            "magnitudeTarget": "maximum_sampled_directional_excursion_within90_minutes_conditional_on_first20_cause",
            "trainingLookbackDays": 28, "trainingCutoff": "issue_day_midnight_Asia_Kuala_Lumpur",
            "trainingCadence": "fixed_four_hour_origins_plus300_seconds",
            "minimumGuardedHourlyIssues": 120, "minimumCompletedTargetDates": 10,
            "trainingClassRequirement": "all_three20_risk_classes_present",
            "physicalSupport": "nonnegative_concentration_before_cause_and_magnitude_calculation",
            "calibrated": False}


def _selected():
    return (SELECTED_CANDIDATE in ("dedicated20", "pooled10_20", "joint_path_forest") and
            MAGNITUDE_METHOD in ("path", "training_direction_median") and
            SELECTION_EVIDENCE != "pending")


def midnight_epoch(issue_epoch):
    return source.midnight_epoch(issue_epoch)


def _paths(model_dir):
    directory = Path(model_dir) if model_dir is not None else PRIVATE
    return directory / ARTIFACT_NAME, directory / STATUS_NAME


def _identity():
    policy = selected_policy()
    return {"adapterSha256": source.h.digest(__file__),
            "hazardHelperSha256": source.h.digest(ROOT / "large_change_hazard.py"),
            "pathHelperSha256": source.h.digest(ROOT / "large_change_paths.py"),
            "selectedPolicy": policy,
            "selectedPolicySha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "guardedSourceIdentity": source._identity()}


def _training_set(bundle, cutoff):
    issues = bundle["issues"]
    if issues.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "trainingSupport": dict.fromkeys(hazard.NAMES, 0),
            "trainingSupport10": dict.fromkeys(hazard.NAMES, 0),
            "originalGuardedHourlyTrainingIssues": 0, "fixedFourHourTrainingIssues": 0,
            "completedTargetDays": 0, "riskRows20": 0, "maxTrainingOutcomeEpoch": None,
            "unavailableReasons": ["missing_guarded_training_inputs"]}
    hourly = issues.loc[issues.allSixCovered & issues.completeEpoch.lt(cutoff) &
                        issues.issueEpoch.ge(cutoff - 28 * 86400) & issues.issueEpoch.lt(cutoff)]
    train = hourly.loc[source.h.sparse_mask(hourly)].sort_values("issueEpoch").copy()
    step_features = bundle["stepFeatures"].loc[bundle["stepFeatures"].issueEpoch.isin(train.issueEpoch)].copy()
    counts = {name: int(train.cause20.eq(cause).sum()) for cause, name in zip(hazard.CLASSES, hazard.NAMES)}
    counts10 = {name: int(train.cause.eq(cause).sum()) for cause, name in zip(hazard.CLASSES, hazard.NAMES)}
    reasons = []
    if len(hourly) < 120: reasons.append("fewer_than120_original_guarded_hourly_training_issues")
    if train.targetDay.nunique() < 10: reasons.append("fewer_than10_completed_target_dates")
    risk = hazard.risk_rows(train, step_features, threshold=20) if len(train) else pd.DataFrame()
    if risk.empty or set(risk.riskLabel) != set(hazard.CLASSES):
        reasons.append("missing20_at_risk_cause_class")
    return train, step_features, {
        "trainingSupport": counts, "trainingSupport10": counts10,
        "originalGuardedHourlyTrainingIssues": len(hourly), "fixedFourHourTrainingIssues": len(train),
        "completedTargetDays": int(train.targetDay.nunique()), "riskRows20": len(risk),
        "maxTrainingOutcomeEpoch": int(train.completeEpoch.max()) if len(train) else None,
        "unavailableReasons": reasons}


def _load_artifact(model_dir=None):
    artifact, _ = _paths(model_dir)
    key = str(artifact.resolve())
    try: signature = (artifact.stat().st_mtime_ns, artifact.stat().st_size)
    except OSError: return None
    identity = _identity()
    with _LOCK:
        cached = _MODELS.get(key)
        if cached is not None and cached[0] == signature and cached[1].get("identity") == identity:
            return cached[1]
    # The service owns this local pickle; atomic replacement prevents partial reads.
    try:
        artifact_value = pickle.loads(artifact.read_bytes())
        if not isinstance(artifact_value, dict): return None
        if artifact_value.get("identity") != identity or artifact_value.get("adapterVersion") != ADAPTER_VERSION: return None
        if artifact_value.get("modelVersion") != MODEL_VERSION: return None
        with _LOCK: _MODELS[key] = (signature, artifact_value)
        return artifact_value
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError, AssertionError):
        return None


def refresh_model(db_path, issue_epoch, *, model_dir=None):
    """Background-only daily fit, with all target labels strictly before midnight.

    Pending selection performs no source reads, fits, or model/status writes.
    Production callers must never invoke this from HTTP or forecast computation.
    """
    cutoff = midnight_epoch(issue_epoch)
    if not _selected():
        return {"available": False, "reason": "evaluated_candidate_selection_pending",
                "modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
                "trainingCutoffEpoch": cutoff, "selectedPolicy": selected_policy()}
    artifact_path, status_path = _paths(model_dir)
    with _FIT_LOCK:
        identity = _identity()
        prior = _load_artifact(model_dir)
        if (prior is not None and prior["trainingCutoffEpoch"] == cutoff and
                prior["metadata"].get("reason") != "background_fit_unavailable"):
            return dict(prior["metadata"], cacheReloaded=True)
        started = time.perf_counter()
        metadata = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
                    "trainingCutoffEpoch": cutoff, "fittedAtEpoch": int(time.time()),
                    "experimental": True, "calibrated": False,
                    "trainingLookbackDays": 28, "featureColumns": list(hazard.FEATURES),
                    "selectedPolicy": selected_policy(), "fitCpuThreads": 1, "identity": identity,
                    "sourceRole": "reconstructed_guarded_original_sensor_publication_and_weather_archive"}
        hazard_model = path_model = training_delta_paths = None
        try:
            sources = source._source_snapshot(db_path, cutoff - 1, cutoff)
            bundle, rebuild = source.p.rebuild_inputs(*sources)
            train, steps, support = _training_set(bundle, cutoff)
            metadata.update(support, sourceSnapshot=sources[-1], inputRebuild=rebuild)
            if not support["unavailableReasons"]:
                if not train.completeEpoch.lt(cutoff).all() or not source.h.feature_clock_valid(train).all():
                    raise ValueError("training_clock_guard_failed")
                if len(train) > 1 and not train.originEpoch.diff().dropna().ge(14400).all():
                    raise ValueError("training_issues_not_four_hour_spaced")
                training_delta_paths = (train[["outcome" + str(lead) for lead in hazard.LEADS]].to_numpy(float)
                                        - train.fresh.to_numpy(float)[:, None])
                if not np.isfinite(training_delta_paths).all():
                    raise ValueError("incomplete_training_paths")
                if SELECTED_CANDIDATE in ("dedicated20", "pooled10_20"):
                    hazard_model = hazard.fit_hazard(train, steps, kind=SELECTED_CANDIDATE)
                    metadata["riskRows"] = hazard_model["riskRows"]
                if SELECTED_CANDIDATE == "joint_path_forest" or MAGNITUDE_METHOD == "path":
                    features = paths.aggregate_issue_features(steps)
                    if features.issueEpoch.tolist() != train.issueEpoch.astype(int).tolist():
                        raise ValueError("path_feature_training_alignment_failed")
                    path_model = paths.fit_path_model(features, training_delta_paths)
                available = (hazard_model is not None if SELECTED_CANDIDATE != "joint_path_forest" else path_model is not None)
            else:
                available = False
            metadata.update(available=bool(available), reason="available" if available else "training_support_unavailable")
        except Exception as error:
            metadata.update(available=False, reason="background_fit_unavailable", unavailableReasons=[type(error).__name__])
            hazard_model = path_model = training_delta_paths = None
        metadata["fittedAtEpoch"] = int(time.time())
        metadata["fitElapsedSeconds"] = round(time.perf_counter() - started, 3)
        artifact = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION, "identity": identity,
                    "trainingCutoffEpoch": cutoff, "metadata": metadata, "hazardModel": hazard_model,
                    "pathModel": path_model, "trainingDeltaPaths": training_delta_paths}
        source._atomic(artifact_path, pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
        source._atomic(status_path, (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        signature = (artifact_path.stat().st_mtime_ns, artifact_path.stat().st_size)
        with _LOCK: _MODELS[str(artifact_path.resolve())] = (signature, artifact)
        return dict(metadata, cacheReloaded=False)


def _magnitude_estimates(raw, reference):
    estimates = {}
    for name in ("drop", "rise"):
        item = raw[name + "ConditionalMagnitude"]
        quantiles = [item.get("q10"), item.get("median"), item.get("q90")]
        available = all(value is not None and math.isfinite(float(value)) for value in quantiles)
        if available:
            q10, median, q90 = map(float, quantiles)
            if not THRESHOLD <= q10 <= median <= q90:
                raise ValueError("invalid_conditional_magnitude_quantiles")
            if name == "drop" and q90 > reference + 1e-8:
                raise ValueError("conditional_drop_exceeds_reference")
        estimates[name] = {
            "available": available, "medianUgM3": float(quantiles[1]) if available else None,
            "q10UgM3": float(quantiles[0]) if available else None,
            "q90UgM3": float(quantiles[2]) if available else None,
            "effectiveSupport": float(item["effectiveCauseCases"]) if available else 0.0,
            "trainingCauseCases": int(item.get("trainingCauseCases", 0)),
            "positiveWeightCauseCases": int(item.get("positiveWeightCauseCases", 0)),
            "method": MAGNITUDE_METHOD,
            "conditioning": "first_at_least20_" + name + "_within90_minutes",
            "target": "maximum_sampled_directional_excursion_within90_minutes",
            "notEndpointEstimate": True, "calibrated": False,
            "estimateRole": "uncalibrated_conditional_training_distribution_diagnostic",
            "operationalUseEligible": False,
            "reason": "available" if available else "no_supported_conditional_training_paths"}
    return estimates


def predict_event(db_path, rows, issue_epoch, sensor_watermark_epoch, reference_pm, *, model_dir=None):
    """Read-only cached inference; never builds training data or fits a model."""
    issue = int(issue_epoch)
    try: reference = float(reference_pm)
    except (TypeError, ValueError, OverflowError): reference = math.nan
    result = {"available": False, "modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
              "experimental": True, "calibrated": False, "forecastIssuedEpoch": issue,
              "validUntilEpoch": issue + 5400, "referencePm": reference if math.isfinite(reference) else None,
              "changeThresholdUgM3": THRESHOLD, "thresholdComparison": "at_least",
              "target": "first_sampled_crossing_within90_minutes", "direction": "unresolved",
              "directionDecision": "three_class_argmax_ties_unresolved", "horizons": [],
              "directionTieTolerance": selected_policy()["directionTieTolerance"],
              "magnitudeEstimates": {}, "selectedPolicy": selected_policy()}
    result.update(qualification_metadata(issue), diagnosticDirection="unresolved",
                  diagnosticDirectionAvailable=False)
    if not _selected():
        result.update(reason="evaluated_candidate_selection_pending", unavailableReasons=["candidate_not_selected_after_evaluation"])
        return result
    try:
        if not math.isfinite(reference) or reference < 0:
            raise ValueError("invalid_reference")
        artifact = _load_artifact(model_dir)
        if artifact is None:
            result.update(reason="background_model_collecting", unavailableReasons=["no_current_cached_estimator"])
            return result
        metadata = artifact["metadata"]
        for key in ("trainingCutoffEpoch", "trainingSupport", "trainingSupport10", "fittedAtEpoch", "completedTargetDays",
                    "fixedFourHourTrainingIssues", "riskRows", "riskRows20", "maxTrainingOutcomeEpoch"):
            if key in metadata: result[key] = metadata[key]
        result["modelIdentity"] = artifact["identity"]
        cutoff = midnight_epoch(issue)
        if artifact["trainingCutoffEpoch"] != cutoff:
            result.update(reason="background_model_cutoff_mismatch", unavailableReasons=["cached_fit_is_not_issue_day_midnight"])
            return result
        if metadata["fittedAtEpoch"] > issue:
            result.update(reason="model_not_yet_fitted_at_issue", unavailableReasons=["cached_fit_recorded_after_issue"])
            return result
        if not metadata["available"]:
            result.update(reason=metadata["reason"], unavailableReasons=metadata.get("unavailableReasons", []))
            return result
        if metadata["maxTrainingOutcomeEpoch"] >= cutoff:
            raise ValueError("training_labels_not_strictly_before_issue_day_midnight")
        inputs, reason = source._current_inputs(db_path, rows, issue, float(sensor_watermark_epoch), reference)
        if inputs is None:
            result.update(reason=reason, unavailableReasons=[reason])
            return result
        steps, provenance = inputs
        path_result = None
        if SELECTED_CANDIDATE == "joint_path_forest" or MAGNITUDE_METHOD == "path":
            features = paths.aggregate_issue_features(steps)
            path_result = paths.predict_path_distribution(artifact["pathModel"], features, threshold=THRESHOLD)[0]
        if SELECTED_CANDIDATE == "joint_path_forest":
            horizons = path_result["horizons"]
            call = int(path_result["direction"])
        else:
            probabilities = hazard.cumulative_incidence(hazard.predict_hazards(artifact["hazardModel"], steps, threshold=THRESHOLD))[0]
            horizons = [{"leadMinutes": lead, "probabilityDrop": float(values[0]),
                         "probabilityNoCrossing": float(values[1]), "probabilityRise": float(values[2])}
                        for lead, values in zip(hazard.LEADS, probabilities)]
            call = int(hazard.direction_calls(probabilities[-1:], rule="argmax")[0])
        if MAGNITUDE_METHOD == "path":
            magnitude_raw = path_result
        else:
            magnitude_raw = paths.predict_training_direction_magnitude(artifact["trainingDeltaPaths"], reference, threshold=THRESHOLD)
        magnitudes = _magnitude_estimates(magnitude_raw, reference)
        final = horizons[-1]
        mass = [final[name] for name in ("probabilityDrop", "probabilityNoCrossing", "probabilityRise")]
        if (not np.isfinite(mass).all() or min(mass) < 0 or not np.isclose(sum(mass), 1, atol=1e-8) or
                (reference < THRESHOLD and final["probabilityDrop"] != 0)):
            raise ValueError("invalid_or_physically_impossible_probability_mass")
        result.update(provenance, horizons=horizons, magnitudeEstimates=magnitudes,
                      **{name: final[name] for name in ("probabilityDrop", "probabilityNoCrossing", "probabilityRise")})
        result.update(available=True, reason="available", unavailableReasons=[],
                      direction="drop" if call == -1 else "rise" if call == 1 else "unresolved",
                      probabilityAnyCrossing=float(final["probabilityDrop"] + final["probabilityRise"]))
        return apply_event_qualification(result)
    except Exception as error:
        result.update(reason="large_change_inference_unavailable", unavailableReasons=[type(error).__name__])
        return result


def record_issued_prediction(result, *, output_path=None, recorded_epoch=None, input_lineage=None):
    """Archive only a newly published issue; replay callers must not call this."""
    recorded = int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue = int(result["forecastIssuedEpoch"])
    if not 0 <= recorded - issue <= 120:
        raise ValueError("Only fresh actual publication may be recorded as originally issued")
    path = Path(output_path) if output_path is not None else PRIVATE / ISSUE_NAME
    key = (str(path.resolve()), issue, result.get("trainingCutoffEpoch"), result.get("featureValuesSha256"), result.get("reason"))
    record = {**result, "recordedEpoch": recorded, "publishedEpoch": recorded,
              "isOriginallyIssued": True, "predictionRole": "genuinely_issued_live_large20_prediction",
              **publication_metadata(result, input_lineage)}
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path.resolve()) not in _ISSUE_PATHS_READ:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2_000_000))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()
                for line in tail:
                    try:
                        old = json.loads(line)
                        old_key = (str(path.resolve()), int(old["forecastIssuedEpoch"]), old.get("trainingCutoffEpoch"),
                                   old.get("featureValuesSha256"), old.get("reason"))
                        _RECORDED[old_key] = None
                    except (ValueError, KeyError, TypeError): continue
            _ISSUE_PATHS_READ.add(str(path.resolve()))
        if key in _RECORDED: return False
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        _RECORDED[key] = None
        while len(_RECORDED) > 4096: _RECORDED.popitem(last=False)
    return True
