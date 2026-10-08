"""Staged cached forecasts of an exact two-hour ride-window planning cutoff.

Daily model fitting belongs to the independent background worker. Inference
never fits. A recheck preserves the originally chosen window rather than moving
the ride later. Pending candidate selection performs no source or cache I/O.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import threading
import time

import numpy as np
import pandas as pd

import first_crossing_live as source
import cycling_window_probability as probability
import cycling_window_distribution as distribution
from forecast_clock import window_weights

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private"
PREPARATION_PATH = ROOT / "research/cycling_window_70/recheck_preparation.py"
MODEL_VERSION = "ttdi_fixed_ride_window70_live_v1"
ADAPTER_VERSION = "ttdi_cycling_window_live_adapter_v1"
ARTIFACT_NAME = "cycling-window70-model.pkl"
STATUS_NAME = "cycling-window70-model-status.json"
ISSUE_NAME = "cycling-window70-issued.jsonl"
CUTOFF_UG_M3 = 70.0
SELECTED_CANDIDATES = {"initial90": "sensor", "recheck60": "sensor"}
SELECTION_EVIDENCE = {
    "initial90": {"role": "historical_development_selection",
                  "policySha256": "a522f388d0afe59763cccff7077bb11e92d6ca30486dd1043fe7ee4869173703",
                  "resultsSha256": "e0703aaf48eae496ddcd231567dbe5fc9c03519d3a6b23bec191b366a9e06a5b"},
    "recheck60": {"role": "historical_development_selection",
                  "policySha256": "7043dc1c6c08870185a699e529967d0650efa9003be690ab298533a3906abf2d",
                  "resultsSha256": "6b2c90ddad56c07ea409bfc3da4a73763ce36a9e98b07d430a673920d1f01267"},
}
MODE_PHASE = {"initial90": "initial", "recheck60": "recheck"}
_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_MODELS = {}
_RECORDED = OrderedDict()
_ISSUE_PATHS_READ = set()
_PREPARATION = None


def selected_policy():
    return {"candidatesByMode": dict(SELECTED_CANDIDATES), "selectionEvidence": SELECTION_EVIDENCE,
            "planningCutoffUgM3": CUTOFF_UG_M3, "targetComparison": "mean_at_or_below",
            "target": "exact_two_hour_ride_window_mean",
            "initialLeadSeconds": 5400, "durationSeconds": 7200,
            "recheckOffsetSeconds": 1800, "supportedRecheckLeadSeconds": [3300, 3900],
            "recheckKeepsOriginalAbsoluteWindow": True,
            "trainingLookbackDays": 28, "trainingCadence": "four_hour_initial_origins_plus300_seconds",
            "trainingCutoff": "issue_day_midnight_Asia_Kuala_Lumpur", "trainingHeads": "separate_initial90_and_recheck60",
            "minimumCasesPerHead": 50, "minimumTargetDatesPerHead": 10, "bothOutcomeClassesRequired": True,
            "targetWeights": "overlap_duration_weighted_complete_15_minute_sensor_medians",
            "calibrated": False, "cutoffMeaning": "user_selected_planning_cutoff"}


def _selected(mode):
    evidence = SELECTION_EVIDENCE.get(mode, "pending") if isinstance(SELECTION_EVIDENCE, dict) else SELECTION_EVIDENCE
    return SELECTED_CANDIDATES.get(mode) in ("sensor", "weather", "forest") and evidence != "pending"


def _preparation():
    global _PREPARATION
    if _PREPARATION is None:
        spec = importlib.util.spec_from_file_location("cycling_window_live_preparation", PREPARATION_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PREPARATION = module
    return _PREPARATION


def _identity():
    policy = selected_policy()
    return {"adapterSha256": source.h.digest(__file__),
            "probabilityHelperSha256": source.h.digest(ROOT / "cycling_window_probability.py"),
            "distributionHelperSha256": source.h.digest(ROOT / "cycling_window_distribution.py"),
            "preparationHelperSha256": source.h.digest(PREPARATION_PATH),
            "selectedPolicy": policy,
            "selectedPolicySha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "guardedSourceIdentity": source._identity()}


def _paths(model_dir=None):
    directory = Path(model_dir) if model_dir is not None else PRIVATE
    return directory / ARTIFACT_NAME, directory / STATUS_NAME


def _load_artifact(model_dir=None):
    path, _ = _paths(model_dir)
    key = str(path.resolve())
    try: signature = (path.stat().st_mtime_ns, path.stat().st_size)
    except OSError: return None
    identity = _identity()
    with _LOCK:
        prior = _MODELS.get(key)
        if prior is not None and prior[0] == signature and prior[1].get("identity") == identity:
            return prior[1]
    try:
        value = pickle.loads(path.read_bytes())
        if not isinstance(value, dict) or value.get("identity") != identity: return None
        if value.get("modelVersion") != MODEL_VERSION or value.get("adapterVersion") != ADAPTER_VERSION: return None
        with _LOCK: _MODELS[key] = (signature, value)
        return value
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError, AssertionError):
        return None


def _training_support(frame, cutoff, mode):
    if frame.empty:
        return frame, {"available": False, "reason": "training_support_unavailable",
                       "unavailableReasons": ["no_completed_guarded_training_cases"],
                       "trainingCases": 0, "targetDates": 0, "maxTrainingOutcomeEpoch": None}
    eligible = frame.valid & frame.completeEpoch.lt(cutoff) & frame.issueEpoch.ge(cutoff - 28 * 86400)
    train = frame.loc[eligible].sort_values("issueEpoch").copy()
    if not train.empty:
        if not probability.feature_clock_valid(train).all(): raise ValueError("training_feature_clock_guard_failed")
        if train.issueEpoch.duplicated().any(): raise ValueError("duplicate_training_issue")
        expected_lead = 5400 if mode == "initial90" else 3600
        if not (train.startEpoch - train.issueEpoch).eq(expected_lead).all(): raise ValueError("training_lead_mismatch")
        if not (train.endEpoch - train.startEpoch).eq(7200).all(): raise ValueError("training_duration_mismatch")
        # Both heads use the same nonoverlapping four-hour plan cadence.
        if len(train) > 1 and not train.issueEpoch.diff().dropna().ge(14400).all():
            raise ValueError("training_plans_not_four_hour_spaced")
    actual = train.actual.to_numpy(float)
    labels = probability.target_label(actual)
    above = train.fresh.gt(CUTOFF_UG_M3).to_numpy()
    reasons = []
    if len(train) < 50: reasons.append("fewer_than50_complete_spaced_training_cases")
    if train.targetDay.nunique() < 10: reasons.append("fewer_than10_completed_target_dates")
    if len(np.unique(labels)) < 2: reasons.append("training_target_requires_both_classes")
    return train, {"available": not reasons, "reason": "available" if not reasons else "training_support_unavailable",
                   "unavailableReasons": reasons, "trainingCases": len(train),
                   "targetDates": int(train.targetDay.nunique()), "trainingSuccesses": int(labels.sum()),
                   "trainingFailures": int(len(labels) - labels.sum()),
                   "trainingCurrentAboveCutoffCases": int(above.sum()),
                   "trainingCurrentAboveCutoffOpportunities": int((above & (labels == 1)).sum()),
                   "maxTrainingOutcomeEpoch": int(train.completeEpoch.max()) if len(train) else None}


def refresh_model(db_path, issue_epoch, *, model_dir=None):
    """Fit selected heads once daily in the service worker, never during a request."""
    cutoff = source.midnight_epoch(issue_epoch)
    if not any(_selected(mode) for mode in MODE_PHASE):
        return {"available": False, "reason": "evaluated_candidate_selection_pending",
                "modelVersion": MODEL_VERSION, "trainingCutoffEpoch": cutoff, "selectedPolicy": selected_policy()}
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
                    "experimental": True, "calibrated": False, "fitCpuThreads": 1,
                    "selectedPolicy": selected_policy(), "identity": identity, "heads": {}}
        models = {}
        try:
            sources = source._source_snapshot(db_path, cutoff - 1, cutoff)
            frames, preparation = _preparation().rebuild_training(*sources)
            metadata.update(sourceSnapshot=sources[-1], preparation=preparation)
            for mode, phase in MODE_PHASE.items():
                if not _selected(mode):
                    metadata["heads"][mode] = {"available": False, "reason": "evaluated_candidate_selection_pending"}
                    continue
                train, support = _training_support(frames[phase], cutoff, mode)
                metadata["heads"][mode] = support
                if not support["available"]: continue
                candidate = SELECTED_CANDIDATES[mode]
                if candidate == "forest":
                    models[mode] = distribution.fit_window_distribution(train, train.actual.to_numpy(float) - train.fresh.to_numpy(float))
                else:
                    models[mode] = probability.fit_window_model(train, candidate=candidate, cutoff_epoch=cutoff)
                support["candidate"] = candidate
            metadata.update(available=bool(models), reason="available" if models else "training_support_unavailable")
        except Exception as error:
            metadata.update(available=False, reason="background_fit_unavailable", unavailableReasons=[type(error).__name__])
            models = {}
        metadata["fittedAtEpoch"] = int(time.time())
        metadata["fitElapsedSeconds"] = round(time.perf_counter() - started, 3)
        value = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION, "identity": identity,
                 "trainingCutoffEpoch": cutoff, "metadata": metadata, "models": models}
        source._atomic(artifact_path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        source._atomic(status_path, (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode())
        signature = (artifact_path.stat().st_mtime_ns, artifact_path.stat().st_size)
        with _LOCK: _MODELS[str(artifact_path.resolve())] = (signature, value)
        return dict(metadata, cacheReloaded=False)


def _window_contract(issue, start, end, mode):
    if mode not in MODE_PHASE: return "unsupported_forecast_mode"
    if not issue < start < end or end - start != 7200: return "invalid_two_hour_ride_window"
    if mode == "initial90" and start - issue != 5400: return "initial_window_must_start90_minutes_after_issue"
    if mode == "recheck60" and not 3300 <= start - issue <= 3900: return "recheck_outside55_to65_minute_remaining_lead"
    return None


def _target_clock(issue, start, end):
    anchor = issue // 900 * 900
    weights = window_weights(start - anchor, end - anchor)
    return {"featureOriginEpoch": anchor, "windowStartEpoch": start, "windowEndEpoch": end,
            "targetMeanDefinition": "overlap_duration_weighted_complete_15_minute_sensor_medians",
            "bucketWeights": [{"bucketEndEpoch": int(anchor + offset * 900), "weight": float(weight)}
                              for offset, weight in weights.items()],
            "observedOutcomeCompleteAfterEpoch": int(anchor + max(weights) * 900)}


def predict_window(db_path, rows, issue_epoch, sensor_watermark_epoch, reference_pm, *,
                   start_epoch=None, end_epoch=None, mode="initial90", model_dir=None):
    """Read-only inference for the rolling initial or same-window30-minute recheck."""
    issue = int(issue_epoch)
    start = issue + 5400 if start_epoch is None else int(start_epoch)
    end = start + 7200 if end_epoch is None else int(end_epoch)
    try: reference = float(reference_pm)
    except (ValueError, TypeError, OverflowError): reference = math.nan
    initial_issue = start - 5400
    result = {"available": False, "modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
              "forecastIssuedEpoch": issue, "startEpoch": start, "endEpoch": end, "mode": mode,
              "referencePm": reference if math.isfinite(reference) else None,
              "cutoffUgM3": CUTOFF_UG_M3, "chanceMeanAtOrBelowCutoff": None,
              "experimental": True, "calibrated": False, "selectedPolicy": selected_policy(),
              "target": "exact_two_hour_ride_window_mean", "cutoffMeaning": "user_selected_planning_cutoff",
              "planId": f"cycling70-{start}-{end}", "planIssuedEpoch": initial_issue,
              "plannedRecheckEpoch": initial_issue + 1800, "remainingLeadMinutes": (start - issue) / 60}
    invalid = _window_contract(issue, start, end, mode)
    if invalid:
        result.update(reason=invalid, unavailableReasons=[invalid])
        return result
    result["targetClock"] = _target_clock(issue, start, end)
    if not _selected(mode):
        result.update(reason="evaluated_candidate_selection_pending", unavailableReasons=["candidate_not_selected_after_evaluation"])
        return result
    try:
        if not math.isfinite(reference) or reference < 0: raise ValueError("invalid_fresh_reference")
        artifact = _load_artifact(model_dir)
        if artifact is None:
            result.update(reason="background_model_collecting", unavailableReasons=["no_current_cached_estimator"])
            return result
        metadata = artifact["metadata"]
        result.update(trainingCutoffEpoch=artifact["trainingCutoffEpoch"], fittedAtEpoch=metadata["fittedAtEpoch"], modelIdentity=artifact["identity"])
        cutoff = source.midnight_epoch(issue)
        if artifact["trainingCutoffEpoch"] != cutoff:
            result.update(reason="background_model_cutoff_mismatch", unavailableReasons=["cached_fit_is_not_issue_day_midnight"])
            return result
        if metadata["fittedAtEpoch"] > issue:
            result.update(reason="model_not_yet_fitted_at_issue", unavailableReasons=["cached_fit_recorded_after_issue"])
            return result
        support = metadata.get("heads", {}).get(mode, {})
        if not metadata["available"] or not support.get("available") or mode not in artifact["models"]:
            result.update(reason=support.get("reason", metadata["reason"]),
                          unavailableReasons=support.get("unavailableReasons", metadata.get("unavailableReasons", [])))
            return result
        if support["maxTrainingOutcomeEpoch"] >= cutoff:
            raise ValueError("training_labels_not_strictly_before_issue_day_midnight")
        result["trainingSupport"] = support
        run, _ = source._latest_weather(db_path, issue)
        if run is None:
            result.update(reason="issue_weather_unavailable", unavailableReasons=["issue_weather_unavailable"])
            return result
        inputs, reason = _preparation().extract_current(rows, run, issue, start, end, float(sensor_watermark_epoch), reference)
        if inputs is None:
            result.update(reason=reason, unavailableReasons=[reason])
            return result
        frame, provenance = inputs
        if len(frame) != 1 or not probability.feature_clock_valid(frame).all():
            raise ValueError("current_input_feature_clock_invalid")
        if not (int(frame.issueEpoch.iloc[0]) == issue and int(frame.startEpoch.iloc[0]) == start and int(frame.endEpoch.iloc[0]) == end):
            raise ValueError("current_input_target_clock_mismatch")
        if not math.isclose(float(frame.fresh.iloc[0]), reference, rel_tol=0, abs_tol=1e-7):
            raise ValueError("current_input_reference_mismatch")
        candidate = SELECTED_CANDIDATES[mode]
        if candidate == "forest":
            predicted = distribution.predict_window_distribution(artifact["models"][mode], frame, limit=CUTOFF_UG_M3)[0]
            chance = float(predicted["probabilityRideMeanAtOrBelowThreshold"])
            result["rideMeanEstimate"] = {"meanUgM3": predicted["predictedRideMean"], "medianUgM3": predicted["rideMeanMedian"],
                "q10UgM3": predicted["rideMeanQ10"], "q90UgM3": predicted["rideMeanQ90"],
                "effectiveTrainingCases": predicted["effectiveTrainingCases"], "calibrated": False}
        else:
            chance = float(probability.predict_window_probability(artifact["models"][mode], frame)[0])
        if not math.isfinite(chance) or not 0 <= chance <= 1: raise ValueError("invalid_planning_probability")
        values = frame[list(probability.FEATURES)].replace([np.inf, -np.inf], np.nan).to_numpy(float)[0]
        normalized = [float(value) if np.isfinite(value) else None for value in values]
        result.update(provenance, available=True, reason="available", unavailableReasons=[], chanceMeanAtOrBelowCutoff=chance,
                      featureColumns=list(probability.FEATURES), featureValues=normalized,
                      featureValuesSha256=hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest(),
                      probabilityCandidate=candidate, unvalidatedLiveCadence=True,
                      trainingFitCadence="four_hour_initial_plans_and_exact30_minute_rechecks")
        return result
    except Exception as error:
        result.update(reason="cycling_window_inference_unavailable", unavailableReasons=[type(error).__name__])
        return result


def record_issued_prediction(result, *, output_path=None, recorded_epoch=None):
    """Call only after the initial forecast or explicit recheck was published."""
    recorded = int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue = int(result["forecastIssuedEpoch"])
    if not 0 <= recorded - issue <= 120:
        raise ValueError("Only a fresh actual publication may be recorded as originally issued")
    path = Path(output_path) if output_path is not None else PRIVATE / ISSUE_NAME
    def identity(item):
        return (str(path.resolve()), int(item["forecastIssuedEpoch"]), item.get("planId"), item.get("mode"),
                item.get("trainingCutoffEpoch"), item.get("featureValuesSha256"), item.get("reason"))
    key = identity(result)
    record = {**result, "recordedEpoch": recorded, "publishedEpoch": recorded, "isOriginallyIssued": True,
              "predictionRole": "genuinely_issued_live_cycling_window70_prediction"}
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path.resolve()) not in _ISSUE_PATHS_READ:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2_000_000))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()
                for line in tail:
                    try: _RECORDED[identity(json.loads(line))] = None
                    except (ValueError, KeyError, TypeError): continue
            _ISSUE_PATHS_READ.add(str(path.resolve()))
        if key in _RECORDED: return False
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        _RECORDED[key] = None
        while len(_RECORDED) > 4096: _RECORDED.popitem(last=False)
    return True
