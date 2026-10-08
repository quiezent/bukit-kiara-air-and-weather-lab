"""Live inference adapter for the frozen first-crossing competing-risk model.

Only ``refresh_model`` trains, in the service's independent background worker.
``predict_event`` does a bounded source read and read-only cached inference.
``record_issued_prediction`` is called separately after real publication; it
never labels an offline replay as a genuinely issued forecast.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sqlite3
import threading
import time
import uuid

import numpy as np
import pandas as pd
from event_qualification import (apply_event_qualification, qualification_metadata,
                                 publication_metadata)

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research" / "first_crossing_hazard"
PRIVATE = ROOT / "private"
MODEL_VERSION = "ttdi_first_crossing_competing_risk_logistic_v1"
ADAPTER_VERSION = "ttdi_first_crossing_live_adapter_v2_qualified_diagnostics"
ARTIFACT_NAME = "first-crossing-model.pkl"
STATUS_NAME = "first-crossing-model-status.json"
ISSUE_NAME = "first-crossing-issued.jsonl"
_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_MODELS = {}
_RECORDED = OrderedDict()
_ISSUE_PATHS_READ = set()
_SOURCE_STATE = {}

_spec = importlib.util.spec_from_file_location("first_crossing_live_frozen_replay", RESEARCH / "prospective_replay.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)
h = p.h


def midnight_epoch(issue_epoch):
    return int(pd.Timestamp(int(issue_epoch), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur").normalize().timestamp())


def _paths(model_dir):
    directory = Path(model_dir) if model_dir is not None else PRIVATE
    return directory / ARTIFACT_NAME, directory / STATUS_NAME


def _identity():
    # Original frozen trial is checked without editing or re-freezing it.
    files = [Path(__file__), RESEARCH / "hazard_trial.py", RESEARCH / "prospective_replay.py",
             RESEARCH / "policy.json", ROOT / "research/evaluated_models/base_features.pkl",
             ROOT / "research/evaluated_models/preparation.json"]
    files.extend(ROOT / name for name in ("afternoon_direction_features.py", "forecast_clock.py", "rain_weather_features.py",
                                         "weather_session_features.py", "window_pm_predictor.py", "forecast_freshness.py", "collection_quality.py"))
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
    with _LOCK:
        if _SOURCE_STATE.get("signature") == signature: return _SOURCE_STATE["identity"]
    h.checked_policy()
    identity = {"adapterSha256": h.digest(__file__), "frozenRunnerSha256": h.digest(RESEARCH / "hazard_trial.py"),
                "replayHelperSha256": h.digest(RESEARCH / "prospective_replay.py"),
                "helperSourceSha256": p.source_hashes(), "researchPolicySha256": h.digest(RESEARCH / "policy.json")}
    with _LOCK: _SOURCE_STATE.update(signature=signature, identity=identity)
    return identity


def _atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _source_snapshot(db_path, cutoff, fit_cutoff):
    earliest = max(h.epoch(h.POLICY["sourceStart"]), fit_cutoff - 28 * 86400)
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        conn.row_factory = sqlite3.Row
        readings = [dict(row) for row in conn.execute(
            "SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch", (earliest - 86400, cutoff))]
        publications = [tuple(row) for row in conn.execute(
            "SELECT issued_epoch,sensor_epoch FROM dashboard_forecast_issues WHERE issued_epoch>=? AND issued_epoch<=? ORDER BY issued_epoch,issue_id",
            (earliest - 86400, cutoff))]
        weather_rows = [tuple(row) for row in conn.execute(
            "SELECT fetched_epoch,source,payload FROM weather_forecast_runs WHERE fetched_epoch>=? AND fetched_epoch<=? ORDER BY fetched_epoch,payload",
            (earliest - 7200, cutoff))]
    runs = h.rain.RunArchive(tuple(h.rain._parse_run(float(f), str(s), str(payload)) for f, s, payload in weather_rows),
                            tuple(float(row[0]) for row in weather_rows), float(cutoff), p._hash_records(weather_rows), str(path))
    manifest = {"cutoffEpoch": cutoff, "earliestTrainingIssueEpoch": earliest, "databaseReadOnly": True,
                "singleReadTransaction": True, "providerFetches": 0, "sensorRows": len(readings),
                "publicationRows": len(publications), "weatherRuns": len(weather_rows),
                "sensorRowsSha256": p._hash_records(readings), "publicationRowsSha256": p._hash_records(publications),
                "weatherRowsSha256": runs.revision,
                "maxSensorEpoch": max((r["epoch"] for r in readings), default=None),
                "maxPublicationEpoch": max((r[0] for r in publications), default=None),
                "maxWeatherFetchedEpoch": max((r[0] for r in weather_rows), default=None)}
    return readings, publications, runs, manifest


def _training_set(bundle, cutoff):
    issues = bundle["issues"]
    if issues.empty:
        return pd.DataFrame(), pd.DataFrame(), {"trainingSupport": dict.fromkeys(h.NAMES, 0),
            "originalGuardedHourlyTrainingIssues": 0, "fixedFourHourTrainingIssues": 0,
            "completedTargetDays": 0, "riskRows": 0, "maxTrainingOutcomeEpoch": None,
            "unavailableReasons": ["missing_guarded_training_inputs"]}
    hourly = issues.loc[issues.allSixCovered & issues.completeEpoch.lt(cutoff) &
                        issues.issueEpoch.ge(cutoff - 28 * 86400) & issues.issueEpoch.lt(cutoff)]
    train = hourly.loc[h.sparse_mask(hourly)].sort_values("issueEpoch").copy()
    risk = h.risk_rows(train, bundle["stepFeatures"])
    counts = h.support(train)
    reasons = []
    if len(hourly) < 120: reasons.append("fewer_than120_original_guarded_hourly_training_issues")
    if train.targetDay.nunique() < 10: reasons.append("fewer_than10_completed_target_dates")
    if len(risk) < 120: reasons.append("fewer_than120_pooled_at_risk_rows_not_independent_cases")
    for name, minimum in h.POLICY["minimumFixedFourHourTrainingSupport"].items():
        if counts[name] < minimum: reasons.append("insufficient_spaced_" + name + "_training_cases")
    return train, risk, {"trainingSupport": counts, "originalGuardedHourlyTrainingIssues": len(hourly),
                        "fixedFourHourTrainingIssues": len(train), "completedTargetDays": int(train.targetDay.nunique()),
                        "riskRows": len(risk), "maxTrainingOutcomeEpoch": int(train.completeEpoch.max()) if len(train) else None,
                        "unavailableReasons": reasons}


def _load_artifact(model_dir=None):
    artifact, _ = _paths(model_dir)
    key = str(artifact.resolve())
    try: signature = (artifact.stat().st_mtime_ns, artifact.stat().st_size)
    except OSError: return None
    identity = _identity()
    with _LOCK:
        cached = _MODELS.get(key)
        if cached is not None and cached[0] == signature and cached[1].get("identity") == identity: return cached[1]
    # Local, service-owned model pickle, atomically replaced by this adapter.
    try:
        value = pickle.loads(artifact.read_bytes())
        if value.get("identity") != identity or value.get("adapterVersion") != ADAPTER_VERSION: return None
        if value.get("modelVersion") != MODEL_VERSION: return None
        with _LOCK: _MODELS[key] = (signature, value)
        return value
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError, AssertionError):
        return None


def refresh_model(db_path, issue_epoch, *, model_dir=None):
    """Fit once per Malaysia calendar day, on CPU1 and before-midnight labels.

    This is the ONLY fitting entry point. Call from a service background thread,
    never from the HTTP handler or ``_compute_analysis``.
    """
    cutoff = midnight_epoch(issue_epoch)
    artifact_path, status_path = _paths(model_dir)
    with _FIT_LOCK:
        identity = _identity()
        prior = _load_artifact(model_dir)
        if (prior is not None and prior["trainingCutoffEpoch"] == cutoff and
                prior["metadata"].get("reason") != "background_fit_unavailable"):
            return dict(prior["metadata"], cacheReloaded=True)
        started = time.perf_counter()
        now = int(time.time())
        metadata = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
                    "trainingCutoffEpoch": cutoff, "fittedAtEpoch": now, "experimental": True,
                    "calibrated": False, "trainingLookbackDays": 28, "featureColumns": list(h.FEATURES),
                    "parameters": h.LOGISTIC, "fitCpuThreads": 1, "identity": identity,
                    "sourceRole": "reconstructed_guarded_original_sensor_publication_and_weather_archive"}
        try:
            sources = _source_snapshot(db_path, cutoff - 1, cutoff)
            bundle, rebuild = p.rebuild_inputs(*sources)
            train, risk, support = _training_set(bundle, cutoff)
            metadata.update(support, sourceSnapshot=sources[-1], inputRebuild=rebuild)
            estimator = None
            if not support["unavailableReasons"]:
                assert train.completeEpoch.lt(cutoff).all() and h.feature_clock_valid(train).all()
                estimator = h.fit_hazard(risk)
            metadata.update(available=estimator is not None, reason="available" if estimator is not None else "training_support_unavailable")
        except Exception as error:
            metadata.update(available=False, reason="background_fit_unavailable", unavailableReasons=[type(error).__name__],
                            trainingSupport=dict.fromkeys(h.NAMES, 0))
            estimator = None
        metadata["fittedAtEpoch"] = int(time.time())
        metadata["fitElapsedSeconds"] = round(time.perf_counter() - started, 3)
        artifact = {"modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION, "identity": identity,
                    "trainingCutoffEpoch": cutoff, "metadata": metadata, "estimator": estimator}
        _atomic(artifact_path, pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
        _atomic(status_path, (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        # Reader sees one complete immutable estimator/metadata snapshot.
        signature = (artifact_path.stat().st_mtime_ns, artifact_path.stat().st_size)
        with _LOCK: _MODELS[str(artifact_path.resolve())] = (signature, artifact)
        return dict(metadata, cacheReloaded=False)


def _latest_weather(db_path, issue):
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)) as conn:
        conn.execute("PRAGMA query_only=ON")
        row = conn.execute("SELECT fetched_epoch,source,payload FROM weather_forecast_runs WHERE fetched_epoch<=? "
                           "ORDER BY fetched_epoch DESC,payload DESC LIMIT 1", (issue,)).fetchone()
    if row is None: return None, None
    run = h.rain._parse_run(float(row[0]), str(row[1]), str(row[2]))
    archive = h.rain.RunArchive((run,), (run.fetched_epoch,), float(issue), run.payload_hash, str(path))
    description = h.weather.describe_window(archive, issue, issue + 5400, issue + 5400)
    return run, description


def _current_inputs(db_path, rows, issue, watermark, reference):
    """Current measurement snapshot; no prior publication watermark required."""
    if not (math.isfinite(watermark) and watermark <= issue): return None, "invalid_sensor_watermark"
    bounded = []
    for source in rows:
        row = dict(source)
        try: observed = float(row["epoch"])
        except (KeyError, TypeError, ValueError, OverflowError): continue
        if math.isfinite(observed) and issue - 4 * 3600 <= observed <= watermark:
            bounded.append({"epoch": observed, **{name: row.get(name) for name in ("pm02", "atmp", "rhum")}})
    bounded.sort(key=lambda row: row["epoch"])
    from forecast_freshness import raw_arrays
    epochs, values = raw_arrays(bounded, issue)
    fresh_mask = epochs > issue - 300
    if not fresh_mask.any(): return None, "fresh_sensor_reference_unavailable"
    fresh = float(np.median(values[fresh_mask])); fresh_epoch = float(epochs[fresh_mask][-1])
    if issue - fresh_epoch > 240: return None, "fresh_sensor_reference_stale"
    if not math.isfinite(reference) or reference < 0 or not math.isclose(reference, fresh, rel_tol=0, abs_tol=1e-7):
        return None, "fresh_sensor_reference_mismatch"
    origin = issue // 900 * 900
    frame = h.causal.strict_frame(bounded, issue)
    stamp = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    if frame.empty or stamp not in frame.index: return None, "closed_sensor_bucket_unavailable"
    base = h.causal.base_features(frame).loc[stamp]
    if not np.isfinite(base.closedLevel): return None, "closed_sensor_bucket_unavailable"
    run, description = _latest_weather(db_path, issue)
    if run is None or not description["available"] or run.fetched_epoch > issue or issue - run.fetched_epoch > 7200:
        return None, "issue_weather_unavailable_or_stale"
    local = {name: (fresh if name == "fresh" else fresh - base.closedLevel if name == "freshOffset" else base[name]) for name in h.LOCAL}
    step_records = []
    for step, lead in enumerate(h.LEADS):
        record = dict(local)
        record.update(h.step_weather(run, issue, step))
        record.update({name: float(i == step) for i, name in enumerate(h.INDICATORS)})
        record.update(issueEpoch=issue, step=step, leadMinutes=lead)
        step_records.append(record)
    steps = pd.DataFrame(step_records)
    maximum = max(row["epoch"] for row in bounded)
    feature_max = max(row["epoch"] for row in bounded if row["epoch"] <= origin)
    feature_array = steps[list(h.FEATURES)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    normalized = [[float(value) if np.isfinite(value) else None for value in row] for row in feature_array]
    provenance = {"featureOriginEpoch": origin, "sensorWatermarkEpoch": int(watermark),
        "freshReferenceEpoch": int(fresh_epoch), "inputMaxMeasurementEpoch": int(maximum),
        "featureSourceMaxEpoch": int(feature_max), "weatherFetchedEpoch": run.fetched_epoch,
        "weatherPayloadHash": run.payload_hash, "weatherAvailabilityProvenance": "legacy_archived_fetch_timestamp",
        "sourceRole": "current_measurement_snapshot_for_publication",
        "featureColumns": list(h.FEATURES), "featureValues": normalized,
        "featureValuesSha256": hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest(),
        "trainingFitCadence": "fixed_four_hour_origins_plus300_seconds",
        "unvalidatedLiveCadence": True, "liveIssueOffsetSeconds": issue - origin}
    assert feature_max <= maximum <= watermark <= issue and fresh_epoch <= watermark
    return (steps, provenance), None


def predict_event(db_path, rows, issue_epoch, sensor_watermark_epoch, reference_pm, *, model_dir=None):
    """Read-only prediction; NEVER prepares training data or fits any estimator."""
    issue = int(issue_epoch)
    result = {"available": False, "modelVersion": MODEL_VERSION, "adapterVersion": ADAPTER_VERSION,
        "experimental": True, "calibrated": False, "forecastIssuedEpoch": issue, "validUntilEpoch": issue + 5400,
        "referencePm": float(reference_pm) if reference_pm is not None and np.isfinite(reference_pm) else None,
        "changeThresholdUgM3": 10, "target": "first_sampled_crossing_within90_minutes",
        "direction": "unresolved", "directionCallProbability": .5, "horizons": []}
    result.update(qualification_metadata(issue), diagnosticDirection="unresolved",
                  diagnosticDirectionAvailable=False)
    try:
        artifact = _load_artifact(model_dir)
        if artifact is None:
            result.update(reason="background_model_collecting", unavailableReasons=["no_current_cached_estimator"])
            return result
        metadata = artifact["metadata"]
        result["modelIdentity"] = artifact["identity"]
        for key in ("trainingCutoffEpoch", "trainingSupport", "fittedAtEpoch", "completedTargetDays",
                    "fixedFourHourTrainingIssues", "riskRows", "maxTrainingOutcomeEpoch"):
            if key in metadata: result[key] = metadata[key]
        cutoff = midnight_epoch(issue)
        # Older replay cannot use a model trained later; yesterday's model is
        # also unavailable until today's independent background fit is ready.
        if artifact["trainingCutoffEpoch"] != cutoff:
            result.update(reason="background_model_cutoff_mismatch", unavailableReasons=["cached_fit_is_not_issue_day_midnight"])
            return result
        if metadata["fittedAtEpoch"] > issue:
            result.update(reason="model_not_yet_fitted_at_issue", unavailableReasons=["cached_fit_recorded_after_issue"])
            return result
        if not metadata["available"] or artifact["estimator"] is None:
            result.update(reason=metadata["reason"], unavailableReasons=metadata.get("unavailableReasons", []))
            return result
        if metadata["maxTrainingOutcomeEpoch"] >= cutoff:
            raise ValueError("training_labels_not_strictly_before_issue_day_midnight")
        inputs, reason = _current_inputs(db_path, rows, issue, float(sensor_watermark_epoch), float(reference_pm))
        if inputs is None:
            result.update(reason=reason, unavailableReasons=[reason])
            return result
        steps, provenance = inputs
        probabilities = h.cumulative_incidence(h.predictions(artifact["estimator"], steps))[0]
        horizons = [{"leadMinutes": lead, "probabilityDrop": float(values[0]),
                     "probabilityNoCrossing": float(values[1]), "probabilityRise": float(values[2])}
                    for lead, values in zip(h.LEADS, probabilities)]
        result.update(provenance, horizons=horizons, **{name: horizons[-1][name] for name in ("probabilityRise", "probabilityDrop", "probabilityNoCrossing")})
        call = int(h.direction_calls(probabilities[-1:])[0])
        result.update(available=True, reason="available", unavailableReasons=[],
                      direction="drop" if call == -1 else "rise" if call == 1 else "unresolved",
                      probabilityAnyCrossing=float(probabilities[-1, 0] + probabilities[-1, 2]))
        return apply_event_qualification(result)
    except Exception as error:
        result.update(reason="event_inference_unavailable", unavailableReasons=[type(error).__name__])
        return result


def record_issued_prediction(result, *, output_path=None, recorded_epoch=None, input_lineage=None):
    """Append ONLY an actually published fresh issue; unavailable cases count.

    Call at publication, after computation. Replay/test callers never call this.
    No history row or target outcome is read here.
    """
    recorded = int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue = int(result["forecastIssuedEpoch"])
    if not 0 <= recorded - issue <= 120:
        raise ValueError("Only fresh actual publication may be recorded as originally issued")
    path = Path(output_path) if output_path is not None else PRIVATE / ISSUE_NAME
    identity = (str(path.resolve()), issue, result.get("trainingCutoffEpoch"), result.get("featureValuesSha256"), result.get("reason"))
    record = {**result, "recordedEpoch": recorded, "publishedEpoch": recorded,
              "isOriginallyIssued": True, "predictionRole": "genuinely_issued_live_event_prediction",
              **publication_metadata(result, input_lineage)}
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if str(path.resolve()) not in _ISSUE_PATHS_READ:
            # Bound restart deduplication to the last ~2 MB, never scan a growing archive.
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 2_000_000))
                    tail = stream.read().decode("utf-8", errors="replace").splitlines()
                for line in tail:
                    try:
                        old = json.loads(line)
                        key = (str(path.resolve()), int(old["forecastIssuedEpoch"]), old.get("trainingCutoffEpoch"), old.get("featureValuesSha256"), old.get("reason"))
                        _RECORDED[key] = None
                    except (ValueError, KeyError, TypeError): continue
            _ISSUE_PATHS_READ.add(str(path.resolve()))
        if identity in _RECORDED: return False
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        _RECORDED[identity] = None
        while len(_RECORDED) > 4096: _RECORDED.popitem(last=False)
    return True
