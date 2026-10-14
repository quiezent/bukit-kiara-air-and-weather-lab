"""Dedicated fixed +90-minute PM2.5 learner with issue-visible inputs.

Only fit/refresh_model train. Prediction uses a sealed cached fit and publishes
its physical concentration literally. The estimator owns both the fresh-
reference target transform and its inverse; the publisher never adds a sensor
offset, selects a forecast, clips a point, or substitutes persistence.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from collection_quality import bucket_coverage
from forecast_clock import TARGET_VERSION, point_weights
import fresh_sensor_features as sensor

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_PATH = ROOT / "private" / "arrival-numeric-model-v1.pkl"
MODEL_VERSION = "fresh_weather_hgb_arrival90_delta_v1_20261008"
OUTPUT_COLUMNS = ("arrivalPoint",)
MODEL_PARAMETERS = {
    "loss": "squared_error", "learning_rate": 0.05, "max_iter": 150,
    "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1.0,
    "early_stopping": False, "random_state": 20261008,
}
TRAINING_LOOKBACK_DAYS = 28
TRAINING_CADENCE_SECONDS = 300
EMBARGO_SECONDS = 120 * 60
MINIMUM_TRAINING_ORIGINS = 300
MINIMUM_TRAINING_DAYS = 10
MYT = timezone(timedelta(hours=8))
_FIT_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_LOADED = {}


def _inputs():
    # Lazy import allows independent target/estimator verification without a
    # database or weather archive. The identity always binds the real builder.
    import arrival_feature_inputs
    return arrival_feature_inputs


def midnight_epoch(issue_epoch):
    local = datetime.fromtimestamp(int(issue_epoch), MYT)
    return int(local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def model_identity():
    inputs = _inputs()
    identity = {
        "modelVersion": MODEL_VERSION, "parameters": dict(MODEL_PARAMETERS),
        "outputColumns": list(OUTPUT_COLUMNS), "featureColumns": list(inputs.FEATURE_COLUMNS),
        "trainingCadenceSeconds": TRAINING_CADENCE_SECONDS,
        "trainingLookbackDays": TRAINING_LOOKBACK_DAYS, "embargoSeconds": EMBARGO_SECONDS,
        "minimumTrainingOrigins": MINIMUM_TRAINING_ORIGINS,
        "minimumTrainingDays": MINIMUM_TRAINING_DAYS,
        "responseRepresentation": "Exact +90 absolute target minus the SAME issue-time freshPm25 feature",
        "inverseTargetTransform": "Estimator.predict adds query freshPm25 to its learned delta; publisher copies its physical output",
        "targetDefinition": "Interpolated complete 15-minute bucket-median proxy at exact issue+90 minutes; no ride outcome dependency",
        "missingInputs": "Native HGB missing-value handling; freshPm25 itself must be finite",
        "numericalSelectionApplied": False, "persistenceFallback": False,
        "numericModuleSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "featureModuleSha256": hashlib.sha256(Path(inputs.__file__).read_bytes()).hexdigest(),
        "sensorFeatureModuleSha256": hashlib.sha256(Path(sensor.__file__).read_bytes()).hexdigest(),
        "targetClockModuleSha256": hashlib.sha256(Path(__import__('forecast_clock').__file__).read_bytes()).hexdigest(),
        "coverageModuleSha256": hashlib.sha256(Path(__import__('collection_quality').__file__).read_bytes()).hexdigest(),
        "sklearnVersion": sklearn.__version__,
    }
    identity["identitySha256"] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def observed_series(rows, cutoff_epoch=None):
    """Completed observed PM medians, preserving holes and coverage failures."""
    records = {}
    for original in rows:
        row = dict(original)
        try:
            stamp, value = row["epoch"], row["pm02"]
            if isinstance(stamp, bool) or isinstance(value, bool):
                continue
            stamp, value = float(stamp), float(value)
            if not math.isfinite(stamp) or not math.isfinite(value) or value < 0:
                continue
            if cutoff_epoch is not None and stamp > cutoff_epoch:
                continue
            records[stamp] = value
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not records:
        return pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    epochs = sorted(records)
    raw = pd.DataFrame({"pm02": [records[stamp] for stamp in epochs]},
                       index=pd.to_datetime(epochs, unit="s", utc=True))
    medians = raw.pm02.resample("900s", closed="right", label="right").median()
    medians.loc[~bucket_coverage(raw).forecastEligible] = np.nan
    if cutoff_epoch is not None:
        medians.loc[medians.index.asi8 // 10**9 > cutoff_epoch] = np.nan
    return medians


def targets_for_issues(pm_series, issue_epochs):
    """Use only +90 target buckets; missing later ride bins cannot erase it."""
    issues = np.asarray(issue_epochs, dtype=np.int64)
    responses = np.full(len(issues), np.nan)
    complete = np.zeros(len(issues), dtype=np.int64)
    if not isinstance(pm_series.index, pd.DatetimeIndex) or not pm_series.index.is_unique:
        raise ValueError("Observed bucket labels must be a unique DatetimeIndex")
    mapping = {int(stamp): float(value) for stamp, value in
               zip(pm_series.index.asi8 // 10**9, pm_series.to_numpy(dtype=float))}
    for index, issue in enumerate(issues):
        anchor = int(issue // 900 * 900)
        weights = point_weights(int(issue - anchor) + 5400)
        values = np.asarray([mapping.get(anchor + offset*900, np.nan) for offset in weights])
        complete[index] = anchor + max(weights)*900
        if np.isfinite(values).all():
            responses[index] = float(values @ np.asarray(list(weights.values())))
    return pd.DataFrame({"arrivalPoint": responses},
                        index=pd.Index(issues, name="issueEpoch")), complete


def _matrix_hash(index, columns, matrix, responses=None):
    digest = hashlib.sha256(json.dumps(list(columns), separators=(",", ":")).encode())
    digest.update(np.asarray(index, dtype="<i8").tobytes())
    for block in (matrix,) if responses is None else (matrix, responses):
        canonical = np.asarray(block, dtype="<f8").copy()
        canonical[~np.isfinite(canonical)] = np.nan
        digest.update(canonical.tobytes())
    return digest.hexdigest()


class FreshReferenceArrivalRegressor(RegressorMixin, BaseEstimator):
    """The trained estimator maps issue features to physical PM concentration."""
    def __init__(self, estimator=None, reference_index=0):
        self.estimator = estimator
        self.reference_index = reference_index

    def fit(self, matrix, absolute_targets):
        matrix = np.asarray(matrix, dtype=float)
        targets = np.asarray(absolute_targets, dtype=float)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if matrix.ndim != 2 or targets.ndim != 1 or len(matrix) != len(targets):
            raise ValueError("training_matrix_or_target_shape_invalid")
        if not 0 <= self.reference_index < matrix.shape[1]:
            raise ValueError("fresh_reference_index_invalid")
        reference = matrix[:, self.reference_index]
        if not np.isfinite(reference).all() or not np.isfinite(targets).all():
            raise ValueError("training_reference_or_target_unavailable")
        estimator = (HistGradientBoostingRegressor(**MODEL_PARAMETERS) if self.estimator is None
                     else clone(self.estimator))
        with threadpool_limits(limits=1):
            self.estimator_ = estimator.fit(matrix, targets - reference)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("query_feature_shape_invalid")
        reference = matrix[:, self.reference_index]
        if not np.isfinite(reference).all():
            raise ValueError("query_fresh_reference_unavailable")
        with threadpool_limits(limits=1):
            learned = np.asarray(self.estimator_.predict(matrix), dtype=float)
        if learned.shape != reference.shape:
            raise ValueError("arrival_estimator_output_shape_invalid")
        return learned + reference


def _make_estimator(columns=None):
    columns = list(_inputs().FEATURE_COLUMNS if columns is None else columns)
    return FreshReferenceArrivalRegressor(reference_index=columns.index("freshPm25"))


def _subset_features(frame, issues):
    selected = frame.loc[issues].copy()
    provenance = frame.attrs.get("featureProvenance", [])
    if len(provenance) == len(frame):
        positions = frame.index.get_indexer(issues)
        selected.attrs["featureProvenance"] = [provenance[index] for index in positions]
    return selected


def _prepare_inputs(rows, end_epoch, db_path=None, *, lookback_days=TRAINING_LOOKBACK_DAYS):
    inputs = _inputs()
    if db_path is not None:
        return inputs.load_inputs(db_path,
                                  int(end_epoch) - int(lookback_days)*86400 - 3*3600,
                                  int(end_epoch))
    if hasattr(rows, "sensorRows"):
        return rows
    return inputs.prepare_inputs(rows)


def training_data(rows, cutoff_epoch, *, issue_epochs=None, db_path=None, feature_frame=None):
    """Match causal raw/issued-weather features with completed arrival labels."""
    cutoff = int(cutoff_epoch)
    prepared = _prepare_inputs(rows, cutoff, db_path)
    causal_rows = [dict(row) for row in _inputs().source_sensor_rows(prepared)
                   if int(row["epoch"]) <= cutoff]
    if issue_epochs is None:
        lower = cutoff - TRAINING_LOOKBACK_DAYS*86400
        issue_epochs = np.arange((lower // TRAINING_CADENCE_SECONDS + 1)*TRAINING_CADENCE_SECONDS,
                                 cutoff - EMBARGO_SECONDS, TRAINING_CADENCE_SECONDS, dtype=np.int64)
    issues = np.asarray(sorted(set(int(value) for value in issue_epochs)), dtype=np.int64)
    issues = issues[(issues >= cutoff - TRAINING_LOOKBACK_DAYS*86400)
                    & (issues < cutoff - EMBARGO_SECONDS)]
    features = (_inputs().issue_feature_frame(prepared, issues)
                if feature_frame is None else _subset_features(feature_frame, issues))
    targets, complete = targets_for_issues(observed_series(causal_rows, cutoff), issues)
    matrix = features.to_numpy(dtype=float)
    reference = matrix[:, list(features.columns).index("freshPm25")]
    provenance = features.attrs.get("featureProvenance", [])
    available = np.asarray([bool(item.get("available")) for item in provenance], dtype=bool)
    if len(available) != len(issues):
        raise ValueError("Arrival feature provenance is missing")
    valid = available & np.isfinite(reference) & np.isfinite(targets.arrivalPoint) & (complete <= cutoff)
    return features.loc[valid], targets.loc[valid], complete[valid]


def _fit_frames(features, targets, complete, cutoff_epoch):
    if list(features.columns) != list(_inputs().FEATURE_COLUMNS):
        raise ValueError("arrival_feature_columns_invalid")
    if len(features) < MINIMUM_TRAINING_ORIGINS:
        raise ValueError("insufficient_completed_training_origins")
    dates = {datetime.fromtimestamp(int(i), MYT).date() for i in features.index}
    if len(dates) < MINIMUM_TRAINING_DAYS:
        raise ValueError("insufficient_training_dates")
    matrix = features.to_numpy(dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    responses = targets.to_numpy(dtype=float)
    estimator = _make_estimator(features.columns).fit(matrix, responses)
    weather_columns = [name for name in features.columns if name not in sensor.FEATURE_COLUMNS]
    weather_matrix = features[weather_columns].to_numpy(dtype=float)
    weather_support = np.isfinite(weather_matrix)
    metadata = {
        "modelVersion": MODEL_VERSION, "modelIdentity": model_identity(),
        "fittedAtEpoch": int(time.time()), "trainingCutoffEpoch": int(cutoff_epoch),
        "trainingOriginCount": len(features),
        "trainingDateCount": len(dates),
        "earliestTrainingOriginEpoch": int(features.index[0]),
        "latestTrainingOriginEpoch": int(features.index[-1]),
        "latestTrainingOutcomeCompleteEpoch": int(max(complete)),
        "trainingDataHash": _matrix_hash(features.index, features.columns, matrix, responses),
        "inputImputation": "None; native HGB missing-value handling, no target imputation",
        "featureCount": len(features.columns), "featureColumns": list(features.columns),
        "outputColumns": list(OUTPUT_COLUMNS), "calibrated": False,
        "prospectivelyValidated": False, "numericalSelectionApplied": False,
        "persistenceFallback": False, "rawOutputArithmeticApplied": False,
        "estimatorOwnsInverseTargetTransform": True,
        "weatherFeatureColumnCount": len(weather_columns),
        "weatherTrainingNonemptyRowCount": int(weather_support.any(axis=1).sum()),
        "weatherTrainingNonemptyValueCount": int(weather_support.sum()),
        "weatherTrainingNonemptyByColumn": {name: int(weather_support[:, index].sum())
                                           for index, name in enumerate(weather_columns)},
        "weatherInputsLearnedFromTraining": bool(weather_support.any()),
    }
    return {"estimator": estimator, "metadata": metadata}


def fit(rows, cutoff_epoch, *, issue_epochs=None, db_path=None, feature_frame=None):
    frames = training_data(rows, cutoff_epoch, issue_epochs=issue_epochs,
                           db_path=db_path, feature_frame=feature_frame)
    return _fit_frames(*frames, cutoff_epoch)


def fit_prepared(data, cutoff_epoch, *, issue_epochs=None):
    """Read-only evaluator interface using the same cached feature/label cohort."""
    features, labels = data["features"], data["issues"]
    cutoff = int(cutoff_epoch)
    if issue_epochs is None:
        available_origins = features.index
        if "isTrainingGrid" in labels:
            available_origins = labels.index[labels.isTrainingGrid.astype(bool)]
        issue_epochs = available_origins
    issues = np.asarray(issue_epochs, dtype=np.int64)
    issues = np.asarray(sorted(set(issues.tolist())), dtype=np.int64)
    selected = labels.loc[issues]
    responses = selected["arrivalPm"].to_numpy(dtype=float)
    complete = selected["completeEpoch"].to_numpy(dtype=np.int64)
    valid = ((issues >= cutoff - TRAINING_LOOKBACK_DAYS*86400)
             & (issues < cutoff - EMBARGO_SECONDS) & (complete <= cutoff)
             & np.isfinite(responses))
    source_max = selected.get("featureSourceMaxEpoch")
    if source_max is not None:
        source_max = source_max.to_numpy(dtype=float)
        valid &= np.isfinite(source_max) & (source_max <= issues)
    matrix = features.loc[issues]
    reference = matrix["freshPm25"].to_numpy(dtype=float)
    valid &= np.isfinite(reference)
    provenance = features.attrs.get("featureProvenance", [])
    if len(provenance) == len(features):
        positions = features.index.get_indexer(issues)
        valid &= np.asarray([bool(provenance[position].get("available")) for position in positions])
    targets = pd.DataFrame({"arrivalPoint": responses[valid]}, index=features.loc[issues[valid]].index)
    return _fit_frames(matrix.loc[issues[valid]], targets, complete[valid], cutoff)


def save_artifact(artifact, artifact_path=None):
    path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)
    with _LOAD_LOCK:
        _LOADED.pop(str(path.resolve()), None)
    return path


def load_artifact(artifact_path=None):
    path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT_PATH
    try:
        stamp, key = (path.stat().st_mtime_ns, path.stat().st_size), str(path.resolve())
        identity = model_identity()
        with _LOAD_LOCK:
            cached = _LOADED.get(key)
            if cached and cached[0] == stamp:
                return cached[1] if cached[1]["metadata"].get("modelIdentity") == identity else None
            artifact = pickle.loads(path.read_bytes())
            metadata = artifact["metadata"]
            if metadata.get("modelIdentity") != identity:
                return None
            if metadata.get("featureColumns") != list(_inputs().FEATURE_COLUMNS):
                return None
            _LOADED[key] = (stamp, artifact)
            return artifact
    except (OSError, ValueError, KeyError, TypeError, ImportError, pickle.UnpicklingError, EOFError):
        return None


def refresh_model(db_path, *, artifact_path=None, issue_epoch=None, cutoff_epoch=None):
    """Background fitting only; archived outcomes cannot pass today's midnight."""
    issue = int(time.time()) if issue_epoch is None else int(issue_epoch)
    cutoff = midnight_epoch(issue) if cutoff_epoch is None else int(cutoff_epoch)
    if cutoff > midnight_epoch(issue):
        raise ValueError("training_cutoff_after_issue_day_midnight")
    with _FIT_LOCK:
        existing = load_artifact(artifact_path)
        if existing and existing["metadata"]["trainingCutoffEpoch"] == cutoff:
            return {"available": True, "refreshed": False, **existing["metadata"]}
        prepared = _prepare_inputs(None, cutoff, db_path)
        artifact = fit(prepared, cutoff)
        path = save_artifact(artifact, artifact_path)
        return {"available": True, "refreshed": True, "artifactPath": str(path), **artifact["metadata"]}


def forecast_clock(issue_epoch):
    issue = int(issue_epoch)
    return {"targetVersion": TARGET_VERSION, "forecastIssuedEpoch": issue,
            "featureAnchorEpoch": issue, "featureAnchorAgeSeconds": 0,
            "arrivalTargetEpoch": issue + 5400, "targetResolutionMinutes": 15,
            "arrivalTargetDefinition": "Interpolated complete 15-minute bucket-median proxy at exact +90 min"}


def predict(rows, issue_epoch, *, db_path=None, artifact_path=None, artifact=None,
            allow_reconstruction=False, query_features=None):
    """Inference only. Never fit, gate on performance, or replace model output."""
    issue = int(issue_epoch)
    base = {
        "available": False, "modelVersion": MODEL_VERSION, "forecastIssuedEpoch": issue,
        "forecastClock": forecast_clock(issue), "arrivalPoint": None, "rawOutputUgM3": None,
        "reason": None, "numericalSelectionApplied": False, "persistenceFallback": False,
        "rawOutputArithmeticApplied": False, "calibrated": False,
        "estimatorOwnsInverseTargetTransform": True, "reconstructed": bool(allow_reconstruction),
        "forecastRole": "counterfactual_reconstruction" if allow_reconstruction else "live_cached_model_output",
        "prospectivelyValidated": False,
    }
    prepared = _prepare_inputs(rows, issue, db_path, lookback_days=0)
    query = (_inputs().issue_feature_frame(prepared, [issue])
             if query_features is None else _subset_features(query_features, [issue]))
    provenance = query.attrs.get("featureProvenance", [])
    if len(provenance) != 1:
        return {**base, "reason": "arrival_feature_provenance_unavailable"}
    lineage = provenance[0]
    base.update(inputLineage=lineage, freshReferencePm25=lineage.get("freshReferencePm25"),
                featureSourceMaxEpoch=lineage.get("featureSourceMaxEpoch"))
    if not lineage.get("available"):
        return {**base, "reason": lineage.get("reason") or "arrival_features_unavailable"}
    for name in ("featureSourceMaxEpoch", "weatherAvailableEpoch"):
        value = lineage.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value > issue):
            return {**base, "reason": "arrival_input_after_issue"}
    artifact = load_artifact(artifact_path) if artifact is None else artifact
    if artifact is None:
        return {**base, "reason": "arrival_numeric_model_asset_unavailable"}
    metadata = artifact.get("metadata") or {}
    fitted = metadata.get("fittedAtEpoch")
    if isinstance(fitted, bool) or not isinstance(fitted, (int, float)) or not math.isfinite(fitted):
        return {**base, "reason": "model_fit_timestamp_unavailable"}
    if fitted > issue and not allow_reconstruction:
        return {**base, "reason": "model_not_fitted_at_issue"}
    cutoff, complete = metadata.get("trainingCutoffEpoch"), metadata.get("latestTrainingOutcomeCompleteEpoch")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (cutoff, complete)):
        return {**base, "reason": "model_training_clock_unavailable"}
    if cutoff > midnight_epoch(issue):
        return {**base, "reason": "training_cutoff_after_issue_day_midnight"}
    if complete > issue:
        return {**base, "reason": "training_outcomes_after_issue"}
    if metadata.get("modelIdentity") != model_identity():
        return {**base, "reason": "arrival_numeric_model_identity_mismatch"}
    if list(query.columns) != metadata.get("featureColumns"):
        return {**base, "reason": "arrival_query_feature_columns_invalid"}
    matrix = query.to_numpy(dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    if not np.isfinite(matrix[:, list(query.columns).index("freshPm25")]).all():
        return {**base, "reason": "query_fresh_reference_unavailable"}
    try:
        raw = np.asarray(artifact["estimator"].predict(matrix), dtype=float)
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        return {**base, "reason": "arrival_numeric_estimator_unavailable", "errorType": type(error).__name__}
    if raw.shape != (1,) or not np.isfinite(raw).all():
        return {**base, "reason": "arrival_numeric_estimator_output_invalid"}
    point = float(raw[0])
    return {**base, "available": True, "arrivalPoint": point,
            "rawOutputUgM3": {"arrivalPoint": point}, "reason": None,
            "queryFeaturesHash": _matrix_hash(query.index, query.columns, matrix),
            "training": dict(metadata), "modelIdentity": metadata.get("modelIdentity"),
            "outputsPublishedWithoutArithmetic": True}
