"""Fixed regularized linear learner of fresh-reference numeric PM2.5 targets.

Only refresh_model/fit train. Prediction loads an already trained private asset
and returns its literal outputs; missing assets/inputs produce unavailable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from collection_quality import bucket_coverage
import fresh_sensor_features as sensor
from forecast_clock import TARGET_VERSION, point_weights, window_weights

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_PATH = ROOT / "private" / "fresh-numeric-model-v3.pkl"
MODEL_VERSION = "fresh_sequence_ridge_delta_numeric_v3_20261008"
OUTPUT_COLUMNS = ("arrivalPoint", "trailMeanPoint", "trailMinimumPoint", "trailMaximumPoint")
MODEL_PARAMETERS = {"alpha": 100.0, "fit_intercept": True, "solver": "auto"}
TRAINING_LOOKBACK_DAYS = 28
TRAINING_CADENCE_SECONDS = 300
EMBARGO_SECONDS = 330 * 60
MINIMUM_TRAINING_ORIGINS = 240
MYT = timezone(timedelta(hours=8))
_FIT_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_LOADED = {}


def midnight_epoch(issue_epoch):
    local = datetime.fromtimestamp(int(issue_epoch), MYT)
    return int(local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def model_identity():
    identity = {"modelVersion": MODEL_VERSION, "parameters": dict(MODEL_PARAMETERS),
                "outputColumns": list(OUTPUT_COLUMNS), "featureColumns": list(sensor.FEATURE_COLUMNS),
                "trainingCadenceSeconds": TRAINING_CADENCE_SECONDS,
                "trainingLookbackDays": TRAINING_LOOKBACK_DAYS, "embargoSeconds": EMBARGO_SECONDS,
                "numericalSelectionApplied": False, "persistenceFallback": False,
                "responseRepresentation": "All four absolute targets minus the SAME issue-time freshPm25 feature during fit",
                "inverseTargetTransform": "Estimator.predict adds its query freshPm25 to each learned delta; adapters copy physical outputs literally",
                "inputScaling": "Training-only StandardScaler inside the target-transformed Ridge estimator",
                "targetDefinition": "Arrival interpolated 15-minute median proxy; exact overlap-weighted ride mean and complete-bucket minimum/maximum, +90..+210 minutes",
                "numericModuleSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "featureModuleSha256": hashlib.sha256(Path(sensor.__file__).read_bytes()).hexdigest()}
    identity["identitySha256"] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity


def observed_series(rows, cutoff_epoch=None):
    """Observed closed bucket medians with the existing sensor coverage rule."""
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
    coverage = bucket_coverage(raw)
    medians.loc[~coverage.forecastEligible] = np.nan
    if cutoff_epoch is not None:
        medians.loc[medians.index.asi8 // 10**9 > cutoff_epoch] = np.nan
    return medians


def targets_for_issues(pm_series, issue_epochs):
    """Construct four exact issue-clock targets; never fill missing outcomes."""
    issues = np.asarray(issue_epochs, dtype=np.int64)
    responses = np.full((len(issues), 4), np.nan)
    complete = np.zeros(len(issues), dtype=np.int64)
    if not isinstance(pm_series.index, pd.DatetimeIndex) or not pm_series.index.is_unique:
        raise ValueError("Observed bucket labels must be a unique DatetimeIndex")
    mapping = {int(stamp): float(value) for stamp, value in
               zip(pm_series.index.asi8 // 10**9, pm_series.to_numpy(dtype=float))}
    for index, issue in enumerate(issues):
        anchor = int(issue // 900 * 900)
        lag = int(issue - anchor)
        arrival_weights = point_weights(lag + 5400)
        ride_weights = window_weights(lag + 5400, lag + 12600)
        arrival = np.asarray([mapping.get(anchor+offset*900, np.nan) for offset in arrival_weights])
        ride = np.asarray([mapping.get(anchor+offset*900, np.nan) for offset in ride_weights])
        complete[index] = anchor + max(max(arrival_weights), max(ride_weights)) * 900
        if not np.isfinite(arrival).all() or not np.isfinite(ride).all():
            continue
        responses[index] = (float(arrival @ np.asarray(list(arrival_weights.values()))),
                            float(ride @ np.asarray(list(ride_weights.values()))),
                            float(ride.min()), float(ride.max()))
    return pd.DataFrame(responses, index=pd.Index(issues, name="issueEpoch"),
                        columns=OUTPUT_COLUMNS), complete


def _matrix_hash(index, columns, matrix, responses=None):
    digest = hashlib.sha256(json.dumps(list(columns), separators=(",", ":")).encode())
    digest.update(np.asarray(index, dtype="<i8").tobytes())
    for block in (matrix,) if responses is None else (matrix, responses):
        canonical = np.asarray(block, dtype="<f8").copy()
        canonical[~np.isfinite(canonical)] = np.nan
        digest.update(canonical.tobytes())
    return digest.hexdigest()


class FreshReferenceRegressor(RegressorMixin, BaseEstimator):
    """A target-transformed regressor whose predict output is physical PM2.5.

    The reference is a declared model input at fitting AND inference. This
    inversion is part of the fitted estimator, with no publication adjustment.
    """
    def __init__(self, estimator=None, reference_index=0):
        self.estimator = estimator
        self.reference_index = reference_index

    def fit(self, matrix, absolute_targets):
        matrix = np.asarray(matrix, dtype=float)
        targets = np.asarray(absolute_targets, dtype=float)
        if matrix.ndim != 2 or targets.ndim != 2 or len(matrix) != len(targets):
            raise ValueError("training_matrix_or_target_shape_invalid")
        reference = matrix[:, self.reference_index]
        if not np.isfinite(reference).all() or not np.isfinite(targets).all():
            raise ValueError("training_reference_or_target_unavailable")
        if self.estimator is None:
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            estimator = make_pipeline(StandardScaler(), Ridge(**MODEL_PARAMETERS))
        else:
            estimator = clone(self.estimator)
        self.estimator_ = estimator.fit(matrix, targets-reference[:, None])
        self.n_features_in_ = matrix.shape[1]
        self.n_outputs_ = targets.shape[1]
        return self

    def predict(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("query_feature_shape_invalid")
        reference = matrix[:, self.reference_index]
        if not np.isfinite(reference).all():
            raise ValueError("query_fresh_reference_unavailable")
        learned = np.asarray(self.estimator_.predict(matrix), dtype=float)
        return learned + reference[:, None]


def _make_estimator():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         FreshReferenceRegressor(make_pipeline(StandardScaler(), Ridge(**MODEL_PARAMETERS)),
                                                 list(sensor.FEATURE_COLUMNS).index("freshPm25")))


def training_data(rows, cutoff_epoch, *, issue_epochs=None):
    """Build matched raw issue features, excluding embargoed/incomplete targets."""
    cutoff = int(cutoff_epoch)
    causal_rows = [dict(row) for row in rows if int(row["epoch"]) <= cutoff]
    if issue_epochs is None:
        lower = cutoff - TRAINING_LOOKBACK_DAYS * 86400
        issue_epochs = np.arange((lower // TRAINING_CADENCE_SECONDS + 1) * TRAINING_CADENCE_SECONDS,
                                 cutoff - EMBARGO_SECONDS, TRAINING_CADENCE_SECONDS, dtype=np.int64)
    issues = np.asarray(sorted(set(int(value) for value in issue_epochs)), dtype=np.int64)
    issues = issues[(issues >= cutoff - TRAINING_LOOKBACK_DAYS * 86400)
                    & (issues < cutoff - EMBARGO_SECONDS)]
    features = sensor.issue_feature_frame(causal_rows, issues)
    targets, complete = targets_for_issues(observed_series(causal_rows, cutoff), issues)
    matrix = features.to_numpy(dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    responses = targets.to_numpy(dtype=float)
    provenance = features.attrs.get("featureProvenance", [])
    available = np.asarray([bool(item.get("available")) for item in provenance], dtype=bool)
    if len(available) != len(issues):
        raise ValueError("Fresh feature provenance is missing")
    valid = available & np.isfinite(responses).all(axis=1) & (complete <= cutoff)
    valid &= np.isfinite(matrix).any(axis=1)
    return features.loc[valid], targets.loc[valid], complete[valid]


def fit(rows, cutoff_epoch, *, issue_epochs=None):
    """Fit the fixed joint model only to pre-cutoff completed historical targets."""
    features, targets, complete = training_data(rows, cutoff_epoch, issue_epochs=issue_epochs)
    if len(features) < MINIMUM_TRAINING_ORIGINS:
        raise ValueError("insufficient_completed_training_origins")
    matrix, responses = features.to_numpy(dtype=float), targets.to_numpy(dtype=float)
    if np.any(responses[:, 2] > responses[:, 3]):
        raise ValueError("observed_extrema_order_invalid")
    estimator = _make_estimator()
    estimator.fit(matrix, responses)
    metadata = {"modelVersion": MODEL_VERSION, "modelIdentity": model_identity(),
                "fittedAtEpoch": int(time.time()),
                "trainingCutoffEpoch": int(cutoff_epoch), "trainingOriginCount": len(features),
                "trainingDateCount": len({datetime.fromtimestamp(int(i), MYT).date() for i in features.index}),
                "earliestTrainingOriginEpoch": int(features.index[0]),
                "latestTrainingOriginEpoch": int(features.index[-1]),
                "latestTrainingOutcomeCompleteEpoch": int(max(complete)),
                "trainingDataHash": _matrix_hash(features.index, features.columns, matrix, responses),
                "inputImputation": "Training-feature medians only; no outcome imputation",
                "inputScaling": "Training-only StandardScaler inside target-transformed estimator",
                "featureCount": len(features.columns), "featureColumns": list(features.columns),
                "outputColumns": list(OUTPUT_COLUMNS), "calibrated": False,
                "prospectivelyValidated": False, "numericalSelectionApplied": False,
                "persistenceFallback": False, "rawOutputArithmeticApplied": False,
                "estimatorOwnsInverseTargetTransform": True}
    return {"estimator": estimator, "metadata": metadata}


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
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        key = str(path.resolve())
        with _LOAD_LOCK:
            cached = _LOADED.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
            artifact = pickle.loads(path.read_bytes())
            metadata = artifact["metadata"]
            if metadata.get("modelIdentity") != model_identity():
                return None
            if metadata.get("featureColumns") != list(sensor.FEATURE_COLUMNS):
                return None
            _LOADED[key] = (stamp, artifact)
            return artifact
    except (OSError, ValueError, KeyError, TypeError, ImportError, pickle.UnpicklingError, EOFError):
        return None


def refresh_model(db_path, *, artifact_path=None, issue_epoch=None, cutoff_epoch=None):
    """Background-worker entry point; read only historical rows and save one asset."""
    import time
    issue = int(time.time()) if issue_epoch is None else int(issue_epoch)
    cutoff = midnight_epoch(issue) if cutoff_epoch is None else int(cutoff_epoch)
    if cutoff > midnight_epoch(issue):
        raise ValueError("training_cutoff_after_issue_day_midnight")
    with _FIT_LOCK:
        existing = load_artifact(artifact_path)
        if existing and existing["metadata"]["trainingCutoffEpoch"] == cutoff:
            return {"available": True, "refreshed": False, **existing["metadata"]}
        database = Path(db_path).resolve()
        with sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute(
                "SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch",
                (cutoff - TRAINING_LOOKBACK_DAYS*86400 - 3*3600, cutoff))]
        artifact = fit(rows, cutoff)
        path = save_artifact(artifact, artifact_path)
        return {"available": True, "refreshed": True, "artifactPath": str(path), **artifact["metadata"]}


def forecast_clock(issue_epoch):
    issue = int(issue_epoch)
    return {"targetVersion": TARGET_VERSION, "forecastIssuedEpoch": issue,
            "featureAnchorEpoch": issue, "featureAnchorAgeSeconds": 0,
            "arrivalTargetEpoch": issue+5400, "windowStartEpoch": issue+5400,
            "windowEndEpoch": issue+12600, "targetResolutionMinutes": 15,
            "arrivalTargetDefinition": "Interpolated complete 15-minute bucket-median proxy at exact +90 min",
            "windowMeanTargetDefinition": "Overlap-weighted complete 15-minute bucket medians in exact +90..+210 min",
            "windowExtremaTargetDefinition": "Minimum/maximum complete 15-minute medians with positive overlap; partial edge buckets extend outside window"}


def predict(rows, issue_epoch, *, artifact_path=None, artifact=None, allow_reconstruction=False):
    """Inference only: literal four-head estimator output, without selection."""
    issue = int(issue_epoch)
    base = {"available": False, "modelVersion": MODEL_VERSION,
            "forecastIssuedEpoch": issue, "forecastClock": forecast_clock(issue),
            "reason": None, "numericalSelectionApplied": False, "persistenceFallback": False,
            "rawOutputArithmeticApplied": False, "calibrated": False,
            "estimatorOwnsInverseTargetTransform": True,
            "reconstructed": bool(allow_reconstruction),
            "forecastRole": "counterfactual_reconstruction" if allow_reconstruction else "live_cached_model_output",
            "prospectivelyValidated": False,
            **dict.fromkeys(OUTPUT_COLUMNS), "rawOutputUgM3": None}
    prepared = sensor.prepare_rows(rows)
    provenance = sensor.feature_metadata(prepared, issue)
    base["inputLineage"] = provenance
    base["freshReferencePm25"] = provenance.get("freshReferencePm25")
    base["featureSourceMaxEpoch"] = provenance.get("featureSourceMaxEpoch")
    if not provenance.get("available"):
        return {**base, "reason": provenance.get("reason") or "fresh_sensor_features_unavailable"}
    artifact = load_artifact(artifact_path) if artifact is None else artifact
    if artifact is None:
        return {**base, "reason": "fresh_numeric_model_asset_unavailable"}
    metadata = artifact.get("metadata") or {}
    fitted = metadata.get("fittedAtEpoch")
    if (isinstance(fitted, bool) or not isinstance(fitted, (int, float))
            or not math.isfinite(fitted)):
        return {**base, "reason": "model_fit_timestamp_unavailable"}
    if fitted > issue and not allow_reconstruction:
        return {**base, "reason": "model_not_fitted_at_issue"}
    if metadata.get("trainingCutoffEpoch", issue+1) > midnight_epoch(issue):
        return {**base, "reason": "training_cutoff_after_issue_day_midnight"}
    if metadata.get("latestTrainingOutcomeCompleteEpoch", issue+1) > issue:
        return {**base, "reason": "training_outcomes_after_issue"}
    query = sensor.issue_feature_frame(prepared, [issue])
    matrix = query.to_numpy(dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    try:
        result = np.asarray(artifact["estimator"].predict(matrix), dtype=float)
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        return {**base, "reason": "fresh_numeric_estimator_unavailable", "errorType": type(error).__name__}
    if result.shape != (1, 4) or not np.isfinite(result).all():
        return {**base, "reason": "fresh_numeric_estimator_output_invalid"}
    raw = {name: float(result[0, index]) for index, name in enumerate(OUTPUT_COLUMNS)}
    return {**base, **raw, "available": True, "rawOutputUgM3": dict(raw), "reason": None,
            "rangeOrdered": raw["trailMinimumPoint"] <= raw["trailMaximumPoint"],
            "queryFeaturesHash": _matrix_hash(query.index, query.columns, matrix),
            "training": dict(metadata), "modelIdentity": metadata.get("modelIdentity"),
            "outputsPublishedWithoutArithmetic": True}
