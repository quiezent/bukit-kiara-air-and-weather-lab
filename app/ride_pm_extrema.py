"""Fixed learned lowest/highest bucket-median PM2.5 during the exact ride.

The two output columns share each tree and its leaf membership. Every observed
training row has minimum <= maximum, so joint leaf averages preserve ordering
without sorting, widening or moving either output around a separate mean.
No database, provider, publication policy or model selection lives here.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

from forecast_clock import ForecastClock, target_series, window_weights


MODEL_VERSION = "local_extra_trees_joint_extrema90_210_v1"
MIN_TRAINING_ORIGINS = 48
MODEL_PARAMETERS = {
    "n_estimators": 64,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "max_features": 1.0,
    "random_state": 20261008,
    "n_jobs": 1,
}
TARGET_DEFINITION = (
    "Lowest and highest complete 15-minute PM2.5 bucket medians among buckets "
    "with positive overlap in the exact issued +90..+210-minute ride window. "
    "Partial edge buckets summarize their full 15-minute observation interval. "
    "These are forecasts of bucket-median extrema, not continuous or "
    "instantaneous minimum and maximum concentrations."
)


def window_extrema_targets(pm_series, clock):
    """Exact contributing-bin extrema; any missing bin invalidates both heads."""
    # Reuse the same index, complete-bin and decision-clock validation as mean.
    target_series(pm_series, clock)
    values = pd.to_numeric(pm_series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    weights = window_weights(
        clock.arrival_lead_seconds, clock.end_lead_seconds, clock.bucket_seconds
    )
    parts = pd.concat(
        {offset: values.shift(-offset) for offset in weights}, axis=1
    )
    return pd.DataFrame({
        "low": parts.min(axis=1, skipna=False),
        "high": parts.max(axis=1, skipna=False),
    })


def _array_hash(index, columns, matrix, responses=None):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(columns), ensure_ascii=True).encode("utf-8"))
    digest.update(np.asarray(index.asi8, dtype="<i8").tobytes())
    blocks = (matrix,) if responses is None else (matrix, responses)
    for block in blocks:
        canonical = np.asarray(block, dtype="<f8").copy()
        canonical[np.isnan(canonical)] = np.nan
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _make_estimator():
    # Lazy imports keep a missing optional runtime from breaking other cards.
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesRegressor(**MODEL_PARAMETERS),
    )


def predict_ride_extrema(features, pm_series, query_time, train_end, clock):
    """Fit one declared joint estimator to completed targets and return it raw.

    ``features`` are the existing origin-known closed-bin particle/met features.
    The caller supplies the same query time and 330-minute training embargo as
    the mean estimator. A second maturity mask prevents accidental future-target
    fitting even if a caller supplies a later train_end.
    """
    base = {
        "available": False, "low": None, "high": None,
        "rawLow": None, "rawHigh": None, "crossing": None,
        "modelVersion": MODEL_VERSION, "modelParameters": dict(MODEL_PARAMETERS),
        "targetDefinition": TARGET_DEFINITION, "targetResolutionMinutes": 15,
        "targetName": "extrema90_210", "outputColumns": ["low", "high"],
        "forecastClock": clock.metadata() if isinstance(clock, ForecastClock) else None,
        "numericalSelection": False, "persistenceFallback": False,
        "outputSorting": False, "outputWidening": False, "meanRecentering": False,
        "calibrated": False, "prospectivelyValidated": False,
        "interpretation": "predicted_lowest_highest_bucket_medians_not_uncertainty_interval",
        "minimumTrainingOrigins": MIN_TRAINING_ORIGINS,
    }
    if not isinstance(clock, ForecastClock):
        return {**base, "reason": "forecast_clock_unavailable"}
    if clock.arrival_minutes != 90 or clock.window_minutes != 120 or clock.bucket_seconds != 900:
        return {**base, "reason": "ride_target_definition_mismatch"}
    if not isinstance(features, pd.DataFrame) or not isinstance(features.index, pd.DatetimeIndex):
        return {**base, "reason": "origin_feature_frame_unavailable"}
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        return {**base, "reason": "origin_feature_index_invalid"}
    query_time, train_end = pd.Timestamp(query_time), pd.Timestamp(train_end)
    if query_time not in features.index or int(query_time.timestamp()) != clock.anchor_epoch:
        return {**base, "reason": "query_feature_anchor_mismatch"}
    if not len(features.columns):
        return {**base, "reason": "origin_features_unavailable"}
    targets = window_extrema_targets(pm_series, clock).reindex(features.index)
    matrix = features.to_numpy(dtype=float, na_value=np.nan)
    matrix[~np.isfinite(matrix)] = np.nan
    response = targets.to_numpy(dtype=float, na_value=np.nan)
    origin_epochs = features.index.asi8 // 10**9
    complete_epochs = origin_epochs + clock.outcome_complete_lead_seconds
    valid = np.isfinite(response).all(axis=1)
    valid &= features.index < train_end
    valid &= complete_epochs <= clock.issued_epoch
    training_index = features.index[valid]
    training_matrix, training_response = matrix[valid], response[valid]
    query_matrix = matrix[[features.index.get_loc(query_time)]]
    base.update({
        "trainingOriginCount": int(valid.sum()),
        "featureCount": len(features.columns), "featureColumns": list(features.columns),
        "trainingCutoffEpoch": int(train_end.timestamp()),
        "observedOutcomeCompletionCutoffEpoch": clock.issued_epoch,
        "trainingDataHash": _array_hash(
            training_index, features.columns, training_matrix, training_response
        ),
        "queryFeaturesHash": _array_hash(
            features.loc[[query_time]].index, features.columns, query_matrix
        ),
        "featureAnchorEpoch": clock.anchor_epoch,
        "latestTrainingOriginEpoch": int(training_index[-1].timestamp()) if len(training_index) else None,
        "latestTrainingOutcomeCompleteEpoch": int(complete_epochs[valid][-1]) if valid.any() else None,
        "inputImputation": "feature medians fitted only on the declared training rows",
    })
    if valid.sum() < MIN_TRAINING_ORIGINS:
        return {**base, "reason": "insufficient_completed_training_origins"}
    if not np.isfinite(training_matrix).any():
        return {**base, "reason": "training_features_unavailable"}
    if np.any(training_response[:, 0] > training_response[:, 1]):
        return {**base, "reason": "observed_extrema_order_invalid"}
    try:
        estimator = _make_estimator()
        estimator.fit(training_matrix, training_response)
        prediction = np.asarray(estimator.predict(query_matrix), dtype=float)
    except (ImportError, ValueError, RuntimeError) as error:
        return {**base, "reason": "range_estimator_unavailable", "errorType": type(error).__name__}
    if prediction.shape != (1, 2) or not np.isfinite(prediction).all():
        return {**base, "reason": "range_estimator_output_invalid"}
    low, high = float(prediction[0, 0]), float(prediction[0, 1])
    return {
        **base, "available": True, "low": low, "high": high,
        "rawLow": low, "rawHigh": high, "crossing": low > high,
        "rangeOrdered": low <= high, "negativeOutput": low < 0 or high < 0,
        "reason": None, "outputsPublishedWithoutArithmetic": True,
    }
