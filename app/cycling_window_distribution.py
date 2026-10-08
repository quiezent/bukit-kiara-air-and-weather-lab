"""Experimental PM2.5 ride-window mean distribution; off-line helpers only.

The target is the mean over the exact named ride window, not an instantaneous
minimum, first crossing, or the +90-minute endpoint. Leaf weights transfer
historical ride-mean deltas to the query reference with a zero concentration
floor. Probabilities and empirical ranges are uncalibrated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

MODEL_VERSION = "ttdi_cycling_window_delta_forest_70_v1"
SENSOR = ("fresh", "freshOffset", "delta15", "delta30", "delta60", "delta120", "sd60", "mean3Offset",
          "temp", "rh", "tempDelta30", "rhDelta30", "sinTarget", "cosTarget", "targetLeadHours", "targetDurationHours")
WEATHER = ("weatherAgeHours", "priorRainMm", "duringRainMm", "priorRainProbMean", "duringRainProbMean",
           "firstRainLeadHours", "lastRainEndLeadHours", "wind10UMeanKmh", "wind10VMeanKmh",
           "wind10UDeltaKmh", "wind10VDeltaKmh", "gustHourlyEnvelopeMaxKmh", "gustHourlyEnvelopeDeltaKmh",
           "ventilationMeanM2S", "ventilationDeltaM2S", "boundaryLayerMeanM", "targetTempDeltaC", "targetRhDeltaPct")
FEATURES = (*SENSOR, *WEATHER)
PARAMETERS = dict(n_estimators=128, max_depth=4, min_samples_leaf=6,
                  max_features=1.0, random_state=1749, n_jobs=1)


def feature_names(): return list(FEATURES)


def _x(features):
    x = features[list(FEATURES)].to_numpy(float) if isinstance(features, pd.DataFrame) else np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError("Expected34 frozen window features per issue")
    if not np.isfinite(x[:, 0]).all() or (x[:, 0] < 0).any():
        raise ValueError("Exact query/training references must be finite and nonnegative")
    return np.where(np.isfinite(x), x, np.nan)


def fit_window_distribution(features, ride_mean_deltas):
    """Fit scalar delta forest and keep empirical training targets for weights."""
    x = _x(features)
    target = np.asarray(ride_mean_deltas, float)
    if target.shape != (len(x),) or len(x) < 50 or not np.isfinite(target).all():
        raise ValueError("Need at least50 complete training ride-mean deltas")
    if (target + x[:, 0] < -1e-8).any():
        raise ValueError("Training ride means cannot be negative")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    tx = imputer.fit_transform(x)
    forest = RandomForestRegressor(**PARAMETERS)
    with threadpool_limits(limits=1):
        forest.fit(tx, target)
        leaves = forest.apply(tx)
    return dict(modelVersion=MODEL_VERSION, featureNames=list(FEATURES), imputer=imputer,
                forest=forest, trainingDeltas=target.copy(), trainingLeaves=leaves,
                trainingCases=len(x))


def window_weights(model, features):
    """Average uniform original-case memberships of each reached tree leaf."""
    if model["modelVersion"] != MODEL_VERSION or model["featureNames"] != list(FEATURES):
        raise ValueError("Incompatible cycling window model")
    tx = model["imputer"].transform(_x(features))
    with threadpool_limits(limits=1): query_leaves = model["forest"].apply(tx)
    train_leaves = model["trainingLeaves"]
    weights = np.empty((len(tx), len(train_leaves)))
    for i, leaf in enumerate(query_leaves):
        membership = train_leaves == leaf[None, :]
        counts = membership.sum(axis=0)
        if (counts == 0).any(): raise ValueError("Empty empirical tree leaf")
        weights[i] = (membership / counts[None, :]).mean(axis=1)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1):
        raise ValueError("Invalid empirical probability mass")
    return weights


def weighted_quantiles(values, weights, quantiles=(.1, .5, .9)):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any(): return [None for _ in quantiles]
    order = np.argsort(values[valid], kind="stable")
    values, weights = values[valid][order], weights[valid][order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return [float(values[min(int(np.searchsorted(cumulative, q, side="left")), len(values) - 1)]) for q in quantiles]


def summarize_window_distribution(training_deltas, reference, weights=None, limit=70.):
    """Transfer ride-mean deltas, preserving physical support before decisions."""
    deltas = np.asarray(training_deltas, float)
    if deltas.ndim != 1 or not len(deltas) or not np.isfinite(deltas).all():
        raise ValueError("Expected finite historical ride-mean deltas")
    if not np.isfinite(reference) or reference < 0 or not np.isfinite(limit) or limit < 0:
        raise ValueError("Reference and planning cutoff must be finite and nonnegative")
    w = np.full(len(deltas), 1 / len(deltas)) if weights is None else np.asarray(weights, float)
    if w.shape != deltas.shape or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("Expected normalized training-case probability weights")
    means = np.maximum(0., reference + deltas)
    eligible = means <= limit
    ew = w * eligible
    probability = float(np.clip(ew.sum(), 0., 1.))
    q10, median, q90 = weighted_quantiles(means, w)
    conditional_median = weighted_quantiles(means, ew, (.5,))[0]
    return dict(referencePm=float(reference), planningThresholdUgM3=float(limit),
                probabilityRideMeanAtOrBelowThreshold=probability,
                predictedRideMean=float(means @ w), rideMeanMedian=median,
                rideMeanQ10=q10, rideMeanQ90=q90,
                conditionalRideMean=float(means @ ew / probability) if probability > 0 else None,
                conditionalRideMeanMedian=conditional_median,
                conditionalPositiveWeightCases=int(((w > 0) & eligible).sum()),
                conditionalEffectiveCases=float(probability ** 2 / np.square(ew).sum()) if probability > 0 else None,
                effectiveTrainingCases=float(1 / np.square(w).sum()), trainingCases=len(deltas),
                experimental=True, calibrated=False,
                targetMeaning="Exact ride-window average PM2.5; not every moment below the cutoff",
                conditionalMeaning="Ride-window average conditional on that average meeting the planning cutoff")


def predict_window_distribution(model, features, limit=70.):
    weights = window_weights(model, features)
    references = _x(features)[:, 0]
    results = []
    for reference, w in zip(references, weights):
        result = summarize_window_distribution(model["trainingDeltas"], reference, w, limit)
        result["modelVersion"] = MODEL_VERSION
        results.append(result)
    return results
