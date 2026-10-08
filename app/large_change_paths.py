"""Experimental distribution of six future PM2.5 changes, trained off-line.

The weighted empirical paths preserve reversals and the competing first cause.
Conditional magnitudes describe the maximum sampled excursion within 90 minutes
given that a first >=20 crossing occurs in that direction, not the endpoint.
No database access, fitting schedule, publication, or live activation is here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

MODEL_VERSION = "ttdi_joint_path_random_forest_20_v1"
LEADS = (15, 30, 45, 60, 75, 90)
LOCAL = ("fresh", "freshOffset", "delta15", "delta30", "delta60", "delta120",
         "sd60", "mean3Offset", "temp", "rh", "tempDelta30", "rhDelta30")
WEATHER = ("stepRainMm", "stepRainProbabilityPct", "stepWindUKmh", "stepWindVKmh",
           "stepWindUDeltaKmh", "stepWindVDeltaKmh", "stepGustEnvelopeKmh",
           "stepTempDeltaC", "stepRhDeltaPct", "stepVentilationM2S", "weatherAgeHours")
FEATURES = (*LOCAL, *("mean_" + n for n in WEATHER), *("last_" + n for n in WEATHER))
FOREST_PARAMETERS = dict(n_estimators=128, max_depth=4, min_samples_leaf=6,
                         max_features=1.0, random_state=1749, n_jobs=1)


def feature_names():
    return list(FEATURES)


def aggregate_issue_features(step_frame):
    """Aggregate six issue-vintage rows; never use later observed weather/PM."""
    rows = []
    for issue, frame in step_frame.groupby("issueEpoch", sort=True):
        frame = frame.sort_values("step")
        if frame.step.tolist() != list(range(6)):
            raise ValueError("Every issue requires six unique ordered forecast steps")
        local = frame[list(LOCAL)].to_numpy(float)
        if not np.all(np.isclose(local, local[0], equal_nan=True)):
            raise ValueError("Local source features differ across one issue")
        weather = frame[list(WEATHER)].replace([np.inf, -np.inf], np.nan)
        row = dict(issueEpoch=int(issue))
        row.update(zip(LOCAL, local[0]))
        row.update(zip(("mean_" + n for n in WEATHER), weather.mean(axis=0).to_numpy()))
        row.update(zip(("last_" + n for n in WEATHER), weather.iloc[-1].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows, columns=["issueEpoch", *FEATURES])


def _x(features):
    a = features[list(FEATURES)].to_numpy(float) if isinstance(features, pd.DataFrame) else np.asarray(features, float)
    if a.ndim != 2 or a.shape[1] != len(FEATURES):
        raise ValueError("Expected one row of 34 fixed features per issue")
    return np.where(np.isfinite(a), a, np.nan)


def first_causes(delta_paths, threshold=20.0):
    paths = np.asarray(delta_paths, float)
    if paths.ndim != 2 or paths.shape[1] != len(LEADS) or not np.isfinite(paths).all():
        raise ValueError("Expected complete finite six-step change paths")
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("Threshold must be positive and finite")
    crossings = np.abs(paths) >= threshold
    first = crossings.argmax(axis=1)
    return np.where(crossings.any(axis=1), np.sign(paths[np.arange(len(paths)), first]), 0).astype(int)


def fit_path_model(features, delta_paths):
    """Fit training-only imputation and fixed multi-output forest."""
    paths = np.asarray(delta_paths, float)
    first_causes(paths)
    x = _x(features)
    if len(x) != len(paths) or len(x) < 12:
        raise ValueError("Need matching feature/path rows and at least 12 cases")
    if not np.isfinite(x[:, 0]).all() or (x[:, 0] < 0).any() or (paths + x[:, [0]] < -1e-8).any():
        raise ValueError("Training references and implied concentrations must be nonnegative")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    tx = imputer.fit_transform(x)
    forest = RandomForestRegressor(**FOREST_PARAMETERS)
    with threadpool_limits(limits=1):
        forest.fit(tx, paths)
        leaves = forest.apply(tx)
    return dict(modelVersion=MODEL_VERSION, featureNames=list(FEATURES),
                imputer=imputer, forest=forest, trainingPaths=paths.copy(),
                trainingLeaves=leaves, trainingCases=len(paths))


def path_weights(model, features):
    """Average uniform membership weights of the reached leaf in every tree.

    All original training paths in a leaf receive equal weight for that tree;
    bootstrap duplicates are not duplicated as independent path observations.
    """
    if model["modelVersion"] != MODEL_VERSION or model["featureNames"] != list(FEATURES):
        raise ValueError("Incompatible path model")
    tx = model["imputer"].transform(_x(features))
    with threadpool_limits(limits=1):
        query_leaves = model["forest"].apply(tx)
    training_leaves = model["trainingLeaves"]
    out = np.empty((len(tx), len(training_leaves)))
    for i, leaves in enumerate(query_leaves):
        membership = training_leaves == leaves[None, :]
        counts = membership.sum(axis=0)
        if (counts == 0).any():
            raise ValueError("Reached tree leaf contains no training paths")
        out[i] = (membership / counts[None, :]).mean(axis=1)
    if not np.isfinite(out).all() or (out < 0).any() or not np.allclose(out.sum(axis=1), 1):
        raise ValueError("Invalid path probability mass")
    return out


def weighted_quantiles(values, weights, quantiles=(.1, .5, .9)):
    """Inverse empirical CDF; each quantile is an observed training magnitude."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return [None for _ in quantiles]
    order = np.argsort(values[valid], kind="stable")
    ordered = values[valid][order]
    cumulative = np.cumsum(weights[valid][order])
    cumulative /= cumulative[-1]
    return [float(ordered[min(int(np.searchsorted(cumulative, q, side="left")), len(ordered) - 1)]) for q in quantiles]


def direction_call(probabilities, minimum_probability=None):
    """Three-way argmax, with ties unresolved; optional .5 directional gate."""
    p = np.asarray(probabilities, float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("Expected [drop, none, rise] probability mass")
    winners = np.flatnonzero(np.isclose(p, p.max(), atol=1e-12, rtol=0))
    if len(winners) != 1:
        return 0
    winner = int(winners[0])
    if minimum_probability is not None and p[winner] < minimum_probability:
        return 0
    return (-1, 0, 1)[winner]


def predict_path_distribution(model, features, threshold=20.0):
    """Return first-cause probabilities and conditional peak-change magnitudes.

    A drop of size m corresponds to reference-m; a rise corresponds to
    reference+m. Magnitude quantiles condition on FIRST cause at this threshold,
    so first >=20 drop probability is not nested inside first >=10 drop.
    """
    paths = model["trainingPaths"]
    references = _x(features)[:, 0]
    if not np.isfinite(references).all() or (references < 0).any():
        raise ValueError("Query requires an exact finite nonnegative reference")
    weights = path_weights(model, features)
    results = []
    for w, reference in zip(weights, references):
        # Empirical changes transfer between reference levels. Enforce physical
        # support BEFORE first-cause classification, not only on displayed size.
        transferred_paths = np.maximum(paths, -reference)
        causes = first_causes(transferred_paths, threshold)
        magnitudes = {-1: np.maximum(0, -transferred_paths.min(axis=1)),
                      1: np.maximum(0, transferred_paths.max(axis=1))}
        p = [float(w[causes == c].sum()) for c in (-1, 0, 1)]
        row = dict(modelVersion=MODEL_VERSION, experimental=True, calibrated=False,
                   changeThresholdUgM3=float(threshold), horizonMinutes=90,
                   probabilityDrop=p[0], probabilityNoCrossing=p[1], probabilityRise=p[2],
                   referencePm=float(reference), physicalSupport="Transferred paths floored at zero concentration before cause and magnitude calculation",
                   direction=direction_call(p), directionAt0_5=direction_call(p, .5),
                   effectiveTrainingCases=float(1 / np.square(w).sum()),
                   trainingCases=len(paths))
        for name, cause in (("drop", -1), ("rise", 1)):
            mask = causes == cause
            cw = w * mask
            mass = cw.sum()
            quantiles = weighted_quantiles(magnitudes[cause], cw)
            row[name + "ConditionalMagnitude"] = dict(
                q10=quantiles[0], median=quantiles[1], q90=quantiles[2],
                definition="Maximum sampled directional excursion within 90 minutes, conditional on that first crossing cause",
                trainingCauseCases=int(mask.sum()), positiveWeightCauseCases=int(((w > 0) & mask).sum()),
                effectiveCauseCases=float(mass ** 2 / np.square(cw).sum()) if mass else None)
        horizons = []
        for step, lead in enumerate(LEADS):
            prefix = transferred_paths[:, :step + 1]
            crossed = np.abs(prefix) >= threshold
            first = crossed.argmax(axis=1)
            prefix_causes = np.where(crossed.any(axis=1), np.sign(prefix[np.arange(len(prefix)), first]), 0).astype(int)
            horizons.append(dict(leadMinutes=lead, probabilityDrop=float(w[prefix_causes == -1].sum()),
                                 probabilityNoCrossing=float(w[prefix_causes == 0].sum()),
                                 probabilityRise=float(w[prefix_causes == 1].sum())))
        row["horizons"] = horizons
        results.append(row)
    return results


def predict_training_direction_magnitude(delta_paths, reference, threshold=20.0):
    """Uniform fold-training baseline with the identical physical support rule.

    Return a drop/rise ConditionalMagnitude mapping using the same field names
    as the forest prediction. This alternative does not use query weather or
    learned similarity weights and is not a forecast of whether an event occurs.
    """
    if not np.isfinite(reference) or reference < 0:
        raise ValueError("Reference must be finite and nonnegative")
    paths = np.maximum(np.asarray(delta_paths, float), -reference)
    causes = first_causes(paths, threshold)
    result = {}
    for name, code in (("drop", -1), ("rise", 1)):
        values = np.maximum(0, -paths.min(axis=1)) if code == -1 else np.maximum(0, paths.max(axis=1))
        mask = causes == code
        q10, median, q90 = weighted_quantiles(values, mask.astype(float))
        count = int(mask.sum())
        result[name + "ConditionalMagnitude"] = dict(q10=q10, median=median, q90=q90,
            definition="Maximum sampled directional excursion within 90 minutes, conditional on that first crossing cause",
            trainingCauseCases=count, positiveWeightCauseCases=count,
            effectiveCauseCases=float(count) if count else None)
    return result
