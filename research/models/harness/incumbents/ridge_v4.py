"""Walk-forward replay of fresh_numeric_model_v4 (Ridge: +90 point, ride mean/min/max).

Production recipe (fresh_numeric_model_v4.training_data/fit), reproduced from
the receipt cache: nightly cutoff = issue-day MYT midnight; origins every 300 s
in (cutoff-28 d, cutoff-330 min); fresh reference available; all four targets
finite and complete <= cutoff; >= 240 origins; estimator = production
``_make_estimator()`` (median imputer -> target-transformed StandardScaler+Ridge
alpha 100 on targets minus freshPm25). Queries use the harness-visible features.
"""
from __future__ import annotations

from contextlib import closing
import sqlite3

import numpy as np
import pandas as pd

from .. import config, prod
from . import common

NAME = "ridge_v4"
OUTPUTS = {"arrivalPoint": "pred_arrival", "trailMeanPoint": "pred_ride_mean",
           "trailMinimumPoint": "pred_ride_min", "trailMaximumPoint": "pred_ride_max"}


def training_set(cache: pd.DataFrame, cutoff: int):
    numeric = prod.module("fresh_numeric_model_v4")
    columns = list(prod.module("fresh_sensor_features").FEATURE_COLUMNS)
    issues = cache.index.to_numpy(np.int64)
    lower = cutoff - numeric.TRAINING_LOOKBACK_DAYS * 86400
    window = (issues > lower) & (issues < cutoff - numeric.EMBARGO_SECONDS)
    part = cache.loc[window]
    matrix = part[columns].to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    responses = part[["y_arrival", "y_ride_mean", "y_ride_min", "y_ride_max"]].to_numpy(float)
    valid = (part["_available"].to_numpy(bool) & part["y_ride_joint_valid"].to_numpy(bool)
             & np.isfinite(responses).all(axis=1)
             & (part["y_ride_complete"].to_numpy(np.int64) <= cutoff) & np.isfinite(matrix).any(axis=1))
    return part.index[valid], matrix[valid], responses[valid]


def fit(cache, cutoff):
    numeric = prod.module("fresh_numeric_model_v4")
    index, matrix, responses = training_set(cache, cutoff)
    if len(index) < numeric.MINIMUM_TRAINING_ORIGINS:
        return None, {"cutoff": cutoff, "origins": len(index), "reason": "insufficient_completed_training_origins"}
    estimator = numeric._make_estimator()
    estimator.fit(matrix, responses)
    return estimator, {"cutoff": cutoff, "origins": len(index),
                       "trainingDataHash": numeric._matrix_hash(index, prod.module("fresh_sensor_features").FEATURE_COLUMNS,
                                                                 matrix, responses)}


def predict(estimator, frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["s_" + c for c in prod.module("fresh_sensor_features").FEATURE_COLUMNS]
    out = pd.DataFrame(index=frame.index, columns=list(OUTPUTS.values()), dtype=float)
    usable = frame["m_fresh_available"].to_numpy(bool) & np.isfinite(frame["s_freshPm25"].to_numpy(float))
    if estimator is None or not usable.any():
        return out
    matrix = frame.loc[usable, columns].to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    values = estimator.predict(matrix)
    out.loc[usable, :] = values
    return out


def replay(ds: pd.DataFrame, days=None, cache=None, log=print) -> pd.DataFrame:
    cache = common.receipt_cache() if cache is None else cache
    days = common.eval_days(ds) if days is None else days
    parts, fits = [], []
    for cutoff in days:
        estimator, info = fit(cache, cutoff)
        fits.append(info)
        day = ds.loc[(ds.index >= cutoff) & (ds.index < cutoff + 86400)]
        parts.append(predict(estimator, day))
        log(f"{NAME} {config.myt(cutoff):%Y-%m-%d} origins={info['origins']}")
    out = pd.concat(parts).add_prefix(f"{NAME}__")
    out.attrs["fits"] = fits
    return out


# ----------------------------------------------------------------------------- verification
def verify(cutoff: int, cache=None) -> dict:
    """Compare the cache recipe with production fit() on the snapshot and with the live artifact."""
    numeric = prod.module("fresh_numeric_model_v4")
    sensor = prod.module("fresh_sensor_features")
    cache = common.receipt_cache() if cache is None else cache
    with closing(sqlite3.connect(f"file:{config.SNAPSHOT_DB.as_posix()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        rows = numeric._training_rows(conn, cutoff - numeric.TRAINING_LOOKBACK_DAYS * 86400 - 3 * 3600, cutoff)
    artifact = numeric.fit(rows, cutoff)
    ours, info = fit(cache, cutoff)
    result = {"cutoff": str(config.myt(cutoff)),
              "productionTrainingHash": artifact["metadata"]["trainingDataHash"],
              "cacheTrainingHash": info.get("trainingDataHash"),
              "productionOrigins": artifact["metadata"]["trainingOriginCount"], "cacheOrigins": info["origins"]}
    probe = cache.loc[(cache.index >= cutoff) & (cache.index < cutoff + 86400) & cache["_available"]]
    matrix = probe[list(sensor.FEATURE_COLUMNS)].to_numpy(float)
    matrix[~np.isfinite(matrix)] = np.nan
    a = artifact["estimator"].predict(matrix)
    b = ours.predict(matrix)
    result["maxAbsPredictionDiffVsProductionFit"] = float(np.max(np.abs(a - b)))
    live = numeric.load_artifact(config.SERVER / "private" / "fresh-numeric-model-v4.pkl")
    if live is not None and live["metadata"]["trainingCutoffEpoch"] == cutoff:
        c = live["estimator"].predict(matrix)
        result["liveArtifactTrainingHash"] = live["metadata"]["trainingDataHash"]
        result["maxAbsPredictionDiffVsLiveArtifact"] = float(np.max(np.abs(c - b)))
    return result
