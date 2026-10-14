"""Chronological walk-forward evaluation shared by every candidate.

For each evaluation day D (MYT), the candidate is refit on rows whose issue is
in (D - lookback, D - embargo) AND whose target label completed strictly
before D (midnight). It then predicts every issue of day D. Only feature
columns may be used for predictions; ``y_*`` columns of the test day are
removed before the candidate sees them.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, features

COMPLETE_COLUMNS = {
    "arrival": "y_arrival_complete", "class5": "y_arrival_complete", "delta": "y_arrival_complete",
    "fall20": "y_arrival_complete", "rise20": "y_arrival_complete",
    "fall40": "y_arrival_complete", "rise40": "y_arrival_complete",
    "ride_mean": "y_ride_complete", "ride_min": "y_ride_complete", "ride_max": "y_ride_complete",
    "ride_le70": "y_ride_complete", "first20": "y_first20_complete",
    "morning": "y_morning_complete", "afternoon": "y_afternoon_complete",
}
DEFAULT_EMBARGO_MINUTES = 330          # fresh_numeric_model_v4.EMBARGO_SECONDS


def feature_columns(ds: pd.DataFrame, groups=("sensor", "time")) -> list[str]:
    prefixes = tuple(features.GROUPS[g] for g in groups)
    return [c for c in ds.columns if c.startswith(prefixes) and not c.startswith("y_")]


def eval_days(ds: pd.DataFrame, start=None, end=None) -> list[int]:
    start = config.EVAL_START_EPOCH if start is None else start
    index = ds.index.to_numpy(np.int64)
    index = index[(index >= start) & ((index < end) if end is not None else True)]
    return sorted({config.midnight(i) for i in index})


def walk_forward(ds: pd.DataFrame, fit_predict, *, targets=("arrival",), lookback_days=28,
                 embargo_minutes=DEFAULT_EMBARGO_MINUTES, days=None, test_mask=None, log=print) -> pd.DataFrame:
    """Run ``fit_predict(train, test, cutoff) -> DataFrame`` for each day.

    ``train`` includes features and targets (label-complete rows only); ``test``
    includes only non-target columns. Returned frames must be indexed by the
    test issues; they are concatenated in time order.
    """
    complete_cols = sorted({COMPLETE_COLUMNS[t] for t in targets})
    target_cols = [c for c in ds.columns if c.startswith("y_")]
    index = ds.index.to_numpy(np.int64)
    days = eval_days(ds) if days is None else days
    outputs = []
    for cutoff in days:
        started = time.time()
        complete = np.ones(len(ds), bool)
        for column in complete_cols:
            complete &= ds[column].to_numpy(np.int64) < cutoff
        train_mask = ((index > cutoff - lookback_days * 86400) & (index < cutoff - embargo_minutes * 60) & complete)
        test_rows = (index >= cutoff) & (index < cutoff + 86400)
        if test_mask is not None:
            test_rows &= np.asarray(test_mask.reindex(ds.index).fillna(False), bool)
        if not test_rows.any():
            continue
        train = ds.loc[train_mask]
        test = ds.loc[test_rows].drop(columns=target_cols)
        prediction = fit_predict(train, test, cutoff)
        if prediction is None:
            continue
        prediction = prediction.reindex(test.index)
        outputs.append(prediction)
        log(f"{config.myt(cutoff):%Y-%m-%d} train={int(train_mask.sum())} test={int(test_rows.sum())} "
            f"{time.time() - started:.1f}s")
    return pd.concat(outputs) if outputs else pd.DataFrame()
