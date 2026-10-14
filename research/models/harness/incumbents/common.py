"""Training caches that reproduce production training inputs exactly.

Production fits rebuild sensor features for every training origin each night.
Those features depend only on rows observed at or before the origin (and, for
receipt rows, available by then), never on the fit cutoff, because every
training origin precedes the cutoff by >= 120 min and receipts lag <= 459 s.
The labels use bucket medians that close >= 120 min before the cutoff. So one
cache per production input semantics reproduces every nightly training set;
``verify_*`` functions in each replay module check this against the
production training routine for chosen cutoffs.

Two production input semantics exist:
* ``receipt``: Ridge v4 and the arrival logistic (receipt rows known with
  max(available, confirmed); legacy rows unknown = visible at observation).
* ``plain``: first-crossing HGB v2 (readings table only, visible at observation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, prod, sources, targets

CACHE_DIR = config.OUT / "cache"


def aligned_origins(start, end):
    return np.arange(-(-int(start) // 300) * 300, int(end), 300, dtype=np.int64)


def variable_origins(start, end):
    """fresh_event_model.dense_origins: bases + 30 s + (base//300 % 5) min."""
    bases = np.arange(int(start) // 300 * 300, int(end), 300, dtype=np.int64)
    epochs = bases + 30 + ((bases // 300) % 5) * 60
    return epochs[(epochs >= start) & (epochs < end)]


def _feature_cache(name, issues, rows):
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        if len(frame) == len(issues) and np.array_equal(frame.index.to_numpy(np.int64), issues):
            return frame
    sensor = prod.module("fresh_sensor_features")
    prepared = sensor.prepare_rows(rows)
    frame = sensor.issue_feature_frame(prepared, list(issues.astype(float)))
    provenance = frame.attrs["featureProvenance"]
    frame.index = pd.Index(issues, name="issue_epoch")
    frame["_available"] = [bool(p["available"]) for p in provenance]
    frame["_featureSourceMaxEpoch"] = [p.get("featureSourceMaxEpoch") for p in provenance]
    frame["_featureSourceMaxEpoch"] = pd.to_numeric(frame["_featureSourceMaxEpoch"], errors="coerce")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def _span():
    table = sources.sensor_table()
    start = int(table.epoch.min())
    end = config.SNAPSHOT_TAKEN_EPOCH
    return start, end


def receipt_cache() -> pd.DataFrame:
    """Aligned 300-s origins, production receipt semantics, plus Ridge/arrival labels."""
    start, end = _span()
    issues = aligned_origins(start, end)
    frame = _feature_cache("prod_receipt_aligned", issues, sources.production_receipt_rows())
    series = targets.bucket_series(sources.plain_rows())
    labels = targets.ride_targets(series, issues)
    labels.index = frame.index
    return frame.join(labels)


def plain_cache() -> pd.DataFrame:
    """Variable-offset origins, plain readings semantics, plus first-crossing labels."""
    start, end = _span()
    issues = variable_origins(start, end)
    frame = _feature_cache("prod_plain_variable", issues, sources.plain_rows())
    buckets = targets.bucket_map(targets.bucket_series(sources.plain_rows()))
    reference = frame["freshPm25"].to_numpy(float)
    labels = targets.first_crossing(issues, reference, buckets)
    return frame.join(labels)


def eval_days(frame_or_issues, start=None, end=None):
    """MYT midnights of days with issues in [start, end)."""
    issues = (frame_or_issues.index.to_numpy(np.int64) if hasattr(frame_or_issues, "index")
              else np.asarray(frame_or_issues, dtype=np.int64))
    start = config.EVAL_START_EPOCH if start is None else start
    days = sorted({config.midnight(i) for i in issues if i >= start and (end is None or i < end)})
    return days
