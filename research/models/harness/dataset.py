"""Build/load the 5-minute issue-grid dataset (features + production targets)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, features, prod, sources, targets


def issue_grid(start=None, end=None) -> np.ndarray:
    start = config.GRID_START_EPOCH if start is None else int(start)
    end = config.SNAPSHOT_TAKEN_EPOCH if end is None else int(end)
    first = -(-start // config.GRID_SECONDS) * config.GRID_SECONDS
    return np.arange(first, end + 1, config.GRID_SECONDS, dtype=np.int64)


def build_frame(issues, *, progress=print) -> pd.DataFrame:
    """Features and targets for arbitrary issue clocks (used for grid and issued clocks)."""
    issues = np.asarray(issues, dtype=np.int64)
    frame = features.build(issues, progress=progress)
    progress("targets")
    reference = frame["s_freshPm25"].where(frame["m_fresh_available"]).to_numpy(float)
    target = targets.build(issues, reference, sources.plain_rows())
    frame = frame.join(target)
    local = pd.to_datetime(issues, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur")
    frame.insert(0, "issue_myt", local.strftime("%Y-%m-%d %H:%M:%S"))
    frame.insert(1, "date_myt", local.strftime("%Y-%m-%d"))
    frame.insert(2, "canonical", [targets.canonical_label(int(i)) for i in issues])
    frame.insert(3, "eval", issues >= config.EVAL_START_EPOCH)
    return frame


def build(*, progress=print) -> pd.DataFrame:
    started = time.time()
    config.OUT.mkdir(parents=True, exist_ok=True)
    manifest = prod.ensure_mirror()
    issues = issue_grid()
    frame = build_frame(issues, progress=lambda s: progress(f"[{time.time()-started:6.1f}s] {s}"))
    frame.to_parquet(config.DATASET_PATH)
    meta = {"builtAtEpoch": int(time.time()), "rows": len(frame), "columns": len(frame.columns),
            "gridStartMYT": frame.issue_myt.iloc[0], "gridEndMYT": frame.issue_myt.iloc[-1],
            "evalStartMYT": str(config.myt(config.EVAL_START_EPOCH)),
            "snapshotDb": str(config.SNAPSHOT_DB), "neighborDb": str(config.NEIGHBOR_DB),
            "legacySensorVisibilityLagSeconds": config.LEGACY_SENSOR_VISIBILITY_LAG_SECONDS,
            "legacySourceVisibilityLagSeconds": config.LEGACY_SOURCE_VISIBILITY_LAG_SECONDS,
            "neighborReconstructedLagSeconds": config.NEIGHBOR_RECONSTRUCTED_LAG_SECONDS,
            "productionMirrorFiles": manifest["fileCount"],
            "groups": {name: [c for c in frame.columns if c.startswith(prefix)]
                       for name, prefix in features.GROUPS.items()},
            "targets": [c for c in frame.columns if c.startswith("y_")],
            "buildSeconds": round(time.time() - started, 1)}
    config.DATASET_META_PATH.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    progress(f"saved {config.DATASET_PATH} {frame.shape} in {meta['buildSeconds']} s")
    return frame


def load(columns=None) -> pd.DataFrame:
    frame = pd.read_parquet(config.DATASET_PATH, columns=columns)
    frame.index.name = "issue_epoch"
    return frame


def meta() -> dict:
    return json.loads(config.DATASET_META_PATH.read_text(encoding="utf-8"))


def feature_columns(frame, groups=("sensor", "time")) -> list[str]:
    prefixes = tuple(features.GROUPS[g] for g in groups)
    return [c for c in frame.columns if c.startswith(prefixes) and not c.startswith("m_")]


if __name__ == "__main__":
    build()
