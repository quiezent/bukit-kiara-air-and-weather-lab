"""Leakage check: rebuild features for sampled issues from sources truncated at the issue.

If any feature used a record that only became visible after its issue clock,
the value computed from the truncated sources would differ from the dataset.
The truncation uses the harness's own visibility clocks (sources.py), so this
checks the feature code, not the clocks. ``harness.audit`` re-derives values
and clocks from raw SQL independently.

    python -B -m harness.checks [n_samples]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, dataset, features, sources


def _truncated(issue):
    table = sources.sensor_table()
    weather = sources.forecast_runs("weather_forecast_runs")
    cams = sources.forecast_runs("air_quality_forecast_runs")
    pm, wind = sources.regional_pm_runs(), sources.regional_wind_runs()
    subang, neighbors = sources.subang_table(), sources.neighbor_table()
    return {
        "sensor_table": lambda: table.loc[table.visible_epoch <= issue].reset_index(drop=True),
        "forecast_runs": lambda name: tuple(r for r in (weather if name == "weather_forecast_runs" else cams) if r[1] <= issue),
        "regional_pm_runs": lambda: tuple(r for r in pm if r[1] <= issue),
        "regional_wind_runs": lambda: tuple(r for r in wind if r[1] <= issue),
        "subang_table": lambda: subang.loc[subang.visible_epoch <= issue].reset_index(drop=True),
        "neighbor_table": lambda: neighbors.loc[neighbors.reconstructed_visible <= issue].assign(
            receipt_epoch=lambda f: f.receipt_epoch.where(f.receipt_epoch <= issue, np.inf)).reset_index(drop=True),
    }


def leak_check(n=40, seed=7) -> pd.DataFrame:
    ds = dataset.load()
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(ds.index.to_numpy(np.int64), size=n, replace=False))
    feature_cols = [c for c in ds.columns if c.startswith(tuple(features.GROUPS.values()))]
    originals = {name: getattr(sources, name) for name in _truncated(0)}
    rows = []
    try:
        for issue in sample:
            for name, function in _truncated(int(issue)).items():
                setattr(sources, name, function)
            rebuilt = features.build([int(issue)], rows=sources.harness_rows(sources.sensor_table()),
                                     progress=lambda _: None)
            for name, function in originals.items():
                setattr(sources, name, function)
            a = ds.loc[issue, feature_cols].astype(float).to_numpy()
            b = rebuilt.loc[issue].reindex(feature_cols).astype(float).to_numpy()
            differ = ~((a == b) | (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True))
            rows.append({"issue_myt": str(config.myt(issue))[:16], "differing_columns": int(differ.sum()),
                         "examples": [c for c, d in zip(feature_cols, differ) if d][:5]})
    finally:
        for name, function in originals.items():
            setattr(sources, name, function)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = leak_check(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    print(result.to_string())
    print("issues with any differing feature:", int((result.differing_columns > 0).sum()), "of", len(result))
