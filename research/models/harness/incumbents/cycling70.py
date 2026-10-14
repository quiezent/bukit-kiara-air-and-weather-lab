"""Walk-forward replay of cycling_window_live (chance the +90..+210 ride mean <= 70).

Production recipe (cycling_window_live.refresh_model, head initial90, candidate
'sensor'): cutoff = issue-day MYT midnight; first_crossing_live._source_snapshot
(readings, dashboard publications, weather runs up to cutoff-1, earliest
max(2026-09-07, cutoff-28 d)); research/cycling_window_70/recheck_preparation
.rebuild_training (fixed four-hour plans issued at origin+300 s, publication
watermark guards); _training_support (>=50 cases, >=10 dates, both classes);
cycling_window_probability.fit_window_model (logistic C=0.1 on 16 sensor
features). All production functions are called unchanged on the snapshot.

Queries mirror predict_window: weather = latest run fetched <= issue
(first_crossing_live._latest_weather on the snapshot), inputs from
recheck_preparation.extract_current on rows visible at the issue, the
watermark = newest visible observation, reference = 5-min median of visible rows.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import config, prod, sources
from . import common

NAME = "cycling70"


def fit(cutoff: int):
    live = prod.module("cycling_window_live")
    probability = prod.module("cycling_window_probability")
    src = live.source
    try:
        snapshot = src._source_snapshot(config.SNAPSHOT_DB, cutoff - 1, cutoff)
        frames, preparation = live._preparation().rebuild_training(*snapshot)
        train, support = live._training_support(frames["initial"], cutoff, "initial90")
        if not support["available"]:
            return None, {"cutoff": cutoff, "cases": support.get("trainingCases", 0),
                          "reason": ";".join(support["unavailableReasons"])}
        model = probability.fit_window_model(train, candidate="sensor", cutoff_epoch=cutoff)
    except Exception as error:  # production marks the head unavailable on any failure
        return None, {"cutoff": cutoff, "cases": 0, "reason": f"{type(error).__name__}: {error}"}
    return model, {"cutoff": cutoff, "cases": support["trainingCases"], "targetDates": support["targetDates"],
                   "successes": support["trainingSuccesses"]}


class _VisibleRows:
    def __init__(self):
        table = sources.sensor_table()
        self.epoch = table.epoch.to_numpy(np.int64)
        self.visible = table.visible_epoch.to_numpy(float)
        self.values = table[["pm02", "atmp", "rhum"]].to_numpy(float)

    def at(self, issue):
        lo = np.searchsorted(self.epoch, issue - 4 * 3600, side="left")
        hi = np.searchsorted(self.epoch, issue, side="right")
        keep = self.visible[lo:hi] <= issue
        rows = []
        for e, (p, t, h) in zip(self.epoch[lo:hi][keep], self.values[lo:hi][keep]):
            rows.append({"epoch": int(e), "pm02": None if math.isnan(p) else float(p),
                         "atmp": None if math.isnan(t) else float(t), "rhum": None if math.isnan(h) else float(h)})
        return rows


def predict_issue(model, issue: int, rows) -> tuple[float, str]:
    live = prod.module("cycling_window_live")
    probability = prod.module("cycling_window_probability")
    freshness = prod.module("forecast_freshness")
    if model is None:
        return np.nan, "no_model"
    if not rows:
        return np.nan, "missing_sensor_inputs"
    origin = issue // 900 * 900
    reference = freshness.references(rows, [origin], issue - origin, issue)[0][0]
    if not math.isfinite(float(reference)):
        return np.nan, "fresh_sensor_reference_unavailable"
    watermark = max(row["epoch"] for row in rows)
    run, _ = live.source._latest_weather(config.SNAPSHOT_DB, issue)
    if run is None:
        return np.nan, "issue_weather_unavailable"
    start, end = issue + 5400, issue + 12600
    inputs, reason = live._preparation().extract_current(rows, run, issue, start, end, float(watermark), float(reference))
    if inputs is None:
        return np.nan, reason
    frame, _ = inputs
    chance = float(probability.predict_window_probability(model, frame)[0])
    return chance, "available"


def replay(ds: pd.DataFrame, days=None, issues=None, log=print) -> pd.DataFrame:
    days = common.eval_days(ds) if days is None else days
    visible = _VisibleRows()
    records, fits = [], []
    for cutoff in days:
        model, info = fit(cutoff)
        fits.append(info)
        day = ds.index[(ds.index >= cutoff) & (ds.index < cutoff + 86400)]
        if issues is not None:
            day = day[np.isin(day, issues)]
        for issue in day:
            chance, reason = predict_issue(model, int(issue), visible.at(int(issue)))
            records.append((int(issue), chance, reason))
        log(f"{NAME} {config.myt(cutoff):%Y-%m-%d} cases={info['cases']} {info.get('reason', '')}")
    out = pd.DataFrame(records, columns=["issue_epoch", f"{NAME}__p_ride_le70", f"{NAME}__reason"]).set_index("issue_epoch")
    out.attrs["fits"] = fits
    return out


def verify_live_artifact(cutoff: int) -> dict:
    """Compare the replay fit with the live private artifact on its own training rows."""
    import pickle
    live = prod.module("cycling_window_live")
    probability = prod.module("cycling_window_probability")
    path = config.SERVER / "private" / live.ARTIFACT_NAME
    artifact = pickle.loads(path.read_bytes())
    if artifact["trainingCutoffEpoch"] != cutoff:
        return {"skipped": "live artifact has a different cutoff"}
    snapshot = live.source._source_snapshot(config.SNAPSHOT_DB, cutoff - 1, cutoff)
    frames, _ = live._preparation().rebuild_training(*snapshot)
    train, support = live._training_support(frames["initial"], cutoff, "initial90")
    ours = probability.fit_window_model(train, candidate="sensor", cutoff_epoch=cutoff)
    theirs = artifact["models"]["initial90"]
    x = probability.feature_frame(train, "sensor")
    a = theirs["pipeline"].predict_proba(x)[:, 1]
    b = ours["pipeline"].predict_proba(x)[:, 1]
    return {"cutoff": str(config.myt(cutoff)), "replayCases": support["trainingCases"],
            "liveCases": artifact["metadata"]["heads"]["initial90"]["trainingCases"],
            "maxAbsProbabilityDiffVsLiveArtifact": float(np.max(np.abs(a - b)))}
