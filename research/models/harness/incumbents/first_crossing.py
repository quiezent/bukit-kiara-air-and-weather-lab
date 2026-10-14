"""Walk-forward replay of fresh_event_model_v2 (first |change| >= 20 within 90 min, HGB).

Production recipe (fresh_event_model_v2.refresh_model): cutoff = issue-day MYT
midnight; variable-offset origins (bases + 30 s + 0..4 min) in [cutoff-28 d,
cutoff); readings table only (no receipts); 208 fresh sensor features; labels
= first of six interpolated proxies (+15..+90) with |delta| >= 20, all six
complete < cutoff; production ``fit_prepared`` (HGB, 3 classes drop/none/rise).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .. import config, prod
from . import common

NAME = "first_crossing"
PROB_COLUMNS = ["p_drop", "p_none", "p_rise"]


def prepared_from_cache(cache: pd.DataFrame, cutoff: int) -> dict:
    sensor = prod.module("fresh_sensor_features")
    columns = list(sensor.FEATURE_COLUMNS)
    issues = cache.index.to_numpy(np.int64)
    window = (issues >= cutoff - 28 * 86400) & (issues < cutoff)
    part = cache.loc[window]
    reference = part["freshPm25"].to_numpy(float)
    keep = (part["_available"].to_numpy(bool) & np.isfinite(reference) & (reference >= 0)
            & np.isfinite(part["y_first20"].to_numpy(float)) & (part["y_first20_complete"].to_numpy(np.int64) < cutoff))
    part = part.loc[keep]
    records = pd.DataFrame({
        "issueEpoch": part.index.to_numpy(np.int64), "referencePm": part["freshPm25"].to_numpy(float),
        "completeEpoch": part["y_first20_complete"].to_numpy(np.int64),
        "cause": part["y_first20"].to_numpy(float).astype(int),
        "firstStep": part["y_first20_step"].to_numpy(float).astype(int),
        "issueDay": [datetime.fromtimestamp(int(i), config.MYT).date().isoformat() for i in part.index],
        "featureSourceMaxEpoch": part["_featureSourceMaxEpoch"].to_numpy(float)})
    return {"features": part[columns].to_numpy(float), "issues": records, "columns": columns}


def fit(cache, cutoff):
    model = prod.module("fresh_event_model_v2")
    prepared = prepared_from_cache(cache, cutoff)
    try:
        artifact = model.fit_prepared(prepared, cutoff, fitted_epoch=cutoff)
    except ValueError as error:
        return None, {"cutoff": cutoff, "issues": len(prepared["issues"]), "reason": str(error)}
    return artifact, {"cutoff": cutoff, "issues": len(prepared["issues"]),
                      "support": artifact["metadata"]["trainingSupport"]}


def predict(artifact, frame: pd.DataFrame) -> pd.DataFrame:
    model = prod.module("fresh_event_model_v2")
    out = pd.DataFrame(index=frame.index, columns=PROB_COLUMNS, dtype=float)
    usable = frame["m_fresh_available"].to_numpy(bool) & (frame["s_freshPm25"].to_numpy(float) >= 0)
    if artifact is None or not usable.any():
        return out
    x = frame.loc[usable, ["s_" + c for c in artifact["columns"]]].to_numpy(float)
    out.loc[usable, :] = model.predict_probabilities(artifact, x)
    return out


def replay(ds: pd.DataFrame, days=None, cache=None, log=print) -> pd.DataFrame:
    cache = common.plain_cache() if cache is None else cache
    days = common.eval_days(ds) if days is None else days
    parts, fits = [], []
    for cutoff in days:
        artifact, info = fit(cache, cutoff)
        fits.append(info)
        parts.append(predict(artifact, ds.loc[(ds.index >= cutoff) & (ds.index < cutoff + 86400)]))
        log(f"{NAME} {config.myt(cutoff):%Y-%m-%d} issues={info['issues']} {info.get('reason', '')}")
    out = pd.concat(parts).add_prefix(f"{NAME}__")
    out.attrs["fits"] = fits
    return out


def verify(cutoff: int, cache=None) -> dict:
    model = prod.module("fresh_event_model_v2")
    cache = common.plain_cache() if cache is None else cache
    rows = model.read_training_rows(config.SNAPSHOT_DB, cutoff)
    prepared = model.prepare_training_data(rows, model.dense_origins(cutoff - 28 * 86400, cutoff), cutoff)
    production = model.fit_prepared(prepared, cutoff, fitted_epoch=cutoff)
    mine = prepared_from_cache(cache, cutoff)
    ours = model.fit_prepared(mine, cutoff, fitted_epoch=cutoff)
    same = len(mine["issues"]) == len(prepared["issues"])
    probe = cache.loc[(cache.index >= cutoff) & (cache.index < cutoff + 86400) & cache["_available"]]
    x = probe[list(ours["columns"])].to_numpy(float)
    a, b = model.predict_probabilities(production, x), model.predict_probabilities(ours, x)
    result = {"cutoff": str(config.myt(cutoff)), "productionIssues": len(prepared["issues"]),
              "cacheIssues": len(mine["issues"]),
              "sameIssueClocks": same and bool(np.array_equal(prepared["issues"].issueEpoch.to_numpy(), mine["issues"].issueEpoch.to_numpy())),
              "sameLabels": same and bool(np.array_equal(prepared["issues"].cause.to_numpy(), mine["issues"].cause.to_numpy())),
              "maxAbsFeatureDiff": float(np.nanmax(np.abs(prepared["features"] - mine["features"]))) if same else None,
              "maxAbsProbabilityDiffVsProductionFit": float(np.max(np.abs(a - b)))}
    path = config.SERVER / "private" / model.ARTIFACT_NAME
    if path.exists():
        import pickle
        live = pickle.loads(path.read_bytes())
        if live.get("trainingCutoffEpoch") == cutoff:
            result["maxAbsProbabilityDiffVsLiveArtifact"] = float(np.max(np.abs(model.predict_probabilities(live, x) - b)))
    return result
