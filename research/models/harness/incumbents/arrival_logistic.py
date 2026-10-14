"""Walk-forward replay of arrival_logistic_model (5-class +90 arrival change, 20/40).

Production recipe (arrival_logistic_model.refresh_model): cutoff = issue-day
MYT midnight; aligned 300-s origins in [cutoff-28 d, cutoff-120 min); fresh
reference available and >= 0; arrival label complete < cutoff; 28 base sensor
features; production ``fit_prepared`` (median imputer + indicators, scaler,
multinomial LogisticRegression C=0.1). The prepared dict is built from the
receipt cache and handed to the production fit function unchanged.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .. import config, prod
from . import common

NAME = "arrival_logistic"
PROB_COLUMNS = ["p_fall40", "p_fall20to40", "p_within20", "p_rise20to40", "p_rise40"]


def prepared_from_cache(cache: pd.DataFrame, cutoff: int) -> dict:
    model = prod.module("arrival_logistic_model")
    change = prod.module("arrival_change_model")
    columns = list(model.FEATURE_COLUMNS)
    issues = cache.index.to_numpy(np.int64)
    window = (issues >= cutoff - 28 * 86400) & (issues < cutoff - 7200)
    part = cache.loc[window]
    reference = part["freshPm25"].to_numpy(float)
    arrival = part["y_arrival"].to_numpy(float)
    complete = part["y_arrival_complete"].to_numpy(np.int64)
    keep = (part["_available"].to_numpy(bool) & np.isfinite(reference) & (reference >= 0)
            & np.isfinite(arrival) & (complete < cutoff))
    part = part.loc[keep]
    delta = part["y_arrival"].to_numpy(float) - part["freshPm25"].to_numpy(float)
    records = pd.DataFrame({
        "issueEpoch": part.index.to_numpy(np.int64), "referencePm": part["freshPm25"].to_numpy(float),
        "arrivalPm": part["y_arrival"].to_numpy(float), "arrivalDelta": delta,
        "arrivalClass": change.label_classes(delta) if len(delta) else np.array([], int),
        "completeEpoch": part["y_arrival_complete"].to_numpy(np.int64),
        "issueDay": [datetime.fromtimestamp(int(i), config.MYT).date().isoformat() for i in part.index],
        "featureSourceMaxEpoch": part["_featureSourceMaxEpoch"].to_numpy(float)})
    return {"features": part[columns].to_numpy(float), "issues": records, "columns": columns,
            "trainingCadence": model.TRAINING_CADENCE}


def fit(cache, cutoff):
    model = prod.module("arrival_logistic_model")
    prepared = prepared_from_cache(cache, cutoff)
    try:
        artifact = model.fit_prepared(prepared, cutoff, fitted_epoch=cutoff)
    except ValueError as error:
        return None, {"cutoff": cutoff, "issues": len(prepared["issues"]), "reason": str(error)}
    return artifact, {"cutoff": cutoff, "issues": len(prepared["issues"]),
                      "support": artifact["metadata"]["trainingSupport"]}


def predict(artifact, frame: pd.DataFrame) -> pd.DataFrame:
    model = prod.module("arrival_logistic_model")
    out = pd.DataFrame(index=frame.index, columns=PROB_COLUMNS, dtype=float)
    usable = frame["m_fresh_available"].to_numpy(bool) & (frame["s_freshPm25"].to_numpy(float) >= 0)
    if artifact is None or not usable.any():
        return out
    x = frame.loc[usable, ["s_" + c for c in model.FEATURE_COLUMNS]].to_numpy(float)
    out.loc[usable, :] = model.predict_probabilities(artifact, x)
    return out


def add_tails(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Production tail_probabilities: fall20 = fall40 + fall20to40, rise20 likewise."""
    frame[prefix + "p_fall20"] = frame[prefix + "p_fall40"] + frame[prefix + "p_fall20to40"]
    frame[prefix + "p_rise20"] = frame[prefix + "p_rise40"] + frame[prefix + "p_rise20to40"]
    return frame


def replay(ds: pd.DataFrame, days=None, cache=None, log=print) -> pd.DataFrame:
    cache = common.receipt_cache() if cache is None else cache
    days = common.eval_days(ds) if days is None else days
    parts, fits = [], []
    for cutoff in days:
        artifact, info = fit(cache, cutoff)
        fits.append(info)
        parts.append(predict(artifact, ds.loc[(ds.index >= cutoff) & (ds.index < cutoff + 86400)]))
        log(f"{NAME} {config.myt(cutoff):%Y-%m-%d} issues={info['issues']} {info.get('reason', '')}")
    out = add_tails(pd.concat(parts).add_prefix(f"{NAME}__"), f"{NAME}__")
    out.attrs["fits"] = fits
    return out


def verify(cutoff: int, cache=None) -> dict:
    """Production load_inputs + prepare_training_data + fit_prepared vs the cache recipe."""
    model = prod.module("arrival_logistic_model")
    inputs = prod.module("arrival_feature_inputs")
    cache = common.receipt_cache() if cache is None else cache
    source = inputs.load_inputs(config.SNAPSHOT_DB, cutoff - 28 * 86400 - 10800, cutoff)
    prepared = model.prepare_training_data(source, model.dense_origins(cutoff - 28 * 86400, cutoff - 7200), cutoff)
    production = model.fit_prepared(prepared, cutoff, fitted_epoch=cutoff)
    ours, info = fit(cache, cutoff)
    mine = prepared_from_cache(cache, cutoff)
    probe = cache.loc[(cache.index >= cutoff) & (cache.index < cutoff + 86400) & cache["_available"]]
    x = probe[list(model.FEATURE_COLUMNS)].to_numpy(float)
    a = model.predict_probabilities(production, x)
    b = model.predict_probabilities(ours, x)
    result = {"cutoff": str(config.myt(cutoff)), "productionIssues": len(prepared["issues"]),
              "cacheIssues": len(mine["issues"]),
              "sameIssueClocks": bool(np.array_equal(prepared["issues"].issueEpoch.to_numpy(), mine["issues"].issueEpoch.to_numpy())),
              "maxAbsFeatureDiff": float(np.nanmax(np.abs(np.asarray(prepared["features"]) - mine["features"])))
              if len(mine["issues"]) == len(prepared["issues"]) else None,
              "sameLabels": bool(np.array_equal(prepared["issues"].arrivalClass.to_numpy(), mine["issues"].arrivalClass.to_numpy()))
              if len(mine["issues"]) == len(prepared["issues"]) else False,
              "maxAbsProbabilityDiffVsProductionFit": float(np.max(np.abs(a - b)))}
    path = config.SERVER / "private" / model.ARTIFACT_NAME
    if path.exists():
        import pickle
        live = pickle.loads(path.read_bytes())
        if live.get("trainingCutoffEpoch") == cutoff:
            c = model.predict_probabilities(live, x)
            result["maxAbsProbabilityDiffVsLiveArtifact"] = float(np.max(np.abs(c - b)))
            result["liveTrainingIssues"] = live["metadata"]["trainingIssueCount"]
    return result
