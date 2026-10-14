"""Replay of the Afternoon PatchTST (sequence_followup_20261008 daily runtime).

Two eras, both with the production trainer (research/sequence_models_20261008/
sequence_trainer.SequenceRegressor 'patchtst', 30 epochs, seed 1749, CUDA fit,
CPU inference) and its raw output max(0, fresh + 20*head) (no deadband; v21+
publishes the raw number):

* cutoffs <= 2026-10-08: walk-forward refits on the certified prefix
  (session_train.npz, 6328 original-input rows, issues 2026-09-07..10-07) with
  the daily_cache._eligible_training rule: issue in [cutoff-14 d, cutoff),
  complete < cutoff, finite label, sequenceValid & valid; >= 120 rows and >= 10
  target dates. Queries are the prefix's own rows issued that day:
  12:00 -> same-day 14:00-16:00 (lead 120; the canonical Afternoon clock) and
  20:00 -> next-day 14:00-16:00 (lead 1080; the prefix has no 18:05 rows, so this
  is the closest next-day head and is labelled ``next_day_2000``).
* cutoffs 2026-10-08..13: the production daily checkpoints
  (receipt_runtime_v2_lifecycle/models/<cutoff>_patchtst.pt) applied to the
  genuinely captured canonical input groups (12:00 afternoon, 18:05 nextAfternoon).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

from .. import config, prod

NAME = "afternoon_patchtst"
_TRAINER = None


def _paths():
    root = prod.setup()
    return (root / "research/sequence_models_20261008",
            root / "research/sequence_followup_20261008/forward")


def trainer():
    global _TRAINER
    if _TRAINER is None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        experiment, _ = _paths()
        spec = importlib.util.spec_from_file_location("harness_sequence_trainer", experiment / "sequence_trainer.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["harness_sequence_trainer"] = module
        spec.loader.exec_module(module)
        _TRAINER = module
    return _TRAINER


def prefix():
    experiment, _ = _paths()
    with np.load(experiment / "data" / "session_train.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    meta = pd.read_pickle(experiment / "data" / "session_train_metadata.pkl").reset_index(drop=True)
    return arrays, meta


def eligible(arrays, meta, cutoff):
    mask = (meta.issueEpoch.ge(cutoff - 14 * 86400) & meta.issueEpoch.lt(cutoff) & meta.completeEpoch.lt(cutoff)
            & np.isfinite(meta.actual) & meta.sequenceValid & arrays["valid"])
    if "valid" in meta:
        mask &= meta.valid
    return np.flatnonzero(mask.to_numpy())


def _bundle(arrays, indices):
    return {key: arrays[key][indices] for key in ("past", "future", "context", "fresh", "y", "valid")}


def fit(arrays, meta, cutoff):
    indices = eligible(arrays, meta, cutoff)
    days = meta.iloc[indices].targetDay.nunique()
    if len(indices) < 120 or days < 10:
        return None, {"cutoff": cutoff, "rows": len(indices), "targetDates": int(days), "reason": "insufficient_support"}
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = trainer().SequenceRegressor("patchtst", device=device).fit(_bundle(arrays, indices), np.arange(len(indices)))
    return model, {"cutoff": cutoff, "rows": len(indices), "targetDates": int(days), "device": device,
                   "trainingSeconds": round(model.training_metadata["trainingSeconds"], 1)}


def _queries(meta, cutoff):
    issue = meta.issueEpoch.to_numpy(np.int64)
    local = pd.to_datetime(issue, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur")
    start_local = pd.to_datetime(meta.startEpoch.to_numpy(np.int64), unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur")
    day = (issue >= cutoff) & (issue < cutoff + 86400)
    at14 = (start_local.hour == 14) & (start_local.minute == 0) & ((meta.endEpoch - meta.startEpoch) == 7200).to_numpy()
    same = day & at14 & (local.strftime("%H:%M:%S") == "12:00:00") & (meta.leadMinutes.to_numpy() == 120)
    nxt = day & at14 & (local.strftime("%H:%M:%S") == "20:00:00") & (meta.leadMinutes.to_numpy() == 1080)
    return {"same_day_1200": np.flatnonzero(same), "next_day_2000": np.flatnonzero(nxt)}


def replay_prefix(days, log=print):
    arrays, meta = prefix()
    records, fits = [], []
    for cutoff in days:
        if cutoff > config.myt_epoch("2026-10-08"):
            continue
        queries = _queries(meta, cutoff)
        if not any(len(v) for v in queries.values()):
            continue
        model, info = fit(arrays, meta, cutoff)
        fits.append(info)
        log(f"{NAME} prefix {config.myt(cutoff):%Y-%m-%d} rows={info['rows']} {info.get('reason', '')}")
        for head, indices in queries.items():
            for i in indices:
                pred = float(model.predict(_bundle(arrays, np.array([i])), np.array([0]))[0]) if model is not None else np.nan
                row = meta.iloc[i]
                records.append({"head": head, "era": "prefix_refit", "issue_epoch": int(row.issueEpoch),
                                "start_epoch": int(row.startEpoch), "fresh": float(row.fresh),
                                "pred": pred, "actual_prefix": float(row.actual), "cutoff": cutoff})
    return pd.DataFrame(records), fits


def replay_lifecycle(days, log=print):
    """Production daily checkpoints on captured canonical inputs (2026-10-08..)."""
    _, forward = _paths()
    lifecycle = forward / "receipt_runtime_v2_lifecycle"
    records = []
    for cutoff in days:
        checkpoint = lifecycle / "models" / f"{cutoff}_patchtst.pt"
        if not checkpoint.exists():
            continue
        model = trainer().SequenceRegressor.load(checkpoint)
        info = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        date = config.myt(cutoff).strftime("%Y-%m-%d")
        for kind, head in (("afternoon", "same_day_1200"), ("nextAfternoon", "next_day_1805")):
            for group_path in sorted((lifecycle / "inputs" / date).glob(f"*_{kind}.json")):
                group = json.loads(group_path.read_text(encoding="utf-8"))
                row = group["rows"][0]
                if info.get("fittedAtEpoch", 0) > row["issueEpoch"]:
                    log(f"{NAME} {date} {kind}: checkpoint fitted after issue; skipped")
                    continue
                array_path = forward / group["arrayPath"].replace("\\", "/")
                with np.load(array_path, allow_pickle=False) as stored:
                    bundle = {key: stored[key] for key in stored.files}
                pred = float(model.predict(bundle, np.array([0]))[0])
                records.append({"head": head, "era": "production_checkpoint", "issue_epoch": int(row["issueEpoch"]),
                                "start_epoch": int(row["startEpoch"]), "fresh": float(row["fresh"]),
                                "pred": pred, "actual_prefix": np.nan, "cutoff": cutoff})
        log(f"{NAME} lifecycle {date}")
    return pd.DataFrame(records)


def replay(days, log=print) -> pd.DataFrame:
    a, fits = replay_prefix(days, log)
    b = replay_lifecycle([d for d in days if d >= config.myt_epoch("2026-10-08")], log)
    out = pd.concat([a, b], ignore_index=True)
    out.attrs["fits"] = fits
    return out


def verify_oct8(log=print) -> dict:
    """Refit at the 2026-10-08 cutoff and compare with the adopted original checkpoint."""
    arrays, meta = prefix()
    cutoff = config.myt_epoch("2026-10-08")
    model, info = fit(arrays, meta, cutoff)
    _, forward = _paths()
    original = trainer().SequenceRegressor.load(forward / "receipt_runtime_v2_lifecycle" / "models" / f"{cutoff}_patchtst.pt")
    probe = np.flatnonzero((meta.issueEpoch >= cutoff - 2 * 86400).to_numpy())
    bundle = _bundle(arrays, probe)
    a = model.predict(bundle, np.arange(len(probe)))
    b = original.predict(bundle, np.arange(len(probe)))
    return {"cutoff": str(config.myt(cutoff)), "rows": info["rows"],
            "maxAbsPredictionDiffVsProductionCheckpoint": float(np.max(np.abs(a - b))),
            "meanAbsPredictionDiff": float(np.mean(np.abs(a - b)))}
