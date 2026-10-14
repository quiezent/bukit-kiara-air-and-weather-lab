"""Run incumbent replays over the evaluation period and save them.

    python -B -m harness.incumbents.run ridge_v4 arrival_logistic first_crossing
    python -B -m harness.incumbents.run cycling70
    python -B -m harness.incumbents.run morning_hgb afternoon_patchtst
    python -B -m harness.incumbents.run issued_clocks   # replays at genuinely issued clocks

Outputs: data/models/incumbents/<name>.parquet (+ <name>_fits.json).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from .. import config, dataset, issued
from . import afternoon_patchtst, arrival_logistic, common, cycling70, first_crossing, morning_hgb, ridge_v4

OUT = config.OUT / "incumbents"
ISSUED_FRAME = config.OUT / "issued_clock_frame.parquet"


def _save(name, frame, fits=None):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / f"{name}.parquet")
    fits = fits if fits is not None else frame.attrs.get("fits")
    if fits is not None:
        (OUT / f"{name}_fits.json").write_text(json.dumps(fits, indent=1, default=str), encoding="utf-8")


def load(name) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def issued_clock_frame(refresh=False) -> pd.DataFrame:
    """Features+targets at the exact clocks of issued forecasts (one per 5-min slot, Oct 8+)."""
    if ISSUED_FRAME.exists() and not refresh:
        return pd.read_parquet(ISSUED_FRAME)
    ledger = issued.ledger()
    sample = issued.slot_sample(ledger, start=config.myt_epoch("2026-10-08"))
    frame = dataset.build_frame(sample.issue_epoch.to_numpy(np.int64), progress=log)
    frame = frame.join(sample.set_index("issue_epoch").add_prefix("issued__"))
    frame.to_parquet(ISSUED_FRAME)
    return frame


def run(names):
    ds = dataset.load()
    days = common.eval_days(ds)
    for name in names:
        started = time.time()
        if name == "ridge_v4":
            _save(name, ridge_v4.replay(ds, days, log=log))
        elif name == "arrival_logistic":
            _save(name, arrival_logistic.replay(ds, days, log=log))
        elif name == "first_crossing":
            _save(name, first_crossing.replay(ds, days, log=log))
        elif name == "cycling70":
            _save(name, cycling70.replay(ds, days, log=log))
        elif name == "morning_hgb":
            _save(name, morning_hgb.replay(ds, days, log=log))
        elif name == "afternoon_patchtst":
            _save(name, afternoon_patchtst.replay(days, log=log))
        elif name == "issued_clocks":
            frame = issued_clock_frame()
            issued_days = common.eval_days(frame, start=config.myt_epoch("2026-10-08"))
            parts = [ridge_v4.replay(frame, issued_days, log=log),
                     arrival_logistic.replay(frame, issued_days, log=log),
                     first_crossing.replay(frame, issued_days, log=log),
                     cycling70.replay(frame, issued_days, log=log)]
            _save("issued_clocks", pd.concat(parts, axis=1), fits=[])
        else:
            raise SystemExit(f"unknown replay {name}")
        log(f"{name} done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    run(sys.argv[1:])
