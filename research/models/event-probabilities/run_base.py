"""Walk the base models forward (daily refit) from 2026-09-01 and save their outputs.

Usage (from this folder, system Python with -B):
    python -B run_base.py <config> [<config> ...]

Outputs: data\\models\\event-probabilities\\base_<config>.parquet (+ .json timing).
Predictions before 2026-09-10 exist only to give the chronological calibrator
out-of-sample history; they are never scored.
"""
from __future__ import annotations

import os

# One thread: the laptop is shared with production and other research runs, and
# OpenMP with 2 threads on a saturated CPU was ~30x slower than 1 thread.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import sys
import time

import base_models as bm
import common
from harness import walkforward

FS = common.FEATURE_SETS
# name -> (factory, lookback days); 60 days = all history since 2026-08-26 during the evaluation
CONFIGS = {
    "lr_emp_28": (lambda: bm.make_logratio_empirical(), 28),
    "lr_emp_all": (lambda: bm.make_logratio_empirical(), 60),
    "lr_hgb_sw": (lambda: bm.make_logratio_hgb(FS["sensor_weather"]), 60),
    "lr_hgb_all": (lambda: bm.make_logratio_hgb(FS["all"]), 60),
    "hgb_sensor_a0": (lambda: bm.make_hgb_class(FS["sensor"], alpha=0.0), 60),
    "hgb_sw_a0": (lambda: bm.make_hgb_class(FS["sensor_weather"], alpha=0.0), 60),
    "hgb_all_a0": (lambda: bm.make_hgb_class(FS["all"], alpha=0.0), 60),
    "hgb_all_a05": (lambda: bm.make_hgb_class(FS["all"], alpha=0.5), 60),
    "hgb_all_a1": (lambda: bm.make_hgb_class(FS["all"], alpha=1.0), 60),
    "hgb_all_a0_reg": (lambda: bm.make_hgb_class(FS["all"], alpha=0.0,
                                                 kw=dict(max_leaf_nodes=7, min_samples_leaf=200, max_iter=150)), 60),
}


def run(name: str, ds=None):
    ds = common.load() if ds is None else ds
    factory, lookback = CONFIGS[name]
    days = common.days_between(ds, common.BASE_START, common.EVAL_END)
    started = time.time()
    pred = walkforward.walk_forward(ds, factory(), targets=("class5", "first20"), lookback_days=lookback,
                                    embargo_minutes=common.EMBARGO_MINUTES, days=days,
                                    log=lambda m: print(name, m, flush=True))
    elapsed = time.time() - started
    common.save(pred, f"base_{name}")
    (common.OUT / f"base_{name}.json").write_text(json.dumps(
        {"name": name, "lookback_days": lookback, "embargo_minutes": common.EMBARGO_MINUTES,
         "days": len(days), "seconds": round(elapsed, 1), "seconds_per_refit": round(elapsed / len(days), 2)}))
    print(f"{name}: {elapsed:.0f}s ({elapsed / len(days):.1f}s per daily refit)", flush=True)
    return pred


if __name__ == "__main__":
    data = common.load()
    for config_name in sys.argv[1:]:
        run(config_name, data)
