"""Robustness check (skeptic): re-run the two base models with other HGB settings.

Usage: python -B robustness_base.py <config> [<config> ...]
Writes data\\models\\event-probabilities\\robust_base_<config>.parquet. Same
walk-forward (daily refit, 60-day lookback, 120-min embargo, labels complete
before D) as run_base.py; only the HGB hyperparameters differ.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import sys
import time

import base_models as bm
import common
from harness import walkforward

ALL = common.FEATURE_SETS["all"]
LOOSE = dict(max_leaf_nodes=31, min_samples_leaf=20, max_iter=200, learning_rate=0.05)
TIGHT = dict(max_leaf_nodes=7, min_samples_leaf=200, max_iter=150)
SLOW = dict(learning_rate=0.035, max_iter=200)
CONFIGS = {
    "hgb_all_a0_loose": lambda: bm.make_hgb_class(ALL, alpha=0.0, kw=LOOSE),
    "hgb_all_a0_slow": lambda: bm.make_hgb_class(ALL, alpha=0.0, kw=SLOW),
    "lr_hgb_all_tight": lambda: bm.make_logratio_hgb(ALL, loc_kw=TIGHT, scale_kw=TIGHT),
    "lr_hgb_all_loose": lambda: bm.make_logratio_hgb(ALL, loc_kw=LOOSE, scale_kw=LOOSE),
}


def run(name, ds):
    days = common.days_between(ds, common.BASE_START, common.EVAL_END)
    started = time.time()
    pred = walkforward.walk_forward(ds, CONFIGS[name](), targets=("class5", "first20"), lookback_days=60,
                                    embargo_minutes=common.EMBARGO_MINUTES, days=days,
                                    log=lambda m: print(name, m, flush=True))
    common.save(pred, f"robust_base_{name}")
    (common.OUT / f"robust_base_{name}.json").write_text(json.dumps(
        {"name": name, "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    data = common.load()
    for config_name in sys.argv[1:]:
        run(config_name, data)
