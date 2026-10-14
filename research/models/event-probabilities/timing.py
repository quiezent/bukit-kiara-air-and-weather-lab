"""Runtime of the chosen candidate on this laptop: one daily refit and one-issue predict.

Usage: python -B timing.py
Refits both base models for the 2026-10-13 cutoff (the largest training set),
fits both calibrators on the saved out-of-sample history, and times predicting a
single issue (features already computed, as in the live feature frame).
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import time

import numpy as np

import base_models as bm
import calibrate as cal
import common
import final
import run_base


def main():
    ds = common.load()
    cutoff = common.config.myt_epoch("2026-10-13")
    index = ds.index.to_numpy(np.int64)
    out = {}
    for name in (final.DIST, final.CLF):
        factory, lookback = run_base.CONFIGS[name]
        complete = ((ds.y_arrival_complete.to_numpy(np.int64) < cutoff)
                    & (ds.y_first20_complete.to_numpy(np.int64) < cutoff))
        train = ds.loc[(index > cutoff - lookback * 86400) & (index < cutoff - common.EMBARGO_MINUTES * 60) & complete]
        one = ds.loc[[cutoff + 17 * 3600]].drop(columns=[c for c in ds.columns if c.startswith("y_")])
        fp = factory()
        t = time.time()
        fp(train, one, cutoff)                 # fit + predict one issue
        out[f"{name} fit+predict(1 issue)"] = time.time() - t
    base = {n: common.load_pred(f"base_{n}") for n in (final.DIST, final.CLF)}
    # Predict latency for one issue: the fitted HGB objects the candidate uses.
    tr = bm.train_rows(train, "class5")
    feats = common.FEATURE_SETS["all"]
    x, xq = tr[feats].to_numpy(float), one[feats].to_numpy(float)
    models = [bm._hgb_reg().fit(x, tr.y_delta.to_numpy(float)), bm._hgb_reg().fit(x, np.abs(tr.y_delta.to_numpy(float))),
              bm._hgb_clf().fit(x, tr.y_class5.to_numpy(int)), bm._hgb_clf().fit(x, tr.y_first20.to_numpy(int))]
    t = time.time()
    for _ in range(20):
        models[0].predict(xq), models[1].predict(xq), models[2].predict_proba(xq), models[3].predict_proba(xq)
    out["predict one issue (4 HGB objects)"] = (time.time() - t) / 20
    t = time.time()
    days = [cutoff]
    cal.calibrate(ds, {final.DIST: base[final.DIST], final.CLF: base[final.CLF]}, "class5", "tempbias",
                  lookback_days=final.CAL_LOOKBACK_DAYS, days=days)
    cal.calibrate(ds, {final.CLF: base[final.CLF]}, "first20", "tempbias", lookback_days=final.CAL_LOOKBACK_DAYS,
                  days=days)
    out["calibrators fit+apply (1 day)"] = time.time() - t
    for k, v in out.items():
        print(f"{k}: {v:.3f} s")


if __name__ == "__main__":
    main()
