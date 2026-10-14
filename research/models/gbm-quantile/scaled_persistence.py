"""Featureless 'scaled persistence' baseline: the training window's empirical distribution of
z = log((y+5)/(ref+5)) applied to the issue's fresh reference (same training rows/embargo as `final`).

Run from research\models: python -B gbm-quantile/scaled_persistence.py test   (or tune)
"""
import sys
sys.path.insert(0, r"research\models")
sys.path.insert(0, r"research\models\gbm-quantile")
import numpy as np, pandas as pd
from harness import config as hc
import evaluate as ev
SPLIT = sys.argv[1]
ds = ev.load_ds()
idx = ds.index.to_numpy(np.int64)
ref = ds.s_freshPm25.to_numpy(float)
fresh = ds.m_fresh_available.to_numpy(bool) & np.isfinite(ref)
out = pd.DataFrame(index=ds.index, dtype=float)
cols = {k: np.full(len(ds), np.nan) for k in ["pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max",
        "pred_ride_min_q10", "pred_ride_max_q90", "p_ride_le70"] + [f"p_class5_{k}" for k in range(5)]}
for cutoff in ev.split_days(SPLIT):
    win = (idx > cutoff - 28 * 86400)
    arr = win & (idx < cutoff - 7200) & fresh & (ds.y_arrival_complete.to_numpy(np.int64) < cutoff) & np.isfinite(ds.y_arrival.to_numpy(float))
    ride = (win & (idx < cutoff - 19800) & fresh & ds.y_ride_joint_valid.astype(bool).to_numpy()
            & (ds.y_ride_complete.to_numpy(np.int64) < cutoff) & (ds.y_arrival_complete.to_numpy(np.int64) < cutoff))
    te = (idx >= cutoff) & (idx < cutoff + 86400) & fresh
    r = ref[te]
    za = np.log((ds.y_arrival.to_numpy(float)[arr] + 5) / (ref[arr] + 5))
    cols["pred_arrival"][te] = (r + 5) * np.exp(np.median(za)) - 5
    za_s = np.sort(za)
    def cdf(z_sorted, x):  # P(Z <= x)
        return np.searchsorted(z_sorted, x, side="right") / len(z_sorted)
    def thr(level):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(level < 0, -np.inf, np.log((level + 5) / (r + 5)))
    c = {x: cdf(za_s, thr(r + x)) for x in (-40, -20, 20, 40)}
    # strict/inclusive boundaries as label_classes: <=-40 | (-40,-20] | (-20,20) | [20,40) | >=40
    c20 = np.searchsorted(za_s, thr(r + 20), side="left") / len(za_s)
    c40 = np.searchsorted(za_s, thr(r + 40), side="left") / len(za_s)
    p0, p1 = c[-40], c[-20] - c[-40]
    p3, p4 = c40 - c20, 1 - c40
    p2 = 1 - p0 - p1 - p3 - p4
    for k, p in enumerate((p0, p1, p2, p3, p4)):
        cols[f"p_class5_{k}"][te] = p
    for head in ("ride_mean", "ride_min", "ride_max"):
        y = ds[f"y_{head}"].to_numpy(float)
        ok = ride & np.isfinite(y)
        z = np.log((y[ok] + 5) / (ref[ok] + 5))
        cols[f"pred_{head}"][te] = (r + 5) * np.exp(np.median(z)) - 5
        if head == "ride_min":
            cols["pred_ride_min_q10"][te] = (r + 5) * np.exp(np.quantile(z, 0.1)) - 5
        if head == "ride_max":
            cols["pred_ride_max_q90"][te] = (r + 5) * np.exp(np.quantile(z, 0.9)) - 5
        if head == "ride_mean":
            cols["p_ride_le70"][te] = cdf(np.sort(z), thr(np.full(len(r), 70.0)))
for k, v in cols.items():
    out[k] = v
out["p_fall20"] = out.p_class5_0 + out.p_class5_1
out["p_rise20"] = out.p_class5_3 + out.p_class5_4
out["p_fall40"], out["p_rise40"] = out.p_class5_0, out.p_class5_4
out = out.loc[out.pred_arrival.notna()]
out.to_parquet(hc.OUT / "gbm-quantile" / "leakcheck" / ("ratio_baseline__" + SPLIT + ".parquet"))
rows, _ = ev.compare(out, SPLIT)
rows_c, _ = ev.compare(ev.load_pred("final", SPLIT).reindex(out.index), SPLIT)
print(ev.table(["Target", "Metric", "Persistence / prior", "Incumbent replay", "Scaled-persistence baseline", "GBM final", "n"],
               [(a[0], a[1], a[2], a[3], a[4], b[4], a[5]) for a, b in zip(rows, rows_c) if a[:2] == b[:2]]))
