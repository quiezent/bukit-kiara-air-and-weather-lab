"""GBM-quantile candidate: direct multi-horizon quantile gradient boosting.

One daily refit of an xgboost multi-quantile booster (``reg:quantileerror``,
one tree per quantile level and round, hist method, 1 thread) per head, on the
harness's leak-free features. (sklearn HistGradientBoosting was 3x slower per
quantile on this loaded laptop: 8.2 s vs 2.7 s for 100 rounds, 284 features.)
Every head is trained on the change relative to the fresh 5-min reference
(``s_freshPm25``, the persistence value), as a difference or (final choice) a
log ratio log((y + 5) / (ref + 5)). Quantile levels of a head are sorted per
row (monotone rearrangement), turned into a predictive CDF (piecewise linear
between knots, exponential tails beyond the outer knots) and read off literally:

* arrival (+90): pred_arrival = median; p_fall20/p_rise20/p_fall40/p_rise40
  and p_class5_0..4 = CDF masses on the production class boundaries, linearly
  pooled with the training window's class frequencies (weight clim_weight_class5).
* ride mean: pred_ride_mean = median; p_ride_le70 = CDF at 70, pooled with the
  training share of ride means <= 70 (weight clim_weight_le70).
* ride min / max: pred_ride_min / pred_ride_max = medians (MAE heads), plus
  pred_ride_min_q10 / pred_ride_max_q90 (pinball heads).

The unpooled probabilities are kept as p_class5_raw_* / p_ride_le70_raw.
No hand-written gates, no persistence substitution: where the fresh reference
is missing the outputs are NaN (unavailable). ``recalibrate`` (walk-forward PIT
recalibration) was tried on the tune split and not adopted.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xgboost as xgb

ARRIVAL_EMBARGO_SECONDS = 120 * 60      # arrival_logistic_model embargo
RIDE_EMBARGO_SECONDS = 330 * 60         # fresh_numeric_model_v4.EMBARGO_SECONDS
LOG_OFFSET = 5.0                        # log((y + c) / (ref + c)); PM2.5 >= 0


@dataclass(frozen=True)
class Config:
    name: str
    groups: tuple = ("sensor", "time", "weather", "cams")
    drop_sequence: bool = True           # drop the 180 s_seq_* 3-min bins
    transform: str = "diff"              # "diff" or "log"
    lookback_days: int = 28
    rounds: int = 100
    eta: float = 0.1
    max_depth: int = 4
    min_child_weight: float = 40.0
    subsample: float = 1.0
    colsample: float = 1.0
    l2: float = 1.0
    max_bin: int = 64
    arrival_q: tuple = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)
    ride_mean_q: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    min_q: tuple = (0.1, 0.5)
    max_q: tuple = (0.5, 0.9)
    heads: tuple = ("arrival", "ride_mean", "ride_min", "ride_max")
    # Linear pool with the training window's own climatology (class frequencies
    # of the arrival training rows / share of ride means <= 70):
    # p = (1 - w) * p_quantile + w * p_climatology. Weights chosen on the tune split.
    clim_weight_class5: float = 0.0
    clim_weight_le70: float = 0.0

    def with_(self, **kw):
        return replace(self, **kw)


# ----------------------------------------------------------------------------- transforms
def to_target(y, ref, transform):
    if transform == "log":
        return np.log((y + LOG_OFFSET) / (ref + LOG_OFFSET))
    return y - ref


def from_target(z, ref, transform):
    ref = np.asarray(ref, float)[:, None] if np.ndim(z) == 2 else np.asarray(ref, float)
    if transform == "log":
        return (ref + LOG_OFFSET) * np.exp(z) - LOG_OFFSET
    return ref + z


def threshold_in_target(level, ref, transform):
    """Target-space value of the concentration ``level`` (array); -inf where level < 0."""
    level = np.asarray(level, float)
    if transform == "log":
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.log((level + LOG_OFFSET) / (ref + LOG_OFFSET))
    else:
        out = level - ref
    return np.where(level < 0, -np.inf, out)


# ----------------------------------------------------------------------------- quantiles -> CDF
def cdf_from_quantiles(q, levels, x):
    """P(Z <= x) per row from sorted quantile knots ``q`` (n, K) at ``levels`` (K,).

    Linear between knots, exponential tails through the two outermost knots
    on each side. ``x`` is (n,); -inf gives 0, +inf gives 1.
    """
    q = np.sort(np.asarray(q, float), axis=1)
    a = np.asarray(levels, float)
    n, k = q.shape
    q = q + np.arange(k)[None, :] * 1e-6           # strictly increasing knots
    x = np.asarray(x, float)
    out = np.empty(n)
    lo, hi = x <= q[:, 0], x >= q[:, -1]
    mid = ~(lo | hi)
    if mid.any():
        qm, xm = q[mid], x[mid]
        j = np.clip((qm <= xm[:, None]).sum(axis=1) - 1, 0, k - 2)
        r = np.arange(len(xm))
        q0, q1 = qm[r, j], qm[r, j + 1]
        out[mid] = a[j] + (a[j + 1] - a[j]) * (xm - q0) / (q1 - q0)
    if lo.any():
        s = np.maximum((q[lo, 1] - q[lo, 0]) / np.log(a[1] / a[0]), 1e-9)
        out[lo] = a[0] * np.exp((x[lo] - q[lo, 0]) / s)
    if hi.any():
        s = np.maximum((q[hi, -1] - q[hi, -2]) / np.log((1 - a[-2]) / (1 - a[-1])), 1e-9)
        out[hi] = 1 - (1 - a[-1]) * np.exp(-(x[hi] - q[hi, -1]) / s)
    out[np.isneginf(x)] = 0.0
    out[np.isposinf(x)] = 1.0
    out[~np.isfinite(q).all(axis=1)] = np.nan
    return np.clip(out, 0.0, 1.0)


def median_from_quantiles(q, levels):
    q = np.sort(np.asarray(q, float), axis=1)
    levels = np.asarray(levels, float)
    return np.array([np.interp(0.5, levels, row) for row in q]) if len(q) else np.empty(0)


# ----------------------------------------------------------------------------- columns
def columns(ds: pd.DataFrame, config: Config) -> list[str]:
    from harness import walkforward
    cols = walkforward.feature_columns(ds, config.groups)
    if config.drop_sequence:
        cols = [c for c in cols if not c.startswith("s_seq_")]
    return cols


def _params(config: Config, levels) -> dict:
    return {"objective": "reg:quantileerror", "quantile_alpha": np.asarray(levels, float),
            "eta": config.eta, "max_depth": config.max_depth, "min_child_weight": config.min_child_weight,
            "subsample": config.subsample, "colsample_bytree": config.colsample, "lambda": config.l2,
            "tree_method": "hist", "max_bin": config.max_bin, "nthread": 1, "seed": 0}


def _fit_quantiles(x, z, levels, config, xq):
    """Sorted predicted quantiles (n_query, len(levels)) of z | x."""
    train = xgb.QuantileDMatrix(x, z, max_bin=config.max_bin, nthread=1)
    booster = xgb.train(_params(config, levels), train, config.rounds)
    out = booster.predict(xgb.DMatrix(xq, nthread=1)).reshape(len(xq), len(levels))
    return np.sort(out, axis=1)


# ----------------------------------------------------------------------------- fit + predict
def make_fit_predict(config: Config, cols: list[str], log=None):
    """fit_predict(train, test, cutoff) for harness.walkforward.walk_forward (embargo 120 there)."""
    t = config.transform

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cutoff: int) -> pd.DataFrame:
        idx = train.index.to_numpy(np.int64)
        ref_tr = train["s_freshPm25"].to_numpy(float)
        fresh_tr = train["m_fresh_available"].to_numpy(bool) & np.isfinite(ref_tr)
        x_tr = train[cols].to_numpy(np.float32)
        ref_te = test["s_freshPm25"].to_numpy(float)
        usable = test["m_fresh_available"].to_numpy(bool) & np.isfinite(ref_te)
        out = pd.DataFrame(index=test.index, dtype=float)
        if not usable.any():
            return out
        xq = test.loc[usable, cols].to_numpy(np.float32)
        rq = ref_te[usable]

        def put(name, values):
            col = np.full(len(test), np.nan)
            col[usable] = values
            out[name] = col

        if "arrival" in config.heads:
            ok = fresh_tr & np.isfinite(train["y_arrival"].to_numpy(float)) \
                & (idx < cutoff - ARRIVAL_EMBARGO_SECONDS)
            z = to_target(train["y_arrival"].to_numpy(float)[ok], ref_tr[ok], t)
            q = _fit_quantiles(x_tr[ok], z, config.arrival_q, config, xq)
            put("pred_arrival", from_target(median_from_quantiles(q, config.arrival_q), rq, t))
            for j, a in enumerate(config.arrival_q):
                put(f"q_arrival_{a:g}", from_target(q[:, j], rq, t))
            cdf = {}
            for x in (-40.0, -20.0, 20.0, 40.0):
                cdf[x] = cdf_from_quantiles(q, config.arrival_q, threshold_in_target(rq + x, rq, t))
            # class boundaries (label_classes): <=-40 | (-40,-20] | (-20,20) | [20,40) | >=40
            p0, p1 = cdf[-40.0], cdf[-20.0] - cdf[-40.0]
            p3, p4 = cdf[40.0] - cdf[20.0], 1 - cdf[40.0]
            p2 = 1 - p0 - p1 - p3 - p4
            labels = train["y_class5"].to_numpy(float)[ok]
            labels = labels[(labels >= 0) & (labels <= 4)]
            clim = np.bincount(labels.astype(int), minlength=5) / max(len(labels), 1)
            w = config.clim_weight_class5
            for k, p in enumerate((p0, p1, p2, p3, p4)):
                put(f"p_class5_raw_{k}", p)
            p0, p1, p2, p3, p4 = ((1 - w) * p + w * clim[k] for k, p in enumerate((p0, p1, p2, p3, p4)))
            for k, p in enumerate((p0, p1, p2, p3, p4)):
                put(f"p_class5_{k}", p)
            put("p_fall20", p0 + p1)
            put("p_rise20", p3 + p4)
            put("p_fall40", p0)
            put("p_rise40", p4)
        ride_ok = (fresh_tr & train["y_ride_joint_valid"].astype(bool).to_numpy()
                   & (train["y_ride_complete"].to_numpy(np.int64) < cutoff)
                   & (idx < cutoff - RIDE_EMBARGO_SECONDS))
        if "ride_mean" in config.heads:
            ok = ride_ok & np.isfinite(train["y_ride_mean"].to_numpy(float))
            z = to_target(train["y_ride_mean"].to_numpy(float)[ok], ref_tr[ok], t)
            q = _fit_quantiles(x_tr[ok], z, config.ride_mean_q, config, xq)
            put("pred_ride_mean", from_target(median_from_quantiles(q, config.ride_mean_q), rq, t))
            for j, a in enumerate(config.ride_mean_q):
                put(f"q_ride_mean_{a:g}", from_target(q[:, j], rq, t))
            raw = cdf_from_quantiles(q, config.ride_mean_q, threshold_in_target(np.full(len(rq), 70.0), rq, t))
            clim = float(np.mean(train["y_ride_mean"].to_numpy(float)[ok] <= 70.0))
            put("p_ride_le70_raw", raw)
            put("p_ride_le70", (1 - config.clim_weight_le70) * raw + config.clim_weight_le70 * clim)
        for head, levels, point, extra_name, extra_level in (
                ("ride_min", config.min_q, "pred_ride_min", "pred_ride_min_q10", 0.1),
                ("ride_max", config.max_q, "pred_ride_max", "pred_ride_max_q90", 0.9)):
            if head not in config.heads:
                continue
            ok = ride_ok & np.isfinite(train[f"y_{head}"].to_numpy(float))
            z = to_target(train[f"y_{head}"].to_numpy(float)[ok], ref_tr[ok], t)
            q = _fit_quantiles(x_tr[ok], z, levels, config, xq)
            for j, a in enumerate(levels):
                if abs(a - 0.5) < 1e-9:
                    put(point, from_target(q[:, j], rq, t))
                if abs(a - extra_level) < 1e-9:
                    put(extra_name, from_target(q[:, j], rq, t))
        return out

    return fit_predict


# ----------------------------------------------------------------------------- walk-forward recalibration
def recalibrate(pred: pd.DataFrame, ds: pd.DataFrame, levels_arrival, levels_ride, window_days=14, min_labels=1000):
    """PIT recalibration of the predictive CDFs from the candidate's own past forecasts.

    For each issue day D, the map G(p) = share of past PIT values <= p is
    learnt from this candidate's forecasts issued in [D - window, D) whose
    labels completed before D (midnight), and applied as F_cal = G(F). Event and
    class probabilities and P(ride mean <= 70) are re-read from F_cal; the
    quantile/point heads are unchanged. Days with fewer than ``min_labels``
    past labelled forecasts get NaN (unavailable).
    """
    out = pd.DataFrame(index=pred.index, dtype=float)
    index = pred.index.to_numpy(np.int64)
    days = np.array([_midnight(i) for i in index])
    ref = ds["s_freshPm25"].reindex(pred.index).to_numpy(float)
    heads = []
    if "q_arrival_0.5" in pred:
        qa = pred[[f"q_arrival_{a:g}" for a in levels_arrival]].to_numpy(float)
        ya = ds["y_arrival"].reindex(pred.index).to_numpy(float)
        ca = ds["y_arrival_complete"].reindex(pred.index).to_numpy(np.int64)
        heads.append(("arrival", qa, levels_arrival, ya, ca))
    if "q_ride_mean_0.5" in pred:
        qr = pred[[f"q_ride_mean_{a:g}" for a in levels_ride]].to_numpy(float)
        yr = ds["y_ride_mean"].reindex(pred.index).to_numpy(float)
        cr = ds["y_ride_complete"].reindex(pred.index).to_numpy(np.int64)
        heads.append(("ride_mean", qr, levels_ride, yr, cr))
    for name, q, levels, y, complete in heads:
        pit = cdf_from_quantiles(np.nan_to_num(q, nan=0.0), levels, np.nan_to_num(y, nan=0.0))
        pit[~(np.isfinite(q).all(axis=1) & np.isfinite(y))] = np.nan
        for day in np.unique(days):
            today = days == day
            past = (index >= day - window_days * 86400) & (index < day) & (complete < day) & np.isfinite(pit)
            if past.sum() < min_labels or not np.isfinite(q[today]).all(axis=1).any():
                continue
            u = np.sort(pit[past])

            def g(p):
                return np.searchsorted(u, p, side="right") / (len(u) + 1) + 0.5 / (len(u) + 1)

            qt, rt = q[today], ref[today]
            if name == "arrival":
                cdf = {x: g(cdf_from_quantiles(qt, levels, np.where(rt + x < 0, -np.inf, rt + x)))
                       for x in (-40.0, -20.0, 20.0, 40.0)}
                p0, p1 = cdf[-40.0], cdf[-20.0] - cdf[-40.0]
                p3, p4 = cdf[40.0] - cdf[20.0], 1 - cdf[40.0]
                # impossible falls (ref + x < 0) keep probability 0
                p0 = np.where(rt - 40 < 0, 0.0, p0)
                p1 = np.where(rt - 20 < 0, 0.0, p1)
                p2 = 1 - p0 - p1 - p3 - p4
                values = {"p_class5_0": p0, "p_class5_1": p1, "p_class5_2": p2, "p_class5_3": p3, "p_class5_4": p4,
                          "p_fall20": p0 + p1, "p_rise20": p3 + p4, "p_fall40": p0, "p_rise40": p4}
            else:
                values = {"p_ride_le70": g(cdf_from_quantiles(qt, levels, np.full(len(qt), 70.0)))}
            for column, v in values.items():
                if column not in out:
                    out[column] = np.nan
                col = out[column].to_numpy(float)
                col[today] = v
                out[column] = col
    return out


def _midnight(epoch):
    from harness import config as hconfig
    return hconfig.midnight(epoch)
