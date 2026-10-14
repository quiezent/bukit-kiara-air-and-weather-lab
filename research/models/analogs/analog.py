"""Analog (nearest-neighbour) forecaster for the TTDI PM2.5 heads.

For each issue, find past issues whose last 3 h of PM2.5 (shape relative to the
current reference), level, time of day and issued weather looked most alike,
and forecast from what followed them. Each neighbour's outcome is transferred
to the current reference level (ratio or additive), then:

* points (+90 arrival, ride mean/min/max): weighted median of the transferred values;
* probabilities (5 arrival-change classes and their >=20/>=40 tails, first
  crossing, ride mean <= 70): weighted share of neighbours whose transferred
  outcome lands in the class, shrunk towards the library's 28-day class
  frequency with ``alpha`` pseudo-neighbours.

Every output is the literal analog estimate: no gates, no persistence fallback.
The library is the walk-forward training frame (label-complete rows before the
day's cutoff); the query uses only feature columns visible at the issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from harness import config

OFFSET = 5.0                                   # log(x + OFFSET): stabilises ratios at low PM
PATH_LEADS = (15, 30, 45, 60, 75, 90)          # first-crossing proxies
CLASS_EDGES = (-40.0, -20.0, 20.0, 40.0)       # arrival_change_model.label_classes boundaries


# ----------------------------------------------------------------------------- feature blocks
def _lr(x, ref):
    return np.log(np.asarray(x, float) + OFFSET) - np.log(np.asarray(ref, float) + OFFSET)


def analog_frame(ds: pd.DataFrame) -> pd.DataFrame:
    """Raw analog coordinates (a_<block>__<name>) plus the future path labels (y_path_<lead>).

    Every a_ column is a function of feature columns visible at the issue. The
    y_path_<lead> labels are the interpolated proxies at issue+lead, which equal
    y_arrival of the issue lead-90 min earlier (the proxy depends only on the
    absolute clock; this reproduces every y_first20 label exactly). They are
    y_ columns, so walk_forward strips them from the test frame, and each one
    completes no later than the row's own y_arrival.
    """
    ref = ds.s_freshPm25.where(ds.m_fresh_available).to_numpy(float)
    out = {}
    seq = {int(c.split("_")[3]): c for c in ds.columns if c.startswith("s_seq_pm25_")}
    # recent: 3-min bins over the last 30 min
    for age in range(0, 30, 3):
        out[f"a_recent__{age}"] = _lr(ds[seq[age]], ref)
    # traj: 15-min block means over 180 min
    for block in range(12):
        cols = [seq[a] for a in range(block * 15, block * 15 + 15, 3)]
        out[f"a_traj__{block * 15}"] = _lr(ds[cols].mean(axis=1, skipna=True), ref)
    out["a_level__logref"] = np.log(ref + OFFSET)
    out["a_tod__sin"] = ds.t_day_sin.to_numpy(float)
    out["a_tod__cos"] = ds.t_day_cos.to_numpy(float)
    out["a_wx__next3h_precip"] = np.log1p(ds.w_next3h_precip_sum)
    out["a_wx__ride_precip"] = np.log1p(ds.w_ride_precip_sum)
    out["a_wx__prior3h_precip"] = np.log1p(ds.w_prior3h_precip_sum)
    out["a_wx__pprob90"] = ds.w_pprob_90 / 100
    out["a_wx__pprob6h"] = ds.w_next6h_pprob_max / 100
    out["a_wx__wspd90"] = ds.w_wspd10_90
    out["a_wx__u90"] = ds.w_wind10_u_90
    out["a_wx__v90"] = ds.w_wind10_v_90
    out["a_wx__logblh90"] = np.log(ds.w_blh_90.clip(lower=10))
    out["a_wx__blh_change"] = np.log(ds.w_blh_90.clip(lower=10)) - np.log(ds.w_blh_0.clip(lower=10))
    out["a_wx__rh_change"] = ds.w_rh_90 - ds.w_rh_0
    out["a_met__dtemp30"] = ds.s_rawTemperatureDelta30
    out["a_met__drh30"] = ds.s_rawHumidityDelta30
    out["a_met__rh"] = ds.s_freshHumidity
    out["a_met__temp"] = ds.s_freshTemperature
    out["a_cams__delta90"] = _lr(ds.c_pm25_90, ds.c_pm25_0)
    out["a_cams__ride"] = _lr(ds.c_pm25_ride_mean, ds.c_pm25_0)
    out["a_cams__gap"] = _lr(ds.c_pm25_0, ref)
    out["a_subang__rain"] = ds.o_rain
    out["a_subang__thunder"] = ds.o_thunder
    out["a_subang__logvis"] = np.log(ds.o_visibility_km.clip(lower=0.1))
    out["a_subang__rh"] = ds.o_rh
    frame = pd.DataFrame({k: np.asarray(v, float) for k, v in out.items()}, index=ds.index)
    arrival = ds.y_arrival
    for lead in PATH_LEADS:
        frame[f"y_path_{lead}"] = arrival.reindex(ds.index - (90 - lead) * 60).to_numpy(float)
    return frame


def slim_dataset(ds: pd.DataFrame) -> pd.DataFrame:
    """Columns the analog needs: a_ coordinates, reference, all y_ labels (walk_forward strips them from test)."""
    keep = ["eval", "m_fresh_available", "s_freshPm25"] + [c for c in ds.columns if c.startswith("y_")]
    return ds[keep].join(analog_frame(ds))


# ----------------------------------------------------------------------------- configuration
BLOCKS = ("recent", "traj", "level", "tod", "wx", "met", "cams", "subang")


@dataclass(frozen=True)
class AnalogConfig:
    weights: dict = field(default_factory=lambda: {"recent": 1.0, "traj": 1.0, "level": 1.0, "tod": 1.0,
                                                   "wx": 1.0, "met": 0.0, "cams": 0.0, "subang": 0.0})
    k: int = 50                 # neighbours
    bandwidth: float = 1.0      # Gaussian kernel exp(-d2 / (bandwidth * d2_K)); inf = uniform
    transfer: str = "ratio"     # ratio | additive
    alpha: float = 5.0          # prior pseudo-neighbours for probabilities
    traj_tau: float = np.inf    # extra age decay (minutes) inside the traj block
    point_stat: str = "median"  # median | mean

    def with_(self, **kw):
        if "weights" in kw:
            kw["weights"] = {**self.weights, **kw["weights"]}
        return replace(self, **kw)

    def label(self):
        w = ",".join(f"{b}={self.weights.get(b, 0):g}" for b in BLOCKS if self.weights.get(b, 0))
        return (f"k={self.k} bw={self.bandwidth:g} {self.transfer} a={self.alpha:g} tau={self.traj_tau:g} "
                f"{self.point_stat} [{w}]")


def _column_weights(columns, cfg: AnalogConfig) -> np.ndarray:
    """Per-column multiplier on standardised coordinates: each block contributes w_b * mean sq diff."""
    blocks = np.array([c.split("__")[0][2:] for c in columns])
    out = np.zeros(len(columns))
    for block in np.unique(blocks):
        weight = cfg.weights.get(block, 0.0)
        if weight <= 0:
            continue
        idx = np.flatnonzero(blocks == block)
        inner = np.ones(len(idx))
        if block == "traj" and np.isfinite(cfg.traj_tau):
            ages = np.array([float(columns[i].split("__")[1]) + 7.5 for i in idx])
            inner = np.exp(-ages / cfg.traj_tau)
            inner = inner / inner.mean()
        out[idx] = weight * inner / len(idx)
    return out


# ----------------------------------------------------------------------------- core
def _transfer(values, nref, qref, mode):
    """Move neighbour outcomes (n_q x K) from their reference nref to the query reference qref."""
    if mode == "ratio":
        return (qref[:, None] + OFFSET) * (values + OFFSET) / (nref + OFFSET) - OFFSET
    return qref[:, None] + values - nref


def _weighted_quantile(values, weights, q=0.5):
    values = np.where(np.isfinite(values), values, np.nan)
    weights = np.where(np.isfinite(values), weights, 0.0)
    order = np.argsort(np.nan_to_num(values, nan=np.inf), axis=1)
    v = np.take_along_axis(values, order, 1)
    w = np.take_along_axis(weights, order, 1)
    total = w.sum(1)
    cum = np.cumsum(w, 1)
    idx = np.argmax(cum >= q * total[:, None], axis=1)
    out = v[np.arange(len(v)), idx]
    out[total <= 0] = np.nan
    return out


def _weighted_mean(values, weights):
    weights = np.where(np.isfinite(values), weights, 0.0)
    total = weights.sum(1)
    out = np.nansum(np.nan_to_num(values) * weights, 1) / np.where(total > 0, total, np.nan)
    return out


def _share(flags, valid, weights, prior, alpha):
    """Shrunk weighted share: (sum w*flag + alpha*prior) / (sum w + alpha)."""
    w = np.where(valid, weights, 0.0)
    return ((w * flags).sum(1) + alpha * prior) / (w.sum(1) + alpha)


def _class5(delta):
    out = np.full(delta.shape, -1, np.int64)
    finite = np.isfinite(delta)
    d = delta[finite]
    out[finite] = np.select([d <= CLASS_EDGES[0], d <= CLASS_EDGES[1], d < CLASS_EDGES[2], d < CLASS_EDGES[3]],
                            [0, 1, 2, 3], 4)
    return out


def _first_crossing(paths, qref):
    """paths: n_q x K x 6 transferred proxies -> -1/0/+1 (NaN if any proxy missing)."""
    delta = paths - qref[:, None, None]
    crossed = np.abs(delta) >= 20
    first = np.where(crossed.any(2), crossed.argmax(2), -1)
    picked = np.take_along_axis(delta, np.maximum(first, 0)[..., None], 2)[..., 0]
    out = np.where(first >= 0, np.sign(picked), 0.0)
    out[~np.isfinite(delta).all(2)] = np.nan
    return out


def _prior(train, column, values, cutoff, complete_col, lookback_days=28):
    recent = train.loc[(train.index >= cutoff - lookback_days * 86400) & (train[complete_col] < cutoff)]
    labels = recent[column].to_numpy(float)
    counts = np.array([(labels == v).sum() for v in values], float)
    return counts / counts.sum() if counts.sum() else np.full(len(values), 1 / len(values))


def make_fit_predict(cfg: AnalogConfig, heads=("arrival", "class", "first", "ride")):
    """fit_predict(train, test, cutoff) for harness.walkforward.walk_forward."""

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cutoff: int) -> pd.DataFrame:
        columns = [c for c in train.columns if c.startswith("a_")]
        colw = _column_weights(columns, cfg)
        use = colw > 0
        columns = [c for c, u in zip(columns, use)]
        lib = train.loc[train.m_fresh_available & train.y_ref.notna() & train.y_arrival.notna()]
        X = lib[columns].to_numpy(float)
        center = np.nanmedian(X, 0)
        scale = np.nanstd(X, 0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        center = np.where(np.isfinite(center), center, 0.0)
        mult = np.sqrt(colw) / scale

        def standardise(frame):
            Z = (frame[columns].to_numpy(float) - center) * mult
            Z[~np.isfinite(Z)] = 0.0              # missing coordinate = library median (no pull either way)
            return Z[:, use]

        L = standardise(lib)
        q_ok = test.m_fresh_available.to_numpy(bool) & test.s_freshPm25.notna().to_numpy()
        query = test.loc[q_ok]
        out = pd.DataFrame(index=test.index)
        if not len(query) or len(lib) < cfg.k:
            return out
        Q = standardise(query)
        d2 = (Q ** 2).sum(1)[:, None] + (L ** 2).sum(1)[None, :] - 2 * Q @ L.T
        d2 = np.maximum(d2, 0)
        k = cfg.k
        nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
        nd2 = np.take_along_axis(d2, nn, 1)
        if np.isfinite(cfg.bandwidth):
            scale_k = nd2.max(1, keepdims=True) * cfg.bandwidth + 1e-12
            w = np.exp(-nd2 / scale_k)
        else:
            w = np.ones_like(nd2)

        qref = query.s_freshPm25.to_numpy(float)
        nref = lib.y_ref.to_numpy(float)[nn]
        res = {}

        def tr(column):
            return _transfer(lib[column].to_numpy(float)[nn], nref, qref, cfg.transfer)

        point = _weighted_quantile if cfg.point_stat == "median" else (lambda v, ww: _weighted_mean(v, ww))
        if "arrival" in heads or "class" in heads:
            arrival = tr("y_arrival")
            if "arrival" in heads:
                res["pred_arrival"] = point(arrival, w)
            if "class" in heads:
                prior = _prior(train, "y_class5", range(5), cutoff, "y_arrival_complete")
                cls = _class5(arrival - qref[:, None])
                valid = cls >= 0
                for c in range(5):
                    res[f"p_class5_{c}"] = _share(cls == c, valid, w, prior[c], cfg.alpha)
                res["p_fall40"], res["p_rise40"] = res["p_class5_0"], res["p_class5_4"]
                res["p_fall20"] = res["p_class5_0"] + res["p_class5_1"]
                res["p_rise20"] = res["p_class5_3"] + res["p_class5_4"]
        if "first" in heads:
            prior = _prior(train, "y_first20", (-1, 0, 1), cutoff, "y_first20_complete")
            paths = np.stack([tr(f"y_path_{lead}") for lead in PATH_LEADS], axis=2)
            first = _first_crossing(paths, qref)
            valid = np.isfinite(first)
            for name, value, p in (("p_first_drop", -1, prior[0]), ("p_first_none", 0, prior[1]),
                                   ("p_first_rise", 1, prior[2])):
                res[name] = _share(first == value, valid, w, p, cfg.alpha)
        if "ride" in heads:
            for head in ("mean", "min", "max"):
                res[f"pred_ride_{head}"] = point(tr(f"y_ride_{head}"), w)
            prior = _prior(train, "y_ride_le70", (0.0, 1.0), cutoff, "y_ride_complete")
            ride = tr("y_ride_mean")
            res["p_ride_le70"] = _share(ride <= config.PLANNING_CUTOFF, np.isfinite(ride), w, prior[1], cfg.alpha)
        res["analog_k_eff"] = w.sum(1) ** 2 / (w ** 2).sum(1)
        res["analog_d2_median"] = np.median(nd2, 1)
        for name, values in res.items():
            out.loc[query.index, name] = values
        return out

    return fit_predict
