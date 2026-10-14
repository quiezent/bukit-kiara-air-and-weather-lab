"""Chronological post-hoc calibration and stacking of saved base-model outputs.

For each evaluation day D the calibrator is fitted only on base-model outputs
that were themselves out-of-sample (walk-forward, refit before their own day)
for issues in [D - lookback, D - embargo) whose labels completed before D's
MYT midnight. It then maps day D's base outputs to probabilities.

Calibrators
* ``tempbias``: logits_k = sum_m a_m * log p_mk + b_k (one temperature per base
  model, one bias per class), multinomial log loss + small ridge on (a - 1/M)
  and b. Few parameters, so it is stable with rare classes.
* ``matrix``: multinomial logistic regression on all log-probabilities of all
  inputs (sklearn, L2 with strength 1/C). Lets one model's tails inform
  another target (e.g. arrival tails -> first crossing).
* ``isotonic``: per binary tail (fall20, rise20, fall40, rise40), isotonic
  regression on the tail probability. Diagnostic only: it does not keep the
  5 classes coherent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import common
from harness import config

EPS = 1e-4
HEADS = {
    "class5": (common.CLASS5, "y_class5", (0, 1, 2, 3, 4), "y_arrival_complete"),
    "first20": (common.FIRST3, "y_first20", (-1, 0, 1), "y_first20_complete"),
}


def _logp(frame, columns):
    return np.log(np.clip(frame[columns].to_numpy(float), EPS, 1))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_tempbias(logps, y_idx, n_classes, ridge=1.0, reachable=None):
    m = len(logps)
    penalty = 0.0 if reachable is None else np.where(reachable, 0.0, -1e9)
    n = len(y_idx)

    def unpack(theta):
        return theta[:m], theta[m:]

    def loss(theta):
        a, b = unpack(theta)
        z = sum(a[j] * logps[j] for j in range(m)) + b[None, :] + penalty
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        nll = np.mean(lse - z[np.arange(n), y_idx])
        p = np.exp(z - lse[:, None])
        onehot = np.zeros_like(p)
        onehot[np.arange(n), y_idx] = 1
        g_z = (p - onehot) / n
        g_a = np.array([np.sum(g_z * logps[j]) for j in range(m)]) + 2 * ridge / n * (a - 1 / m)
        g_b = g_z.sum(axis=0) + 2 * ridge / n * b
        reg = ridge / n * (np.sum((a - 1 / m) ** 2) + np.sum(b ** 2))
        return nll + reg, np.concatenate([g_a, g_b])

    theta0 = np.concatenate([np.full(m, 1 / m), np.zeros(n_classes)])
    result = minimize(loss, theta0, jac=True, method="L-BFGS-B")
    return unpack(result.x)


def apply_tempbias(params, logps, reachable):
    a, b = params
    z = sum(a[j] * logps[j] for j in range(len(logps))) + b[None, :]
    z = np.where(reachable, z, -np.inf)
    return _softmax(z)


def reachable_class5(ref):
    from base_models import r_edges
    out = np.ones((len(ref), 5), bool)
    out[:, :4] = np.isfinite(r_edges(ref))
    return out


def calibrate(ds: pd.DataFrame, inputs: dict, head: str, method="tempbias", lookback_days=21,
              embargo_minutes=common.EMBARGO_MINUTES, days=None, ridge=1.0, C=0.1,
              extra: dict | None = None, log=None) -> pd.DataFrame:
    """inputs: {name: base prediction frame}; their ``head`` columns are calibrated.

    ``extra`` (matrix only): {name: (frame, columns)} additional log-prob inputs,
    e.g. arrival tails for the first-crossing head.
    """
    columns, label, classes, complete_col = HEADS[head]
    frames = {k: v.reindex(ds.index) for k, v in inputs.items()}
    extra = {k: (v[0].reindex(ds.index), v[1]) for k, v in (extra or {}).items()}
    index = ds.index.to_numpy(np.int64)
    y = ds[label].to_numpy(float)
    complete = ds[complete_col].to_numpy(np.int64)
    ref = ds["s_freshPm25"].to_numpy(float)
    finite = np.ones(len(ds), bool)
    for f in frames.values():
        finite &= np.isfinite(f[columns].to_numpy(float)).all(axis=1)
    for f, cols in extra.values():
        finite &= np.isfinite(f[cols].to_numpy(float)).all(axis=1)
    labelled = np.isin(y, classes) & common.usable(ds)
    days = common.days_between(ds, common.TUNE_START, common.EVAL_END) if days is None else days
    out = pd.DataFrame(np.nan, index=ds.index, columns=columns)
    for cutoff in days:
        train = ((index >= cutoff - lookback_days * 86400) & (index < cutoff - embargo_minutes * 60)
                 & (complete < cutoff) & finite & labelled)
        test = (index >= cutoff) & (index < cutoff + 86400) & finite
        if not test.any() or train.sum() < 200:
            continue
        y_idx = np.searchsorted(np.asarray(classes), y[train].astype(int))
        reach_tr = reachable_class5(ref[train]) if head == "class5" else np.ones((train.sum(), len(classes)), bool)
        reach_te = reachable_class5(ref[test]) if head == "class5" else np.ones((test.sum(), len(classes)), bool)
        if method == "tempbias":
            lp_tr = [_logp(f.loc[train], columns) for f in frames.values()]
            lp_te = [_logp(f.loc[test], columns) for f in frames.values()]
            params = fit_tempbias(lp_tr, y_idx, len(classes), ridge=ridge, reachable=reach_tr)
            p = apply_tempbias(params, lp_te, reach_te)
        elif method == "matrix":
            def design(mask):
                parts = [_logp(f.loc[mask], columns) for f in frames.values()]
                parts += [_logp(f.loc[mask], cols) for f, cols in extra.values()]
                return np.hstack(parts)
            x_tr, x_te = design(train), design(test)
            mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0) + 1e-6
            model = LogisticRegression(C=C, max_iter=2000)
            model.fit((x_tr - mu) / sd, y_idx)
            raw = model.predict_proba((x_te - mu) / sd)
            p = np.full((test.sum(), len(classes)), 0.0)
            for j, c in enumerate(model.classes_):
                p[:, c] = raw[:, j]
            p = np.where(reach_te, np.maximum(p, 1e-6), 0)
            p = p / p.sum(axis=1, keepdims=True)
        else:
            raise ValueError(method)
        out.loc[test, columns] = p
        if log:
            log(f"{head} {method} {config.myt(cutoff):%Y-%m-%d} train={int(train.sum())}")
    return common.with_tails(out)


def isotonic_tails(ds: pd.DataFrame, base: pd.DataFrame, lookback_days=21,
                   embargo_minutes=common.EMBARGO_MINUTES, days=None) -> pd.DataFrame:
    base = base.reindex(ds.index)
    index = ds.index.to_numpy(np.int64)
    complete = ds["y_arrival_complete"].to_numpy(np.int64)
    days = common.days_between(ds, common.TUNE_START, common.EVAL_END) if days is None else days
    out = pd.DataFrame(np.nan, index=ds.index, columns=["p_fall20", "p_rise20", "p_fall40", "p_rise40"])
    for tail in out.columns:
        p = base[tail].to_numpy(float)
        y = ds["y_" + tail[2:]].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(y) & common.usable(ds)
        for cutoff in days:
            train = ((index >= cutoff - lookback_days * 86400) & (index < cutoff - embargo_minutes * 60)
                     & (complete < cutoff) & ok)
            test = (index >= cutoff) & (index < cutoff + 86400) & np.isfinite(p)
            if not test.any() or train.sum() < 200:
                continue
            iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p[train], y[train])
            out.loc[test, tail] = iso.predict(p[test])
    return out
