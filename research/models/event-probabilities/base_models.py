"""Base model families for the +90 arrival-change classes and the first crossing.

Each ``make_*`` returns a ``fit_predict(train, test, cutoff)`` for
``harness.walkforward.walk_forward``. Outputs are the 5 arrival-change class
probabilities (``p_class5_0..4``, coherent tails added) and, where the family
supports it, the 3 first-crossing probabilities (``p_first_drop/none/rise``).

Families
* ``logratio_empirical``: unconditional empirical distribution of the log ratio
  r = log((arrival + 10) / (ref + 10)) over the training window, mapped through
  each issue's reference. It only encodes "changes scale with the level".
* ``logratio_hgb``: a numeric predictive distribution. HGB for the location of r,
  HGB for its scale (fitted on out-of-fold absolute residuals, folds by day),
  and the empirical CDF of the standardised out-of-fold residual. Class
  probabilities = CDF mass between the class edges mapped into r space.
* ``hgb_class``: direct HGB classifiers (5-class arrival change and 3-class first
  crossing) with optional class weights w_c = (n / n_c) ** alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import common

OFFSET = 10.0


def r_edges(ref):
    """Class5 edges (-40, -20, 20, 40 on delta) in log-ratio space; -inf where unreachable."""
    ref = np.asarray(ref, float)[:, None]
    edges = np.asarray(common.CHANGE_EDGES)[None, :]
    level = ref + edges + OFFSET
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(level > 0, np.log(level / (ref + OFFSET)), -np.inf)
    return out


def probs_from_cdf(cdf_values: np.ndarray, edges: np.ndarray, eps=1e-3) -> np.ndarray:
    """cdf at the 4 edges -> 5 class masses, floored at eps on reachable classes.

    A class is unreachable only when its upper edge is -inf (e.g. a fall >= 40
    from a reference below 30 cannot happen); it keeps exactly 0.
    """
    c = np.clip(cdf_values, 0, 1)
    c = np.maximum.accumulate(c, axis=1)
    p = np.diff(np.concatenate([np.zeros((len(c), 1)), c, np.ones((len(c), 1))], axis=1), axis=1)
    p = np.clip(p, 0, 1)
    reachable = np.ones_like(p, bool)
    reachable[:, :4] = np.isfinite(edges)        # class k < 4 needs its upper edge reachable
    p = np.where(reachable, np.maximum(p, eps), 0.0)
    return p / p.sum(axis=1, keepdims=True)


TRAIN_STRIDE_SECONDS = 600     # train on every second 5-min issue: neighbours share ~all of the 90-min path


def train_rows(train: pd.DataFrame, head: str, stride=TRAIN_STRIDE_SECONDS) -> pd.DataFrame:
    ok = common.usable(train) & (train.index.to_numpy(np.int64) % stride == 0)
    if head == "class5":
        ok &= train["y_class5"].to_numpy(float) >= 0
    elif head == "first20":
        ok &= np.isin(train["y_first20"].to_numpy(float), (-1, 0, 1))
    return train.loc[ok]


def floor_probs(p, reachable, eps=1e-4):
    """Keep every reachable class above eps (a class absent from a training window
    would otherwise get exactly 0); unreachable classes stay 0."""
    p = np.where(reachable, np.maximum(p, eps), 0.0)
    return p / p.sum(axis=1, keepdims=True)


def _empty(test, columns):
    return pd.DataFrame(np.nan, index=test.index, columns=columns)


# ----------------------------------------------------------------------------- log-ratio empirical
def make_logratio_empirical():
    def fit_predict(train, test, cutoff):
        tr = train_rows(train, "class5")
        r = np.log((tr.y_arrival.to_numpy(float) + OFFSET) / (tr.s_freshPm25.to_numpy(float) + OFFSET))
        r = np.sort(r[np.isfinite(r)])
        out = _empty(test, common.CLASS5)
        ok = common.usable(test)
        edges = r_edges(test.loc[ok, "s_freshPm25"].to_numpy(float))
        # P(r <= e) with the class boundary conventions (<= -40 / <= -20 / < 20 / < 40)
        cdf = np.column_stack([
            np.searchsorted(r, edges[:, 0], side="right"), np.searchsorted(r, edges[:, 1], side="right"),
            np.searchsorted(r, edges[:, 2], side="left"), np.searchsorted(r, edges[:, 3], side="left")]) / len(r)
        out.loc[ok, common.CLASS5] = probs_from_cdf(cdf, edges)
        return common.with_tails(out)
    return fit_predict


# ----------------------------------------------------------------------------- log-ratio HGB distribution
def _hgb_reg(seed=0, **kw):
    params = dict(max_iter=100, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=60,
                  l2_regularization=1.0, early_stopping=False, random_state=seed)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def make_logratio_hgb(features, folds=3, loc_kw=None, scale_kw=None, shrink_loc=1.0):
    loc_kw, scale_kw = loc_kw or {}, scale_kw or {}

    def fit_predict(train, test, cutoff):
        tr = train_rows(train, "class5")
        r = np.log((tr.y_arrival.to_numpy(float) + OFFSET) / (tr.s_freshPm25.to_numpy(float) + OFFSET))
        keep = np.isfinite(r)
        tr, r = tr.loc[keep], r[keep]
        x = tr[features].to_numpy(float)
        days = np.array([common.config.midnight(i) for i in tr.index.to_numpy(np.int64)])
        unique_days = np.unique(days)
        fold_of_day = {d: k % folds for k, d in enumerate(unique_days)}
        fold = np.array([fold_of_day[d] for d in days])
        # Out-of-fold location, then out-of-fold scale on |residual|.
        mu_oof = np.zeros(len(r))
        for k in range(folds):
            m = fold == k
            mu_oof[m] = _hgb_reg(**loc_kw).fit(x[~m], r[~m]).predict(x[m]) * shrink_loc
        abs_res = np.abs(r - mu_oof)
        s_oof = np.zeros(len(r))
        for k in range(folds):
            m = fold == k
            s_oof[m] = _hgb_reg(**scale_kw).fit(x[~m], abs_res[~m]).predict(x[m])
        floor = np.quantile(abs_res, 0.05) + 1e-3
        s_oof = np.maximum(s_oof, floor)
        z = np.sort((r - mu_oof) / s_oof)
        loc = _hgb_reg(**loc_kw).fit(x, r)
        scale = _hgb_reg(**scale_kw).fit(x, abs_res)
        out = _empty(test, common.CLASS5 + ["pred_r_loc", "pred_r_scale"])
        ok = common.usable(test)
        if ok.any():
            xt = test.loc[ok, features].to_numpy(float)
            mu = loc.predict(xt) * shrink_loc
            s = np.maximum(scale.predict(xt), floor)
            edges = r_edges(test.loc[ok, "s_freshPm25"].to_numpy(float))
            ze = (edges - mu[:, None]) / s[:, None]
            cdf = np.column_stack([np.searchsorted(z, ze[:, j], side=("right" if j < 2 else "left"))
                                   for j in range(4)]) / len(z)
            out.loc[ok, common.CLASS5] = probs_from_cdf(cdf, edges)
            out.loc[ok, "pred_r_loc"] = mu
            out.loc[ok, "pred_r_scale"] = s
        return common.with_tails(out)
    return fit_predict


# ----------------------------------------------------------------------------- direct classifiers
def _hgb_clf(seed=0, **kw):
    params = dict(max_iter=100, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=60,
                  l2_regularization=1.0, early_stopping=False, random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def _class_weights(y, classes, alpha):
    if alpha == 0:
        return np.ones(len(y))
    counts = np.array([(y == c).sum() for c in classes], float)
    w_c = np.where(counts > 0, (len(y) / np.maximum(counts, 1)) ** alpha, 0)
    w = w_c[np.searchsorted(classes, y)]
    return w * len(y) / w.sum()


def _fit_proba(x, y, classes, xt, alpha, kw):
    present = np.unique(y)
    if len(present) < 2:
        p = np.zeros((len(xt), len(classes)))
        p[:, list(classes).index(present[0])] = 1.0
        return p
    model = _hgb_clf(**kw).fit(x, y, sample_weight=_class_weights(y, np.asarray(classes), alpha))
    p = np.zeros((len(xt), len(classes)))
    raw = model.predict_proba(xt)
    for j, c in enumerate(model.classes_):
        p[:, list(classes).index(c)] = raw[:, j]
    return p


def make_hgb_class(features, alpha=0.0, heads=("class5", "first20"), kw=None):
    kw = kw or {}

    def fit_predict(train, test, cutoff):
        out = _empty(test, (common.CLASS5 if "class5" in heads else []) + (common.FIRST3 if "first20" in heads else []))
        ok = common.usable(test)
        if not ok.any():
            return out
        xt = test.loc[ok, features].to_numpy(float)
        if "class5" in heads:
            tr = train_rows(train, "class5")
            p = _fit_proba(tr[features].to_numpy(float), tr.y_class5.to_numpy(int), (0, 1, 2, 3, 4), xt, alpha, kw)
            reachable = np.ones_like(p, bool)
            reachable[:, :4] = np.isfinite(r_edges(test.loc[ok, "s_freshPm25"].to_numpy(float)))
            out.loc[ok, common.CLASS5] = floor_probs(p, reachable)
        if "first20" in heads:
            tr = train_rows(train, "first20")
            p = _fit_proba(tr[features].to_numpy(float), tr.y_first20.to_numpy(int), (-1, 0, 1), xt, alpha, kw)
            out.loc[ok, common.FIRST3] = floor_probs(p, np.ones_like(p, bool))
        return common.with_tails(out)
    return fit_predict
