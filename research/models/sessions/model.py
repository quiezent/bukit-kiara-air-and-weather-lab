"""Session candidates: anchored gradient boosting (the hypothesis) and a log-linear ablation.

A candidate predicts the next feasible Morning (09-11) or Afternoon (14-16)
session mean for every issue. One pooled model covers both sessions and every
lead (90 min .. ~23 h), so tomorrow's Afternoon at 18:05 (lead 1195 min) is
inside the training lead domain, unlike the PatchTST head.

HGB target: residual of the session mean against an anchor, in linear or log
space; loss absolute_error (MAE is the scoring metric). The anchor is a fixed
input transformation, not a selection rule: the model output is published as is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features

OFFSET = 5.0          # log(y + 5): keeps clean-air sessions (~20) from dominating log errors
HGB_PARAMS = dict(loss="absolute_error", max_iter=150, learning_rate=0.05, max_depth=3, max_leaf_nodes=8,
                  min_samples_leaf=150, l2_regularization=1.0, early_stopping=False, random_state=1749)
ANCHORS = {"ref": "x_freshPm25", "sess7": "x_sess_mean7", "diurnal": "x_anchor_diurnal"}


def _anchor(frame, name):
    return frame[ANCHORS[name]].to_numpy(float)


def _to_target(y, anchor, space):
    return np.log((y + OFFSET) / (anchor + OFFSET)) if space == "log" else y - anchor


def _from_target(z, anchor, space):
    out = (anchor + OFFSET) * np.exp(z) - OFFSET if space == "log" else anchor + z
    return np.maximum(out, 0.0)


def hgb(anchor="ref", space="log", groups=("base",), params=None):
    cols = features.columns(groups)
    params = {**HGB_PARAMS, **(params or {})}

    def fit_predict(train, test):
        a_train = _anchor(train, anchor)
        ok = np.isfinite(train.y.to_numpy()) & np.isfinite(a_train) & (a_train >= 0)
        if ok.sum() < 200:
            return pd.Series(np.nan, index=test.index)
        z = _to_target(train.y.to_numpy()[ok], a_train[ok], space)
        model = HistGradientBoostingRegressor(**params).fit(train[cols].to_numpy(float)[ok], z)
        a_test = _anchor(test, anchor)
        pred = _from_target(model.predict(test[cols].to_numpy(float)), a_test, space)
        pred[~np.isfinite(a_test) | (a_test < 0)] = np.nan          # no anchor -> unavailable
        return pd.Series(pred, index=test.index)

    fit_predict.label = f"hgb anchor={anchor} space={space} groups={'+'.join(groups)}"
    fit_predict.columns = cols
    return fit_predict


LINEAR_LEVELS = ("x_freshPm25", "x_ref_mean_24h", "x_sess_last", "x_sess_mean7", "x_ref_sameclock_7d", "x_c_sess_pm25")


def linear(alpha=1.0):
    """Log-space ridge blend of levels, each also scaled by lead/24 h and per session."""

    def design(frame):
        lead = frame.x_lead_min.to_numpy(float) / 1440
        sess = frame.x_session.to_numpy(float)
        blocks = []
        for column in LINEAR_LEVELS:
            v = np.log(np.clip(frame[column].to_numpy(float), 0, None) + OFFSET)
            blocks += [v, v * lead, v * sess]
        blocks += [sess, lead, frame.x_c_log_ratio.to_numpy(float), frame.x_w_sess_precip_sum.to_numpy(float),
                   np.log(frame.x_w_sess_blh.to_numpy(float).clip(50, None))]
        return np.column_stack(blocks)

    def fit_predict(train, test):
        x = design(train)
        y = np.log(train.y.to_numpy(float) + OFFSET)
        ok = np.isfinite(x).all(1) & np.isfinite(y)
        if ok.sum() < 200:
            return pd.Series(np.nan, index=test.index)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[ok], y[ok])
        xt = design(test)
        good = np.isfinite(xt).all(1)
        pred = np.full(len(test), np.nan)
        pred[good] = np.maximum(np.exp(model.predict(xt[good])) - OFFSET, 0)
        return pd.Series(pred, index=test.index)

    fit_predict.label = f"log-linear ridge alpha={alpha}"
    return fit_predict


def naive(column):
    def fit_predict(train, test):
        return pd.Series(test[column].to_numpy(float), index=test.index)

    fit_predict.label = f"naive {column}"
    return fit_predict


def hgb_windows(anchor="ref", space="log", groups=("base", "weather"), hours=None, params=None):
    """The window-augmented candidate: train on generic 2-h windows (windows.py), predict the sessions.

    ``hours`` = (first, last) window start hour kept in training (None = all 24)."""
    from . import windows
    cols = windows.columns(groups)
    params = {**HGB_PARAMS, **(params or {})}

    def fit_predict(train, test):
        if hours is not None:
            keep = train.x_start_hour.between(*hours).to_numpy()
            train = train.loc[keep]
        a_train = _anchor(train, anchor)
        ok = np.isfinite(train.y.to_numpy()) & np.isfinite(a_train) & (a_train >= 0)
        if ok.sum() < 200:
            return pd.Series(np.nan, index=test.index)
        z = _to_target(train.y.to_numpy()[ok], a_train[ok], space)
        model = HistGradientBoostingRegressor(**params).fit(train[cols].to_numpy(float)[ok], z)
        a_test = _anchor(test, anchor)
        pred = _from_target(model.predict(test[cols].to_numpy(float)), a_test, space)
        pred[~np.isfinite(a_test) | (a_test < 0)] = np.nan
        out = pd.Series(pred, index=test.index)
        out.attrs["model"] = model
        return out

    fit_predict.label = f"window-hgb anchor={anchor} space={space} groups={'+'.join(groups)} hours={hours}"
    fit_predict.columns = cols
    fit_predict.windows = True
    return fit_predict
