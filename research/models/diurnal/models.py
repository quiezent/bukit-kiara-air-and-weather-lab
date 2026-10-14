"""Two diurnal candidates, each a family of walk-forward ``fit_predict`` functions.

* ``KernelClimatology``: conditional climatology of the change from the current
  reference. Training issues are weighted by a Gaussian kernel in clock of day
  (circular, hours), log level and 30-min trend; point heads are the weighted
  median change, probabilities the weighted class frequencies (smoothed toward
  the unweighted training frequencies with ``alpha`` pseudo-rows).
* ``DiurnalHGB``: HistGradientBoosting on the change (regression heads) or the
  class (probability heads) with strong time-of-day features, compact sensor
  level/trend, diurnal analog/profile features and optionally the Open-Meteo
  mixing-layer/rain drivers.

Every head is a literal model output: ref + predicted change, or the class
probabilities. Nothing is gated, selected or replaced by persistence.

Head groups (one walk-forward per group, each with its own label-completion
column and embargo): near = +90 arrival and ride mean/min/max; events = the
5-class arrival change and the first crossing; le70 = P(ride mean <= 70);
morning / afternoon = next feasible session mean.
"""
from __future__ import annotations

from harness import config  # noqa: F401  (sets the BLAS/OpenMP thread limits before sklearn loads)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import features as F

NEAR = {"arrival": "y_arrival", "ride_mean": "y_ride_mean", "ride_min": "y_ride_min", "ride_max": "y_ride_max"}
CLASS5 = (0, 1, 2, 3, 4)
FIRST = (-1, 0, 1)
EPS = 1e-3          # floor mixing toward the training class frequencies, so no class gets probability 0
GROUPS = {          # walk-forward settings per head group
    "near": dict(targets=("arrival", "ride_mean"), embargo_minutes=330),
    "events": dict(targets=("class5", "first20"), embargo_minutes=120),
    "le70": dict(targets=("ride_le70",), embargo_minutes=330),
    "morning": dict(targets=("morning",), embargo_minutes=330),
    "afternoon": dict(targets=("afternoon",), embargo_minutes=330),
}


def _class5_columns(p: np.ndarray, index) -> pd.DataFrame:
    out = pd.DataFrame(p, index=index, columns=[f"p_class5_{k}" for k in CLASS5])
    out["p_fall20"] = out.p_class5_0 + out.p_class5_1
    out["p_rise20"] = out.p_class5_3 + out.p_class5_4
    out["p_fall40"] = out.p_class5_0
    out["p_rise40"] = out.p_class5_4
    return out


def _frequencies(labels, classes) -> np.ndarray:
    counts = np.array([(labels == c).sum() for c in classes], float) + 0.5
    return counts / counts.sum()


def _rows(train: pd.DataFrame, target: str) -> pd.Series:
    ok = train[target].notna() & train.s_freshPm25.notna()
    if target.startswith("y_ride_"):
        ok &= train.y_ride_joint_valid.astype(bool)
    if target == "y_class5":
        ok &= train.y_class5 >= 0
    if target == "y_first20":
        ok &= train.y_first20.isin(FIRST)
    return ok


# ============================================================================ kernel climatology
class KernelClimatology:
    """Nadaraya-Watson conditional climatology of the change from the reference."""

    def __init__(self, bw_hour=1.0, bw_level=0.25, bw_trend=10.0, alpha=5.0, bw_hour_session=0.5):
        self.bw_hour, self.bw_level, self.bw_trend = bw_hour, bw_level, bw_trend
        self.alpha, self.bw_hour_session = alpha, bw_hour_session

    def name(self):
        return f"kernel_h{self.bw_hour:g}_l{self.bw_level:g}_t{self.bw_trend:g}_a{self.alpha:g}"

    def _weights(self, train, test, bw_hour):
        h_tr, h_te = train.t_hour.to_numpy(float), test.t_hour.to_numpy(float)
        dh = np.abs(h_te[:, None] - h_tr[None, :])
        dh = np.minimum(dh, 24 - dh)
        logw = -0.5 * (dh / bw_hour) ** 2
        if np.isfinite(self.bw_level):
            l_tr = np.log(np.maximum(train.s_freshPm25.to_numpy(float), 1))
            l_te = np.log(np.maximum(test.s_freshPm25.to_numpy(float), 1))
            logw += -0.5 * ((l_te[:, None] - l_tr[None, :]) / self.bw_level) ** 2
        if np.isfinite(self.bw_trend):
            t_tr = np.nan_to_num(train.s_rawDelta30.to_numpy(float))
            t_te = np.nan_to_num(test.s_rawDelta30.to_numpy(float))
            logw += -0.5 * ((t_te[:, None] - t_tr[None, :]) / self.bw_trend) ** 2
        logw -= logw.max(axis=1, keepdims=True)          # far-out queries fall back to their nearest rows
        return np.exp(logw)

    @staticmethod
    def _weighted_quantile(w, values, q=0.5):
        order = np.argsort(values)
        cw = np.cumsum(w[:, order], axis=1)
        idx = np.argmax(cw >= q * cw[:, -1:], axis=1)
        return values[order][idx]

    def _probs(self, w, labels, classes):
        prior = _frequencies(labels, classes)
        onehot = (labels[:, None] == np.asarray(classes)[None, :]).astype(float)
        p = (w @ onehot + self.alpha * prior) / (w.sum(axis=1, keepdims=True) + self.alpha)
        return (1 - EPS) * p + EPS * prior

    def fit_predict(self, group):
        def run(train, test, cutoff):
            ref = test.s_freshPm25.to_numpy(float)
            out = pd.DataFrame(index=test.index)
            if group == "near":
                for head, target in NEAR.items():
                    tr = train.loc[_rows(train, target)]
                    w = self._weights(tr, test, self.bw_hour)
                    delta = (tr[target] - tr.s_freshPm25).to_numpy(float)
                    out[f"pred_{head}"] = ref + self._weighted_quantile(w, delta)
            elif group == "events":
                tr = train.loc[_rows(train, "y_class5")]
                p = self._probs(self._weights(tr, test, self.bw_hour), tr.y_class5.to_numpy(float), CLASS5)
                out = out.join(_class5_columns(p, test.index))
                tr = train.loc[_rows(train, "y_first20")]
                p = self._probs(self._weights(tr, test, self.bw_hour), tr.y_first20.to_numpy(float), FIRST)
                out["p_first_drop"], out["p_first_none"], out["p_first_rise"] = p[:, 0], p[:, 1], p[:, 2]
            elif group == "le70":
                tr = train.loc[_rows(train, "y_ride_mean")]
                w = self._weights(tr, test, self.bw_hour)
                delta = (tr.y_ride_mean - tr.s_freshPm25).to_numpy(float)
                below = ((ref[:, None] + delta[None, :]) <= config.PLANNING_CUTOFF).astype(float)
                base = below.mean(axis=1)
                out["p_ride_le70"] = ((w * below).sum(axis=1) + self.alpha * base) / (w.sum(axis=1) + self.alpha)
            else:                                            # morning / afternoon
                target = f"y_{group}_mean"
                tr = train.loc[_rows(train, target)]
                w = self._weights(tr, test, self.bw_hour_session)
                delta = (tr[target] - tr.s_freshPm25).to_numpy(float)
                out[f"pred_{group}"] = ref + self._weighted_quantile(w, delta)
            out.loc[~np.isfinite(ref)] = np.nan
            return out
        return run


# ============================================================================ HGB
class DiurnalHGB:
    def __init__(self, feature_set="diurnal", min_samples_leaf=100, loss="absolute_error", max_iter=100,
                 learning_rate=0.1, max_leaf_nodes=15, l2=1.0, seed=20261013):
        self.feature_set, self.loss = feature_set, loss
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=min_samples_leaf, l2_regularization=l2, early_stopping=False,
                           random_state=seed)
        self.fit_seconds = []

    def name(self):
        return f"hgb_{self.feature_set}_leaf{self.params['min_samples_leaf']}_{self.loss}"

    def _regress(self, train, test, target, cols):
        tr = train.loc[_rows(train, target)]
        model = HistGradientBoostingRegressor(loss=self.loss, **self.params)
        model.fit(tr[cols].to_numpy(float), (tr[target] - tr.s_freshPm25).to_numpy(float))
        return test.s_freshPm25.to_numpy(float) + model.predict(test[cols].to_numpy(float))

    def _classify(self, train, test, target, cols, classes):
        tr = train.loc[_rows(train, target)]
        labels = tr[target].to_numpy(float)
        model = HistGradientBoostingClassifier(**self.params)
        model.fit(tr[cols].to_numpy(float), labels)
        raw = model.predict_proba(test[cols].to_numpy(float))
        p = np.zeros((len(test), len(classes)))
        for j, c in enumerate(model.classes_):
            p[:, list(classes).index(c)] = raw[:, j]
        return (1 - EPS) * p + EPS * _frequencies(labels, classes)

    def fit_predict(self, group):
        def run(train, test, cutoff):
            import time
            started = time.perf_counter()
            out = pd.DataFrame(index=test.index)
            session = group if group in ("morning", "afternoon") else None
            cols = F.feature_set(self.feature_set, session)
            if group == "near":
                for head, target in NEAR.items():
                    out[f"pred_{head}"] = self._regress(train, test, target, cols)
            elif group == "events":
                out = out.join(_class5_columns(self._classify(train, test, "y_class5", cols, CLASS5), test.index))
                p = self._classify(train, test, "y_first20", cols, FIRST)
                out["p_first_drop"], out["p_first_none"], out["p_first_rise"] = p[:, 0], p[:, 1], p[:, 2]
            elif group == "le70":
                out["p_ride_le70"] = self._classify(train, test, "y_ride_le70", cols, (0.0, 1.0))[:, 1]
            else:
                out[f"pred_{group}"] = self._regress(train, test, f"y_{group}_mean", cols)
            out.loc[test.s_freshPm25.isna().to_numpy()] = np.nan
            self.fit_seconds.append((group, time.perf_counter() - started))
            return out
        return run
