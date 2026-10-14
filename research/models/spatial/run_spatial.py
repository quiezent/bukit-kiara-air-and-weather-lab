"""Walk-forward runs for the spatial hypothesis (neighbours, upwind weighting, regional CAMS).

    python -B -m spatial.run_spatial <variant> [<variant> ...] [--lag 300]

Each variant is fit daily with the shared harness walk_forward (refit at MYT midnight,
label-complete rows only) and writes data/models/spatial/pred_<variant>_lag<lag>.parquet
with the scorer's standard columns. Pairs differ only by the spatial feature block, so
every comparison is an ablation of that block.

Variants
  lr28 / lr28_sp      production logistic pipeline (median impute + indicators, scaler,
                      multinomial C=0.1) on the 28 incumbent inputs, without / with the
                      12 robust spatial inputs, clipped at the fit's own 1st/99th percentiles.
                      Embargo 120 min (the logistic's).
  hgbc / hgbc_sp      5-class HGB classifier on base (28 sensor + mean-reversion controls +
                      time + compact weather + CAMS) without / with all spatial features.
  hgbr / hgbr_sp      HGB absolute-error regressors for +90 arrival and ride mean/min/max
                      deltas, plus an HGB P(ride mean <= 70). Embargo 330 min (Ridge v4's).
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")     # before numpy/sklearn load (harness.config does the same)

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness import config, dataset, walkforward

OUT = config.OUT / "spatial"

LOG28 = ['s_freshPm25', 's_rawLag5', 's_rawLag10', 's_rawLag15', 's_rawLag30', 's_rawLag60', 's_rawDelta5',
         's_rawDelta10', 's_rawDelta15', 's_rawDelta30', 's_rawDelta60', 's_rawSd15', 's_rawSd30', 's_rawSd60',
         's_rawRange15', 's_rawRange30', 's_rawSlope15PerMinute', 's_rawSlope30PerMinute', 's_freshTemperature',
         's_freshHumidity', 's_rawTemperatureDelta30', 's_rawHumidityDelta30', 's_freshSampleCount',
         's_rawSampleCount30', 's_timeSin', 's_timeCos', 's_issueOffsetSin', 's_issueOffsetCos']
CONTROLS = ["x_dev90", "x_dev180", "x_mean180", "x_slope180"]       # TTDI-only mean reversion
TIME = ["t_hour", "t_day_sin", "t_day_cos", "t_dow", "t_weekend"]
WEATHER = ["w_next3h_precip_sum", "w_prior3h_precip_sum", "w_next6h_pprob_max", "w_first_pprob50_lead_min",
           "w_pprob_0", "w_pprob_90", "w_precip_90", "w_wspd10_0", "w_wspd10_90", "w_wind10_u_90",
           "w_wind10_v_90", "w_wind925_u_90", "w_wind925_v_90", "w_blh_0", "w_blh_90", "w_rh_0", "w_rh_90",
           "w_temp_0", "w_temp_90", "w_cape_90", "w_cloud_90", "w_rev60_pprob_90", "w_ride_precip_sum",
           "w_ride_blh_mean"]
CAMS = ["c_pm25_0", "c_pm25_90", "c_pm25_delta90", "c_pm25_ride_mean"]
SPATIAL_CORE = ["sp_net_level_minus_ref", "sp_net_diff", "sp_net_rel60", "sp_net_chg60", "sp_net_chg30",
                "sp_net_diff_mad", "sp_w10_up_diff", "sp_w10_adv90", "sp_w10_up_rel60", "sp_w925_up_diff",
                "sp_w925_adv90", "sp_w925_up_rel60"]   # robust (median) aggregates only
SPATIAL_EXTRA_PREFIXES = ("sp_",)

THIN = 900   # HGB trains on 15-min origins (5-min neighbours are near duplicates; laptop is shared); LR uses all 5-min origins
HGB_C = dict(max_iter=120, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=60,
             l2_regularization=1.0, early_stopping=False, random_state=0)
HGB_R = dict(loss="absolute_error", max_iter=120, learning_rate=0.1, max_leaf_nodes=15,
             min_samples_leaf=60, l2_regularization=1.0, early_stopping=False, random_state=0)
HGB_B = dict(max_iter=120, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=60,
             l2_regularization=1.0, early_stopping=False, random_state=0)


def load(lag: int) -> pd.DataFrame:
    ds = dataset.load()
    sp = pd.read_parquet(OUT / f"spatial_features_lag{lag}.parquet")
    ds = ds.join(sp)
    seq = [c for c in ds.columns if c.startswith("s_seq_pm25_")]
    ages = np.array([int(c.split("_")[3]) for c in seq])
    bins = ds[seq].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean90 = np.nanmean(bins[:, ages < 90], axis=1)
        mean180 = np.nanmean(bins, axis=1)
        old = np.nanmean(bins[:, ages >= 150], axis=1)
        new = np.nanmean(bins[:, ages < 30], axis=1)
    ds["x_dev90"] = ds.s_freshPm25 - mean90
    ds["x_dev180"] = ds.s_freshPm25 - mean180
    ds["x_mean180"] = mean180
    ds["x_slope180"] = new - old
    return ds


def feature_sets(ds):
    spatial_all = [c for c in ds.columns if c.startswith(SPATIAL_EXTRA_PREFIXES)]
    base = LOG28 + CONTROLS + TIME + WEATHER + CAMS
    return {"lr28": LOG28, "lr28_sp": LOG28 + SPATIAL_CORE,
            "hgb": base, "hgb_sp": base + spatial_all}


def _usable(frame):
    return frame.m_fresh_available.to_numpy(bool) & (frame.s_freshPm25.to_numpy(float) >= 0)


def _class_frame(proba, classes, index):
    full = np.zeros((len(index), 5))
    full[:, list(classes)] = proba
    out = pd.DataFrame({f"p_class5_{k}": full[:, k] for k in range(5)}, index=index)
    out["p_fall20"] = out.p_class5_0 + out.p_class5_1
    out["p_rise20"] = out.p_class5_3 + out.p_class5_4
    out["p_fall40"] = out.p_class5_0
    out["p_rise40"] = out.p_class5_4
    return out


def classifier_fp(cols, kind, params=None):
    def fit_predict(train, test, cutoff):
        ok = _usable(train) & train.y_class5.isin(range(5)).to_numpy()
        if kind == "hgb":
            ok &= (train.index.to_numpy(np.int64) % THIN) == 0
        tr = train.loc[ok]
        y = tr.y_class5.to_numpy(int)
        if len(tr) < 300 or len(set(y)) < 2:
            return None
        xtr, xte = tr[cols].copy(), test[cols].copy()
        if kind == "lr":
            # Linear model: clip spatial inputs at the training 1st/99th percentiles (fit-local).
            sp = [c for c in cols if c.startswith("sp_")]
            if sp:
                lo, hi = xtr[sp].quantile(0.01), xtr[sp].quantile(0.99)
                xtr[sp], xte[sp] = xtr[sp].clip(lo, hi, axis=1), xte[sp].clip(lo, hi, axis=1)
        if kind == "lr":
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                  StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=500, solver="lbfgs", tol=1e-4))
        else:
            model = HistGradientBoostingClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(xtr.to_numpy(float), y)
        use = _usable(test)
        out = pd.DataFrame(index=test.index, columns=[f"p_class5_{k}" for k in range(5)], dtype=float)
        if use.any():
            classes = model.classes_ if kind == "hgb" else model[-1].classes_
            probs = _class_frame(model.predict_proba(xte.loc[use].to_numpy(float)), classes, test.index[use])
            return probs.reindex(test.index)
        return out
    return fit_predict


def regressor_fp(cols):
    heads = {"arrival": "y_arrival", "ride_mean": "y_ride_mean", "ride_min": "y_ride_min", "ride_max": "y_ride_max"}

    def fit_predict(train, test, cutoff):
        base_ok = _usable(train) & ((train.index.to_numpy(np.int64) % THIN) == 0)
        use = _usable(test)
        out = pd.DataFrame(index=test.index)
        xt = test.loc[use, cols].to_numpy(float)
        ref_t = test.loc[use, "s_freshPm25"].to_numpy(float)
        for head, target in heads.items():
            ok = base_ok & np.isfinite(train[target].to_numpy(float))
            tr = train.loc[ok]
            model = HistGradientBoostingRegressor(**HGB_R)
            model.fit(tr[cols].to_numpy(float), (tr[target] - tr.s_freshPm25).to_numpy(float))
            pred = np.full(len(test), np.nan)
            pred[use] = ref_t + model.predict(xt)
            out[f"pred_{head}"] = pred
        ok = base_ok & np.isfinite(train.y_ride_mean.to_numpy(float))
        tr = train.loc[ok]
        y = (tr.y_ride_mean <= config.PLANNING_CUTOFF).astype(int).to_numpy()
        p = np.full(len(test), np.nan)
        if len(set(y)) == 2:
            model = HistGradientBoostingClassifier(**HGB_B).fit(tr[cols].to_numpy(float), y)
            p[use] = model.predict_proba(xt)[:, 1]
        else:
            p[use] = float(y[0])
        out["p_ride_le70"] = p
        return out
    return fit_predict


def _log(text):
    print("  " + text, flush=True)


def run(variant: str, ds: pd.DataFrame, lag: int):
    sets = feature_sets(ds)
    started = time.time()
    if variant in ("lr28", "lr28_sp"):
        fp = classifier_fp(sets[variant], "lr")
        pred = walkforward.walk_forward(ds, fp, targets=("class5",), embargo_minutes=120, log=_log)
    elif variant in ("hgbc", "hgbc_sp"):
        cols = sets["hgb_sp" if variant.endswith("_sp") else "hgb"]
        fp = classifier_fp(cols, "hgb", HGB_C)
        pred = walkforward.walk_forward(ds, fp, targets=("class5",), embargo_minutes=120, log=_log)
    elif variant in ("hgbr", "hgbr_sp"):
        cols = sets["hgb_sp" if variant.endswith("_sp") else "hgb"]
        fp = regressor_fp(cols)
        pred = walkforward.walk_forward(ds, fp, targets=("arrival", "ride_mean", "ride_le70"),
                                        embargo_minutes=330, log=_log)
    else:
        raise SystemExit(f"unknown variant {variant}")
    path = OUT / f"pred_{variant}_lag{lag}.parquet"
    pred.to_parquet(path)
    print(f"{variant} lag{lag}: {len(pred)} rows, {time.time() - started:.0f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--lag", type=int, default=300)
    args = parser.parse_args()
    data = load(args.lag)
    for name in args.variants:
        run(name, data, args.lag)
