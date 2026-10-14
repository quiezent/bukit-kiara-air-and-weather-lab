"""Weather-inputs candidate: literal model outputs, walk-forward daily refits.

Heads
* class5  : 5-class +90 arrival change (<=-40, (-40,-20], (-20,20), [20,40), >=40).
            Tails p_fall20 = p0 + p1, p_rise20 = p3 + p4, p_fall40 = p0, p_rise40 = p4.
            Embargo 120 min (production arrival logistic).
* arrival, ride_mean, ride_min, ride_max : point forecasts = reference + HGB(delta),
            absolute-error loss. Embargo 330 min (production Ridge v4).
* le70    : P(ride mean <= 70), HGB binary classifier. Embargo 330 min.

Every head is run twice with the same algorithm and hyperparameters: sensor-only
(the 28 production arrival-logistic inputs) and sensor + issued weather
(wxfeatures.build). The difference is the value of the weather inputs.

    python -B weather-inputs/candidate.py tune     # tuning grid, tune days only (2026-09-10..09-26)
    python -B weather-inputs/candidate.py final [head,head]   # frozen config (final_config.json), all eval days, both feature sets
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # research/models -> harness
sys.path.insert(0, str(HERE))

from harness import config, dataset, walkforward   # noqa: E402  (sets OMP/MKL threads = 2 first)

import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor   # noqa: E402
from sklearn.impute import SimpleImputer   # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.pipeline import make_pipeline   # noqa: E402
from sklearn.preprocessing import StandardScaler   # noqa: E402

import wxfeatures   # noqa: E402

OUT = config.OUT / "weather-inputs"
TUNE_END = config.myt_epoch("2026-09-27")          # tune on 09-10..09-26, report on 09-27..10-13
SEED = 20261013
EMBARGO = {"class5": 120, "arrival": 330, "ride_mean": 330, "ride_min": 330, "ride_max": 330, "le70": 330}
TARGET_KEY = {"class5": "class5", "arrival": "arrival", "ride_mean": "ride_mean", "ride_min": "ride_mean",
              "ride_max": "ride_mean", "le70": "ride_le70"}


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
    ds = dataset.load()
    wx = wxfeatures.build(ds)
    ds = ds.join(wx.add_prefix("x_"))        # x_ prefix: candidate-only columns, never y_*
    sensor = list(wxfeatures.SENSOR28)
    weather = ["x_" + c for c in wx.columns]
    return ds, sensor, weather


def weather_columns(cfg, weather):
    """cfg['wx'] = 'full' (default, all wxfeatures columns) or 'core' (wxfeatures.CORE)."""
    return weather if cfg.get("wx", "full") == "full" else ["x_" + c for c in wxfeatures.CORE]


# ----------------------------------------------------------------------------- estimators
def thin(train, cfg):
    """Training origins every cfg['stride'] minutes (default 5 = every grid row). Test rows are never thinned."""
    stride = int(cfg.get("stride", 5)) * 60
    return train.loc[np.asarray(train.index, np.int64) % stride == 0] if stride > 300 else train


def hgb_params(cfg):
    return dict(learning_rate=cfg.get("lr", 0.05), max_iter=cfg.get("iters", 200), max_depth=3,
                max_leaf_nodes=7, min_samples_leaf=cfg.get("leaf", 80), l2_regularization=10.0,
                early_stopping=False, random_state=SEED)


def class5_fit_predict(cols, cfg):
    def fit_predict(train, test, cutoff):
        train = thin(train, cfg)
        ok = train.m_fresh_available.to_numpy(bool) & train.y_class5.isin([0, 1, 2, 3, 4]).to_numpy()
        x, y = train.loc[ok, cols].to_numpy(float), train.loc[ok, "y_class5"].to_numpy(int)
        if cfg["algo"] == "hgb":
            model = HistGradientBoostingClassifier(**hgb_params(cfg))
        else:   # production logistic recipe with the extra columns
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                  StandardScaler(), LogisticRegression(C=cfg.get("C", 0.1), max_iter=2000))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x, y)
        usable = test.m_fresh_available.to_numpy(bool)
        p = np.full((len(test), 5), np.nan)
        if usable.any():
            raw = model.predict_proba(test.loc[usable, cols].to_numpy(float))
            full = np.zeros((usable.sum(), 5))
            classes = model.classes_ if cfg["algo"] == "hgb" else model[-1].classes_
            for k, c in enumerate(classes):
                full[:, int(c)] = raw[:, k]
            p[usable] = full
        out = pd.DataFrame(p, index=test.index, columns=[f"p_class5_{k}" for k in range(5)])
        out["p_fall20"] = out.p_class5_0 + out.p_class5_1
        out["p_rise20"] = out.p_class5_3 + out.p_class5_4
        out["p_fall40"] = out.p_class5_0
        out["p_rise40"] = out.p_class5_4
        return out
    return fit_predict


def point_fit_predict(head, cols, cfg):
    target = f"y_{head}"

    def fit_predict(train, test, cutoff):
        train = thin(train, cfg)
        ref = train.s_freshPm25.to_numpy(float)
        ok = train.m_fresh_available.to_numpy(bool) & np.isfinite(train[target].to_numpy(float)) & np.isfinite(ref)
        if head.startswith("ride"):
            ok &= train.y_ride_joint_valid.fillna(False).to_numpy(bool)
        model = HistGradientBoostingRegressor(loss=cfg.get("loss", "absolute_error"), **hgb_params(cfg))
        model.fit(train.loc[ok, cols].to_numpy(float), (train[target].to_numpy(float) - ref)[ok])
        usable = test.m_fresh_available.to_numpy(bool)
        pred = np.full(len(test), np.nan)
        if usable.any():
            pred[usable] = test.s_freshPm25.to_numpy(float)[usable] + model.predict(test.loc[usable, cols].to_numpy(float))
        return pd.DataFrame({f"pred_{head}": pred}, index=test.index)
    return fit_predict


def le70_fit_predict(cols, cfg):
    def fit_predict(train, test, cutoff):
        train = thin(train, cfg)
        ok = (train.m_fresh_available.to_numpy(bool) & train.y_ride_joint_valid.fillna(False).to_numpy(bool)
              & np.isfinite(train.y_ride_le70.to_numpy(float)))
        y = train.loc[ok, "y_ride_le70"].to_numpy(int)
        usable = test.m_fresh_available.to_numpy(bool)
        p = np.full(len(test), np.nan)
        if len(set(y)) == 2 and usable.any():
            model = HistGradientBoostingClassifier(**hgb_params(cfg)).fit(train.loc[ok, cols].to_numpy(float), y)
            p[usable] = model.predict_proba(test.loc[usable, cols].to_numpy(float))[:, 1]
        return pd.DataFrame({"p_ride_le70": p}, index=test.index)
    return fit_predict


def run_head(ds, head, cols, cfg, days, log=lambda *_: None) -> pd.DataFrame:
    if head == "class5":
        fp = class5_fit_predict(cols, cfg)
    elif head == "le70":
        fp = le70_fit_predict(cols, cfg)
    else:
        fp = point_fit_predict(head, cols, cfg)
    return walkforward.walk_forward(ds, fp, targets=(TARGET_KEY[head],), lookback_days=cfg.get("lookback", 28),
                                    embargo_minutes=EMBARGO[head], days=days, log=log)


# ----------------------------------------------------------------------------- configs
# Kept small on purpose: the laptop is shared with other jobs and throttled (a 17-day HGB class run on
# 5-min origins x 200 iterations had not finished after 33 min at idle priority and was abandoned).
# Only the sensor + weather model is scored for selection (the sensor-only logistic row checks that the
# harness reproduces the incumbent recipe). Point and P(<=70) heads: 28-d lookback (Ridge v4), leaf 80,
# absolute error; only the weather set (full / core) is chosen. ride_min/ride_max follow ride_mean.
CHEAP = dict(stride=15, iters=100, lr=0.1)      # 6x less compute than 5-min origins x 200 iterations
TUNE_GRID = {
    "class5": [dict(algo="logit", lookback=28), dict(algo="hgb", lookback=28, leaf=80, **CHEAP),
               dict(algo="hgb", lookback=28, leaf=80, wx="core", **CHEAP), dict(algo="hgb", lookback=60, leaf=80, **CHEAP),
               dict(algo="logit", lookback=28, wx="core")],
    "arrival": [dict(lookback=28, leaf=80, **CHEAP), dict(lookback=28, leaf=80, wx="core", **CHEAP)],
    "ride_mean": [dict(lookback=28, leaf=80, **CHEAP), dict(lookback=28, leaf=80, wx="core", **CHEAP)],
    "le70": [dict(lookback=28, leaf=80, **CHEAP), dict(lookback=28, leaf=80, wx="core", **CHEAP)],
}


def tune():
    ds, sensor, weather = load()
    days = walkforward.eval_days(ds, end=TUNE_END)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tune_runs.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for head, grid in TUNE_GRID.items():
        for i, cfg in enumerate(grid):
            sets = ((("S", sensor), ("SW", sensor + weather)) if cfg.get("algo") == "logit" and i == 0
                    else (("SW", sensor + weather_columns(cfg, weather)),))
            for fs_name, cols in sets:
                key = f"{head}__{i}__{fs_name}"
                if key in results and results[key]["cfg"] == cfg and (OUT / f"tune_{key}.parquet").exists():
                    continue
                started = time.time()
                pred = run_head(ds, head, cols, cfg, days)
                pred.to_parquet(OUT / f"tune_{key}.parquet")
                results[key] = {"cfg": cfg, "seconds": round(time.time() - started, 1)}
                path.write_text(json.dumps(results, indent=1))
                print(key, cfg, f"{time.time() - started:.0f}s", flush=True)


def final(cfg_path=HERE / "final_config.json"):
    ds, sensor, weather = load()
    days = walkforward.eval_days(ds)
    chosen = json.loads(Path(cfg_path).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    timing_path = OUT / "final_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    heads = sys.argv[2].split(",") if len(sys.argv) > 2 else ["arrival", "ride_mean", "ride_min", "ride_max", "le70", "class5"]
    for head in heads:
        for fs_name in ("SW", "S"):
            path = OUT / f"final_{fs_name}_{head}.parquet"
            if path.exists():
                continue
            cfg = chosen["ride_mean" if head in ("ride_min", "ride_max") else head]
            cols = sensor + (weather_columns(cfg, weather) if fs_name == "SW" else [])
            fits = []
            pred = run_head(ds, head, cols, cfg, days,
                            log=lambda line: fits.append(float(line.rsplit(" ", 1)[-1].rstrip("s"))))
            pred.to_parquet(path)
            timing[f"{head}__{fs_name}"] = {"days": len(fits), "median_fit_predict_s": float(np.median(fits)),
                                            "max_s": float(np.max(fits)), "features": len(cols)}
            timing_path.write_text(json.dumps(timing, indent=1))
            print(head, fs_name, timing[f"{head}__{fs_name}"], flush=True)
    for fs_name in ("SW", "S"):
        parts = [OUT / f"final_{fs_name}_{h}.parquet" for h in
                 ("class5", "arrival", "ride_mean", "ride_min", "ride_max", "le70")]
        if all(p.exists() for p in parts):
            pd.concat([pd.read_parquet(p) for p in parts], axis=1).to_parquet(OUT / f"final_{fs_name}.parquet")


if __name__ == "__main__":
    {"tune": tune, "final": final}[sys.argv[1]]()
