"""Robustness check of spatial_v1 vs the incumbents (skeptic's pass).

    python -B -m spatial.robust_spatial stats [--cand spatial_v1] [--B 2000]
    python -B -m spatial.robust_spatial hp <config> [<config> ...]     # hyperparameter / data-phase reruns

stats: on identical rows (as score_spatial), per-day sums of every error, then
  * day-block bootstrap CIs of the paired differences (candidate - comparator),
  * leave-one-day-out and leave-one-big-event-window-out extremes,
  * week-block and receipt-era (Oct 8+) splits,
  * event errors counted per issue (misses + wrong-way + false) and per episode
    (missed episodes + false clusters + wrong-way clusters).
hp: walk-forward reruns of the spatial heads with perturbed settings (written to
  data/models/spatial_robust/pred_hp_<config>.parquet), scored by `stats --hp`.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, walkforward

OUT = config.OUT / "spatial_robust"
SP = config.OUT / "spatial"
FINAL = config.myt_epoch("2026-10-06")
RECEIPT_ERA = config.myt_epoch("2026-10-08")
CASES = [(c[0], c[1], c[2], c[3]) for c in report.CASES]


# ----------------------------------------------------------------------------- frames
def frames(ds, cand_name="spatial_v1", lag=300, extra=None):
    inc = report.standard_replay(ds)
    cand = pd.read_parquet(SP / f"pred_{cand_name}_lag{lag}.parquet").reindex(ds.index)
    abl_c = pd.read_parquet(SP / f"pred_lr28_lag{lag}.parquet").reindex(ds.index)
    abl_r = pd.read_parquet(SP / f"pred_hgbr_lag{lag}.parquet").reindex(ds.index)
    ev = ds.loc[ds["eval"] & ds.m_fresh_available]
    prior = scoring.walk_forward_prior(ev, "y_class5", list(range(5)), "y_arrival_complete", history=ds)
    pr = pd.DataFrame({f"p_class5_{k}": prior[f"prior_{k}"] for k in range(5)}, index=ev.index).reindex(ds.index)
    pr["p_fall20"], pr["p_rise20"] = pr.p_class5_0 + pr.p_class5_1, pr.p_class5_3 + pr.p_class5_4
    pr["p_fall40"], pr["p_rise40"] = pr.p_class5_0, pr.p_class5_4
    per = scoring.persistence_predictions(ds)
    out = {"cand": cand, "inc": inc, "abl_c": abl_c, "abl_r": abl_r, "prior": pr, "persist": per}
    if extra:
        out.update({k: v.reindex(ds.index) for k, v in extra.items()})
    return ev, out


def masks(ds, ev, f, cls_models, reg_models):
    idx = ev.index
    mc = f["inc"].p_fall20.reindex(idx).notna() & ev.y_class5.isin(range(5))
    for k in cls_models:
        mc &= f[k].p_fall20.reindex(idx).notna()
    mr = f["inc"].pred_arrival.reindex(idx).notna() & f["inc"].pred_ride_mean.reindex(idx).notna()
    for k in reg_models:
        mr &= f[k].pred_arrival.reindex(idx).notna() & f[k].pred_ride_mean.reindex(idx).notna()
    mp = f["inc"].p_ride_le70.reindex(idx).notna() & ev.y_ride_mean.notna()
    for k in reg_models:
        mp &= f[k].p_ride_le70.reindex(idx).notna()
    return mc, mr, mp


# ----------------------------------------------------------------------------- per-issue error columns
def issue_errors(ev, f, mc, mr, mp, cls_models, reg_models):
    """Long frame: one row per issue, columns '<metric>|<model>' with per-issue losses."""
    cols = {}
    y = ev.y_arrival.to_numpy(float)
    ref = ev.y_ref.to_numpy(float)
    for m in reg_models:
        p = f[m].reindex(ev.index)
        for head in ("arrival", "ride_mean", "ride_min", "ride_max"):
            yt = ev[f"y_{head}"].to_numpy(float)
            e = np.abs(p[f"pred_{head}"].to_numpy(float) - yt)
            cols[f"mae_{head}|{m}"] = np.where(mr, e, np.nan)
        big = np.abs(y - ref) >= 20
        cols[f"mae_arr20|{m}"] = np.where(mr & big, np.abs(p.pred_arrival.to_numpy(float) - y), np.nan)
        cols[f"n_arr20|{m}"] = np.where(mr & big, 1.0, np.nan)
        q = p.pred_ride_min.to_numpy(float); d = ev.y_ride_min.to_numpy(float) - q
        cols[f"pin10_min|{m}"] = np.where(mr, np.maximum(0.1 * d, -0.9 * d), np.nan)
        q = p.pred_ride_max.to_numpy(float); d = ev.y_ride_max.to_numpy(float) - q
        cols[f"pin90_max|{m}"] = np.where(mr, np.maximum(0.9 * d, -0.1 * d), np.nan)
        yl = ev.y_ride_le70.to_numpy(float)
        cols[f"brier_le70|{m}"] = np.where(mp, (p.p_ride_le70.to_numpy(float) - yl) ** 2, np.nan)
    p5 = [f"p_class5_{k}" for k in range(5)]
    yc = ev.y_class5.to_numpy(float)
    onehot = (yc[:, None] == np.arange(5)[None, :]).astype(float)
    for m in cls_models:
        p = f[m].reindex(ev.index)
        pm = p[p5].to_numpy(float)
        cols[f"brier5|{m}"] = np.where(mc, ((pm - onehot) ** 2).sum(1), np.nan)
        picked = np.clip((pm * onehot).sum(1), 1e-15, 1)
        cols[f"logloss5|{m}"] = np.where(mc, -np.log(picked), np.nan)
        for b in ("fall20", "rise20", "fall40", "rise40"):
            cols[f"brier_{b}|{m}"] = np.where(mc, (p[f"p_{b}"].to_numpy(float) - ev[f"y_{b}"].to_numpy(float)) ** 2, np.nan)
    return pd.DataFrame(cols, index=ev.index)


def event_tables(ev, f, mc, models, threshold=20, rule="argmax"):
    """Per-issue event outcome codes and per-episode / per-cluster records for each model."""
    e = ev.loc[mc]
    issues = e.index.to_numpy(np.int64)
    truth = scoring.truth_direction(e.y_delta, threshold)
    rec = {}
    for m in models:
        p = f[m].reindex(e.index)
        if m == "persist":
            call = np.zeros(len(e))
        else:
            call = scoring.calls_from_probs(p[f"p_fall{threshold}"], p[f"p_rise{threshold}"], rule)
        ev_ = truth != 0
        code = np.where(ev_ & (call == truth), "hit", np.where(ev_ & (call == 0), "miss",
                        np.where(ev_ & (call == -truth), "wrong", np.where(~ev_ & (call != 0), "false", "tn"))))
        lookup = dict(zip(issues, call))
        eps = []
        for direction in (-1.0, 1.0):
            for run in scoring._runs(issues, truth == direction):
                eps.append({"start": int(run[0]), "end": int(run[-1]), "dir": direction,
                            "caught": any(lookup[i] == direction for i in run)})
        false_cl = [int(r[0]) for r in scoring._runs(issues, (~ev_) & (call != 0))]
        wrong_cl = [int(r[0]) for r in scoring._runs(issues, ev_ & (call == -truth))]
        rec[m] = {"code": pd.Series(code, index=e.index), "episodes": pd.DataFrame(eps),
                  "false_cl": np.array(false_cl, np.int64), "wrong_cl": np.array(wrong_cl, np.int64)}
    return rec


# ----------------------------------------------------------------------------- per-day aggregation
def day_of(epochs):
    return np.array([config.midnight(int(x)) for x in np.asarray(epochs, np.int64)], np.int64)


def per_day(err, rec):
    """Per-day sums (index = MYT midnight epoch)."""
    days = pd.Series(day_of(err.index), index=err.index)
    sums = err.groupby(days.values).sum(min_count=1)
    counts = err.notna().groupby(days.values).sum()
    counts.columns = ["N:" + c for c in counts.columns]
    out = pd.concat([sums, counts], axis=1)
    for m, r in rec.items():
        code_day = day_of(r["code"].index)
        for k in ("hit", "miss", "wrong", "false"):
            out[f"ev_{k}|{m}"] = pd.Series((r["code"].values == k).astype(float)).groupby(code_day).sum()
        ep = r["episodes"]
        if len(ep):
            d = day_of(ep.start.values)
            out[f"ep_n|{m}"] = pd.Series(1.0, index=range(len(ep))).groupby(d).sum()
            out[f"ep_caught|{m}"] = pd.Series(ep.caught.astype(float).values).groupby(d).sum()
            for direction, tag in ((-1.0, "fall"), (1.0, "rise")):
                sel = ep.dir.values == direction
                out[f"ep_n_{tag}|{m}"] = pd.Series(sel.astype(float)).groupby(d).sum()
                out[f"ep_caught_{tag}|{m}"] = pd.Series((ep.caught.values & sel).astype(float)).groupby(d).sum()
        out[f"false_cl|{m}"] = pd.Series(1.0, index=range(len(r["false_cl"]))).groupby(day_of(r["false_cl"])).sum() if len(r["false_cl"]) else 0.0
        out[f"wrong_cl|{m}"] = pd.Series(1.0, index=range(len(r["wrong_cl"]))).groupby(day_of(r["wrong_cl"])).sum() if len(r["wrong_cl"]) else 0.0
    return out.fillna(0.0)


# ----------------------------------------------------------------------------- metrics from day sums
def metric_defs(cand="cand"):
    """name -> (function(day_sum_frame) -> value per model, list of (label, model_a, model_b))."""
    def mean(col):
        return lambda S, m: S[f"{col}|{m}"].sum() / np.maximum(S[f"N:{col}|{m}"].sum(), 1)

    def ev_err(S, m):           # issue-level errors as counted by the user: misses + wrong + false
        return S[f"ev_miss|{m}"].sum() + S[f"ev_wrong|{m}"].sum() + S[f"ev_false|{m}"].sum()

    def ep_err(S, m):           # episode-level: missed episodes + false clusters + wrong clusters
        return (S[f"ep_n|{m}"].sum() - S[f"ep_caught|{m}"].sum()) + S[f"false_cl|{m}"].sum() + S[f"wrong_cl|{m}"].sum()

    def get(col):
        return lambda S, m: S[f"{col}|{m}"].sum()

    return {
        "+90 MAE": (mean("mae_arrival"), [(cand, "persist"), (cand, "inc"), (cand, "abl_r")]),
        "+90 MAE |chg|>=20": (lambda S, m: S[f"mae_arr20|{m}"].sum() / np.maximum(S[f"n_arr20|{m}"].sum(), 1), [(cand, "persist"), (cand, "inc")]),
        "ride mean MAE": (mean("mae_ride_mean"), [(cand, "persist"), (cand, "inc"), (cand, "abl_r")]),
        "ride min MAE": (mean("mae_ride_min"), [(cand, "persist"), (cand, "inc")]),
        "ride max MAE": (mean("mae_ride_max"), [(cand, "persist"), (cand, "inc")]),
        "ride min pinball10": (mean("pin10_min"), [(cand, "inc")]),
        "ride max pinball90": (mean("pin90_max"), [(cand, "inc")]),
        "P(ride<=70) Brier": (mean("brier_le70"), [(cand, "persist"), (cand, "inc"), (cand, "abl_r")]),
        "5-class log loss": (mean("logloss5"), [(cand, "prior"), (cand, "inc"), (cand, "abl_c")]),
        "5-class Brier": (mean("brier5"), [(cand, "prior"), (cand, "inc"), (cand, "abl_c")]),
        "P(fall20) Brier": (mean("brier_fall20"), [(cand, "prior"), (cand, "inc"), (cand, "abl_c")]),
        "P(rise20) Brier": (mean("brier_rise20"), [(cand, "prior"), (cand, "inc")]),
        ">=20 hits": (get("ev_hit"), [(cand, "inc"), (cand, "abl_c")]),
        ">=20 wrong-way": (get("ev_wrong"), [(cand, "inc")]),
        ">=20 false issues": (get("ev_false"), [(cand, "inc")]),
        ">=20 issue errors (miss+wrong+false)": (ev_err, [(cand, "persist"), (cand, "inc"), (cand, "abl_c")]),
        ">=20 episodes caught": (get("ep_caught"), [(cand, "inc"), (cand, "abl_c")]),
        ">=20 fall episodes caught": (get("ep_caught_fall"), [(cand, "inc")]),
        ">=20 false clusters": (get("false_cl"), [(cand, "inc"), (cand, "abl_c")]),
        ">=20 episode errors (missed ep+false cl+wrong cl)": (ep_err, [(cand, "persist"), (cand, "inc"), (cand, "abl_c")]),
    }


class _Sum:
    def __init__(self, x):
        self.x = x

    def sum(self):
        return self.x


class Weighted:
    """S-like view whose column .sum() returns W @ column (one total per row of W)."""
    def __init__(self, S, W):
        self.S, self.W = S, W

    def __getitem__(self, column):
        return _Sum(self.W @ self.S[column].to_numpy(float))


def bootstrap(S, defs, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    days = S.index.to_numpy()
    n = len(days)
    W = np.zeros((B, n))
    for b in range(B):
        np.add.at(W[b], rng.integers(0, n, n), 1.0)
    boot, one, lodo_w = Weighted(S, W), Weighted(S, np.ones(n)), Weighted(S, 1 - np.eye(n))
    rows = []
    for name, (fn, pairs) in defs.items():
        for a, b in pairs:
            try:
                point = float(fn(one, a) - fn(one, b))
            except KeyError:
                continue
            vals = fn(boot, a) - fn(boot, b)
            lodo = fn(lodo_w, a) - fn(lodo_w, b)
            worst = days[int(np.argmax(lodo))]
            rows.append({"metric": name, "vs": b, "cand": float(fn(one, a)), "comp": float(fn(one, b)), "diff": point,
                         "ci_lo": np.percentile(vals, 2.5), "ci_hi": np.percentile(vals, 97.5),
                         "P(diff<0)": float((vals < 0).mean()), "P(diff>0)": float((vals > 0).mean()),
                         "lodo_min": float(lodo.min()), "lodo_max": float(lodo.max()),
                         "lodo_worst_day": str(config.myt(worst))[:10]})
    return pd.DataFrame(rows)


def case_windows(ds):
    out = []
    for label, start, end, direction in CASES:
        a, b = config.myt_epoch(start), config.myt_epoch(end)
        part = ds.loc[(ds.index >= a) & (ds.index <= b)]
        truth = scoring.truth_direction(part.y_delta, 20)
        hits = part.index[truth == direction]
        if len(hits):
            out.append((label, int(hits[0]) - 3 * 3600, int(hits[-1]) + 3 * 3600))
    return out


def summarize(ds, ev, f, cls_models, reg_models, event_models, split_mask, label, B, defs):
    sub = ev.loc[split_mask.reindex(ev.index).fillna(False).to_numpy(bool)]
    mc, mr, mp = masks(ds, sub, f, cls_models, reg_models)
    err = issue_errors(sub, f, mc.to_numpy(), mr.to_numpy(), mp.to_numpy(), cls_models + ["inc", "prior"],
                       reg_models + ["inc", "persist"])
    rec = event_tables(sub, f, mc, event_models)
    S = per_day(err, rec)
    res = bootstrap(S, defs, B=B)
    res.insert(0, "split", label)
    res["n_days"] = len(S)
    return res, sub, mc, mr, mp


def leave_event_out(ds, ev, f, cls_models, reg_models, event_models, split_mask, defs, keep):
    rows = []
    for label, a, b in case_windows(ds):
        m = split_mask & ~((ev.index >= a) & (ev.index <= b))
        if m.sum() == split_mask.sum():
            continue
        sub = ev.loc[m.reindex(ev.index).fillna(False).to_numpy(bool)]
        mc, mr, mp = masks(ds, sub, f, cls_models, reg_models)
        err = issue_errors(sub, f, mc.to_numpy(), mr.to_numpy(), mp.to_numpy(), cls_models + ["inc", "prior"],
                           reg_models + ["inc", "persist"])
        rec = event_tables(sub, f, mc, event_models)
        S = Weighted(per_day(err, rec), None)
        S.W = np.ones(len(S.S))
        for name, (fn, pairs) in defs.items():
            if name not in keep:
                continue
            for a_, b_ in pairs:
                try:
                    rows.append({"left out": label, "metric": name, "vs": b_, "diff": float(fn(S, a_) - fn(S, b_))})
                except KeyError:
                    pass
    return pd.DataFrame(rows)


def fmt(df, cols=None, digits=4):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: f"{x:.{digits}g}" if np.isfinite(x) else "–")
    return df.to_markdown(index=False)


# ----------------------------------------------------------------------------- hyperparameter reruns
HP = {
    # regressor heads (arrival, ride_mean) -- (params override, thin phase, feature set)
    "r_base": ({}, 0, "hgb_sp"),
    "r_phase300": ({}, 300, "hgb_sp"),
    "r_phase600": ({}, 600, "hgb_sp"),
    "r_leaf20": ({"min_samples_leaf": 20}, 0, "hgb_sp"),
    "r_leaf150": ({"min_samples_leaf": 150}, 0, "hgb_sp"),
    "r_iter60": ({"max_iter": 60}, 0, "hgb_sp"),
    "r_lr05_iter240": ({"learning_rate": 0.05, "max_iter": 240}, 0, "hgb_sp"),
    "r_nodes31": ({"max_leaf_nodes": 31}, 0, "hgb_sp"),
    "r_nodes7": ({"max_leaf_nodes": 7}, 0, "hgb_sp"),
    "r_core12": ({}, 0, "hgb_core"),            # only the 12 robust spatial aggregates (no per-station/upwind extras)
    "r_nosp_phase300": ({}, 300, "hgb"),        # ablation at a different phase
    "r_lb14": ({"lookback": 14}, 0, "hgb_sp"),
    # classifier (lr28_sp recipe)
    "c_C0.03": ({"C": 0.03}, 0, "lr28_sp"),
    "c_C0.3": ({"C": 0.3}, 0, "lr28_sp"),
    "c_C1": ({"C": 1.0}, 0, "lr28_sp"),
    "c_noclip": ({"clip": False}, 0, "lr28_sp"),
    "c_net6": ({}, 0, "lr28_net6"),            # network aggregates only, no upwind
    "c_lb14": ({"lookback": 14}, 0, "lr28_sp"),
    "c_base_C0.3": ({"C": 0.3}, 0, "lr28"),
    "c_base_C1": ({"C": 1.0}, 0, "lr28"),
    # placebo: every sp_ input replaced by its value 24 h earlier (same time of day, wrong day)
    "c_placebo24h": ({"shift": 86400}, 0, "lr28_sp"),
    "r_placebo24h": ({"shift": 86400}, 0, "hgb_sp"),
}


def hp_run(name, data, final_only=False):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from spatial import run_spatial as rs

    over, phase, fset = HP[name]
    over = dict(over)
    lookback = over.pop("lookback", 28)
    shift = over.pop("shift", 0)
    if shift:
        data = data.copy()
        sp_cols = [c for c in data.columns if c.startswith("sp_")]
        moved = data[sp_cols].reindex(data.index - shift)
        data[sp_cols] = moved.to_numpy()
    sets = rs.feature_sets(data)
    sets["hgb_core"] = rs.LOG28 + rs.CONTROLS + rs.TIME + rs.WEATHER + rs.CAMS + rs.SPATIAL_CORE
    sets["lr28_net6"] = rs.LOG28 + rs.SPATIAL_CORE[:6]
    cols = sets[fset]
    started = time.time()
    days = [d for d in walkforward.eval_days(data) if d >= FINAL] if final_only else None
    if name.startswith("r_"):
        params = {**rs.HGB_R, **over}
        bparams = {k: v for k, v in params.items() if k != "loss"}

        def fp(train, test, cutoff):
            base_ok = rs._usable(train) & ((train.index.to_numpy(np.int64) % rs.THIN) == phase)
            use = rs._usable(test)
            out = pd.DataFrame(index=test.index)
            xt = test.loc[use, cols].to_numpy(float)
            ref_t = test.loc[use, "s_freshPm25"].to_numpy(float)
            for head, target in (("arrival", "y_arrival"), ("ride_mean", "y_ride_mean")):
                ok = base_ok & np.isfinite(train[target].to_numpy(float))
                tr = train.loc[ok]
                model = HistGradientBoostingRegressor(**params)
                model.fit(tr[cols].to_numpy(float), (tr[target] - tr.s_freshPm25).to_numpy(float))
                pred = np.full(len(test), np.nan)
                pred[use] = ref_t + model.predict(xt)
                out[f"pred_{head}"] = pred
            # P(ride mean <= 70) head omitted here: no robust gain on the original run, and compute is scarce.
            return out
        pred = walkforward.walk_forward(data, fp, targets=("arrival", "ride_mean"),
                                        lookback_days=lookback, embargo_minutes=330, days=days,
                                        log=lambda s: print("   " + s, flush=True))
    else:
        C = over.get("C", 0.1)
        clip = over.get("clip", True)

        def fp(train, test, cutoff):
            ok = rs._usable(train) & train.y_class5.isin(range(5)).to_numpy()
            tr = train.loc[ok]
            y = tr.y_class5.to_numpy(int)
            if len(tr) < 300 or len(set(y)) < 2:
                return None
            xtr, xte = tr[cols].copy(), test[cols].copy()
            sp = [c for c in cols if c.startswith("sp_")]
            if sp and clip:
                lo, hi = xtr[sp].quantile(0.01), xtr[sp].quantile(0.99)
                xtr[sp], xte[sp] = xtr[sp].clip(lo, hi, axis=1), xte[sp].clip(lo, hi, axis=1)
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                                  StandardScaler(), LogisticRegression(C=C, max_iter=500, solver="lbfgs", tol=1e-4))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(xtr.to_numpy(float), y)
            use = rs._usable(test)
            if not use.any():
                return None
            return rs._class_frame(model.predict_proba(xte.loc[use].to_numpy(float)), model[-1].classes_,
                                   test.index[use]).reindex(test.index)
        pred = walkforward.walk_forward(data, fp, targets=("class5",), lookback_days=lookback,
                                        embargo_minutes=120, days=days, log=lambda s: None)
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(OUT / f"pred_hp_{name}.parquet")
    print(f"{name}: {len(pred)} rows {time.time() - started:.0f}s", flush=True)


def hp_table(ds, ev, f):
    """Score every available hp run against persistence / prior and the incumbent on its own identical rows."""
    rows = []
    for name in HP:
        path = OUT / f"pred_hp_{name}.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path).reindex(ds.index)
        for split, m in (("final", ev.index >= FINAL), ("tune", ev.index < FINAL), ("all", np.ones(len(ev), bool))):
            sub = ev.loc[m]
            r = {"config": name, "split": split}
            if "pred_arrival" in p:
                ok = (p.pred_arrival.reindex(sub.index).notna() & f["inc"].pred_arrival.reindex(sub.index).notna()
                      & f["cand"].pred_arrival.reindex(sub.index).notna() & sub.y_arrival.notna()
                      & f["inc"].pred_ride_mean.reindex(sub.index).notna() & p.pred_ride_mean.reindex(sub.index).notna()
                      & f["abl_r"].pred_arrival.reindex(sub.index).notna())
                s = sub.loc[ok]
                if not len(s):
                    continue
                for head in ("arrival", "ride_mean"):
                    yy = s[f"y_{head}"]
                    k = yy.notna()
                    r[f"{head} MAE"] = (p[f"pred_{head}"].reindex(s.index)[k] - yy[k]).abs().mean()
                    r[f"{head} persist"] = (s.y_ref[k] - yy[k]).abs().mean()
                    r[f"{head} Ridge"] = (f["inc"][f"pred_{head}"].reindex(s.index)[k] - yy[k]).abs().mean()
                    r[f"{head} spatial_v1"] = (f["cand"][f"pred_{head}"].reindex(s.index)[k] - yy[k]).abs().mean()
                    r[f"{head} no-spatial"] = (f["abl_r"][f"pred_{head}"].reindex(s.index)[k] - yy[k]).abs().mean()
                r["n"] = int(ok.sum())
            else:
                ok = (p.p_fall20.reindex(sub.index).notna() & f["inc"].p_fall20.reindex(sub.index).notna()
                      & sub.y_class5.isin(range(5)))
                s = sub.loc[ok]
                if not len(s):
                    continue
                sc = scoring.score_candidate(s, p[[c for c in p.columns if c.startswith("p_")]])
                si = scoring.score_candidate(s, f["inc"][[c for c in p.columns if c.startswith("p_")]])
                spr = scoring.score_candidate(s, f["prior"][[c for c in p.columns if c.startswith("p_")]])
                scand = scoring.score_candidate(s, f["cand"][[c for c in p.columns if c.startswith("p_")]])
                e = sc["events20_prob_argmax"]
                ei = si["events20_prob_argmax"]
                r.update({"logloss": sc["class5"]["log_loss"], "logloss inc": si["class5"]["log_loss"],
                          "logloss prior": spr["class5"]["log_loss"], "logloss spatial_v1": scand["class5"]["log_loss"],
                          "Pfall20 Brier": sc["brier_fall20"]["brier"], "Pfall20 inc": si["brier_fall20"]["brier"],
                          "Pfall20 prior": spr["brier_fall20"]["brier"],
                          "ep20": f"{e['episodes_caught']}/{e['episodes']}", "ep20 inc": f"{ei['episodes_caught']}/{ei['episodes']}",
                          "hit/wrong/false (cl)": f"{e['hits']}/{e['wrong_way']}/{e['false_calls']} ({e['false_clusters']})",
                          "inc hit/wrong/false (cl)": f"{ei['hits']}/{ei['wrong_way']}/{ei['false_calls']} ({ei['false_clusters']})",
                          "ep40": f"{sc['events40_prob_argmax']['episodes_caught']}/{sc['events40_prob_argmax']['episodes']}",
                          "n": int(ok.sum())})
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- operating curves
def operating_curve(ev, f, mc, models, split_mask, threshold=20, taus=np.round(np.arange(0.05, 0.651, 0.025), 3)):
    """Episodes caught vs false clusters when any direction with p >= tau is called (larger wins)."""
    e = ev.loc[mc & split_mask.reindex(ev.index).fillna(False)]
    issues = e.index.to_numpy(np.int64)
    truth = scoring.truth_direction(e.y_delta, threshold)
    rows = []
    for m in models:
        p = f[m].reindex(e.index)
        for tau in taus:
            r = scoring.event_confusion(issues, truth, scoring.calls_from_probs(p[f"p_fall{threshold}"], p[f"p_rise{threshold}"], str(tau)))
            rows.append({"model": m, "tau": tau, "episodes": r["episodes_caught"], "of": r["episodes"],
                         "false_cl": r["false_clusters"], "hits": r["hits"], "false": r["false_calls"],
                         "wrong": r["wrong_way"]})
        r = scoring.event_confusion(issues, truth, scoring.calls_from_probs(p[f"p_fall{threshold}"], p[f"p_rise{threshold}"], "argmax"))
        rows.append({"model": m, "tau": "argmax", "episodes": r["episodes_caught"], "of": r["episodes"],
                     "false_cl": r["false_clusters"], "hits": r["hits"], "false": r["false_calls"], "wrong": r["wrong_way"]})
    return pd.DataFrame(rows)


def best_at_budget(curve, budgets):
    out = []
    for m, g in curve.loc[curve.tau != "argmax"].groupby("model", sort=False):
        row = {"model": m}
        for b in budgets:
            ok = g.loc[g.false_cl <= b]
            row[f"ep caught @false_cl<={b}"] = int(ok.episodes.max()) if len(ok) else 0
        out.append(row)
    return pd.DataFrame(out)


def ranking_scores(ev, f, mc, models, split_mask, B=1000, seed=0):
    """Issue-level average precision / ROC AUC of p_fall20 and p_rise20, day-block bootstrap of differences."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    e = ev.loc[mc & split_mask.reindex(ev.index).fillna(False)]
    days = day_of(e.index)
    uniq = np.unique(days)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(uniq), len(uniq)) for _ in range(B)]
    pos = {u: np.flatnonzero(days == u) for u in uniq}
    rows = []
    for side in ("fall20", "rise20"):
        y = e[f"y_{side}"].to_numpy(float)
        sc = {m: f[m].reindex(e.index)[f"p_{side}"].to_numpy(float) for m in models}
        point = {m: (average_precision_score(y, sc[m]), roc_auc_score(y, sc[m])) for m in models}
        boots = {m: [] for m in models}
        for d in draws:
            idx = np.concatenate([pos[uniq[k]] for k in d])
            if y[idx].sum() == 0 or y[idx].sum() == len(idx):
                continue
            for m in models:
                boots[m].append(average_precision_score(y[idx], sc[m][idx]))
        for m in models:
            row = {"target": side, "model": m, "AP": point[m][0], "AUC": point[m][1], "base rate": y.mean()}
            if m != models[0]:
                diff = np.array(boots[models[0]]) - np.array(boots[m])
                row.update({f"AP {models[0]}-this": point[models[0]][0] - point[m][0],
                            "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5),
                            "P(cand better)": float((diff > 0).mean())})
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["stats", "hp", "hptable", "curve"])
    p.add_argument("configs", nargs="*")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--lag", type=int, default=300)
    p.add_argument("--final-only", action="store_true", help="hp: refit only the final-split days")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = dataset.load()
    if a.mode == "hp":
        from spatial import run_spatial as rs
        data = rs.load(a.lag)
        for c in a.configs:
            hp_run(c, data, final_only=a.final_only)
    elif a.mode == "curve":
        extra = {}
        for name in ("c_placebo24h", "c_C1", "c_C0.03"):
            path = OUT / f"pred_hp_{name}.parquet"
            if path.exists():
                extra[name] = pd.read_parquet(path)
        ev, f = frames(ds, lag=a.lag, extra=extra)
        mc, _, _ = masks(ds, ev, f, ["cand", "abl_c"] + list(extra), ["cand"])
        models = ["cand", "inc", "abl_c"] + list(extra)
        pd.set_option("display.width", 250)
        splits = {"final": pd.Series(ev.index >= FINAL, index=ev.index), "tune": pd.Series(ev.index < FINAL, index=ev.index),
                  "all": pd.Series(True, index=ev.index)}
        for label, m in splits.items():
            for t in (20, 40):
                cv = operating_curve(ev, f, mc, models, m, threshold=t)
                cv.to_csv(OUT / f"curve_{label}_{t}.csv", index=False)
                budgets = (5, 10, 15, 20, 25, 30) if label == "final" else (10, 20, 30, 40, 50, 60)
                print(f"## >= {t}, {label}: most episodes caught within a false-cluster budget (any tau)")
                print(best_at_budget(cv, budgets).to_markdown(index=False))
                print(cv.loc[cv.tau == "argmax"].to_markdown(index=False))
            rk = ranking_scores(ev, f, mc, models, m)
            print(f"## ranking, {label}")
            print(rk.round(4).to_markdown(index=False))
    elif a.mode == "hptable":
        ev, f = frames(ds, lag=a.lag)
        t = hp_table(ds, ev, f)
        t.to_csv(OUT / "hp_table.csv", index=False)
        pd.set_option("display.width", 250)
        for kind in ("r_", "c_"):
            sub = t.loc[t.config.str.startswith(kind)].dropna(axis=1, how="all")
            print(sub.round(4).to_markdown(index=False))
            print()
    else:
        ev, f = frames(ds, lag=a.lag)
        cls_models = ["cand", "abl_c"]
        reg_models = ["cand", "abl_r"]
        event_models = ["cand", "inc", "abl_c", "persist"]
        defs = metric_defs()
        allm = pd.Series(True, index=ev.index)
        splits = {"final Oct06-13": pd.Series(ev.index >= FINAL, index=ev.index),
                  "tune Sep10-Oct05": pd.Series(ev.index < FINAL, index=ev.index),
                  "all 34d": allm,
                  "receipt era Oct08-13": pd.Series(ev.index >= RECEIPT_ERA, index=ev.index)}
        for i, (lo, hi) in enumerate((("2026-09-10", "2026-09-17"), ("2026-09-17", "2026-09-24"),
                                      ("2026-09-24", "2026-10-01"), ("2026-10-01", "2026-10-08"),
                                      ("2026-10-08", "2026-10-14"))):
            splits[f"week {lo[5:]}..{hi[5:]}"] = pd.Series((ev.index >= config.myt_epoch(lo)) & (ev.index < config.myt_epoch(hi)), index=ev.index)
        res = []
        for label, m in splits.items():
            r, *_ = summarize(ds, ev, f, cls_models, reg_models, event_models, m, label, a.B if "week" not in label else 500, defs)
            res.append(r)
            print(f"done {label}", flush=True)
        res = pd.concat(res)
        res.to_csv(OUT / "bootstrap.csv", index=False)
        keep = ["+90 MAE", "ride mean MAE", "P(ride<=70) Brier", "5-class log loss", "P(fall20) Brier",
                ">=20 hits", ">=20 issue errors (miss+wrong+false)", ">=20 episodes caught",
                ">=20 episode errors (missed ep+false cl+wrong cl)"]
        leo = []
        for label in ("final Oct06-13", "all 34d"):
            t = leave_event_out(ds, ev, f, cls_models, reg_models, event_models, splits[label], defs, keep)
            t.insert(0, "split", label)
            leo.append(t)
        leo = pd.concat(leo)
        leo.to_csv(OUT / "leave_event_out.csv", index=False)
        pd.set_option("display.width", 250)
        print(fmt(res))
        print()
        print(fmt(leo))
