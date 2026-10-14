"""Skeptic's robustness pass on spatial_v1, written independently of robust_spatial.py.

    python -B -m spatial.skeptic_spatial stats [--B 5000]
    python -B -m spatial.skeptic_spatial hp <config> [<config> ...]
    python -B -m spatial.skeptic_spatial hpstats

stats: losses on identical rows (as score_spatial), summed per MYT day, then
  * day-block bootstrap of the paired difference candidate - comparator (lower = candidate better)
    on the final, tune and whole splits and on alternative splits (halves, weeks, receipt era);
  * leave-one-day-out and leave-one-big-event-window-out (case window +-2 h) extremes;
  * event errors per issue (misses + wrong-way + false calls) and per episode
    (missed episodes + false-call clusters + wrong-way clusters).
hp: walk-forward reruns with perturbed hyperparameters, neighbour lags (600/900 s, after `features`) or a
  24-168 h time-shifted placebo of the spatial block, written to data/models/spatial_skeptic2/pred_<config>.parquet
  (--final: held-out days only); hpstats scores them against matched ablations.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, walkforward

SP = config.OUT / "spatial"
OUT = config.OUT / "spatial_skeptic2"
FINAL = config.myt_epoch("2026-10-06")
P5 = [f"p_class5_{k}" for k in range(5)]


# ----------------------------------------------------------------------------- inputs
def model_frames(ds, cand=None, abl_c=None, abl_r=None):
    inc = report.standard_replay(ds)
    cand = pd.read_parquet(SP / "pred_spatial_v1_lag300.parquet") if cand is None else cand
    abl_c = pd.read_parquet(SP / "pred_lr28_lag300.parquet") if abl_c is None else abl_c
    abl_r = pd.read_parquet(SP / "pred_hgbr_lag300.parquet") if abl_r is None else abl_r
    ev = ds.loc[ds["eval"] & ds.m_fresh_available]
    prior = scoring.walk_forward_prior(ev, "y_class5", list(range(5)), "y_arrival_complete", history=ds)
    pr = pd.DataFrame({f"p_class5_{k}": prior[f"prior_{k}"] for k in range(5)}, index=ev.index)
    pr["p_fall20"], pr["p_rise20"] = pr.p_class5_0 + pr.p_class5_1, pr.p_class5_3 + pr.p_class5_4
    pr["p_fall40"], pr["p_rise40"] = pr.p_class5_0, pr.p_class5_4
    per = scoring.persistence_predictions(ev)
    r = lambda f: f.reindex(ev.index)
    return ev, {"cand": r(cand), "inc": r(inc), "abl_c": r(abl_c), "abl_r": r(abl_r), "prior": r(pr), "persist": per}


def episode_records(issues, truth, call):
    """(start_epoch, kind) for missed episodes, caught episodes, false clusters, wrong-way clusters."""
    rec = []
    for d in (-1.0, 1.0):
        for run in scoring._runs(issues, truth == d):
            pos = np.searchsorted(issues, run)
            rec.append((int(run[0]), "caught" if np.any(call[pos] == d) else "missed"))
    for run in scoring._runs(issues, (truth == 0) & (call != 0)):
        rec.append((int(run[0]), "false_cl"))
    for run in scoring._runs(issues, (truth != 0) & (call == -truth)):
        rec.append((int(run[0]), "wrong_cl"))
    return rec


def build_losses(ev, f):
    """Per-issue loss columns {metric: {model: Series}} plus per-metric row masks and episode records."""
    idx = ev.index
    mc = f["inc"].p_fall20.notna() & f["cand"].p_fall20.notna() & f["abl_c"].p_fall20.notna() & ev.y_class5.isin(range(5))
    mr = (f["inc"].pred_arrival.notna() & f["inc"].pred_ride_mean.notna() & f["cand"].pred_arrival.notna()
          & f["cand"].pred_ride_mean.notna() & f["abl_r"].pred_arrival.notna() & f["abl_r"].pred_ride_mean.notna())
    mp = f["inc"].p_ride_le70.notna() & ev.y_ride_mean.notna() & f["cand"].p_ride_le70.notna() & f["abl_r"].p_ride_le70.notna()
    L, M = {}, {}

    def add(metric, mask, losses):
        M[metric] = mask & np.all([np.isfinite(v) for v in losses.values()], axis=0)
        L[metric] = {k: pd.Series(v, index=idx) for k, v in losses.items()}

    reg_models = {"cand": f["cand"], "inc": f["inc"], "abl": f["abl_r"], "persist": f["persist"]}
    for head in ("arrival", "ride_mean", "ride_min", "ride_max"):
        y = ev[f"y_{head}"].to_numpy(float)
        add(f"{head} MAE", mr, {k: np.abs(m[f"pred_{head}"].to_numpy(float) - y) for k, m in reg_models.items()})
    big = mr & ((ev.y_arrival - ev.y_ref).abs() >= 20)
    add("arrival MAE |chg|>=20", big, {k: np.abs(m.pred_arrival.to_numpy(float) - ev.y_arrival.to_numpy(float))
                                        for k, m in reg_models.items()})
    for head, tau in (("ride_min", 0.1), ("ride_max", 0.9)):
        y = ev[f"y_{head}"].to_numpy(float)
        add(f"{head} pinball{tau}", mr, {k: np.maximum(tau * (y - m[f"pred_{head}"].to_numpy(float)),
                                                        (tau - 1) * (y - m[f"pred_{head}"].to_numpy(float)))
                                          for k, m in reg_models.items()})
    y70 = ev.y_ride_le70.to_numpy(float)
    add("P(ride<=70) Brier", mp, {k: (m.p_ride_le70.to_numpy(float) - y70) ** 2 for k, m in reg_models.items()})

    cls_models = {"cand": f["cand"], "inc": f["inc"], "abl": f["abl_c"], "prior": f["prior"]}
    y5 = ev.y_class5.to_numpy(float)
    onehot = (y5[:, None] == np.arange(5)[None, :]).astype(float)
    add("5-class log loss", mc, {k: -np.log(np.clip(np.sum(m[P5].to_numpy(float) * onehot, axis=1), 1e-15, 1))
                                  for k, m in cls_models.items()})
    add("5-class Brier", mc, {k: np.sum((m[P5].to_numpy(float) - onehot) ** 2, axis=1) for k, m in cls_models.items()})
    for b in ("fall20", "rise20", "fall40"):
        yb = ev[f"y_{b}"].to_numpy(float)
        add(f"P({b}) Brier", mc, {k: (m[f"p_{b}"].to_numpy(float) - yb) ** 2 for k, m in cls_models.items()})

    # events: per-issue errors and hits; per-episode records
    ev_models = {"cand": f["cand"], "inc": f["inc"], "abl": f["abl_c"], "persist": f["persist"]}
    issues_all = idx.to_numpy(np.int64)
    E = {}
    for t in (20, 40):
        truth = scoring.truth_direction(ev.y_delta, t)
        for rule in ("argmax", "0.2"):
            calls = {k: scoring.calls_from_probs(m[f"p_fall{t}"], m[f"p_rise{t}"], rule) for k, m in ev_models.items()}
            err = {k: ((truth != 0) & (c != truth)) | ((truth == 0) & (c != 0)) for k, c in calls.items()}
            add(f">={t} {rule} issue errors", mc, {k: v.astype(float) for k, v in err.items()})
            add(f">={t} {rule} false issues", mc, {k: ((truth == 0) & (c != 0)).astype(float) for k, c in calls.items()})
            add(f">={t} {rule} non-hits", mc, {k: ((truth != 0) & (c != truth)).astype(float) for k, c in calls.items()})
            m = mc.to_numpy(bool)
            for k, c in calls.items():
                E[(t, rule, k)] = episode_records(issues_all[m], truth[m], c[m])
    return L, M, E


# ----------------------------------------------------------------------------- aggregation
def day_vectors(ev, L, M, E, days, drop=None):
    """num[metric][model] and den[metric] arrays over ``days`` (drop: bool Series of issues to exclude)."""
    day_of = ev.date_myt.astype(str)
    pos = {d: i for i, d in enumerate(days)}
    di = day_of.map(pos)
    num, den, kind = {}, {}, {}
    for metric, losses in L.items():
        mask = M[metric].copy()
        if drop is not None:
            mask &= ~drop
        mask &= di.notna()
        g = di[mask].astype(int).to_numpy()
        den[metric] = np.bincount(g, minlength=len(days)).astype(float)
        num[metric] = {k: np.bincount(g, weights=v[mask].to_numpy(float), minlength=len(days)) for k, v in losses.items()}
        kind[metric] = "sum" if ("errors" in metric or "false" in metric or "non-hits" in metric) else "mean"
    drop_epochs = set(ev.index[drop].to_numpy(np.int64)) if drop is not None else set()
    for (t, rule, k), rec in E.items():
        for name, kinds in ((f">={t} {rule} episode errors", ("missed", "false_cl", "wrong_cl")),
                            (f">={t} {rule} missed episodes", ("missed",)),
                            (f">={t} {rule} false clusters", ("false_cl",))):
            vec = np.zeros(len(days))
            for start, kd in rec:
                if kd in kinds and start not in drop_epochs:
                    d = str(config.myt(start).date())
                    if d in pos:
                        vec[pos[d]] += 1
            num.setdefault(name, {})[k] = vec
            den[name] = np.ones(len(days))
            kind[name] = "sum"
    return num, den, kind


def stat(num, den, kind, metric, model, sel=None):
    a = num[metric][model] if sel is None else num[metric][model][sel]
    if kind[metric] == "sum":
        return float(a.sum())
    b = den[metric] if sel is None else den[metric][sel]
    return float(a.sum() / b.sum()) if b.sum() else np.nan


def compare(num, den, kind, metric, a, b, B, rng):
    """Point difference, bootstrap 95% CI, share of resamples where a is better (lower), days a better."""
    n = len(den[metric])
    d = num[metric][a] - num[metric][b]
    keep = den[metric] > 0 if kind[metric] == "mean" else np.ones(n, bool)
    point = stat(num, den, kind, metric, a) - stat(num, den, kind, metric, b)
    idx = rng.integers(0, n, (B, n))
    if kind[metric] == "sum":
        boot = d[idx].sum(axis=1)
    else:
        dd = den[metric][idx].sum(axis=1)
        boot = np.where(dd > 0, d[idx].sum(axis=1) / np.where(dd > 0, dd, 1), np.nan)
    boot = boot[np.isfinite(boot)]
    per_day = np.where(keep, d / np.where(den[metric] > 0, den[metric], 1), np.nan) if kind[metric] == "mean" else d
    lodo = [stat(num, den, kind, metric, a, np.arange(n) != i) - stat(num, den, kind, metric, b, np.arange(n) != i)
            for i in range(n)] if n > 1 else [np.nan]
    if not len(boot):
        boot = np.array([np.nan])
    return {"diff": point, "lo": np.nanpercentile(boot, 2.5) if np.isfinite(boot).any() else np.nan,
            "hi": np.nanpercentile(boot, 97.5) if np.isfinite(boot).any() else np.nan,
            "P(cand better)": float(np.mean(boot < 0)), "P(tie)": float(np.mean(boot == 0)),
            "days better/worse": f"{int(np.sum(per_day < 0))}/{int(np.sum(per_day > 0))}",
            "LODO worst": float(np.nanmax(lodo)), "LODO best": float(np.nanmin(lodo))}


METRICS = [  # (metric, comparators)
    ("arrival MAE", ("inc", "persist", "abl")), ("arrival MAE |chg|>=20", ("inc", "persist", "abl")),
    ("ride_mean MAE", ("inc", "persist", "abl")), ("ride_min MAE", ("inc", "persist")),
    ("ride_max MAE", ("inc", "persist")), ("ride_min pinball0.1", ("inc",)), ("ride_max pinball0.9", ("inc",)),
    ("P(ride<=70) Brier", ("inc", "persist", "abl")),
    ("5-class log loss", ("inc", "prior", "abl")), ("5-class Brier", ("inc", "prior")),
    ("P(fall20) Brier", ("inc", "prior", "abl")), ("P(rise20) Brier", ("inc", "prior")), ("P(fall40) Brier", ("inc", "prior")),
    (">=20 argmax issue errors", ("inc", "persist", "abl")), (">=20 argmax non-hits", ("inc", "persist", "abl")),
    (">=20 argmax false issues", ("inc", "abl")),
    (">=20 argmax episode errors", ("inc", "persist", "abl")), (">=20 argmax missed episodes", ("inc", "persist", "abl")),
    (">=20 argmax false clusters", ("inc", "abl")),
    (">=20 0.2 issue errors", ("inc", "persist", "abl")), (">=20 0.2 episode errors", ("inc", "persist", "abl")),
    (">=20 0.2 missed episodes", ("inc", "abl")),
    (">=40 argmax issue errors", ("inc", "persist")), (">=40 argmax episode errors", ("inc", "persist")),
    (">=40 argmax missed episodes", ("inc",)),
    (">=40 0.2 issue errors", ("inc", "persist")), (">=40 0.2 episode errors", ("inc", "persist")),
    (">=40 0.2 missed episodes", ("inc",)),
]


def splits(all_days):
    d = pd.to_datetime(pd.Series(all_days))
    sel = lambda a, b: [x for x, t in zip(all_days, d) if pd.Timestamp(a) <= t <= pd.Timestamp(b)]
    return {"final Oct06-13": sel("2026-10-06", "2026-10-13"), "tune Sep10-Oct05": sel("2026-09-10", "2026-10-05"),
            "all 34d": list(all_days), "half A Sep10-26": sel("2026-09-10", "2026-09-26"),
            "half B Sep27-Oct13": sel("2026-09-27", "2026-10-13"), "receipt era Oct08-13": sel("2026-10-08", "2026-10-13"),
            "wk Sep10-16": sel("2026-09-10", "2026-09-16"), "wk Sep17-23": sel("2026-09-17", "2026-09-23"),
            "wk Sep24-30": sel("2026-09-24", "2026-09-30"), "wk Oct01-07": sel("2026-10-01", "2026-10-07"),
            "wk Oct08-13": sel("2026-10-08", "2026-10-13")}


def run_stats(B):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = dataset.load()
    ev, f = model_frames(ds)
    L, M, E = build_losses(ev, f)
    all_days = sorted(ev.date_myt.astype(str).unique())
    rows = []
    for split, days in splits(all_days).items():
        num, den, kind = day_vectors(ev, L, M, E, days)
        rng = np.random.default_rng(20261014)
        for metric, comps in METRICS:
            for c in comps:
                r = compare(num, den, kind, metric, "cand", c, B, rng)
                rows.append({"split": split, "metric": metric, "vs": c, "cand": stat(num, den, kind, metric, "cand"),
                             "comp": stat(num, den, kind, metric, c), **r, "n_days": len(days)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "bootstrap.csv", index=False)

    # leave-one-big-event-window-out (window +-2 h around each report case), on final and all
    lrows = []
    for split in ("final Oct06-13", "all 34d"):
        days = splits(all_days)[split]
        for label, a, b, _ in [("none", None, None, 0)] + list(report.CASES):
            if a is None:
                drop = None
            else:
                lo, hi = config.myt_epoch(a) - 7200, config.myt_epoch(b) + 7200
                if str(config.myt(lo).date()) not in days and str(config.myt(hi).date()) not in days:
                    continue
                drop = pd.Series((ev.index >= lo) & (ev.index <= hi), index=ev.index)
            num, den, kind = day_vectors(ev, L, M, E, days, drop)
            for metric, comps in METRICS:
                for c in comps:
                    lrows.append({"split": split, "dropped": label, "metric": metric, "vs": c,
                                  "cand": stat(num, den, kind, metric, "cand"), "comp": stat(num, den, kind, metric, c),
                                  "diff": stat(num, den, kind, metric, "cand") - stat(num, den, kind, metric, c)})
    leo = pd.DataFrame(lrows)
    leo.to_csv(OUT / "leave_event_out.csv", index=False)
    pd.set_option("display.width", 250)
    for split in ("final Oct06-13", "tune Sep10-Oct05", "all 34d"):
        print(f"\n## {split}")
        print(table.loc[table.split == split].drop(columns=["split", "n_days"]).round(4).to_markdown(index=False))
    print("\n## alternative splits (diff [CI], P(cand better))")
    alt = table.loc[~table.split.isin(["final Oct06-13", "tune Sep10-Oct05", "all 34d"])].copy()
    alt["cell"] = alt.apply(lambda r: f"{r['diff']:+.3g} [{r['lo']:+.3g},{r['hi']:+.3g}] {r['P(cand better)']:.2f}", axis=1)
    print(alt.pivot_table(index=["metric", "vs"], columns="split", values="cell", aggfunc="first").to_markdown())
    print("\n## leave-one-event-window-out: diff (cand - comp)")
    leo["cell"] = leo["diff"].round(3)
    print(leo.pivot_table(index=["split", "metric", "vs"], columns="dropped", values="cell", aggfunc="first").to_markdown())


# ----------------------------------------------------------------------------- hyperparameter / placebo reruns
def placebo(ds, seconds=86400):
    sp = [c for c in ds.columns if c.startswith("sp_")]
    shifted = ds[sp].reindex(ds.index - seconds)
    shifted.index = ds.index
    out = ds.copy()
    out[sp] = shifted
    return out


HP = {  # name: (kind, settings)
    "r_base": ("reg", {}), "r_iter60": ("reg", {"max_iter": 60}), "r_iter240_lr05": ("reg", {"max_iter": 240, "learning_rate": 0.05}),
    "r_leaf20": ("reg", {"min_samples_leaf": 20}), "r_leaf150": ("reg", {"min_samples_leaf": 150}),
    "r_nodes31": ("reg", {"max_leaf_nodes": 31}), "r_nodes7": ("reg", {"max_leaf_nodes": 7}),
    "r_phase600": ("reg", {"_phase": 600}), "r_lb14": ("reg", {"_lookback": 14}), "r_placebo24": ("reg", {"_placebo": True}),
    "ra_leaf20": ("reg", {"min_samples_leaf": 20, "_abl": True}), "ra_phase600": ("reg", {"_phase": 600, "_abl": True}),
    "ra_iter240_lr05": ("reg", {"max_iter": 240, "learning_rate": 0.05, "_abl": True}),
    "c_base": ("cls", {}), "c_C0.03": ("cls", {"_C": 0.03}), "c_C0.3": ("cls", {"_C": 0.3}), "c_C1": ("cls", {"_C": 1.0}),
    "c_noclip": ("cls", {"_noclip": True}), "c_lb14": ("cls", {"_lookback": 14}), "c_lb42": ("cls", {"_lookback": 42}),
    "c_placebo24": ("cls", {"_placebo": True}), "c_net6": ("cls", {"_net6": True}),
    "ca_C0.03": ("cls", {"_C": 0.03, "_abl": True}), "ca_C0.3": ("cls", {"_C": 0.3, "_abl": True}),
    "ca_C1": ("cls", {"_C": 1.0, "_abl": True}), "ca_lb14": ("cls", {"_lookback": 14, "_abl": True}),
    "c_lag600": ("cls", {"_lag": 600}), "c_lag900": ("cls", {"_lag": 900}),
    "r_lag600": ("reg", {"_lag": 600}), "r_lag900": ("reg", {"_lag": 900}),
    **{f"c_placebo{d * 24}": ("cls", {"_placebo": True, "_pdays": d}) for d in (2, 3, 4, 5, 6, 7)},
    "r_placebo72": ("reg", {"_placebo": True, "_pdays": 3}),
    "r_placebo120": ("reg", {"_placebo": True, "_pdays": 5}), "ra_lb14": ("reg", {"_lookback": 14, "_abl": True}),
    "ra_nodes31": ("reg", {"max_leaf_nodes": 31, "_abl": True}),
}


def run_hp(name, final_only=False):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from spatial import run_spatial as rs

    kind, s = HP[name]
    ds = rs.load(300)
    if s.get("_lag"):   # neighbour visibility lag sensitivity: features built by `features <lag>` into OUT
        alt = pd.read_parquet(OUT / f"spatial_features_lag{s['_lag']}.parquet").reindex(ds.index)
        ds[alt.columns] = alt
    if s.get("_placebo"):
        ds = placebo(ds, 86400 * s.get("_pdays", 1))
    sets = rs.feature_sets(ds)
    lookback = s.get("_lookback", 28)
    days = walkforward.eval_days(ds, start=FINAL) if final_only else None   # contended laptop: held-out days only
    started = time.time()
    if kind == "reg":
        cols = sets["hgb" if s.get("_abl") else "hgb_sp"]
        params = {**rs.HGB_R, **{k: v for k, v in s.items() if not k.startswith("_")}}
        bparams = {**rs.HGB_B, **{k: v for k, v in s.items() if not k.startswith("_")}}
        phase = s.get("_phase", 0)
        heads = {"arrival": "y_arrival", "ride_mean": "y_ride_mean", "ride_min": "y_ride_min", "ride_max": "y_ride_max"}

        def fp(train, test, cutoff):
            base_ok = rs._usable(train) & (((train.index.to_numpy(np.int64) - phase) % rs.THIN) == 0)
            use = rs._usable(test)
            out = pd.DataFrame(index=test.index)
            xt = test.loc[use, cols].to_numpy(float)
            ref_t = test.loc[use, "s_freshPm25"].to_numpy(float)
            for head, target in heads.items():
                tr = train.loc[base_ok & np.isfinite(train[target].to_numpy(float))]
                model = HistGradientBoostingRegressor(**params).fit(tr[cols].to_numpy(float),
                                                                    (tr[target] - tr.s_freshPm25).to_numpy(float))
                pred = np.full(len(test), np.nan)
                pred[use] = ref_t + model.predict(xt)
                out[f"pred_{head}"] = pred
            tr = train.loc[base_ok & np.isfinite(train.y_ride_mean.to_numpy(float))]
            y = (tr.y_ride_mean <= config.PLANNING_CUTOFF).astype(int).to_numpy()
            p = np.full(len(test), np.nan)
            if len(set(y)) == 2:
                p[use] = HistGradientBoostingClassifier(**bparams).fit(tr[cols].to_numpy(float), y).predict_proba(xt)[:, 1]
            else:
                p[use] = float(y[0])
            out["p_ride_le70"] = p
            return out
        pred = walkforward.walk_forward(ds, fp, targets=("arrival", "ride_mean", "ride_le70"), lookback_days=lookback,
                                        embargo_minutes=330, days=days, log=lambda t: None)
    else:
        if s.get("_abl"):
            cols = rs.LOG28
        elif s.get("_net6"):
            cols = rs.LOG28 + [c for c in rs.SPATIAL_CORE if c.startswith("sp_net_")]
        else:
            cols = sets["lr28_sp"]
        C, clip = s.get("_C", 0.1), not s.get("_noclip")

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
        pred = walkforward.walk_forward(ds, fp, targets=("class5",), lookback_days=lookback, embargo_minutes=120,
                                        days=days, log=lambda t: None)
    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(OUT / f"pred_{name}.parquet")
    print(f"{name}: {len(pred)} rows {time.time() - started:.0f}s", flush=True)


def run_hpstats():
    """Classifier reruns (all 34 days, vs the matched-C ablation and 24-168 h placebos) and regressor reruns
    (final days, day-block bootstrap vs incumbent / persistence / matched ablation)."""
    warnings.simplefilter("ignore")
    pd.set_option("display.width", 300)
    ds = dataset.load()
    ev0, f0 = model_frames(ds)
    reg = f0["cand"][["pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max", "p_ride_le70"]]
    load = lambda n: pd.read_parquet(OUT / f"pred_{n}.parquet").reindex(ev0.index)
    mets = ["5-class log loss", "P(fall20) Brier", ">=20 argmax missed episodes", ">=20 argmax false clusters",
            ">=20 argmax episode errors", ">=20 argmax issue errors", ">=40 argmax missed episodes",
            ">=20 0.2 missed episodes", ">=20 0.2 episode errors"]
    rows = []
    for name, (kind, s) in HP.items():
        if kind != "cls" or s.get("_abl") or not (OUT / f"pred_{name}.parquet").exists():
            continue
        matched = "ca_" + name[2:]
        abl = load(matched) if (OUT / f"pred_{matched}.parquet").exists() else None
        ev, f = model_frames(ds, cand=pd.concat([load(name), reg], axis=1), abl_c=abl)
        L, M, E = build_losses(ev, f)
        all_days = sorted(ev.date_myt.astype(str).unique())
        for split in ("final Oct06-13", "all 34d"):
            num, den, k = day_vectors(ev, L, M, E, splits(all_days)[split])
            r = {"config": name, "split": split, "ablation": matched if abl is not None else "lr28 C=0.1"}
            for m in mets:
                r[m] = stat(num, den, k, m, "cand")
            for m in mets[:5]:
                r[f"{m} (abl)"] = stat(num, den, k, m, "abl")
            rows.append(r)
    cls = pd.DataFrame(rows)
    cls.to_csv(OUT / "cls_sensitivity.csv", index=False)
    print(cls.round(4).to_markdown(index=False))

    final = ev0.index >= FINAL
    regs = {n: load(n) for n, (kind, _) in HP.items() if kind == "reg" and (OUT / f"pred_{n}.parquet").exists()}
    mask = final & f0["inc"].pred_arrival.notna() & f0["inc"].pred_ride_mean.notna() & f0["cand"].pred_arrival.notna()
    mask &= f0["abl_r"].pred_arrival.notna()
    for p in regs.values():
        mask &= p.pred_arrival.notna() & p.pred_ride_mean.notna()
    mask &= ev0.y_arrival.notna() & ev0.y_ride_mean.notna() & ev0.y_ride_min.notna() & ev0.y_ride_max.notna()
    mp = mask & f0["inc"].p_ride_le70.notna()
    day = ev0.date_myt.astype(str)[mask]
    days = sorted(day.unique())
    dpos = day.map({d: i for i, d in enumerate(days)}).to_numpy()
    idx = np.random.default_rng(7).integers(0, len(days), (5000, len(days)))
    err = lambda fr, h: (fr[f"pred_{h}"] - ev0[f"y_{h}"]).abs()[mask].to_numpy(float)

    def boot(a, b):
        d, n = np.bincount(dpos, weights=a - b, minlength=len(days)), np.bincount(dpos, minlength=len(days))
        bs = d[idx].sum(1) / n[idx].sum(1)
        return f"{(a - b).mean():+.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]"

    rows = []
    base = {"incumbent": f0["inc"], "persistence": f0["persist"], "spatial_v1": f0["cand"], "ablation hgbr": f0["abl_r"]}
    for name, fr in list(base.items()) + list(regs.items()):
        r = {"config": name, **{f"{h} MAE": round(err(fr, h).mean(), 3) for h in ("arrival", "ride_mean", "ride_min", "ride_max")}}
        a = err(fr, "arrival")
        r["arrival vs inc"] = boot(a, err(f0["inc"], "arrival"))
        r["arrival vs persist"] = boot(a, err(f0["persist"], "arrival"))
        matched = regs.get("ra_" + name[2:]) if name.startswith("r_") else None
        r["arrival vs ablation"] = boot(a, err(matched if matched is not None else f0["abl_r"], "arrival")) + (
            " (matched)" if matched is not None else "")
        r["ride_mean vs inc"] = boot(err(fr, "ride_mean"), err(f0["inc"], "ride_mean"))
        r["P(ride<=70) Brier"] = round(float(((fr.p_ride_le70 - ev0.y_ride_le70) ** 2)[mp].mean()), 4)
        rows.append(r)
    regt = pd.DataFrame(rows)
    regt.to_csv(OUT / "reg_sensitivity_final.csv", index=False)
    print()
    print(f"final split, identical rows n={int(mask.sum())}, days={len(days)}")
    print(regt.to_markdown(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("configs", nargs="*")
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    if a.cmd == "stats":
        run_stats(a.B)
    elif a.cmd == "hp":
        for c in a.configs:
            run_hp(c, a.final)
    elif a.cmd == "hpstats":
        run_hpstats()
    elif a.cmd == "features":
        from spatial import spatial_features
        for lag in a.configs:
            OUT.mkdir(parents=True, exist_ok=True)
            spatial_features.build(dataset.load(), int(lag)).to_parquet(OUT / f"spatial_features_lag{lag}.parquet")
            print("features", lag, flush=True)
