"""Robustness review of the GBM-quantile `final` candidate against the incumbents.

Post-hoc on the saved walk-forward predictions (no refits):

    python -B gbm-quantile/robustness.py posthoc      # bootstrap CIs, jackknife, other splits, attribution, bands
    python -B gbm-quantile/robustness.py variants     # score the hyperparameter refits of robustness_variants.py

Rows are identical across models (where the incumbent replay has an output, as
evaluate.compare). Bootstrap: resample MYT days with replacement (and 3-day
circular blocks); a difference is candidate minus incumbent (negative = the
candidate has lower loss). Event errors are counted as the user does:
misses + false calls + wrong-way calls, per issue and per episode/cluster.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from harness import config as hc, report, scoring  # noqa: E402
import evaluate as E  # noqa: E402
import gbmq  # noqa: E402

ROB = hc.OUT / "gbm-quantile" / "robustness"
RNG = np.random.default_rng(20261014)
B = 4000
LOG_OFFSET = gbmq.LOG_OFFSET
ARR_EMB, RIDE_EMB = gbmq.ARRIVAL_EMBARGO_SECONDS, gbmq.RIDE_EMBARGO_SECONDS
CLASS_EDGES = (-40.0, -20.0, 20.0, 40.0)


def f(v, d=3):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------- baselines without features
def zero_feature_baseline(ds, ev):
    """Persistence scaled by the training window's empirical log-ratio distribution.

    Same training rows, embargo, target transform, class boundaries and
    climatology pool as the candidate, but no features: the predictive
    distribution of y given ref r is {(r + 5) exp(z_i) - 5} over the training
    log ratios z_i. Leak-free (labels complete before the day).
    """
    index = ds.index.to_numpy(np.int64)
    ref_all = ds.s_freshPm25.to_numpy(float)
    fresh = ds.m_fresh_available.to_numpy(bool) & np.isfinite(ref_all)
    arr_c = ds.y_arrival_complete.to_numpy(np.int64)
    ride_c = ds.y_ride_complete.to_numpy(np.int64)
    joint = ds.y_ride_joint_valid.astype(bool).to_numpy()
    out = pd.DataFrame(index=ev.index, dtype=float)
    cols = {}
    days = np.array([hc.midnight(i) for i in ev.index])
    ev_ref = ev.s_freshPm25.to_numpy(float)
    for d in np.unique(days):
        rows = np.flatnonzero(days == d)
        r = ev_ref[rows]
        win = (index > d - 28 * 86400) & fresh & (arr_c < d)
        arr = win & (index < d - ARR_EMB) & np.isfinite(ds.y_arrival.to_numpy(float))
        z = np.log((ds.y_arrival.to_numpy(float)[arr] + LOG_OFFSET) / (ref_all[arr] + LOG_OFFSET))
        delta = (r[:, None] + LOG_OFFSET) * (np.exp(z)[None, :] - 1)          # (n_test, n_train)
        bins = [delta <= -40, (delta > -40) & (delta <= -20), (delta > -20) & (delta < 20),
                (delta >= 20) & (delta < 40), delta >= 40]
        raw = np.stack([b.mean(axis=1) for b in bins], axis=1)
        lab = ds.y_class5.to_numpy(float)[arr]
        lab = lab[(lab >= 0) & (lab <= 4)]
        clim = np.bincount(lab.astype(int), minlength=5) / max(len(lab), 1)
        pooled = 0.7 * raw + 0.3 * clim
        cols.setdefault("pred_arrival", []).append((rows, (r + LOG_OFFSET) * np.exp(np.median(z)) - LOG_OFFSET))
        for k in range(5):
            cols.setdefault(f"p_class5_{k}", []).append((rows, pooled[:, k]))
            cols.setdefault(f"p_class5_raw_{k}", []).append((rows, raw[:, k]))
        ride = (win & (index < d - RIDE_EMB) & joint & (ride_c < d))
        for head, qs in (("ride_mean", (0.5,)), ("ride_min", (0.1, 0.5)), ("ride_max", (0.5, 0.9))):
            ok = ride & np.isfinite(ds[f"y_{head}"].to_numpy(float))
            zr = np.log((ds[f"y_{head}"].to_numpy(float)[ok] + LOG_OFFSET) / (ref_all[ok] + LOG_OFFSET))
            for q in qs:
                name = f"pred_{head}" if q == 0.5 else f"pred_{head}_q{int(q * 100)}"
                cols.setdefault(name, []).append((rows, (r + LOG_OFFSET) * np.exp(np.quantile(zr, q)) - LOG_OFFSET))
            if head == "ride_mean":
                ym = (r[:, None] + LOG_OFFSET) * np.exp(zr)[None, :] - LOG_OFFSET
                raw70 = (ym <= 70.0).mean(axis=1)
                clim70 = float(np.mean(ds.y_ride_mean.to_numpy(float)[ok] <= 70.0))
                cols.setdefault("p_ride_le70", []).append((rows, 0.95 * raw70 + 0.05 * clim70))
    for name, parts in cols.items():
        v = np.full(len(ev), np.nan)
        for rows, values in parts:
            v[rows] = values
        out[name] = v
    out["p_fall20"] = out.p_class5_0 + out.p_class5_1
    out["p_rise20"] = out.p_class5_3 + out.p_class5_4
    out["p_fall40"], out["p_rise40"] = out.p_class5_0, out.p_class5_4
    bad = ~np.isfinite(ev_ref)
    out.loc[bad] = np.nan
    return out


def prior_frames(ds, ev):
    prior = scoring.walk_forward_prior(ev, "y_class5", range(5), "y_arrival_complete", history=ds)
    prior.columns = [f"p_class5_{k}" for k in range(5)]
    prior["p_fall20"] = prior.p_class5_0 + prior.p_class5_1
    prior["p_rise20"] = prior.p_class5_3 + prior.p_class5_4
    prior["p_fall40"], prior["p_rise40"] = prior.p_class5_0, prior.p_class5_4
    p70 = scoring.walk_forward_prior(ev, "y_ride_le70", (0.0, 1.0), "y_ride_complete", history=ds)
    prior["p_ride_le70"] = p70["prior_1.0"]
    return prior


def pooled_incumbent(inc, prior, w_class, w70):
    out = inc.copy()
    P = (1 - w_class) * inc[[f"p_class5_{k}" for k in range(5)]].to_numpy(float) \
        + w_class * prior[[f"p_class5_{k}" for k in range(5)]].to_numpy(float)
    for k in range(5):
        out[f"p_class5_{k}"] = P[:, k]
    out["p_fall20"], out["p_rise20"] = P[:, 0] + P[:, 1], P[:, 3] + P[:, 4]
    out["p_fall40"], out["p_rise40"] = P[:, 0], P[:, 4]
    out["p_ride_le70"] = (1 - w70) * inc.p_ride_le70 + w70 * prior.p_ride_le70
    return out


# ----------------------------------------------------------------------------- per-row losses
POINT = (("arr_mae", "pred_arrival", "y_arrival"), ("ridemean_mae", "pred_ride_mean", "y_ride_mean"),
         ("ridemin_mae", "pred_ride_min", "y_ride_min"), ("ridemax_mae", "pred_ride_max", "y_ride_max"))
PROB = ("fall20", "rise20", "fall40", "rise40")


def losses(ev, inc, model):
    """Per-row loss columns for ``model`` on the identical-row masks of evaluate.compare (NaN outside)."""
    out = pd.DataFrame(index=ev.index)
    for key, col, y in POINT:
        if col not in model:
            continue
        m = inc[col].notna() & model[col].notna() & ev[y].notna()
        out[key] = (model[col] - ev[y]).abs().where(m)
        if key == "arr_mae":
            out["arr_se"] = ((model[col] - ev[y]) ** 2).where(m)
    for key, col, y, tau in (("ridemin_pin10", "pred_ride_min_q10", "y_ride_min", 0.1),
                             ("ridemax_pin90", "pred_ride_max_q90", "y_ride_max", 0.9),
                             ("ridemin_pin10_pt", "pred_ride_min", "y_ride_min", 0.1),
                             ("ridemax_pin90_pt", "pred_ride_max", "y_ride_max", 0.9)):
        if col in model:
            base = col.replace("_q10", "").replace("_q90", "")
            m = inc[base].notna() & model[col].notna() & ev[y].notna()
            diff = ev[y] - model[col]
            out[key] = np.maximum(tau * diff, (tau - 1) * diff).where(m)
    if "p_fall20" in model:
        m = inc.p_fall20.notna() & model.p_fall20.notna() & ev.y_delta.notna()
        for s in PROB:
            out[f"brier_{s}"] = ((model[f"p_{s}"] - ev[f"y_{s}"]) ** 2).where(m)
        if all(f"p_class5_{k}" in model for k in range(5)):
            P = model[[f"p_class5_{k}" for k in range(5)]].to_numpy(float)
            y = ev.y_class5.to_numpy(float)
            ok = m.to_numpy(bool) & np.isin(y, range(5))
            onehot = (y[:, None] == np.arange(5)[None, :]).astype(float)
            out["c5_brier"] = np.where(ok, ((P - onehot) ** 2).sum(axis=1), np.nan)
            out["c5_logloss"] = np.where(ok, -np.log(np.clip((P * onehot).sum(axis=1), 1e-15, 1)), np.nan)
    if "p_ride_le70" in model:
        m = inc.p_ride_le70.notna() & model.p_ride_le70.notna() & ev.y_ride_le70.notna()
        p = model.p_ride_le70.clip(1e-15, 1 - 1e-15)
        out["le70_brier"] = ((model.p_ride_le70 - ev.y_ride_le70) ** 2).where(m)
        out["le70_logloss"] = (-(ev.y_ride_le70 * np.log(p) + (1 - ev.y_ride_le70) * np.log(1 - p))).where(m)
    return out


# ----------------------------------------------------------------------------- events
def event_units(ev, model, threshold, rule, mask):
    """Per-issue event outcome and episode/cluster units with the day they start.

    Returns (issue frame, unit frame). Units: true episodes (caught or missed),
    false-call clusters and wrong-way clusters, as scoring.event_confusion.
    """
    sub = ev.loc[mask]
    mm = model.reindex(sub.index)
    truth = scoring.truth_direction(sub.y_delta, threshold)
    if rule == "point":
        call = scoring.calls_from_point(mm.pred_arrival, sub.y_ref, threshold)
    else:
        call = scoring.calls_from_probs(mm[f"p_fall{threshold}"], mm[f"p_rise{threshold}"], rule)
    ok = np.isfinite(truth) & np.isfinite(call)
    issues = sub.index.to_numpy(np.int64)[ok]
    truth, call = truth[ok], call[ok]
    event = truth != 0
    issue = pd.DataFrame({"hit": event & (call == truth), "miss": event & (call == 0),
                          "wrong": event & (call == -truth), "false": ~event & (call != 0)}, index=issues)
    lookup = dict(zip(issues, call))
    units = []
    for direction in (-1.0, 1.0):
        for run in scoring._runs(issues, truth == direction):
            caught = any(lookup[e] == direction for e in run)
            units.append({"start": int(run[0]), "kind": "episode", "caught": caught})
    for run in scoring._runs(issues, (~event) & (call != 0)):
        units.append({"start": int(run[0]), "kind": "false_cluster", "caught": False})
    for run in scoring._runs(issues, event & (call == -truth)):
        units.append({"start": int(run[0]), "kind": "wrong_cluster", "caught": False})
    return issue, pd.DataFrame(units)


def event_day_stats(ev, model, threshold, rule, mask):
    issue, units = event_units(ev, model, threshold, rule, mask)
    issue["day"] = [hc.midnight(i) for i in issue.index]
    g = issue.groupby("day")[["hit", "miss", "wrong", "false"]].sum()
    if len(units):
        units["day"] = [hc.midnight(i) for i in units.start]
        ep = units[units.kind == "episode"]
        g["episodes"] = ep.groupby("day").size()
        g["caught"] = ep[ep.caught].groupby("day").size()
        g["false_cl"] = units[units.kind == "false_cluster"].groupby("day").size()
        g["wrong_cl"] = units[units.kind == "wrong_cluster"].groupby("day").size()
    g = g.fillna(0)
    for c in ("episodes", "caught", "false_cl", "wrong_cl"):
        if c not in g:
            g[c] = 0
    g["missed_ep"] = g.episodes - g.caught
    g["issue_errors"] = g.miss + g.wrong + g["false"]
    g["unit_errors"] = g.missed_ep + g.false_cl + g.wrong_cl
    return g


# ----------------------------------------------------------------------------- bootstrap
def day_groups(frame):
    days = np.array([hc.midnight(i) for i in frame.index])
    return days


def boot_indices(n_days, block=1):
    """B resamples of day positions (circular blocks of ``block`` consecutive days)."""
    if block == 1:
        return RNG.integers(0, n_days, size=(B, n_days))
    starts = RNG.integers(0, n_days, size=(B, int(np.ceil(n_days / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_days
    return idx.reshape(B, -1)[:, :n_days]


def mean_diff_boot(la, lb, days, block=1):
    """Bootstrap of mean(la) - mean(lb) over rows where both are finite, resampling days."""
    ok = np.isfinite(la) & np.isfinite(lb)
    la, lb, d = la[ok], lb[ok], days[ok]
    uniq = np.unique(d)
    pos = np.searchsorted(uniq, d)
    sa = np.bincount(pos, la, len(uniq))
    sb = np.bincount(pos, lb, len(uniq))
    n = np.bincount(pos, None, len(uniq)).astype(float)
    idx = boot_indices(len(uniq), block)
    diff = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    point = (sa.sum() - sb.sum()) / n.sum()
    return point, np.percentile(diff, [2.5, 97.5]), float((diff < 0).mean()), len(uniq), int(n.sum())


def auc_diff_boot(y, pa, pb, days, reps=1000):
    ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb, d = y[ok], pa[ok], pb[ok], days[ok]
    uniq = np.unique(d)
    rows = [np.flatnonzero(d == u) for u in uniq]
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    out = []
    for r in range(reps):
        pick = np.concatenate([rows[i] for i in RNG.integers(0, len(uniq), len(uniq))])
        if 0 < y[pick].sum() < len(pick):
            out.append(roc_auc_score(y[pick], pa[pick]) - roc_auc_score(y[pick], pb[pick]))
    out = np.array(out)
    return point, np.percentile(out, [2.5, 97.5]), float((out > 0).mean())


def count_diff_boot(ga, gb, column, block=1):
    days = sorted(set(ga.index) | set(gb.index))
    a = ga[column].reindex(days).fillna(0).to_numpy(float)
    b = gb[column].reindex(days).fillna(0).to_numpy(float)
    idx = boot_indices(len(days), block)
    diff = a[idx].sum(1) - b[idx].sum(1)
    return a.sum() - b.sum(), np.percentile(diff, [2.5, 97.5]), float((diff > 0).mean()), float((diff < 0).mean())


# ----------------------------------------------------------------------------- report sections
LOSS_ROWS = [("+90 MAE", "arr_mae"), ("+90 MSE", "arr_se"), ("ride mean MAE", "ridemean_mae"),
             ("ride min MAE", "ridemin_mae"), ("ride max MAE", "ridemax_mae"),
             ("P(fall>=20) Brier", "brier_fall20"), ("P(rise>=20) Brier", "brier_rise20"),
             ("P(fall>=40) Brier", "brier_fall40"), ("P(rise>=40) Brier", "brier_rise40"),
             ("5-class Brier", "c5_brier"), ("5-class log loss", "c5_logloss"),
             ("P(ride mean<=70) Brier", "le70_brier"), ("P(ride mean<=70) log loss", "le70_logloss")]


def bootstrap_section(L_c, L_ref, label, split_rows):
    days = day_groups(L_c)
    rows = []
    for title, key in LOSS_ROWS:
        if key not in L_c or key not in L_ref:
            continue
        a, b = L_c[key].to_numpy(float), L_ref[key].to_numpy(float)
        p1, ci1, pr1, nd, n = mean_diff_boot(a, b, days, 1)
        p3, ci3, pr3, *_ = mean_diff_boot(a, b, days, 3)
        ma, mb = np.nanmean(np.where(np.isfinite(b), a, np.nan)), np.nanmean(np.where(np.isfinite(a), b, np.nan))
        d = 4 if "Brier" in title or "log" in title else 2
        rows.append((title, f(ma, d), f(mb, d), f(p1, d + 1 if d == 4 else 3), f"[{f(ci1[0], d)}, {f(ci1[1], d)}]",
                     f"{pr1:.2f}", f"[{f(ci3[0], d)}, {f(ci3[1], d)}]", f"{pr3:.2f}", nd, n))
    print(f"\n### Day-block bootstrap: candidate vs {label} ({split_rows})\n")
    print(table(["Metric", "cand", label, "diff", "95% CI (1-day blocks)", "P(cand better)",
                 "95% CI (3-day blocks)", "P(cand better, 3d)", "days", "n"], rows))


def event_section(ev, cand, inc, mask_p, mask_a, label):
    rows = []
    stats = {}
    for t in (20, 40):
        for rule, src_c, src_i, m in (("argmax", cand, inc, mask_p), ("0.2", cand, inc, mask_p),
                                      ("point", cand, inc, mask_a)):
            gc = event_day_stats(ev, src_c, t, rule, m)
            gi = event_day_stats(ev, src_i, t, rule, m)
            stats[(t, rule)] = (gc, gi)
            for col, nice in (("caught", "episodes caught"), ("unit_errors", "errors: missed ep + false cl + wrong cl"),
                              ("issue_errors", "errors: miss + false + wrong issues")):
                pt, ci, p_gt, p_lt = count_diff_boot(gc, gi, col)
                better = p_gt if col == "caught" else p_lt
                rows.append((f">={t} {rule}", nice, int(gc[col].sum()), int(gi[col].sum()), f"{pt:+.0f}",
                             f"[{ci[0]:+.0f}, {ci[1]:+.0f}]", f"{better:.2f}"))
    print(f"\n### Event errors (user's count) with day-block bootstrap: candidate vs {label}\n")
    print("Episodes are assigned to the day they start. 'P(cand better)' = share of resamples where the candidate"
          " catches more episodes / makes fewer errors.\n")
    print(table(["Events", "Count", "cand", label, "diff", "95% CI", "P(cand better)"], rows))
    return stats


def auc_section(ev, cand, inc, mask_p, label):
    days = day_groups(ev)
    rows = []
    for s in PROB:
        y = ev[f"y_{s}"].to_numpy(float)
        m = mask_p.to_numpy(bool)
        pa = np.where(m, cand[f"p_{s}"].reindex(ev.index).to_numpy(float), np.nan)
        pb = np.where(m, inc[f"p_{s}"].reindex(ev.index).to_numpy(float), np.nan)
        if np.nansum(y[m]) < 5:
            continue
        p, ci, pr = auc_diff_boot(y, pa, pb, days)
        rows.append((f"P({s})", f(p), f"[{f(ci[0])}, {f(ci[1])}]", f"{pr:.2f}"))
    print(f"\n### ROC AUC difference (candidate - {label}), day-block bootstrap\n")
    print(table(["Probability", "AUC diff", "95% CI", "P(cand better)"], rows))


def jackknife_section(L_c, L_i, stats, label):
    days = day_groups(L_c)
    uniq = np.unique(days)
    keys = [("+90 MAE", "arr_mae"), ("ride mean MAE", "ridemean_mae"), ("ride min MAE", "ridemin_mae"),
            ("P(fall>=20) Brier", "brier_fall20"), ("5-class Brier", "c5_brier"), ("5-class log loss", "c5_logloss"),
            ("P(<=70) Brier", "le70_brier"), ("P(<=70) log loss", "le70_logloss")]
    case_days = {}
    for name, start, end, direction in report.CASES:
        d = hc.midnight(hc.myt_epoch(start))
        if uniq.min() <= d <= uniq.max():
            case_days.setdefault(d, []).append(name)
    rows = []
    for title, key in keys:
        a, b = L_c[key].to_numpy(float), L_i[key].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        full = a[ok].mean() - b[ok].mean()
        loo = {}
        for u in uniq:
            k = ok & (days != u)
            loo[u] = a[k].mean() - b[k].mean()
        vals = np.array(list(loo.values()))
        flips = int(np.sum(np.sign(vals) != np.sign(full)))
        worst_day = max(loo, key=lambda u: loo[u] * np.sign(-full))
        per_day = pd.Series(a[ok] - b[ok]).groupby(days[ok]).sum()
        top = per_day.sort_values().iloc[0] if full < 0 else per_day.sort_values().iloc[-1]
        top_share = top / (a[ok] - b[ok]).sum()
        case_vals = "; ".join(f"{'/'.join(n.split(',')[0] for n in names)} {loo[d]:+.4f}" for d, names in case_days.items() if d in loo)
        rows.append((title, f"{full:+.4f}", f"[{vals.min():+.4f}, {vals.max():+.4f}]", flips,
                     f"{hc.myt(worst_day):%m-%d} ({loo[worst_day]:+.4f})", f"{top_share:.2f}", case_vals))
    print(f"\n### Leave-one-day-out and leave-one-big-event-day-out: candidate - {label}\n")
    print("The case columns drop the whole MYT day of each big event. 'top-day share' = the single best"
          " (for the sign of the full difference) day's share of the total summed difference.\n")
    print(table(["Metric", "full diff", "LODO range", "LODO sign flips", "worst-case day dropped",
                 "top-day share", "drop each big-event day"], rows))
    # events: leave-one-big-event-day-out for >=20 at argmax and 0.2
    rows = []
    for (t, rule), (gc, gi) in stats.items():
        if t != 20 or rule == "point":
            continue
        for d, names in case_days.items():
            c = gc.drop(index=d, errors="ignore")
            i = gi.drop(index=d, errors="ignore")
            rows.append((f">=20 {rule}", "/".join(names), f"{int(c.caught.sum())}/{int(c.episodes.sum())}",
                         f"{int(i.caught.sum())}/{int(i.episodes.sum())}", int(c.unit_errors.sum()), int(i.unit_errors.sum())))
    print(f"\n#### Events with each big-event day dropped (candidate vs {label})\n")
    print(table(["Events", "dropped", "cand caught/ep", f"{label} caught/ep", "cand unit errors",
                 f"{label} unit errors"], rows))


def subperiod_section(L_c, L_i, label):
    days = day_groups(L_c)
    periods = [("tune half 09-10..09-26 (selection in-sample)", "2026-09-10", "2026-09-27"),
               ("test half 09-27..10-13", "2026-09-27", "2026-10-14"),
               ("wk 09-10..09-16", "2026-09-10", "2026-09-17"), ("wk 09-17..09-23", "2026-09-17", "2026-09-24"),
               ("wk 09-24..09-30", "2026-09-24", "2026-10-01"), ("wk 10-01..10-07", "2026-10-01", "2026-10-08"),
               ("10-08..10-13", "2026-10-08", "2026-10-14")]
    keys = [("+90 MAE", "arr_mae"), ("ride mean MAE", "ridemean_mae"), ("ride min MAE", "ridemin_mae"),
            ("ride max MAE", "ridemax_mae"), ("P(fall20) Brier", "brier_fall20"), ("5-cl Brier", "c5_brier"),
            ("5-cl logloss", "c5_logloss"), ("P(<=70) Brier", "le70_brier")]
    rows = []
    for name, a, b in periods:
        lo, hi = hc.myt_epoch(a), hc.myt_epoch(b)
        sel = (days >= lo) & (days < hi)
        row = [name]
        for title, key in keys:
            x, y = L_c[key].to_numpy(float)[sel], L_i[key].to_numpy(float)[sel]
            ok = np.isfinite(x) & np.isfinite(y)
            row.append(f"{x[ok].mean() - y[ok].mean():+.4f}" if ok.any() else "–")
        rows.append(row)
    odd = np.array([hc.myt(d).day % 2 == 1 for d in days])
    for name, sel in (("odd days (all)", odd), ("even days (all)", ~odd)):
        row = [name]
        for title, key in keys:
            x, y = L_c[key].to_numpy(float)[sel], L_i[key].to_numpy(float)[sel]
            ok = np.isfinite(x) & np.isfinite(y)
            row.append(f"{x[ok].mean() - y[ok].mean():+.4f}")
        rows.append(row)
    print(f"\n### Other splits: candidate - {label} (negative = candidate better)\n")
    print(table(["Period"] + [k[0] for k in keys], rows))


def events_by_period(ev, cand, inc, mask_p, label):
    rows = []
    periods = [("tune half", "2026-09-10", "2026-09-27"), ("test half", "2026-09-27", "2026-10-14"),
               ("09-10..09-16", "2026-09-10", "2026-09-17"), ("09-17..09-23", "2026-09-17", "2026-09-24"),
               ("09-24..09-30", "2026-09-24", "2026-10-01"), ("10-01..10-07", "2026-10-01", "2026-10-08"),
               ("10-08..10-13", "2026-10-08", "2026-10-14")]
    idx = ev.index.to_numpy(np.int64)
    for name, a, b in periods:
        sel = mask_p & pd.Series((idx >= hc.myt_epoch(a)) & (idx < hc.myt_epoch(b)), index=ev.index)
        row = [name]
        for rule in ("argmax", "0.2"):
            gc = event_day_stats(ev, cand, 20, rule, sel)
            gi = event_day_stats(ev, inc, 20, rule, sel)
            row.append(f"{int(gc.caught.sum())}/{int(gc.episodes.sum())} ({int(gc.unit_errors.sum())}) | "
                       f"{int(gi.caught.sum())}/{int(gi.episodes.sum())} ({int(gi.unit_errors.sum())})")
        rows.append(row)
    print(f"\n### >=20 events by period: cand | {label} = caught/episodes (unit errors)\n")
    print(table(["Period", "argmax", "p>=0.2"], rows))


def posthoc():
    ds = E.load_ds()
    ev = E.evaluation_frame("all")
    inc = report.standard_replay(ds).reindex(ev.index)
    cand = pd.read_parquet(E.pred_path("final", "all")).reindex(ev.index)
    prior = prior_frames(ds, ev)
    per = scoring.persistence_predictions(ev)
    zero = zero_feature_baseline(ds, ev)
    idx = ev.index.to_numpy(np.int64)
    test = pd.Series(idx >= hc.myt_epoch("2026-09-27"), index=ev.index)
    tune = ~test

    L_c, L_i = losses(ev, inc, cand), losses(ev, inc, inc)
    # persistence for points and the 0/1 call for P(<=70); the walk-forward class prior for arrival probabilities
    L_p = losses(ev, inc, per.reindex(ev.index).assign(**{c: prior[c] for c in prior.columns if c != "p_ride_le70"}))
    L_z = losses(ev, inc, zero)
    # incumbent pooled with the walk-forward prior, weights tuned on the tune half (lowest log loss)
    scan = []
    for w in (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        Lw = losses(ev, inc, pooled_incumbent(inc, prior, w, w))
        scan.append((w, Lw.c5_logloss[tune].mean(), Lw.le70_logloss[tune].mean(),
                     Lw.c5_logloss[test].mean(), Lw.le70_logloss[test].mean(), Lw.brier_fall20[test].mean(),
                     Lw.c5_brier[test].mean(), Lw.le70_brier[test].mean()))
    scan = pd.DataFrame(scan, columns=["w", "tune_c5_ll", "tune_le70_ll", "test_c5_ll", "test_le70_ll",
                                       "test_brier_fall20", "test_c5_brier", "test_le70_brier"])
    w_c5 = float(scan.loc[scan.tune_c5_ll.idxmin(), "w"])
    w_70 = float(scan.loc[scan.tune_le70_ll.idxmin(), "w"])
    inc_pool = pooled_incumbent(inc, prior, w_c5, w_70)
    L_ip = losses(ev, inc, inc_pool)

    print("# GBM-quantile `final`: robustness review (post hoc on saved walk-forward predictions)\n")
    print(f"Rows: grid evaluation issues with a fresh reference, 2026-09-10..10-13; identical-row masks as evaluate.compare. B={B}.")
    print("\n## 1. Test half (2026-09-27..10-13), the split scored once")
    Lt = {k: v.loc[test] for k, v in (("c", L_c), ("i", L_i), ("p", L_p), ("z", L_z), ("ip", L_ip))}
    bootstrap_section(Lt["c"], Lt["i"], "incumbent", "test")
    bootstrap_section(Lt["c"], Lt["p"], "persistence/prior", "test; prior for probabilities, 0/1 for P(<=70)")
    mask_p = inc.p_fall20.notna() & cand.p_fall20.notna() & ev.y_delta.notna()
    mask_a = inc.pred_arrival.notna() & cand.pred_arrival.notna() & ev.y_arrival.notna()
    evt = ev.loc[test]
    stats = event_section(evt, cand, inc, mask_p[test], mask_a[test], "incumbent")
    auc_section(evt, cand.loc[test], inc.loc[test], mask_p[test], "incumbent")
    jackknife_section(Lt["c"], Lt["i"], stats, "incumbent")

    print("\n## 2. Other splits (walk-forward predictions over the full period; the tune half was used for selection)")
    subperiod_section(L_c, L_i, "incumbent")
    events_by_period(ev, cand, inc, mask_p, "incumbent")
    print("\nFull period (09-10..10-13) bootstrap:")
    bootstrap_section(L_c, L_i, "incumbent", "full period, partly in-sample for selection")

    print("\n## 3. Attribution: is the probability gain the model or the climatology pool?")
    print("\nIncumbent probabilities pooled linearly with the walk-forward prior, weight chosen on the tune half by lowest"
          " log loss (5-class for the logistic, P(<=70) for the cycling logistic):\n")
    print(table(list(scan.columns), [[f(v, 4) for v in r] for r in scan.itertuples(index=False)]))
    print(f"\nChosen on tune: w_class5={w_c5}, w_le70={w_70}.")
    bootstrap_section(Lt["c"], Lt["ip"], "incumbent pooled", "test")
    print("\nZero-feature baseline: persistence scaled by the training window's empirical log-ratio distribution,"
          " same rows, embargo, CDF read-out and pooling (0.3 / 0.05) as the candidate, no features:")
    bootstrap_section(Lt["c"], Lt["z"], "zero-feature log-ratio", "test")
    bootstrap_section(Lt["z"], Lt["i"], "incumbent [zero-feature baseline vs incumbent]", "test")
    # bands
    rows = []
    for title, key in (("ride min pinball q0.1", "ridemin_pin10"), ("ride max pinball q0.9", "ridemax_pin90")):
        for name, L in (("candidate q-head", Lt["c"]), ("zero-feature empirical quantile", Lt["z"])):
            rows.append((title, name, f(L[key].mean(), 2)))
        pt = "ridemin_pin10_pt" if "min" in key else "ridemax_pin90_pt"
        rows.append((title, "Ridge point (as in the candidate's table)", f(Lt["i"][pt].mean(), 2)))
        rows.append((title, "persistence point", f(Lt["p"][pt].mean(), 2)))
    print("\n### Quantile bands vs fair quantile baselines (test)\n")
    print(table(["Target", "Model", "pinball"], rows))
    cov = []
    for name, col, y in (("candidate q0.1 min", cand.pred_ride_min_q10, ev.y_ride_min),
                         ("zero-feature q0.1 min", zero.pred_ride_min_q10, ev.y_ride_min),
                         ("candidate q0.9 max", cand.pred_ride_max_q90, ev.y_ride_max),
                         ("zero-feature q0.9 max", zero.pred_ride_max_q90, ev.y_ride_max)):
        m = test & col.notna() & y.notna() & inc.pred_ride_min.notna()
        cov.append((name, f((y[m] <= col[m]).mean())))
    print("\n" + table(["Band", "coverage P(y <= q)"], cov))
    b = []
    for key, title in (("ridemin_pin10", "ride min pinball q0.1"), ("ridemax_pin90", "ride max pinball q0.9")):
        p, ci, pr, nd, n = mean_diff_boot(Lt["c"][key].to_numpy(float), Lt["z"][key].to_numpy(float), day_groups(Lt["c"]))
        b.append((title, f(p), f"[{f(ci[0])}, {f(ci[1])}]", f"{pr:.2f}"))
    print("\n" + table(["Band: candidate - zero-feature", "diff", "95% CI", "P(cand better)"], b))

    print("\n## 4. Pool-weight sensitivity of the candidate on test (post hoc, same fits)")
    rows = []
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        pw = E.pool(cand.loc[test], E.CONFIGS["final"], w, min(w, 0.05) if w > 0 else 0.0)
        Lw = losses(evt, inc.loc[test], pw)
        gc = event_day_stats(evt, pw, 20, "argmax", mask_p[test])
        g2 = event_day_stats(evt, pw, 20, "0.2", mask_p[test])
        rows.append((w, f(Lw.c5_brier.mean(), 4), f(Lw.c5_logloss.mean(), 4), f(Lw.brier_fall20.mean(), 4),
                     f(Lw.brier_rise20.mean(), 4), f"{int(gc.caught.sum())}/{int(gc.episodes.sum())} ({int(gc.unit_errors.sum())})",
                     f"{int(g2.caught.sum())}/{int(g2.episodes.sum())} ({int(g2.unit_errors.sum())})"))
    print("\n" + table(["w_class5", "5-cl Brier", "5-cl logloss", "Brier fall20", "Brier rise20",
                        ">=20 argmax caught (unit err)", ">=20 p>=0.2 caught (unit err)"], rows))
    print(f"\nIncumbent on the same rows: 5-cl Brier {f(Lt['i'].c5_brier.mean(), 4)}, logloss {f(Lt['i'].c5_logloss.mean(), 4)},"
          f" Brier fall20 {f(Lt['i'].brier_fall20.mean(), 4)}, rise20 {f(Lt['i'].brier_rise20.mean(), 4)}")
    ROB.mkdir(parents=True, exist_ok=True)
    zero.to_parquet(ROB / "zero_feature__all.parquet")


# ----------------------------------------------------------------------------- hyperparameter variants
def variants():
    ds = E.load_ds()
    ev = E.evaluation_frame("test")
    inc = report.standard_replay(ds).reindex(ev.index)
    final = pd.read_parquet(E.pred_path("final", "test")).reindex(ev.index)
    L_i = losses(ev, inc, inc)
    mask_p = inc.p_fall20.notna() & final.p_fall20.notna() & ev.y_delta.notna()
    rows = []
    specs = [("final (reference)", final)]
    for path in sorted(ROB.glob("v_*__test.parquet")):
        specs.append((path.name.split("__")[0], pd.read_parquet(path).reindex(ev.index)))
    for name, pred in specs:
        L = losses(ev, inc, pred)
        days = day_groups(L)
        cells = [name]
        for key in ("arr_mae", "ridemean_mae", "brier_fall20", "c5_brier", "c5_logloss", "le70_brier"):
            p, ci, pr, *_ = mean_diff_boot(L[key].to_numpy(float), L_i[key].to_numpy(float), days)
            d = 2 if "mae" in key else 4
            cells.append(f"{f(np.nanmean(np.where(np.isfinite(L_i[key]), L[key], np.nan)), d)} ({pr:.2f})")
        m = mask_p & pred.p_fall20.notna()
        a = event_day_stats(ev, pred, 20, "argmax", m)
        b = event_day_stats(ev, pred, 20, "0.2", m)
        cells += [f"{int(a.caught.sum())}/{int(a.episodes.sum())} ({int(a.unit_errors.sum())})",
                  f"{int(b.caught.sum())}/{int(b.episodes.sum())} ({int(b.unit_errors.sum())})"]
        y = ev.y_fall20.to_numpy(float)
        ok = m.to_numpy(bool) & np.isfinite(y)
        cells.append(f(roc_auc_score(y[ok], pred.p_fall20.to_numpy(float)[ok])))
        rows.append(cells)
    a = event_day_stats(ev, inc, 20, "argmax", mask_p)
    b = event_day_stats(ev, inc, 20, "0.2", mask_p)
    y = ev.y_fall20.to_numpy(float)
    ok = mask_p.to_numpy(bool)
    rows.append(["incumbent", f(L_i.arr_mae.mean(), 2), f(L_i.ridemean_mae.mean(), 2), f(L_i.brier_fall20.mean(), 4),
                 f(L_i.c5_brier.mean(), 4), f(L_i.c5_logloss.mean(), 4), f(L_i.le70_brier.mean(), 4),
                 f"{int(a.caught.sum())}/{int(a.episodes.sum())} ({int(a.unit_errors.sum())})",
                 f"{int(b.caught.sum())}/{int(b.episodes.sum())} ({int(b.unit_errors.sum())})",
                 f(roc_auc_score(y[ok], inc.p_fall20.to_numpy(float)[ok]))])
    print("## Hyperparameter sensitivity on test (arrival + ride-mean heads refit; value (P(better than incumbent) by day bootstrap))\n")
    print(table(["Variant", "+90 MAE", "ride mean MAE", "Brier fall20", "5-cl Brier", "5-cl logloss", "P(<=70) Brier",
                 ">=20 argmax caught (unit err)", ">=20 p>=0.2 caught (unit err)", "AUC fall20"], rows))


if __name__ == "__main__":
    {"posthoc": posthoc, "variants": variants}[sys.argv[1]]()
