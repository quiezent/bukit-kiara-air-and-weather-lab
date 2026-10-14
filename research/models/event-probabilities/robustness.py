"""Robustness check (skeptic) of the frozen candidate stack14tb.

Usage: python -B robustness.py [bootstrap|events|select|hyper|all]

* bootstrap: day-block (1-day and circular 3-day) bootstrap CIs of per-row
  Brier / log-loss differences candidate - incumbent and candidate - prior;
  leave-one-day-out and leave-one-big-event-day-out; weekly sign consistency.
* events: per-day event confusion for the fixed rules, bootstrap of episode and
  error-count differences, matched-false-call bootstrap, which episodes each
  model catches.
* select: every tune variant recomputed on all days and scored on the holdout
  (selection luck), and a reversed split (choose on holdout, score on tune).
* hyper: calibrator lookback / ridge and alternative HGB settings (needs
  robustness_base.py outputs).

Writes data\\models\\event-probabilities\\robust_<section>.json and prints Markdown.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import sys

import numpy as np
import pandas as pd

import calibrate as cal
import common
import evaluate as E
from harness import config, report, scoring

B = 4000
RNG = np.random.default_rng(20261014)
METRICS = ["brier_fall20", "brier_rise20", "brier_fall40", "brier_rise40", "brier_class5", "logloss_class5",
           "brier_first20", "logloss_first20"]
BIG_DAYS = ["2026-09-28", "2026-09-29", "2026-10-01", "2026-10-06", "2026-10-08", "2026-10-09", "2026-10-13"]


# ----------------------------------------------------------------------------- setup
def setup():
    ds = common.load()
    cand = common.load_pred("candidate")
    inc = E.incumbent_frame(ds)
    prior = E.prior_frame(ds)
    mask = E.comparison_mask(ds, {"candidate": cand})
    return ds, cand, inc, prior, mask


def rows(ds, mask, period):
    a, b = E.PERIODS[period]
    return ds.loc[mask & (ds.index >= a) & (ds.index < b)]


def losses(frame, pred) -> dict:
    p = pred.reindex(frame.index)
    out = {}
    for tail in ("fall20", "rise20", "fall40", "rise40"):
        out[f"brier_{tail}"] = (p[f"p_{tail}"].to_numpy(float) - frame[f"y_{tail}"].to_numpy(float)) ** 2
    for name, cols, label, classes in (("class5", common.CLASS5, "y_class5", (0, 1, 2, 3, 4)),
                                       ("first20", common.FIRST3, "y_first20", (-1, 0, 1))):
        y = frame[label].to_numpy(float)
        P = p[cols].to_numpy(float)
        ok = np.isin(y, classes) & np.isfinite(P).all(axis=1)
        onehot = (y[:, None] == np.asarray(classes, float)[None, :]).astype(float)
        brier = np.sum((P - onehot) ** 2, axis=1)
        ll = -np.log(np.clip(np.sum(P * onehot, axis=1), 1e-15, 1))
        brier[~ok], ll[~ok] = np.nan, np.nan
        out[f"brier_{name}"], out[f"logloss_{name}"] = brier, ll
    return out


def day_of(frame):
    return np.array([config.midnight(i) for i in frame.index.to_numpy(np.int64)])


def resample_index(n_days, block, b=B):
    if block == 1:
        return RNG.integers(0, n_days, (b, n_days))
    k = int(np.ceil(n_days / block))
    starts = RNG.integers(0, n_days, (b, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_days
    return idx.reshape(b, -1)[:, :n_days]


def boot_diff(la, lb, days, block=1):
    ok = np.isfinite(la) & np.isfinite(lb)
    uniq, inv = np.unique(days[ok], return_inverse=True)
    sa = np.bincount(inv, la[ok], len(uniq))
    sb = np.bincount(inv, lb[ok], len(uniq))
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    point = (sa.sum() - sb.sum()) / n.sum()
    idx = resample_index(len(uniq), block)
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    return {"diff": float(point), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975)),
            "p_better": float(np.mean(d < 0)), "n_days": int(len(uniq)),
            "a": float(sa.sum() / n.sum()), "b": float(sb.sum() / n.sum())}


def fmt_ci(r, digits=4):
    return f"{r['diff']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}] P(better)={r['p_better']:.2f}"


# ----------------------------------------------------------------------------- bootstrap section
def section_bootstrap(ds, cand, inc, prior, mask):
    res, lines = {}, []
    for period in ("holdout", "full", "tune", "issued_era"):
        frame = rows(ds, mask, period)
        days = day_of(frame)
        lc, li, lp = losses(frame, cand), losses(frame, inc), losses(frame, prior)
        res[period] = {}
        lines += [f"### {period} (n={len(frame)}, days={len(np.unique(days))})", "",
                  "| metric | cand | inc | prior | cand-inc 1-day [95% CI] | cand-inc 3-day block | cand-prior 1-day | cand-prior 3-day block |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in METRICS:
            r1 = boot_diff(lc[m], li[m], days, 1)
            r3 = boot_diff(lc[m], li[m], days, 3)
            q1 = boot_diff(lc[m], lp[m], days, 1)
            q3 = boot_diff(lc[m], lp[m], days, 3)
            res[period][m] = {"vs_inc_1d": r1, "vs_inc_3d": r3, "vs_prior_1d": q1, "vs_prior_3d": q3}
            lines.append(f"| {m} | {r1['a']:.4f} | {r1['b']:.4f} | {q1['b']:.4f} | {fmt_ci(r1)} | {fmt_ci(r3)} | "
                         f"{fmt_ci(q1)} | {fmt_ci(q3)} |")
        lines.append("")
    # leave-one-day-out on the holdout
    frame = rows(ds, mask, "holdout")
    days = day_of(frame)
    lc, li, lp = losses(frame, cand), losses(frame, inc), losses(frame, prior)
    uniq = np.unique(days)
    loo = {}
    lines += ["### Holdout leave-one-day-out (diff cand-inc / cand-prior after dropping the day)", "",
              "| dropped day | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for d in uniq:
        keep = days != d
        row = {}
        cells = []
        for m in METRICS:
            a, b, c = lc[m][keep], li[m][keep], lp[m][keep]
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            di, dp = float(np.mean(a[ok] - b[ok])), float(np.mean(a[ok] - c[ok]))
            row[m] = (di, dp)
            cells.append(f"{di:+.4f} / {dp:+.4f}")
        label = str(config.myt(int(d)))[:10]
        loo[label] = row
        lines.append(f"| {label}{' *' if label in BIG_DAYS else ''} | " + " | ".join(cells) + " |")
    keep = ~np.isin([str(config.myt(int(d)))[:10] for d in days], BIG_DAYS)
    cells = []
    allbig = {}
    for m in METRICS:
        a, b, c = lc[m][keep], li[m][keep], lp[m][keep]
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        allbig[m] = (float(np.mean(a[ok] - b[ok])), float(np.mean(a[ok] - c[ok])))
        cells.append(f"{allbig[m][0]:+.4f} / {allbig[m][1]:+.4f}")
    lines.append(f"| ALL 7 big-event days dropped (n={int(keep.sum())}) | " + " | ".join(cells) + " |")
    # only big days
    cells = []
    onlybig = {}
    for m in METRICS:
        a, b, c = lc[m][~keep], li[m][~keep], lp[m][~keep]
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        onlybig[m] = (float(np.mean(a[ok] - b[ok])), float(np.mean(a[ok] - c[ok])))
        cells.append(f"{onlybig[m][0]:+.4f} / {onlybig[m][1]:+.4f}")
    lines.append(f"| ONLY the 7 big-event days (n={int((~keep).sum())}) | " + " | ".join(cells) + " |")
    res["holdout_loo"] = loo
    res["holdout_drop_all_big"] = allbig
    res["holdout_only_big"] = onlybig
    lines.append("")
    # weekly sign consistency over the full period
    frame = rows(ds, mask, "full")
    days = day_of(frame)
    lc, li, lp = losses(frame, cand), losses(frame, inc), losses(frame, prior)
    week = (days - days.min()) // (7 * 86400)
    lines += ["### Weekly blocks (full period; diff cand-inc / cand-prior)", "",
              "| week starting | n | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
    weekly = {}
    for w in np.unique(week):
        k = week == w
        label = str(config.myt(int(days[k].min())))[:10]
        cells = []
        weekly[label] = {}
        for m in METRICS:
            a, b, c = lc[m][k], li[m][k], lp[m][k]
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            weekly[label][m] = (float(np.mean(a[ok] - b[ok])), float(np.mean(a[ok] - c[ok])))
            cells.append(f"{weekly[label][m][0]:+.4f} / {weekly[label][m][1]:+.4f}")
        lines.append(f"| {label} | {int(k.sum())} | " + " | ".join(cells) + " |")
    res["weekly"] = weekly
    return res, lines


# ----------------------------------------------------------------------------- events section
SPECS = {
    "arrival20": (lambda f: scoring.truth_direction(f.y_delta, 20), "p_fall20", "p_rise20"),
    "arrival40": (lambda f: scoring.truth_direction(f.y_delta, 40), "p_fall40", "p_rise40"),
    "first20": (lambda f: f.y_first20.to_numpy(float), "p_first_drop", "p_first_rise"),
}


def calls(pred, spec, rule):
    _, neg, pos = SPECS[spec]
    if spec == "first20":
        P = pred[common.FIRST3].to_numpy(float)
        c = np.argmax(np.nan_to_num(P, nan=-1), axis=1) - 1.0
        c[~np.isfinite(P).all(axis=1)] = np.nan
        if rule != "argmax":
            tau = float(rule)
            c = np.where((P[:, 0] >= tau) | (P[:, 2] >= tau), np.where(P[:, 0] >= P[:, 2], -1.0, 1.0), 0.0)
        return c
    return scoring.calls_from_probs(pred[neg], pred[pos], rule)


KEYS = ("episodes", "episodes_caught", "hits", "misses", "wrong_way", "false_calls", "false_clusters",
        "wrong_way_clusters")


def per_day_conf(frame, truth, call):
    days = day_of(frame)
    issues = frame.index.to_numpy(np.int64)
    uniq = np.unique(days)
    out = np.zeros((len(uniq), len(KEYS)))
    for j, d in enumerate(uniq):
        k = days == d
        r = scoring.event_confusion(issues[k], truth[k], call[k])
        out[j] = [r[x] for x in KEYS]
    return uniq, out


def errors(arr):
    """issue-level errors = misses + wrong-way + false; episode-level = missed episodes + false clusters + wrong-way clusters."""
    K = {k: i for i, k in enumerate(KEYS)}
    issue = arr[..., K["misses"]] + arr[..., K["wrong_way"]] + arr[..., K["false_calls"]]
    episode = (arr[..., K["episodes"]] - arr[..., K["episodes_caught"]]) + arr[..., K["false_clusters"]] \
        + arr[..., K["wrong_way_clusters"]]
    return issue, episode


def curve_per_day(frame, truth, p_neg, p_pos, taus):
    days = day_of(frame)
    issues = frame.index.to_numpy(np.int64)
    uniq = np.unique(days)
    score = np.maximum(p_neg, p_pos)
    direction = np.where(p_neg >= p_pos, -1.0, 1.0)
    false = np.zeros((len(uniq), len(taus)))
    caught = np.zeros((len(uniq), len(taus)))
    for j, d in enumerate(uniq):
        k = days == d
        for t, tau in enumerate(taus):
            r = scoring.event_confusion(issues[k], truth[k], np.where(score[k] >= tau, direction[k], 0.0))
            false[j, t], caught[j, t] = r["false_calls"], r["episodes_caught"]
    return false, caught


def section_events(ds, cand, inc, prior, mask):
    res, lines = {}, []
    persistence = pd.DataFrame({"p_fall20": 0.0, "p_rise20": 0.0, "p_fall40": 0.0, "p_rise40": 0.0,
                                "p_first_drop": 0.0, "p_first_none": 1.0, "p_first_rise": 0.0}, index=ds.index)
    for period in ("holdout", "full"):
        frame = rows(ds, mask, period)
        res[period] = {}
        lines += [f"### {period}: event errors (issue errors = miss+wrong+false; episode errors = missed ep + false clusters + wrong-way clusters)", "",
                  "| target | rule | model | caught/ep | miss | wrong | false | false cl | wrong cl | issue errors | episode errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for spec in SPECS:
            truth = SPECS[spec][0](frame)
            for rule in ("argmax", "0.2"):
                per = {}
                for name, pred in (("persistence", persistence), ("incumbent", inc), ("candidate", cand)):
                    pr = pred.reindex(frame.index)
                    c = calls(pr, spec, rule)
                    g = scoring.event_confusion(frame.index.to_numpy(np.int64), truth, c)
                    uniq, arr = per_day_conf(frame, truth, c)
                    per[name] = arr
                    ie, ee = errors(np.array([g[k] for k in KEYS], float))
                    lines.append(f"| {spec} | {rule} | {name} | {g['episodes_caught']}/{g['episodes']} | {g['misses']} | "
                                 f"{g['wrong_way']} | {g['false_calls']} | {g['false_clusters']} | {g['wrong_way_clusters']} | "
                                 f"{int(ie)} | {int(ee)} |")
                    res[period][f"{spec}|{rule}|{name}"] = {k: g[k] for k in KEYS}
                # bootstrap of the differences (per-day sums; an episode crossing midnight counts in both days)
                idx = resample_index(len(uniq), 1)
                K = {k: i for i, k in enumerate(KEYS)}
                out = {}
                for other in ("incumbent", "persistence"):
                    a, b = per["candidate"][idx].sum(1), per[other][idx].sum(1)
                    ia, ea = errors(a)
                    ib, eb = errors(b)
                    dc = a[:, K["episodes_caught"]] - b[:, K["episodes_caught"]]
                    out[other] = {
                        "caught_diff": [float(np.quantile(dc, q)) for q in (0.025, 0.5, 0.975)],
                        "p_more_caught": float(np.mean(dc > 0)), "p_fewer_caught": float(np.mean(dc < 0)),
                        "issue_err_diff": [float(np.quantile(ia - ib, q)) for q in (0.025, 0.5, 0.975)],
                        "p_fewer_issue_err": float(np.mean(ia < ib)),
                        "episode_err_diff": [float(np.quantile(ea - eb, q)) for q in (0.025, 0.5, 0.975)],
                        "p_fewer_episode_err": float(np.mean(ea < eb))}
                res[period][f"{spec}|{rule}|boot"] = out
        lines += ["", f"### {period}: bootstrap (1-day blocks) of candidate minus incumbent / persistence", "",
                  "| target | rule | vs | caught diff [2.5, 50, 97.5] | P(cand catches more) | P(fewer) | issue-error diff | P(fewer issue errors) | episode-error diff | P(fewer episode errors) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for spec in SPECS:
            for rule in ("argmax", "0.2"):
                for other, r in res[period][f"{spec}|{rule}|boot"].items():
                    lines.append(f"| {spec} | {rule} | {other} | {r['caught_diff']} | {r['p_more_caught']:.2f} | "
                                 f"{r['p_fewer_caught']:.2f} | {r['issue_err_diff']} | {r['p_fewer_issue_err']:.2f} | "
                                 f"{r['episode_err_diff']} | {r['p_fewer_episode_err']:.2f} |")
        lines.append("")
    # matched false-call bootstrap on the holdout
    frame = rows(ds, mask, "holdout")
    lines += ["### Holdout matched-false-call bootstrap (budget = incumbent argmax false calls in each resample; both read on their own threshold curve)", "",
              "| target | point: cand / inc caught at budget | diff [2.5, 50, 97.5] | P(cand more) | P(cand fewer) |", "|---|---|---|---|---|"]
    res["matched_boot"] = {}
    for spec in SPECS:
        truth = SPECS[spec][0](frame)
        _, neg, pos = SPECS[spec]
        cp, ip = cand.reindex(frame.index), inc.reindex(frame.index)
        sc = np.maximum(cp[neg].to_numpy(float), cp[pos].to_numpy(float))
        si = np.maximum(ip[neg].to_numpy(float), ip[pos].to_numpy(float))
        taus_c = np.unique(np.concatenate([np.quantile(sc, np.linspace(0.5, 1, 150)), [1.01]]))
        taus_i = np.unique(np.concatenate([np.quantile(si, np.linspace(0.5, 1, 150)), [1.01]]))
        fc, cc = curve_per_day(frame, truth, cp[neg].to_numpy(float), cp[pos].to_numpy(float), taus_c)
        fi, ci = curve_per_day(frame, truth, ip[neg].to_numpy(float), ip[pos].to_numpy(float), taus_i)
        uniq, arr_inc = per_day_conf(frame, truth, calls(ip, spec, "argmax"))
        K = {k: i for i, k in enumerate(KEYS)}

        def at(false, caught, budget):
            ok = false <= budget
            return caught[ok].max() if ok.any() else 0

        budget0 = arr_inc[:, K["false_calls"]].sum()
        point = (at(fc.sum(0), cc.sum(0), budget0), at(fi.sum(0), ci.sum(0), budget0))
        idx = resample_index(len(uniq), 1, 2000)
        diffs = []
        for row in idx:
            budget = arr_inc[row, K["false_calls"]].sum()
            diffs.append(at(fc[row].sum(0), cc[row].sum(0), budget) - at(fi[row].sum(0), ci[row].sum(0), budget))
        diffs = np.array(diffs)
        res["matched_boot"][spec] = {"point": [float(x) for x in point], "budget": float(budget0),
                                     "q": [float(np.quantile(diffs, q)) for q in (0.025, 0.5, 0.975)],
                                     "p_more": float(np.mean(diffs > 0)), "p_fewer": float(np.mean(diffs < 0))}
        r = res["matched_boot"][spec]
        lines.append(f"| {spec} @{int(budget0)} | {point[0]:.0f} / {point[1]:.0f} | {r['q']} | {r['p_more']:.2f} | {r['p_fewer']:.2f} |")
    lines.append("")
    # which episodes each model catches at the global budget (holdout)
    lines += ["### Holdout episodes caught at the incumbent-argmax false-call budget (global thresholds)", "",
              "| target | episode start (MYT) | dir | issues | candidate | incumbent |", "|---|---|---|---|---|---|"]
    res["episodes"] = {}
    for spec in SPECS:
        truth = SPECS[spec][0](frame)
        _, neg, pos = SPECS[spec]
        issues = frame.index.to_numpy(np.int64)
        flags = {}
        for name, pred in (("candidate", cand), ("incumbent", inc)):
            pr = pred.reindex(frame.index)
            budget = scoring.event_confusion(issues, truth, calls(inc.reindex(frame.index), spec, "argmax"))["false_calls"]
            curve = E.threshold_curve(issues, truth, pr[neg], pr[pos])
            tau = E.at_budget(curve, budget)["tau"]
            s = np.maximum(pr[neg].to_numpy(float), pr[pos].to_numpy(float))
            dirn = np.where(pr[neg].to_numpy(float) >= pr[pos].to_numpy(float), -1.0, 1.0)
            flags[name] = np.where(s >= tau, dirn, 0.0)
        ep = scoring.episodes_table(issues, truth)
        res["episodes"][spec] = []
        for _, e in ep.iterrows():
            k = (issues >= e.start) & (issues <= e.end)
            d = -1.0 if e.direction == "fall" else 1.0
            got = {n: bool(np.any((flags[n][k] == d) & (truth[k] == d))) for n in flags}
            if got["candidate"] or got["incumbent"]:
                lines.append(f"| {spec} | {e.start_myt} | {e.direction} | {e.issues} | {'Y' if got['candidate'] else '.'} | "
                             f"{'Y' if got['incumbent'] else '.'} |")
                res["episodes"][spec].append({"start": e.start_myt, "dir": e.direction, **got})
    lines.append("")
    return res, lines


# ----------------------------------------------------------------------------- selection section
def all_variants(ds, base, days):
    """tune.py variants, recomputed for every day of the evaluation period."""
    import tune
    tune_days = tune.common.days_between  # keep import side effects identical
    out = {f"raw:{n}": f for n, f in base.items()}
    for n, f in base.items():
        if n != "lr_emp_28":
            out[f"tb21:{n}"] = tune._tb(ds, {n: f}, days)
    out["tb14:hgb_all_a0"] = tune._tb(ds, {"hgb_all_a0": base["hgb_all_a0"]}, days, lookback=14)
    out["tb28:hgb_all_a0"] = tune._tb(ds, {"hgb_all_a0": base["hgb_all_a0"]}, days, lookback=28)
    out["mx21:hgb_all_a0"] = cal.calibrate(ds, {"a": base["hgb_all_a0"]}, "class5", "matrix", lookback_days=21,
                                           days=days, C=0.1)
    for clf in [n for n in base if n.startswith("hgb_all")]:
        out[f"stack:lr_hgb_all+{clf}"] = tune._stack(ds, base, "lr_hgb_all", clf, days)
    out["stack14:lr_hgb_all+hgb_all_a0"] = tune._stack(ds, base, "lr_hgb_all", "hgb_all_a0", days, lookback=14)
    out["stack14tb:lr_hgb_all+hgb_all_a0"] = tune._stack(ds, base, "lr_hgb_all", "hgb_all_a0", days, lookback=14,
                                                         first_method="tempbias")
    out["stack:lr_hgb_sw+hgb_sw_a0"] = tune._stack(ds, base, "lr_hgb_sw", "hgb_sw_a0", days)
    out["stack:lr_hgb_all+hgb_sensor_a0"] = tune._stack(ds, base, "lr_hgb_all", "hgb_sensor_a0", days)
    return out


def score_variants(ds, variants, mask, period):
    frame = rows(ds, mask, period)
    out = {}
    for name, pred in variants.items():
        L = losses(frame, pred.reindex(frame.index).reindex(columns=sorted(set(pred.columns) | set(common.CLASS5) | set(common.FIRST3) | {"p_fall20", "p_rise20", "p_fall40", "p_rise40"})))
        out[name] = {m: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for m, v in L.items()}
        out[name]["coverage5"] = float(np.isfinite(L["logloss_class5"]).mean())
    return out


def section_select(ds, cand, inc, prior, mask):
    names = sorted(p.stem[5:] for p in common.OUT.glob("base_*.parquet"))
    base = {n: common.load_pred(f"base_{n}") for n in names}
    days = common.days_between(ds, common.TUNE_START, common.EVAL_END)
    variants = all_variants(ds, base, days)
    # identical rows: comparison mask plus every variant that outputs class5 having it
    res, lines = {}, []
    for period in ("tune", "holdout"):
        sc = score_variants(ds, variants, mask, period)
        sc["incumbent"] = score_variants(ds, {"i": inc}, mask, period)["i"]
        sc["prior"] = score_variants(ds, {"p": prior}, mask, period)["p"]
        res[period] = sc
    lines += ["### Every tune variant on tune vs holdout (identical comparison rows)", "",
              "| variant | tune 5c LL | hold 5c LL | tune 1st LL | hold 1st LL | hold fall20 Brier | hold rise20 Brier | hold fall40 Brier | hold 5c Brier | hold 1st Brier |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(res["tune"], key=lambda n: (res["tune"][n]["logloss_class5"]
                                               if np.isfinite(res["tune"][n]["logloss_class5"]) else 9))
    for n in order:
        t, h = res["tune"][n], res["holdout"][n]
        lines.append(f"| {n} | {t['logloss_class5']:.4f} | {h['logloss_class5']:.4f} | {t['logloss_first20']:.4f} | "
                     f"{h['logloss_first20']:.4f} | {h['brier_fall20']:.4f} | {h['brier_rise20']:.4f} | "
                     f"{h['brier_fall40']:.4f} | {h['brier_class5']:.4f} | {h['brier_first20']:.4f} |")
    # rank of the chosen one on holdout, and reversed selection
    calibrated = [n for n in variants if not n.startswith("raw:")]
    for metric in ("logloss_class5", "logloss_first20", "brier_fall20", "brier_rise20", "brier_fall40"):
        vals = {n: res["holdout"][n][metric] for n in calibrated if np.isfinite(res["holdout"][n][metric])}
        ranked = sorted(vals, key=vals.get)
        chosen = "stack14tb:lr_hgb_all+hgb_all_a0"
        beats_inc = sum(v < res["holdout"]["incumbent"][metric] for v in vals.values())
        beats_prior = sum(v < res["holdout"]["prior"][metric] for v in vals.values())
        rev = min((n for n in calibrated if np.isfinite(res["holdout"][n][metric])), key=lambda n: res["holdout"][n][metric])
        lines.append("")
        lines.append(f"- {metric}: chosen ranks {ranked.index(chosen) + 1}/{len(ranked)} among calibrated variants on holdout; "
                     f"{beats_inc}/{len(vals)} beat the incumbent, {beats_prior}/{len(vals)} beat the prior "
                     f"(inc {res['holdout']['incumbent'][metric]:.4f}, prior {res['holdout']['prior'][metric]:.4f}, "
                     f"worst calibrated {max(vals.values()):.4f}). Reversed split: best on holdout = {rev} "
                     f"({res['holdout'][rev][metric]:.4f}); its tune value {res['tune'][rev][metric]:.4f} vs chosen tune "
                     f"{res['tune'][chosen][metric]:.4f}.")
        res[f"rank_{metric}"] = {"rank": ranked.index(chosen) + 1, "of": len(ranked), "beats_inc": beats_inc,
                                 "beats_prior": beats_prior, "reversed_choice": rev}
    return res, lines


# ----------------------------------------------------------------------------- hyperparameter section
def section_hyper(ds, cand, inc, prior, mask):
    base = {n: common.load_pred(f"base_{n}") for n in ("lr_hgb_all", "hgb_all_a0")}
    for p in sorted(common.OUT.glob("robust_base_*.parquet")):
        base[p.stem[12:]] = pd.read_parquet(p)
    days = common.days_between(ds, common.TUNE_START, common.EVAL_END)
    variants = {}

    def stack(dist, clf, lookback=14, ridge=1.0):
        five = cal.calibrate(ds, {dist: base[dist], clf: base[clf]}, "class5", "tempbias", lookback_days=lookback,
                             days=days, ridge=ridge)
        first = cal.calibrate(ds, {clf: base[clf]}, "first20", "tempbias", lookback_days=lookback, days=days,
                              ridge=ridge)
        return five.join(first[common.FIRST3])

    for lb in (7, 10, 14, 21, 28):
        variants[f"lookback{lb}"] = stack("lr_hgb_all", "hgb_all_a0", lookback=lb)
    for rg in (0.1, 10.0, 100.0):
        variants[f"ridge{rg:g}"] = stack("lr_hgb_all", "hgb_all_a0", ridge=rg)
    dists = [n for n in base if n.startswith("lr_hgb")]
    clfs = [n for n in base if n.startswith("hgb_all")]
    for d in dists:
        for c in clfs:
            if (d, c) != ("lr_hgb_all", "hgb_all_a0") and ("lr_hgb_all" in (d,) or c == "hgb_all_a0"):
                variants[f"{d}+{c}"] = stack(d, c)
    res, lines = {}, []
    for period in ("holdout", "full"):
        sc = score_variants(ds, variants, mask, period)
        sc["candidate"] = score_variants(ds, {"c": cand}, mask, period)["c"]
        sc["incumbent"] = score_variants(ds, {"i": inc}, mask, period)["i"]
        sc["prior"] = score_variants(ds, {"p": prior}, mask, period)["p"]
        res[period] = sc
        lines += [f"### Hyperparameter sensitivity, {period} (identical rows)", "",
                  "| variant | fall20 Brier | rise20 Brier | fall40 Brier | 5c Brier | 5c LL | 1st Brier | 1st LL |",
                  "|---|---|---|---|---|---|---|---|"]
        for n in ["prior", "incumbent", "candidate"] + list(variants):
            r = sc[n]
            lines.append(f"| {n} | {r['brier_fall20']:.4f} | {r['brier_rise20']:.4f} | {r['brier_fall40']:.4f} | "
                         f"{r['brier_class5']:.4f} | {r['logloss_class5']:.4f} | {r['brier_first20']:.4f} | "
                         f"{r['logloss_first20']:.4f} |")
        lines.append("")
    # events for each hyper variant on the holdout (fixed rules)
    frame = rows(ds, mask, "holdout")
    issues = frame.index.to_numpy(np.int64)
    lines += ["### Hyperparameter sensitivity: holdout events (caught/episodes, false calls)", "",
              "| variant | >=20 argmax | >=20 0.2 | >=40 0.2 | first argmax |", "|---|---|---|---|---|"]
    for n, pred in [("incumbent", inc), ("candidate", cand)] + list(variants.items()):
        pr = pred.reindex(frame.index)
        cells = []
        for spec, rule in (("arrival20", "argmax"), ("arrival20", "0.2"), ("arrival40", "0.2"), ("first20", "argmax")):
            g = scoring.event_confusion(issues, SPECS[spec][0](frame), calls(pr, spec, rule))
            cells.append(f"{g['episodes_caught']}/{g['episodes']} f{g['false_calls']} w{g['wrong_way']}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return res, lines


# ----------------------------------------------------------------------------- stronger baseline section
def section_levelclim(ds, cand, inc, prior, mask):
    """Candidate vs a level-only climatology (empirical log-ratio distribution, no
    features), raw and tempbias-calibrated (21 d); plus Oct 6 and alarm-rate checks."""
    days = common.days_between(ds, common.TUNE_START, common.EVAL_END)
    emp = common.load_pred("base_lr_emp_all")
    emp_cal = cal.calibrate(ds, {"e": emp}, "class5", "tempbias", lookback_days=21, days=days)
    res, lines = {}, []
    for period in ("holdout", "full"):
        frame = rows(ds, mask, period)
        d = day_of(frame)
        lc = losses(frame, cand)
        res[period] = {}
        lines += [f"### {period}: candidate minus level-only climatology (1-day bootstrap)", "",
                  "| metric | cand | level-clim raw | cand-raw [95% CI] | level-clim calibrated | cand-cal [95% CI] |",
                  "|---|---|---|---|---|---|"]
        lr = losses(frame, emp.reindex(columns=list(emp.columns) + common.FIRST3))
        lk = losses(frame, emp_cal.reindex(columns=list(emp_cal.columns) + common.FIRST3))
        for m in METRICS[:6]:
            a = boot_diff(lc[m], lr[m], d)
            b = boot_diff(lc[m], lk[m], d)
            res[period][m] = {"vs_raw": a, "vs_cal": b}
            lines.append(f"| {m} | {a['a']:.4f} | {a['b']:.4f} | {fmt_ci(a)} | {b['b']:.4f} | {fmt_ci(b)} |")
        lines.append("")
    # Oct 6: what drives the incumbent's loss
    a, b = config.myt_epoch("2026-10-06"), config.myt_epoch("2026-10-07")
    f6 = ds.loc[mask & (ds.index >= a) & (ds.index < b)]
    c6, i6 = cand.reindex(f6.index), inc.reindex(f6.index)
    lines += ["### Oct 6 (hourly means)", "", "| hour | ref | true delta | fall20 frac | rise20 frac | cand P(fall20) | inc P(fall20) | cand P(rise20) | inc P(rise20) |",
              "|---|---|---|---|---|---|---|---|---|"]
    hours = np.array([config.myt(i).hour for i in f6.index.to_numpy(np.int64)])
    for h in np.unique(hours):
        k = hours == h
        lines.append(f"| {h:02d} | {f6.s_freshPm25[k].mean():.0f} | {f6.y_delta[k].mean():+.0f} | {f6.y_fall20[k].mean():.2f} | "
                     f"{f6.y_rise20[k].mean():.2f} | {c6.p_fall20[k].mean():.2f} | {i6.p_fall20[k].mean():.2f} | "
                     f"{c6.p_rise20[k].mean():.2f} | {i6.p_rise20[k].mean():.2f} |")
    lines.append("")
    # alarm rate: share of holdout issues per hour band with P(fall20) >= 0.2
    frame = rows(ds, mask, "holdout")
    hours = np.array([config.myt(i).hour for i in frame.index.to_numpy(np.int64)])
    cp, ip = cand.reindex(frame.index), inc.reindex(frame.index)
    lines += ["### Holdout: share of issues with P(fall20) >= 0.2 by MYT hour band (base rate of the 'p>=0.2 at onset-60' signal)", "",
              "| hours | n | candidate | incumbent | true fall20 rate |", "|---|---|---|---|---|"]
    for lo, hi in ((0, 6), (6, 12), (12, 15), (15, 18), (18, 21), (21, 24)):
        k = (hours >= lo) & (hours < hi)
        lines.append(f"| {lo:02d}-{hi:02d} | {int(k.sum())} | {np.mean(cp.p_fall20[k] >= 0.2):.2f} | "
                     f"{np.mean(ip.p_fall20[k] >= 0.2):.2f} | {frame.y_fall20[k].mean():.3f} |")
    lines.append("")
    # big-fall onsets: was the candidate's p>=0.2 at onset-60 special that day?
    lines += ["### Candidate P(fall20) at onset-60 vs the same clock on the other holdout days", "",
              "| case | onset-60 clock | cand P at onset-60 | share of other holdout days with P>=0.2 at that clock | inc P at onset-60 | inc share |",
              "|---|---|---|---|---|---|"]
    for label, start, end, direction in report.CASES:
        if direction > 0:
            continue
        a, b = config.myt_epoch(start), config.myt_epoch(end)
        part = ds.loc[(ds.index >= a) & (ds.index <= b)]
        hits = part.index[scoring.truth_direction(part.y_delta, 20) == direction]
        if not len(hits):
            continue
        t = int(hits[0]) - 3600
        clock = t - config.midnight(t)
        others = [int(dd) + clock for dd in np.unique(day_of(frame)) if int(dd) != config.midnight(t)]
        others = [o for o in others if o in cand.index and np.isfinite(cand.p_fall20.get(o, np.nan))]
        sc = np.mean([cand.p_fall20[o] >= 0.2 for o in others])
        si = np.mean([inc.p_fall20.get(o, np.nan) >= 0.2 for o in others])
        lines.append(f"| {label} | {str(config.myt(t))[11:16]} | {cand.p_fall20.get(t, np.nan):.2f} | {sc:.2f} (n={len(others)}) | "
                     f"{inc.p_fall20.get(t, np.nan):.2f} | {si:.2f} |")
    return res, lines


def main(which):
    ds, cand, inc, prior, mask = setup()
    sections = {"bootstrap": section_bootstrap, "events": section_events, "select": section_select,
                "hyper": section_hyper, "levelclim": section_levelclim}
    for name in (list(sections) if which == "all" else [which]):
        res, lines = sections[name](ds, cand, inc, prior, mask)
        (common.OUT / f"robust_{name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        (common.OUT / f"robust_{name}.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
