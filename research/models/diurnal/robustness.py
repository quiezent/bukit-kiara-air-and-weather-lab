"""Robustness checks of diurnal_v1 against the incumbents (skeptic pass, reads saved predictions).

    python -B -m diurnal.robustness stats            # bootstrap CIs, leave-one-day/event-out, splits, event errors
    python -B -m diurnal.robustness hp <variant ...> # final-half reruns of hyperparameter neighbours
    python -B -m diurnal.robustness hp_stats         # score the hp reruns on identical rows

Day-block bootstrap: resample MYT evaluation days with replacement (and 3-day moving blocks),
recompute the pooled metric (sum of row losses / rows) for each model, report candidate minus
incumbent / persistence. Rows are those of evaluate.full_table (identical for every model).
Outputs: data\\models\\diurnal_robust\\.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from harness import config, report, scoring, walkforward
from . import evaluate
from .chosen import CHOSEN, COMPOSE
from .models import GROUPS, DiurnalHGB, KernelClimatology

OUT = config.OUT / "diurnal_robust"
SRC = config.OUT / "diurnal"
PERIODS = {"final": evaluate.FINAL, "tune": evaluate.TUNE, "all": (evaluate.TUNE[0], evaluate.FINAL[1])}
B = 4000
RNG = np.random.default_rng(20261014)


def load_pred(name):
    if name == "diurnal_v1":
        parts = COMPOSE.items()
    else:
        parts = [(g, name) for g in CHOSEN[name]]
    return pd.concat([pd.read_parquet(SRC / f"pred_{src}__{g}.parquet") for g, src in parts], axis=1)


# ----------------------------------------------------------------------------- per-row losses
def _ll(p, onehot):
    return -np.log(np.clip((p * onehot).sum(axis=1), 1e-15, 1))


def row_losses(ds, preds: dict, replay, period) -> dict:
    """metric -> DataFrame(day, <model>...) of per-row losses on identical rows."""
    ev = evaluate.evaluation_frame(ds, period)
    pers = scoring.persistence_predictions(ev)
    day = pd.Series([config.midnight(i) for i in ev.index], index=ev.index)
    names = list(preds)
    main = preds["diurnal_v1"]
    out = {}

    def frame(mask, fn_by_model: dict):
        idx = ev.index[mask.to_numpy(bool)]
        f = pd.DataFrame({"day": day.loc[idx]})
        for k, fn in fn_by_model.items():
            f[k] = fn(idx)
        return f

    for col, target in (("pred_arrival", "y_arrival"), ("pred_ride_mean", "y_ride_mean"),
                        ("pred_ride_min", "y_ride_min"), ("pred_ride_max", "y_ride_max")):
        m = evaluate._mask(ev, replay[col], ev[target], main[col], *[p[col] for p in preds.values() if col in p])
        y = ev[target]
        fns = {"persistence": lambda i: (pers.loc[i, col] - y.loc[i]).abs().to_numpy(),
               "incumbent": lambda i: (replay.loc[i, col] - y.loc[i]).abs().to_numpy()}
        for n, p in preds.items():
            if col in p:
                fns[n] = (lambda pp: lambda i: (pp.loc[i, col] - y.loc[i]).abs().to_numpy())(p)
        key = col.replace("pred_", "")
        out[f"{key}_mae"] = frame(m, fns)
        big = m & ((ev[target] - ev.y_ref).abs() >= 20)
        out[f"{key}_mae_change20"] = frame(big, fns)
        if key in ("ride_min", "ride_max"):
            tau = 0.1 if key == "ride_min" else 0.9

            def pin(q, i, tau=tau):
                d = (y.loc[i] - q.loc[i, col]).to_numpy()
                return np.maximum(tau * d, (tau - 1) * d)
            fns = {"persistence": lambda i: pin(pers, i), "incumbent": lambda i: pin(replay, i)}
            for n, p in preds.items():
                if col in p:
                    fns[n] = (lambda pp: lambda i: pin(pp, i))(p)
            out[f"{key}_pinball"] = frame(m, fns)

    # arrival-change probabilities
    m = evaluate._mask(ev, replay.p_fall20, ev.y_delta, main.p_fall20, *[p.p_fall20 for p in preds.values()
                                                                        if "p_fall20" in p])
    prior = evaluate.class5_prior(ev, ds)
    models = {"prior": prior, "incumbent": replay, **{n: p for n, p in preds.items() if "p_fall20" in p}}
    for side in ("fall20", "rise20", "fall40", "rise40"):
        y = ev[f"y_{side}"]
        mm = m & y.notna()
        out[f"brier_{side}"] = frame(mm, {n: (lambda pp: lambda i: ((pp.loc[i, f"p_{side}"] - y.loc[i]) ** 2).to_numpy())(p)
                                          for n, p in models.items()})
    mm = m & ev.y_class5.isin(range(5))
    oh = lambda i: (ev.loc[i, "y_class5"].to_numpy()[:, None] == np.arange(5)[None, :]).astype(float)
    p5 = [f"p_class5_{k}" for k in range(5)]
    out["class5_brier"] = frame(mm, {n: (lambda pp: lambda i: ((pp.loc[i, p5].to_numpy() - oh(i)) ** 2).sum(axis=1))(p)
                                     for n, p in models.items()})
    out["class5_ll"] = frame(mm, {n: (lambda pp: lambda i: _ll(pp.loc[i, p5].to_numpy(), oh(i)))(p)
                                  for n, p in models.items()})
    # first crossing
    cols = ["p_first_drop", "p_first_none", "p_first_rise"]
    m = evaluate._mask(ev, replay.p_first_drop, ev.y_first20, main.p_first_drop) & ev.y_first20.isin((-1, 0, 1))
    fprior = evaluate.first_prior(ev, ds)
    models = {"prior": fprior, "incumbent": replay, **{n: p for n, p in preds.items() if "p_first_drop" in p}}
    oh3 = lambda i: (ev.loc[i, "y_first20"].to_numpy()[:, None] == np.array([-1, 0, 1])[None, :]).astype(float)
    out["first_brier"] = frame(m, {n: (lambda pp: lambda i: ((pp.loc[i, cols].to_numpy() - oh3(i)) ** 2).sum(axis=1))(p)
                                   for n, p in models.items()})
    out["first_ll"] = frame(m, {n: (lambda pp: lambda i: _ll(pp.loc[i, cols].to_numpy(), oh3(i)))(p)
                                for n, p in models.items()})
    # P(ride mean <= 70)
    m = evaluate._mask(ev, replay.p_ride_le70, ev.y_ride_le70, main.p_ride_le70)
    y = ev.y_ride_le70
    models = {"persistence": pers, "prior": evaluate.le70_prior(ev, ds), "incumbent": replay,
              **{n: p for n, p in preds.items() if "p_ride_le70" in p}}
    out["le70_brier"] = frame(m, {n: (lambda pp: lambda i: ((pp.loc[i, "p_ride_le70"] - y.loc[i]) ** 2).to_numpy())(p)
                                  for n, p in models.items()})

    def bll(pp, i):
        p = np.clip(pp.loc[i, "p_ride_le70"].to_numpy(), 1e-15, 1 - 1e-15)
        t = y.loc[i].to_numpy()
        return -(t * np.log(p) + (1 - t) * np.log(1 - p))
    out["le70_ll"] = frame(m, {n: (lambda pp: lambda i: bll(pp, i))(p) for n, p in models.items()
                               if n != "persistence"})
    return out


# ----------------------------------------------------------------------------- bootstrap
def _day_sums(f, models):
    g = f.groupby("day")
    return g.size().to_numpy(float), {m: g[m].sum().to_numpy(float) for m in models}, g.size().index.to_numpy()


def bootstrap(f, cand="diurnal_v1", others=("incumbent", "persistence", "prior"), block=1, reps=B):
    models = [c for c in f.columns if c != "day"]
    n, s, days = _day_sums(f, models)
    D = len(days)
    if block == 1:
        idx = RNG.integers(0, D, size=(reps, D))
    else:                                                    # moving blocks of consecutive days (circular)
        starts = RNG.integers(0, D, size=(reps, int(np.ceil(D / block))))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % D).reshape(reps, -1)[:, :D]
    N = n[idx].sum(axis=1)
    point = {m: s[m].sum() / n.sum() for m in models}
    res = {"n_rows": int(n.sum()), "n_days": D, **{f"{m}": point[m] for m in models}}
    for o in others:
        if o not in models or o == cand:
            continue
        d = (s[cand][idx].sum(axis=1) - s[o][idx].sum(axis=1)) / N
        res[f"d_vs_{o}"] = point[cand] - point[o]
        res[f"ci_vs_{o}"] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        res[f"p_notbetter_vs_{o}"] = float((d >= 0).mean())
    return res


def leave_one_day_out(f, cand="diurnal_v1", other="incumbent"):
    n, s, days = _day_sums(f, [cand, other])
    tot = (s[cand].sum() - s[other].sum())
    rows = []
    for j, d in enumerate(days):
        diff = (tot - (s[cand][j] - s[other][j])) / (n.sum() - n[j])
        rows.append((str(config.myt(int(d)))[:10], diff, (s[cand][j] - s[other][j]) / n[j], int(n[j])))
    return pd.DataFrame(rows, columns=["day_left_out", "diff_without_day", "diff_on_day", "rows_on_day"])


# ----------------------------------------------------------------------------- events
def _episode_outcomes(issues, truth, call, direction):
    """Episodes of one direction: list of (start, caught, wrong_only); false clusters: list of starts."""
    lookup = dict(zip(issues, call))
    eps = []
    for run in scoring._runs(issues, truth == direction):
        calls = np.array([lookup[e] for e in run])
        caught = bool((calls == direction).any())
        eps.append((int(run[0]), caught, (not caught) and bool((calls == -direction).any())))
    false = [int(r[0]) for r in scoring._runs(issues, (truth == 0) & (call == direction))]
    wrong_issue = int(((truth == -direction) & (call == direction)).sum())
    return eps, false, wrong_issue


def event_errors(ds, preds, replay, period):
    ev = evaluate.evaluation_frame(ds, period)
    main = preds["diurnal_v1"]
    m = evaluate._mask(ev, replay.pred_arrival, replay.p_fall20, ev.y_delta, main.pred_arrival, main.p_fall20)
    ev = ev.loc[m]
    issues, ref = ev.index.to_numpy(np.int64), ev.y_ref.to_numpy(float)
    rows, per_day = [], []
    for t in (20, 40):
        truth = scoring.truth_direction(ev.y_delta, t)
        sources = {"persistence": ("point", None), "incumbent": ("both", replay)}
        sources.update({n: ("both", p) for n, p in preds.items() if "p_fall20" in p and "pred_arrival" in p})
        for name, (_, p) in sources.items():
            for rule in ("point", "argmax", "0.2"):
                if p is None:
                    call = np.zeros(len(ev))
                elif rule == "point":
                    call = scoring.calls_from_point(p.pred_arrival.reindex(ev.index), ref, t)
                else:
                    call = scoring.calls_from_probs(p[f"p_fall{t}"].reindex(ev.index), p[f"p_rise{t}"].reindex(ev.index), rule)
                for d, side in ((-1.0, "fall"), (1.0, "rise")):
                    ok = np.isfinite(truth) & np.isfinite(call)
                    eps, false, wrong_issue = _episode_outcomes(issues[ok], truth[ok], call[ok], d)
                    caught = sum(e[1] for e in eps)
                    wrong_only = sum(e[2] for e in eps)
                    rows.append({"t": t, "side": side, "model": name, "rule": rule, "episodes": len(eps),
                                 "caught": caught, "missed": len(eps) - caught - wrong_only, "wrong_only": wrong_only,
                                 "false_clusters": len(false), "wrong_way_issues": wrong_issue,
                                 "errors": len(eps) - caught + len(false)})
                    for e in eps:
                        per_day.append({"t": t, "side": side, "model": name, "rule": rule,
                                        "day": config.midnight(e[0]), "ep": 1, "caught": int(e[1]), "false": 0})
                    for s in false:
                        per_day.append({"t": t, "side": side, "model": name, "rule": rule,
                                        "day": config.midnight(s), "ep": 0, "caught": 0, "false": 1})
    return pd.DataFrame(rows), pd.DataFrame(per_day)


def first_errors(ds, preds, replay, period):
    ev = evaluate.evaluation_frame(ds, period)
    main = preds["diurnal_v1"]
    m = evaluate._mask(ev, replay.p_first_drop, ev.y_first20, main.p_first_drop) & ev.y_first20.isin((-1, 0, 1))
    ev = ev.loc[m]
    issues = ev.index.to_numpy(np.int64)
    truth = ev.y_first20.to_numpy(float)
    cols = ["p_first_drop", "p_first_none", "p_first_rise"]
    rows = []
    for name, p in {"incumbent": replay, **{n: p for n, p in preds.items() if "p_first_drop" in p}}.items():
        call = np.argmax(p.loc[ev.index, cols].to_numpy(float), axis=1) - 1.0
        for d, side in ((-1.0, "drop"), (1.0, "rise")):
            eps, false, wi = _episode_outcomes(issues, truth, call, d)
            caught = sum(e[1] for e in eps)
            rows.append({"side": side, "model": name, "episodes": len(eps), "caught": caught,
                         "wrong_only": sum(e[2] for e in eps), "false_clusters": len(false), "wrong_way_issues": wi,
                         "errors": len(eps) - caught + len(false)})
    return pd.DataFrame(rows)


def event_bootstrap(per_day, days, a="diurnal_v1", b="incumbent", reps=B):
    """Day bootstrap of (errors_a - errors_b) and (caught_a - caught_b) for each t/side/rule."""
    out = []
    days = np.array(sorted(days))
    for (t, side, rule), g in per_day.groupby(["t", "side", "rule"]):
        def counts(model):
            x = g[g.model == model].groupby("day")[["ep", "caught", "false"]].sum().reindex(days).fillna(0)
            return (x.ep - x.caught + x.false).to_numpy(), x.caught.to_numpy()
        if a not in set(g.model) or b not in set(g.model):
            continue
        ea, ca = counts(a)
        eb, cb = counts(b)
        idx = RNG.integers(0, len(days), size=(reps, len(days)))
        de = ea[idx].sum(1) - eb[idx].sum(1)
        dc = ca[idx].sum(1) - cb[idx].sum(1)
        out.append({"t": t, "side": side, "rule": rule, "errors_diff": int(ea.sum() - eb.sum()),
                    "errors_ci": (float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))),
                    "p_errors_not_lower": float((de >= 0).mean()),
                    "caught_diff": int(ca.sum() - cb.sum()),
                    "caught_ci": (float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5)))})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- driver
KEY = ["arrival_mae", "ride_mean_mae", "ride_min_mae", "ride_max_mae", "ride_min_pinball", "ride_max_pinball",
       "arrival_mae_change20", "ride_mean_mae_change20", "brier_fall20", "brier_rise20", "brier_fall40",
       "class5_brier", "class5_ll", "first_brier", "first_ll", "le70_brier", "le70_ll"]
WEEKS = [("09-10..09-16", "2026-09-10", "2026-09-17"), ("09-17..09-23", "2026-09-17", "2026-09-24"),
         ("09-24..09-30", "2026-09-24", "2026-10-01"), ("10-01..10-07", "2026-10-01", "2026-10-08"),
         ("10-08..10-13", "2026-10-08", "2026-10-14")]


def _fmt_ci(r, o):
    if f"d_vs_{o}" not in r:
        return "–"
    lo, hi = r[f"ci_vs_{o}"]
    return f"{r[f'd_vs_{o}']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={r[f'p_notbetter_vs_{o}']:.3f}"


def stats(ds):
    OUT.mkdir(parents=True, exist_ok=True)
    replay = report.standard_replay(ds)
    preds = {n: load_pred(n) for n in ("diurnal_v1", "kernel_best", "hgb_best", "hgb_time_nwp", "pure_diurnal")}
    result = {}
    lines = []
    for label, period in PERIODS.items():
        losses = row_losses(ds, preds, replay, period)
        pd.to_pickle(losses, OUT / f"losses_{label}.pkl")
        lines.append(f"\n## Day-block bootstrap, {label} half ({B} reps): candidate minus other, 95% CI, "
                     f"p = share of resamples where the candidate is not better\n")
        lines.append("| metric | n rows | days | incumbent | baseline | diurnal_v1 | vs incumbent (1-day blocks) | "
                     "vs incumbent (3-day blocks) | vs persistence/prior (1-day) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for k in KEY:
            if k not in losses:
                continue
            f = losses[k]
            r1 = bootstrap(f)
            r3 = bootstrap(f, block=3)
            base = "persistence" if "persistence" in f else "prior"
            result[f"{label}/{k}"] = {"block1": r1, "block3": r3}
            lines.append(f"| {k} | {r1['n_rows']} | {r1['n_days']} | {r1['incumbent']:.4f} | {r1[base]:.4f} ({base}) | "
                         f"{r1['diurnal_v1']:.4f} | {_fmt_ci(r1, 'incumbent')} | {_fmt_ci(r3, 'incumbent')} | "
                         f"{_fmt_ci(r1, base)} |")
        # same bootstrap: the other families vs the incumbent (reverse-split view)
        if label == "final":
            lines.append("\n### Final half: other families (each vs incumbent), point estimate and 1-day-block CI\n")
            lines.append("| metric | kernel_best | hgb_best | hgb_time_nwp | pure_diurnal |")
            lines.append("|---|---|---|---|---|")
            for k in KEY:
                f = losses[k]
                cells = []
                for n in ("kernel_best", "hgb_best", "hgb_time_nwp", "pure_diurnal"):
                    if n not in f:
                        cells.append("–")
                        continue
                    r = bootstrap(f, cand=n, others=("incumbent",))
                    cells.append(f"{r[n]:.4f} ({_fmt_ci(r, 'incumbent')})")
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
            # leave one day out
            lines.append("\n## Final half, leave-one-day-out (candidate minus incumbent; worst = least favourable)\n")
            lines.append("| metric | full | worst without one day (day) | best without one day (day) | "
                         "days where candidate worse | sign flips? |")
            lines.append("|---|---|---|---|---|---|")
            for k in KEY:
                lo = leave_one_day_out(losses[k])
                full = result[f"final/{k}"]["block1"]["d_vs_incumbent"]
                w, b = lo.loc[lo.diff_without_day.idxmax()], lo.loc[lo.diff_without_day.idxmin()]
                lines.append(f"| {k} | {full:+.4f} | {w.diff_without_day:+.4f} ({w.day_left_out}) | "
                             f"{b.diff_without_day:+.4f} ({b.day_left_out}) | {(lo.diff_on_day > 0).sum()}/{len(lo)} | "
                             f"{'YES' if (np.sign(lo.diff_without_day) != np.sign(full)).any() else 'no'} |")
                lo.to_csv(OUT / f"lodo_{k}.csv", index=False)
            # leave one big event out (drop that event's day)
            lines.append("\n## Final half, leave-one-big-event-out (drop the event's MYT day): candidate minus incumbent\n")
            heads = ["arrival_mae", "ride_mean_mae", "brier_fall20", "class5_ll", "first_ll", "le70_brier"]
            lines.append("| case dropped | " + " | ".join(heads) + " |")
            lines.append("|---|" + "---|" * len(heads))
            for case, start, _, _ in report.CASES:
                dday = config.midnight(config.myt_epoch(start))
                cells = []
                for k in heads:
                    f = losses[k]
                    g = f[f.day != dday]
                    cells.append(f"{(g.diurnal_v1.sum() - g.incumbent.sum()) / len(g):+.4f}")
                lines.append(f"| {case} | " + " | ".join(cells) + " |")
            # drop all eight event days at once
            ddays = {config.midnight(config.myt_epoch(s)) for _, s, _, _ in report.CASES}
            cells = []
            for k in heads:
                g = losses[k][~losses[k].day.isin(ddays)]
                cells.append(f"{(g.diurnal_v1.sum() - g.incumbent.sum()) / len(g):+.4f}")
            lines.append(f"| all 8 case days | " + " | ".join(cells) + " |")
            # only the event days
            cells = []
            for k in heads:
                g = losses[k][losses[k].day.isin(ddays)]
                cells.append(f"{(g.diurnal_v1.sum() - g.incumbent.sum()) / len(g):+.4f}")
            lines.append(f"| only the 8 case days | " + " | ".join(cells) + " |")
        if label == "all":
            lines.append("\n## Weekly splits (candidate minus incumbent | candidate minus persistence/prior)\n")
            heads = ["arrival_mae", "ride_mean_mae", "ride_min_mae", "ride_max_mae", "brier_fall20", "class5_ll",
                     "first_ll", "le70_brier"]
            lines.append("| week | " + " | ".join(heads) + " |")
            lines.append("|---|" + "---|" * len(heads))
            for wl, a, b in WEEKS + [("odd days", None, 1), ("even days", None, 0)]:
                cells = []
                for k in heads:
                    f = losses[k]
                    if a is None:
                        dn = ((f.day - config.myt_epoch("2026-09-10")) // 86400) % 2
                        g = f[dn == b]
                    else:
                        g = f[(f.day >= config.myt_epoch(a)) & (f.day < config.myt_epoch(b))]
                    base = "persistence" if "persistence" in g else "prior"
                    cells.append(f"{(g.diurnal_v1.sum() - g.incumbent.sum()) / len(g):+.3f} / "
                                 f"{(g.diurnal_v1.sum() - g[base].sum()) / len(g):+.3f}")
                lines.append(f"| {wl} | " + " | ".join(cells) + " |")
    # events
    for label in ("final", "tune"):
        tab, per_day = event_errors(ds, preds, replay, PERIODS[label])
        tab.to_csv(OUT / f"event_errors_{label}.csv", index=False)
        lines.append(f"\n## Episode-level event errors, {label} half (errors = uncaught episodes + false-call clusters)\n")
        piv = tab[tab.model.isin(["persistence", "incumbent", "diurnal_v1", "kernel_best", "hgb_time_nwp"])]
        lines.append("| t | side | rule | model | caught/episodes | wrong-only | false clusters | wrong-way issues | errors |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in piv.itertuples():
            if r.model == "persistence" and r.rule != "point":
                continue
            lines.append(f"| {r.t} | {r.side} | {r.rule} | {r.model} | {r.caught}/{r.episodes} | {r.wrong_only} | "
                         f"{r.false_clusters} | {r.wrong_way_issues} | {r.errors} |")
        pooled = tab.groupby(["model", "rule"])[["episodes", "caught", "false_clusters", "errors"]].sum().reset_index()
        lines.append(f"\nPooled over t in (20, 40) and both sides, {label} half:\n")
        lines.append("| model | rule | caught/episodes | false clusters | errors |")
        lines.append("|---|---|---|---|---|")
        for r in pooled.itertuples():
            lines.append(f"| {r.model} | {r.rule} | {r.caught}/{r.episodes} | {r.false_clusters} | {r.errors} |")
        days = sorted({config.midnight(i) for i in evaluate.evaluation_frame(ds, PERIODS[label]).index})
        eb = event_bootstrap(per_day, days)
        eb.to_csv(OUT / f"event_bootstrap_{label}.csv", index=False)
        lines.append(f"\nDay bootstrap of diurnal_v1 minus incumbent, {label} half (negative errors_diff = fewer errors):\n")
        lines.append("| t | side | rule | errors diff [95% CI] | p(errors not lower) | caught diff [95% CI] |")
        lines.append("|---|---|---|---|---|---|")
        for r in eb.itertuples():
            lines.append(f"| {r.t} | {r.side} | {r.rule} | {r.errors_diff:+d} [{r.errors_ci[0]:+.0f}, {r.errors_ci[1]:+.0f}] | "
                         f"{r.p_errors_not_lower:.3f} | {r.caught_diff:+d} [{r.caught_ci[0]:+.0f}, {r.caught_ci[1]:+.0f}] |")
        fe = first_errors(ds, preds, replay, PERIODS[label])
        fe.to_csv(OUT / f"first_errors_{label}.csv", index=False)
        lines.append(f"\nFirst crossing (argmax), {label} half:\n")
        lines.append("| side | model | caught/episodes | wrong-only | false clusters | wrong-way issues | errors |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in fe.itertuples():
            lines.append(f"| {r.side} | {r.model} | {r.caught}/{r.episodes} | {r.wrong_only} | {r.false_clusters} | "
                         f"{r.wrong_way_issues} | {r.errors} |")
    text = "\n".join(lines)
    (OUT / "stats.md").write_text(text, encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- hyperparameter neighbours
INF = np.inf


def _hgb(leaf=1000, lr=0.05, iters=100, nodes=15, fs="diurnal_nwp"):
    return lambda: DiurnalHGB(feature_set=fs, min_samples_leaf=leaf, learning_rate=lr, max_iter=iters,
                              max_leaf_nodes=nodes, loss="absolute_error")


HP = {
    # near heads: kernel neighbours of (h inf, level 0.3, trend 3, lookback 28)
    "near_l015": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=0.15, bw_trend=3.0, alpha=5.0), 28),
    "near_l05": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=0.5, bw_trend=3.0, alpha=5.0), 28),
    "near_t15": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=0.3, bw_trend=1.5, alpha=5.0), 28),
    "near_t6": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=0.3, bw_trend=6.0, alpha=5.0), 28),
    "near_linf_t5": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=INF, bw_trend=5.0, alpha=5.0), 28),
    "near_lb14": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=0.3, bw_trend=3.0, alpha=5.0), 14),
    "near_lb45": ("near", lambda: KernelClimatology(bw_hour=INF, bw_level=0.3, bw_trend=3.0, alpha=5.0), 45),
    # event heads: HGB neighbours of (leaf 1000, lr 0.05, 100 iters, 15 leaves, lookback 45)
    "events_leaf500": ("events", _hgb(leaf=500), 45),
    "events_leaf2000": ("events", _hgb(leaf=2000), 45),
    "events_lr01": ("events", _hgb(lr=0.1), 45),
    "events_it200": ("events", _hgb(iters=200), 45),
    "events_it50": ("events", _hgb(iters=50), 45),
    "events_lb28": ("events", _hgb(), 28),
    "events_nodes31": ("events", _hgb(nodes=31), 45),
    # P(<=70)
    "le70_leaf500": ("le70", _hgb(leaf=500), 45),
    "le70_leaf2000": ("le70", _hgb(leaf=2000), 45),
    "le70_lb28": ("le70", _hgb(), 28),
    "le70_lr01": ("le70", _hgb(lr=0.1), 45),
}


def hp(ds, names):
    from .run import days_in, run
    (OUT / "hp").mkdir(parents=True, exist_ok=True)
    days = days_in(ds, evaluate.FINAL)
    for name in names:
        group, factory, lookback = HP[name]
        path = OUT / "hp" / f"pred_{name}.parquet"
        if path.exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        started = time.perf_counter()
        pred, timing = run(factory(), ds, days, (group,), lookback_days=lookback)
        pred.to_parquet(path)
        print(f"{name} {group} {timing[group]:.2f} s/day total {time.perf_counter() - started:.0f}s", flush=True)


def hp_stats(ds):
    replay = report.standard_replay(ds)
    base = load_pred("diurnal_v1")
    lines = ["| variant | metric | chosen | variant | incumbent | persistence/prior | variant minus incumbent [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    metrics = {"near": ["arrival_mae", "ride_mean_mae", "ride_min_mae", "ride_max_mae"],
               "events": ["brier_fall20", "brier_fall40", "class5_brier", "class5_ll", "first_brier", "first_ll"],
               "le70": ["le70_brier", "le70_ll"]}
    ev_lines = ["| variant | rule | >=20 fall caught/ep (false cl) | >=20 rise caught/ep (false cl) | "
                ">=40 fall caught/ep (false cl) | pooled errors | first-crossing caught/ep (false cl) |",
                "|---|---|---|---|---|---|---|"]
    for name, (group, _, _) in HP.items():
        path = OUT / "hp" / f"pred_{name}.parquet"
        if not path.exists():
            continue
        v = pd.read_parquet(path)
        variant = base.copy()
        for c in v.columns:
            variant[c] = v[c]
        losses = row_losses(ds, {"diurnal_v1": base, "variant": variant}, replay, evaluate.FINAL)
        for k in metrics[group]:
            f = losses[k]
            r = bootstrap(f, cand="variant", others=("incumbent",))
            b = "persistence" if "persistence" in f else "prior"
            lo, hi = r["ci_vs_incumbent"]
            lines.append(f"| {name} | {k} | {f.diurnal_v1.mean():.4f} | {f.variant.mean():.4f} | {f.incumbent.mean():.4f} | "
                         f"{f[b].mean():.4f} | {r['d_vs_incumbent']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        if group == "events":
            tab, _ = event_errors(ds, {"diurnal_v1": base, "variant": variant}, replay, evaluate.FINAL)
            fe = first_errors(ds, {"diurnal_v1": base, "variant": variant}, replay, evaluate.FINAL)
            fv = fe[fe.model == "variant"]
            for rule in ("argmax", "0.2"):
                t = tab[(tab.model == "variant") & (tab.rule == rule)].set_index(["t", "side"])
                cell = lambda tt, s: f"{t.loc[(tt, s), 'caught']}/{t.loc[(tt, s), 'episodes']} ({t.loc[(tt, s), 'false_clusters']})"
                fc = (f"{fv.caught.sum()}/{fv.episodes.sum()} ({fv.false_clusters.sum()})" if rule == "argmax" else "–")
                ev_lines.append(f"| {name} | {rule} | {cell(20, 'fall')} | {cell(20, 'rise')} | {cell(40, 'fall')} | "
                                f"{t.errors.sum()} | {fc} |")
    text = "\n".join(lines) + "\n\n" + "\n".join(ev_lines)
    (OUT / "hp_stats.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    from .run import load
    command = sys.argv[1]
    data = load()
    if command == "stats":
        stats(data)
    elif command == "hp":
        hp(data, sys.argv[2:] or list(HP))
    elif command == "hp_stats":
        hp_stats(data)
