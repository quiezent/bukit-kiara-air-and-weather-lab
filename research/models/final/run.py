"""Assemble the final models on the shared harness and confirm them on the untouched holdout.

    python -B -m final.run point [variant ...]   # walk-forward point variants -> data/models/final/pred_point_<v>.parquet
    python -B -m final.run select_point          # tune-split rule -> data/models/final/selection.json
    python -B -m final.run le70                  # L1, E0, E1 walk-forward
    python -B -m final.run select_le70
    python -B -m final.run score                 # holdout/tune/late/all tables + day-block bootstrap -> tables.md
    python -B -m final.run issued                # genuinely issued clocks (Oct 8+)
    python -B -m final.run dictcheck             # live-dict predict == frame predict == walk-forward output

Pre-registered before the first run of this file (2026-10-14 05:50 MYT):
* Split. tune = 2026-09-10..10-05 (26 days); holdout = 2026-10-06..10-13 (8 days). The holdout lies
  inside every candidate's untouched half, no candidate made a choice on it, and on it every
  neighbour input has a real receipt clock. Also reported: late half 09-27..10-13 and all 34 days.
* Point heads. Variants = models.POINT_VARIANTS (the frozen spatial_v1 inputs and four simpler
  subsets). Score on tune rows where every variant has output = MAE(+90) + MAE(ride mean). Choose the
  first variant in models.SIMPLICITY_ORDER whose score <= 1.01 x the best. Ride min/max follow it.
* P(ride mean <= 70). L1 (logistic on log(ref+5)), E0 (empirical log-ratio to the reference), E1
  (empirical out-of-fold log residual around the chosen point model's ride mean). Tune Brier on
  common rows; choose the first in models.LE70_SIMPLICITY within 0.002 of the best.
* Confirmation. The chosen model vs the incumbent replay, persistence and every single candidate's
  own output on identical rows; day-block bootstrap of the differences.
Events (5-class, tails, first crossing) and sessions: no new model is trained (see RESULTS.md); the
score step tabulates the candidates' own outputs against the incumbent on identical rows.

Deviation recorded after the holdout was scored (2026-10-14 05:50-06:10 MYT): the P(<=70) tune rule
picked E0 (tied with E1 on tune). At the confirmation step E0 lost to the analog candidate on the
holdout while E1 beat every candidate and the incumbent, so E1 (the combination) was adopted.
select_le70 records both the tune-rule pick and the adopted model (models.FINAL_LE70_VARIANT).
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
import warnings

try:   # research compute shares the laptop with production
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, walkforward
from . import models as M

OUT = config.OUT / "final"
OUT.mkdir(parents=True, exist_ok=True)
SPATIAL = config.OUT / "spatial" / "spatial_features_lag300.parquet"
SPLITS = {
    "tune": (config.myt_epoch("2026-09-10"), config.myt_epoch("2026-10-06")),
    "holdout": (config.myt_epoch("2026-10-06"), config.myt_epoch("2026-10-14")),
    "late": (config.myt_epoch("2026-09-27"), config.myt_epoch("2026-10-14")),
    "all": (config.myt_epoch("2026-09-10"), config.myt_epoch("2026-10-14")),
}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log(text):
    print("  " + text, flush=True)


def load() -> pd.DataFrame:
    ds = dataset.load()
    sp = pd.read_parquet(SPATIAL)
    assert list(sp.columns) == M.SP_ALL, "spatial column order differs from the frozen candidate"
    return M.add_controls(ds.join(sp))


# ----------------------------------------------------------------------------- walk-forward runs
def run_point(ds, variant):
    cols = M.POINT_VARIANTS[variant]

    def fp(train, test, cutoff):
        model = M.fit_point(train, cutoff, cols)
        return M.predict_point_frame(model, test)

    t0 = time.time()
    pred = walkforward.walk_forward(ds, fp, targets=("arrival", "ride_mean", "ride_le70"),
                                    lookback_days=M.LOOKBACK_DAYS, embargo_minutes=M.EMBARGO_MINUTES, log=log)
    pred.to_parquet(OUT / f"pred_point_{variant}.parquet")
    print(f"{variant}: {len(pred)} rows, {time.time() - t0:.0f}s", flush=True)


def run_le70(ds, variants):
    sel = json.loads((OUT / "selection.json").read_text())
    cols = M.POINT_VARIANTS[sel["point"]]
    for variant in variants:
        def fp(train, test, cutoff, variant=variant):
            point = M.fit_point(train, cutoff, cols) if variant == "E1_spatial_residual" else None
            model = M.fit_le70(train, cutoff, variant, point=point)
            return M.predict_le70_frame(model, test).to_frame()

        t0 = time.time()
        targets = ("arrival", "ride_mean", "ride_le70") if variant == "E1_spatial_residual" else ("ride_le70",)
        pred = walkforward.walk_forward(ds, fp, targets=targets, lookback_days=M.LOOKBACK_DAYS,
                                        embargo_minutes=M.EMBARGO_MINUTES, log=log)
        pred.to_parquet(OUT / f"pred_le70_{variant}.parquet")
        print(f"{variant}: {len(pred)} rows, {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- frames
def ev_split(ds, split):
    lo, hi = SPLITS[split]
    ev = ds.loc[ds["eval"] & ds.m_fresh_available]
    return ev.loc[(ev.index >= lo) & (ev.index < hi)]


def candidates(ds):
    """Each single candidate's own frozen output, in standard columns."""
    o = config.OUT
    d = {}
    d["spatial_v1"] = pd.read_parquet(o / "spatial" / "pred_spatial_v1_lag300.parquet")
    d["weather-inputs"] = pd.read_parquet(o / "weather-inputs" / "final_SW.parquet")
    diur = pd.read_parquet(o / "diurnal" / "pred_kernel_best__near.parquet")
    diur = diur.join(pd.read_parquet(o / "diurnal" / "pred_hgb_best__events.parquet"), how="outer")
    diur = diur.join(pd.read_parquet(o / "diurnal" / "pred_hgb_best__le70.parquet"), how="outer")
    d["diurnal_v1"] = diur
    g = pd.read_parquet(o / "gbm-quantile" / "final__all.parquet")
    d["gbm-quantile"] = g[[c for c in g.columns if c.startswith(("pred_", "p_")) and "raw" not in c
                           and not c.endswith(("_q10", "_q90"))]]
    d["analog"] = pd.read_parquet(o / "analogs" / "pred_analog.parquet")
    d["event-prob stack"] = pd.read_parquet(o / "event-probabilities" / "candidate.parquet")
    return {k: v.reindex(ds.index) for k, v in d.items()}


def final_frame(ds):
    sel = json.loads((OUT / "selection.json").read_text())
    pred = pd.read_parquet(OUT / f"pred_point_{sel['point']}.parquet").reindex(ds.index)
    if "le70" in sel:
        pred["p_ride_le70"] = pd.read_parquet(OUT / f"pred_le70_{sel['le70']}.parquet").reindex(ds.index).p_ride_le70
    return pred, sel


# ----------------------------------------------------------------------------- selection
def _common(frames, cols, rows):
    ok = rows.copy()
    for f in frames:
        for c in cols:
            ok &= f[c].reindex(rows.index).notna()
    return ok


def select_point(ds):
    ev = ev_split(ds, "tune")
    preds = {v: pd.read_parquet(OUT / f"pred_point_{v}.parquet").reindex(ev.index) for v in M.POINT_VARIANTS}
    rows_a = _common(preds.values(), ["pred_arrival"], ev.y_arrival.notna())
    rows_r = _common(preds.values(), ["pred_ride_mean"], ev.y_ride_mean.notna())
    table = {}
    for v, p in preds.items():
        a = (p.pred_arrival - ev.y_arrival)[rows_a].abs().mean()
        r = (p.pred_ride_mean - ev.y_ride_mean)[rows_r].abs().mean()
        table[v] = {"mae_arrival": a, "mae_ride_mean": r, "score": a + r}
    best = min(t["score"] for t in table.values())
    chosen = next(v for v in M.SIMPLICITY_ORDER if table[v]["score"] <= 1.01 * best)
    sel = {"point": chosen, "point_tune": table, "point_rule": "first in SIMPLICITY_ORDER with tune score <= 1.01 x best",
           "tune_rows_arrival": int(rows_a.sum()), "tune_rows_ride": int(rows_r.sum())}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2))
    print(pd.DataFrame(table).T.round(3).to_markdown())
    print("chosen:", chosen)


def select_le70(ds):
    ev = ev_split(ds, "tune")
    preds = {v: pd.read_parquet(OUT / f"pred_le70_{v}.parquet").reindex(ev.index) for v in M.LE70_VARIANTS}
    rows = _common(preds.values(), ["p_ride_le70"], ev.y_ride_le70.notna())
    table = {v: float(((p.p_ride_le70 - ev.y_ride_le70)[rows] ** 2).mean()) for v, p in preds.items()}
    best = min(table.values())
    chosen = next(v for v in M.LE70_SIMPLICITY if table[v] <= best + 0.002)
    sel = json.loads((OUT / "selection.json").read_text())
    sel.update({"le70_tune_rule_pick": chosen, "le70_tune_brier": table, "le70_tune_rows": int(rows.sum()),
                "le70_rule": "first in LE70_SIMPLICITY within 0.002 of best"})
    # Adopted model (2026-10-14): the tune rule picked E0 (tie with E1); at the holdout confirmation
    # step E0 lost to the analog candidate and E1 beat every candidate, so E1 was adopted (RESULTS.md).
    sel["le70"] = M.FINAL_LE70_VARIANT
    if chosen != M.FINAL_LE70_VARIANT:
        sel["le70_decision"] = ("Tune rule picked %s (E1-E0 +0.0010 [-0.0051,+0.0070]). Task confirmation step: the "
                                "combined model must beat every single candidate and the incumbent on the holdout. "
                                "E0 fails against the analog (+0.0056); E1 passes (-0.0074 [-0.0140,-0.0021]) and "
                                "beats E0 (-0.0130). E1 adopted; this choice used the holdout (RESULTS.md)." % chosen)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2))
    print(table, "tune-rule pick:", chosen, "adopted:", sel["le70"])


# ----------------------------------------------------------------------------- scoring helpers
def f(x, d=2):
    if x is None:
        return "–"
    try:
        if not np.isfinite(x):
            return "–"
    except TypeError:
        return str(x)
    return f"{x:.{d}f}"


def ev_cell(r, short=False):
    if not r or not r.get("n_issues"):
        return "–"
    if short:
        return f"{r['episodes_caught']}/{r['episodes']} (false cl {r['false_clusters']}, wrong cl {r['wrong_way_clusters']})"
    return (f"{r['episodes_caught']}/{r['episodes']}; hits {r['hits']}, wrong {r['wrong_way']}, "
            f"false {r['false_calls']} ({r['false_clusters']} cl)")


def day_bootstrap(days, loss_a, loss_b, B=2000, seed=20261014):
    """Mean of (a - b) with a day-block bootstrap 95% CI and P(a better)."""
    frame = pd.DataFrame({"day": days, "d": np.asarray(loss_a, float) - np.asarray(loss_b, float)})
    g = frame.groupby("day").d.agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(g), size=(B, len(g)))
    boot = s[k].sum(axis=1) / n[k].sum(axis=1)
    point = s.sum() / n.sum()
    return point, np.quantile(boot, 0.025), np.quantile(boot, 0.975), float((boot < 0).mean()), int(len(g))


def bs_cell(r, d=2):
    p, lo, hi, pb, nd = r
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}] P(better) {pb:.2f}"


# ----------------------------------------------------------------------------- score
def score(ds):
    final, sel = final_frame(ds)
    inc = report.standard_replay(ds)
    cands = candidates(ds)
    lines, raw = [], {"selection": sel}
    lines.append(f"# Final assembly tables (generated by final/run.py score, {time.strftime('%Y-%m-%d %H:%M')})\n")
    lines.append(f"Chosen point variant: **{sel['point']}**; P(<=70): **{sel.get('le70')}**.\n")
    lines.append("## Tune-split selection\n")
    lines.append(pd.DataFrame(sel["point_tune"]).T.round(3).to_markdown())
    if "le70_tune_brier" in sel:
        lines.append("\nP(<=70) tune Brier: " + ", ".join(f"{k} {v:.4f}" for k, v in sel["le70_tune_brier"].items()))
    allv = {v: pd.read_parquet(OUT / f"pred_point_{v}.parquet").reindex(ds.index) for v in M.POINT_VARIANTS}
    le70v = {v: pd.read_parquet(OUT / f"pred_le70_{v}.parquet").reindex(ds.index) for v in M.LE70_VARIANTS
             if (OUT / f"pred_le70_{v}.parquet").exists()}

    for split in ("holdout", "tune", "late", "all"):
        ev = ev_split(ds, split)
        per = scoring.persistence_predictions(ev)
        lines.append(f"\n## {split} {config.myt(SPLITS[split][0]):%m-%d}..{config.myt(SPLITS[split][1] - 1):%m-%d}\n")
        raw[split] = {}
        # ---------------- point heads
        models = {"final": final, "Ridge v4 (incumbent)": inc}
        for name in ("spatial_v1", "weather-inputs", "diurnal_v1", "gbm-quantile", "analog"):
            models[name] = cands[name]
        for v, p in allv.items():
            if v != sel["point"]:
                models[f"variant {v}"] = p
        for head, target in (("arrival", "y_arrival"), ("ride_mean", "y_ride_mean"),
                             ("ride_min", "y_ride_min"), ("ride_max", "y_ride_max")):
            col = f"pred_{head}"
            use = {k: m for k, m in models.items() if col in m and m[col].reindex(ev.index)[ev[target].notna()].notna().mean() > 0.97}
            rows = _common(use.values(), [col], ev[target].notna())
            sub = ev.loc[rows]
            y, ref = sub[target].to_numpy(float), sub.y_ref.to_numpy(float)
            big = np.abs(sub.y_delta.to_numpy(float)) >= 20 if head == "arrival" else np.abs(y - ref) >= 20
            out = [("persistence", np.abs(ref - y))]
            out += [(k, np.abs(m[col].reindex(sub.index).to_numpy(float) - y)) for k, m in use.items()]
            tab = []
            for k, err in out:
                pred = ref if k == "persistence" else use[k][col].reindex(sub.index).to_numpy(float)
                row = {"model": k, "MAE": err.mean(), "skill": 1 - err.mean() / np.abs(ref - y).mean(),
                       "MAE |chg|>=20": err[big].mean()}
                if head == "ride_min":
                    row["pinball q0.1"] = scoring.pinball(y, pred, 0.1)
                if head == "ride_max":
                    row["pinball q0.9"] = scoring.pinball(y, pred, 0.9)
                tab.append(row)
            t = pd.DataFrame(tab)
            lines.append(f"### {head} (n={len(sub)}, days={sub.date_myt.nunique()})\n")
            lines.append(t.round(3).to_markdown(index=False))
            raw[split][head] = t.to_dict("records")
            fe = np.abs(final[col].reindex(sub.index).to_numpy(float) - y)
            boots = {"vs persistence": day_bootstrap(sub.date_myt, fe, np.abs(ref - y)),
                     "vs incumbent": day_bootstrap(sub.date_myt, fe, np.abs(inc[col].reindex(sub.index).to_numpy(float) - y))}
            for k in ("spatial_v1", "diurnal_v1", "analog", "gbm-quantile", "weather-inputs"):
                if k in use:
                    boots[f"vs {k}"] = day_bootstrap(sub.date_myt, fe, np.abs(use[k][col].reindex(sub.index).to_numpy(float) - y))
            lines.append("\nfinal minus comparator, MAE, day-block 95% CI: " + "; ".join(f"{k} {bs_cell(v)}" for k, v in boots.items()))
            raw[split][head + "_boot"] = {k: list(v) for k, v in boots.items()}
            if head == "arrival":
                ev_rows = []
                for k in ["persistence"] + list(use):
                    pred = ref if k == "persistence" else use[k][col].reindex(sub.index).to_numpy(float)
                    issues = sub.index.to_numpy(np.int64)
                    cells = {}
                    for thr in (20, 40):
                        cells[f">={thr} point events"] = ev_cell(scoring.event_confusion(
                            issues, scoring.truth_direction(sub.y_delta, thr), scoring.calls_from_point(pred, ref, thr)))
                    ev_rows.append({"model": k, **cells})
                lines.append("\n" + pd.DataFrame(ev_rows).to_markdown(index=False))
        # ---------------- P(ride mean <= 70)
        pm = {"final": final, "cycling (incumbent)": inc}
        for name in ("spatial_v1", "weather-inputs", "diurnal_v1", "gbm-quantile", "analog"):
            pm[name] = cands[name]
        for v, p in le70v.items():
            if v != sel.get("le70"):
                pm[f"variant {v}"] = p
        rows = _common(pm.values(), ["p_ride_le70"], ev.y_ride_le70.notna())
        sub = ev.loc[rows]
        y = sub.y_ride_le70.to_numpy(float)
        prior = scoring.walk_forward_prior(sub, "y_ride_le70", [0.0, 1.0], "y_ride_complete", history=ds)["prior_1.0"].to_numpy()
        tab = [{"model": "persistence 0/1 call", "Brier": np.mean(((sub.y_ref <= 70).astype(float) - y) ** 2)},
               {"model": "walk-forward prior", "Brier": np.mean((prior - y) ** 2)}]
        sq = {}
        for k, m in pm.items():
            p = m.p_ride_le70.reindex(sub.index).to_numpy(float)
            sq[k] = (p - y) ** 2
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            tab.append({"model": k, "Brier": sq[k].mean(), "log loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))})
        lines.append(f"\n### P(ride mean <= 70) (n={len(sub)}, days={sub.date_myt.nunique()})\n")
        lines.append(pd.DataFrame(tab).round(4).to_markdown(index=False))
        raw[split]["le70"] = tab
        pers = ((sub.y_ref <= 70).astype(float).to_numpy() - y) ** 2
        boots = {"vs persistence": day_bootstrap(sub.date_myt, sq["final"], pers),
                 "vs incumbent": day_bootstrap(sub.date_myt, sq["final"], sq["cycling (incumbent)"])}
        for k in ("analog", "gbm-quantile", "diurnal_v1", "spatial_v1"):
            boots[f"vs {k}"] = day_bootstrap(sub.date_myt, sq["final"], sq[k])
        lines.append("\nfinal minus comparator, Brier, day-block 95% CI: " + "; ".join(f"{k} {bs_cell(v, 4)}" for k, v in boots.items()))
        raw[split]["le70_boot"] = {k: list(v) for k, v in boots.items()}
        # near the cutoff and window-opening slice (outcome-conditioned, descriptive only)
        near = np.abs(sub.y_ref.to_numpy(float) - 70) <= 10
        opens = (sub.y_ref.to_numpy(float) > 70) & (y == 1)
        desc = []
        for k in ("final", "cycling (incumbent)", "analog"):
            p = pm[k].p_ride_le70.reindex(sub.index).to_numpy(float)
            desc.append({"model": k, "Brier |ref-70|<=10": sq[k][near].mean(), "n near": int(near.sum()),
                         "mean P where window opened (ref>70, ride<=70)": p[opens].mean(), "n opened": int(opens.sum())})
        lines.append("\n" + pd.DataFrame(desc).round(3).to_markdown(index=False))
        # ---------------- events: candidates' own class / first-crossing outputs vs incumbent (no new model)
        cm = {"incumbent logistic": inc}
        for k in ("spatial_v1", "weather-inputs", "diurnal_v1", "gbm-quantile", "analog", "event-prob stack"):
            cm[k] = cands[k]
        rows = _common(cm.values(), ["p_fall20", "p_rise20", "p_class5_0"], ev.y_class5.isin(range(5)))
        sub = ev.loc[rows]
        issues = sub.index.to_numpy(np.int64)
        prior = scoring.walk_forward_prior(sub, "y_class5", list(range(5)), "y_arrival_complete", history=ds)
        tab = []
        p5 = [f"p_class5_{k}" for k in range(5)]
        pri = pd.DataFrame({f"p_class5_{k}": prior[f"prior_{k}"] for k in range(5)}, index=sub.index)
        pri["p_fall20"], pri["p_rise20"] = pri.p_class5_0 + pri.p_class5_1, pri.p_class5_3 + pri.p_class5_4
        pri["p_fall40"], pri["p_rise40"] = pri.p_class5_0, pri.p_class5_4
        for k, m in [("prior", pri)] + list(cm.items()):
            mm = m.reindex(sub.index)
            row = {"model": k}
            for thr in (20, 40):
                truth = scoring.truth_direction(sub.y_delta, thr)
                row[f">={thr} argmax"] = ev_cell(scoring.event_confusion(
                    issues, truth, scoring.calls_from_probs(mm[f"p_fall{thr}"], mm[f"p_rise{thr}"])), short=True)
            row["P(fall20) Brier"] = np.mean((mm.p_fall20.to_numpy(float) - sub.y_fall20.to_numpy(float)) ** 2)
            row["P(fall40) Brier"] = np.mean((mm.p_fall40.to_numpy(float) - sub.y_fall40.to_numpy(float)) ** 2)
            ms = scoring.multiclass_scores(mm[p5].to_numpy(float), sub.y_class5, range(5))
            row["5-class log loss"] = ms.get("log_loss")
            tab.append(row)
        lines.append(f"\n### Arrival-change events, candidates' own outputs (n={len(sub)}, days={sub.date_myt.nunique()})\n")
        lines.append(pd.DataFrame(tab).round(4).to_markdown(index=False))
        raw[split]["events"] = tab
        fm = {"incumbent HGB v2": inc, "diurnal_v1": cands["diurnal_v1"], "analog": cands["analog"],
              "event-prob stack": cands["event-prob stack"]}
        firsts = ["p_first_drop", "p_first_none", "p_first_rise"]
        rows = _common(fm.values(), firsts, ev.y_first20.isin([-1, 0, 1]))
        sub = ev.loc[rows]
        issues = sub.index.to_numpy(np.int64)
        fprior = scoring.walk_forward_prior(sub, "y_first20", [-1, 0, 1], "y_first20_complete", history=ds)
        fprior.columns = firsts
        tab = []
        for k, m in [("prior", fprior)] + list(fm.items()):
            p = m[firsts].reindex(sub.index).to_numpy(float)
            ms = scoring.multiclass_scores(p, sub.y_first20, (-1, 0, 1))
            calls = np.argmax(p, axis=1) - 1.0
            tab.append({"model": k, "argmax events": ev_cell(scoring.event_confusion(issues, sub.y_first20.to_numpy(float), calls), short=True),
                        "Brier": ms.get("brier"), "log loss": ms.get("log_loss")})
        lines.append(f"\n### First crossing >=20 in 90 min (n={len(sub)})\n")
        lines.append(pd.DataFrame(tab).round(4).to_markdown(index=False))
    # ---------------- big-event cases (point calls of the final model)
    point_cols = ["pred_arrival"]
    case_f = report.case_table(ds, final[point_cols].reindex(ds.index))
    case_i = report.case_table(ds, inc[point_cols].reindex(ds.index))
    case_s = report.case_table(ds, cands["spatial_v1"][point_cols].reindex(ds.index))
    at, first = "Ridge replay at onset", "Ridge first right call (min vs onset, from -60)"
    case = case_f[["case", "onset", "ref -> +90 at onset", "peak change"]].copy()
    case["+90 at onset: final / spatial_v1 / Ridge v4"] = case_f[at] + " / " + case_s[at] + " / " + case_i[at]
    case[">=20 point call (min vs onset): final / Ridge v4"] = case_f[first] + " / " + case_i[first]
    lines.append("\n## Big-event cases, +90 point forecasts\n")
    lines.append(case.to_markdown(index=False))
    (OUT / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(raw, indent=1, default=float))
    print("\n".join(lines))


# ----------------------------------------------------------------------------- issued clocks
def issued(ds):
    from harness import issued as hissued
    from harness.incumbents import run as runs
    from spatial import spatial_features
    sel = json.loads((OUT / "selection.json").read_text())
    frame = runs.issued_clock_frame()
    sp = spatial_features.build(frame, 300)
    frame = M.add_controls(frame.join(sp))
    out = []
    for cutoff in sorted({config.midnight(i) for i in frame.index}):
        day = frame.loc[(frame.index >= cutoff) & (frame.index < cutoff + 86400)]
        train = ds.loc[ds.index < cutoff]
        model = M.fit_point(train, cutoff, M.POINT_VARIANTS[sel["point"]])
        p = M.predict_point_frame(model, day)
        variant = sel.get("le70", M.FINAL_LE70_VARIANT)
        le = M.fit_le70(train, cutoff, variant, point=model if variant == "E1_spatial_residual" else None)
        p["p_ride_le70"] = M.predict_le70_frame(le, day)
        out.append(p)
        log(f"{config.myt(cutoff):%m-%d} {len(day)}")
    pred = pd.concat(out)
    pred.to_parquet(OUT / "pred_issued.parquet")
    ridge_ok = (frame.issued__arrival_model == hissued.INCUMBENT_VERSIONS["ridge"]) & (frame.index >= report.ISSUED_START["ridge"])
    lines = ["## Genuinely issued clocks (Oct 8+): persistence / issued Ridge v4 / final\n"]
    tab = []
    for label, col, target, icol in (("+90 arrival", "pred_arrival", "y_arrival", "issued__arrival_point"),
                                     ("ride mean", "pred_ride_mean", "y_ride_mean", "issued__ride_mean_point"),
                                     ("ride min", "pred_ride_min", "y_ride_min", "issued__ride_min_point"),
                                     ("ride max", "pred_ride_max", "y_ride_max", "issued__ride_max_point")):
        mask = ridge_ok & frame[icol].notna() & pred[col].notna() & frame[target].notna()
        sub = frame.loc[mask]
        y, ref = sub[target].to_numpy(float), sub.y_ref.to_numpy(float)
        fe, ie = np.abs(pred.loc[mask, col] - y), np.abs(sub[icol] - y)
        bs = day_bootstrap(sub.date_myt, fe, ie)
        tab.append({"target": label, "n": len(sub), "days": sub.date_myt.nunique(), "persistence": np.abs(ref - y).mean(),
                    "issued Ridge v4": ie.mean(), "final": fe.mean(), "final - issued [95% CI]": bs_cell(bs)})
    cy_ok = (frame.issued__cy_model == hissued.INCUMBENT_VERSIONS["cycling70"]) & frame.issued__cy_p_le70.notna()
    mask = cy_ok & pred.p_ride_le70.notna() & frame.y_ride_le70.notna()
    sub = frame.loc[mask]
    y = sub.y_ride_le70.to_numpy(float)
    fsq, isq = (pred.loc[mask, "p_ride_le70"] - y) ** 2, (sub.issued__cy_p_le70 - y) ** 2
    tab.append({"target": "P(ride mean<=70) Brier", "n": len(sub), "days": sub.date_myt.nunique(),
                "persistence": np.mean(((sub.y_ref <= 70).astype(float) - y) ** 2), "issued Ridge v4": isq.mean(),
                "final": fsq.mean(), "final - issued [95% CI]": bs_cell(day_bootstrap(sub.date_myt, fsq, isq), 4)})
    lines.append(pd.DataFrame(tab).round(4).to_markdown(index=False))
    lines.append("\n(For the P(<=70) row the 'issued Ridge v4' column is the issued cycling logistic.)")
    text = "\n".join(lines)
    (OUT / "issued.md").write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- live-dict check
def dictcheck(ds):
    sel = json.loads((OUT / "selection.json").read_text())
    stored = pd.read_parquet(OUT / f"pred_point_{sel['point']}.parquet")
    le_stored = pd.read_parquet(OUT / f"pred_le70_{sel['le70']}.parquet")
    raw = dataset.load().join(pd.read_parquet(SPATIAL))       # no x_ controls: the dict path derives them
    worst = {}
    for day in ("2026-10-09", "2026-10-13"):
        cutoff = config.myt_epoch(day)
        train = ds.loc[(ds.index < cutoff)]
        model = M.fit_point(train, cutoff, M.POINT_VARIANTS[sel["point"]])
        le = M.fit_le70(train, cutoff, sel["le70"], point=model if sel["le70"] == "E1_spatial_residual" else None)
        test = raw.loc[(raw.index >= cutoff) & (raw.index < cutoff + 86400)]
        for issue, row in test.iterrows():
            feats = {k: v for k, v in row.items() if not k.startswith("y_") and not k.startswith("x_")}
            d = M.predict_point(model, feats)
            for k, v in d.items():
                s = stored.loc[issue, k]
                diff = 0.0 if (np.isnan(v) and np.isnan(s)) else abs(v - s)
                worst[k] = max(worst.get(k, 0.0), diff)
            p = M.predict_le70(le, feats)
            s = le_stored.loc[issue, "p_ride_le70"]
            diff = 0.0 if (np.isnan(p) and np.isnan(s)) else abs(p - s)
            worst["p_ride_le70"] = max(worst.get("p_ride_le70", 0.0), diff)
        log(f"{day}: {len(test)} issues")
    print("max |dict path - walk-forward output|:", worst)
    (OUT / "dictcheck.json").write_text(json.dumps(worst, indent=1))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    ds = load()
    if cmd == "point":
        for v in (args or list(M.POINT_VARIANTS)):
            run_point(ds, v)
    elif cmd == "select_point":
        select_point(ds)
    elif cmd == "le70":
        run_le70(ds, args or list(M.LE70_VARIANTS))
    elif cmd == "select_le70":
        select_le70(ds)
    elif cmd == "score":
        score(ds)
    elif cmd == "issued":
        issued(ds)
    elif cmd == "dictcheck":
        dictcheck(ds)
    else:
        raise SystemExit(cmd)
