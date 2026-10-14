"""Walk-forward runs and scoring for the GBM-quantile candidate.

Run from research\\models with the system interpreter (set OMP_NUM_THREADS=1;
xgboost runs single-threaded, which was 3x faster than 2 threads on the loaded laptop):

    python -B gbm-quantile/evaluate.py run <config> <split>      # daily refits -> data/models/gbm-quantile/<config>__<split>.parquet
    python -B gbm-quantile/evaluate.py score <spec> <split>      # vs persistence/prior and the incumbent replay
    python -B gbm-quantile/evaluate.py cases <spec> <split>      # big-event cases (harness.report.CASES)
    python -B gbm-quantile/evaluate.py issued <config>           # genuinely issued clocks, Oct 8+
    python -B gbm-quantile/evaluate.py tunetable <spec> ...      # tune-split summary on identical rows
    python -B gbm-quantile/evaluate.py poolscan <config>         # tune-split scan of the climatology weights

<spec> is a config name, optionally with '+raw' (climatology weights 0) or
'+recal' (walk-forward PIT recalibration, tried and not adopted).

Splits (cutoff days, MYT): tune = 2026-09-10..09-26 (17 days, used for every
choice), test = 2026-09-27..10-13 (17 days, scored once with the frozen choice).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from harness import config as hconfig, dataset, report, scoring, walkforward  # noqa: E402
import gbmq  # noqa: E402

OUT = hconfig.OUT / "gbm-quantile"
SPLITS = {"tune": ("2026-09-10", "2026-09-27"), "test": ("2026-09-27", "2026-10-14"),
          "all": ("2026-09-10", "2026-10-14")}

BASE = gbmq.Config(name="base")      # sensor (no 3-min bins) + time + weather + CAMS, y - ref, depth 4, 100 rounds
CONFIGS = {c.name: c for c in [
    BASE,
    BASE.with_(name="log", transform="log"),                                # log-ratio target
    BASE.with_(name="sensor_only", groups=("sensor", "time")),              # no weather/CAMS
]}
# Round 2 (round 1 picked the log-ratio target on the tune split; the remaining
# difference-target variants were stopped unfinished and are not reported)
LOG = CONFIGS["log"]
CONFIGS.update({c.name: c for c in [
    LOG.with_(name="log_sensor_only", groups=("sensor", "time")),
    LOG.with_(name="log_wide", groups=("sensor", "time", "weather", "cams", "subang", "regional_wind")),
    LOG.with_(name="log_seq", drop_sequence=False),
    LOG.with_(name="log_shallow", max_depth=2, rounds=200),
    LOG.with_(name="log_tails", arrival_q=(0.02, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.98),
              ride_mean_q=(0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)),
]})
# Frozen choice from the tune split (see REPORT.md): log-ratio target, sensor (no 3-min bins) + time +
# weather + CAMS, depth 4 / 100 rounds; climatology weights by lowest tune log loss (pool_scan).
CONFIGS["final"] = LOG.with_(name="final", clim_weight_class5=0.3, clim_weight_le70=0.05)
TUNE_HEADS = ("arrival", "ride_mean")   # tuning runs fit these two heads only (compute)


def split_days(split):
    a, b = SPLITS[split]
    return walkforward.eval_days(load_ds(), hconfig.myt_epoch(a), hconfig.myt_epoch(b))


_DS = None


def load_ds():
    global _DS
    if _DS is None:
        _DS = dataset.load()
    return _DS


def pred_path(name, split):
    return OUT / f"{name}__{split}.parquet"


def run(name, split):
    cfg = CONFIGS[name]
    if split == "tune":
        cfg = cfg.with_(heads=TUNE_HEADS)
    ds = load_ds()
    cols = gbmq.columns(ds, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fit_predict = gbmq.make_fit_predict(cfg, cols)
    pred = walkforward.walk_forward(ds, fit_predict, targets=("arrival",), lookback_days=cfg.lookback_days,
                                    embargo_minutes=gbmq.ARRIVAL_EMBARGO_SECONDS // 60, days=split_days(split),
                                    log=lambda s: print(f"[{name}/{split}] {s}", flush=True))
    pred.to_parquet(pred_path(name, split))
    meta = {"config": {k: v for k, v in cfg.__dict__.items() if k != "extra"}, "features": len(cols),
            "split": split, "days": len(split_days(split)), "seconds": round(time.time() - started, 1)}
    (OUT / f"{name}__{split}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"saved {pred_path(name, split)} rows={len(pred)} in {meta['seconds']} s", flush=True)
    return pred


# ----------------------------------------------------------------------------- scoring
def _f(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return str(int(v)) if isinstance(v, (int, np.integer)) else f"{v:.{d}f}"


def _ev(e):
    if not e or not e.get("n_issues"):
        return "–"
    return (f"{e['episodes_caught']}/{e['episodes']} ep; hits {e['hits']}, miss {e['misses']}, "
            f"wrong {e['wrong_way']}, false {e['false_calls']} ({e['false_clusters']} cl)")


def evaluation_frame(split):
    ds = load_ds()
    a, b = (hconfig.myt_epoch(s) for s in SPLITS[split])
    idx = ds.index.to_numpy(np.int64)
    return ds.loc[ds["eval"].to_numpy(bool) & ds.m_fresh_available.to_numpy(bool) & (idx >= a) & (idx < b)]


def compare(pred: pd.DataFrame, split: str, label="candidate"):
    """Rows: (target, metric, persistence/prior, incumbent replay, candidate, n) on identical rows."""
    ds = load_ds()
    ev = evaluation_frame(split)
    inc = report.standard_replay(ds).reindex(ev.index)
    mine = pred.reindex(ev.index)
    per = scoring.persistence_predictions(ev)
    rows, summary = [], {}
    for title, col, key, y in (("+90 arrival", "pred_arrival", "arrival", "y_arrival"),
                               ("ride mean +90..210", "pred_ride_mean", "ride_mean", "y_ride_mean"),
                               ("ride min", "pred_ride_min", "ride_min", "y_ride_min"),
                               ("ride max", "pred_ride_max", "ride_max", "y_ride_max")):
        if col not in mine:
            continue
        m = inc[col].notna() & mine[col].notna() & ev[y].notna()
        p = scoring.score_candidate(ev, per[[col]], mask=m)
        r = scoring.score_candidate(ev, inc[[col]], mask=m)
        c = scoring.score_candidate(ev, mine[[col]], mask=m)
        rows.append((title, "MAE", _f(p[key]["mae"]), _f(r[key]["mae"]), _f(c[key]["mae"]), c[key]["n"]))
        rows.append((title, "skill vs persistence", "0", _f(r[key]["skill_vs_persistence"], 3),
                     _f(c[key]["skill_vs_persistence"], 3), c[key]["n"]))
        rows.append((title, "RMSE", _f(p[key]["rmse"]), _f(r[key]["rmse"]), _f(c[key]["rmse"]), c[key]["n"]))
        rows.append((title, "MAE where |true change|>=20", _f(p[key]["mae_change20"]), _f(r[key]["mae_change20"]),
                     _f(c[key]["mae_change20"]), c[key]["n_change20"]))
        summary[f"{key}_mae"] = c[key]["mae"]
        summary[f"{key}_mae_inc"] = r[key]["mae"]
        summary[f"{key}_mae_per"] = p[key]["mae"]
        if key == "ride_min" and "pred_ride_min_q10" in mine:
            tau, extra = 0.1, "pred_ride_min_q10"
        elif key == "ride_max" and "pred_ride_max_q90" in mine:
            tau, extra = 0.9, "pred_ride_max_q90"
        else:
            tau = None
        if tau is not None:
            yy = ev.loc[m, y]
            rows.append((title, f"pinball q{tau:g} (candidate: q{tau:g} head | median head)",
                         _f(scoring.pinball(yy, per.loc[m, col], tau)), _f(scoring.pinball(yy, inc.loc[m, col], tau)),
                         f"{_f(scoring.pinball(yy, mine.loc[m, extra], tau))} | {_f(scoring.pinball(yy, mine.loc[m, col], tau))}",
                         int(m.sum())))
            cover = float((yy <= mine.loc[m, extra]).mean())
            rows.append((title, f"coverage P(y <= q{tau:g} head)", "–", "–", _f(cover, 3), int(m.sum())))
        if key == "arrival":
            for t in (20, 40):
                rows.append((f"arrival change >={t} (point)", "events", _ev(p[f"events{t}_point"]),
                             _ev(r[f"events{t}_point"]), _ev(c[f"events{t}_point"]), c[f"events{t}_point"]["n_issues"]))
                summary[f"ev{t}_point"] = c[f"events{t}_point"]
    if "p_fall20" in mine:
        m = inc.p_fall20.notna() & mine.p_fall20.notna() & ev.y_delta.notna()
        cols = ["p_fall20", "p_rise20", "p_fall40", "p_rise40"] + [f"p_class5_{k}" for k in range(5)]
        r = scoring.score_candidate(ev, inc[cols], mask=m)
        c = scoring.score_candidate(ev, mine[cols], mask=m)
        prior = scoring.walk_forward_prior(ev, "y_class5", range(5), "y_arrival_complete", history=ds)
        prior.columns = [f"p_class5_{k}" for k in range(5)]
        prior["p_fall20"] = prior.p_class5_0 + prior.p_class5_1
        prior["p_rise20"] = prior.p_class5_3 + prior.p_class5_4
        prior["p_fall40"], prior["p_rise40"] = prior.p_class5_0, prior.p_class5_4
        q = scoring.score_candidate(ev, prior, mask=m)
        for t in (20, 40):
            for rule in ("argmax", "0.2"):
                k = f"events{t}_prob_{rule}"
                rows.append((f"arrival change >={t} (probabilities, {rule})", "events", "0 calls (prior: " + _ev(q[k]) + ")",
                             _ev(r[k]), _ev(c[k]), c[k]["n_issues"]))
                summary[f"ev{t}_{rule}"] = c[k]
                summary[f"ev{t}_{rule}_inc"] = r[k]
            for side in ("fall", "rise"):
                k = f"brier_{side}{t}"
                rows.append((f"P({side} >={t})", "Brier", _f(q[k]["brier"], 4), _f(r[k]["brier"], 4), _f(c[k]["brier"], 4), c[k]["n"]))
                yy = ev.loc[m, f"y_{side}{t}"].to_numpy(float)
                if 0 < yy.sum() < len(yy):
                    auc_r = roc_auc_score(yy, inc.loc[m, f"p_{side}{t}"].to_numpy(float))
                    auc_c = roc_auc_score(yy, mine.loc[m, f"p_{side}{t}"].to_numpy(float))
                    rows.append((f"P({side} >={t})", "ROC AUC (discrimination)", "0.5", _f(auc_r, 3), _f(auc_c, 3), c[k]["n"]))
                    summary[f"auc_{side}{t}"], summary[f"auc_{side}{t}_inc"] = auc_c, auc_r
                summary[f"brier_{side}{t}"] = c[k]["brier"]
                summary[f"brier_{side}{t}_inc"] = r[k]["brier"]
                summary[f"brier_{side}{t}_prior"] = q[k]["brier"]
        for metric in ("brier", "log_loss"):
            rows.append(("5-class arrival change", metric.replace("_", " "), _f(q["class5"][metric], 4),
                         _f(r["class5"][metric], 4), _f(c["class5"][metric], 4), c["class5"]["n"]))
            summary[f"class5_{metric}"] = c["class5"][metric]
            summary[f"class5_{metric}_inc"] = r["class5"][metric]
            summary[f"class5_{metric}_prior"] = q["class5"][metric]
    if "p_ride_le70" in mine:
        m = inc.p_ride_le70.notna() & mine.p_ride_le70.notna() & ev.y_ride_le70.notna()
        p = scoring.score_candidate(ev, per[["p_ride_le70"]], mask=m)
        r = scoring.score_candidate(ev, inc[["p_ride_le70"]], mask=m)
        c = scoring.score_candidate(ev, mine[["p_ride_le70"]], mask=m)
        prior = scoring.walk_forward_prior(ev, "y_ride_le70", (0.0, 1.0), "y_ride_complete", history=ds)
        q = scoring.score_candidate(ev, prior[["prior_1.0"]].rename(columns={"prior_1.0": "p_ride_le70"}), mask=m)
        rows.append(("P(ride mean <=70)", "Brier (persistence 0/1 | prior)",
                     f"{_f(p['ride_le70']['brier'], 4)} | {_f(q['ride_le70']['brier'], 4)}",
                     _f(r["ride_le70"]["brier"], 4), _f(c["ride_le70"]["brier"], 4), c["ride_le70"]["n"]))
        rows.append(("P(ride mean <=70)", "log loss", f"– | {_f(q['ride_le70']['log_loss'], 4)}",
                     _f(r["ride_le70"]["log_loss"], 4), _f(c["ride_le70"]["log_loss"], 4), c["ride_le70"]["n"]))
        summary["le70_brier"] = c["ride_le70"]["brier"]
        summary["le70_brier_inc"] = r["ride_le70"]["brier"]
        summary["le70_brier_per"] = p["ride_le70"]["brier"]
    return rows, summary


def calibration(pred, split, column, truth_col, bins=(0, .05, .1, .2, .3, .5, 1.0001)):
    ev = evaluation_frame(split)
    p = pred[column].reindex(ev.index)
    y = ev[truth_col]
    ok = p.notna() & y.notna()
    p, y = p[ok], y[ok]
    cut = pd.cut(p, bins, right=False)
    g = pd.DataFrame({"p": p, "y": y, "bin": cut}).groupby("bin", observed=True)
    return g.agg(n=("y", "size"), mean_p=("p", "mean"), rate=("y", "mean")).reset_index()


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def load_pred(name, split, recal=False):
    """Candidate predictions; with recal=True the probability columns come from
    the walk-forward PIT recalibration fitted on this config's earlier days
    (tune-split forecasts are the history for the first test days)."""
    if name == "final" and split == "tune" and not pred_path(name, split).exists():
        # The final config is the 'log' fit plus a post-hoc climatology pool, so its
        # tune-split forecasts are the 'log' tune run pooled with the final weights.
        cfg = CONFIGS["final"]
        pred = pool(pd.read_parquet(pred_path("log", "tune")), cfg, cfg.clim_weight_class5, cfg.clim_weight_le70)
    else:
        pred = pd.read_parquet(pred_path(name, split))
    if not recal:
        return pred
    cfg = CONFIGS[name]
    history = [pd.read_parquet(pred_path(name, s)) for s in ("tune", "test") if pred_path(name, s).exists()]
    allp = pd.concat(history + [pred])
    allp = allp.loc[~allp.index.duplicated(keep="last")].sort_index()
    cal = gbmq.recalibrate(allp, load_ds(), cfg.arrival_q, cfg.ride_mean_q).reindex(pred.index)
    out = pred.copy()
    for column in cal.columns:
        out[column] = cal[column]
    return out


def load_variant(spec, split):
    """'name', 'name+recal' (PIT recalibration) or 'name+raw' (climatology weights 0)."""
    name = spec.split("+")[0]
    pred = load_pred(name, split, recal=spec.endswith("+recal"))
    if spec.endswith("+raw"):
        pred = pool(pred, CONFIGS[name], 0.0, 0.0)
    return pred


def coverage(pred, split):
    ev = evaluation_frame(split)
    rows = []
    for head, y in (("arrival", "y_arrival"), ("ride_mean", "y_ride_mean")):
        cols = [c for c in pred.columns if c.startswith(f"q_{head}_")]
        if not cols:
            continue
        sub = pred.reindex(ev.index)
        ok = sub[cols].notna().all(axis=1) & ev[y].notna()
        rows.append((head, int(ok.sum()), " ".join(f"{c.rsplit('_', 1)[1]}:{(ev.loc[ok, y] <= sub.loc[ok, c]).mean():.3f}"
                                                  for c in cols)))
    return rows


def score(spec, split):
    pred = load_variant(spec, split)
    rows, summary = compare(pred, split)
    tag = spec
    print(f"## {tag} on {split} ({SPLITS[split][0]} .. {SPLITS[split][1]} exclusive)\n")
    print(table(["Target", "Metric", "Persistence / prior", "Incumbent replay", f"GBM-quantile ({tag})", "n"], rows))
    print("\nNominal level: empirical share of y <= predicted quantile\n")
    print(table(["Head", "n", "coverage"], coverage(pred, split)))
    return rows, summary


# ----------------------------------------------------------------------------- issued clocks (Oct 8+)
def issued(name):
    """Refit daily on the grid (same rule as walk_forward) and predict at the exact issued clocks.

    Compared with the genuinely issued incumbent outputs and the walk-forward
    prior, on the rows where the published model version is the current
    incumbent (as harness.report.issued_rows).
    """
    from harness import issued as hissued
    from harness.incumbents import run as runs
    cfg = CONFIGS[name]
    ds = load_ds()
    cols = gbmq.columns(ds, cfg)
    frame = runs.issued_clock_frame()
    path = OUT / f"{name}__issued.parquet"
    if path.exists():
        pred = pd.read_parquet(path)
    else:
        fit_predict = gbmq.make_fit_predict(cfg, cols)
        index = ds.index.to_numpy(np.int64)
        days = sorted({hconfig.midnight(i) for i in frame.index})
        parts = []
        for cutoff in days:
            started = time.time()
            train_mask = ((index > cutoff - cfg.lookback_days * 86400)
                          & (index < cutoff - gbmq.ARRIVAL_EMBARGO_SECONDS)
                          & (ds["y_arrival_complete"].to_numpy(np.int64) < cutoff))
            fi = frame.index.to_numpy(np.int64)
            test = frame.loc[(fi >= cutoff) & (fi < cutoff + 86400)]
            test = test.drop(columns=[c for c in test.columns if c.startswith("y_") or c.startswith("issued__")])
            parts.append(fit_predict(ds.loc[train_mask], test, cutoff))
            print(f"[issued] {hconfig.myt(cutoff):%Y-%m-%d} test={len(test)} {time.time() - started:.1f}s", flush=True)
        pred = pd.concat(parts)
        pred.to_parquet(path)
    pred = pred.reindex(frame.index)
    rows = []
    ridge_ok = ((frame.issued__arrival_model == hissued.INCUMBENT_VERSIONS["ridge"])
                & (frame.index >= report.ISSUED_START["ridge"]))
    for title, col, issued_col, y in (("+90 arrival", "pred_arrival", "issued__arrival_point", "y_arrival"),
                                      ("ride mean", "pred_ride_mean", "issued__ride_mean_point", "y_ride_mean"),
                                      ("ride min", "pred_ride_min", "issued__ride_min_point", "y_ride_min"),
                                      ("ride max", "pred_ride_max", "issued__ride_max_point", "y_ride_max")):
        m = ridge_ok & frame[issued_col].notna() & pred[col].notna() & frame[y].notna()
        sub = frame.loc[m]
        i = scoring.point_metrics(sub[y], sub[issued_col], sub.y_ref)
        c = scoring.point_metrics(sub[y], pred.loc[m, col], sub.y_ref)
        rows.append((title, "MAE", _f(i.get("persistence_mae")), _f(i.get("mae")), _f(c.get("mae")), c.get("n", 0)))
        if title == "+90 arrival":
            for t in (20, 40):
                truth = scoring.truth_direction(sub.y_delta, t)
                issues = sub.index.to_numpy(np.int64)
                rows.append((f"change >={t} (point)", "events", _ev(scoring.event_confusion(issues, truth, np.zeros(len(sub)))),
                             _ev(scoring.event_confusion(issues, truth, scoring.calls_from_point(sub.issued__arrival_point, sub.y_ref, t))),
                             _ev(scoring.event_confusion(issues, truth, scoring.calls_from_point(pred.loc[m, col], sub.y_ref, t))),
                             len(sub)))
    ac_ok = (frame.issued__ac_model == hissued.INCUMBENT_VERSIONS["arrival_logistic"]) & frame.issued__ac_p_fall20.notna()
    m = ac_ok & pred.p_fall20.notna() & frame.y_delta.notna()
    sub = frame.loc[m]
    issues = sub.index.to_numpy(np.int64)
    prior = scoring.walk_forward_prior(sub, "y_class5", range(5), "y_arrival_complete", history=ds)
    for t in (20, 40):
        truth = scoring.truth_direction(sub.y_delta, t)
        for rule in ("argmax", "0.2"):
            rows.append((f"change >={t} (probabilities, {rule})", "events", "0 calls",
                         _ev(scoring.event_confusion(issues, truth, scoring.calls_from_probs(sub[f"issued__ac_p_fall{t}"], sub[f"issued__ac_p_rise{t}"], rule))),
                         _ev(scoring.event_confusion(issues, truth, scoring.calls_from_probs(pred.loc[m, f"p_fall{t}"], pred.loc[m, f"p_rise{t}"], rule))),
                         len(sub)))
        classes = {("fall", 20): [0, 1], ("rise", 20): [3, 4], ("fall", 40): [0], ("rise", 40): [4]}
        for side in ("fall", "rise"):
            pp = prior[[f"prior_{k}" for k in classes[(side, t)]]].sum(axis=1)
            yy = sub[f"y_{side}{t}"]
            rows.append((f"P({side} >={t})", "Brier (prior | issued | candidate)", _f(scoring.binary_brier(pp, yy)["brier"], 4),
                         _f(scoring.binary_brier(sub[f"issued__ac_p_{side}{t}"], yy)["brier"], 4),
                         _f(scoring.binary_brier(pred.loc[m, f"p_{side}{t}"], yy)["brier"], 4), len(sub)))
    cy_ok = (frame.issued__cy_model == hissued.INCUMBENT_VERSIONS["cycling70"]) & frame.issued__cy_p_le70.notna()
    m = cy_ok & pred.p_ride_le70.notna() & frame.y_ride_le70.notna()
    sub = frame.loc[m]
    rows.append(("P(ride mean <=70)", "Brier (persistence 0/1 | issued | candidate)",
                 _f(scoring.binary_brier((sub.y_ref <= 70).astype(float), sub.y_ride_le70)["brier"], 4),
                 _f(scoring.binary_brier(sub.issued__cy_p_le70, sub.y_ride_le70)["brier"], 4),
                 _f(scoring.binary_brier(pred.loc[m, "p_ride_le70"], sub.y_ride_le70)["brier"], 4), len(sub)))
    print(table(["Target", "Metric", "Persistence / prior", "Issued incumbent", f"GBM-quantile ({name})", "n"], rows))
    return rows


# ----------------------------------------------------------------------------- big-event cases
def cases(pred):
    """Per big-event case (harness.report.CASES): candidate vs incumbent around the onset.

    Onset = first grid issue whose +90 change is >=20 in the case direction.
    First call = minutes from onset to the first right-direction call, searching
    from onset-60 to the end of the episode (negative = before onset).
    """
    ds = load_ds()
    inc = report.standard_replay(ds)
    rows = []
    for label, start, end, direction in report.CASES:
        a, b = hconfig.myt_epoch(start), hconfig.myt_epoch(end)
        part = ds.loc[(ds.index >= a) & (ds.index <= b)]
        truth = scoring.truth_direction(part.y_delta, 20)
        hits = part.index[truth == direction]
        if not len(hits):
            continue
        onset = int(hits[0])
        window = ds.loc[(ds.index >= onset - 3600) & (ds.index <= int(hits[-1]))]
        side = "fall" if direction < 0 else "rise"

        def first(calls):
            idx = np.flatnonzero(np.asarray(calls) == direction)
            return f"{(int(window.index[idx[0]]) - onset) / 60:+.0f}" if len(idx) else "never"

        def calls(frame, rule):
            f = frame.reindex(window.index)
            if rule == "point":
                return scoring.calls_from_point(f.pred_arrival, window.y_ref, 20)
            return scoring.calls_from_probs(f.p_fall20, f.p_rise20, rule)

        def at(frame, column, epoch):
            v = frame[column].get(epoch, np.nan) if column in frame else np.nan
            return v

        o = ds.loc[onset]
        rows.append((label, str(hconfig.myt(onset))[5:16], f"{o.y_ref:.0f} -> {o.y_arrival:.0f}",
                     f"{at(inc, 'pred_arrival', onset):.0f} / {at(pred, 'pred_arrival', onset):.0f}",
                     f"{at(inc, f'p_{side}20', onset - 1800):.2f} / {at(inc, f'p_{side}20', onset):.2f}",
                     f"{at(pred, f'p_{side}20', onset - 1800):.2f} / {at(pred, f'p_{side}20', onset):.2f}",
                     f"{first(calls(inc, 'point'))} / {first(calls(pred, 'point'))}",
                     f"{first(calls(inc, 'argmax'))} / {first(calls(pred, 'argmax'))}",
                     f"{first(calls(inc, '0.2'))} / {first(calls(pred, '0.2'))}"))
    header = ["case", "onset", "ref -> +90", "point at onset (Ridge / cand.)",
              "logistic P(dir20) onset-30 / onset", "candidate P(dir20) onset-30 / onset",
              "first point call (Ridge / cand.)", "first argmax call (logistic / cand.)",
              "first p>=0.2 call (logistic / cand.)"]
    return header, rows


def tune_table(specs, split="tune"):
    """Key metrics for several configs ('name' or 'name+recal') on the rows where all have outputs."""
    preds = {spec: load_variant(spec, split) for spec in specs}
    common = None
    for pred in preds.values():
        ok = pred.p_fall20.notna() & pred.pred_arrival.notna()
        common = ok if common is None else common.reindex(ok.index).fillna(False) & ok
    rows = []
    for spec, pred in preds.items():
        pred = pred.where(common.reindex(pred.index).fillna(False), np.nan, axis=0)
        _, m = compare(pred, split)
        e20a, e20b = m["ev20_argmax"], m["ev20_0.2"]
        rows.append((spec, _f(m["arrival_mae"]), _f(m["ride_mean_mae"]), _f(m["brier_fall20"], 4), _f(m["brier_rise20"], 4),
                     _f(m["auc_fall20"], 3), _f(m["auc_rise20"], 3), _f(m["class5_brier"], 4), _f(m["class5_log_loss"], 4),
                     f"{e20a['episodes_caught']}/{e20a['episodes']}, false {e20a['false_calls']}",
                     f"{e20b['episodes_caught']}/{e20b['episodes']}, false {e20b['false_calls']}",
                     _f(m.get("le70_brier"), 4)))
    rows.append(("persistence / prior", _f(m["arrival_mae_per"]), _f(m["ride_mean_mae_per"]), _f(m["brier_fall20_prior"], 4),
                 _f(m["brier_rise20_prior"], 4), "0.5", "0.5", _f(m["class5_brier_prior"], 4), _f(m["class5_log_loss_prior"], 4),
                 "0", "0", _f(m.get("le70_brier_per"), 4)))
    rows.append(("incumbent replay", _f(m["arrival_mae_inc"]), _f(m["ride_mean_mae_inc"]), _f(m["brier_fall20_inc"], 4),
                 _f(m["brier_rise20_inc"], 4), _f(m["auc_fall20_inc"], 3), _f(m["auc_rise20_inc"], 3), _f(m["class5_brier_inc"], 4),
                 _f(m["class5_log_loss_inc"], 4),
                 f"{m['ev20_argmax_inc']['episodes_caught']}/{m['ev20_argmax_inc']['episodes']}, false {m['ev20_argmax_inc']['false_calls']}",
                 f"{m['ev20_0.2_inc']['episodes_caught']}/{m['ev20_0.2_inc']['episodes']}, false {m['ev20_0.2_inc']['false_calls']}",
                 _f(m.get("le70_brier_inc"), 4)))
    header = ["config", "+90 MAE", "ride mean MAE", "Brier fall20", "Brier rise20", "AUC fall20", "AUC rise20",
              "5-class Brier", "5-class log loss", ">=20 argmax ep, false", ">=20 p>=0.2 ep, false", "P(<=70) Brier"]
    print(f"rows: {int(common.sum())}")
    print(table(header, rows))
    return header, rows


# ----------------------------------------------------------------------------- climatology pool (tuning)
def daily_climatology(cfg, days):
    """Per-day climatology exactly as gbmq.make_fit_predict computes it from its training rows."""
    ds = load_ds()
    index = ds.index.to_numpy(np.int64)
    fresh = ds.m_fresh_available.to_numpy(bool) & np.isfinite(ds.s_freshPm25.to_numpy(float))
    out = {}
    for cutoff in days:
        window = (index > cutoff - cfg.lookback_days * 86400)
        arr = (window & (index < cutoff - gbmq.ARRIVAL_EMBARGO_SECONDS) & fresh
               & (ds.y_arrival_complete.to_numpy(np.int64) < cutoff) & np.isfinite(ds.y_arrival.to_numpy(float)))
        labels = ds.y_class5.to_numpy(float)[arr]
        labels = labels[(labels >= 0) & (labels <= 4)]
        ride = (window & (index < cutoff - gbmq.RIDE_EMBARGO_SECONDS) & fresh & ds.y_ride_joint_valid.astype(bool).to_numpy()
                & (ds.y_ride_complete.to_numpy(np.int64) < cutoff) & (ds.y_arrival_complete.to_numpy(np.int64) < cutoff)
                & np.isfinite(ds.y_ride_mean.to_numpy(float)))
        out[cutoff] = (np.bincount(labels.astype(int), minlength=5) / max(len(labels), 1),
                       float(np.mean(ds.y_ride_mean.to_numpy(float)[ride] <= 70.0)))
    return out


def pool(pred, cfg, w_class, w_le70):
    """Re-read probabilities from the raw quantile-derived ones with climatology weights."""
    out = pred.copy()
    raw = [f"p_class5_raw_{k}" if f"p_class5_raw_{k}" in pred else f"p_class5_{k}" for k in range(5)]
    raw70 = "p_ride_le70_raw" if "p_ride_le70_raw" in pred else "p_ride_le70"
    days = np.array([hconfig.midnight(i) for i in pred.index])
    clim = daily_climatology(cfg, sorted(set(days)))
    c5 = np.vstack([clim[d][0] for d in days])
    c70 = np.array([clim[d][1] for d in days])
    P = pred[raw].to_numpy(float)
    M = (1 - w_class) * P + w_class * c5
    for k in range(5):
        out[f"p_class5_{k}"] = M[:, k]
    out["p_fall20"], out["p_rise20"] = M[:, 0] + M[:, 1], M[:, 3] + M[:, 4]
    out["p_fall40"], out["p_rise40"] = M[:, 0], M[:, 4]
    if raw70 in pred:
        out["p_ride_le70"] = (1 - w_le70) * pred[raw70].to_numpy(float) + w_le70 * c70
    return out


def pool_scan(name, split="tune", weights=(0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)):
    """Tune-split scan of the climatology weights (choice = lowest log loss of the head)."""
    cfg = CONFIGS[name]
    pred = load_pred(name, split)
    ev = evaluation_frame(split)
    rows = []
    for w in weights:
        pooled = pool(pred, cfg, w, w).reindex(ev.index)
        P = pooled[[f"p_class5_{k}" for k in range(5)]].to_numpy(float)
        c5 = scoring.multiclass_scores(P, ev.y_class5, range(5))
        t = scoring.truth_direction(ev.y_delta, 20)
        ok = np.isfinite(P).all(axis=1)
        e = scoring.event_confusion(ev.index[ok], t[ok], scoring.calls_from_probs(pooled.p_fall20[ok], pooled.p_rise20[ok], "0.2"))
        a = scoring.event_confusion(ev.index[ok], t[ok], scoring.calls_from_probs(pooled.p_fall20[ok], pooled.p_rise20[ok], "argmax"))
        le = scoring.binary_brier(pooled.p_ride_le70, ev.y_ride_le70)
        rows.append((w, _f(c5["brier"], 4), _f(c5["log_loss"], 4),
                     _f(scoring.binary_brier(pooled.p_fall20, ev.y_fall20)["brier"], 4),
                     _f(scoring.binary_brier(pooled.p_rise20, ev.y_rise20)["brier"], 4),
                     f"{a['episodes_caught']}/{a['episodes']}, false {a['false_calls']}",
                     f"{e['episodes_caught']}/{e['episodes']}, false {e['false_calls']}",
                     _f(le["brier"], 4), _f(le["log_loss"], 4), c5["n"]))
    header = ["w", "5-class Brier", "5-class log loss", "Brier fall20", "Brier rise20", ">=20 argmax ep, false",
              ">=20 p>=0.2 ep, false", "P(<=70) Brier", "P(<=70) log loss", "n"]
    print(table(header, rows))
    return header, rows


if __name__ == "__main__":
    action, name = sys.argv[1], sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "all"
    if action == "poolscan":
        pool_scan(sys.argv[2])
        sys.exit()
    if action == "tunetable":
        tune_table(sys.argv[2:])
        sys.exit()
    if action == "issued":
        issued(name)
    elif action == "run":
        run(name, split)
    elif action == "score":
        score(name, split)
    elif action == "cases":
        header, rows = cases(load_variant(name, split))
        print(table(header, rows))
    elif action == "both":
        run(name, split)
        score(name, split)
