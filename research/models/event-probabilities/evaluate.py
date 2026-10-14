"""Score candidate event probabilities against the prior, persistence and the incumbents.

Identical rows for every model in a comparison: evaluation issues with a fresh
reference where the Ridge replay produced an output (the harness standard
mask), the arrival logistic and the first-crossing replay produced outputs and
every candidate being compared produced an output.

Periods: tune = 2026-09-10..09-26 (choices were made here only), holdout =
2026-09-27..10-13 (reported, never used for a choice), full = both.

Matched false-call comparison: for a probability pair (fall, rise) the call
rule "call the larger direction when its probability >= tau" is swept over tau;
at a false-call budget B (the incumbent's argmax false-call count on the same
rows) each model is read at its most lenient tau with false calls <= B. The
threshold is chosen per model on the scored rows, so this is a ranking
diagnostic (like a point on a ROC curve), not a deployable rule.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common
from harness import config, report, scoring

PERIODS = {"tune": (common.TUNE_START, common.HOLDOUT_START),
           "holdout": (common.HOLDOUT_START, common.EVAL_END),
           "full": (common.TUNE_START, common.EVAL_END),
           # Sub-period of the holdout where the logistic was genuinely issued (replay = issued in 99.2%).
           "issued_era": (config.myt_epoch("2026-10-08 19:15"), common.EVAL_END)}


# ----------------------------------------------------------------------------- reference frames
def incumbent_frame(ds):
    replay = report.standard_replay(ds)
    keep = [c for c in replay.columns if c.startswith("p_") or c == "pred_arrival"]
    return replay[keep]


def prior_frame(ds):
    ev = ds.loc[ds["eval"]]
    p5 = scoring.walk_forward_prior(ev, "y_class5", list(range(5)), "y_arrival_complete", history=ds)
    p3 = scoring.walk_forward_prior(ev, "y_first20", [-1, 0, 1], "y_first20_complete", history=ds)
    out = pd.DataFrame(index=ev.index)
    for k in range(5):
        out[f"p_class5_{k}"] = p5[f"prior_{k}"]
    out["p_first_drop"], out["p_first_none"], out["p_first_rise"] = p3["prior_-1"], p3["prior_0"], p3["prior_1"]
    return common.with_tails(out)


def comparison_mask(ds, frames: dict) -> pd.Series:
    ev = ds["eval"] & ds["m_fresh_available"]
    inc = incumbent_frame(ds)
    mask = ev & inc["pred_arrival"].notna() & inc["p_fall20"].notna() & inc["p_first_drop"].notna()
    for f in frames.values():
        f = f.reindex(ds.index)
        for c in ("p_fall20", "p_first_drop"):
            if c in f:
                mask &= f[c].notna()
    return mask


# ----------------------------------------------------------------------------- matched false-call curves
def threshold_curve(issues, truth, p_neg, p_pos, taus=None):
    p_neg, p_pos = np.asarray(p_neg, float), np.asarray(p_pos, float)
    score = np.maximum(p_neg, p_pos)
    direction = np.where(p_neg >= p_pos, -1.0, 1.0)
    if taus is None:
        taus = np.unique(np.concatenate([np.quantile(score, np.linspace(0.5, 1, 300)), [1.01]]))
    rows = []
    for tau in taus:
        call = np.where(score >= tau, direction, 0.0)
        r = scoring.event_confusion(issues, truth, call)
        rows.append({"tau": float(tau), "false": r["false_calls"], "caught": r["episodes_caught"],
                     "episodes": r["episodes"], "hits": r["hits"], "wrong": r["wrong_way"],
                     "false_clusters": r["false_clusters"]})
    return pd.DataFrame(rows)


def at_budget(curve: pd.DataFrame, budget: int) -> dict:
    ok = curve.loc[curve["false"] <= budget]
    if ok.empty:
        return {"tau": np.nan, "false": 0, "caught": 0, "hits": 0, "wrong": 0}
    best = ok.sort_values(["false", "caught"]).iloc[-1]
    return {k: (float(best[k]) if k == "tau" else int(best[k])) for k in ("tau", "false", "caught", "hits", "wrong")}


# ----------------------------------------------------------------------------- one model, one period
def evaluate(frame: pd.DataFrame, pred: pd.DataFrame, budgets: dict | None = None) -> dict:
    """frame: dataset rows to score (already masked); pred: standard columns."""
    pred = pred.reindex(frame.index)
    res = scoring.score_candidate(frame, pred)
    issues = frame.index.to_numpy(np.int64)
    out = {"n": len(frame)}
    for tail in ("fall20", "rise20", "fall40", "rise40"):
        if f"brier_{tail}" in res:
            out[f"brier_{tail}"] = res[f"brier_{tail}"]["brier"]
            out[f"logloss_{tail}"] = res[f"brier_{tail}"]["log_loss"]
    if "class5" in res:
        out["brier_class5"], out["logloss_class5"] = res["class5"]["brier"], res["class5"]["log_loss"]
    if "first20" in res:
        out["brier_first20"], out["logloss_first20"] = res["first20"]["brier"], res["first20"]["log_loss"]
    for key in ("events20_prob_argmax", "events20_prob_0.2", "events40_prob_argmax", "events40_prob_0.2",
                "first20_events"):
        if key in res:
            out[key] = {k: res[key][k] for k in ("episodes_caught", "episodes", "hits", "misses", "wrong_way",
                                                 "false_calls", "false_clusters", "median_minutes_to_first_hit")}
    curves = {}
    for t in (20, 40):
        if f"p_fall{t}" in pred:
            curves[f"arrival{t}"] = threshold_curve(issues, scoring.truth_direction(frame.y_delta, t),
                                                    pred[f"p_fall{t}"], pred[f"p_rise{t}"])
    if "p_first_drop" in pred:
        curves["first20"] = threshold_curve(issues, frame.y_first20.to_numpy(float), pred.p_first_drop, pred.p_first_rise)
    out["curves"] = curves
    if budgets:
        out["matched"] = {f"{key}@{b}": at_budget(curves[key], b) for key, bs in budgets.items() if key in curves
                          for b in bs}
    return out


def budgets_from(incumbent_result: dict) -> dict:
    """False-call budgets = the incumbent's own false calls under the fixed rules."""
    b = {}
    for key, rules in (("arrival20", ("events20_prob_argmax", "events20_prob_0.2")),
                       ("arrival40", ("events40_prob_argmax", "events40_prob_0.2")),
                       ("first20", ("first20_events",))):
        b[key] = sorted({incumbent_result[r]["false_calls"] for r in rules if r in incumbent_result})
    return b


def run(ds, candidates: dict, periods=("tune", "holdout", "full")) -> dict:
    """candidates: {name: prediction frame}. Returns {period: {model: result}}."""
    inc = incumbent_frame(ds)
    prior = prior_frame(ds)
    persistence = scoring.persistence_predictions(ds)[["p_fall20", "p_rise20", "p_fall40", "p_rise40"]]
    mask = comparison_mask(ds, candidates)
    results = {}
    for period in periods:
        a, b = PERIODS[period]
        rows = mask & (ds.index >= a) & (ds.index < b)
        frame = ds.loc[rows]
        res = {"incumbent": evaluate(frame, inc)}
        budgets = budgets_from(res["incumbent"])
        res["incumbent"] = evaluate(frame, inc, budgets)
        res["prior"] = evaluate(frame, prior, budgets)
        res["persistence"] = evaluate(frame, persistence)
        for name, pred in candidates.items():
            res[name] = evaluate(frame, pred, budgets)
        results[period] = res
    return results


# ----------------------------------------------------------------------------- rendering
def _ev(e):
    if not e:
        return "–"
    return (f"{e['episodes_caught']}/{e['episodes']} ep; hits {e['hits']}, miss {e['misses']}, "
            f"wrong {e['wrong_way']}, false {e['false_calls']} ({e['false_clusters']} cl)")


def prob_table(results: dict, period: str, models) -> str:
    res = results[period]
    cols = ["brier_fall20", "brier_rise20", "brier_fall40", "brier_rise40", "brier_class5", "logloss_class5",
            "brier_first20", "logloss_first20"]
    head = "| model | n | P(fall>=20) Brier | P(rise>=20) Brier | P(fall>=40) Brier | P(rise>=40) Brier | 5-class Brier | 5-class log loss | first-crossing Brier | first-crossing log loss |"
    lines = [head, "|" + "---|" * 10]
    for m in models:
        r = res[m]
        cells = [f"{r[c]:.4f}" if c in r else "–" for c in cols]
        lines.append(f"| {m} | {r['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def event_table(results: dict, period: str, models) -> str:
    res = results[period]
    keys = [("events20_prob_argmax", ">=20 argmax"), ("events20_prob_0.2", ">=20 p>=0.2"),
            ("events40_prob_argmax", ">=40 argmax"), ("events40_prob_0.2", ">=40 p>=0.2"),
            ("first20_events", "first crossing argmax")]
    lines = ["| model | " + " | ".join(k[1] for k in keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for m in models:
        r = res[m]
        lines.append(f"| {m} | " + " | ".join(_ev(r.get(k)) for k, _ in keys) + " |")
    return "\n".join(lines)


def matched_table(results: dict, period: str, models) -> str:
    res = results[period]
    keys = list(res["incumbent"].get("matched", {}).keys())
    lines = ["| model | " + " | ".join(f"{k} false-call budget" for k in keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for m in models:
        r = res[m].get("matched", {})
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append(f"{v['caught']}/{res[m]['curves'][k.split('@')[0]]['episodes'].iloc[0]} ep, hits {v['hits']}, "
                         f"wrong {v['wrong']}, false {v['false']}" if v else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def case_table(ds, preds: dict) -> str:
    """Per big event: P(direction>=20) at onset-60/-30/0, first right call under argmax and p>=0.2."""
    lines = ["| case | onset | model | P(dir>=20) onset-60 / -30 / 0 | first argmax call | first p>=0.2 call | P(first dir) onset | first-crossing first argmax call |",
             "|---|---|---|---|---|---|---|---|"]
    for label, start, end, direction in report.CASES:
        a, b = config.myt_epoch(start), config.myt_epoch(end)
        part = ds.loc[(ds.index >= a) & (ds.index <= b)]
        truth = scoring.truth_direction(part.y_delta, 20)
        hits = part.index[truth == direction]
        if not len(hits):
            continue
        onset = int(hits[0])
        window = ds.loc[(ds.index >= onset - 3600) & (ds.index <= int(hits[-1]))]
        side = "fall" if direction < 0 else "rise"
        first_col = "p_first_drop" if direction < 0 else "p_first_rise"
        for name, pred in preds.items():
            rp = pred.reindex(window.index)

            def first_right(calls):
                idx = np.flatnonzero(np.asarray(calls) == direction)
                return f"{(int(window.index[idx[0]]) - onset) / 60:+.0f}" if len(idx) else "never"

            def at(col, epoch):
                v = pred[col].get(epoch, np.nan) if col in pred else np.nan
                return "–" if not np.isfinite(v) else f"{v:.2f}"

            argmax = scoring.calls_from_probs(rp.p_fall20, rp.p_rise20)
            loose = scoring.calls_from_probs(rp.p_fall20, rp.p_rise20, "0.2")
            first = rp[["p_first_drop", "p_first_none", "p_first_rise"]].to_numpy(float) if first_col in rp else None
            first_call = "–"
            if first is not None:
                calls = np.argmax(np.nan_to_num(first, nan=-1), axis=1) - 1.0
                calls[~np.isfinite(first).all(axis=1)] = np.nan
                first_call = first_right(calls)
            lines.append(f"| {label} | {str(config.myt(onset))[5:16]} | {name} | "
                         f"{at(f'p_{side}20', onset - 3600)} / {at(f'p_{side}20', onset - 1800)} / {at(f'p_{side}20', onset)} | "
                         f"{first_right(argmax)} | {first_right(loose)} | {at(first_col, onset)} | {first_call} |")
    return "\n".join(lines)


def to_json(results: dict) -> dict:
    out = {}
    for period, res in results.items():
        out[period] = {}
        for m, r in res.items():
            out[period][m] = {k: v for k, v in r.items() if k != "curves"}
    return json.loads(json.dumps(out, default=float))
