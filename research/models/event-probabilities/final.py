"""Build the chosen candidate for every evaluation day and report tune / holdout / full.

Usage: python -B final.py
The candidate (chosen in tune.py on 2026-09-10..09-26 only) is defined in
``build``. Writes data\\models\\event-probabilities\\candidate.parquet (standard
prediction columns), final_results.json and final_tables.md.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import time
from pathlib import Path

import calibrate as cal
import common
import evaluate as E

# Chosen on the tune period only (tune.py variant "stack14tb:lr_hgb_all+hgb_all_a0":
# lowest tune 5-class log loss and lowest tune first-crossing log loss); fixed
# before the holdout was scored.
DIST = "lr_hgb_all"
CLF = "hgb_all_a0"
CAL_LOOKBACK_DAYS = 14


def build(ds, base: dict):
    days = common.days_between(ds, common.TUNE_START, common.EVAL_END)
    started = time.time()
    five = cal.calibrate(ds, {DIST: base[DIST], CLF: base[CLF]}, "class5", "tempbias",
                         lookback_days=CAL_LOOKBACK_DAYS, days=days)
    first = cal.calibrate(ds, {CLF: base[CLF]}, "first20", "tempbias", lookback_days=CAL_LOOKBACK_DAYS, days=days)
    seconds = (time.time() - started) / len(days)
    out = five.join(first[common.FIRST3])
    return out, seconds


def prespecified(ds, cand, inc):
    """Out-of-sample operating point: tau fixed on the TUNE period so that the
    candidate's false calls there do not exceed the incumbent's argmax false
    calls on the same tune rows; then applied unchanged to the holdout and
    compared with the incumbent's argmax calls there."""
    import numpy as np
    from harness import scoring
    mask = E.comparison_mask(ds, {"candidate": cand})
    rows = {}
    for period in ("tune", "holdout"):
        a, b = E.PERIODS[period]
        rows[period] = ds.loc[mask & (ds.index >= a) & (ds.index < b)]
    specs = [("arrival >=20", lambda f: scoring.truth_direction(f.y_delta, 20), "p_fall20", "p_rise20",
              lambda p: scoring.calls_from_probs(p.p_fall20, p.p_rise20)),
             ("arrival >=40", lambda f: scoring.truth_direction(f.y_delta, 40), "p_fall40", "p_rise40",
              lambda p: scoring.calls_from_probs(p.p_fall40, p.p_rise40)),
             ("first crossing", lambda f: f.y_first20.to_numpy(float), "p_first_drop", "p_first_rise",
              lambda p: np.argmax(p[["p_first_drop", "p_first_none", "p_first_rise"]].to_numpy(float), axis=1) - 1.0)]
    lines = ["| target | tau fixed on tune | incumbent argmax, holdout | candidate at tau, holdout |", "|---|---|---|---|"]
    out = {}
    for label, truth_fn, neg, pos, inc_call in specs:
        tune, hold = rows["tune"], rows["holdout"]
        ip = inc.reindex(tune.index)
        budget = scoring.event_confusion(tune.index, truth_fn(tune), inc_call(ip))["false_calls"]
        cp = cand.reindex(tune.index)
        curve = E.threshold_curve(tune.index.to_numpy(np.int64), truth_fn(tune), cp[neg], cp[pos])
        tau = E.at_budget(curve, budget)["tau"]
        hp = cand.reindex(hold.index)
        score = np.maximum(hp[neg].to_numpy(float), hp[pos].to_numpy(float))
        direction = np.where(hp[neg].to_numpy(float) >= hp[pos].to_numpy(float), -1.0, 1.0)
        mine = scoring.event_confusion(hold.index, truth_fn(hold), np.where(score >= tau, direction, 0.0))
        theirs = scoring.event_confusion(hold.index, truth_fn(hold), inc_call(inc.reindex(hold.index)))
        out[label] = {"tau": tau, "tune_budget": budget, "candidate": mine, "incumbent": theirs}
        lines.append(f"| {label} | {tau:.3f} (tune budget {budget} false) | {E._ev(theirs)} | {E._ev(mine)} |")
    return "\n".join(lines), out


def main():
    ds = common.load()
    base = {n: common.load_pred(f"base_{n}") for n in (DIST, CLF, "lr_emp_all")}
    cand, cal_seconds = build(ds, base)
    common.save(cand, "candidate")
    raw = {"raw:" + DIST: base[DIST], "raw:" + CLF: base[CLF]}
    models = {"candidate": cand, **raw}
    results = E.run(ds, models, periods=("tune", "holdout", "full", "issued_era"))
    names = ["prior", "persistence", "incumbent"] + list(models)
    text = []
    for period in ("tune", "holdout", "full", "issued_era"):
        text += [f"## {period}", "", E.prob_table(results, period, names), "", E.event_table(results, period, names), "",
                 E.matched_table(results, period, ["incumbent", "prior"] + list(models)), ""]
    inc = E.incumbent_frame(ds)
    pre_table, pre = prespecified(ds, cand, inc)
    text += ["## Pre-specified operating point (tau fixed on tune, scored on holdout)", "", pre_table, ""]
    text += ["## Big-event cases", "", E.case_table(ds, {"candidate": cand, "incumbent": inc}), ""]
    text += [f"Calibrator fit: {cal_seconds:.2f} s per daily refit (both heads)."]
    Path(common.OUT / "final_tables.md").write_text("\n".join(text), encoding="utf-8")
    payload = E.to_json(results)
    payload["calibrator_seconds_per_refit"] = cal_seconds
    payload["prespecified"] = json.loads(json.dumps(pre, default=float))
    Path(common.OUT / "final_results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
