"""Holdout diagnostics for the chosen candidate (run after final.py; changes no choice).

* Episodes caught by direction (fall / rise) at the incumbent's false-call budgets.
* Reliability: mean forecast vs observed frequency in probability bins.
Writes data\\models\\event-probabilities\\diagnostics.md.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path

import numpy as np
import pandas as pd

import common
import evaluate as E
from harness import scoring


def by_direction(issues, truth, p_neg, p_pos, budget):
    curve = E.threshold_curve(issues, truth, p_neg, p_pos)
    tau = E.at_budget(curve, budget)["tau"]
    score = np.maximum(p_neg, p_pos)
    call = np.where(score >= tau, np.where(p_neg >= p_pos, -1.0, 1.0), 0.0)
    out = {}
    for direction, name in ((-1.0, "fall"), (1.0, "rise")):
        t = np.where(truth == direction, direction, 0.0)
        c = np.where(call == direction, direction, 0.0)       # only calls in this direction
        r = scoring.event_confusion(issues, t, c)
        out[name] = f"{r['episodes_caught']}/{r['episodes']}"
    return out


def directional(issues, truth, call):
    """Per-direction confusion: hits / misses / wrong-way on that direction's events,
    false calls in that direction on non-events, episodes caught."""
    out = {}
    for d, name in ((-1.0, "fall"), (1.0, "rise")):
        ev = truth == d
        t = np.where(ev, d, 0.0)
        c = np.where(call == d, d, 0.0)
        r = scoring.event_confusion(issues, t, c)
        out[name] = (f"{r['episodes_caught']}/{r['episodes']} ep; hits {int((ev & (call == d)).sum())}, "
                     f"miss {int((ev & (call == 0)).sum())}, wrong {int((ev & (call == -d)).sum())}, "
                     f"false {int(((truth == 0) & (call == d)).sum())}")
    return out


def reliability(p, y, edges=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01)):
    p, y = np.asarray(p, float), np.asarray(y, float)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) & np.isfinite(y)
        if m.sum():
            rows.append(f"[{a:.2f},{min(b, 1):.2f}): n={int(m.sum())}, mean p={p[m].mean():.3f}, observed={y[m].mean():.3f}")
    return rows


def main():
    ds = common.load()
    cand = common.load_pred("candidate")
    inc = E.incumbent_frame(ds)
    mask = E.comparison_mask(ds, {"candidate": cand})
    a, b = E.PERIODS["holdout"]
    hold = ds.loc[mask & (ds.index >= a) & (ds.index < b)]
    issues = hold.index.to_numpy(np.int64)
    lines = ["# Holdout diagnostics (2026-09-27..10-13)", "",
             "## Episodes caught by direction at the incumbent's false-call budgets", "",
             "| target | budget (false calls) | incumbent fall | incumbent rise | candidate fall | candidate rise |",
             "|---|---|---|---|---|---|"]
    for label, t, neg, pos in (("arrival >=20", 20, "p_fall20", "p_rise20"), ("arrival >=40", 40, "p_fall40", "p_rise40"),
                               ("first crossing", None, "p_first_drop", "p_first_rise")):
        truth = (scoring.truth_direction(hold.y_delta, t) if t else hold.y_first20.to_numpy(float))
        ip, cp = inc.reindex(hold.index), cand.reindex(hold.index)
        if t:
            inc_calls = {"argmax": scoring.calls_from_probs(ip[neg], ip[pos]), "0.2": scoring.calls_from_probs(ip[neg], ip[pos], "0.2")}
        else:
            arr = ip[["p_first_drop", "p_first_none", "p_first_rise"]].to_numpy(float)
            inc_calls = {"argmax": np.argmax(arr, axis=1) - 1.0}
        for rule, calls in inc_calls.items():
            budget = scoring.event_confusion(issues, truth, calls)["false_calls"]
            i = by_direction(issues, truth, ip[neg].to_numpy(float), ip[pos].to_numpy(float), budget)
            c = by_direction(issues, truth, cp[neg].to_numpy(float), cp[pos].to_numpy(float), budget)
            lines.append(f"| {label} | {budget} (incumbent {rule}) | {i['fall']} | {i['rise']} | {c['fall']} | {c['rise']} |")
    for period in ("holdout", "full"):
        pa, pb = E.PERIODS[period]
        rows = ds.loc[mask & (ds.index >= pa) & (ds.index < pb)]
        iss = rows.index.to_numpy(np.int64)
        lines += ["", f"## Per-direction event confusion, {period} (fixed rules)", "",
                  "| target | rule | model | falls | rises |", "|---|---|---|---|---|"]
        for t in (20, 40):
            truth = scoring.truth_direction(rows.y_delta, t)
            for rule in ("argmax", "0.2"):
                for name, frame in (("persistence", None), ("incumbent", inc), ("candidate", cand)):
                    if frame is None:
                        call = np.zeros(len(rows))
                    else:
                        f = frame.reindex(rows.index)
                        call = scoring.calls_from_probs(f[f"p_fall{t}"], f[f"p_rise{t}"], rule)
                    r = directional(iss, truth, call)
                    lines.append(f"| >={t} | {rule} | {name} | {r['fall']} | {r['rise']} |")
        truth = rows.y_first20.to_numpy(float)
        for name, frame in (("persistence", None), ("incumbent", inc), ("candidate", cand)):
            if frame is None:
                call = np.zeros(len(rows))
            else:
                call = np.argmax(frame.reindex(rows.index)[["p_first_drop", "p_first_none", "p_first_rise"]].to_numpy(float), axis=1) - 1.0
            r = directional(iss, truth, call)
            lines.append(f"| first crossing | argmax | {name} | {r['fall']} | {r['rise']} |")
    lines += ["", "## Reliability (holdout)", ""]
    for col, y in (("p_fall20", "y_fall20"), ("p_rise20", "y_rise20"), ("p_fall40", "y_fall40")):
        for name, frame in (("candidate", cand), ("incumbent", inc)):
            lines.append(f"- {col} {name}: " + "; ".join(reliability(frame.reindex(hold.index)[col], hold[y])))
    for name, frame in (("candidate", cand), ("incumbent", inc)):
        drop = frame.reindex(hold.index)["p_first_drop"]
        lines.append(f"- p_first_drop {name}: " + "; ".join(reliability(drop, (hold.y_first20 == -1).astype(float))))
    text = "\n".join(lines)
    Path(common.OUT / "diagnostics.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
