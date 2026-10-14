"""Walk-forward runs for the diurnal candidate (run from research\models with -B).

    python -B -m diurnal.run tune_kernel grid|hours          # kernel bandwidths, tuning half only
    python -B -m diurnal.run tune_kernel_rest                # kernel alpha, P(<=70), sessions, tuning half only
    python -B -m diurnal.run tune_hgb <groups> <feature sets> <leaf[/lr/iters]> <loss> <lookback> <tag>
    python -B -m diurnal.run final [names] [groups]          # chosen.py configs over the whole eval period
    python -B -m diurnal.run tables                          # final-half (primary) and tuning-half tables, cases
    python -B -m diurnal.run bench                           # fit/predict cost at the 2026-10-13 cutoff
    python -B -m diurnal.run leak_check                      # d_* features unchanged when the future is blanked

Tuning half: 2026-09-10..09-26 MYT (17 days). Final half (never tuned on):
2026-09-27..10-13 MYT (17 days). Outputs go to data\models\diurnal\.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from harness import config, dataset, report, walkforward
from . import evaluate, features as F
from .models import GROUPS, DiurnalHGB, KernelClimatology

OUT = config.OUT / "diurnal"


def load():
    return F.add(dataset.load())


def days_in(ds, period):
    return [d for d in walkforward.eval_days(ds) if period[0] <= d < period[1]]


def run(model, ds, days, groups, lookback_days=28):
    preds, timing = [], {}
    for group in groups:
        started = time.perf_counter()
        pred = walkforward.walk_forward(ds, model.fit_predict(group), lookback_days=lookback_days, days=days,
                                        test_mask=ds.m_fresh_available, log=lambda s: None, **GROUPS[group])
        timing[group] = (time.perf_counter() - started) / max(len(days), 1)
        preds.append(pred)
    return pd.concat(preds, axis=1), timing


def _print(rows):
    frame = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        print(frame.round(4).to_string(index=False))


def tune_kernel(ds, stage="grid"):
    """stage 'grid': hour x level x trend bandwidths; 'hours': broad hour bandwidths at level 0.3, trend 5."""
    days = days_in(ds, evaluate.TUNE)
    replay = report.standard_replay(ds)
    rows = []
    grid = ([(bh, bl, bt) for bh in (0.5, 1.0, 2.0, np.inf) for bl in (0.15, 0.3, np.inf) for bt in (5.0, 15.0, np.inf)]
            if stage == "grid" else [(bh, 0.3, 5.0) for bh in (3.0, 4.0, 6.0, 8.0, 12.0)] + [(np.inf, 0.3, 3.0)])
    for bh, bl, bt in grid:
        model = KernelClimatology(bw_hour=bh, bw_level=bl, bw_trend=bt, alpha=5.0)
        pred, timing = run(model, ds, days, ("near", "events"))
        rows.append({"model": model.name(), **evaluate.quick(ds, pred, evaluate.TUNE, replay),
                     "s_per_day": sum(timing.values())})
        print(rows[-1], flush=True)
    _print(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(OUT / f"tune_kernel_{stage}.json", orient="records", indent=1)


def tune_kernel_rest(ds):
    """Tuning half: alpha for the event heads at the chosen event bandwidths (h6, l0.3, t5);
    bandwidths/alpha for P(ride mean <= 70); session bandwidths."""
    days = days_in(ds, evaluate.TUNE)
    replay = report.standard_replay(ds)
    rows = []
    for alpha in (2.0, 5.0, 20.0, 50.0):
        model = KernelClimatology(bw_hour=6.0, bw_level=0.3, bw_trend=5.0, alpha=alpha)
        pred, _ = run(model, ds, days, ("events",))
        rows.append({"model": f"events_{model.name()}", **evaluate.quick(ds, pred, evaluate.TUNE, replay)})
        print(rows[-1], flush=True)
    for bh in (6.0, np.inf):
        for bt in (3.0, np.inf):
            for alpha in (5.0, 50.0):
                model = KernelClimatology(bw_hour=bh, bw_level=0.3, bw_trend=bt, alpha=alpha)
                pred, _ = run(model, ds, days, ("le70",))
                rows.append({"model": f"le70_{model.name()}", **evaluate.quick(ds, pred, evaluate.TUNE, replay)})
                print(rows[-1], flush=True)
    for bhs in (0.25, 0.5, 1.0, 2.0):
        for bls in (0.15, 0.3, np.inf):
            model = KernelClimatology(bw_hour=np.inf, bw_level=bls, bw_trend=np.inf, bw_hour_session=bhs)
            pred, _ = run(model, ds, days, ("morning", "afternoon"))
            rows.append({"model": f"session_h{bhs}_l{bls}", **evaluate.quick(ds, pred, evaluate.TUNE, replay)})
            print(rows[-1], flush=True)
    _print(rows)
    pd.DataFrame(rows).to_json(OUT / "tune_kernel_rest.json", orient="records", indent=1)


def tune_hgb(ds, groups, feature_sets, leaves, losses, lookbacks, tag):
    """Grid over comma-separated options on the tuning half, e.g.
    tune_hgb near time,diurnal,diurnal_nwp 100 absolute_error 28 stage1_near"""
    days = days_in(ds, evaluate.TUNE)
    replay = report.standard_replay(ds)
    rows = []
    for fs in feature_sets.split(","):
        for leaf in leaves.split(","):                  # "leaf" or "leaf/learning_rate/max_iter"
            leaf, lr, iters = (leaf.split("/") + ["0.1", "100"])[:3] if "/" in leaf else (leaf, "0.1", "100")
            for loss in losses.split(","):
                for lb in lookbacks.split(","):
                    model = DiurnalHGB(feature_set=fs, min_samples_leaf=int(leaf), loss=loss, learning_rate=float(lr),
                                       max_iter=int(iters))
                    pred, timing = run(model, ds, days, tuple(groups.split(",")), lookback_days=int(lb))
                    rows.append({"model": f"{model.name()}_lr{lr}_it{iters}_lb{lb}",
                                 **evaluate.quick(ds, pred, evaluate.TUNE, replay),
                                 "s_per_day": round(sum(timing.values()), 2)})
                    print(rows[-1], flush=True)
    _print(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(OUT / f"tune_hgb_{tag}.json", orient="records", indent=1)


# ----------------------------------------------------------------------------- final
def final(ds, names=None, groups=None):
    """Walk-forward over the whole eval period for the configurations in chosen.py."""
    from .chosen import CHOSEN
    days = days_in(ds, (evaluate.TUNE[0], evaluate.FINAL[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in CHOSEN.items():
        if names and name not in names:
            continue
        for group, (factory, lookback) in spec.items():
            if groups and group not in groups:
                continue
            pred, timing = run(factory(), ds, days, (group,), lookback_days=lookback)
            pred.to_parquet(OUT / f"pred_{name}__{group}.parquet")
            print(f"{name} {group} {timing[group]:.2f} s/day", flush=True)


def load_pred(name):
    from .chosen import CANDIDATE, CHOSEN, COMPOSE
    parts = COMPOSE.items() if name == CANDIDATE else [(g, name) for g in CHOSEN[name]]
    return pd.concat([pd.read_parquet(OUT / f"pred_{source}__{group}.parquet") for group, source in parts], axis=1)


def tables(ds):
    from .chosen import CANDIDATE, CHOSEN
    replay = report.standard_replay(ds)
    result = {}
    for name in [CANDIDATE] + list(CHOSEN):
        pred = load_pred(name)
        for label, period in (("final", evaluate.FINAL), ("tune", evaluate.TUNE)):
            rows = evaluate.full_table(ds, pred, period, replay)
            result[f"{name}/{label}"] = rows
            print(f"\n### {name}: {label}\n")
            print(evaluate.markdown(rows))
        if name == CANDIDATE:
            rows = evaluate.direction_table(ds, pred, evaluate.FINAL, replay)
            result[f"{name}/final_directions"] = rows
            print(f"\n### {name}: final, falls and rises separately\n")
            print(evaluate.markdown(rows, header=("Event", "Call rule", "Incumbent", "Candidate", "n")))
        cases = report.case_table(ds, pred)
        result[f"{name}/cases"] = cases.to_dict(orient="records")
        print(f"\n### {name}: big-event cases\n")
        print(evaluate.markdown(cases.astype(str).values.tolist(), header=list(cases.columns)))
    (OUT / "results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")


def bench(ds):
    """Fit and predict cost of the candidate on the last cutoff (2026-10-13), one process, one thread."""
    from .chosen import CHOSEN, COMPOSE
    cutoff = config.myt_epoch("2026-10-13")
    out = {}
    for group, source in COMPOSE.items():
        factory, lookback = CHOSEN[source][group]
        spec = GROUPS[group]
        one = ds.m_fresh_available & (ds.index == cutoff + 12 * 3600)          # a single issue (12:00)
        day = ds.m_fresh_available & (ds.index >= cutoff) & (ds.index < cutoff + 86400)
        timings = []
        for mask in (one, day):
            started = time.perf_counter()
            walkforward.walk_forward(ds, factory().fit_predict(group), lookback_days=lookback, days=[cutoff],
                                     test_mask=mask, log=lambda s: None, **spec)
            timings.append(time.perf_counter() - started)
        rows = int(day.sum())
        out[group] = {"source": source, "fit_plus_one_issue_s": round(timings[0], 2), "day_s": round(timings[1], 2),
                      "issues_per_day": rows, "per_issue_predict_ms": round(1000 * (timings[1] - timings[0]) / (rows - 1), 2)}
        print(group, out[group], flush=True)
    (OUT / "bench.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    command = sys.argv[1]
    data = load()
    if command == "tune_kernel":
        tune_kernel(data, *sys.argv[2:3])
    elif command == "tune_kernel_rest":
        tune_kernel_rest(data)
    elif command == "tune_hgb":
        tune_hgb(data, *sys.argv[2:8])
    elif command == "final":
        final(data, *(arg.split(",") for arg in sys.argv[2:4]))
    elif command == "bench":
        bench(data)
    elif command == "leak_check":
        F.leak_check(data)
    elif command == "tables":
        tables(data)
