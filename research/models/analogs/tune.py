"""Coordinate-descent tuning of the analog configs on the TUNING split only (issue days 09-10 .. 09-27).

    python -B -m analogs.tune prob     # probability config: max mean Brier skill vs prior of fall20, rise20, first20
    python -B -m analogs.tune point    # point config: min arrival MAE + ride-mean MAE
    python -B -m analogs.tune event    # event variant: max (>=20 episodes caught - false - wrong-way clusters), argmax

Every evaluated config is appended to data/models/analogs/tune_log_<mode>.csv. The
holdout days (09-28 .. 10-13) are never run here.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from . import analog, common

GRID = {
    "k": [10, 20, 35, 50, 75, 100, 150, 250],
    "bandwidth": [0.25, 0.5, 1.0, 2.0, np.inf],
    "transfer": ["ratio", "additive"],
    "alpha": [0.0, 2.0, 5.0, 10.0, 20.0],
    "traj_tau": [30.0, 60.0, 120.0, np.inf],
    "point_stat": ["median", "mean"],
    "lookback": [28, 60],
}
WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SKIP = {"prob": {"point_stat"}, "event": {"point_stat"}, "point": {"alpha"}}


def objective(s: dict, mode: str) -> float:
    """Selection rules, fixed before any search (higher is better)."""
    if mode == "prob":
        return float(np.mean([s["bs_fall20"], s["bs_rise20"], s["f_skill"]]))
    if mode == "event":   # caught >=20 episodes minus false and wrong-way call clusters (argmax); Brier breaks ties
        return float(s["ep20_argmax"] - s["fc20_argmax"] - s["wc20_argmax"]
                     + 0.01 * np.mean([s["bs_fall20"], s["bs_rise20"]]))
    return -float(s["arr_mae"] + s["ride_mean"])


def evaluate(cfg, lookback, mode, slim, log_rows):
    started = time.time()
    pred = common.run(cfg, "tune", lookback_days=lookback, slim=slim)
    s = common.summary(common.score_all(pred, "tune", others=False))
    row = {"mode": mode, "config": cfg.label(), "lookback": lookback, "objective": objective(s, mode),
           "seconds": round(time.time() - started, 1), **{k: v for k, v in s.items()}}
    log_rows.append(row)
    return row["objective"], s


def descend(mode: str, sweeps=2):
    slim = common.load_slim()
    cfg, lookback = analog.AnalogConfig(), 60
    log_rows = []
    best, s = evaluate(cfg, lookback, mode, slim, log_rows)
    print(f"start {best:.4f} {cfg.label()}")
    for sweep in range(sweeps):
        improved = False
        dims = [d for d in GRID if d not in SKIP[mode]] + [f"w:{b}" for b in analog.BLOCKS]
        for dim in dims:
            values = WEIGHT_GRID if dim.startswith("w:") else GRID[dim]
            for value in values:
                if dim == "lookback":
                    trial_cfg, trial_lb = cfg, value
                    if value == lookback:
                        continue
                elif dim.startswith("w:"):
                    block = dim[2:]
                    if cfg.weights.get(block, 0.0) == value:
                        continue
                    trial_cfg, trial_lb = cfg.with_(weights={block: value}), lookback
                    if not any(v > 0 for v in trial_cfg.weights.values()):
                        continue
                else:
                    if getattr(cfg, dim) == value:
                        continue
                    trial_cfg, trial_lb = cfg.with_(**{dim: value}), lookback
                score, trial_s = evaluate(trial_cfg, trial_lb, mode, slim, log_rows)
                if score > best + 1e-6:
                    best, cfg, lookback, s, improved = score, trial_cfg, trial_lb, trial_s, True
                    print(f"  sweep {sweep} {dim}={value}: {best:.4f} {cfg.label()} lb={lookback}", flush=True)
        if not improved:
            break
    log = pd.DataFrame(log_rows)
    path = common.OUT / f"tune_log_{mode}.csv"
    log.to_csv(path, mode="a", header=not path.exists(), index=False)
    chosen = {"mode": mode, "objective": best, "lookback": lookback, "k": cfg.k, "bandwidth": cfg.bandwidth,
              "transfer": cfg.transfer, "alpha": cfg.alpha, "traj_tau": cfg.traj_tau, "point_stat": cfg.point_stat,
              "weights": cfg.weights, "label": cfg.label(), "evaluations": len(log_rows), "tune_summary": s}
    (common.OUT / f"chosen_{mode}.json").write_text(json.dumps(chosen, indent=1, default=str), encoding="utf-8")
    print(json.dumps(chosen, indent=1, default=str))
    return chosen


def extend_k(mode: str, ks=(400, 600, 1000)):
    """K ended at the top of GRID: try larger K from the chosen config (tuning split only)."""
    path = common.OUT / f"chosen_{mode}.json"
    chosen = json.loads(path.read_text(encoding="utf-8"))
    cfg = analog.AnalogConfig(weights=chosen["weights"], k=int(chosen["k"]), bandwidth=float(chosen["bandwidth"]),
                              transfer=chosen["transfer"], alpha=float(chosen["alpha"]),
                              traj_tau=float(chosen["traj_tau"]), point_stat=chosen["point_stat"])
    slim, log_rows = common.load_slim(), []
    best, best_cfg, s = chosen["objective"], cfg, chosen["tune_summary"]
    for k in ks:
        score, trial_s = evaluate(cfg.with_(k=k), chosen["lookback"], mode, slim, log_rows)
        print(f"  k={k}: {score:.4f} (best {best:.4f})", flush=True)
        if score > best + 1e-6:
            best, best_cfg, s = score, cfg.with_(k=k), trial_s
    log_path = common.OUT / f"tune_log_{mode}_extend.csv"
    pd.DataFrame(log_rows).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    chosen.update({"objective": best, "k": best_cfg.k, "label": best_cfg.label(), "tune_summary": s,
                   "evaluations": chosen["evaluations"] + len(log_rows), "k_extension": list(ks)})
    path.write_text(json.dumps(chosen, indent=1, default=str), encoding="utf-8")
    print(json.dumps(chosen, indent=1, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "extend":
        extend_k(sys.argv[2])
    else:
        descend(sys.argv[1] if len(sys.argv) > 1 else "prob")
