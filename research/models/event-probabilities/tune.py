"""Choose the candidate on the tune period only (2026-09-10..09-26).

Usage: python -B tune.py
Reads the saved base outputs, builds the calibrated / stacked variants, scores
every variant on the tune period and writes
data\\models\\event-probabilities\\tune_results.json plus tune_tables.md.
The holdout period is not touched here.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import json
from pathlib import Path

import pandas as pd

import calibrate as cal
import common
import evaluate as E

TUNE_DAYS_END = common.HOLDOUT_START


def _tb(ds, inputs, days, lookback=21):
    """tempbias-calibrated class5 (+ first20 when every input has it)."""
    five = cal.calibrate(ds, inputs, "class5", "tempbias", lookback_days=lookback, days=days)
    if all(all(c in f for c in common.FIRST3) for f in inputs.values()):
        five = five.join(cal.calibrate(ds, inputs, "first20", "tempbias", lookback_days=lookback, days=days)[common.FIRST3])
    return five


def _stack(ds, base, dist, clf, days, lookback=21, first_method="matrix"):
    """class5: tempbias over the distribution and the classifier. first20:
    'matrix' = multinomial calibration of the classifier's first-crossing
    probabilities with the distribution's fall20/rise20 as extra inputs;
    'tempbias' = the classifier's first-crossing probabilities alone."""
    five = cal.calibrate(ds, {dist: base[dist], clf: base[clf]}, "class5", "tempbias", lookback_days=lookback, days=days)
    if first_method == "matrix":
        first = cal.calibrate(ds, {clf: base[clf]}, "first20", "matrix", lookback_days=lookback, days=days, C=0.1,
                              extra={dist: (base[dist], ["p_fall20", "p_rise20"])})
    else:
        first = cal.calibrate(ds, {clf: base[clf]}, "first20", "tempbias", lookback_days=lookback, days=days)
    return five.join(first[common.FIRST3])


def variants(ds, base: dict) -> dict:
    """Candidate variants; calibrators produce tune-period days only."""
    days = common.days_between(ds, common.TUNE_START, TUNE_DAYS_END)
    out = {f"raw:{n}": f for n, f in base.items()}
    for n, f in base.items():
        if n != "lr_emp_28":
            out[f"tb21:{n}"] = _tb(ds, {n: f}, days)
    if "hgb_all_a0" in base:
        out["tb14:hgb_all_a0"] = _tb(ds, {"hgb_all_a0": base["hgb_all_a0"]}, days, lookback=14)
        out["tb28:hgb_all_a0"] = _tb(ds, {"hgb_all_a0": base["hgb_all_a0"]}, days, lookback=28)
        out["mx21:hgb_all_a0"] = cal.calibrate(ds, {"a": base["hgb_all_a0"]}, "class5", "matrix", lookback_days=21,
                                               days=days, C=0.1)
        out["iso21:hgb_all_a0"] = cal.isotonic_tails(ds, base["hgb_all_a0"], days=days)
    for clf in [n for n in base if n.startswith("hgb_all")]:
        if "lr_hgb_all" in base:
            out[f"stack:lr_hgb_all+{clf}"] = _stack(ds, base, "lr_hgb_all", clf, days)
    if "lr_hgb_all" in base and "hgb_all_a0" in base:
        out["stack14:lr_hgb_all+hgb_all_a0"] = _stack(ds, base, "lr_hgb_all", "hgb_all_a0", days, lookback=14)
        out["stack14tb:lr_hgb_all+hgb_all_a0"] = _stack(ds, base, "lr_hgb_all", "hgb_all_a0", days, lookback=14,
                                                        first_method="tempbias")
    if "lr_hgb_sw" in base and "hgb_sw_a0" in base:
        out["stack:lr_hgb_sw+hgb_sw_a0"] = _stack(ds, base, "lr_hgb_sw", "hgb_sw_a0", days)
    if "lr_hgb_all" in base and "hgb_sensor_a0" in base:
        out["stack:lr_hgb_all+hgb_sensor_a0"] = _stack(ds, base, "lr_hgb_all", "hgb_sensor_a0", days)
    return out


def main(names=None):
    ds = common.load()
    names = names or sorted(p.stem[5:] for p in common.OUT.glob("base_*.parquet"))
    base = {n: common.load_pred(f"base_{n}") for n in names}
    cands = variants(ds, base)
    results = E.run(ds, cands, periods=("tune",))
    models = ["prior", "persistence", "incumbent"] + list(cands)
    text = ["# Tune period (2026-09-10..09-26) only", "", E.prob_table(results, "tune", models), "",
            E.event_table(results, "tune", models), "", E.matched_table(results, "tune", ["incumbent"] + list(cands))]
    Path(common.OUT / "tune_tables.md").write_text("\n".join(text), encoding="utf-8")
    Path(common.OUT / "tune_results.json").write_text(json.dumps(E.to_json(results), indent=1), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
