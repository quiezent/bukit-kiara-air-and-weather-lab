"""Leak check of the derived features (x_*): recompute them from a dataset cut at the issue.

For each sampled issue t the dataset is truncated to rows with issue <= t, and
every label of an earlier row whose completion clock is not before t is blanked
(it was not known at t). ``common.add_derived`` is rerun on that truncated frame
and its last row must equal the full-data value for every x_* column.
Harness feature groups are covered by harness.checks / harness.audit.

Usage: python -B leakcheck.py [n_issues]
"""
from __future__ import annotations

import sys

import numpy as np

import common
from harness import dataset


def leaky_derived(frame):
    """Positive control: two deliberately leaky features (future reference, own label)."""
    out = common.add_derived(frame)
    out["x_leak_future_ref"] = out["s_freshPm25"].shift(-6)
    out["x_leak_own_label"] = out["y_delta"]
    return out


def main(n=40, seed=7, derive=common.add_derived):
    raw = dataset.load()
    full = derive(raw)
    derived = [c for c in full.columns if c.startswith("x_")]
    rng = np.random.default_rng(seed)
    index = raw.index.to_numpy(np.int64)
    eval_index = index[raw["eval"].to_numpy(bool)]
    sample = rng.choice(eval_index, n, replace=False)
    # Add the big-event onsets used in the case table.
    sample = np.concatenate([sample, [common.config.myt_epoch(t) for t in
                                      ("2026-10-09 14:00", "2026-10-13 16:45", "2026-10-06 04:00")]])
    mismatches = 0
    for t in sample:
        cut = raw.loc[raw.index <= t].copy()
        unknown = cut["y_arrival_complete"].to_numpy(np.int64) >= t
        for c in [c for c in cut.columns if c.startswith("y_")]:
            if cut[c].dtype.kind == "f":
                cut.loc[unknown, c] = np.nan
        part = derive(cut)
        a = part.loc[t, derived].to_numpy(float)
        b = full.loc[t, derived].to_numpy(float)
        same = (np.isclose(a, b, rtol=1e-9, atol=1e-9) | (np.isnan(a) & np.isnan(b)))
        if not same.all():
            mismatches += 1
            bad = [derived[i] for i in np.flatnonzero(~same)]
            if derive is common.add_derived:
                print(f"MISMATCH {common.config.myt(t)}: {bad}")
    print(f"{len(sample)} issues, {len(derived)} derived columns, {mismatches} issues with a mismatch")
    return mismatches


if __name__ == "__main__":
    n_issues = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    bad = main(n_issues)
    print("positive control (two leaky features added):")
    caught = main(10, derive=leaky_derived)
    sys.exit(1 if bad or not caught else 0)
