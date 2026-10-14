"""The frozen spatial candidate and its held-out evidence.

    python -B -m spatial.candidate [--lag 300]

spatial_v1 = lr28_sp (5-class arrival change: production logistic recipe + 12 robust spatial
inputs) + hgbr_sp (HGB absolute-error regressors for +90 arrival and ride mean/min/max, HGB
P(ride mean <= 70)). Selected on the tuning split only (2026-09-10 .. 10-05): lr28_sp beat
hgbc_sp on 5-class log loss (0.295 vs 0.432) and on >=20 episodes; hgbr_sp was the only point
variant. Writes data/models/spatial/pred_spatial_v1_lag<lag>.parquet and prints the
big-event case table and day-level paired comparisons on the final split.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring
from spatial.score_spatial import OUT, SPLIT_EPOCH, load_pred


def combine(lag):
    cls = load_pred("lr28_sp", lag)
    reg = load_pred("hgbr_sp", lag)
    return pd.concat([cls, reg], axis=1)


def day_table(ds, cand, inc, split_epoch):
    ev = ds.loc[ds["eval"] & ds.m_fresh_available & (ds.index >= split_epoch)]
    rows = []
    for day, part in ev.groupby("date_myt"):
        idx = part.index
        ok = cand.pred_arrival.reindex(idx).notna() & inc.pred_arrival.reindex(idx).notna() & part.y_arrival.notna()
        y, ref = part.y_arrival[ok], part.y_ref[ok]
        c, r = cand.pred_arrival.reindex(idx)[ok], inc.pred_arrival.reindex(idx)[ok]
        lab = part.y_class5.isin(range(5)) & inc.p_class5_0.reindex(idx).notna() & cand.p_class5_0.reindex(idx).notna()
        p5 = [f"p_class5_{k}" for k in range(5)]
        ll_c = scoring.multiclass_scores(cand.loc[idx[lab], p5].to_numpy(float), part.y_class5[lab], range(5))
        ll_i = scoring.multiclass_scores(inc.loc[idx[lab], p5].to_numpy(float), part.y_class5[lab], range(5))
        rows.append({"day": day, "n": int(ok.sum()),
                     "MAE persistence": (ref - y).abs().mean(), "MAE Ridge v4": (r - y).abs().mean(),
                     "MAE spatial_v1": (c - y).abs().mean(),
                     "logloss logistic": ll_i.get("log_loss"), "logloss spatial_v1": ll_c.get("log_loss"),
                     "events>=20 issues": int((part.y_delta.abs() >= 20).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, default=300)
    a = p.parse_args()
    ds = dataset.load()
    cand = combine(a.lag)
    cand.to_parquet(OUT / f"pred_spatial_v1_lag{a.lag}.parquet")
    inc = report.standard_replay(ds)
    print("## day-level, final split")
    print(day_table(ds, cand, inc, SPLIT_EPOCH).round(3).to_markdown(index=False))
    print("\n## big-event cases (spatial_v1; 'Ridge' columns = spatial_v1 points, 'logistic' = spatial_v1 classes)")
    full = cand.reindex(ds.index)
    print(report.case_table(ds, full).drop(columns=["issued at onset (Oct 8+)", "first-crossing P at onset",
                                                    "first-crossing first argmax call"]).to_markdown(index=False))
