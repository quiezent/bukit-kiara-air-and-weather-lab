"""E1 vs E0 vs analog for P(ride mean <= 70): paired day-block bootstrap per split, weekly Brier, and
the outcome-conditioned 'window opened' slice (descriptive only). Run after final.run le70.

    python -B -m final.le70_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import config, dataset, report
from .run import OUT, SPLITS, day_bootstrap, bs_cell, candidates

ds = dataset.load()
inc = report.standard_replay(ds)
cands = candidates(ds)
P = {v: pd.read_parquet(OUT / f"pred_le70_{v}.parquet").p_ride_le70.reindex(ds.index)
     for v in ("E0_level_ratio", "E1_spatial_residual", "L1_level_logistic")}
P["analog"] = cands["analog"].p_ride_le70
P["cycling"] = inc.p_ride_le70
lines = ["# P(ride mean <= 70): E1 vs E0 checks\n"]
for split, (lo, hi) in SPLITS.items():
    ev = ds.loc[ds["eval"] & ds.m_fresh_available & (ds.index >= lo) & (ds.index < hi) & ds.y_ride_le70.notna()]
    for need_inc in (False, True):
        ok = P["E0_level_ratio"].reindex(ev.index).notna() & P["E1_spatial_residual"].reindex(ev.index).notna() \
            & P["analog"].reindex(ev.index).notna()
        if need_inc:
            ok &= P["cycling"].reindex(ev.index).notna()
        sub = ev.loc[ok]
        y = sub.y_ride_le70.to_numpy(float)
        sq = {k: (p.reindex(sub.index).to_numpy(float) - y) ** 2 for k, p in P.items()}
        tag = f"{split} ({'rows with cycling' if need_inc else 'all rows'}; n={len(sub)}, days={sub.date_myt.nunique()})"
        cells = [f"{k} {sq[k].mean():.4f}" for k in ("E0_level_ratio", "E1_spatial_residual", "L1_level_logistic", "analog")
                 if np.isfinite(sq[k]).all()]
        if need_inc:
            cells.append(f"cycling {sq['cycling'].mean():.4f}")
        lines.append(f"- **{tag}**: " + ", ".join(cells))
        lines.append(f"  - E1 - E0: {bs_cell(day_bootstrap(sub.date_myt, sq['E1_spatial_residual'], sq['E0_level_ratio']), 4)}")
        lines.append(f"  - E1 - analog: {bs_cell(day_bootstrap(sub.date_myt, sq['E1_spatial_residual'], sq['analog']), 4)}")
        if need_inc:
            lines.append(f"  - E1 - cycling: {bs_cell(day_bootstrap(sub.date_myt, sq['E1_spatial_residual'], sq['cycling']), 4)}")
        if split == "all" and not need_inc:
            ref = sub.y_ref.to_numpy(float)
            opened = (ref > 70) & (y == 1)
            closed = (ref <= 70) & (y == 0)
            near = np.abs(ref - 70) <= 10
            rows = []
            for k in P:
                p = P[k].reindex(sub.index).to_numpy(float)
                if not np.isfinite(p).all():
                    continue
                rows.append({"model": k, "mean P, window opened (ref>70, ride<=70)": p[opened].mean(),
                             "mean P, window closed (ref<=70, ride>70)": p[closed].mean(),
                             "Brier |ref-70|<=10": sq[k][near].mean()})
            lines.append(f"\nOutcome-conditioned slices, all rows ({opened.sum()} opened, {closed.sum()} closed, {near.sum()} near):\n")
            lines.append(pd.DataFrame(rows).round(3).to_markdown(index=False))
            week = (sub.index.to_numpy(np.int64) - config.myt_epoch("2026-09-10")) // (7 * 86400)
            wk = pd.DataFrame({k: sq[k] for k in ("E0_level_ratio", "E1_spatial_residual", "analog")}, index=sub.index)
            wk["week from"] = [str(config.myt(config.myt_epoch("2026-09-10") + 7 * 86400 * w))[:10] for w in week]
            lines.append("\nWeekly Brier, all rows:\n")
            lines.append(wk.groupby("week from").mean().round(4).to_markdown())
text = "\n".join(lines)
(OUT / "le70_check.md").write_text(text, encoding="utf-8")
print(text)
