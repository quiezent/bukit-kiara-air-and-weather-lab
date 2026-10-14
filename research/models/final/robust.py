"""Robustness of the final models: leave-one-day-out, weekly splits, and a stale-input stress test.

    python -B -m final.robust       # writes data/models/final/robust.md

Stale-input stress (holdout days): models are refit exactly as in the walk-forward, but every
input (sensor, controls, spatial, wind) is taken from the grid issue 10 min earlier, as if the
live inputs arrived late. Targets stay at the true issue. Persistence is shown both fresh and
10 min stale.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from harness import config, report
from . import models as M
from .run import OUT, SPLITS, load, day_bootstrap, bs_cell


def lodo(days, a, b):
    frame = pd.DataFrame({"day": np.asarray(days), "d": np.asarray(a, float) - np.asarray(b, float)})
    g = frame.groupby("day").d.agg(["sum", "count"])
    out = [(g["sum"].sum() - g.loc[d, "sum"]) / (g["count"].sum() - g.loc[d, "count"]) for d in g.index]
    return min(out), max(out)


def main():
    ds = load()
    sel = json.loads((OUT / "selection.json").read_text())
    point = pd.read_parquet(OUT / f"pred_point_{sel['point']}.parquet").reindex(ds.index)
    le70 = pd.read_parquet(OUT / f"pred_le70_{sel['le70']}.parquet").p_ride_le70.reindex(ds.index)
    inc = report.standard_replay(ds)
    lines = ["# Final models: robustness checks\n"]
    for split in ("holdout", "all"):
        lo, hi = SPLITS[split]
        ev = ds.loc[ds["eval"] & ds.m_fresh_available & (ds.index >= lo) & (ds.index < hi)]
        lines.append(f"## {split}\n")
        rows = []
        for head, target in (("arrival", "y_arrival"), ("ride_mean", "y_ride_mean"), ("ride_min", "y_ride_min"),
                             ("ride_max", "y_ride_max")):
            col = f"pred_{head}"
            ok = ev[target].notna() & point[col].reindex(ev.index).notna() & inc[col].reindex(ev.index).notna()
            sub = ev.loc[ok]
            y = sub[target].to_numpy(float)
            fe = np.abs(point.loc[sub.index, col].to_numpy(float) - y)
            ie = np.abs(inc.loc[sub.index, col].to_numpy(float) - y)
            pe = np.abs(sub.y_ref.to_numpy(float) - y)
            for name, other in (("Ridge v4", ie), ("persistence", pe)):
                lo_, hi_ = lodo(sub.date_myt, fe, other)
                daily = pd.DataFrame({"d": sub.date_myt.to_numpy(), "f": fe, "o": other}).groupby("d").mean()
                rows.append({"head": head, "vs": name, "diff": fe.mean() - other.mean(),
                             "leave-one-day-out range": f"{lo_:+.2f} .. {hi_:+.2f}",
                             "days better": f"{int((daily.f < daily.o).sum())}/{len(daily)}"})
        ok = ev.y_ride_le70.notna() & le70.reindex(ev.index).notna() & inc.p_ride_le70.reindex(ev.index).notna()
        sub = ev.loc[ok]
        y = sub.y_ride_le70.to_numpy(float)
        fs = (le70.loc[sub.index].to_numpy(float) - y) ** 2
        for name, other in (("cycling", (inc.p_ride_le70.loc[sub.index].to_numpy(float) - y) ** 2),
                            ("persistence 0/1", ((sub.y_ref <= 70).astype(float).to_numpy() - y) ** 2)):
            lo_, hi_ = lodo(sub.date_myt, fs, other)
            daily = pd.DataFrame({"d": sub.date_myt.to_numpy(), "f": fs, "o": other}).groupby("d").mean()
            rows.append({"head": "P(<=70) Brier", "vs": name, "diff": fs.mean() - other.mean(),
                         "leave-one-day-out range": f"{lo_:+.4f} .. {hi_:+.4f}",
                         "days better": f"{int((daily.f < daily.o).sum())}/{len(daily)}"})
        lines.append(pd.DataFrame(rows).round(4).to_markdown(index=False))

    # stale-input stress on the holdout
    lo, hi = SPLITS["holdout"]
    cols = M.POINT_VARIANTS[sel["point"]]
    shifted = ds.copy()
    feat = [c for c in ds.columns if c.startswith(("s_", "x_", "sp_", "w_", "c_", "rp_", "m_"))]
    prev = ds[feat].reindex(ds.index - 600)
    prev.index = ds.index
    shifted[feat] = prev
    out = []
    for cutoff in range(lo, hi, 86400):
        train = ds.loc[ds.index < cutoff]                  # training exactly as in the walk-forward
        model = M.fit_point(train, cutoff, cols)
        test = shifted.loc[(shifted.index >= cutoff) & (shifted.index < cutoff + 86400)]
        p = M.predict_point_frame(model, test)
        le = M.fit_le70(train, cutoff, sel["le70"], point=model)
        p["p_ride_le70"] = M.predict_le70_frame(le, test, p)
        out.append(p)
    stale = pd.concat(out)
    ev = ds.loc[ds["eval"] & ds.m_fresh_available & (ds.index >= lo) & (ds.index < hi)]
    rows = []
    for head, target in (("arrival", "y_arrival"), ("ride_mean", "y_ride_mean")):
        col = f"pred_{head}"
        ok = ev[target].notna() & stale[col].reindex(ev.index).notna() & inc[col].reindex(ev.index).notna() \
            & shifted.s_freshPm25.reindex(ev.index).notna()
        sub = ev.loc[ok]
        y = sub[target].to_numpy(float)
        rows.append({"head": head, "n": len(sub), "final fresh": np.abs(point.loc[sub.index, col] - y).mean(),
                     "final, inputs 10 min stale": np.abs(stale.loc[sub.index, col] - y).mean(),
                     "Ridge v4 fresh": np.abs(inc.loc[sub.index, col] - y).mean(),
                     "persistence fresh": np.abs(sub.y_ref - y).mean(),
                     "persistence 10 min stale": np.abs(shifted.loc[sub.index, "s_freshPm25"] - y).mean()})
    ok = ev.y_ride_le70.notna() & stale.p_ride_le70.reindex(ev.index).notna() & inc.p_ride_le70.reindex(ev.index).notna()
    sub = ev.loc[ok]
    y = sub.y_ride_le70.to_numpy(float)
    rows.append({"head": "P(<=70) Brier", "n": len(sub), "final fresh": np.mean((le70.loc[sub.index] - y) ** 2),
                 "final, inputs 10 min stale": np.mean((stale.loc[sub.index, "p_ride_le70"] - y) ** 2),
                 "Ridge v4 fresh": np.mean((inc.loc[sub.index, "p_ride_le70"] - y) ** 2),
                 "persistence fresh": np.mean(((sub.y_ref <= 70).astype(float) - y) ** 2),
                 "persistence 10 min stale": np.mean(((shifted.loc[sub.index, "s_freshPm25"] <= 70).astype(float) - y) ** 2)})
    lines.append("\n## Stale-input stress, holdout (P(<=70) row: 'Ridge v4 fresh' column = cycling logistic)\n")
    lines.append(pd.DataFrame(rows).round(4).to_markdown(index=False))
    text = "\n".join(lines)
    (OUT / "robust.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
