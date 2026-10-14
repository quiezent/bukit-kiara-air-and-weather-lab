"""Leak check of the session history features by brute-force recomputation.

    & $py -B -m sessions.check

For random issues and every canonical issue it recomputes, with plain loops
over data at or before the issue:
* rolling means / extremes of the fresh reference over grid issues in (t - w, t];
* the same-clock 7-day mean;
* the last and 7-session mean of completed sessions, admitting a session only
  if its window end + 15 min <= t (independent of features.session_table);
and checks that no weather/CAMS column of the stacked frame differs from the
harness dataset value (those carry the harness visibility clocks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import dataset
from . import features, run


def main(sample=300, seed=7):
    ds = dataset.load()
    stack = run.load_stack(ds)
    index = ds.index.to_numpy(np.int64)
    ref = ds.s_freshPm25.to_numpy(float)
    rng = np.random.default_rng(seed)
    chosen = np.unique(np.concatenate([rng.choice(index[index > index[0] + 3 * 86400], sample, replace=False),
                                       ds.index[ds.canonical != ""].to_numpy(np.int64)]))
    sessions = {}
    for name in features.SESSIONS:
        starts, means = ds[f"y_{name}_start"].to_numpy(np.int64), ds[f"y_{name}_mean"].to_numpy(float)
        unique = {}
        for s, m in zip(starts, means):
            if np.isfinite(m):
                unique[int(s)] = float(m)
        sessions[name] = sorted(unique.items())
    bad = 0
    checked = 0
    for t in chosen:
        for name in features.SESSIONS:
            row = stack.loc[(int(t), name)]
            past = (index > t - 86400) & (index <= t) & np.isfinite(ref)
            expected = {"x_ref_mean_24h": ref[past].mean() if past.sum() >= 6 else np.nan}
            past3 = (index > t - 3 * 3600) & (index <= t) & np.isfinite(ref)
            expected["x_ref_mean_3h"] = ref[past3].mean() if past3.sum() >= 6 else np.nan
            expected["x_ref_max_24h"] = ref[past].max() if past.sum() >= 6 else np.nan
            same = [ref[index == t - k * 86400][0] for k in range(1, 8) if (index == t - k * 86400).any()]
            same = [v for v in same if np.isfinite(v)]
            expected["x_ref_sameclock_7d"] = np.mean(same) if len(same) >= 3 else np.nan
            done = [m for s, m in sessions[name] if s + 7200 + 900 <= t]
            expected["x_sess_last"] = done[-1] if done else np.nan
            expected["x_sess_mean7"] = np.mean(done[-7:]) if done else np.nan
            for column, value in expected.items():
                checked += 1
                got = row[column]
                if not (np.isnan(value) and np.isnan(got)) and not np.isclose(got, value, rtol=0, atol=1e-9):
                    bad += 1
                    print("MISMATCH", t, name, column, got, value)
            for short in ("temp", "pprob", "blh", "precip_sum"):
                column = f"w_{name}_{short}" if short == "precip_sum" else f"w_{name}_{short}_mean"
                checked += 1
                a, b = row[f"x_w_sess_{short}"], ds.at[int(t), column]
                if not (np.isnan(a) and np.isnan(b)) and a != b:
                    bad += 1
                    print("MISMATCH weather", t, name, short, a, b)
            checked += 1
            a, b = row["x_c_sess_pm25"], ds.at[int(t), f"c_pm25_{name}_mean"]
            if not (np.isnan(a) and np.isnan(b)) and a != b:
                bad += 1
    # the visible session must have ended before the issue (no in-progress or future session)
    issue = stack.index.get_level_values(0).to_numpy(np.int64)
    used_end = issue - stack.x_sess_age_h.to_numpy(float) * 3600
    ok = np.isfinite(used_end)
    late = int(np.sum(used_end[ok] + features.SESSION_VISIBLE_LAG > issue[ok] + 1e-6))
    print(f"issues checked: {len(chosen)}; values checked: {checked}; mismatches: {bad}; "
          f"session history used before end+15min: {late}")
    return bad + late


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
