"""Second robustness pass on GBM-quantile `final` (post hoc, saved predictions, no refits).

    python -B gbm-quantile/robustness_extra.py

1. Independent re-score of the headline test numbers (own code, not robustness.losses).
2. Day-level sign tests: on how many test days does the candidate beat the incumbent?
3. Is the predictive distribution better than a zero-feature one? Mean pinball over the
   candidate's own quantile levels (a CRPS proxy) against the empirical log-ratio
   quantiles of the same training rows (same embargo, no features).
4. P(ride mean <= T) for T = 40..100 (unpooled and pooled), candidate vs zero-feature.
5. P(ride mean <= 70) on the decision-relevant rows (reference 50..100).
6. Issued clocks (Oct 8+): per-day differences.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness import config as hc, report, scoring  # noqa: E402
import evaluate as E  # noqa: E402
import gbmq  # noqa: E402

OFF = gbmq.LOG_OFFSET
RNG = np.random.default_rng(7)
B = 4000
ARR_Q = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)
RIDE_Q = (0.1, 0.3, 0.5, 0.7, 0.9)
THRESHOLDS = (40, 50, 60, 70, 80, 90, 100)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])


def boot(diff_rows, days):
    """Mean row difference and 95% CI resampling whole MYT days; P(diff < 0)."""
    ok = np.isfinite(diff_rows)
    d, g = diff_rows[ok], days[ok]
    u, pos = np.unique(g, return_inverse=True)
    s, n = np.bincount(pos, d, len(u)), np.bincount(pos, None, len(u))
    idx = RNG.integers(0, len(u), size=(B, len(u)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return d.mean(), np.percentile(bs, [2.5, 97.5]), float((bs < 0).mean()), len(u)


def day_wins(a, b, days):
    """Days on which mean loss a < mean loss b (candidate better), out of days with rows."""
    ok = np.isfinite(a) & np.isfinite(b)
    frame = pd.DataFrame({"a": a[ok], "b": b[ok], "d": days[ok]}).groupby("d").mean()
    return int((frame.a < frame.b).sum()), len(frame)


def pin(y, q, tau):
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def zero_feature_z(ds, days):
    """Per evaluation day: training log ratios for arrival and ride heads, with the candidate's masks."""
    index = ds.index.to_numpy(np.int64)
    ref = ds.s_freshPm25.to_numpy(float)
    fresh = ds.m_fresh_available.to_numpy(bool) & np.isfinite(ref)
    arr_c = ds.y_arrival_complete.to_numpy(np.int64)
    ride_c = ds.y_ride_complete.to_numpy(np.int64)
    joint = ds.y_ride_joint_valid.astype(bool).to_numpy()
    ya, ym = ds.y_arrival.to_numpy(float), ds.y_ride_mean.to_numpy(float)
    out = {}
    for d in days:
        win = (index > d - 28 * 86400) & fresh
        arr = win & (index < d - gbmq.ARRIVAL_EMBARGO_SECONDS) & (arr_c < d) & np.isfinite(ya)
        ride = win & (index < d - gbmq.RIDE_EMBARGO_SECONDS) & (ride_c < d) & joint & np.isfinite(ym)
        out[d] = {"arr": np.log((ya[arr] + OFF) / (ref[arr] + OFF)),
                  "ride": np.log((ym[ride] + OFF) / (ref[ride] + OFF)), "ride_y": ym[ride]}
    return out


def main():
    ds = E.load_ds()
    ev = E.evaluation_frame("all")
    inc = report.standard_replay(ds).reindex(ev.index)
    cand = pd.read_parquet(E.pred_path("final", "all")).reindex(ev.index)
    idx = ev.index.to_numpy(np.int64)
    days = np.array([hc.midnight(i) for i in idx])
    test = idx >= hc.myt_epoch("2026-09-27")
    ref = ev.s_freshPm25.to_numpy(float)
    zf = zero_feature_z(ds, sorted(set(days)))

    # ---------------------------------------------------------------- zero-feature quantiles and CDFs
    zq_arr = np.full((len(ev), len(ARR_Q)), np.nan)
    zq_ride = np.full((len(ev), len(RIDE_Q)), np.nan)
    z_le = {t: np.full(len(ev), np.nan) for t in THRESHOLDS}
    clim_le = {t: np.full(len(ev), np.nan) for t in THRESHOLDS}
    for d, z in zf.items():
        rows = np.flatnonzero(days == d)
        r = ref[rows]
        zq_arr[rows] = (r[:, None] + OFF) * np.exp(np.quantile(z["arr"], ARR_Q))[None, :] - OFF
        zq_ride[rows] = (r[:, None] + OFF) * np.exp(np.quantile(z["ride"], RIDE_Q))[None, :] - OFF
        ym = (r[:, None] + OFF) * np.exp(z["ride"])[None, :] - OFF
        for t in THRESHOLDS:
            z_le[t][rows] = (ym <= t).mean(axis=1)
            clim_le[t][rows] = np.mean(z["ride_y"] <= t)

    # candidate CDF of the ride mean at T, exactly as gbmq (z-space knots)
    cq_ride = cand[[f"q_ride_mean_{a:g}" for a in RIDE_Q]].to_numpy(float)
    cq_arr = cand[[f"q_arrival_{a:g}" for a in ARR_Q]].to_numpy(float)
    okc = np.isfinite(cq_ride).all(axis=1) & np.isfinite(ref)
    c_le = {}
    for t in THRESHOLDS:
        v = np.full(len(ev), np.nan)
        zk = np.log((cq_ride[okc] + OFF) / (ref[okc, None] + OFF))
        v[okc] = gbmq.cdf_from_quantiles(zk, RIDE_Q, gbmq.threshold_in_target(np.full(okc.sum(), float(t)), ref[okc], "log"))
        c_le[t] = v

    # ---------------------------------------------------------------- 1. independent re-score (test half)
    yle = ev.y_ride_le70.to_numpy(float)
    m70 = test & inc.p_ride_le70.notna().to_numpy() & cand.p_ride_le70.notna().to_numpy() & np.isfinite(yle)
    check = cand.p_ride_le70.to_numpy(float)
    recon = 0.95 * c_le[70] + 0.05 * clim_le[70]
    print("# GBM-quantile `final`: second robustness pass\n")
    print("## 1. Independent re-score (test half 09-27..10-13)\n")
    print(f"- P(ride mean <=70) reconstructed from the saved ride-mean quantiles and the training share: "
          f"max |diff| vs saved = {np.nanmax(np.abs(recon[m70] - check[m70])):.2e} (n={m70.sum()})")
    rows = []
    zle70 = 0.95 * z_le[70] + 0.05 * clim_le[70]
    for name, p in (("candidate", check), ("cycling logistic (incumbent)", inc.p_ride_le70.to_numpy(float)),
                    ("zero-feature log-ratio (pooled 0.05)", zle70), ("training share only", clim_le[70]),
                    ("0/1 persistence call", (ev.y_ref.to_numpy(float) <= 70).astype(float))):
        rows.append((name, f"{np.mean((p[m70] - yle[m70]) ** 2):.4f}"))
    print("\n" + table(["P(ride mean <=70) on test", "Brier"], rows))

    # ---------------------------------------------------------------- 2. day-level sign tests
    print("\n## 2. Day-level sign tests (test half): days on which the candidate's mean loss is lower\n")
    ya, yr = ev.y_arrival.to_numpy(float), ev.y_ride_mean.to_numpy(float)
    m_a_all = inc.pred_arrival.notna().to_numpy() & cand.pred_arrival.notna().to_numpy() & np.isfinite(ya)
    m_r_all = inc.pred_ride_mean.notna().to_numpy() & cand.pred_ride_mean.notna().to_numpy() & np.isfinite(yr)
    m_a, m_r = test & m_a_all, test & m_r_all
    m_p = test & inc.p_fall20.notna().to_numpy() & cand.p_fall20.notna().to_numpy() & ev.y_delta.notna().to_numpy()
    yf = ev.y_fall20.to_numpy(float)
    y5 = ev.y_class5.to_numpy(float)
    onehot = (y5[:, None] == np.arange(5)[None, :]).astype(float)
    P5c = cand[[f"p_class5_{k}" for k in range(5)]].to_numpy(float)
    P5i = inc[[f"p_class5_{k}" for k in range(5)]].to_numpy(float)
    specs = [
        ("+90 MAE", m_a, np.abs(cand.pred_arrival.to_numpy(float) - ya), np.abs(inc.pred_arrival.to_numpy(float) - ya)),
        ("ride mean MAE", m_r, np.abs(cand.pred_ride_mean.to_numpy(float) - yr), np.abs(inc.pred_ride_mean.to_numpy(float) - yr)),
        ("P(fall>=20) Brier", m_p, (cand.p_fall20.to_numpy(float) - yf) ** 2, (inc.p_fall20.to_numpy(float) - yf) ** 2),
        ("5-class Brier", m_p, ((P5c - onehot) ** 2).sum(1), ((P5i - onehot) ** 2).sum(1)),
        ("P(<=70) Brier", m70, (check - yle) ** 2, (inc.p_ride_le70.to_numpy(float) - yle) ** 2),
        ("P(<=70) Brier vs zero-feature", m70, (check - yle) ** 2, (zle70 - yle) ** 2),
    ]
    rows = []
    for title, m, a, b in specs:
        w, n = day_wins(np.where(m, a, np.nan), np.where(m, b, np.nan), days)
        from math import comb
        p_two = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(w, n - w) + 1)) / 2 ** n)
        rows.append((title, f"{w}/{n}", f"{p_two:.3f}"))
    print(table(["Metric (candidate vs incumbent unless noted)", "days candidate better", "two-sided sign-test p"], rows))

    # ---------------------------------------------------------------- 3. distribution skill vs zero-feature
    print("\n## 3. Predictive distribution vs zero-feature log-ratio quantiles (mean pinball over levels)\n")
    print("Both use the same training rows and embargo; the zero-feature model has no inputs except the reference.\n")
    rows = []
    for title, qc, qz, y, levels, m in (("+90 arrival", cq_arr, zq_arr, ya, ARR_Q, m_a_all),
                                        ("ride mean", cq_ride, zq_ride, yr, RIDE_Q, m_r_all)):
        for split_name, sel in (("test", test), ("tune (selection in-sample)", ~test)):
            mm = m & sel & np.isfinite(qc).all(1) & np.isfinite(qz).all(1) & np.isfinite(y)
            lc = np.mean([pin(y, qc[:, j], a) for j, a in enumerate(levels)], axis=0)
            lz = np.mean([pin(y, qz[:, j], a) for j, a in enumerate(levels)], axis=0)
            d, ci, pr, nd = boot(np.where(mm, lc - lz, np.nan), days)
            rows.append((title, split_name, f"{np.nanmean(lc[mm]):.3f}", f"{np.nanmean(lz[mm]):.3f}", f"{d:+.3f}",
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", f"{pr:.2f}", nd))
            for j, a in enumerate(levels):
                if a in (0.05, 0.1, 0.5, 0.9, 0.95):
                    pc, pz = pin(y, qc[:, j], a), pin(y, qz[:, j], a)
                    d, ci, pr, nd = boot(np.where(mm, pc - pz, np.nan), days)
                    rows.append((f"  {title} q{a:g}", split_name, f"{np.nanmean(pc[mm]):.3f}", f"{np.nanmean(pz[mm]):.3f}",
                                 f"{d:+.3f}", f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", f"{pr:.2f}", nd))
    print(table(["Head", "split", "candidate", "zero-feature", "diff", "95% CI (day bootstrap)", "P(cand better)", "days"], rows))

    # ---------------------------------------------------------------- 4. threshold sweep
    print("\n## 4. P(ride mean <= T): candidate vs zero-feature (test half, Brier)\n")
    rows = []
    for t in THRESHOLDS:
        yt = (yr <= t).astype(float)
        m = m_r & np.isfinite(c_le[t]) & np.isfinite(z_le[t])
        raw_c, raw_z = (c_le[t] - yt) ** 2, (z_le[t] - yt) ** 2
        pc = 0.95 * c_le[t] + 0.05 * clim_le[t]
        pz = 0.95 * z_le[t] + 0.05 * clim_le[t]
        d, ci, pr, _ = boot(np.where(m, (pc - yt) ** 2 - (pz - yt) ** 2, np.nan), days)
        rows.append((t, f"{yt[m].mean():.3f}", f"{raw_c[m].mean():.4f}", f"{raw_z[m].mean():.4f}",
                     f"{np.mean((pc[m] - yt[m]) ** 2):.4f}", f"{np.mean((pz[m] - yt[m]) ** 2):.4f}",
                     f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]", f"{pr:.2f}"))
    print(table(["T", "base rate", "cand raw", "zero raw", "cand pooled", "zero pooled", "pooled diff [95% CI]",
                 "P(cand better)"], rows))

    # ---------------------------------------------------------------- 5. decision-relevant rows
    print("\n## 5. P(ride mean <=70) by reference band (test half)\n")
    rows = []
    for lo, hi in ((0, 50), (50, 70), (70, 100), (50, 100), (100, 1000)):
        m = m70 & (ref >= lo) & (ref < hi)
        if m.sum() < 30:
            continue
        bc, bi, bz = ((check - yle) ** 2), ((inc.p_ride_le70.to_numpy(float) - yle) ** 2), ((zle70 - yle) ** 2)
        d, ci, pr, nd = boot(np.where(m, bc - bi, np.nan), days)
        d2, ci2, pr2, _ = boot(np.where(m, bc - bz, np.nan), days)
        rows.append((f"{lo}-{hi}", int(m.sum()), nd, f"{yle[m].mean():.2f}", f"{bc[m].mean():.4f}", f"{bi[m].mean():.4f}",
                     f"{bz[m].mean():.4f}", f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]",
                     f"{d2:+.4f} [{ci2[0]:+.4f}, {ci2[1]:+.4f}]"))
    print(table(["ref band", "n", "days", "share <=70", "cand", "incumbent", "zero-feature",
                 "cand - incumbent [95% CI]", "cand - zero [95% CI]"], rows))

    # ---------------------------------------------------------------- 6. events: are there any argmax calls at all?
    print("\n## 6. Event-call census (test half, rows where the logistic has an output)\n")
    rows = []
    for name, frame in (("candidate", cand), ("incumbent", inc)):
        pf, pr_ = frame.p_fall20.to_numpy(float)[m_p], frame.p_rise20.to_numpy(float)[m_p]
        am = scoring.calls_from_probs(pf, pr_, "argmax")
        rows.append((name, int((am != 0).sum()), int((pf >= 0.2).sum() + (pr_ >= 0.2).sum()),
                     f"{np.nanmax(pf):.3f}", f"{np.nanmax(pr_):.3f}", f"{np.mean(pf >= 0.5):.4f}"))
    print(table(["model", "argmax calls (issues)", "p>=0.2 issues", "max P(fall20)", "max P(rise20)", "share P(fall20)>=0.5"], rows))
    raw = cand[[f"p_class5_raw_{k}" for k in range(5)]].to_numpy(float)[m_p]
    rf, rr = raw[:, 0] + raw[:, 1], raw[:, 3] + raw[:, 4]
    print(f"\nUnpooled candidate: max P(fall20) {np.nanmax(rf):.3f}; pooled argmax needs P_raw(fall20) above about "
          f"{(0.5 - 0.3 * 0.05) / 0.7:.2f} when the class-2 climatology is 0.9.")

    # ---------------------------------------------------------------- 7. issued clocks, per day
    print("\n## 7. Issued clocks (Oct 8+): candidate - issued incumbent per day\n")
    from harness import issued as hissued
    from harness.incumbents import run as runs
    frame = runs.issued_clock_frame()
    pred = pd.read_parquet(E.OUT / "final__issued.parquet").reindex(frame.index)
    fdays = np.array([hc.midnight(i) for i in frame.index])
    ridge_ok = ((frame.issued__arrival_model == hissued.INCUMBENT_VERSIONS["ridge"])
                & (frame.index >= report.ISSUED_START["ridge"])).to_numpy()
    cy_ok = ((frame.issued__cy_model == hissued.INCUMBENT_VERSIONS["cycling70"]) & frame.issued__cy_p_le70.notna()).to_numpy()
    out = []
    for title, ok, a, b in (
            ("+90 MAE", ridge_ok & frame.issued__arrival_point.notna().to_numpy() & pred.pred_arrival.notna().to_numpy() & frame.y_arrival.notna().to_numpy(),
             np.abs(pred.pred_arrival - frame.y_arrival).to_numpy(float), np.abs(frame.issued__arrival_point - frame.y_arrival).to_numpy(float)),
            ("P(<=70) Brier", cy_ok & pred.p_ride_le70.notna().to_numpy() & frame.y_ride_le70.notna().to_numpy(),
             ((pred.p_ride_le70 - frame.y_ride_le70) ** 2).to_numpy(float), ((frame.issued__cy_p_le70 - frame.y_ride_le70) ** 2).to_numpy(float))):
        g = pd.DataFrame({"d": fdays[ok], "diff": (a - b)[ok]}).groupby("d")["diff"].agg(["mean", "size"])
        out.append((title, "; ".join(f"{hc.myt(d):%m-%d} {r['mean']:+.4f} (n={int(r['size'])})" for d, r in g.iterrows())))
    print(table(["Metric", "per-day mean difference (negative = candidate better)"], out))


if __name__ == "__main__":
    main()
