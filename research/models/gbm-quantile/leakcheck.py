"""Leakage stress test of the GBM-quantile `final` candidate (skeptic review).

  build            delayed weather/CAMS features: each vintage becomes visible only when
                   the NEXT vintage did, and never earlier than its own clock + 600 s
  run <variant>    17-day test walk-forward of `final` on a delayed dataset, with
                   cutoff/embargo assertions on every refit
     wc          weather + CAMS one fetch cycle older (sensor unchanged)
     all_anchor  wc + every s_ sensor feature taken from the issue 10 min earlier,
                 except the fresh reference anchor s_freshPm25 (the label anchor)
     all         wc + every s_ feature incl. the reference (and its availability flag)
                 from 10 min earlier: the whole input set is >= 10 min stale
  score <variant>  evaluate.compare on the test split + paired original final on the same rows
"""
import sys, time
sys.path.insert(0, r"research\models")
sys.path.insert(0, r"research\models\gbm-quantile")
import numpy as np, pandas as pd
from harness import config as hc, dataset, features, sources, walkforward, scoring
import gbmq, evaluate as ev

OUT = hc.OUT / "gbm-quantile" / "leakcheck"
DELAY = 600


def delayed_runs(table):
    runs = sources.forecast_runs(table)            # sorted by visibility clock
    out = []
    for i, (f, v, k, p) in enumerate(runs):
        nxt = runs[i + 1][1] if i + 1 < len(runs) else np.inf
        out.append((f, max(nxt, v + DELAY), k, p))
    out.sort(key=lambda r: (r[1], r[0]))
    return tuple(out)


def build():
    ds = dataset.load()
    issues = ds.index.to_numpy(np.int64)
    orig = sources.forecast_runs
    cache = {t: delayed_runs(t) for t in ("weather_forecast_runs", "air_quality_forecast_runs")}
    sources.forecast_runs = lambda t: cache[t]
    t0 = time.time()
    w = features.weather_group(issues)
    c = features.cams_group(issues)
    sources.forecast_runs = orig
    out = pd.concat([w, c], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "delayed_wc.parquet")
    base = ds[out.columns]
    print(f"built {out.shape} in {time.time() - t0:.0f}s")
    print("weather age (min) orig p50/p90:", np.nanpercentile(ds.w_age_min, [50, 90]), "delayed:", np.nanpercentile(out.w_age_min, [50, 90]))
    print("cams age (min) orig p50/p90:", np.nanpercentile(ds.c_age_min, [50, 90]), "delayed:", np.nanpercentile(out.c_age_min, [50, 90]))
    same = (base.w_age_min == out.w_age_min).mean()
    print("share of rows with the same weather vintage as before:", round(float(same), 4))
    print("weather missing orig/delayed:", int(ds.w_age_min.isna().sum()), int(out.w_age_min.isna().sum()),
          "cams missing:", int(ds.c_age_min.isna().sum()), int(out.c_age_min.isna().sum()))


def variant_ds(name):
    ds = dataset.load().copy()
    wc = pd.read_parquet(OUT / "delayed_wc.parquet").reindex(ds.index)
    for col in wc.columns:
        ds[col] = wc[col]
    assert (np.diff(ds.index.to_numpy()) == 300).all()
    steps = DELAY // 300
    s_cols = [c for c in ds.columns if c.startswith("s_")]
    if name == "all_anchor":
        move = [c for c in s_cols if c != "s_freshPm25"]
        ds[move] = ds[move].shift(steps)
    elif name == "all":
        ds[s_cols] = ds[s_cols].shift(steps)
        ds["m_fresh_available"] = ds["m_fresh_available"].shift(steps).fillna(False).astype(bool)
    else:
        assert name == "wc"
    return ds


def checked(fp, log):
    def wrapped(train, test, cutoff):
        assert not any(c.startswith("y_") for c in test.columns), "labels in test frame"
        idx = train.index.to_numpy(np.int64)
        assert idx.max() < cutoff - gbmq.ARRIVAL_EMBARGO_SECONDS and idx.min() > cutoff - 28 * 86400
        assert (train.y_arrival_complete.to_numpy(np.int64) < cutoff).all()
        ride = (train.y_ride_joint_valid.astype(bool).to_numpy() & (train.y_ride_complete.to_numpy(np.int64) < cutoff)
                & (idx < cutoff - gbmq.RIDE_EMBARGO_SECONDS))
        assert test.index.min() >= cutoff and test.index.max() < cutoff + 86400
        log(f"  cutoff {hc.myt(cutoff):%m-%d}: arrival train max issue {(cutoff - idx.max()) / 60:.0f} min before, "
            f"max label close {(cutoff - train.y_arrival_complete.max()) / 60:.0f} min before; "
            f"ride rows {int(ride.sum())}, max ride close {(cutoff - train.y_ride_complete[ride].max()) / 60:.0f} min before")
        return fp(train, test, cutoff)
    return wrapped


def run(name):
    ds = variant_ds(name)
    cfg = ev.CONFIGS["final"]
    cols = gbmq.columns(ds, cfg)
    assert len(cols) == 284
    log = lambda s: print(f"[{name}] {s}", flush=True)
    fp = checked(gbmq.make_fit_predict(cfg, cols), log)
    t0 = time.time()
    pred = walkforward.walk_forward(ds, fp, targets=("arrival",), lookback_days=cfg.lookback_days,
                                    embargo_minutes=gbmq.ARRIVAL_EMBARGO_SECONDS // 60, days=ev.split_days("test"), log=log)
    pred.to_parquet(OUT / f"{name}__test.parquet")
    log(f"saved rows={len(pred)} in {time.time() - t0:.0f}s")


def score(name):
    pred = pd.read_parquet(OUT / f"{name}__test.parquet")
    orig = pd.read_parquet(ev.pred_path("final", "test"))
    if name == "all":
        # reference-relative class probabilities are relative to the stale anchor; score absolute heads only
        pred = pred[[c for c in pred.columns if c.startswith("pred_") or c in ("p_ride_le70",)]]
    common = pred.notna() & orig.reindex(pred.index)[pred.columns].notna()
    pred = pred.where(common)
    orig = orig.reindex(pred.index)[pred.columns].where(common)
    rows_d, _ = ev.compare(pred, "test")
    rows_o, _ = ev.compare(orig, "test")
    print(f"## variant {name}: delayed candidate vs original candidate on identical rows\n")
    header = ["Target", "Metric", "Persistence / prior", "Incumbent replay", "Original final", f"Delayed ({name})", "n"]
    rows = []
    for a, b in zip(rows_o, rows_d):
        assert a[:2] == b[:2]
        rows.append((a[0], a[1], b[2], b[3], a[4], b[4], b[5]))
    print(ev.table(header, rows))
    if name == "all":
        dsx = dataset.load()
        e = ev.evaluation_frame("test")
        stale = dsx.s_freshPm25.shift(2).reindex(e.index)
        for col, y in (("pred_arrival", "y_arrival"), ("pred_ride_mean", "y_ride_mean")):
            m = pred[col].reindex(e.index).notna() & e[y].notna() & stale.notna()
            print(f"\n{col}: persistence with a 10-min-stale reference MAE {np.mean(np.abs(e.loc[m, y] - stale[m])):.2f} "
                  f"vs fresh reference {np.mean(np.abs(e.loc[m, y] - e.loc[m, 'y_ref'])):.2f} (n={int(m.sum())})")


if __name__ == "__main__":
    action = sys.argv[1]
    {"build": lambda: build(), "run": lambda: run(sys.argv[2]), "score": lambda: score(sys.argv[2])}[action]()
