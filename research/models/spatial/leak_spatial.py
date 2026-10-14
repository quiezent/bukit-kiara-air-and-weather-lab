"""Leakage stress test of spatial_v1 (leakage skeptic's pass).

    python -B -m spatial.leak_spatial reproduce            # current code vs the stored features and final-day preds
    python -B -m spatial.leak_spatial audit [N]            # independent raw-SQL re-derivation of station and aggregate inputs
    python -B -m spatial.leak_spatial build                # delayed inputs (see below)
    python -B -m spatial.leak_spatial run A|C <variant ...> [--final]
    python -B -m spatial.leak_spatial score A|C [--split final|tune|all]

Delay configurations (both use the unchanged spatial_v1 / ablation code in run_spatial):
  A  exogenous inputs late: neighbour buckets visible at max(receipt + 10 min, end + 15 min),
     TTDI comparison buckets + 10 min, and the previous vintage of Open-Meteo weather (one
     15-min fetch cycle), CAMS (one 75-min cycle) and the regional CAMS grid (one 60-min cycle).
     The TTDI sensor block (the incumbent's inputs and the persistence anchor) stays at its clock.
  C  A plus the whole TTDI sensor block (s_*, m_*, mean-reversion controls) taken from the issue
     10 min earlier: every input is at least 10 min late. Targets stay those of the issue.
Outputs: data/models/spatial_leak/.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import argparse
import gzip
import json
import math
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd

try:                                   # be polite to production: idle priority on Windows
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass

from harness import config, dataset, features, report, scoring, sources, walkforward
from spatial import run_spatial as rs
from spatial import score_spatial as ss
from spatial import spatial_features as sf

OUT = config.OUT / "spatial_leak"
EXTRA = 600                            # extra seconds on every clocked input
FINAL = config.myt_epoch("2026-10-06")
EXO_PREFIXES = ("w_", "c_", "rp_", "rw_")
SENSOR_PREFIXES = ("s_", "m_")


# ----------------------------------------------------------------------------- delayed sources
def _previous_vintage(runs, vis_index=1):
    """Each run becomes visible only when the next run did: issues always see the previous vintage."""
    runs = sorted(runs, key=lambda r: (r[vis_index], r[0]))
    out = []
    for i, r in enumerate(runs):
        nxt = runs[i + 1][vis_index] if i + 1 < len(runs) else float("inf")
        r = list(r)
        r[vis_index] = float(nxt)
        out.append(tuple(r))
    return tuple(out)


class Patched:
    """Context manager that delays sources module-wide (sources functions are looked up at call time)."""

    def __init__(self, exo=True, sensor=True, neighbours=True):
        self.exo, self.sensor, self.neighbours = exo, sensor, neighbours

    def __enter__(self):
        self.saved = {k: getattr(sources, k) for k in ("forecast_runs", "regional_pm_runs", "sensor_table", "neighbor_table")}
        fr, rp, st, nt = (self.saved[k] for k in ("forecast_runs", "regional_pm_runs", "sensor_table", "neighbor_table"))
        if self.exo:
            sources.forecast_runs = lambda table: _previous_vintage(fr(table))
            sources.regional_pm_runs = lambda: _previous_vintage(rp())
        if self.sensor:
            def sensor_late():
                t = st().copy()
                t["visible_epoch"] = t.visible_epoch + EXTRA
                return t
            sources.sensor_table = sensor_late
        if self.neighbours:
            def neighbours_late():
                t = nt().copy()
                t["receipt_epoch"] = t.receipt_epoch + EXTRA
                return t
            sources.neighbor_table = neighbours_late
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(sources, k, v)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = dataset.load()
    issues = ds.index.to_numpy(np.int64)
    t0 = time.time()
    # Sanity: the unpatched feature functions reproduce the dataset on a sample.
    rng = np.random.default_rng(3)
    sample = np.sort(rng.choice(issues, 150, replace=False))
    for fn in (features.weather_group, features.cams_group, features.regional_group):
        got = fn(sample)
        ref = ds.loc[sample, got.columns]
        diff = np.nanmax(np.abs(got.to_numpy(float) - ref.to_numpy(float)))
        nan_mismatch = int((got.isna().to_numpy() != ref.isna().to_numpy()).sum())
        print(f"reproduce {fn.__name__}: max|diff| {diff:.2e}, NaN mismatches {nan_mismatch}", flush=True)
    with Patched(exo=True, sensor=False, neighbours=False):
        w = features.weather_group(issues)
        c = features.cams_group(issues)
        r = features.regional_group(issues)
    exo = pd.concat([w, c, r], axis=1)
    exo.to_parquet(OUT / "exo_previous_vintage.parquet")
    print(f"exogenous previous-vintage columns {exo.shape} {time.time() - t0:.0f}s", flush=True)
    for col in ("w_age_min", "c_age_min", "rp_age_min"):
        print(f"  {col}: original median {ds[col].median():.1f} -> delayed {exo[col].median():.1f}; "
              f"coverage {ds[col].notna().mean():.3f} -> {exo[col].notna().mean():.3f}")
    dsd = ds.copy()
    for col in exo.columns:
        dsd[col] = exo[col]
    with Patched(exo=False, sensor=True, neighbours=True):
        sp = sf.build(dsd, 300 + EXTRA)       # history end + 15 min; current max(receipt + 10, end + 15)
    sp.to_parquet(OUT / "spatial_features_A.parquet")
    ref_late = ds[["s_freshPm25", "s_freshHumidity"]].shift(2)
    sp_c = sp.copy()
    sp_c["sp_net_level_minus_ref"] = sp.sp_net_level.to_numpy() - sf.epa_correct(
        ref_late.s_freshPm25.to_numpy(float), ref_late.s_freshHumidity.to_numpy(float))
    sp_c.to_parquet(OUT / "spatial_features_C.parquet")
    orig = pd.read_parquet(config.OUT / "spatial" / "spatial_features_lag300.parquet")
    fin = sp.index >= FINAL
    print("station ages (min), final split: original vs delayed")
    for name in sf.STATIONS.values():
        a, b = orig.loc[fin, f"sp_{name}_age"], sp.loc[fin, f"sp_{name}_age"]
        print(f"  {name:10s} {a.median():5.1f} -> {b.median():5.1f}  (min {a.min():.1f} -> {b.min():.1f})")
    print(f"done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- frames for configs
def load_config(cfg: str) -> pd.DataFrame:
    ds = dataset.load()
    index = ds.index.to_numpy(np.int64)
    assert (np.diff(index) == 300).all(), "grid must be contiguous for the 10-min sensor shift"
    exo = pd.read_parquet(OUT / "exo_previous_vintage.parquet")
    for col in exo.columns:
        ds[col] = exo[col]
    if cfg == "C":
        cols = [c for c in ds.columns if c.startswith(SENSOR_PREFIXES)]
        ds[cols] = ds[cols].shift(2)
        ds["m_fresh_available"] = ds.m_fresh_available.astype("boolean").fillna(False).astype(bool)
    sp = pd.read_parquet(OUT / f"spatial_features_{cfg}.parquet")
    ds = ds.join(sp)
    seq = [c for c in ds.columns if c.startswith("s_seq_pm25_")]
    ages = np.array([int(c.split("_")[3]) for c in seq])
    bins = ds[seq].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean90 = np.nanmean(bins[:, ages < 90], axis=1)
        mean180 = np.nanmean(bins, axis=1)
        old = np.nanmean(bins[:, ages >= 150], axis=1)
        new = np.nanmean(bins[:, ages < 30], axis=1)
    ds["x_dev90"] = ds.s_freshPm25 - mean90
    ds["x_dev180"] = ds.s_freshPm25 - mean180
    ds["x_mean180"] = mean180
    ds["x_slope180"] = new - old
    return ds


def _final_days_only():
    original = walkforward.eval_days
    walkforward.eval_days = lambda ds, start=None, end=None: original(ds, start=FINAL, end=end)
    return original


def run(cfg, variants, final_only):
    ds = load_config(cfg)
    rs.OUT = OUT
    if final_only:
        _final_days_only()
    for v in variants:
        rs.run(v, ds, f"{cfg}{'_final' if final_only else ''}")


def reproduce():
    """Current code reproduces the stored lag-300 features and the final-day candidate preds."""
    ds = dataset.load()
    feats = sf.build(ds, 300)
    stored = pd.read_parquet(config.OUT / "spatial" / "spatial_features_lag300.parquet")
    a, b = feats[stored.columns].to_numpy(float), stored.to_numpy(float)
    print(f"features: max|diff| {np.nanmax(np.abs(a - b)):.2e}, NaN mismatches {int((np.isnan(a) != np.isnan(b)).sum())}")
    data = rs.load(300)
    rs.OUT = OUT
    _final_days_only()
    for v in ("lr28_sp", "hgbr_sp"):
        rs.run(v, data, "repro_final")
        new = pd.read_parquet(OUT / f"pred_{v}_lagrepro_final.parquet")
        old = pd.read_parquet(config.OUT / "spatial" / f"pred_{v}_lag300.parquet").reindex(new.index)
        d = np.nanmax(np.abs(new.to_numpy(float) - old[new.columns].to_numpy(float)))
        print(f"{v}: final-day preds max|diff| vs stored {d:.2e} over {len(new)} rows")


# ----------------------------------------------------------------------------- independent raw-SQL audit
def _decode(blob):
    raw = bytes(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else blob
    if isinstance(raw, bytes):
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _wmedian(values, weights):
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return np.nan
    v, w = values[ok], weights[ok]
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    k = int(np.argmax(np.cumsum(w) >= 0.5 * w.sum()))
    return float(v[k])


def audit(n, shift=0):
    """Recompute station and aggregate inputs from raw SQL at random issues (both splits)."""
    feats = pd.read_parquet(config.OUT / "spatial" / "spatial_features_lag300.parquet")
    ds = dataset.load(columns=["s_freshPm25", "s_freshHumidity"])
    rng = np.random.default_rng(11 + shift)
    fin = feats.index[feats.index >= FINAL].to_numpy(np.int64)
    tune = feats.index[(feats.index >= config.EVAL_START_EPOCH) & (feats.index < FINAL)].to_numpy(np.int64)
    issues = np.concatenate([rng.choice(fin, n, replace=False), rng.choice(tune, n // 2, replace=False)])
    nconn = sqlite3.connect(f"file:{config.NEIGHBOR_DB.as_posix()}?mode=ro", uri=True)
    sconn = sqlite3.connect(f"file:{config.SNAPSHOT_DB.as_posix()}?mode=ro", uri=True)
    geo = {}
    for loc, lat, lon in nconn.execute("SELECT location_id, latitude, longitude FROM locations"):
        if loc in sf.STATIONS:
            dn = (lat - sf.TTDI[0]) * 111.2
            de = (lon - sf.TTDI[1]) * 111.2 * math.cos(math.radians(sf.TTDI[0]))
            d = math.hypot(de, dn)
            geo[loc] = (d, de / d, dn / d)
    # weather runs with their own visibility (receipt when known, else fetched + 5 s)
    rec = dict(sconn.execute(
        "SELECT r.source_fetched_epoch, MIN(MAX(r.available_epoch, v.confirmed_epoch)) FROM source_forecast_receipts r "
        "JOIN source_forecast_visibility v ON v.receipt_id=r.receipt_id WHERE r.source_table='weather_forecast_runs' "
        "GROUP BY r.source_fetched_epoch").fetchall())
    wruns = [(float(rec.get(f, f + 5)), f) for (f,) in sconn.execute(
        "SELECT fetched_epoch FROM weather_forecast_runs WHERE source='Open-Meteo Best Match'")]
    wruns.sort()
    wvis = np.array([v for v, _ in wruns])
    mism, checked, compared = [], 0, 0
    keys_station = ("diff", "chg60", "age")
    for issue in issues:
        issue = int(issue)
        clock = issue + shift
        row = feats.loc[issue]
        per = {}
        for loc, name in sf.STATIONS.items():
            hist = nconn.execute(
                "SELECT o.bucket_start_epoch + 900, o.pm25 FROM observations o WHERE o.location_id=? AND o.kind='history' "
                "AND o.valid=1 AND o.measure='pm25' AND o.bucket_start_epoch + 900 + 300 <= ?", (loc, clock)).fetchall()
            cur = nconn.execute(
                "SELECT o.bucket_start_epoch, o.pm25, MAX(r.received_epoch, r.recorded_epoch) FROM observations o "
                "JOIN requests r ON r.request_id=o.request_id WHERE o.location_id=? AND o.kind='current' AND o.valid=1 "
                "AND o.measure='pm25' AND MAX(r.received_epoch, r.recorded_epoch) <= ?", (loc, clock)).fetchall()
            buckets = {int(e): float(v) for e, v in hist}
            first = {}
            for m, v, rcp in sorted(cur, key=lambda x: x[2]):
                first.setdefault(m, (v, rcp))
            cb = {}
            for m, (v, rcp) in first.items():
                e = int(math.ceil(m / 900) * 900)
                cb.setdefault(e, []).append(v)
            # a current bucket counts only when every poll measured in it was received by the clock
            all_cur = nconn.execute(
                "SELECT o.bucket_start_epoch, MIN(MAX(r.received_epoch, r.recorded_epoch)) FROM observations o JOIN requests r "
                "ON r.request_id=o.request_id WHERE o.location_id=? AND o.kind='current' AND o.valid=1 AND o.measure='pm25' "
                "GROUP BY o.bucket_start_epoch", (loc,)).fetchall()
            late = {int(math.ceil(m / 900) * 900) for m, rcp in all_cur if rcp > clock}
            for e, vals in cb.items():
                if e in buckets or e in late or e + 300 > clock:
                    continue
                buckets[e] = float(np.mean(vals))
            if not buckets:
                per[loc] = None
                continue
            e = max(buckets)
            if clock - e > 5400:
                per[loc] = None
                continue
            per[loc] = (e, buckets[e], buckets.get(e - 1800, np.nan), buckets.get(e - 3600, np.nan))
        # TTDI comparison buckets (receipt clock, else observation + 60 s)
        def ttdi(end):
            rows = sconn.execute(
                "SELECT r.epoch, r.pm02, r.rhum, (SELECT MIN(MAX(x.available_epoch, v.confirmed_epoch)) FROM "
                "sensor_observation_receipts x JOIN sensor_observation_visibility v ON v.receipt_id=x.receipt_id "
                "WHERE x.observed_epoch=r.epoch) FROM readings r WHERE r.epoch > ? AND r.epoch <= ? AND r.pm02 IS NOT NULL",
                (end - 900, end)).fetchall()
            if len(rows) < 2 or max((r[3] if r[3] is not None else r[0] + 60) for r in rows) > clock:
                return np.nan
            return float(sf.epa_correct(np.array([np.mean([r[1] for r in rows])]),
                                        np.array([np.nanmean([np.nan if r[2] is None else r[2] for r in rows])]))[0])
        level, diff, chg30, chg60, rel60, age = ([] for _ in range(6))
        for loc, name in sf.STATIONS.items():
            p = per[loc]
            if p is None:
                vals = dict(diff=np.nan, chg60=np.nan, age=np.nan)
                level.append(np.nan); diff.append(np.nan); chg30.append(np.nan); chg60.append(np.nan)
                rel60.append(np.nan); age.append(np.nan)
            else:
                e, now, l30, l60 = p
                t_now, t60 = ttdi(e), ttdi(e - 3600)
                vals = dict(diff=now - t_now, chg60=now - l60, age=(issue - e) / 60)
                level.append(now); diff.append(now - t_now); chg30.append(now - l30); chg60.append(now - l60)
                rel60.append((now - l60) - (t_now - t60)); age.append((clock - e) / 60)
            for k in keys_station:
                got, want = vals[k], row[f"sp_{name}_{k}"]
                compared += 1
                if not ((np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-6):
                    mism.append((config.myt(issue), name, k, got, want))
        level, diff, chg30, chg60, rel60, age = map(np.array, (level, diff, chg30, chg60, rel60, age))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg = {"sp_net_level": np.nanmedian(level), "sp_net_diff": np.nanmedian(diff),
                   "sp_net_chg30": np.nanmedian(chg30), "sp_net_chg60": np.nanmedian(chg60),
                   "sp_net_rel60": np.nanmedian(rel60)}
            agg["sp_net_diff_mad"] = np.nanmedian(np.abs(diff - agg["sp_net_diff"]))
            ref = ds.loc[issue]
            agg["sp_net_level_minus_ref"] = agg["sp_net_level"] - float(
                sf.epa_correct(np.array([ref.s_freshPm25]), np.array([ref.s_freshHumidity]))[0])
        # issued wind from the raw payload of the newest weather run visible at the clock
        k = int(np.searchsorted(wvis, clock, side="right")) - 1
        if k >= 0 and clock - wruns[k][1] <= 7200:
            (blob,) = sconn.execute("SELECT payload FROM weather_forecast_runs WHERE fetched_epoch=? AND "
                                    "source='Open-Meteo Best Match'", (wruns[k][1],)).fetchone()
            hourly = {int(p["epoch"]): p for p in _decode(blob).get("hourly", []) if p.get("epoch") is not None}
            ep = np.array(sorted(hourly), float)
            for tag, spd_f, dir_f in (("w10", "wind_speed_10m", "wind_direction_10m"),
                                      ("w925", "wind_speed_925hPa", "wind_direction_925hPa")):
                s = np.array([np.nan if hourly[int(x)].get(spd_f) is None else float(hourly[int(x)][spd_f]) for x in ep])
                d = np.array([np.nan if hourly[int(x)].get(dir_f) is None else float(hourly[int(x)][dir_f]) for x in ep])
                uu, vv = -s * np.sin(np.deg2rad(d % 360)), -s * np.cos(np.deg2rad(d % 360))
                times = issue + np.array([0, 30, 60, 90]) * 60
                def interp(y):
                    out = []
                    for t in times:
                        hi = int(np.searchsorted(ep, t, side="left"))
                        if hi < len(ep) and ep[hi] == t:
                            out.append(y[hi])
                        elif 0 < hi < len(ep) and ep[hi] - ep[hi - 1] <= 3600:
                            out.append(y[hi - 1] + (y[hi] - y[hi - 1]) * (t - ep[hi - 1]) / (ep[hi] - ep[hi - 1]))
                        else:
                            out.append(np.nan)
                    out = np.array(out, float)
                    return float(np.nanmean(out)) if np.isfinite(out).any() else np.nan   # pandas row mean skips NaN
                u, v = interp(uu), interp(vv)
                speed = math.hypot(u, v)
                if not np.isfinite(speed) or speed == 0:
                    continue
                align = np.array([-(geo[loc][1] * u + geo[loc][2] * v) / speed for loc in sf.STATIONS])
                up = np.where(np.isfinite(diff), np.clip(align, 0, None) ** 2, 0.0)
                agg[f"sp_{tag}_up_diff"] = _wmedian(diff, up)
                agg[f"sp_{tag}_up_rel60"] = _wmedian(rel60, up)
                dist = np.array([geo[loc][0] for loc in sf.STATIONS])
                along = dist * np.clip(align, 1e-3, None)
                reach = np.clip((90 + np.nan_to_num(age)) / 60 * speed / along, 0, 1)
                agg[f"sp_{tag}_adv90"] = _wmedian(reach * diff, up)
        for key, got in agg.items():
            want = row[key]
            compared += 1
            if not ((np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-6):
                mism.append((config.myt(issue), "agg", key, got, want))
        checked += 1
    tag = "independent re-derivation" if shift == 0 else f"positive control (clock +{shift // 60} min)"
    print(f"{tag}: {checked} issues ({n} final, {n // 2} tune), {compared} values compared, {len(mism)} mismatches")
    for m in mism[:15]:
        print("  ", m)
    return len(mism)


# ----------------------------------------------------------------------------- scoring
def score(cfg, split, final_only):
    ds = dataset.load()
    suffix = f"{cfg}{'_final' if final_only else ''}"
    load = lambda name, lag: pd.read_parquet(OUT / f"pred_{name}_lag{lag}.parquet")
    orig = ss.load_pred
    preds = {
        "spatial_v1 (reported)": pd.concat([orig("lr28_sp", 300), orig("hgbr_sp", 300)], axis=1),
        f"spatial_v1 delayed {cfg}": pd.concat([load("lr28_sp", suffix), load("hgbr_sp", suffix)], axis=1),
        f"ablation delayed {cfg}": pd.concat([load("lr28", suffix) if cfg == "C" else orig("lr28", 300),
                                              load("hgbr", suffix)], axis=1),
    }
    rows, raw = ss.score(ds, split, preds, suffix)
    print(f"config {cfg} split={split}")
    print(ss.table(rows))
    # equal-delay persistence for C (anchor 10 min old) on the point rows
    ev = ss.ev_frame(ds, split)
    inc = report.standard_replay(ds)
    m = inc.pred_arrival.reindex(ev.index).notna() & ev.y_arrival.notna()
    for v in preds.values():
        m &= v.pred_arrival.reindex(ev.index).notna()
    late_ref = ds.y_ref.shift(2).reindex(ev.index)
    m &= late_ref.notna()
    y = ev.y_arrival[m]
    print(f"\n+90 MAE on rows with a 10-min-old reference too (n={int(m.sum())}): persistence(t) "
          f"{(ev.y_ref[m] - y).abs().mean():.2f}, persistence(t-10) {(late_ref[m] - y).abs().mean():.2f}, "
          + ", ".join(f"{k} {(v.pred_arrival.reindex(ev.index)[m] - y).abs().mean():.2f}" for k, v in preds.items())
          + f", Ridge v4 {(inc.pred_arrival.reindex(ev.index)[m] - y).abs().mean():.2f}")
    # day-level paired view for the +90 point
    fin = ev.loc[m]
    days = []
    for day, part in fin.groupby("date_myt"):
        idx = part.index
        yy = part.y_arrival
        days.append({"day": day, "n": len(idx), "persist": (part.y_ref - yy).abs().mean(),
                     "ridge": (inc.pred_arrival.reindex(idx) - yy).abs().mean(),
                     **{k: (v.pred_arrival.reindex(idx) - yy).abs().mean() for k, v in preds.items()}})
    print(pd.DataFrame(days).round(2).to_markdown(index=False))
    (OUT / f"scores_{split}_{suffix}.json").write_text(json.dumps(raw, default=float, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("args", nargs="*")
    p.add_argument("--final", action="store_true")
    p.add_argument("--split", default="final")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        build()
    elif a.cmd == "reproduce":
        reproduce()
    elif a.cmd == "audit":
        n = int(a.args[0]) if a.args else 40
        audit(n)
        audit(n, shift=EXTRA)
    elif a.cmd == "run":
        run(a.args[0], a.args[1:], a.final)
    elif a.cmd == "score":
        score(a.args[0], a.split, a.final)
    else:
        raise SystemExit(a.cmd)
