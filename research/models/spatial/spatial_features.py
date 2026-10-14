"""Spatial features for the TTDI issue grid: AirGradient neighbours, upwind weighting, regional CAMS.

Representation (identical in training and test so the model sees one kind of input):
* Every neighbour is a 15-min bucket series (bucket (E-900, E], value = mean PM2.5 on the
  AirGradient map scale, i.e. EPA-corrected). Before 2026-10-05 19:15 the buckets are the
  provider's history buckets (backfilled on 2026-10-05 19:30); after that they are means of
  the live 'current' polls (every 5 min, real receipt clocks).
* A bucket is visible at max(real receipt of its newest poll, E + lag). After Oct 5 the
  receipts are real, so on test days from 2026-10-06 the inputs are leak-free for any lag
  >= the poll latency (max 5 min). History buckets have no real receipt (the backfill came
  later); for them E + lag is an assumption. That assumption only touches training rows and
  tuning-period test rows, never the final test days.
* TTDI on the same scale: our sensor's raw pm02/RH bucket means through the published
  AirGradient EPA correction (checked against the TTDI map duplicate 956, see REPORT.md),
  using only readings visible at the issue (harness visibility clock).

Wind is a feature input, never a rule: per-station upwind alignment from the issued
Open-Meteo wind (10 m and 925 hPa, mean of +0..+90) weights station changes and gradients.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from harness import config, sources

TTDI = (3.1411106, 101.6274985)          # sensor / map duplicate 956 coordinates
STATIONS = {                             # 956 (TTDI duplicate) excluded as a predictor
    1087: "setia", 52426946: "cyber", 98982064: "enviro",
    99667152: "montkiara", 109266655: "setapak", 111145052: "klcc",
}
BUCKET = 900
MAX_AGE = 90 * 60


def epa_correct(pm, rh):
    """AirGradient map EPA correction (published getEpaCorrectedPM), vectorised, no rounding."""
    pm, rh = np.asarray(pm, float), np.asarray(rh, float)
    out = np.full(pm.shape, np.nan)
    a1 = pm < 30
    out[a1] = 0.524 * pm[a1] - 0.0862 * rh[a1] + 5.75
    a2 = (pm >= 30) & (pm < 50)
    a = pm[a2] / 20 - 1.5
    out[a2] = (0.786 * a + 0.524 * (1 - a)) * pm[a2] - 0.0862 * rh[a2] + 5.75
    a3 = (pm >= 50) & (pm < 210)
    out[a3] = 0.786 * pm[a3] - 0.0862 * rh[a3] + 5.75
    a4 = (pm >= 210) & (pm < 260)
    a = pm[a4] / 50 - 21 / 5
    out[a4] = (0.69 * a + 0.786 * (1 - a)) * pm[a4] - 0.0862 * rh[a4] * (1 - a) + 2.966 * a + 5.75 * (1 - a)
    a5 = pm >= 260
    out[a5] = 2.966 + 0.69 * pm[a5] + 8.84e-4 * pm[a5] ** 2
    nan_rh = ~np.isfinite(rh) & np.isfinite(pm)
    out[nan_rh] = pm[nan_rh]
    out[pm == 0] = 0.0
    return np.maximum(out, 0.0)


def geometry():
    """Distance (km) and unit vector (east, north) from TTDI to each station."""
    out = {}
    with __import__("contextlib").closing(__import__("sqlite3").connect(
            f"file:{config.NEIGHBOR_DB.as_posix()}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT location_id, latitude, longitude FROM locations").fetchall()
    for location, lat, lon in rows:
        if location not in STATIONS:
            continue
        dn = (lat - TTDI[0]) * 111.2
        de = (lon - TTDI[1]) * 111.2 * math.cos(math.radians(TTDI[0]))
        d = math.hypot(de, dn)
        out[location] = (d, de / d, dn / d)
    return out


def ttdi_buckets() -> pd.DataFrame:
    """TTDI 15-min bucket means with the newest visibility clock of the readings inside."""
    table = sources.sensor_table()
    t = table.loc[np.isfinite(table.pm02)].copy()
    t["E"] = (np.ceil(t.epoch / BUCKET) * BUCKET).astype(np.int64)
    g = t.groupby("E").agg(pm=("pm02", "mean"), rh=("rhum", "mean"), n=("pm02", "size"),
                           vis=("visible_epoch", "max"))
    g = g.loc[g.n >= 2]
    g["epa"] = epa_correct(g.pm.to_numpy(), g.rh.to_numpy())
    return g


def neighbour_buckets(lag_seconds: int) -> dict:
    """{location: DataFrame(E, value, vis)} on one 15-min representation."""
    table = sources.neighbor_table()
    out = {}
    for location in STATIONS:
        g = table.loc[table.location_id == location]
        hist = g.loc[g.kind == "history", ["end_epoch", "pm25"]].rename(columns={"end_epoch": "E", "pm25": "value"})
        hist["vis"] = hist.E + lag_seconds                       # assumption: no real receipt for backfill
        cur = g.loc[g.kind == "current"].copy()
        cur["E"] = (np.ceil(cur.end_epoch / BUCKET) * BUCKET).astype(np.int64)
        cur = cur.groupby("E").agg(value=("pm25", "mean"), receipt=("receipt_epoch", "max")).reset_index()
        cur["vis"] = np.maximum(cur.receipt, cur.E + lag_seconds)  # real receipt clock
        cur = cur.loc[~cur.E.isin(hist.E)]
        frame = pd.concat([hist, cur[["E", "value", "vis"]]]).sort_values("E").reset_index(drop=True)
        out[location] = frame
    return out


def weighted_median(values, weights):
    """Column-wise weighted median of a (stations x issues) array; NaN where no weight."""
    v = np.where(np.isfinite(values), values, np.nan)
    w = np.where(np.isfinite(v), weights, 0.0)
    order = np.argsort(np.where(np.isfinite(v), v, np.inf), axis=0)
    vs = np.take_along_axis(v, order, axis=0)
    ws = np.take_along_axis(w, order, axis=0)
    total = ws.sum(axis=0)
    cum = np.cumsum(ws, axis=0)
    k = np.argmax(cum >= 0.5 * total[None, :], axis=0)
    out = vs[k, np.arange(v.shape[1])]
    return np.where(total > 0, out, np.nan)


def _latest_visible(E, vis, issues):
    """Index of the newest bucket end visible by each issue (prefix max over vis order)."""
    order = np.argsort(vis, kind="stable")
    v, e = vis[order], E[order]
    best = np.maximum.accumulate(e)
    k = np.searchsorted(v, issues, side="right") - 1
    latest = np.where(k >= 0, best[np.maximum(k, 0)], -1)
    return latest


def build(ds: pd.DataFrame, lag_seconds: int = 300) -> pd.DataFrame:
    issues = ds.index.to_numpy(np.int64)
    geo = geometry()
    ttdi = ttdi_buckets()
    t_epa = ttdi.epa.to_dict()
    t_vis = ttdi.vis.to_dict()

    def ttdi_at(ends):
        val = np.array([t_epa.get(int(e), np.nan) if e > 0 else np.nan for e in ends])
        vis = np.array([t_vis.get(int(e), np.inf) if e > 0 else np.inf for e in ends])
        val[vis > issues] = np.nan                      # strict: only readings visible at the issue
        return val

    out = pd.DataFrame(index=ds.index)
    buckets = neighbour_buckets(lag_seconds)
    per = {}
    for location, frame in buckets.items():
        name = STATIONS[location]
        E = frame.E.to_numpy(np.int64)
        val = dict(zip(E, frame.value.to_numpy(float)))
        vis = dict(zip(E, frame.vis.to_numpy(float)))
        latest = _latest_visible(E, frame.vis.to_numpy(float), issues)
        ok = (latest > 0) & (issues - latest <= MAX_AGE)
        latest = np.where(ok, latest, -1)

        def value_at(ends):
            v = np.array([val.get(int(e), np.nan) if e > 0 else np.nan for e in ends])
            w = np.array([vis.get(int(e), np.inf) if e > 0 else np.inf for e in ends])
            v[w > issues] = np.nan
            return v

        now = value_at(latest)
        lag30 = value_at(np.where(latest > 0, latest - 1800, -1))
        lag60 = value_at(np.where(latest > 0, latest - 3600, -1))
        t_now = ttdi_at(latest)
        t60 = ttdi_at(np.where(latest > 0, latest - 3600, -1))
        per[location] = {
            "age": np.where(latest > 0, (issues - latest) / 60, np.nan),
            "level": now,
            "diff": now - t_now,                          # spatial gradient at a common bucket
            "chg30": now - lag30,
            "chg60": now - lag60,
            "rel60": (now - lag60) - (t_now - t60),       # neighbour change minus TTDI change
        }
        for key in ("diff", "chg60", "age"):
            out[f"sp_{name}_{key}"] = per[location][key]

    def stack(key):
        return np.vstack([per[loc][key] for loc in STATIONS])

    diff, chg30, chg60, rel60 = stack("diff"), stack("chg30"), stack("chg60"), stack("rel60")
    level, age = stack("level"), stack("age")
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        # Network aggregates are medians: one glitching station (e.g. KLCC 264 on Sep 11) must not
        # move them. Min/max/sd are kept for the tree models only.
        out["sp_count"] = np.isfinite(level).sum(axis=0)
        out["sp_net_level"] = np.nanmedian(level, axis=0)
        out["sp_net_diff"] = np.nanmedian(diff, axis=0)
        out["sp_net_diff_mad"] = np.nanmedian(np.abs(diff - out["sp_net_diff"].to_numpy()[None, :]), axis=0)
        out["sp_net_chg30"] = np.nanmedian(chg30, axis=0)
        out["sp_net_chg60"] = np.nanmedian(chg60, axis=0)
        out["sp_net_chg60_min"] = np.nanmin(chg60, axis=0)
        out["sp_net_chg60_max"] = np.nanmax(chg60, axis=0)
        out["sp_net_rel60"] = np.nanmedian(rel60, axis=0)
        # TTDI fresh reference on the map scale vs the newest neighbour network level
        t_ref_epa = epa_correct(ds.s_freshPm25.to_numpy(float), ds.s_freshHumidity.to_numpy(float))
        out["sp_net_level_minus_ref"] = out["sp_net_level"].to_numpy() - t_ref_epa

        # Upwind weighting by issued wind (toward-vector u, v in km/h), mean of +0..+90.
        east = np.array([geo[loc][1] for loc in STATIONS])[:, None]
        north = np.array([geo[loc][2] for loc in STATIONS])[:, None]
        dist = np.array([geo[loc][0] for loc in STATIONS])[:, None]
        for tag, u_cols, v_cols in (("w10", [f"w_wind10_u_{l}" for l in (0, 30, 60, 90)],
                                     [f"w_wind10_v_{l}" for l in (0, 30, 60, 90)]),
                                    ("w925", [f"w_wind925_u_{l}" for l in (0, 30, 60, 90)],
                                     [f"w_wind925_v_{l}" for l in (0, 30, 60, 90)])):
            u = ds[u_cols].mean(axis=1).to_numpy(float)[None, :]
            v = ds[v_cols].mean(axis=1).to_numpy(float)[None, :]
            speed = np.hypot(u, v)
            align = -(east * u + north * v) / np.where(speed > 0, speed, np.nan)   # +1 = station upwind
            up = np.where(np.isfinite(diff), np.clip(align, 0, None) ** 2, 0.0)
            down = np.where(np.isfinite(diff), np.clip(-align, 0, None) ** 2, 0.0)
            out[f"sp_{tag}_up_diff"] = weighted_median(diff, up)
            out[f"sp_{tag}_up_chg60"] = weighted_median(chg60, up)
            out[f"sp_{tag}_up_chg30"] = weighted_median(chg30, up)
            out[f"sp_{tag}_up_rel60"] = weighted_median(rel60, up)
            out[f"sp_{tag}_down_diff"] = weighted_median(diff, down)
            # Fraction of the upwind gradient that plain advection would bring by +90 min.
            along = dist * np.clip(align, 1e-3, None)                       # km along the wind
            reach = np.clip((90 + np.nan_to_num(age)) / 60 * speed / along, 0, 1)
            out[f"sp_{tag}_adv90"] = weighted_median(reach * diff, up)
            best = np.argmax(np.where(np.isfinite(diff), align, -9), axis=0)
            cols = np.arange(diff.shape[1])
            eta = along[best, cols] / np.where(speed[0] > 0, speed[0], np.nan) * 60 - age[best, cols]
            out[f"sp_{tag}_up_eta_min"] = np.where(align[best, cols] > 0.5, eta, np.nan)
            out[f"sp_{tag}_up_align"] = align[best, cols]

        # Regional CAMS grid (from 2026-10-01): along-wind gradient and +90 change.
        u = ds[[f"w_wind10_u_{l}" for l in (0, 30, 60, 90)]].mean(axis=1).to_numpy(float)
        v = ds[[f"w_wind10_v_{l}" for l in (0, 30, 60, 90)]].mean(axis=1).to_numpy(float)
        speed = np.hypot(u, v)
        out["sp_rp_up_grad"] = -(ds.rp_ew_grad_0.to_numpy(float) * u + ds.rp_ns_grad_0.to_numpy(float) * v) / np.where(speed > 0, speed, np.nan)
        out["sp_rp_delta90"] = ds.rp_mean_90.to_numpy(float) - ds.rp_mean_0.to_numpy(float)
        out["sp_cams_delta90"] = ds.c_pm25_90.to_numpy(float) - ds.c_pm25_0.to_numpy(float)
    return out


AGGREGATE_PREFIXES = ("sp_net_", "sp_w10_", "sp_w925_", "sp_rp_", "sp_cams_", "sp_count")


def columns(frame, per_station=True):
    cols = [c for c in frame.columns if c.startswith(AGGREGATE_PREFIXES)]
    if per_station:
        cols += [c for c in frame.columns if c.startswith("sp_") and c not in cols]
    return cols


if __name__ == "__main__":
    import sys
    import time
    from harness import dataset
    lag = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    t0 = time.time()
    ds = dataset.load()
    feats = build(ds, lag)
    path = config.OUT / "spatial" / f"spatial_features_lag{lag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path)
    print(path, feats.shape, f"{time.time() - t0:.0f}s")
    print(feats.describe().T.to_string())
