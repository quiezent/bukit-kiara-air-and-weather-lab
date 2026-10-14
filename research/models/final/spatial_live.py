"""Per-issue spatial (neighbour) features for the final point model, harness-free for porting.

This is the single-issue form of research/models/spatial/spatial_features.build (the grid
version used for training and evaluation). Given whatever neighbour polls and TTDI readings it
is handed, it applies the grid's visibility rules, so on the research snapshot it reproduces
the grid features exactly (final.spatial_live verify) and live it uses what has been received.

Inputs at issue time t
* Neighbour polls: production already archives them every 300 s
  (server/neighbor_pm_inputs.py -> research/external_airgradient/neighbors.sqlite3, table
  observations JOIN requests). Columns: location_id, kind ('current'; 'history' only for the
  Oct 5 backfill), end_epoch (= bucket_start_epoch + bucket_seconds), pm25 (AirGradient map
  scale), receipt_epoch (= MAX(received_epoch, recorded_epoch)).
* TTDI readings: epoch, pm02 (raw), rhum, visible_epoch (receipt clock; production uses 0 s
  latency for legacy rows).
* The live feature dict: s_freshPm25, s_freshHumidity, w_wind10_u/v_{0,30,60,90},
  w_wind925_u/v_{0,30,60,90}, rp_ew_grad_0, rp_ns_grad_0, rp_mean_0, rp_mean_90, c_pm25_0,
  c_pm25_90 (the last six only for the three regional columns; NaN is fine).

Rules (identical to the grid build)
* 15-min buckets (E-900, E]. A neighbour 'current' bucket = mean of its polls, visible at
  max(newest poll receipt, E + lag); lag = 300 s. A history bucket is visible at E + lag.
* A station's newest bucket must be at most 90 min old. Lagged values use buckets E-1800 and
  E-3600 and must themselves be visible.
* TTDI on the map scale: mean raw pm02 and RH of the bucket's readings (at least 2) through the
  AirGradient EPA correction; usable only if every reading in it is visible at t.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TTDI = (3.1411106, 101.6274985)
STATIONS = {  # location_id: (name, latitude, longitude); 956 (TTDI duplicate) excluded
    1087: ("setia", 3.1113299, 101.476412),
    52426946: ("cyber", 2.913, 101.640583),
    98982064: ("enviro", 3.02072, 101.714186),
    99667152: ("montkiara", 3.1696, 101.6528),
    109266655: ("setapak", 3.1922, 101.72318),
    111145052: ("klcc", 3.15665, 101.70853),
}
BUCKET = 900
MAX_AGE = 90 * 60
LAG = 300


def geometry():
    out = {}
    for location, (_, lat, lon) in STATIONS.items():
        dn = (lat - TTDI[0]) * 111.2
        de = (lon - TTDI[1]) * 111.2 * math.cos(math.radians(TTDI[0]))
        d = math.hypot(de, dn)
        out[location] = (d, de / d, dn / d)
    return out


GEO = geometry()


def epa_correct(pm, rh):
    """AirGradient map EPA correction (published getEpaCorrectedPM), vectorised, no rounding."""
    pm, rh = np.atleast_1d(np.asarray(pm, float)), np.atleast_1d(np.asarray(rh, float))
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


def weighted_median(values, weights):
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


class NeighbourBuckets:
    """Per-station bucket tables {E: (value, visible)} from raw polls (any subset of them)."""

    def __init__(self, polls: pd.DataFrame, lag: int = LAG):
        self.tables = {}
        for location in STATIONS:
            g = polls.loc[polls.location_id == location]
            hist = g.loc[g.kind == "history"]
            table = {int(e): (float(v), float(e + lag)) for e, v in zip(hist.end_epoch, hist.pm25)}
            cur = g.loc[g.kind == "current"].copy()
            if len(cur):
                cur["E"] = (np.ceil(cur.end_epoch / BUCKET) * BUCKET).astype(np.int64)
                agg = cur.groupby("E").agg(value=("pm25", "mean"), receipt=("receipt_epoch", "max"))
                for e, row in agg.iterrows():
                    if int(e) not in table:
                        table[int(e)] = (float(row.value), float(max(row.receipt, e + lag)))
            self.tables[location] = table

    def latest(self, location, issue):
        best = -1
        for e, (_, vis) in self.tables[location].items():
            if vis <= issue and e > best:
                best = e
        return best if best > 0 and issue - best <= MAX_AGE else -1

    def value(self, location, e, issue):
        hit = self.tables[location].get(int(e)) if e > 0 else None
        return hit[0] if hit is not None and hit[1] <= issue else np.nan


class TTDIBuckets:
    """TTDI 15-min bucket EPA means with the newest visibility clock of their readings."""

    def __init__(self, readings: pd.DataFrame):
        t = readings.loc[np.isfinite(readings.pm02)].copy()
        t["E"] = (np.ceil(t.epoch / BUCKET) * BUCKET).astype(np.int64)
        g = t.groupby("E").agg(pm=("pm02", "mean"), rh=("rhum", "mean"), n=("pm02", "size"),
                               vis=("visible_epoch", "max"))
        g = g.loc[g.n >= 2]
        self.epa = dict(zip(g.index.astype(int), epa_correct(g.pm.to_numpy(), g.rh.to_numpy())))
        self.vis = dict(zip(g.index.astype(int), g.vis.astype(float)))

    def at(self, e, issue):
        if e <= 0 or self.vis.get(int(e), np.inf) > issue:
            return np.nan
        return self.epa[int(e)]


def _num(features, key):
    v = features.get(key)
    return float(v) if v is not None else np.nan


def spatial_features(issue: int, nb: NeighbourBuckets, tb: TTDIBuckets, features: dict) -> dict:
    """All 47 sp_* inputs for one issue (UTC epoch seconds)."""
    out, per = {}, {}
    for location, (name, _, _) in STATIONS.items():
        latest = nb.latest(location, issue)
        now = nb.value(location, latest, issue)
        lag30 = nb.value(location, latest - 1800 if latest > 0 else -1, issue)
        lag60 = nb.value(location, latest - 3600 if latest > 0 else -1, issue)
        t_now = tb.at(latest, issue)
        t60 = tb.at(latest - 3600 if latest > 0 else -1, issue)
        per[location] = {"age": (issue - latest) / 60 if latest > 0 else np.nan, "level": now,
                         "diff": now - t_now, "chg30": now - lag30, "chg60": now - lag60,
                         "rel60": (now - lag60) - (t_now - t60)}
        for key in ("diff", "chg60", "age"):
            out[f"sp_{name}_{key}"] = per[location][key]

    def stack(key):
        return np.array([[per[loc][key]] for loc in STATIONS], float)

    diff, chg30, chg60, rel60 = stack("diff"), stack("chg30"), stack("chg60"), stack("rel60")
    level, age = stack("level"), stack("age")
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["sp_count"] = float(np.isfinite(level).sum())
            out["sp_net_level"] = float(np.nanmedian(level))
            out["sp_net_diff"] = float(np.nanmedian(diff))
            out["sp_net_diff_mad"] = float(np.nanmedian(np.abs(diff - out["sp_net_diff"])))
            out["sp_net_chg30"] = float(np.nanmedian(chg30))
            out["sp_net_chg60"] = float(np.nanmedian(chg60))
            out["sp_net_chg60_min"] = float(np.nanmin(chg60))
            out["sp_net_chg60_max"] = float(np.nanmax(chg60))
            out["sp_net_rel60"] = float(np.nanmedian(rel60))
            ref_epa = epa_correct(_num(features, "s_freshPm25"), _num(features, "s_freshHumidity"))[0]
            out["sp_net_level_minus_ref"] = out["sp_net_level"] - ref_epa
            east = np.array([GEO[loc][1] for loc in STATIONS])[:, None]
            north = np.array([GEO[loc][2] for loc in STATIONS])[:, None]
            dist = np.array([GEO[loc][0] for loc in STATIONS])[:, None]
            for tag, level_name in (("w10", "wind10"), ("w925", "wind925")):
                u = np.nanmean([_num(features, f"w_{level_name}_u_{l}") for l in (0, 30, 60, 90)])
                v = np.nanmean([_num(features, f"w_{level_name}_v_{l}") for l in (0, 30, 60, 90)])
                u, v = np.array([[u]]), np.array([[v]])
                speed = np.hypot(u, v)
                align = -(east * u + north * v) / np.where(speed > 0, speed, np.nan)
                up = np.where(np.isfinite(diff), np.clip(align, 0, None) ** 2, 0.0)
                down = np.where(np.isfinite(diff), np.clip(-align, 0, None) ** 2, 0.0)
                out[f"sp_{tag}_up_diff"] = float(weighted_median(diff, up)[0])
                out[f"sp_{tag}_up_chg60"] = float(weighted_median(chg60, up)[0])
                out[f"sp_{tag}_up_chg30"] = float(weighted_median(chg30, up)[0])
                out[f"sp_{tag}_up_rel60"] = float(weighted_median(rel60, up)[0])
                out[f"sp_{tag}_down_diff"] = float(weighted_median(diff, down)[0])
                along = dist * np.clip(align, 1e-3, None)
                reach = np.clip((90 + np.nan_to_num(age)) / 60 * speed / along, 0, 1)
                out[f"sp_{tag}_adv90"] = float(weighted_median(reach * diff, up)[0])
                best = int(np.argmax(np.where(np.isfinite(diff), align, -9), axis=0)[0])
                eta = along[best, 0] / (speed[0, 0] if speed[0, 0] > 0 else np.nan) * 60 - age[best, 0]
                out[f"sp_{tag}_up_eta_min"] = float(eta if align[best, 0] > 0.5 else np.nan)
                out[f"sp_{tag}_up_align"] = float(align[best, 0])
            u = np.nanmean([_num(features, f"w_wind10_u_{l}") for l in (0, 30, 60, 90)])
            v = np.nanmean([_num(features, f"w_wind10_v_{l}") for l in (0, 30, 60, 90)])
            speed = math.hypot(u, v) if np.isfinite(u) and np.isfinite(v) else np.nan
            out["sp_rp_up_grad"] = (-(_num(features, "rp_ew_grad_0") * u + _num(features, "rp_ns_grad_0") * v)
                                    / (speed if speed > 0 else np.nan))
            out["sp_rp_delta90"] = _num(features, "rp_mean_90") - _num(features, "rp_mean_0")
            out["sp_cams_delta90"] = _num(features, "c_pm25_90") - _num(features, "c_pm25_0")
    return out


def verify(n=200, seed=7, start=None):
    """Compare per-issue features with the grid parquet at random eval issues (research only)."""
    import json
    from harness import config, dataset, sources
    ds = dataset.load()
    grid = pd.read_parquet(config.OUT / "spatial" / "spatial_features_lag300.parquet")
    polls = sources.neighbor_table()
    readings = sources.sensor_table()[["epoch", "pm02", "rhum", "visible_epoch"]]
    rng = np.random.default_rng(seed)
    ev = ds.index[ds["eval"].to_numpy(bool)].to_numpy(np.int64)
    if start is not None:
        ev = ev[ev >= start]
    issues = np.sort(rng.choice(ev, size=n, replace=False))
    full_nb, full_tb = NeighbourBuckets(polls), TTDIBuckets(readings)
    worst, mism, live_mism, cells = {}, 0, 0, 0
    for issue in issues:
        feats = ds.loc[issue].to_dict()
        a = spatial_features(int(issue), full_nb, full_tb, feats)
        # live view: only polls/readings received by the issue
        seen_p = polls.loc[(polls.receipt_epoch <= issue) | (polls.kind == "history")]
        seen_r = readings.loc[readings.visible_epoch <= issue]
        lo = issue - 3 * 3600
        live = spatial_features(int(issue), NeighbourBuckets(seen_p.loc[seen_p.end_epoch > lo]),
                                TTDIBuckets(seen_r.loc[seen_r.epoch > lo]), feats)
        for c in grid.columns:
            g = float(grid.loc[issue, c])
            for val, is_live in ((a[c], False), (live[c], True)):
                same = (np.isnan(g) and np.isnan(val)) or (np.isfinite(g) and np.isfinite(val) and abs(g - val) <= 1e-9)
                if not same:
                    if is_live:
                        live_mism += 1
                    else:
                        mism += 1
                        worst[c] = worst.get(c, 0) + 1
            cells += 1
    result = {"issues": int(n), "from": None if start is None else int(start), "cells": cells, "mismatches_full_data": mism, "mismatches_live_view": live_mism,
              "by_column": worst}
    name = "spatial_live_verify.json" if start is None else "spatial_live_verify_holdout.json"
    (config.OUT / "final" / name).write_text(json.dumps(result, indent=1))
    print(result)


if __name__ == "__main__":
    verify()
