"""Independent raw-SQL check of the spatial features on final-period issues (real receipt clocks).

    python -B -m spatial.audit_spatial [n_issues] [--lag 300]

For random issues from 2026-10-06, every station value is recomputed from the neighbour
archive with plain SQL (current polls whose receipt <= issue, 15-min buckets whose end + lag
<= issue) and the TTDI bucket from the readings table (receipt-or-legacy clock <= issue).
A positive control recomputes with a clock 20 min later and must disagree somewhere.
"""
from __future__ import annotations

import argparse
import math
import sqlite3

import numpy as np
import pandas as pd

from harness import config

STATIONS = {1087: "setia", 52426946: "cyber", 98982064: "enviro",
            99667152: "montkiara", 109266655: "setapak", 111145052: "klcc"}


def station_values(conn, location, issue, lag):
    rows = conn.execute(
        "SELECT o.bucket_start_epoch, o.pm25, MAX(r.received_epoch, r.recorded_epoch) FROM observations o "
        "JOIN requests r ON r.request_id=o.request_id WHERE o.location_id=? AND o.kind='current' "
        "AND o.valid=1 AND o.measure='pm25' AND MAX(r.received_epoch, r.recorded_epoch) <= ?",
        (location, issue)).fetchall()
    if not rows:
        return None
    first = {}
    for measured, pm, receipt in sorted(rows, key=lambda r: r[2]):
        first.setdefault(measured, pm)          # first receipt of each measurement
    buckets = {}
    for measured, pm in first.items():
        end = int(math.ceil(measured / 900) * 900)
        buckets.setdefault(end, []).append(pm)
    ends = sorted(e for e in buckets if e + lag <= issue)
    if not ends or issue - ends[-1] > 5400:
        return None
    e = ends[-1]
    mean = lambda k: float(np.mean(buckets[k])) if k in buckets and k + lag <= issue else np.nan
    return e, mean(e), mean(e - 3600)


def ttdi_bucket(conn, end, issue):
    rows = conn.execute(
        "SELECT r.epoch, r.pm02, r.rhum, (SELECT MIN(MAX(x.available_epoch, v.confirmed_epoch)) "
        "FROM sensor_observation_receipts x JOIN sensor_observation_visibility v ON v.receipt_id=x.receipt_id "
        "WHERE x.observed_epoch=r.epoch) FROM readings r WHERE r.epoch > ? AND r.epoch <= ?",
        (end - 900, end)).fetchall()
    rows = [r for r in rows if r[1] is not None]
    if len(rows) < 2:
        return np.nan
    visible = [(r[3] if r[3] is not None else r[0] + 60) for r in rows]
    if max(visible) > issue:
        return np.nan
    from spatial.spatial_features import epa_correct
    return float(epa_correct(np.array([np.mean([r[1] for r in rows])]), np.array([np.mean([r[2] for r in rows])]))[0])


def main(n, lag):
    feats = pd.read_parquet(config.OUT / "spatial" / f"spatial_features_lag{lag}.parquet")
    final = feats.loc[feats.index >= config.myt_epoch("2026-10-06")]
    rng = np.random.default_rng(7)
    issues = rng.choice(final.index.to_numpy(np.int64), n, replace=False)
    nconn = sqlite3.connect(f"file:{config.NEIGHBOR_DB.as_posix()}?mode=ro", uri=True)
    sconn = sqlite3.connect(f"file:{config.SNAPSHOT_DB.as_posix()}?mode=ro", uri=True)
    checked = mismatched = control_diff = 0
    for issue in issues:
        for location, name in STATIONS.items():
            got = station_values(nconn, location, int(issue), lag)
            row = final.loc[issue]
            if got is None:
                ok = not np.isfinite(row[f"sp_{name}_chg60"]) and not np.isfinite(row[f"sp_{name}_diff"])
            else:
                e, now, old = got
                tt = ttdi_bucket(sconn, e, int(issue))
                vals = {"age": (issue - e) / 60, "chg60": now - old, "diff": now - tt}
                ok = all((np.isnan(v) and np.isnan(row[f"sp_{name}_{k}"])) or abs(v - row[f"sp_{name}_{k}"]) < 1e-6
                         for k, v in vals.items())
                later = station_values(nconn, location, int(issue) + 1200, lag)
                control_diff += int(later is not None and abs(later[1] - now) > 1e-9)
            checked += 1
            mismatched += int(not ok)
            if not ok:
                print("MISMATCH", config.myt(issue), name, got, row[[f"sp_{name}_age", f"sp_{name}_chg60", f"sp_{name}_diff"]].to_dict())
    print(f"lag={lag}: {checked} station-issues checked, {mismatched} mismatches; "
          f"positive control (+20 min clock) changed the value in {control_diff}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n", nargs="?", type=int, default=25)
    p.add_argument("--lag", type=int, default=300)
    a = p.parse_args()
    main(a.n, a.lag)
