"""Independent audit of the dataset against raw SQL (no harness loader, feature or target code).

``checks.py`` rebuilds features from sources truncated by the harness's own
visibility clocks, so it cannot catch a wrong clock. This module re-derives
values straight from the snapshot tables with its own short implementations:

* leakage: for sampled issues, the sensor reference/lags/bins, the weather,
  CAMS, regional, Subang and neighbour records each group used, and the clock
  at which each of those records became visible (all must be <= the issue);
* the legacy sensor rule: on post-ledger grid issues, how often observation +
  lag would admit a reading that had not been received yet;
* targets: bucket medians, +90 arrival, ride mean/min/max, classes, first
  crossing and sessions recomputed from raw readings with the production
  definitions (coverage: >= 3 samples, both edges within 240 s, gaps <= 480 s).

    python -B -m harness.audit
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta
import gzip
import json
import sqlite3

import numpy as np
import pandas as pd

from . import config, dataset

EVENT_ISSUES = ("2026-09-28 12:15", "2026-09-29 19:15", "2026-10-01 16:05", "2026-10-06 04:00",
                "2026-10-06 09:25", "2026-10-08 15:15", "2026-10-09 14:00", "2026-10-13 16:45",
                "2026-10-09 13:30", "2026-10-13 16:15")
SENSOR_SQL = """SELECT r.epoch, r.pm02, r.atmp, r.rhum,
  (SELECT MIN(MAX(s.available_epoch, v.confirmed_epoch)) FROM sensor_observation_receipts s
   JOIN sensor_observation_visibility v ON v.receipt_id=s.receipt_id WHERE s.observed_epoch=r.epoch)
FROM readings r WHERE r.epoch > ? AND r.epoch <= ? ORDER BY r.epoch"""
SOURCE_SQL = """SELECT w.fetched_epoch, COALESCE(rc.vis, w.fetched_epoch + 5) AS vis, w.payload FROM {table} w
LEFT JOIN (SELECT r.source_fetched_epoch AS f, MIN(MAX(r.available_epoch, v.confirmed_epoch)) AS vis
           FROM source_forecast_receipts r JOIN source_forecast_visibility v ON v.receipt_id=r.receipt_id
           WHERE r.source_table='{table}' GROUP BY r.source_fetched_epoch) rc ON rc.f=w.fetched_epoch
WHERE COALESCE(rc.vis, w.fetched_epoch + 5) <= ? {extra} ORDER BY vis DESC LIMIT 1"""


def _connect(path):
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _blob(value):
    if isinstance(value, (bytes, memoryview)):
        value = bytes(value)
        value = gzip.decompress(value) if value[:2] == b"\x1f\x8b" else value
    return json.loads(value)


def _ref(rows, end, column=1):
    mask = (rows[:, 0] > end - 300) & (rows[:, 0] <= end) & np.isfinite(rows[:, column])
    if not mask.any() or end - rows[mask, 0].max() > 240:
        return np.nan
    return float(np.median(rows[mask, column]))


def _hourly(payload, field, x, amount=False):
    points = sorted((int(p["epoch"]), p.get(field)) for p in payload["hourly"])
    for epoch, value in points:
        if epoch == x:
            return np.nan if value is None else float(value)
    upper = next((i for i, (epoch, _) in enumerate(points) if epoch > x), None)
    if not upper or points[upper][0] - points[upper - 1][0] > 3600:
        return np.nan
    (e0, a), (e1, b) = points[upper - 1], points[upper]
    if amount:
        return np.nan if b is None else float(b)
    return np.nan if a is None or b is None else a + (b - a) * (x - e0) / (e1 - e0)


def expected_features(conn, neighbors, t: int) -> tuple[dict, dict]:
    """Selected feature values and the visibility clock of every record used."""
    rows = np.array(conn.execute(SENSOR_SQL, (t - 4 * 3600, t)).fetchall(), dtype=float)
    visible = np.where(np.isnan(rows[:, 4]), rows[:, 0] + config.LEGACY_SENSOR_VISIBILITY_LAG_SECONDS, rows[:, 4])
    rows, visible = rows[visible <= t], visible[visible <= t]
    out = {"s_freshPm25": _ref(rows, t), "s_freshTemperature": _ref(rows, t, 2), "s_freshHumidity": _ref(rows, t, 3),
           **{f"s_rawLag{k}": _ref(rows, t - 60 * k) for k in (5, 15, 30, 60)},
           "s_freshSampleCount": float((rows[:, 0] > t - 300).sum()),
           "s_rawSampleCount30": float((rows[:, 0] > t - 1800).sum())}
    recent = rows[:, 0] > t - 900
    out["s_rawSd15"] = float(np.std(rows[recent, 1])) if recent.sum() >= 3 else np.nan
    for age in (0, 3, 30, 177):
        mask = (rows[:, 0] > t - (age + 3) * 60) & (rows[:, 0] <= t - age * 60)
        out[f"s_seq_pm25_{age}_{age + 3}min"] = float(np.median(rows[mask, 1])) if mask.any() else np.nan
    clocks = {"sensor": float(visible.max()) if len(visible) else np.nan}
    for table, short, max_age, extra, fields in (
            ("weather_forecast_runs", "w", 7200, "", (("temp_90", "temperature_2m", 5400, False),
                                                      ("rh_0", "relative_humidity_2m", 0, False),
                                                      ("pprob_180", "precipitation_probability", 10800, False),
                                                      ("precip_90", "precipitation", 5400, True))),
            ("air_quality_forecast_runs", "c", 10800, "AND w.model_version IN ('cams_anchor_v1','cams_anchor_v2')",
             (("pm25_90", "pm2_5", 5400, False),))):
        run = conn.execute(SOURCE_SQL.format(table=table, extra=extra), (t,)).fetchone()
        if run is None or t - run[0] > max_age:
            out[f"{short}_age_min"] = np.nan
            continue
        payload = json.loads(run[2])
        out[f"{short}_age_min"] = (t - run[0]) / 60
        for name, field, lead, amount in fields:
            out[f"{short}_{name}"] = _hourly(payload, field, t + lead, amount)
        clocks[table] = run[1]
    run = conn.execute("SELECT fetched_epoch, recorded_epoch, payload FROM regional_pm_runs WHERE recorded_epoch <= ? "
                       "ORDER BY recorded_epoch DESC LIMIT 1", (t,)).fetchone()
    if run and t - run[0] <= 6 * 3600:
        payload = _blob(run[2])
        times = np.asarray(payload["times"], float)
        values = [np.interp(t, times, [np.nan if v is None else v for v in location["hourly"]["pm2_5"]],
                            left=np.nan, right=np.nan) for location in payload["rawResponse"]]
        out["rp_age_min"], out["rp_mean_0"] = (t - run[0]) / 60, float(np.mean(values)) if np.isfinite(values).all() else np.nan
        clocks["regional_pm"] = run[1]
    else:
        out["rp_age_min"] = np.nan
    run = conn.execute("SELECT fetched_epoch FROM regional_wind_runs WHERE fetched_epoch + 5 <= ? "
                       "ORDER BY fetched_epoch DESC LIMIT 1", (t,)).fetchone()
    out["rw_age_min"] = (t - run[0]) / 60 if run and t - run[0] <= 12 * 3600 else np.nan
    if run:
        clocks["regional_wind"] = run[0] + 5
    run = conn.execute("SELECT report_epoch, atmp, first_fetched_epoch FROM subang_observations WHERE first_fetched_epoch <= ? "
                       "ORDER BY report_epoch DESC LIMIT 1", (t,)).fetchone()
    if run and t - run[0] <= 3 * 3600:
        out["o_age_min"], out["o_temp"] = (t - run[0]) / 60, run[1]
        clocks["subang"] = run[2]
    else:
        out["o_age_min"] = np.nan
    stations = neighbors.execute(
        "SELECT o.location_id, MAX(o.bucket_start_epoch + o.bucket_seconds), MAX(MAX(r.received_epoch, r.recorded_epoch)) "
        "FROM observations o JOIN requests r ON r.request_id=o.request_id WHERE o.valid=1 AND o.measure='pm25' "
        "AND MAX(r.received_epoch, r.recorded_epoch) <= ? GROUP BY o.location_id", (t,)).fetchall()
    out["n_count"] = float(sum(1 for _, end, _ in stations if t - end <= config.NEIGHBOR_MAX_AGE_SECONDS))
    if stations:
        clocks["neighbors"] = max(clock for _, _, clock in stations)
    return out, clocks


def leak_audit(n_random=30, seed=20261013) -> pd.DataFrame:
    ds = dataset.load()
    rng = np.random.default_rng(seed)
    issues = [("random", int(i)) for i in np.sort(rng.choice(ds.index.to_numpy(np.int64), n_random, replace=False))]
    issues += [("event", config.myt_epoch(text)) for text in EVENT_ISSUES]
    records = []
    with closing(_connect(config.SNAPSHOT_DB)) as conn, closing(_connect(config.NEIGHBOR_DB)) as neighbors:
        for kind, t in issues:
            expected, clocks = expected_features(conn, neighbors, t)
            wrong = []
            for column, value in expected.items():
                have = float(ds.at[t, column])
                same = (np.isnan(have) and np.isnan(value)) or abs(have - value) <= 1e-6 * max(1.0, abs(value))
                if not same:
                    wrong.append(f"{column}: sql={value} dataset={have}")
            late = {k: round(v - t, 1) for k, v in clocks.items() if np.isfinite(v) and v > t}
            records.append({"kind": kind, "issue_myt": str(config.myt(t))[:16], "values_checked": len(expected),
                            "mismatches": "; ".join(wrong), "records_visible_after_issue": late or "",
                            "newest_input_age_s": {k: round(t - v) for k, v in clocks.items() if np.isfinite(v)}})
    return pd.DataFrame(records)


def legacy_rule_audit(lags=(0, 30, 60, 90)) -> pd.DataFrame:
    """Post-ledger grid issues: does observation + lag admit readings not yet received?"""
    with closing(_connect(config.SNAPSHOT_DB)) as conn:
        rows = np.array(conn.execute(
            "SELECT s.observed_epoch, MIN(MAX(s.available_epoch, v.confirmed_epoch)) FROM sensor_observation_receipts s "
            "JOIN sensor_observation_visibility v ON v.receipt_id=s.receipt_id GROUP BY s.observed_epoch").fetchall(), float)
    observed, received = rows[:, 0], rows[:, 1]
    start = -(-int(observed.min() + 3 * 3600) // config.GRID_SECONDS) * config.GRID_SECONDS
    issues = np.arange(start, int(observed.max()), config.GRID_SECONDS)
    out = []
    for lag in lags:
        leaky = hidden = 0
        for t in issues:
            window = (observed > t - 10800) & (observed <= t)
            rule, real = observed[window] + lag <= t, received[window] <= t
            leaky += bool((rule & ~real).any())
            hidden += bool((~rule & real).any())
        out.append({"lag_s": lag, "grid_issues": len(issues), "share_admitting_unreceived_reading": leaky / len(issues),
                    "share_hiding_received_reading": hidden / len(issues)})
    return pd.DataFrame(out)


def _buckets(conn):
    raw = np.array(conn.execute("SELECT epoch, pm02 FROM readings WHERE pm02 IS NOT NULL AND pm02 >= 0 ORDER BY epoch").fetchall(), float)
    epochs, values = raw[:, 0], raw[:, 1]

    def bucket(label):
        lo, hi = np.searchsorted(epochs, label - 900, "right"), np.searchsorted(epochs, label, "right")
        e = epochs[lo:hi]
        if len(e) < 3 or e[0] - (label - 900) > 240 or label - e[-1] > 240 or np.diff(e).max() > 480:
            return None
        return float(np.median(values[lo:hi]))
    return bucket


def target_audit(n=50, seed=50) -> pd.DataFrame:
    ds = dataset.load()
    with closing(_connect(config.SNAPSHOT_DB)) as conn:
        bucket = _buckets(conn)

    def proxy(t, lead):
        anchor = t // 900 * 900
        lower = anchor + (t + lead - anchor) // 900 * 900
        if t + lead == lower:
            value = bucket(lower)
            return np.nan if value is None else value
        a, b = bucket(lower), bucket(lower + 900)
        fraction = (t + lead - lower) / 900
        return np.nan if a is None or b is None else a * (1 - fraction) + b * fraction

    def ride(t):
        start, end, values, weights = t + 5400, t + 12600, [], []
        for label in range((start // 900 + 1) * 900, -(-end // 900) * 900 + 1, 900):
            overlap = min(end, label) - max(start, label - 900)
            if overlap > 0:
                value = bucket(label)
                if value is None:
                    return np.nan, np.nan, np.nan
                values.append(value)
                weights.append(overlap / 7200)
        return float(np.dot(values, weights)), min(values), max(values)

    def session(t, hour):
        local = config.myt(t)
        for days in range(4):
            day = (local + timedelta(days=days)).date()
            start = int(datetime(day.year, day.month, day.day, hour, tzinfo=config.MYT).timestamp())
            if start >= t + 5400:
                values = [bucket(start + 900 * k) for k in range(1, 9)]
                return np.nan if None in values else float(np.mean(values))

    rng = np.random.default_rng(seed)
    pool = ds.index[ds.y_ref.notna()].to_numpy(np.int64)
    issues = sorted(rng.choice(pool, n - len(EVENT_ISSUES), replace=False).tolist()) + [config.myt_epoch(s) for s in EVENT_ISSUES]
    records = []
    for t in issues:
        row = ds.loc[t]
        ref, arrival = row.y_ref, proxy(t, 5400)
        mean, low, high = ride(t)
        joint = np.isfinite(arrival) and np.isfinite(mean)
        deltas = np.array([proxy(t, 60 * m) for m in (15, 30, 45, 60, 75, 90)]) - ref
        crossed = np.flatnonzero(np.abs(deltas) >= 20)
        first = (float(np.sign(deltas[crossed[0]])) if len(crossed) else 0.0) if np.isfinite(deltas).all() else np.nan
        delta = arrival - ref
        klass = -1 if not np.isfinite(delta) else 0 if delta <= -40 else 1 if delta <= -20 else 2 if delta < 20 else 3 if delta < 40 else 4
        expected = {"y_arrival": arrival, "y_ride_mean": mean if joint else np.nan, "y_ride_min": low if joint else np.nan,
                    "y_ride_max": high if joint else np.nan, "y_ride_joint_valid": float(joint), "y_class5": float(klass),
                    "y_first20": first, "y_morning_mean": session(t, 9), "y_afternoon_mean": session(t, 14),
                    "y_ride_le70": float(mean <= 70) if joint else np.nan}
        wrong = [f"{k}: raw={v} dataset={float(row[k])}" for k, v in expected.items()
                 if not ((np.isnan(v) and np.isnan(float(row[k]))) or abs(v - float(row[k])) < 1e-9)]
        records.append({"issue_myt": str(config.myt(t))[:16], "y_arrival": arrival, "y_ride_mean": expected["y_ride_mean"],
                        "y_first20": first, "mismatches": "; ".join(wrong)})
    return pd.DataFrame(records)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 160)
    leaks = leak_audit()
    print(leaks.to_string())
    print("leak audit: issues with a mismatch", int((leaks.mismatches != "").sum()), "of", len(leaks),
          "| issues using a record visible after the issue", int((leaks.records_visible_after_issue != "").sum()))
    print(legacy_rule_audit().to_string())
    found = target_audit()
    print(found.to_string())
    print("target audit: issues with a mismatch", int((found.mismatches != "").sum()), "of", len(found))
