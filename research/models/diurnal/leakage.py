"""Leakage checks of diurnal_v1 (skeptic pass; never changes the candidate or the harness).

    python -B -m diurnal.leakage rederive [n]       # raw-SQL re-derivation of every diurnal_v1 input
    python -B -m diurnal.leakage weather_clocks     # vintage clocks: payload fetchedEpoch vs table, gaps
    python -B -m diurnal.leakage cutoffs            # walk-forward train clocks vs cutoff - embargo, per group
    python -B -m diurnal.leakage truncate [n]       # issue-level truncation: predictions with every row after t removed
    python -B -m diurnal.leakage delay <scenario>   # final-half rerun with delayed inputs (base, s10, w1, s10w1)
    python -B -m diurnal.leakage score              # tables for the delay scenarios on identical rows

Scenarios (inputs only; targets, y_ref for scoring and the clock t_* are unchanged):
  base    as published (reproduces data/models/diurnal/pred_*.parquet)
  s10     every sensor-derived input (s_*, d_*) taken from the grid row 10 min earlier; Open-Meteo
          vintage = latest visible at t - 10 min
  w1      Open-Meteo vintage one fetch cycle (15 min) older than the latest visible at t; sensor unchanged
  s10w1   s10, and the vintage one more fetch cycle older (latest visible at t-10 min, then one back)
Outputs: data\\models\\diurnal_leak\\.
"""
from __future__ import annotations

import ctypes
import json
import sqlite3
import sys
import time
from contextlib import closing

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, sources, walkforward
from harness import features as HF
from . import evaluate, features as F
from .chosen import CHOSEN, COMPOSE
from .models import GROUPS

OUT = config.OUT / "diurnal_leak"
SRC = config.OUT / "diurnal"
SHIFT_ROWS = 2                                   # 10 min on the 5-min grid
NWP = F.NWP
CAND_GROUPS = ("near", "events", "le70", "morning", "afternoon")


def _low_priority():
    try:                                          # IDLE_PRIORITY_CLASS: research shares the laptop with production
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:
        pass


def _conn():
    return sqlite3.connect(f"file:{config.SNAPSHOT_DB.as_posix()}?mode=ro", uri=True)


# ============================================================================ independent raw re-derivation
class RawSensor:
    """Independent implementation from the raw tables (does not call production or harness feature code)."""

    def __init__(self):
        with closing(_conn()) as c:
            r = np.array(c.execute("SELECT epoch, pm02, atmp, rhum FROM readings ORDER BY epoch").fetchall(), float)
            rec = dict(c.execute(
                "SELECT r.observed_epoch, MIN(MAX(r.available_epoch, v.confirmed_epoch)) "
                "FROM sensor_observation_receipts r JOIN sensor_observation_visibility v USING(receipt_id) "
                "GROUP BY r.observed_epoch").fetchall())
        self.epoch = r[:, 0]
        self.val = r[:, 1:4].copy()
        self.val[self.val[:, 0] < 0, 0] = np.nan
        self.vis = np.array([rec.get(int(e), e + 60.0) for e in self.epoch])

    def window(self, t):
        lo = np.searchsorted(self.epoch, t - 10800, side="right")
        hi = np.searchsorted(self.epoch, t, side="right")
        ok = self.vis[lo:hi] <= t
        return self.epoch[lo:hi][ok], self.val[lo:hi][ok]

    @staticmethod
    def _ref(e, v, end, col=0):
        m = (e > end - 300) & (e <= end) & np.isfinite(v[:, col])
        if not m.any() or end - e[m].max() > 240:
            return np.nan
        return float(np.median(v[m, col]))

    def features(self, t):
        e, v = self.window(t)
        out = {"s_freshPm25": self._ref(e, v, t)}
        for m in (5, 10, 15, 30, 60):
            out[f"s_rawDelta{m}"] = out["s_freshPm25"] - self._ref(e, v, t - 60 * m)
        for m in (15, 30, 60):
            mask = (e > t - 60 * m) & np.isfinite(v[:, 0])
            x, y = (e[mask] - t) / 60, v[mask, 0]
            out[f"s_rawSd{m}"] = float(np.std(y)) if len(y) >= 3 else np.nan
            if m in (15, 30):
                out[f"s_rawRange{m}"] = float(y.max() - y.min()) if len(y) >= 2 else np.nan
                out[f"s_rawSlope{m}PerMinute"] = float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 and np.ptp(x) > 0 else np.nan
        out["s_freshTemperature"] = self._ref(e, v, t, 1)
        out["s_freshHumidity"] = self._ref(e, v, t, 2)
        out["s_rawTemperatureDelta30"] = out["s_freshTemperature"] - self._ref(e, v, t - 1800, 1)
        out["s_rawHumidityDelta30"] = out["s_freshHumidity"] - self._ref(e, v, t - 1800, 2)
        return out

    def ref(self, t):
        e, v = self.window(t)
        return self._ref(e, v, t)


class RawWeather:
    """Independent Open-Meteo vintage reader: latest run visible at t (receipt clock, else fetched + 5 s)."""

    FIELDS = {"blh": "boundary_layer_height", "pprob": "precipitation_probability", "cloud": "cloud_cover",
              "wspd10": "wind_speed_10m", "temp": "temperature_2m", "precip": "precipitation"}

    def __init__(self):
        with closing(_conn()) as c:
            rec = dict(c.execute(
                "SELECT r.source_fetched_epoch, MIN(MAX(r.available_epoch, v.confirmed_epoch)) "
                "FROM source_forecast_receipts r JOIN source_forecast_visibility v USING(receipt_id) "
                "WHERE r.source_table='weather_forecast_runs' GROUP BY r.source_fetched_epoch").fetchall())
            rows = c.execute("SELECT fetched_epoch, payload FROM weather_forecast_runs "
                             "WHERE source='Open-Meteo Best Match' ORDER BY fetched_epoch").fetchall()
        self.runs = []
        for fetched, blob in rows:
            payload = sources._decode(blob)
            hours = sorted((int(p["epoch"]), p) for p in payload["hourly"])
            x = np.array([h for h, _ in hours], float)
            vals = {k: np.array([np.nan if p.get(f) is None else float(p[f]) for _, p in hours]) for k, f in self.FIELDS.items()}
            self.runs.append((int(fetched), float(rec.get(int(fetched), fetched + 5.0)), x, vals))
        self.runs.sort(key=lambda r: (r[1], r[0]))
        self.visible = np.array([r[1] for r in self.runs])

    def pick(self, t, delay=0, back=0):
        k = int(np.searchsorted(self.visible, t - delay, side="right")) - 1 - back
        if k < 0 or t - self.runs[k][0] > 7200:
            return None
        return self.runs[k]

    @staticmethod
    def _interp(x, y, ts):
        out = []
        for t in np.atleast_1d(ts):
            j = np.searchsorted(x, t)
            if j < len(x) and x[j] == t:
                out.append(y[j])
            elif 0 < j < len(x) and x[j] - x[j - 1] <= 3600:
                out.append(y[j - 1] + (y[j] - y[j - 1]) * (t - x[j - 1]) / (x[j] - x[j - 1]))
            else:
                out.append(np.nan)
        return np.array(out)

    @staticmethod
    def _amount(x, y, start, end):
        total, covered = 0.0, 0.0
        for right, value in zip(x, y):
            ov = min(end, right) - max(start, right - 3600)
            if ov > 0:
                if not np.isfinite(value):
                    return np.nan
                total += value * ov / 3600
                covered += ov
        return total if covered >= end - start - 1 else np.nan

    def features(self, t, delay=0, back=0):
        run = self.pick(t, delay, back)
        if run is None:
            return {c: np.nan for c in NWP}
        _, _, x, v = run
        i = lambda k, lead: float(self._interp(x, v[k], t + 60 * lead)[0])
        mids = np.arange(t + 5400 + 450, t + 12600, 900)
        blh = self._interp(x, v["blh"], mids)
        return {"w_blh_0": i("blh", 0), "w_blh_90": i("blh", 90), "w_blh_180": i("blh", 180),
                "w_ride_blh_mean": float(blh.mean()) if np.isfinite(blh).all() else np.nan,
                "w_pprob_90": i("pprob", 90), "w_next3h_precip_sum": self._amount(x, v["precip"], t, t + 10800),
                "w_ride_precip_sum": self._amount(x, v["precip"], t + 5400, t + 12600),
                "w_cloud_90": i("cloud", 90), "w_wspd10_90": i("wspd10", 90),
                "w_temp_0": i("temp", 0), "w_temp_90": i("temp", 90),
                "fetched": run[0], "visible": run[1]}


def raw_analogs(sensor: RawSensor, t, ref_cache):
    """d_* analog/profile features from the independent reference (grid points strictly visible at each point)."""
    def ref(s):
        if s not in ref_cache:
            ref_cache[s] = sensor.ref(s)
        return ref_cache[s]
    D, A, step = 86400, 5400, 300
    lag0 = np.array([ref(t - k * D) for k in range(1, 8)])
    lag90 = np.array([ref(t - k * D + A) for k in range(1, 8)])
    rides = []
    for k in range(1, 8):
        s = t - k * D
        vals = np.array([ref(s + A + j * step) for j in range(25)])
        rides.append(np.nanmean(vals) if np.isfinite(vals).sum() >= 12 else np.nan)
    rides = np.array(rides)
    cur = ref(t)
    last24 = np.array([ref(t - j * step) for j in range(288)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = {"d_an1_delta90": lag90[0] - lag0[0], "d_an7_delta90": np.nanmedian(lag90 - lag0),
                   "d_an7_ride": np.nanmedian(rides - lag0), "d_an7_absdelta90": np.nanmean(np.abs(lag90 - lag0)),
                   "d_ref_lag24": lag0[0], "d_prof7_0": np.nanmedian(lag0), "d_prof7_90": np.nanmedian(lag90)}
            ok = np.isfinite(last24).sum() >= 144
            out.update({"d_prof7_delta90": out["d_prof7_90"] - out["d_prof7_0"], "d_anom7": cur - out["d_prof7_0"],
                        "d_mean24": np.nanmean(last24) if ok else np.nan,
                        "d_min24": np.nanmin(last24) if ok else np.nan, "d_max24": np.nanmax(last24) if ok else np.nan})
    out["d_anom24"] = cur - out["d_mean24"]
    return out


def rederive(n=120, seed=7):
    ds = F.add(dataset.load())
    ev = ds.loc[ds["eval"] & ds.m_fresh_available]
    rng = np.random.default_rng(seed)
    picks = list(rng.choice(ev.index.to_numpy(np.int64), n, replace=False))
    # plus the minutes around every big-event onset (where the candidate's early calls are claimed)
    for text in ("2026-10-01 16:05", "2026-10-08 15:15", "2026-10-09 14:00", "2026-10-13 16:45"):
        onset = config.myt_epoch(text)
        picks += [onset + k * 300 for k in range(-12, 4) if onset + k * 300 in ev.index]
    picks = sorted(set(int(p) for p in picks))
    sensor, weather = RawSensor(), RawWeather()
    cache, rows = {}, []
    for t in picks:
        rec = {"issue": t, **sensor.features(t), **weather.features(t), **raw_analogs(sensor, t, cache)}
        rec["y_ref_raw"] = sensor.ref(t)
        rows.append(rec)
    raw = pd.DataFrame(rows).set_index("issue")
    cols = F.SENSOR + NWP + F.ANALOG
    report_rows = []
    for c in cols + ["y_ref"]:
        a = ds.loc[raw.index, c].to_numpy(float)
        b = raw["y_ref_raw" if c == "y_ref" else c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
        diff = np.abs(a - b)[both]
        report_rows.append({"column": c, "n": int(both.sum()), "nan_mismatch": nan_mismatch,
                            "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
                            "n_diff_gt_1e-6": int((diff > 1e-6).sum())})
    frame = pd.DataFrame(report_rows)
    # vintage clock at each sampled issue
    age = (raw.index.to_numpy(float) - raw.fetched.to_numpy(float)) / 60
    lead = (raw.index.to_numpy(float) - raw.visible.to_numpy(float))
    print(frame.to_string(index=False))
    print(f"issues {len(raw)}; vintage visible before issue by min {np.nanmin(lead):.1f} s; age min {np.nanmin(age):.1f} "
          f"median {np.nanmedian(age):.1f} max {np.nanmax(age):.1f} min")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_json(OUT / "rederive.json", orient="records", indent=1)
    raw.to_parquet(OUT / "rederive_raw.parquet")
    return frame


def weather_clocks():
    with closing(_conn()) as c:
        rows = c.execute("SELECT fetched_epoch, fetched_timestamp, payload FROM weather_forecast_runs "
                         "WHERE source='Open-Meteo Best Match' ORDER BY fetched_epoch").fetchall()
        rec = dict(c.execute(
            "SELECT r.source_fetched_epoch, MIN(MAX(r.available_epoch, v.confirmed_epoch)) "
            "FROM source_forecast_receipts r JOIN source_forecast_visibility v USING(receipt_id) "
            "WHERE r.source_table='weather_forecast_runs' GROUP BY r.source_fetched_epoch").fetchall())
    mismatch, recv_lag, first_hour, receipt_lag = 0, [], [], []
    for fetched, stamp, blob in rows:
        p = sources._decode(blob)
        mismatch += int(int(p.get("fetchedEpoch", -1)) != fetched)
        if p.get("receivedEpoch") is not None:
            recv_lag.append(fetched - float(p["receivedEpoch"]))
        hours = [int(h["epoch"]) for h in p["hourly"]]
        first_hour.append((min(hours) - fetched) / 3600)
        if fetched in rec:
            receipt_lag.append(rec[fetched] - fetched)
    gaps = np.diff([r[0] for r in rows])
    out = {"runs": len(rows), "payload_fetchedEpoch_mismatch": mismatch,
           "fetched_minus_received_s": np.percentile(recv_lag, [0, 50, 100]).round(3).tolist() if recv_lag else None,
           "runs_with_receipt": len(receipt_lag),
           "receipt_minus_fetched_s": np.percentile(receipt_lag, [0, 50, 99, 100]).round(2).tolist() if receipt_lag else None,
           "first_hour_minus_fetched_h": np.percentile(first_hour, [0, 50, 100]).round(2).tolist(),
           "fetch_gap_s_p5_p50_p95": np.percentile(gaps, [5, 50, 95]).tolist()}
    print(json.dumps(out, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weather_clocks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ============================================================================ walk-forward clocks
def cutoffs():
    ds = F.add(dataset.load())
    days = [d for d in walkforward.eval_days(ds) if evaluate.TUNE[0] <= d < evaluate.FINAL[1]]
    result = {}
    for group, source in COMPOSE.items():
        _, lookback = CHOSEN[source][group]
        spec = GROUPS[group]
        complete_cols = sorted({walkforward.COMPLETE_COLUMNS[t] for t in spec["targets"]})
        worst = {"max_train_issue_minus_cutoff_min": -np.inf, "max_label_complete_minus_cutoff_s": -np.inf,
                 "y_cols_in_test": 0, "min_test_minus_cutoff_s": np.inf, "days": 0,
                 "max_target_label_complete_minus_cutoff_s": -np.inf}
        target_cols = {"near": ["y_arrival_complete", "y_ride_complete"],
                       "events": ["y_arrival_complete", "y_first20_complete"],
                       "le70": ["y_ride_complete"], "morning": ["y_morning_complete"],
                       "afternoon": ["y_afternoon_complete"]}[group]

        def check(train, test, cutoff):
            idx = train.index.to_numpy(np.int64)
            worst["days"] += 1
            worst["max_train_issue_minus_cutoff_min"] = max(worst["max_train_issue_minus_cutoff_min"], (idx.max() - cutoff) / 60)
            for c in complete_cols:
                worst["max_label_complete_minus_cutoff_s"] = max(worst["max_label_complete_minus_cutoff_s"],
                                                                 float(train[c].max() - cutoff))
            for c in target_cols:
                worst["max_target_label_complete_minus_cutoff_s"] = max(
                    worst["max_target_label_complete_minus_cutoff_s"], float(train[c].max() - cutoff))
            worst["y_cols_in_test"] += sum(c.startswith("y_") for c in test.columns)
            worst["min_test_minus_cutoff_s"] = min(worst["min_test_minus_cutoff_s"], float(test.index.min() - cutoff))
            worst["min_train_issue_minus_cutoff_days"] = min(worst.get("min_train_issue_minus_cutoff_days", np.inf),
                                                              (idx.min() - cutoff) / 86400)
            return None

        walkforward.walk_forward(ds, check, lookback_days=lookback, days=days, test_mask=ds.m_fresh_available,
                                 log=lambda s: None, **spec)
        worst.update({"embargo_min": spec["embargo_minutes"], "lookback_days": lookback,
                      "complete_columns": complete_cols})
        result[group] = worst
        print(group, worst, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cutoffs.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")


# ============================================================================ issue-level truncation
def _saved():
    return pd.concat([pd.read_parquet(SRC / f"pred_{src}__{g}.parquet") for g, src in COMPOSE.items()], axis=1)


def truncate(n=34, seed=11):
    """For sampled final-half issues t: drop every grid row after t, rebuild the d_* features, refit each
    diurnal_v1 group for t's day and predict t alone. Any difference from the saved prediction = future use."""
    base = dataset.load()
    saved = _saved()
    final_idx = base.index[(base.index >= evaluate.FINAL[0]) & base.m_fresh_available & base["eval"]].to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    picks = set(rng.choice(final_idx, n, replace=False).tolist())
    picks |= {config.myt_epoch(s) for s in ("2026-10-08 14:50", "2026-10-13 16:30", "2026-10-09 13:00", "2026-10-01 16:10")}
    rows = []
    for t in sorted(picks):
        started = time.perf_counter()
        ds = F.add(base.loc[base.index <= t])
        day = config.midnight(t)
        mask = pd.Series(ds.index == t, index=ds.index)
        preds = []
        for group, source in COMPOSE.items():
            factory, lookback = CHOSEN[source][group]
            preds.append(walkforward.walk_forward(ds, factory().fit_predict(group), lookback_days=lookback, days=[day],
                                                  test_mask=mask, log=lambda s: None, **GROUPS[group]))
        pred = pd.concat(preds, axis=1)
        a = pred.loc[t].astype(float)
        b = saved.loc[t, a.index].astype(float)
        diff = (a - b).abs()
        same_nan = a.isna() == b.isna()
        rows.append({"issue": config.myt(t).strftime("%m-%d %H:%M"), "max_abs_diff": float(diff.max(skipna=True)),
                     "nan_mismatch": int((~same_nan).sum()), "seconds": round(time.perf_counter() - started, 1)})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_json(OUT / "truncate.json", orient="records", indent=1)
    print(frame.to_string(index=False))


# ============================================================================ delayed inputs
def _weather_columns(ds, delay, back):
    """The 11 NWP inputs (harness interpolation) from an older vintage."""
    runs = [HF._Run(f, v, p, HF.WEATHER_FIELDS, HF.WEATHER_DIRECTIONS)
            for f, v, _, p in sources.forecast_runs("weather_forecast_runs")]
    visible = np.array([r.visible for r in runs])
    records = []
    for issue in ds.index.to_numpy(np.int64):
        k = HF._latest(visible, issue - delay) - back
        rec = {}
        if k >= 0 and issue - runs[k].fetched <= config.WEATHER_MAX_AGE_SECONDS:
            run = runs[k]
            for lead in (0, 90, 180):
                rec[f"w_blh_{lead}"] = run.at("boundary_layer_height", issue + lead * 60)[0]
            rec["w_ride_blh_mean"] = run.window_mean("boundary_layer_height", issue + 5400, issue + 12600)
            rec["w_pprob_90"] = run.at("precipitation_probability", issue + 5400)[0]
            rec["w_next3h_precip_sum"] = run.window_sum_amount("precipitation", issue, issue + 10800)
            rec["w_ride_precip_sum"] = run.window_sum_amount("precipitation", issue + 5400, issue + 12600)
            rec["w_cloud_90"] = run.at("cloud_cover", issue + 5400)[0]
            rec["w_wspd10_90"] = run.at("wind_speed_10m", issue + 5400)[0]
            rec["w_temp_0"] = run.at("temperature_2m", issue)[0]
            rec["w_temp_90"] = run.at("temperature_2m", issue + 5400)[0]
            rec["w_vintage_fetched"] = run.fetched
        records.append(rec)
    return pd.DataFrame(records, index=ds.index).reindex(columns=NWP + ["w_vintage_fetched"])


def delayed_frame(scenario):
    ds = F.add(dataset.load())
    if scenario == "base":
        return ds
    assert (np.diff(ds.index.to_numpy(np.int64)) == config.GRID_SECONDS).all(), "grid must be contiguous"
    out = ds.copy()
    if scenario in ("s10", "s10w1"):
        sensor_cols = [c for c in ds.columns if c.startswith("s_")]
        derived = [c for c in F.ANALOG] + [c for c in ds.columns if c.startswith("d_") and c.endswith(("_last", "_last_minus_ref"))]
        ref = ds.s_freshPm25.where(ds.m_fresh_available)
        out[sensor_cols] = ds[sensor_cols].shift(SHIFT_ROWS)
        out["s_freshPm25"] = ref.shift(SHIFT_ROWS)
        out[derived] = ds[derived].shift(SHIFT_ROWS)
    delay = 600 if scenario in ("s10", "s10w1") else 0
    back = 1 if scenario in ("w1", "s10w1") else 0
    w = _weather_columns(ds, delay, back)
    out[NWP] = w[NWP]
    out["d_w_blh_change90"] = out.w_blh_90 - out.w_blh_0
    out["d_w_temp_change90"] = out.w_temp_90 - out.w_temp_0
    age = (out.index.to_numpy(float) - w.w_vintage_fetched.to_numpy(float)) / 60
    print(f"{scenario}: vintage age min median p95 = {np.nanmin(age):.1f} {np.nanmedian(age):.1f} "
          f"{np.nanpercentile(age, 95):.1f} min; base age median {np.nanmedian(ds.w_age_min):.1f}", flush=True)
    return out


def delay(scenario):
    _low_priority()
    ds = delayed_frame(scenario)
    days = [d for d in walkforward.eval_days(ds) if evaluate.FINAL[0] <= d < evaluate.FINAL[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    preds = []
    for group, source in COMPOSE.items():
        factory, lookback = CHOSEN[source][group]
        started = time.perf_counter()
        pred = walkforward.walk_forward(ds, factory().fit_predict(group), lookback_days=lookback, days=days,
                                        test_mask=ds.m_fresh_available, log=lambda s: None, **GROUPS[group])
        preds.append(pred)
        print(f"{scenario} {group} {time.perf_counter() - started:.0f}s", flush=True)
    pd.concat(preds, axis=1).to_parquet(OUT / f"pred_{scenario}.parquet")


def score():
    ds = dataset.load()
    replay = report.standard_replay(ds)
    saved = _saved()
    final = (ds.index >= evaluate.FINAL[0]) & (ds.index < evaluate.FINAL[1])
    saved = saved.loc[saved.index.isin(ds.index[final])]
    preds = {"published": saved}
    for name in ("base", "s10", "w1", "s10w1"):
        path = OUT / f"pred_{name}.parquet"
        if path.exists():
            preds[name] = pd.read_parquet(path)
    # identical rows: a column is kept only where every scenario has it
    cols = [c for c in saved.columns]
    common = pd.concat([p.reindex(saved.index)[cols].notna() for p in preds.values()]).groupby(level=0).all()
    aligned = {k: p.reindex(saved.index)[cols].where(common) for k, p in preds.items()}
    # delayed persistence: the reference 10 min earlier (same rows)
    ref10 = ds.y_ref.shift(SHIFT_ROWS)
    pers10 = pd.DataFrame({c: ref10 for c in ("pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max")},
                          index=ds.index).reindex(saved.index).where(common[["pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max"]])
    out = {}
    for name, pred in list(aligned.items()) + [("persistence_ref_t-10", pers10)]:
        rows = evaluate.full_table(ds, pred, evaluate.FINAL, replay)
        out[name] = rows
        if name != "persistence_ref_t-10":
            out[name + "/directions"] = evaluate.direction_table(ds, pred, evaluate.FINAL, replay)
            out[name + "/cases"] = report.case_table(ds, pred).astype(str).to_dict(orient="records")
        print(f"\n### {name}\n")
        print(evaluate.markdown(rows))
    diffs = {k: float((aligned[k] - aligned["published"]).abs().max().max()) for k in aligned}
    print("\nmax |scenario - published| over all heads:", diffs)
    out["max_abs_diff_vs_published"] = diffs
    (OUT / "score.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    _low_priority()
    command = sys.argv[1]
    if command == "rederive":
        rederive(*(int(a) for a in sys.argv[2:3]))
    elif command == "weather_clocks":
        weather_clocks()
    elif command == "cutoffs":
        cutoffs()
    elif command == "truncate":
        truncate(*(int(a) for a in sys.argv[2:3]))
    elif command == "delay":
        delay(sys.argv[2])
    elif command == "score":
        score()
