"""Generic two-hour-window rows: the training set of the window-augmented candidate.

Only two session labels exist per day (Morning 09-11, Afternoon 14-16), so 14
days give 28 distinct targets. Production's Morning HGB instead pools 2-h
windows starting at :00/:30 origins x 7 leads. Here every training issue (each
full hour) is paired with every 2-h window starting on a full MYT hour with
lead 90 min .. 24 h, labelled with the production session-mean definition
(mean of the 8 closed 15-min bucket medians, ``targets.session_mean``). The
model learns the diurnal shape from the target hour and the same-hour history,
and is queried only at the real Morning/Afternoon windows.

Inputs per (issue, window), all visible at the issue:
* sensor and grid-history features of the issue (features.history_features);
* same-start-hour window history: the most recent completed window at that
  hour and the mean of up to 7, each admitted only from window end + 15 min;
* weather means/sums over the window from the latest Open-Meteo vintage visible
  at the issue (age <= 2 h) and CAMS PM2.5 over the window (age <= 3 h), using
  the harness ``_Run`` interpolation, so session rows equal the harness
  w_morning_*/w_afternoon_* and c_pm25_*_mean columns exactly (checked in build).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from harness import config, sources, targets
from harness import features as hf
from . import features

TRAIN_ISSUE_STEP = 3600
MIN_LEAD, MAX_LEAD = 5400, 86400
VISIBLE_LAG = features.SESSION_VISIBLE_LAG
WEATHER_MEANS = (("temperature_2m", "temp"), ("relative_humidity_2m", "rh"), ("precipitation_probability", "pprob"),
                 ("wind_speed_10m", "wspd10"), ("wind10_u", "wind10_u"), ("wind10_v", "wind10_v"),
                 ("wind925_u", "wind925_u"), ("wind925_v", "wind925_v"), ("boundary_layer_height", "blh"),
                 ("cloud_cover", "cloud"), ("cape", "cape"))
ISSUE_COLUMNS = {"s_freshPm25": "x_freshPm25", "s_rawDelta30": "x_rawDelta30", "s_rawDelta60": "x_rawDelta60",
                 "s_rawSd60": "x_rawSd60", "s_freshTemperature": "x_freshTemperature",
                 "s_freshHumidity": "x_freshHumidity", "t_hour": "x_issue_hour",
                 **{c: "x_" + c for c in features.WEATHER_NOW}, "c_pm25_0": "x_c_pm25_0", "c_aod_0": "x_c_aod_0"}

COLUMNS = {
    "base": ["x_lead_min", "x_same_day", "x_issue_hour", "x_start_hour", "x_freshPm25", "x_rawDelta30",
             "x_rawDelta60", "x_rawSd60", "x_freshTemperature", "x_freshHumidity", "x_ref_mean_3h", "x_ref_mean_6h",
             "x_ref_mean_24h", "x_ref_min_24h", "x_ref_max_24h", "x_ref_sameclock_7d", "x_ref_sameclock_1d",
             "x_sess_last", "x_sess_mean7", "x_sess_age_h", "x_diurnal_ratio", "x_anchor_diurnal"],
    "weather": [f"x_w_sess_{s}" for _, s in WEATHER_MEANS] + ["x_w_sess_precip_sum"]
               + ["x_" + c for c in features.WEATHER_NOW] + ["x_w_blh_ratio", "x_w_temp_change"],
    "cams": ["x_c_sess_pm25", "x_c_pm25_0", "x_c_aod_0", "x_c_log_ratio"],
}


def columns(groups) -> list[str]:
    return [c for g in groups for c in COLUMNS[g]]


def _hour_starts(issue):
    first = -(-(issue + MIN_LEAD) // 3600) * 3600
    return np.arange(first, issue + MAX_LEAD, 3600, dtype=np.int64)


class Sources:
    def __init__(self):
        self.buckets = targets.bucket_map(targets.bucket_series(sources.plain_rows()))
        self.weather = [hf._Run(f, v, p, hf.WEATHER_FIELDS, hf.WEATHER_DIRECTIONS)
                        for f, v, _, p in sources.forecast_runs("weather_forecast_runs")]
        self.weather_visible = np.array([r.visible for r in self.weather])
        self.cams = [hf._Run(f, v, p, hf.CAMS_FIELDS) for f, v, _, p in sources.forecast_runs("air_quality_forecast_runs")]
        self.cams_visible = np.array([r.visible for r in self.cams])
        self._labels = {}

    def label(self, start):
        if start not in self._labels:
            self._labels[start] = targets.session_mean(self.buckets, int(start))
        return self._labels[start]

    def run(self, kind, issue):
        runs, visible, max_age = ((self.weather, self.weather_visible, config.WEATHER_MAX_AGE_SECONDS) if kind == "w"
                                  else (self.cams, self.cams_visible, config.CAMS_MAX_AGE_SECONDS))
        index = hf._latest(visible, issue)
        return runs[index] if index >= 0 and issue - runs[index].fetched <= max_age else None


def _same_hour_history(src, issue, start):
    """Most recent completed window at the same start hour, and the mean of the 7 most recent with a
    label (visible at issue: window end + 15 min). Same rule as features.session_table for sessions."""
    k_min = max(1, int(np.ceil((start + 7200 + VISIBLE_LAG - issue) / 86400)))
    values, last, age = [], np.nan, np.nan
    for k in range(k_min, k_min + 60):
        if len(values) == features.HISTORY_SESSIONS or start - k * 86400 < config.GRID_START_EPOCH:
            break
        value = src.label(start - k * 86400)
        if np.isfinite(value):
            if not values:
                last, age = value, (issue - (start - k * 86400 + 7200)) / 3600
            values.append(value)
    return last, (np.mean(values) if values else np.nan), age


def rows_for(ds, history, src, pairs):
    """Feature (and label) rows for (issue, window start[, session name]) pairs."""
    records = []
    for issue, start, name in pairs:
        issue, start = int(issue), int(start)
        local_start = config.myt(start)
        record = {"issue_epoch": issue, "session": name, "start": start, "complete": start + 7200,
                  "y": src.label(start), "x_lead_min": (start - issue) / 60,
                  "x_same_day": float(local_start.date() == config.myt(issue).date()),
                  "x_start_hour": local_start.hour + local_start.minute / 60}
        record["x_sess_last"], record["x_sess_mean7"], record["x_sess_age_h"] = _same_hour_history(src, issue, start)
        run = src.run("w", issue)
        if run is not None:
            for field, short in WEATHER_MEANS:
                record[f"x_w_sess_{short}"] = run.window_mean(field, start, start + 7200)
            record["x_w_sess_precip_sum"] = run.window_sum_amount("precipitation", start, start + 7200)
        cams = src.run("c", issue)
        if cams is not None:
            record["x_c_sess_pm25"] = cams.window_mean("pm2_5", start, start + 7200)
        records.append(record)
    out = pd.DataFrame(records)
    issues = out.issue_epoch.to_numpy(np.int64)
    for source, target in ISSUE_COLUMNS.items():
        out[target] = ds[source].reindex(issues).to_numpy(float)
    for column in ("h_ref_mean_3h", "h_ref_mean_6h", "h_ref_mean_24h", "h_ref_min_24h", "h_ref_max_24h",
                   "h_ref_sameclock_7d", "h_ref_sameclock_1d"):
        out["x_" + column[2:]] = history[column].reindex(issues).to_numpy(float)
    for column in [c for g in COLUMNS.values() for c in g]:
        if column not in out:
            out[column] = np.nan
    ratio = (out.x_sess_mean7 + 5) / (out.x_ref_sameclock_7d + 5)
    out["x_diurnal_ratio"] = ratio
    out["x_anchor_diurnal"] = (out.x_freshPm25 + 5) * ratio - 5
    out["x_w_blh_ratio"] = out.x_w_sess_blh / ds.w_blh_0.reindex(issues).clip(lower=50).to_numpy(float)
    out["x_w_temp_change"] = out.x_w_sess_temp - ds.w_temp_0.reindex(issues).to_numpy(float)
    with np.errstate(all="ignore"):
        out["x_c_log_ratio"] = np.log((out.x_c_sess_pm25 + 5) / (out.x_c_pm25_0 + 5))
    return out.set_index(["issue_epoch", "session"])


def build(ds, log=print):
    """Training rows (hourly issues x hourly windows) and query rows (every grid issue x next sessions)."""
    started = time.time()
    history = features.history_features(ds)
    src = Sources()
    index = ds.index.to_numpy(np.int64)
    train_issues = index[index % TRAIN_ISSUE_STEP == 0]
    pairs = [(i, s, "window") for i in train_issues for s in _hour_starts(int(i))]
    train = rows_for(ds, history, src, pairs)
    log(f"window training rows {train.shape} {time.time() - started:.0f}s")
    pairs = [(i, targets.next_session(int(i), name)[0], name) for i in index for name in features.SESSIONS]
    query = rows_for(ds, history, src, pairs)
    log(f"session query rows {query.shape} {time.time() - started:.0f}s")
    return train, query


def check_query(query, stack) -> dict:
    """Session query rows must equal the stacked (harness-column) frame where both are defined."""
    out = {}
    pairs = {"x_w_sess_temp": "x_w_sess_temp", "x_w_sess_blh": "x_w_sess_blh", "x_w_sess_precip_sum": "x_w_sess_precip_sum",
             "x_c_sess_pm25": "x_c_sess_pm25", "x_sess_last": "x_sess_last", "x_sess_mean7": "x_sess_mean7",
             "y": "y", "start": "start"}
    joined = query.join(stack[list(set(pairs.values()))], rsuffix="_stack", how="inner")
    for a, b in pairs.items():
        x, y = joined[a].to_numpy(float), joined[b + "_stack"].to_numpy(float)
        both = np.isfinite(x) & np.isfinite(y)
        out[a] = {"n": int(both.sum()), "max_abs_diff": float(np.max(np.abs(x[both] - y[both]))) if both.any() else None,
                  "nan_mismatch": int(np.sum(np.isfinite(x) != np.isfinite(y)))}
    return out


def load(ds=None, stack=None, log=print):
    """Cached (train, query) frames under data/models/sessions/."""
    from . import run
    train_path, query_path = run.OUT / "window_train.parquet", run.OUT / "window_query.parquet"
    if train_path.exists() and query_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(query_path)
    from harness import dataset
    ds = dataset.load() if ds is None else ds
    stack = run.load_stack(ds) if stack is None else stack
    train, query = build(ds, log)
    check = check_query(query, stack)
    log(f"query vs harness columns: {check}")
    bad = [k for k, v in check.items() if v["nan_mismatch"] or (v["max_abs_diff"] or 0) > 1e-9]
    if bad:
        raise AssertionError(f"session query rows differ from the harness columns: {bad}")
    train.to_parquet(train_path)
    query.to_parquet(query_path)
    return train, query


if __name__ == "__main__":
    load()
