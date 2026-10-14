"""Session-model inputs: causal history features and the stacked (issue x session) frame.

Everything here is computed from values visible at each issue clock:

* ``s_freshPm25`` on the 5-min grid is the fresh reference at that grid issue,
  so rolling means over grid issues <= t are visible at t.
* A completed session mean (production ``_session_actual``: 8 closed 15-min
  bucket medians) counts as visible from its window end + ``SESSION_VISIBLE_LAG``
  (15 min; measured receipt latency max is 459 s). Canonical clocks are >= 1 h
  after any session end, so the lag never binds there.
* Weather (w_) and CAMS (c_) session aggregates are the harness columns, already
  restricted to the latest vintage visible at the issue.

The session start/lead for each issue is recomputed from the issue clock
(production ``next_feasible_ride_period``), never read from y_ columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import config, targets

SESSIONS = ("morning", "afternoon")
SESSION_VISIBLE_LAG = 900
HISTORY_SESSIONS = 7

WEATHER_SESSION = ("temp", "rh", "pprob", "wspd10", "wind10_u", "wind10_v", "wind925_u", "wind925_v",
                   "blh", "cloud", "cape", "precip_sum")
WEATHER_NOW = ("w_temp_0", "w_rh_0", "w_blh_0", "w_wspd10_0", "w_prior3h_precip_sum", "w_next6h_pprob_max")
SENSOR_NOW = ("s_freshPm25", "s_rawDelta30", "s_rawDelta60", "s_rawSd60", "s_freshTemperature", "s_freshHumidity")


def session_table(ds: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row per completed session: start, end, visible clock, actual mean (label history)."""
    table = (ds[[f"y_{name}_start", f"y_{name}_mean"]].dropna()
             .drop_duplicates(f"y_{name}_start").rename(columns={f"y_{name}_start": "start", f"y_{name}_mean": "mean"}))
    table["start"] = table.start.astype(np.int64)
    table["end"] = table.start + 7200
    table["visible"] = table.end + SESSION_VISIBLE_LAG
    return table.sort_values("start").reset_index(drop=True)


def _session_history(issues: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    visible = table.visible.to_numpy(np.int64)
    means = table["mean"].to_numpy(float)
    count = np.searchsorted(visible, issues, side="right")          # sessions visible at issue
    last = np.full(len(issues), np.nan)
    mean7 = np.full(len(issues), np.nan)
    med7 = np.full(len(issues), np.nan)
    age = np.full(len(issues), np.nan)
    ends = table.end.to_numpy(np.int64)
    for i, k in enumerate(count):
        if k == 0:
            continue
        window = means[max(0, k - HISTORY_SESSIONS):k]
        last[i] = means[k - 1]
        mean7[i] = window.mean()
        med7[i] = np.median(window)
        age[i] = (issues[i] - ends[k - 1]) / 3600
    return pd.DataFrame({"last": last, "mean7": mean7, "med7": med7, "age_h": age})


def history_features(ds: pd.DataFrame) -> pd.DataFrame:
    """Per-issue causal history features (no y_ column of the issue itself is used)."""
    index = ds.index.to_numpy(np.int64)
    ref = pd.Series(ds.s_freshPm25.to_numpy(float), index=pd.to_datetime(index, unit="s"))
    out = pd.DataFrame(index=ds.index)
    for hours in (3, 6, 24):
        out[f"h_ref_mean_{hours}h"] = ref.rolling(f"{hours}h", min_periods=6).mean().to_numpy()
    out["h_ref_min_24h"] = ref.rolling("24h", min_periods=6).min().to_numpy()
    out["h_ref_max_24h"] = ref.rolling("24h", min_periods=6).max().to_numpy()
    # Same clock on each of the previous 7 days (diurnal climatology of the issue clock).
    lookup = pd.Series(ds.s_freshPm25.to_numpy(float), index=index)
    lags = np.column_stack([lookup.reindex(index - k * 86400).to_numpy() for k in range(1, 8)])
    enough = np.isfinite(lags).sum(1) >= 3
    out["h_ref_sameclock_7d"] = np.nan
    out.loc[enough, "h_ref_sameclock_7d"] = np.nanmean(lags[enough], axis=1)
    out["h_ref_sameclock_1d"] = lags[:, 0]
    for name in SESSIONS:
        history = _session_history(index, session_table(ds, name))
        for column in history:
            out[f"h_{name}_{column}"] = history[column].to_numpy()
    return out


def stacked(ds: pd.DataFrame, history: pd.DataFrame, issues=None) -> pd.DataFrame:
    """Long frame: one row per (issue, session). Columns: x_* inputs, plus y/complete when present."""
    frame = ds if issues is None else ds.loc[issues]
    hist = history.loc[frame.index]
    index = frame.index.to_numpy(np.int64)
    parts = []
    for code, name in enumerate(SESSIONS):
        other = SESSIONS[1 - code]
        starts = np.array([targets.next_session(int(i), name)[0] for i in index], dtype=np.int64)
        part = pd.DataFrame(index=pd.MultiIndex.from_arrays([index, np.full(len(index), name)],
                                                            names=["issue_epoch", "session"]))
        part["x_session"] = float(code)
        part["x_lead_min"] = (starts - index) / 60
        part["x_same_day"] = np.array([config.myt(s).date() == config.myt(int(i)).date()
                                       for s, i in zip(starts, index)], dtype=float)
        part["x_issue_hour"] = frame.t_hour.to_numpy(float)              # fractional MYT hour
        for column in SENSOR_NOW:
            part["x_" + column[2:]] = frame[column].to_numpy(float)
        for column in ("h_ref_mean_3h", "h_ref_mean_6h", "h_ref_mean_24h", "h_ref_min_24h", "h_ref_max_24h",
                       "h_ref_sameclock_7d", "h_ref_sameclock_1d"):
            part["x_" + column[2:]] = hist[column].to_numpy(float)
        for column in ("last", "mean7", "med7", "age_h"):
            part[f"x_sess_{column}"] = hist[f"h_{name}_{column}"].to_numpy(float)
            part[f"x_other_{column}"] = hist[f"h_{other}_{column}"].to_numpy(float)
        ref = part.x_freshPm25.to_numpy(float)
        with np.errstate(all="ignore"):
            ratio = (part.x_sess_mean7 + 5) / (part.x_ref_sameclock_7d + 5)
            part["x_diurnal_ratio"] = ratio.to_numpy()
            part["x_anchor_diurnal"] = ((ref + 5) * ratio - 5).to_numpy()
        # weather for the target session (latest visible vintage)
        for short in WEATHER_SESSION:
            column = f"w_{name}_{short}" if short == "precip_sum" else f"w_{name}_{short}_mean"
            part[f"x_w_sess_{short}"] = frame[column].to_numpy(float)
        for column in WEATHER_NOW:
            part["x_" + column] = frame[column].to_numpy(float)
        part["x_w_blh_ratio"] = (part.x_w_sess_blh / frame.w_blh_0.clip(lower=50)).to_numpy(float)
        part["x_w_temp_change"] = (part.x_w_sess_temp - frame.w_temp_0).to_numpy(float)
        # CAMS for the target session
        part["x_c_sess_pm25"] = frame[f"c_pm25_{name}_mean"].to_numpy(float)
        part["x_c_pm25_0"] = frame.c_pm25_0.to_numpy(float)
        part["x_c_aod_0"] = frame.c_aod_0.to_numpy(float)
        with np.errstate(all="ignore"):
            part["x_c_log_ratio"] = np.log((part.x_c_sess_pm25 + 5) / (part.x_c_pm25_0 + 5)).to_numpy(float)
        if f"y_{name}_mean" in frame:
            part["y"] = frame[f"y_{name}_mean"].to_numpy(float)
            part["complete"] = frame[f"y_{name}_complete"].to_numpy(np.int64)
            assert (frame[f"y_{name}_start"].to_numpy(np.int64) == starts).all(), "session start mismatch"
        part["start"] = starts
        parts.append(part)
    return pd.concat(parts).sort_index()


GROUPS = {
    "base": ["x_session", "x_lead_min", "x_same_day", "x_issue_hour", "x_freshPm25", "x_rawDelta30", "x_rawDelta60",
             "x_rawSd60", "x_freshTemperature", "x_freshHumidity", "x_ref_mean_3h", "x_ref_mean_6h", "x_ref_mean_24h",
             "x_ref_min_24h", "x_ref_max_24h", "x_ref_sameclock_7d", "x_ref_sameclock_1d", "x_sess_last",
             "x_sess_mean7", "x_sess_med7", "x_sess_age_h", "x_other_last", "x_other_mean7", "x_diurnal_ratio",
             "x_anchor_diurnal"],
    "weather": [f"x_w_sess_{s}" for s in WEATHER_SESSION] + ["x_" + c for c in WEATHER_NOW]
               + ["x_w_blh_ratio", "x_w_temp_change"],
    "cams": ["x_c_sess_pm25", "x_c_pm25_0", "x_c_aod_0", "x_c_log_ratio"],
}


def columns(groups) -> list[str]:
    return [c for g in groups for c in GROUPS[g]]
