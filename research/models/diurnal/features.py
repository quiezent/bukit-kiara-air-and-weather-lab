"""Diurnal features derived from the issue grid itself (no new sources).

Every value uses only the fresh reference ``y_ref`` of issues strictly before the
issue being described (or the issue itself for level summaries). ``y_ref`` at an
issue s was computed from readings visible at s, so it is visible at any t > s.

* Analogs: what happened on earlier days from this clock of day: the change from
  t-k*24h to t-k*24h+90 min and the mean over t-k*24h+90..+210 min (k = 1..7).
  The latest instant used is t-24h+210 min = t-20.5 h.
* Profile: median reference at this clock over the last 7 days and the anomaly
  of the current reference against it.
* Sessions: mean reference over the most recent Morning (09-11) / Afternoon
  (14-16) window that had closed at t, and the lead to the next feasible
  session (computed from the clock, as production's next_feasible_ride_period).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from harness import config

STEP = config.GRID_SECONDS                 # 300 s
DAY = 86400 // STEP                        # 288 grid steps
A90 = config.ARRIVAL_SECONDS // STEP       # 18
R210 = config.RIDE_END_SECONDS // STEP     # 42
ANALOG_DAYS = 7


def _grid_ref(ds: pd.DataFrame) -> pd.Series:
    index = ds.index.to_numpy(np.int64)
    full = np.arange(index.min(), index.max() + STEP, STEP, dtype=np.int64)
    return ds["y_ref"].reindex(full)


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    """Value at t - steps*STEP (steps > 0 looks back)."""
    assert steps > 0
    out = np.full_like(values, np.nan)
    out[steps:] = values[:-steps]
    return out


def next_session_start(issue: int, hours: tuple[int, int]) -> int:
    """First session start >= issue + 90 min (production next_feasible_ride_period)."""
    local = config.myt(issue)
    start = local.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    while start.timestamp() < issue + config.ARRIVAL_SECONDS:
        start = start + pd.Timedelta(days=1)
    return int(start.timestamp())


def build(ds: pd.DataFrame) -> pd.DataFrame:
    ref_series = _grid_ref(ds)
    ref = ref_series.to_numpy(float)
    full_index = ref_series.index.to_numpy(np.int64)
    out = pd.DataFrame(index=full_index)

    # ride-window mean of the reference starting at s+90 (uses s+90..s+210)
    ride_fwd = pd.Series(ref).rolling(R210 - A90 + 1, min_periods=12).mean().shift(-(R210)).to_numpy()
    lag0, lag90, lagride = [], [], []
    for k in range(1, ANALOG_DAYS + 1):
        lag0.append(_shift(ref, k * DAY))
        lag90.append(_shift(ref, k * DAY - A90))          # t - k*24h + 90 min
        lagride.append(_shift(ride_fwd, k * DAY))         # mean over t-k*24h+90..+210
    lag0, lag90, lagride = np.array(lag0), np.array(lag90), np.array(lagride)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN slices -> NaN
        out["d_an1_delta90"] = lag90[0] - lag0[0]
        out["d_an7_delta90"] = np.nanmedian(lag90 - lag0, axis=0)
        out["d_an7_ride"] = np.nanmedian(lagride - lag0, axis=0)
        out["d_an7_absdelta90"] = np.nanmean(np.abs(lag90 - lag0), axis=0)
        out["d_ref_lag24"] = lag0[0]
        out["d_prof7_0"] = np.nanmedian(lag0, axis=0)
        out["d_prof7_90"] = np.nanmedian(lag90, axis=0)
    out["d_prof7_delta90"] = out.d_prof7_90 - out.d_prof7_0
    out["d_anom7"] = ref - out.d_prof7_0.to_numpy()
    roll = pd.Series(ref).rolling(DAY, min_periods=DAY // 2)
    out["d_mean24"] = roll.mean().to_numpy()
    out["d_anom24"] = ref - out.d_mean24.to_numpy()
    out["d_min24"] = roll.min().to_numpy()
    out["d_max24"] = roll.max().to_numpy()

    # sessions: latest closed session mean of the reference and lead to the next one
    local = pd.to_datetime(full_index, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur")
    minute = local.hour * 60 + local.minute
    dates = local.date
    for name, (h0, h1) in config.SESSION_HOURS.items():
        inside = (minute >= h0 * 60) & (minute < h1 * 60)
        daily = pd.Series(ref[inside]).groupby(np.asarray(dates)[inside]).mean()
        closes = {d: int(pd.Timestamp(d).tz_localize("Asia/Kuala_Lumpur").timestamp()) + h1 * 3600 for d in daily.index}
        close_epochs = np.array([closes[d] for d in daily.index], np.int64)
        means = daily.to_numpy(float)
        pos = np.searchsorted(close_epochs, full_index, side="right") - 1      # close <= t
        last = np.where(pos >= 0, means[np.clip(pos, 0, None)], np.nan)
        out[f"d_{name}_last"] = last
        out[f"d_{name}_last_minus_ref"] = last - ref
        starts = np.array([next_session_start(int(i), (h0, h1)) for i in full_index], np.int64)
        out[f"d_{name}_lead_min"] = (starts - full_index) / 60
    return out.reindex(ds.index)


ANALOG = ["d_an1_delta90", "d_an7_delta90", "d_an7_ride", "d_an7_absdelta90", "d_ref_lag24", "d_prof7_0",
          "d_prof7_90", "d_prof7_delta90", "d_anom7", "d_mean24", "d_anom24", "d_min24", "d_max24"]
SESSION = {"morning": ["d_morning_last", "d_morning_last_minus_ref", "d_morning_lead_min"],
           "afternoon": ["d_afternoon_last", "d_afternoon_last_minus_ref", "d_afternoon_lead_min"]}

TIME = ["t_hour", "t_day_sin", "t_day_cos", "t_dow", "t_weekend"]
SENSOR = ["s_freshPm25", "s_rawDelta5", "s_rawDelta10", "s_rawDelta15", "s_rawDelta30", "s_rawDelta60",
          "s_rawSlope15PerMinute", "s_rawSlope30PerMinute", "s_rawSd15", "s_rawSd30", "s_rawSd60",
          "s_rawRange15", "s_rawRange30", "s_freshTemperature", "s_freshHumidity",
          "s_rawTemperatureDelta30", "s_rawHumidityDelta30"]
# Mixing-layer and rain drivers of the diurnal cycle (Open-Meteo, issued vintage visible at t).
NWP = ["w_blh_0", "w_blh_90", "w_blh_180", "w_ride_blh_mean", "w_pprob_90", "w_next3h_precip_sum",
       "w_ride_precip_sum", "w_cloud_90", "w_wspd10_90", "w_temp_0", "w_temp_90"]
NWP_SESSION = {"morning": ["w_morning_blh_mean", "w_morning_pprob_mean", "w_morning_precip_sum",
                           "w_morning_wspd10_mean", "w_morning_covered"],
               "afternoon": ["w_afternoon_blh_mean", "w_afternoon_pprob_mean", "w_afternoon_precip_sum",
                             "w_afternoon_wspd10_mean", "w_afternoon_covered"]}


def add(ds: pd.DataFrame) -> pd.DataFrame:
    """Return ds with the d_* columns and two NWP differences appended."""
    extra = build(ds)
    frame = ds.join(extra)
    frame["d_w_blh_change90"] = frame.w_blh_90 - frame.w_blh_0
    frame["d_w_temp_change90"] = frame.w_temp_90 - frame.w_temp_0
    return frame


def feature_set(name: str, session: str | None = None) -> list[str]:
    cols = TIME + SENSOR
    if name in ("diurnal", "diurnal_nwp"):
        cols = cols + ANALOG
    if name in ("diurnal_nwp", "time_nwp"):
        cols = cols + NWP + ["d_w_blh_change90", "d_w_temp_change90"]
    if session:
        cols = cols + SESSION[session]
        if name in ("diurnal_nwp", "time_nwp"):
            cols = cols + NWP_SESSION[session]
    return cols


def leak_check(ds: pd.DataFrame, n=25, seed=0) -> int:
    """Recompute the d_* features with every reference after a sampled issue t blanked out.

    A feature that used anything after t would change. Returns the mismatch count.
    """
    full = build(ds)
    cols = ANALOG + SESSION["morning"] + SESSION["afternoon"]
    index = ds.index.to_numpy(np.int64)
    picks = np.random.default_rng(seed).choice(index[index > index[0] + 8 * 86400], n, replace=False)
    bad = 0
    for t in picks:
        cut = ds[["y_ref"]].copy()
        cut.loc[cut.index > t, "y_ref"] = np.nan
        a = full.loc[t, cols].to_numpy(float)
        b = build(cut).loc[t, cols].to_numpy(float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        bad += int((~same).sum())
    print(f"leak check: {bad} mismatches over {n} issues x {len(cols)} columns")
    return bad
