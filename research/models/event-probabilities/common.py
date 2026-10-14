"""Shared pieces of the event-probability candidate: data, derived features, split.

Every derived feature uses only values visible at the issue clock:
* columns of the issue row itself (harness feature groups, already leak-checked);
* ``s_freshPm25`` of EARLIER grid rows (each visible at its own, earlier issue);
* realised labels of earlier rows only once they were complete (``y_delta`` of
  the row 24 h earlier completes about 105 min after that row, i.e. ~22 h before
  the issue).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # research\models -> harness

from harness import config, dataset, scoring, walkforward   # noqa: E402

OUT = config.OUT / "event-probabilities"
OUT.mkdir(parents=True, exist_ok=True)

# Tune on the first half of the evaluation period, report on the second.
TUNE_START = config.myt_epoch("2026-09-10")
HOLDOUT_START = config.myt_epoch("2026-09-27")
EVAL_END = config.myt_epoch("2026-10-14")
# Base models start earlier so the chronological calibrator has out-of-sample
# history on the first evaluation day.
BASE_START = config.myt_epoch("2026-09-01")
EMBARGO_MINUTES = 120                                     # arrival logistic's production embargo

CLASS5 = [f"p_class5_{k}" for k in range(5)]
FIRST3 = ["p_first_drop", "p_first_none", "p_first_rise"]
CHANGE_EDGES = (-40.0, -20.0, 20.0, 40.0)                 # class5 boundaries on delta

SENSOR_BASE = [
    "s_freshPm25", "s_rawLag5", "s_rawLag10", "s_rawLag15", "s_rawLag30", "s_rawLag60",
    "s_rawDelta5", "s_rawDelta10", "s_rawDelta15", "s_rawDelta30", "s_rawDelta60",
    "s_rawSd15", "s_rawSd30", "s_rawSd60", "s_rawRange15", "s_rawRange30",
    "s_rawSlope15PerMinute", "s_rawSlope30PerMinute", "s_freshTemperature", "s_freshHumidity",
    "s_rawTemperatureDelta30", "s_rawHumidityDelta30", "s_timeSin", "s_timeCos"]
DERIVED = ["x_log_ref", "x_d90", "x_d180", "x_ref_minus_6h", "x_ref_minus_24h", "x_ref_over_24h",
           "x_sd_6h", "x_yday_delta", "x_yday_absdelta_3h"]
WEATHER = [
    "w_precip_0", "w_precip_30", "w_precip_60", "w_precip_90", "w_precip_120",
    "w_pprob_0", "w_pprob_30", "w_pprob_60", "w_pprob_90", "w_pprob_120",
    "w_next3h_precip_sum", "w_prior3h_precip_sum", "w_next6h_pprob_max", "w_first_pprob50_lead_min",
    "w_rev60_pprob_90", "w_cape_0", "w_cape_90", "w_cloud_0", "w_cloud_90",
    "w_blh_0", "w_blh_90", "x_blh_change90", "w_wspd10_0", "w_wspd10_90", "w_gust_90",
    "w_wind10_u_90", "w_wind10_v_90", "w_wind925_u_90", "w_wind925_v_90",
    "x_temp_change90", "x_rh_change90", "w_rh_90"]
CAMS = ["c_pm25_0", "c_pm25_90", "c_pm25_delta90", "x_cams_ratio"]
SUBANG = ["o_rain", "o_thunder", "o_visibility_km", "o_wind_kmh", "o_rh"]

FEATURE_SETS = {
    "sensor": SENSOR_BASE + DERIVED,
    "sensor_weather": SENSOR_BASE + DERIVED + WEATHER,
    "all": SENSOR_BASE + DERIVED + WEATHER + CAMS + SUBANG,
}


def load() -> pd.DataFrame:
    ds = dataset.load()
    return add_derived(ds)


def add_derived(ds: pd.DataFrame) -> pd.DataFrame:
    """Derived features from the issue row and from earlier rows only."""
    ds = ds.copy()
    index = ds.index.to_numpy(np.int64)
    assert np.all(np.diff(index) == config.GRID_SECONDS), "grid must be complete and sorted"
    ref = ds["s_freshPm25"].where(ds["m_fresh_available"])
    ds["x_log_ref"] = np.log(ref + 10)
    # Past grid rows (strictly earlier issues): shift by k rows = k * 5 min.
    ds["x_d90"] = ref - ref.shift(18)
    ds["x_d180"] = ref - ref.shift(36)
    mean6 = ref.rolling(72, min_periods=36).mean()
    mean24 = ref.rolling(288, min_periods=144).mean()
    ds["x_ref_minus_6h"] = ref - mean6
    ds["x_ref_minus_24h"] = ref - mean24
    ds["x_ref_over_24h"] = np.log((ref + 10) / (mean24 + 10))
    ds["x_sd_6h"] = ref.rolling(72, min_periods=36).std()
    # Realised +90 change of the issue 24 h earlier (complete ~22 h before now).
    delta = ds["y_delta"]
    complete = ds["y_arrival_complete"]
    yday = delta.shift(288)
    yday_complete = complete.shift(288)
    ds["x_yday_delta"] = yday.where(yday_complete < ds.index.to_series())
    absd = delta.abs().shift(288 - 18)               # rows centred on 24 h ago, +-90 min
    yday_c = complete.shift(288 - 18)
    ds["x_yday_absdelta_3h"] = absd.where(yday_c < ds.index.to_series()).rolling(36, min_periods=12).mean()
    ds["x_blh_change90"] = ds["w_blh_90"] - ds["w_blh_0"]
    ds["x_temp_change90"] = ds["w_temp_90"] - ds["w_temp_0"]
    ds["x_rh_change90"] = ds["w_rh_90"] - ds["w_rh_0"]
    ds["x_cams_ratio"] = np.log((ref + 10) / (ds["c_pm25_0"] + 10))
    return ds


def class5_from_delta(delta) -> np.ndarray:
    delta = np.asarray(delta, float)
    return np.select([delta <= -40, delta <= -20, delta < 20, delta < 40], [0, 1, 2, 3], 4)


def usable(frame: pd.DataFrame) -> np.ndarray:
    return frame["m_fresh_available"].to_numpy(bool) & (frame["s_freshPm25"].to_numpy(float) >= 0)


def with_tails(pred: pd.DataFrame) -> pd.DataFrame:
    """Coherent binary tails from the 5-class vector (production tail_probabilities)."""
    if all(c in pred for c in CLASS5):
        pred["p_fall40"] = pred["p_class5_0"]
        pred["p_fall20"] = pred["p_class5_0"] + pred["p_class5_1"]
        pred["p_rise20"] = pred["p_class5_3"] + pred["p_class5_4"]
        pred["p_rise40"] = pred["p_class5_4"]
    return pred


def save(pred: pd.DataFrame, name: str) -> Path:
    path = OUT / f"{name}.parquet"
    pred.to_parquet(path)
    return path


def load_pred(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


def days_between(ds, start, end):
    return walkforward.eval_days(ds, start=start, end=end)
