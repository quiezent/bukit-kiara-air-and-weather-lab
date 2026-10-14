"""Issued-weather inputs for the weather-inputs candidate.

Every column here is a function of one dataset row: the harness ``w_`` group
(latest Open-Meteo vintage visible at the issue, receipt clock or fetched + 5 s,
max age 2 h) and the harness ``s_`` sensor group (visible at the issue). No
fitting happens here, so nothing can leak across rows or days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR28 = ['s_freshPm25', 's_rawLag5', 's_rawLag10', 's_rawLag15', 's_rawLag30', 's_rawLag60',
            's_rawDelta5', 's_rawDelta10', 's_rawDelta15', 's_rawDelta30', 's_rawDelta60',
            's_rawSd15', 's_rawSd30', 's_rawSd60', 's_rawRange15', 's_rawRange30',
            's_rawSlope15PerMinute', 's_rawSlope30PerMinute', 's_freshTemperature', 's_freshHumidity',
            's_rawTemperatureDelta30', 's_rawHumidityDelta30', 's_freshSampleCount', 's_rawSampleCount30',
            's_timeSin', 's_timeCos', 's_issueOffsetSin', 's_issueOffsetCos']   # arrival_logistic FEATURE_COLUMNS

RAW = (
    # rain: probability and right-labelled hourly amounts over the next 3 h, ride window, 6 h
    ['w_pprob_0', 'w_pprob_30', 'w_pprob_60', 'w_pprob_90', 'w_pprob_120', 'w_pprob_180',
     'w_precip_0', 'w_precip_30', 'w_precip_60', 'w_precip_90', 'w_precip_120', 'w_precip_180',
     'w_showers_60', 'w_showers_90', 'w_next3h_precip_sum', 'w_prior3h_precip_sum', 'w_ride_precip_sum',
     'w_ride_pprob_mean', 'w_next6h_pprob_max', 'w_first_pprob50_lead_min']
    # mixing / boundary layer, cloud, convection, wind
    + ['w_blh_0', 'w_blh_90', 'w_ride_blh_mean', 'w_cloud_90', 'w_cape_0', 'w_cape_90',
       'w_wspd10_0', 'w_wspd10_90', 'w_gust_90', 'w_wind10_u_90', 'w_wind10_v_90',
       'w_wind925_u_90', 'w_wind925_v_90', 'w_temp_0', 'w_rh_0']
    # vintage revisions and age
    + ['w_rev60_pprob_90', 'w_rev60_temp_90', 'w_age_min'])


# A small physically motivated subset (rain amount/probability to arrival and over the ride,
# washout scaled by the current level, mixing-height growth, forecast cooling).
CORE = ['w_pprob_60', 'w_pprob_90', 'w_precip_60', 'w_precip_90', 'w_next3h_precip_sum', 'w_ride_precip_sum',
        'wx_ref_x_pprob90', 'wx_ref_x_rain3h', 'wx_ref_x_rain60_90', 'wx_dblh_log_90', 'wx_tgap_90', 'wx_dtemp_90']


def build(ds: pd.DataFrame) -> pd.DataFrame:
    """Raw selected weather columns plus changes, forecast-vs-observed gaps and level interactions."""
    out = ds[RAW].astype(float).copy()
    ref = ds['s_freshPm25'].astype(float)
    out['wx_dtemp_90'] = ds.w_temp_90 - ds.w_temp_0               # forecast cooling/warming to arrival
    out['wx_dtemp_180'] = ds.w_temp_180 - ds.w_temp_0
    out['wx_drh_90'] = ds.w_rh_90 - ds.w_rh_0
    out['wx_dpmsl_90'] = ds.w_pmsl_90 - ds.w_pmsl_0
    out['wx_dblh_log_90'] = np.log(ds.w_blh_90.clip(lower=0) + 50) - np.log(ds.w_blh_0.clip(lower=0) + 50)
    out['wx_dwspd_90'] = ds.w_wspd10_90 - ds.w_wspd10_0
    out['wx_tgap_0'] = ds.w_temp_0 - ds.s_freshTemperature        # forecast says cooler than observed now
    out['wx_tgap_90'] = ds.w_temp_90 - ds.s_freshTemperature
    out['wx_rhgap_0'] = ds.w_rh_0 - ds.s_freshHumidity
    rain3 = np.log1p(ds.w_next3h_precip_sum.clip(lower=0))
    out['wx_ref_x_pprob90'] = ref * ds.w_pprob_90 / 100            # washout can only remove what is there
    out['wx_ref_x_rain3h'] = ref * rain3
    out['wx_ref_x_rain60_90'] = ref * np.log1p((ds.w_precip_60 + ds.w_precip_90).clip(lower=0))
    out['wx_ref_x_dblh'] = ref * out['wx_dblh_log_90']
    return out
