"""Leak-free feature groups for arbitrary issue clocks.

Every group selects only records whose visibility clock (sources.py) is at or
before the issue epoch. Column prefixes name the group:

  m_   issue metadata (fresh reference clock, availability, watermark)
  s_   production fresh_sensor_features (208 columns, production function)
  t_   calendar/time of day (MYT)
  w_   latest issued Open-Meteo vintage (age <= 2 h): point values 0..6 h,
       ride-window and session-window aggregates, 60-min revisions
  c_   latest issued CAMS run (age <= 3 h)
  rp_  regional CAMS PM grid (since 2026-10-01)
  rw_  regional GFS 925/850 hPa wind (since 2026-09-05)
  o_   Subang airport reports (first-fetch clock)
  n_   neighbour AirGradient stations, strict receipt clock (live from 2026-10-05)
  nr_  neighbour stations, reconstructed clock (bucket end + 30 min; an assumption)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config, prod, sources

WEATHER_FIELDS = ("temperature_2m", "relative_humidity_2m", "apparent_temperature",
                  "precipitation_probability", "precipitation", "showers", "cloud_cover",
                  "pressure_msl", "wind_speed_10m", "wind_gusts_10m", "wind_speed_180m",
                  "wind_speed_925hPa", "boundary_layer_height", "cape")
WEATHER_DIRECTIONS = (("wind_direction_10m", "wind_speed_10m", "wind10"),
                      ("wind_direction_180m", "wind_speed_180m", "wind180"),
                      ("wind_direction_925hPa", "wind_speed_925hPa", "wind925"))
WEATHER_LEADS = (0, 30, 60, 90, 120, 180, 240, 300, 360)
WEATHER_SHORT = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "apparent_temperature": "feels",
                 "precipitation_probability": "pprob", "precipitation": "precip", "showers": "showers",
                 "cloud_cover": "cloud", "pressure_msl": "pmsl", "wind_speed_10m": "wspd10",
                 "wind_gusts_10m": "gust", "wind_speed_180m": "wspd180", "wind_speed_925hPa": "wspd925",
                 "boundary_layer_height": "blh", "cape": "cape"}
AMOUNT_FIELDS = ("precipitation", "showers")       # hourly interval amounts, never interpolated
CAMS_FIELDS = ("pm2_5", "pm10", "aerosol_optical_depth", "dust")
TTDI = (3.1411, 101.6275)


# ----------------------------------------------------------------------------- sensor
def sensor_group(issues, rows=None) -> pd.DataFrame:
    """Production fresh_sensor_features on harness-visible rows, plus metadata."""
    sensor = prod.module("fresh_sensor_features")
    rows = sources.harness_rows() if rows is None else rows
    frame = sensor.issue_feature_frame(sensor.prepare_rows(rows), list(np.asarray(issues, dtype=float)))
    provenance = frame.attrs.get("featureProvenance", [])
    # Harness rows always carry an availability clock, so production's
    # unknownReceiptRows is 0 even for legacy rows. Count the visible input
    # rows (180-min window) that have no real receipt instead.
    table = sources.sensor_table()
    legacy = table.loc[~table.receipt_known]
    legacy_epoch = legacy.epoch.to_numpy(float)
    legacy_visible = legacy.visible_epoch.to_numpy(float)
    legacy_inputs = []
    for issue in np.asarray(issues, dtype=float):
        lo = np.searchsorted(legacy_epoch, issue - sensor.LOOKBACK_MINUTES * 60, side="right")
        hi = np.searchsorted(legacy_epoch, issue, side="right")
        legacy_inputs.append(int((legacy_visible[lo:hi] <= issue).sum()))
    meta = pd.DataFrame({
        "m_fresh_available": [bool(p.get("available")) for p in provenance],
        "m_fresh_ref_epoch": [p.get("freshReferenceEpoch") for p in provenance],
        "m_sensor_watermark": [p.get("featureSourceMaxEpoch") for p in provenance],
        "m_input_rows_180min": [p.get("inputRowCount") for p in provenance],
        "m_receipts_all_known": [count == 0 and (p.get("inputRowCount") or 0) > 0
                                 for count, p in zip(legacy_inputs, provenance)],
    }, index=frame.index)
    out = frame.add_prefix("s_").join(meta)
    out.index = pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch")
    for column in ("m_fresh_ref_epoch", "m_sensor_watermark"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


# ----------------------------------------------------------------------------- time
def time_group(issues) -> pd.DataFrame:
    issues = np.asarray(issues, dtype=np.int64)
    local = pd.to_datetime(issues, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur")
    hour = local.hour + local.minute / 60
    dow = local.dayofweek.to_numpy()
    return pd.DataFrame({
        "t_hour": hour, "t_day_sin": np.sin(2 * np.pi * hour / 24), "t_day_cos": np.cos(2 * np.pi * hour / 24),
        "t_dow": dow, "t_dow_sin": np.sin(2 * np.pi * dow / 7), "t_dow_cos": np.cos(2 * np.pi * dow / 7),
        "t_weekend": (dow >= 5).astype(float),
        "t_days_since_start": (issues - config.GRID_START_EPOCH) / 86400,
    }, index=pd.Index(issues, name="issue_epoch"))


# ----------------------------------------------------------------------------- helpers
def _latest(visible: np.ndarray, issue: float) -> int:
    return int(np.searchsorted(visible, issue, side="right")) - 1


class _Run:
    """One issued weather/CAMS payload as interpolable hourly arrays."""

    def __init__(self, fetched, visible, payload, fields, directions=()):
        self.fetched, self.visible = fetched, visible
        names = list(fields) + [d for d, _, _ in directions]
        epochs, matrix = sources.hourly_matrix(payload, names)
        self.epochs = epochs
        self.values = {name: matrix[:, i] for i, name in enumerate(fields)}
        for direction, speed, short in directions:
            angle = np.deg2rad(matrix[:, names.index(direction)] % 360)
            spd = self.values[speed]
            self.values[short + "_u"] = -spd * np.sin(angle)
            self.values[short + "_v"] = -spd * np.cos(angle)

    def at(self, name, times):
        """Bounded linear interpolation; amounts use the containing right-labelled hour."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        out = np.full(len(times), np.nan)
        x, y = self.epochs, self.values[name]
        if len(x) < 2:
            return out
        hi = np.searchsorted(x, times, side="left")          # first valid time >= t
        hi_c = np.minimum(hi, len(x) - 1)
        exact = (hi < len(x)) & (x[hi_c] == times)
        lo_c = np.maximum(hi - 1, 0)
        inside = (hi > 0) & (hi < len(x)) & ~exact & ((x[hi_c] - x[lo_c]) <= 3600)
        if name in AMOUNT_FIELDS:
            out[inside] = y[hi_c][inside]
        else:
            span = np.where(inside, x[hi_c] - x[lo_c], 1.0)
            frac = (times - x[lo_c]) / span
            value = y[lo_c] + (y[hi_c] - y[lo_c]) * frac
            out[inside] = value[inside]
        out[exact] = y[hi_c][exact]
        return out

    def window_mean(self, name, start, end, step=900):
        times = np.arange(start + step / 2, end, step)
        values = self.at(name, times)
        return float(np.mean(values)) if len(values) and np.isfinite(values).all() else np.nan

    def window_sum_amount(self, name, start, end):
        """Sum hourly amounts whose right-labelled hour overlaps (start, end], prorated."""
        x, y = self.epochs, self.values[name]
        total, covered = 0.0, 0.0
        for right, value in zip(x, y):
            overlap = min(end, right) - max(start, right - 3600)
            if overlap > 0:
                if not np.isfinite(value):
                    return np.nan
                total += value * overlap / 3600
                covered += overlap
        return total if covered >= (end - start) - 1 else np.nan


def _session_windows(issue):
    from .targets import next_session
    return {name: next_session(int(issue), name) for name in ("morning", "afternoon")}


# ----------------------------------------------------------------------------- weather
def weather_group(issues) -> pd.DataFrame:
    runs = [_Run(f, v, p, WEATHER_FIELDS, WEATHER_DIRECTIONS)
            for f, v, _, p in sources.forecast_runs("weather_forecast_runs")]
    visible = np.array([r.visible for r in runs])
    point_names = [*WEATHER_FIELDS, *(s + c for _, _, s in WEATHER_DIRECTIONS for c in ("_u", "_v"))]
    records = []
    for issue in np.asarray(issues, dtype=np.int64):
        record = {}
        index = _latest(visible, issue)
        run = runs[index] if index >= 0 and issue - runs[index].fetched <= config.WEATHER_MAX_AGE_SECONDS else None
        if run is None:
            records.append(record)
            continue
        record["w_age_min"] = (issue - run.fetched) / 60
        for name in point_names:
            short = WEATHER_SHORT.get(name, name)
            values = run.at(name, issue + np.asarray(WEATHER_LEADS) * 60)
            record.update({f"w_{short}_{lead}": v for lead, v in zip(WEATHER_LEADS, values)})
        ride = (issue + config.ARRIVAL_SECONDS, issue + config.RIDE_END_SECONDS)
        windows = {"ride": ride, **_session_windows(issue)}
        for key, (start, end) in windows.items():
            for name in ("temperature_2m", "relative_humidity_2m", "precipitation_probability",
                         "wind_speed_10m", "wind10_u", "wind10_v", "wind925_u", "wind925_v",
                         "boundary_layer_height", "cloud_cover", "cape"):
                record[f"w_{key}_{WEATHER_SHORT.get(name, name)}_mean"] = run.window_mean(name, start, end)
            record[f"w_{key}_precip_sum"] = run.window_sum_amount("precipitation", start, end)
            if key != "ride":
                record[f"w_{key}_covered"] = float(run.epochs[-1] >= end) if len(run.epochs) else 0.0
        record["w_prior3h_precip_sum"] = run.window_sum_amount("precipitation", issue - 10800, issue)
        record["w_next3h_precip_sum"] = run.window_sum_amount("precipitation", issue, issue + 10800)
        probs = run.at("precipitation_probability", issue + np.arange(0, 361, 30) * 60)
        record["w_next6h_pprob_max"] = float(np.nanmax(probs)) if np.isfinite(probs).any() else np.nan
        wet = np.flatnonzero(probs >= 50)
        record["w_first_pprob50_lead_min"] = float(wet[0] * 30) if len(wet) else np.nan
        previous_index = _latest(visible, issue - 3600)
        if previous_index >= 0 and issue - 3600 - runs[previous_index].fetched <= config.WEATHER_MAX_AGE_SECONDS:
            previous = runs[previous_index]
            for name in ("temperature_2m", "relative_humidity_2m", "precipitation_probability", "wind_speed_10m"):
                short = WEATHER_SHORT[name]
                record[f"w_rev60_{short}_90"] = float(run.at(name, issue + 5400)[0] - previous.at(name, issue + 5400)[0])
        records.append(record)
    return pd.DataFrame(records, index=pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch"))


# ----------------------------------------------------------------------------- CAMS
def cams_group(issues) -> pd.DataFrame:
    runs = [_Run(f, v, p, CAMS_FIELDS) for f, v, _, p in sources.forecast_runs("air_quality_forecast_runs")]
    visible = np.array([r.visible for r in runs])
    records = []
    for issue in np.asarray(issues, dtype=np.int64):
        record = {}
        index = _latest(visible, issue)
        run = runs[index] if index >= 0 and issue - runs[index].fetched <= config.CAMS_MAX_AGE_SECONDS else None
        if run is not None:
            record["c_age_min"] = (issue - run.fetched) / 60
            for name, short in (("pm2_5", "pm25"), ("pm10", "pm10"), ("aerosol_optical_depth", "aod"), ("dust", "dust")):
                values = run.at(name, issue + np.array([0, 90, 180, 360]) * 60)
                record.update({f"c_{short}_{lead}": v for lead, v in zip((0, 90, 180, 360), values)})
            record["c_pm25_delta90"] = record["c_pm25_90"] - record["c_pm25_0"]
            record["c_pm25_ride_mean"] = run.window_mean("pm2_5", issue + config.ARRIVAL_SECONDS, issue + config.RIDE_END_SECONDS)
            for key, (start, end) in _session_windows(issue).items():
                record[f"c_pm25_{key}_mean"] = run.window_mean("pm2_5", start, end)
        records.append(record)
    return pd.DataFrame(records, index=pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch"))


# ----------------------------------------------------------------------------- regional
def regional_group(issues) -> pd.DataFrame:
    pm_runs = sources.regional_pm_runs()
    pm_visible = np.array([r[1] for r in pm_runs])
    wind_runs = sources.regional_wind_runs()
    wind_visible = np.array([r[1] for r in wind_runs])
    records = []
    for issue in np.asarray(issues, dtype=np.int64):
        record = {}
        index = _latest(pm_visible, issue)
        if index >= 0 and issue - pm_runs[index][0] <= 6 * 3600:
            _, _, times, lats, lons, pm = pm_runs[index]
            for lead in (0, 90, 180):
                t = issue + lead * 60
                values = np.array([np.interp(t, times, pm[:, j], left=np.nan, right=np.nan) for j in range(pm.shape[1])])
                if not np.isfinite(values).all():
                    continue
                record[f"rp_mean_{lead}"] = float(values.mean())
                record[f"rp_ns_grad_{lead}"] = float(values[lats > lats.mean()].mean() - values[lats < lats.mean()].mean())
                record[f"rp_ew_grad_{lead}"] = float(values[lons > lons.mean()].mean() - values[lons < lons.mean()].mean())
                record[f"rp_max_{lead}"] = float(values.max())
            record["rp_age_min"] = (issue - pm_runs[index][0]) / 60
        index = _latest(wind_visible, issue)
        if index >= 0 and issue - wind_runs[index][0] <= 12 * 3600:
            _, _, times, lats, lons, data = wind_runs[index]
            nearest = int(np.argmin((lats - TTDI[0]) ** 2 + (lons - TTDI[1]) ** 2))
            for lead in (0, 90):
                t = issue + lead * 60
                for level, (si, di) in (("925", (0, 1)), ("850", (2, 3))):
                    speed = np.interp(t, times, data[nearest, :, si], left=np.nan, right=np.nan)
                    direction = data[nearest, int(np.clip(np.searchsorted(times, t), 0, len(times) - 1)), di]
                    angle = math.radians(direction % 360) if np.isfinite(direction) else np.nan
                    record[f"rw_{level}_u_{lead}"] = -speed * math.sin(angle) if np.isfinite(angle) else np.nan
                    record[f"rw_{level}_v_{lead}"] = -speed * math.cos(angle) if np.isfinite(angle) else np.nan
            k = int(np.clip(np.searchsorted(times, issue), 0, len(times) - 1))
            speed, direction = data[:, k, 0], np.deg2rad(data[:, k, 1] % 360)
            record["rw_925_domain_u_0"] = float(np.nanmean(-speed * np.sin(direction)))
            record["rw_925_domain_v_0"] = float(np.nanmean(-speed * np.cos(direction)))
            record["rw_age_min"] = (issue - wind_runs[index][0]) / 60
        records.append(record)
    return pd.DataFrame(records, index=pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch"))


# ----------------------------------------------------------------------------- Subang
def subang_group(issues) -> pd.DataFrame:
    table = sources.subang_table()
    visible = table.visible_epoch.to_numpy(float)
    order = np.argsort(visible, kind="stable")
    table, visible = table.iloc[order].reset_index(drop=True), visible[order]
    report = table.report_epoch.to_numpy(float)
    best = np.maximum.accumulate(report)               # newest report visible so far
    text = table.present_weather.fillna("").str.upper()
    rain = text.str.contains("RAIN|SHOWER|DRIZZLE|THUNDERSTORM, SLIGHT|THUNDERSTORM, HEAVY|PRECIPITATION WITHIN SIGHT, REACHING THE GROUND OR THE SURFACE OF THE SEA, NEAR").to_numpy()
    thunder = text.str.contains("THUNDER|LIGHTNING").to_numpy()
    haze = text.str.contains("HAZE|SMOKE|MIST").to_numpy()
    by_report = {}
    for i, epoch in enumerate(report):
        by_report.setdefault(epoch, i)
    records = []
    for issue in np.asarray(issues, dtype=np.int64):
        k = _latest(visible, issue)
        record = {}
        if k >= 0 and issue - best[k] <= 3 * 3600:
            i = by_report[best[k]]
            row = table.iloc[i]
            direction = row.wind_direction
            speed = row.wind_speed_kmh
            angle = math.radians(direction % 360) if pd.notna(direction) else np.nan
            record = {"o_age_min": (issue - best[k]) / 60, "o_temp": row.atmp, "o_rh": row.rhum,
                      "o_dewpoint": row.dewpoint, "o_visibility_km": row.visibility_km,
                      "o_wind_kmh": speed,
                      "o_wind_u": -speed * math.sin(angle) if pd.notna(speed) and np.isfinite(angle) else np.nan,
                      "o_wind_v": -speed * math.cos(angle) if pd.notna(speed) and np.isfinite(angle) else np.nan,
                      "o_rain": float(rain[i]), "o_thunder": float(thunder[i]), "o_haze": float(haze[i]),
                      "o_precip_mm": row.precipitation_mm}
        records.append(record)
    return pd.DataFrame(records, index=pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch"))


# ----------------------------------------------------------------------------- neighbours
def neighbor_group(issues, *, reconstructed: bool, ttdi_reference=None) -> pd.DataFrame:
    """Per station: latest visible PM2.5 (age <= 90 min) and its 60-min change."""
    table = sources.neighbor_table()
    prefix = "nr_" if reconstructed else "n_"
    clock = "reconstructed_visible" if reconstructed else "receipt_epoch"
    issues = np.asarray(issues, dtype=np.int64)
    out = pd.DataFrame(index=pd.Index(issues, name="issue_epoch"))
    station_values = []
    for location, group in table.groupby("location_id"):
        name = f"{prefix}{int(location)}"
        visible = group[clock].to_numpy(float)
        ends = group.end_epoch.to_numpy(float)
        values = group.pm25.to_numpy(float)
        order = np.argsort(visible, kind="stable")
        visible, ends, values = visible[order], ends[order], values[order]
        latest_val = np.full(len(issues), np.nan)
        latest_end = np.full(len(issues), np.nan)
        change = np.full(len(issues), np.nan)
        # Prefix maxima: newest observation end among rows visible by each issue.
        best_end = np.maximum.accumulate(ends)
        position = {}
        for i, end in enumerate(ends):
            position.setdefault(end, i)
        for j, issue in enumerate(issues):
            k = _latest(visible, issue)
            if k < 0 or issue - best_end[k] > config.NEIGHBOR_MAX_AGE_SECONDS:
                continue
            end = best_end[k]
            latest_end[j], latest_val[j] = end, values[position[end]]
            # Value about 60 min earlier among rows already visible by this issue.
            visible_mask_end = ends[: k + 1]
            older = visible_mask_end[(visible_mask_end <= end - 2700) & (visible_mask_end >= end - 5400)]
            if len(older):
                target = older[np.argmin(np.abs(older - (end - 3600)))]
                change[j] = latest_val[j] - values[position[target]]
        out[f"{name}_pm25"] = latest_val
        out[f"{name}_age_min"] = (issues - latest_end) / 60
        out[f"{name}_chg60"] = change
        station_values.append(latest_val)
    stack = np.vstack(station_values) if station_values else np.full((1, len(issues)), np.nan)
    count = np.isfinite(stack).sum(axis=0)
    total = np.where(np.isfinite(stack), stack, 0.0).sum(axis=0)
    out[f"{prefix}count"] = count
    out[f"{prefix}mean"] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if ttdi_reference is not None:
        out[f"{prefix}mean_minus_ref"] = out[f"{prefix}mean"] - np.asarray(ttdi_reference, dtype=float)
    return out


GROUPS = {"meta": "m_", "sensor": "s_", "time": "t_", "weather": "w_", "cams": "c_",
          "regional_pm": "rp_", "regional_wind": "rw_", "subang": "o_",
          "neighbors_strict": "n_", "neighbors_reconstructed": "nr_"}


def build(issues, rows=None, *, progress=print) -> pd.DataFrame:
    issues = np.asarray(issues, dtype=np.int64)
    parts = []
    progress("sensor features")
    sensor = sensor_group(issues, rows)
    parts.append(sensor)
    parts.append(time_group(issues))
    progress("weather features")
    parts.append(weather_group(issues))
    progress("cams features")
    parts.append(cams_group(issues))
    progress("regional features")
    parts.append(regional_group(issues))
    progress("subang features")
    parts.append(subang_group(issues))
    progress("neighbour features")
    reference = sensor["s_freshPm25"].to_numpy(float)
    parts.append(neighbor_group(issues, reconstructed=False, ttdi_reference=reference))
    parts.append(neighbor_group(issues, reconstructed=True, ttdi_reference=reference))
    return pd.concat(parts, axis=1)
