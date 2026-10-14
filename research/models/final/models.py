"""Final PM2.5 forecast models assembled on 2026-10-14 (see research/models/RESULTS.md).

Port-ready on purpose: this module imports only numpy, pandas and scikit-learn (the service's
pinned sklearn 1.5.1 / numpy 1.26.4), never the research harness. A port needs three calls:

    model = fit_point(history, cutoff)              # nightly, at MYT midnight `cutoff`
    le70 = fit_le70(history, cutoff, point=model)   # nightly; reuses the point model
    out = predict_point(model, features)            # per issue; `features` is the live dict
    p = predict_le70(le70, features)

Final choices (RESULTS.md):
* +90 arrival and ride mean/min/max: FINAL_POINT_VARIANT = P1_no_weather. Four
  HistGradientBoosting absolute-error regressors on (target - fresh reference); inputs = the
  28 arrival-logistic sensor columns, 4 TTDI mean-reversion controls, 5 time columns and 44
  spatial neighbour columns (per-station gap/60-min change/age, network medians, upwind-weighted
  medians using the issued Open-Meteo 10 m / 925 hPa wind). No weather or CAMS model inputs.
* P(ride mean <= 70): FINAL_LE70_VARIANT = E1_spatial_residual. The point model's ride-mean
  prediction plus the empirical distribution of out-of-fold log residuals
  log((y + 5) / (pred + 5)) from three day blocks of the same 28-day training window:
  P = share of residuals <= log(75 / (pred + 5)).

``history`` is a DataFrame indexed by issue epoch (UTC seconds, 5-min grid) holding the feature
columns plus the production targets and their completion clocks (``y_arrival``, ``y_ride_mean``,
``y_ride_min``, ``y_ride_max``, ``y_arrival_complete``, ``y_ride_complete``) and
``m_fresh_available``. Target definitions are production's own (fresh_numeric_model_v4
targets_for_issues); the harness reproduces them bit for bit.

``features`` is a plain dict with the model's input names. Missing keys count as NaN (the
gradient-boosting heads handle NaN natively). ``x_*`` mean-reversion controls are derived here
from the ``s_seq_pm25_*`` bins when absent. The ``sp_*`` spatial block comes from
final.spatial_live.spatial_features (or spatial.spatial_features on the research grid).

Outputs are literal model outputs: no gates, clipping, selection or persistence substitution.
Without a fresh reference (``m_fresh_available`` False or ``s_freshPm25`` missing/negative) every
output is NaN, i.e. unavailable.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------------- inputs
SENSOR28 = ['s_freshPm25', 's_rawLag5', 's_rawLag10', 's_rawLag15', 's_rawLag30', 's_rawLag60', 's_rawDelta5',
            's_rawDelta10', 's_rawDelta15', 's_rawDelta30', 's_rawDelta60', 's_rawSd15', 's_rawSd30', 's_rawSd60',
            's_rawRange15', 's_rawRange30', 's_rawSlope15PerMinute', 's_rawSlope30PerMinute', 's_freshTemperature',
            's_freshHumidity', 's_rawTemperatureDelta30', 's_rawHumidityDelta30', 's_freshSampleCount',
            's_rawSampleCount30', 's_timeSin', 's_timeCos', 's_issueOffsetSin', 's_issueOffsetCos']
CONTROLS = ["x_dev90", "x_dev180", "x_mean180", "x_slope180"]          # TTDI-only mean reversion
TIME = ["t_hour", "t_day_sin", "t_day_cos", "t_dow", "t_weekend"]
WEATHER = ["w_next3h_precip_sum", "w_prior3h_precip_sum", "w_next6h_pprob_max", "w_first_pprob50_lead_min",
           "w_pprob_0", "w_pprob_90", "w_precip_90", "w_wspd10_0", "w_wspd10_90", "w_wind10_u_90",
           "w_wind10_v_90", "w_wind925_u_90", "w_wind925_v_90", "w_blh_0", "w_blh_90", "w_rh_0", "w_rh_90",
           "w_temp_0", "w_temp_90", "w_cape_90", "w_cloud_90", "w_rev60_pprob_90", "w_ride_precip_sum",
           "w_ride_blh_mean"]
CAMS = ["c_pm25_0", "c_pm25_90", "c_pm25_delta90", "c_pm25_ride_mean"]

STATIONS = ("setia", "cyber", "enviro", "montkiara", "setapak", "klcc")
SP_STATION = [f"sp_{s}_{k}" for s in STATIONS for k in ("diff", "chg60", "age")]
SP_NET = ["sp_count", "sp_net_level", "sp_net_diff", "sp_net_diff_mad", "sp_net_chg30", "sp_net_chg60",
          "sp_net_chg60_min", "sp_net_chg60_max", "sp_net_rel60", "sp_net_level_minus_ref"]
SP_WIND = [f"sp_{tag}_{k}" for tag in ("w10", "w925")
           for k in ("up_diff", "up_chg60", "up_chg30", "up_rel60", "down_diff", "adv90", "up_eta_min", "up_align")]
SP_REGIONAL = ["sp_rp_up_grad", "sp_rp_delta90", "sp_cams_delta90"]
# Column order of spatial.spatial_features.build (kept so the frozen candidate reproduces exactly).
SP_ALL = SP_STATION + SP_NET + SP_WIND + SP_REGIONAL

# Point-head input sets compared on the shared harness (pre-registered in final/run.py).
POINT_VARIANTS = {
    "P0_spatial_v1": SENSOR28 + CONTROLS + TIME + WEATHER + CAMS + SP_ALL,      # frozen spatial candidate (112)
    "P1_no_weather": SENSOR28 + CONTROLS + TIME + SP_STATION + SP_NET + SP_WIND,  # drop weather/CAMS inputs (81)
    "P2_aggregates": SENSOR28 + CONTROLS + TIME + WEATHER + CAMS + SP_NET + SP_WIND + SP_REGIONAL,  # no per-station (94)
    "P3_agg_no_weather": SENSOR28 + CONTROLS + TIME + SP_NET + SP_WIND,           # (63)
    "P4_network_only": SENSOR28 + CONTROLS + TIME + SP_NET,                       # no wind weighting either (47)
}
SIMPLICITY_ORDER = ["P4_network_only", "P3_agg_no_weather", "P1_no_weather", "P2_aggregates", "P0_spatial_v1"]
FINAL_POINT_VARIANT = "P1_no_weather"     # chosen by the pre-registered tune-split rule (final/run.py); see RESULTS.md

POINT_HEADS = {"arrival": "y_arrival", "ride_mean": "y_ride_mean", "ride_min": "y_ride_min", "ride_max": "y_ride_max"}
LOOKBACK_DAYS = 28
EMBARGO_MINUTES = 330          # fresh_numeric_model_v4.EMBARGO_SECONDS (the Ridge v4 rule)
ORIGIN_SECONDS = 900           # point heads train on 15-min origins (5-min neighbours are near duplicates)
HGB_PARAMS = dict(loss="absolute_error", max_iter=120, learning_rate=0.1, max_leaf_nodes=15,
                  min_samples_leaf=60, l2_regularization=1.0, early_stopping=False, random_state=0)
PLANNING_CUTOFF = 70.0


# ----------------------------------------------------------------------------- helpers
def time_features(issue_epoch: int) -> dict:
    """t_* inputs exactly as the harness builds them (MYT clock, seconds ignored)."""
    local = pd.Timestamp(int(issue_epoch), unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    hour = local.hour + local.minute / 60
    dow = local.dayofweek
    return {"t_hour": hour, "t_day_sin": math.sin(2 * math.pi * hour / 24), "t_day_cos": math.cos(2 * math.pi * hour / 24),
            "t_dow": float(dow), "t_weekend": float(dow >= 5)}


def add_controls(frame: pd.DataFrame) -> pd.DataFrame:
    """x_dev90/x_dev180/x_mean180/x_slope180 from the 3-min s_seq_pm25 bins (ages in minutes)."""
    seq = [c for c in frame.columns if c.startswith("s_seq_pm25_")]
    out = frame.copy()
    if not seq:
        for c in CONTROLS:
            if c not in out:
                out[c] = np.nan
        return out
    ages = np.array([int(c.split("_")[3]) for c in seq])
    bins = frame[seq].to_numpy(float)
    ref = frame["s_freshPm25"].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean90 = np.nanmean(bins[:, ages < 90], axis=1)
        mean180 = np.nanmean(bins, axis=1)
        old = np.nanmean(bins[:, ages >= 150], axis=1)
        new = np.nanmean(bins[:, ages < 30], axis=1)
    out["x_dev90"] = ref - mean90
    out["x_dev180"] = ref - mean180
    out["x_mean180"] = mean180
    out["x_slope180"] = new - old
    return out


def usable(frame: pd.DataFrame) -> np.ndarray:
    """Rows with a fresh reference (production's fresh_sensor_features availability)."""
    fresh = frame["m_fresh_available"].to_numpy(bool) if "m_fresh_available" in frame else np.ones(len(frame), bool)
    ref = frame["s_freshPm25"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        return fresh & (ref >= 0)


def training_rows(frame: pd.DataFrame, cutoff: int, *, lookback_days=LOOKBACK_DAYS, embargo_minutes=EMBARGO_MINUTES,
                  complete=("y_arrival_complete", "y_ride_complete"), origin_seconds=None) -> np.ndarray:
    """Issue in (cutoff - lookback, cutoff - embargo), every label closed before cutoff, fresh ref."""
    idx = frame.index.to_numpy(np.int64)
    ok = (idx > cutoff - lookback_days * 86400) & (idx < cutoff - embargo_minutes * 60)
    for column in complete:
        ok &= frame[column].to_numpy(np.int64) < cutoff
    ok &= usable(frame)
    if origin_seconds:
        ok &= (idx % origin_seconds) == 0
    return ok


def _frame_from_dict(features: dict, columns) -> pd.DataFrame:
    row = {c: features.get(c, np.nan) for c in set(columns) | set(features)}
    frame = pd.DataFrame([row])
    if any(c not in features for c in CONTROLS) and any(k.startswith("s_seq_pm25_") for k in features):
        frame = add_controls(frame)
    for c in columns:
        if c not in frame:
            frame[c] = np.nan
    return frame


def _dict_usable(features: dict) -> bool:
    ref = features.get("s_freshPm25")
    if ref is None or not np.isfinite(ref) or ref < 0:
        return False
    return bool(features.get("m_fresh_available", True))


# ----------------------------------------------------------------------------- point heads
@dataclass
class PointModel:
    cutoff: int
    columns: list
    heads: dict = field(default_factory=dict)      # head -> fitted regressor on (target - ref)
    n_train: dict = field(default_factory=dict)


def fit_point(history: pd.DataFrame, cutoff: int, columns=None) -> PointModel:
    """Fit the +90 arrival and ride mean/min/max heads at MYT midnight ``cutoff``."""
    columns = list(POINT_VARIANTS[FINAL_POINT_VARIANT] if columns is None else columns)
    if any(c not in history for c in CONTROLS):
        history = add_controls(history)
    base = training_rows(history, cutoff, origin_seconds=ORIGIN_SECONDS)
    model = PointModel(cutoff=int(cutoff), columns=columns)
    for head, target in POINT_HEADS.items():
        y = history[target].to_numpy(float)
        ok = base & np.isfinite(y)
        tr = history.loc[ok]
        reg = HistGradientBoostingRegressor(**HGB_PARAMS)
        reg.fit(tr[columns].to_numpy(float), (tr[target] - tr["s_freshPm25"]).to_numpy(float))
        model.heads[head] = reg
        model.n_train[head] = int(ok.sum())
    return model


def predict_point_frame(model: PointModel, frame: pd.DataFrame) -> pd.DataFrame:
    if any(c not in frame for c in CONTROLS):
        frame = add_controls(frame)
    use = usable(frame)
    out = pd.DataFrame(index=frame.index)
    x = frame.loc[use, model.columns].to_numpy(float)
    ref = frame.loc[use, "s_freshPm25"].to_numpy(float)
    for head, reg in model.heads.items():
        pred = np.full(len(frame), np.nan)
        if use.any():
            pred[use] = ref + reg.predict(x)
        out[f"pred_{head}"] = pred
    return out


def predict_point(model: PointModel, features: dict) -> dict:
    """One live issue -> {'pred_arrival', 'pred_ride_mean', 'pred_ride_min', 'pred_ride_max'}."""
    if not _dict_usable(features):
        return {f"pred_{h}": float("nan") for h in model.heads}
    frame = _frame_from_dict(features, model.columns)
    x = frame[model.columns].to_numpy(float)
    ref = float(features["s_freshPm25"])
    return {f"pred_{h}": float(ref + reg.predict(x)[0]) for h, reg in model.heads.items()}


# ----------------------------------------------------------------------------- P(ride mean <= 70)
LE70_VARIANTS = ("L1_level_logistic", "E0_level_ratio", "E1_spatial_residual")
LE70_SIMPLICITY = ["L1_level_logistic", "E0_level_ratio", "E1_spatial_residual"]
FINAL_LE70_VARIANT = "E1_spatial_residual"  # the combination; adopted at the holdout confirmation step (see RESULTS.md)
RATIO_OFFSET = 5.0


@dataclass
class Le70Model:
    variant: str
    cutoff: int
    coef: float = float("nan")          # L1: logistic on standardised log(ref + 5)
    intercept: float = float("nan")
    mu: float = 0.0
    sd: float = 1.0
    z: np.ndarray | None = None          # E0/E1: sorted log((y + 5) / (anchor + 5)) of training rows
    point: PointModel | None = None      # E1: anchor = point model's ride-mean prediction
    n_train: int = 0


def _le70_rows(history, cutoff, origin_seconds=None):
    ok = training_rows(history, cutoff, complete=("y_ride_complete",), origin_seconds=origin_seconds)
    return ok & np.isfinite(history["y_ride_mean"].to_numpy(float))


def _block_oof_ride_mean(history, rows, columns, blocks=3):
    """Out-of-fold ride-mean predictions: contiguous day blocks inside the training window."""
    tr = history.loc[rows]
    days = (tr.index.to_numpy(np.int64) // 86400)
    uniq = np.unique(days)
    groups = np.array_split(uniq, blocks)
    pred = np.full(len(tr), np.nan)
    x = tr[columns].to_numpy(float)
    y = (tr["y_ride_mean"] - tr["s_freshPm25"]).to_numpy(float)
    for g in groups:
        hold = np.isin(days, g)
        if hold.all() or not hold.any():
            continue
        reg = HistGradientBoostingRegressor(**HGB_PARAMS).fit(x[~hold], y[~hold])
        pred[hold] = tr["s_freshPm25"].to_numpy(float)[hold] + reg.predict(x[hold])
    return tr, pred


def fit_le70(history: pd.DataFrame, cutoff: int, variant: str | None = None,
             point: PointModel | None = None) -> Le70Model:
    variant = FINAL_LE70_VARIANT if variant is None else variant
    if variant == "L1_level_logistic":
        ok = _le70_rows(history, cutoff)
        tr = history.loc[ok]
        x = np.log(tr["s_freshPm25"].to_numpy(float) + RATIO_OFFSET)
        y = (tr["y_ride_mean"].to_numpy(float) <= PLANNING_CUTOFF).astype(int)
        mu, sd = float(x.mean()), float(x.std() or 1.0)
        m = Le70Model(variant, int(cutoff), mu=mu, sd=sd, n_train=int(ok.sum()))
        if len(set(y)) < 2:
            m.intercept = math.inf if y[0] == 1 else -math.inf
            m.coef = 0.0
            return m
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(((x - mu) / sd)[:, None], y)
        m.coef, m.intercept = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return m
    if variant == "E0_level_ratio":
        ok = _le70_rows(history, cutoff)
        tr = history.loc[ok]
        z = np.log((tr["y_ride_mean"].to_numpy(float) + RATIO_OFFSET) / (tr["s_freshPm25"].to_numpy(float) + RATIO_OFFSET))
        return Le70Model(variant, int(cutoff), z=np.sort(z), n_train=int(ok.sum()))
    if variant == "E1_spatial_residual":
        if any(c not in history for c in CONTROLS):
            history = add_controls(history)
        point = fit_point(history, cutoff) if point is None else point
        ok = training_rows(history, cutoff, origin_seconds=ORIGIN_SECONDS) & np.isfinite(history["y_ride_mean"].to_numpy(float))
        tr, oof = _block_oof_ride_mean(history, ok, point.columns)
        good = np.isfinite(oof)
        anchor = np.maximum(oof[good], 0.0)
        z = np.log((tr["y_ride_mean"].to_numpy(float)[good] + RATIO_OFFSET) / (anchor + RATIO_OFFSET))
        return Le70Model(variant, int(cutoff), z=np.sort(z), point=point, n_train=int(good.sum()))
    raise ValueError(variant)


def _le70_from_anchor(model: Le70Model, anchor: np.ndarray) -> np.ndarray:
    thr = np.log((PLANNING_CUTOFF + RATIO_OFFSET) / (np.maximum(anchor, 0.0) + RATIO_OFFSET))
    return np.searchsorted(model.z, thr, side="right") / len(model.z)


def predict_le70_frame(model: Le70Model, frame: pd.DataFrame, point_pred: pd.DataFrame | None = None) -> pd.Series:
    use = usable(frame)
    p = np.full(len(frame), np.nan)
    if use.any():
        ref = frame.loc[use, "s_freshPm25"].to_numpy(float)
        if model.variant == "L1_level_logistic":
            if math.isinf(model.intercept):
                p[use] = 1.0 if model.intercept > 0 else 0.0
            else:
                x = (np.log(ref + RATIO_OFFSET) - model.mu) / model.sd
                p[use] = 1.0 / (1.0 + np.exp(-(model.intercept + model.coef * x)))
        elif model.variant == "E0_level_ratio":
            p[use] = _le70_from_anchor(model, ref)
        else:
            if point_pred is None:
                point_pred = predict_point_frame(model.point, frame)
            p[use] = _le70_from_anchor(model, point_pred["pred_ride_mean"].to_numpy(float)[use])
    return pd.Series(p, index=frame.index, name="p_ride_le70")


def predict_le70(model: Le70Model, features: dict) -> float:
    """One live issue -> P(ride mean over +90..+210 <= 70)."""
    if not _dict_usable(features):
        return float("nan")
    ref = float(features["s_freshPm25"])
    if model.variant == "L1_level_logistic":
        if math.isinf(model.intercept):
            return 1.0 if model.intercept > 0 else 0.0
        x = (math.log(ref + RATIO_OFFSET) - model.mu) / model.sd
        return float(1.0 / (1.0 + np.exp(-(model.intercept + model.coef * x))))
    if model.variant == "E0_level_ratio":
        return float(_le70_from_anchor(model, np.array([ref]))[0])
    anchor = predict_point(model.point, features)["pred_ride_mean"]
    return float(_le70_from_anchor(model, np.array([anchor]))[0])
