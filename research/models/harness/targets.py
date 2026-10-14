"""Targets with the production definitions (functions reused where possible).

* 15-minute bucket medians: ``fresh_numeric_model_v4.observed_series`` (right-
  closed medians; a bucket counts only if collection_quality.bucket_coverage
  marks it forecastEligible: >=3 samples, both edges within 4 min, gaps <=8 min).
* +90 arrival / ride mean / min / max: ``fresh_numeric_model_v4.targets_for_issues``
  (interpolated 15-min median proxy at exact issue+90 min; overlap-weighted
  mean of buckets in exact [issue+90, issue+210]; min/max over buckets with
  positive overlap). Missing buckets leave the target NaN, never filled.
* Arrival change classes: delta = arrival - fresh 5-min reference, classes
  from ``arrival_change_model.label_classes`` (<=-40, (-40,-20], (-20,20),
  [20,40), >=40). Same target the arrival logistic is trained on.
* First crossing >=20 within 90 min: copy of ``fresh_event_model.prepare_training_data``
  label logic (six interpolated proxies at +15..+90; first |delta|>=20; sign).
* Ride mean <=70: ``cycling_window_probability.target_label`` on the ride mean.
* Sessions: production ``next_feasible_ride_period`` (Morning 09:00-11:00,
  Afternoon 14:00-16:00 MYT, first such window starting >= issue+90 min) and
  ``window_pm_predictor._session_actual`` (mean of the 8 closed bucket medians).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config, prod

FIRST_CROSSING_LEADS = (15, 30, 45, 60, 75, 90)


def bucket_series(rows) -> pd.Series:
    """Production observed_series over plain readings dicts (no cutoff)."""
    numeric = prod.module("fresh_numeric_model_v4")
    return numeric.observed_series(rows)


def bucket_map(series: pd.Series) -> dict:
    return {int(t): float(v) for t, v in zip(series.index.asi8 // 10**9, series.to_numpy(float))
            if np.isfinite(v)}


def ride_targets(series: pd.Series, issues) -> pd.DataFrame:
    """arrival, ride mean/min/max and their completion epochs (production function)."""
    numeric = prod.module("fresh_numeric_model_v4")
    frame, complete = numeric.targets_for_issues(series, np.asarray(issues, dtype=np.int64))
    frame = frame.rename(columns={"arrivalPoint": "y_arrival", "trailMeanPoint": "y_ride_mean",
                                  "trailMinimumPoint": "y_ride_min", "trailMaximumPoint": "y_ride_max"})
    # targets_for_issues blanks all four heads when any ride bucket is missing
    # (Ridge trains only on jointly complete rows). The arrival target itself
    # (the arrival-change models' label) needs only its own two buckets.
    clock = prod.module("forecast_clock")
    buckets = bucket_map(series)
    point_complete, arrival = [], []
    for issue in np.asarray(issues, dtype=np.int64):
        anchor = int(issue // 900 * 900)
        weights = clock.point_weights(int(issue - anchor + 5400))
        labels = [anchor + offset * 900 for offset in weights]
        point_complete.append(max(labels))
        arrival.append(sum(buckets[label] * w for label, w in zip(labels, weights.values()))
                       if all(label in buckets for label in labels) else np.nan)
    joint = frame["y_arrival"].to_numpy(float)
    arrival = np.asarray(arrival, dtype=float)
    both = np.isfinite(joint)
    if not np.allclose(joint[both], arrival[both], rtol=0, atol=1e-9):
        raise AssertionError("independent arrival target disagrees with production targets_for_issues")
    frame["y_arrival"] = arrival
    frame["y_ride_joint_valid"] = both
    frame["y_arrival_complete"] = np.asarray(point_complete, dtype=np.int64)
    frame["y_ride_complete"] = complete
    return frame


def arrival_classes(delta) -> np.ndarray:
    """Five production classes; NaN delta -> -1 (no label)."""
    change = prod.module("arrival_change_model")
    delta = np.asarray(delta, dtype=float)
    out = np.full(len(delta), -1, dtype=np.int64)
    finite = np.isfinite(delta)
    if finite.any():
        out[finite] = change.label_classes(delta[finite])
    return out


def first_crossing(issues, references, buckets: dict, threshold=20.0) -> pd.DataFrame:
    """Production first-crossing labels (fresh_event_model.prepare_training_data)."""
    clock = prod.module("forecast_clock")
    issues = np.asarray(issues, dtype=np.int64)
    cause = np.full(len(issues), np.nan)
    step = np.full(len(issues), np.nan)
    complete = np.zeros(len(issues), dtype=np.int64)
    for index, (issue, reference) in enumerate(zip(issues, references)):
        origin = int(issue) // 900 * 900
        values, completions = [], []
        for lead in FIRST_CROSSING_LEADS:
            weights = clock.point_weights(int(issue - origin + lead * 60))
            labels = [origin + offset * 900 for offset in weights]
            completions.append(max(labels))
            if any(label not in buckets for label in labels):
                values.append(np.nan)
            else:
                values.append(sum(buckets[label] * weight for label, weight in zip(labels, weights.values())))
        complete[index] = max(completions)
        if not np.isfinite(reference) or reference < 0 or not np.isfinite(values).all():
            continue
        delta = np.asarray(values) - reference
        crossed = np.flatnonzero(np.abs(delta) >= threshold)
        first = int(crossed[0]) if len(crossed) else -1
        cause[index] = int(np.sign(delta[first])) if first >= 0 else 0
        step[index] = first
    return pd.DataFrame({"y_first20": cause, "y_first20_step": step, "y_first20_complete": complete},
                        index=pd.Index(issues, name="issue_epoch"))


def next_session(issue: int, name: str) -> tuple[int, int]:
    """Production next_feasible_ride_period: first window start >= issue + 90 min."""
    start_hour, end_hour = config.SESSION_HOURS[name]
    local = config.myt(issue)
    for offset in range(4):
        day = (local + timedelta(days=offset)).date()
        start = int(datetime(day.year, day.month, day.day, start_hour, tzinfo=config.MYT).timestamp())
        if start >= issue + config.ARRIVAL_SECONDS:
            return start, start + 7200
    raise ValueError("no feasible session")


def session_mean(buckets: dict, start: int) -> float:
    """window_pm_predictor._session_actual: 8 closed buckets start+15..start+120."""
    values = [buckets.get(start + 900 * k) for k in range(1, 9)]
    if any(value is None for value in values):
        return np.nan
    return float(np.mean(values))


def session_targets(issues, buckets: dict) -> pd.DataFrame:
    records = []
    for issue in np.asarray(issues, dtype=np.int64):
        record = {}
        for name in ("morning", "afternoon"):
            start, end = next_session(int(issue), name)
            record[f"y_{name}_start"] = start
            record[f"y_{name}_mean"] = session_mean(buckets, start)
            record[f"y_{name}_complete"] = end
            record[f"y_{name}_lead_min"] = (start - int(issue)) / 60
            record[f"y_{name}_same_day"] = config.myt(start).date() == config.myt(int(issue)).date()
        records.append(record)
    return pd.DataFrame(records, index=pd.Index(np.asarray(issues, dtype=np.int64), name="issue_epoch"))


def canonical_label(issue: int) -> str:
    local = config.myt(issue)
    return local.strftime("%H:%M") if local.strftime("%H:%M") in config.CANONICAL_CLOCKS and local.second == 0 else ""


def build(issues, references, rows) -> pd.DataFrame:
    """All targets for the given issue clocks and their fresh references."""
    issues = np.asarray(issues, dtype=np.int64)
    references = np.asarray(references, dtype=float)
    series = bucket_series(rows)
    buckets = bucket_map(series)
    frame = ride_targets(series, issues)
    frame.index = pd.Index(issues, name="issue_epoch")
    frame["y_ref"] = references
    frame["y_delta"] = frame.y_arrival - references
    frame["y_class5"] = arrival_classes(frame.y_delta.to_numpy())
    label = frame.y_class5 >= 0
    for name, condition in (("y_fall20", frame.y_class5 <= 1), ("y_rise20", frame.y_class5 >= 3),
                            ("y_fall40", frame.y_class5 == 0), ("y_rise40", frame.y_class5 == 4)):
        frame[name] = condition.where(label).astype("float")
    frame["y_ride_le70"] = (frame.y_ride_mean <= config.PLANNING_CUTOFF).where(frame.y_ride_mean.notna()).astype("float")
    frame = frame.join(first_crossing(issues, references, buckets))
    frame = frame.join(session_targets(issues, buckets))
    return frame
