"""Experimental 15- and 30-minute PM2.5 change probabilities.

Historical features are reconstructed at archived, quarter-hour issue clocks.
Only sensor observations at or before each issue enter features. Targets are
interpolated proxies from strictly covered, right-labelled 15-minute medians.
The daily fit is frozen at local midnight, and this module never writes the DB
or changes a published PM2.5 point or interval.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sqlite3
import threading

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from collection_quality import bucket_coverage
from forecast_clock import point_weights
from event_qualification import qualification_metadata


MODEL_VERSION = "short_horizon_direction_logistic_v1_experimental"
TARGET_VERSION = "interpolated_covered_15min_median_at_exact_issue_lead_v1"
CHANGE_THRESHOLD = 10.0
LARGE_EVENT_THRESHOLD = 20.0
HORIZONS_MINUTES = (15, 30)
TRAIN_DAYS = 21
MIN_TRAINING = 150
MIN_DIRECTION_EVENTS = 12
MIN_NON_EVENTS = 50
BUCKET_SECONDS = 900
MYT = timezone(timedelta(hours=8))
_FITS: OrderedDict[tuple[str, int], dict] = OrderedDict()
_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
TRAINING_CADENCE = "archived_final120seconds_quarterhour_publications"


def _raw_arrays(rows, as_of_epoch):
    """Sort and deduplicate finite PM readings without crossing the issue."""
    finite = {}
    for row in rows:
        try:
            epoch = int(row["epoch"])
            pm = float(row["pm02"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if epoch <= as_of_epoch and math.isfinite(pm) and pm >= 0:
            finite[epoch] = pm
    epochs = np.array(sorted(finite), dtype=np.int64)
    values = np.array([finite[int(epoch)] for epoch in epochs], dtype=float)
    return epochs, values


def _eligible_medians(epochs, values):
    """Apply the shared forecast bucket rule to every completed sensor bucket."""
    if not len(epochs):
        return {}
    index = pd.to_datetime(epochs, unit="s", utc=True)
    raw = pd.DataFrame({"pm02": values}, index=index)
    coverage = bucket_coverage(raw, bucket_seconds=BUCKET_SECONDS)
    medians = raw["pm02"].resample("900s", label="right", closed="right").median()
    medians = medians.where(coverage["forecastEligible"].reindex(medians.index).fillna(False))
    return {int(stamp.timestamp()): float(value)
            for stamp, value in medians.items() if pd.notna(value)}


def _fresh_reference(epochs, values, issue_epoch, latest_known_epoch=None):
    """A current 5-minute observed median, never extrapolated across a gap."""
    lo = np.searchsorted(epochs, issue_epoch - 300, side="right")
    hi = np.searchsorted(epochs, min(issue_epoch, latest_known_epoch)
                       if latest_known_epoch is not None else issue_epoch, side="right")
    if hi <= lo or issue_epoch - int(epochs[hi - 1]) > 240:
        return None
    return float(np.median(values[lo:hi]))


def _features_at(issue_epoch, epochs, values, medians, latest_known_epoch=None):
    """Return features whose latest possible source is the issue itself."""
    anchor = issue_epoch // BUCKET_SECONDS * BUCKET_SECONDS
    if latest_known_epoch is not None and latest_known_epoch < anchor:
        return None
    closed = [medians.get(anchor - lag * BUCKET_SECONDS) for lag in (0, 1, 2, 4)]
    if any(value is None for value in closed):
        return None
    fresh = _fresh_reference(epochs, values, issue_epoch, latest_known_epoch)
    if fresh is None:
        return None
    current, lag15, lag30, lag60 = closed
    local = datetime.fromtimestamp(issue_epoch, MYT)
    hour = local.hour + local.minute / 60 + local.second / 3600
    features = np.array([
        fresh, fresh - current, current - lag15, current - lag30,
        current - lag60, abs(current - lag15),
        math.sin(math.pi * hour / 12), math.cos(math.pi * hour / 12),
    ], dtype=float)
    return {"features": features, "reference": fresh, "anchor": anchor}


def _target_clock(issue_epoch, horizon_minutes):
    target_epoch = int(issue_epoch + horizon_minutes * 60)
    anchor = issue_epoch // BUCKET_SECONDS * BUCKET_SECONDS
    weights = point_weights(target_epoch - anchor, BUCKET_SECONDS)
    labels = {anchor + offset * BUCKET_SECONDS: weight
              for offset, weight in weights.items()}
    return target_epoch, max(labels), labels


def _target_at(issue_epoch, horizon_minutes, medians, as_of_epoch):
    """Require every touched target bucket to be covered and fully closed."""
    target, complete, labels = _target_clock(issue_epoch, horizon_minutes)
    if complete > as_of_epoch:
        return None
    parts = [(medians.get(label), weight) for label, weight in labels.items()]
    if any(value is None for value, _ in parts):
        return None
    return float(sum(value * weight for value, weight in parts))


def _archived_quarter_issues(conn, start_epoch, cutoff_epoch):
    """Select the archived published clock, not the later DB write timestamp."""
    archives = conn.execute(
        "SELECT issued_epoch,forecast_epoch,sensor_epoch FROM dashboard_forecast_issues "
        "WHERE issued_epoch>=? AND issued_epoch<? AND issued_epoch%900>=780 "
        "ORDER BY issued_epoch",
        (int(start_epoch), int(cutoff_epoch)),
    )
    slots = {}
    for archived, published, sensor_epoch in archives:
        try:
            issue = int(published)
            sensor_epoch = int(sensor_epoch)
        except (TypeError, ValueError, KeyError, OverflowError):
            continue
        if (start_epoch <= issue < cutoff_epoch and 0 <= archived - issue <= 120
                and sensor_epoch <= issue):
            slots[int(archived) // 900] = (issue, sensor_epoch)
    return list(slots.values())


def _training_rows(conn, cutoff_epoch):
    start = cutoff_epoch - TRAIN_DAYS * 86400
    raw = conn.execute(
        "SELECT epoch,pm02 FROM readings WHERE epoch>=? AND epoch<? ORDER BY epoch",
        (start - 3600 - BUCKET_SECONDS, cutoff_epoch),
    ).fetchall()
    epochs, values = _raw_arrays(({"epoch": epoch, "pm02": pm} for epoch, pm in raw), cutoff_epoch - 1)
    medians = _eligible_medians(epochs, values)
    rows = {minutes: [] for minutes in HORIZONS_MINUTES}
    for issue, sensor_epoch in _archived_quarter_issues(conn, start, cutoff_epoch):
        case = _features_at(issue, epochs, values, medians, sensor_epoch)
        if case is None:
            continue
        for minutes in HORIZONS_MINUTES:
            actual = _target_at(issue, minutes, medians, cutoff_epoch)
            if actual is None:
                continue
            target, complete, _ = _target_clock(issue, minutes)
            delta = actual - case["reference"]
            rows[minutes].append((case["features"], delta, issue, complete))
    return rows


def _fit_one(rows):
    count = len(rows)
    if count < MIN_TRAINING:
        return {"available": False, "reason": "insufficient_completed_training_issues",
                "trainingCount": count}
    x = np.stack([row[0] for row in rows])
    deltas = np.array([row[1] for row in rows], dtype=float)
    labels = np.select((deltas <= -CHANGE_THRESHOLD, deltas >= CHANGE_THRESHOLD),
                       (-1, 1), default=0)
    fall_events = int(np.sum(labels == -1))
    rise_events = int(np.sum(labels == 1))
    stable_events = int(np.sum(labels == 0))
    support = {"trainingCount": count, "riseEvents": rise_events,
               "fallEvents": fall_events, "stableEvents": stable_events,
               "largeRiseEvents": int(np.sum(deltas >= LARGE_EVENT_THRESHOLD)),
               "largeFallEvents": int(np.sum(deltas <= -LARGE_EVENT_THRESHOLD)),
               "lastTrainingIssueEpoch": max(row[2] for row in rows),
               "lastTrainingOutcomeCompleteEpoch": max(row[3] for row in rows)}
    if (rise_events < MIN_DIRECTION_EVENTS or fall_events < MIN_DIRECTION_EVENTS
            or stable_events < MIN_NON_EVENTS):
        return {**support, "available": False,
                "reason": "insufficient_direction_events"}
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    model.fit(x, labels)
    return {**support, "available": True, "model": model}


def _daily_fit(db_path, cutoff_epoch):
    key = (str(Path(db_path).resolve()), cutoff_epoch)
    with _LOCK:
        if key in _FITS:
            _FITS.move_to_end(key)
            return _FITS[key]
    with _FIT_LOCK:
        with _LOCK:
            if key in _FITS:
                return _FITS[key]
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=2)) as conn:
            conn.execute("PRAGMA query_only=ON")
            rows = _training_rows(conn, cutoff_epoch)
        fits = {minutes: _fit_one(rows[minutes]) for minutes in HORIZONS_MINUTES}
        with _LOCK:
            _FITS[key] = fits
            while len(_FITS) > 4:
                _FITS.popitem(last=False)
        return fits


def refresh_model(db_path, issue_epoch):
    """Background-only daily fit, separate from forecast publication."""
    day = datetime.fromtimestamp(int(issue_epoch), MYT).date()
    cutoff = int(datetime(day.year, day.month, day.day, tzinfo=MYT).timestamp())
    fits = _daily_fit(db_path, cutoff)
    return {"available": any(item["available"] for item in fits.values()),
            "reason": "cached_diagnostic_fit_ready", "modelVersion": MODEL_VERSION,
            "trainingCutoffEpoch": cutoff,
            "targets": {str(name): {key: value for key, value in item.items() if key != "model"}
                        for name, item in fits.items()}}


def _cached_daily_fit(db_path, cutoff):
    with _LOCK:
        return _FITS.get((str(Path(db_path).resolve()), cutoff))


def predict_short_horizon(db_path, rows, issue_epoch):
    """Report experimental rise/fall probabilities or a reason for absence.

    Historical fits stop at the current Malaysia-day midnight.  The current
    issue's features are computed from caller-provided readings only through
    ``issue_epoch``. This method has no write side effects.
    """
    issue_epoch = int(issue_epoch)
    day = datetime.fromtimestamp(issue_epoch, MYT).date()
    cutoff = int(datetime(day.year, day.month, day.day, tzinfo=MYT).timestamp())
    base = {
        "available": False, "experimental": True, "prospectivelyValidated": False,
        "calibrated": False, "appliedToPrimaryForecast": False,
        "modelVersion": MODEL_VERSION, "targetVersion": TARGET_VERSION,
        "thresholdUgM3": CHANGE_THRESHOLD,
        "largeEventThresholdUgM3": LARGE_EVENT_THRESHOLD,
        "targetDefinition": "Interpolated strictly covered 15-minute PM2.5 bucket medians at exact issue-time lead",
        "referenceDefinition": "Median observed PM2.5 in the five minutes ending at issue; latest reading at most four minutes old",
        "sourceIssueEpoch": issue_epoch, "trainingCutoffEpoch": cutoff,
        "trainingWindowDays": TRAIN_DAYS, "targets": {},
    }
    base.update(qualification_metadata(issue_epoch, training_cadence=TRAINING_CADENCE))
    epochs, values = _raw_arrays(rows, issue_epoch)
    case = _features_at(issue_epoch, epochs, values, _eligible_medians(epochs, values))
    for minutes in HORIZONS_MINUTES:
        target, complete, _ = _target_clock(issue_epoch, minutes)
        base["targets"][f"minutes{minutes}"] = {
            "available": False, "horizonMinutes": minutes, "targetEpoch": target,
            "observedOutcomeCompleteAfterEpoch": complete,
            "operationalUseEligible": False,
        }
    if case is None:
        return {**base, "reason": "insufficient_current_sensor_bucket_coverage"}
    base["referencePm25UgM3"] = round(case["reference"], 3)
    base["featureAnchorEpoch"] = case["anchor"]
    fits = _cached_daily_fit(db_path, cutoff)
    if fits is None:
        return {**base, "reason": "background_model_collecting"}
    for minutes in HORIZONS_MINUTES:
        fit = fits[minutes]
        target_result = base["targets"][f"minutes{minutes}"]
        target_result["trainingCount"] = fit["trainingCount"]
        for name in ("riseEvents", "fallEvents", "stableEvents",
                     "largeRiseEvents", "largeFallEvents"):
            if name in fit:
                target_result[name] = fit[name]
        target_result["largeEventProbabilityAvailable"] = False
        target_result["largeEventReason"] = "large_event_classifier_not_fitted_for_this_head"
        if not fit["available"]:
            target_result["reason"] = fit["reason"]
            continue
        x = case["features"].reshape(1, -1)
        probabilities = fit["model"].predict_proba(x)[0]
        by_class = dict(zip(fit["model"].classes_, probabilities))
        target_result.update({
            "available": True,
            "riseProbability": round(float(by_class[1]), 4),
            "fallProbability": round(float(by_class[-1]), 4),
            "stableProbability": round(float(by_class[0]), 4),
            "lastTrainingIssueEpoch": fit["lastTrainingIssueEpoch"],
            "lastTrainingOutcomeCompleteEpoch": fit["lastTrainingOutcomeCompleteEpoch"],
        })
    base["available"] = any(t["available"] for t in base["targets"].values())
    if not base["available"]:
        base["reason"] = "insufficient_completed_event_examples"
    return base
