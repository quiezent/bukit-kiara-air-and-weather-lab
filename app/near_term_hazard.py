"""Experimental probability of a large near-term PM2.5 change.

The published point forecast and interval do not use this model. Fits use only
archived, as-issued dashboard features and targets fully observed before the
current local day's midnight. A fit is held fixed for that day.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from forecast_payload import loads as archive_loads
import math
from pathlib import Path
import sqlite3
import threading

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from collection_quality import bucket_coverage
from forecast_clock import TARGET_VERSION, point_weights, window_weights
from event_qualification import qualification_metadata


MODEL_VERSION = "near_term_absolute_change_logistic_v1_experimental"
CHANGE_THRESHOLD = 20.0
FLAG_THRESHOLD = 0.20
TRAIN_DAYS = 14
MIN_TRAINING = 100
MYT = timezone(timedelta(hours=8))
FEATURES = ("fresh", "change30", "change60", "fastSlowGap", "fineShare",
            "coarseParticles", "pm10ChangePct30", "rhum", "sinHour", "cosHour")
_FITS: OrderedDict[tuple[str, int], dict] = OrderedDict()
_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
TRAINING_CADENCE = "archived_halfhour_publications_with_120second_delay_guard"


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _features(air_window, current, fresh, target_epoch):
    mix = air_window.get("particleMix") or {}
    target = datetime.fromtimestamp(target_epoch, MYT)
    hour = target.hour + target.minute / 60
    values = {
        "fresh": fresh,
        "change30": _number(air_window.get("change30")),
        "change60": _number(air_window.get("change60")),
        "fastSlowGap": _number(air_window.get("fastSlowGap")),
        "fineShare": _number(mix.get("fineShare")),
        "coarseParticles": _number(mix.get("coarseParticles")),
        "pm10ChangePct30": _number(air_window.get("pm10ChangePct30")),
        "rhum": _number((current or {}).get("rhum")),
        "sinHour": math.sin(hour * math.pi / 12),
        "cosHour": math.cos(hour * math.pi / 12),
    }
    return np.array([values[name] for name in FEATURES], dtype=float)


def _clock_and_features(air_window, current, issue_epoch):
    clock = air_window.get("forecastClock") or {}
    issue = clock.get("forecastIssuedEpoch")
    anchor = clock.get("featureAnchorEpoch")
    target = clock.get("arrivalTargetEpoch")
    start = clock.get("windowStartEpoch")
    end = clock.get("windowEndEpoch")
    if clock.get("targetVersion") != TARGET_VERSION or not all(
            isinstance(x, (int, float)) for x in (issue, anchor, target, start, end)):
        return None
    if int(issue) != int(issue_epoch) or not (anchor <= issue < target == start < end):
        return None
    arrival = air_window.get("arrival") or {}
    trail = air_window.get("trail") or {}
    if arrival.get("pointRole") != "persistence_anchor" or trail.get("pointRole") != "persistence_anchor":
        return None
    fresh = _number(arrival.get("point"))
    if not math.isfinite(fresh) or abs(fresh - _number(trail.get("point"))) > 1e-6:
        return None
    point_complete = max(int(anchor + offset * 900) for offset in point_weights(target - anchor))
    mean_complete = int(clock.get("observedOutcomeCompleteAfterEpoch", math.ceil(end / 900) * 900))
    if mean_complete < end:
        return None
    return {
        "issue": int(issue), "anchor": int(anchor), "target": int(target),
        "start": int(start), "end": int(end), "fresh": fresh,
        "pointComplete": point_complete, "meanComplete": mean_complete,
        "features": _features(air_window, current, fresh, target),
    }


def _closed_buckets(conn, from_epoch, through_epoch):
    raw = pd.read_sql_query(
        "SELECT epoch,pm02 FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch",
        conn, params=(int(from_epoch), int(through_epoch)),
    )
    if raw.empty:
        return {}
    raw["pm02"] = pd.to_numeric(raw["pm02"], errors="coerce")
    raw.index = pd.to_datetime(raw.pop("epoch"), unit="s", utc=True)
    medians = raw["pm02"].resample("900s", label="right", closed="right").median()
    coverage = bucket_coverage(raw)
    medians.loc[~coverage["forecastEligible"]] = np.nan
    return {int(t.timestamp()): float(v) for t, v in medians.items() if pd.notna(v)}


def _actual(buckets, clock, horizon):
    anchor = clock["anchor"]
    if horizon == "arrival90":
        weights = point_weights(clock["target"] - anchor)
    else:
        weights = window_weights(clock["start"] - anchor, clock["end"] - anchor)
    parts = [(buckets.get(anchor + offset * 900), weight) for offset, weight in weights.items()]
    if any(value is None for value, _ in parts):
        return None
    return float(sum(value * weight for value, weight in parts))


def _training_rows(conn, midnight):
    start = midnight - TRAIN_DAYS * 86400
    buckets = _closed_buckets(conn, start, midnight)
    archives = conn.execute(
        "SELECT issued_epoch,payload FROM dashboard_forecast_issues "
        "WHERE issued_epoch>=? AND issued_epoch<? "
        "AND issued_epoch%3600 BETWEEN 1800 AND 1860 ORDER BY issued_epoch",
        (start, midnight),
    )
    rows = {"arrival90": [], "mean90to210": []}
    seen_slots = set()
    for archived, raw in archives:
        slot = archived // 3600
        if slot in seen_slots:
            continue
        seen_slots.add(slot)
        try:
            payload = archive_loads(raw)
        except (TypeError, ValueError):
            continue
        air_window = payload.get("airWindow") or {}
        arrival = air_window.get("arrival") or {}
        trail = air_window.get("trail") or {}
        if arrival.get("pointRole") != "persistence_anchor" or trail.get("pointRole") != "persistence_anchor":
            continue
        clock = air_window.get("forecastClock") or {}
        issued = clock.get("forecastIssuedEpoch")
        if not isinstance(issued, (int, float)) or not archived - 120 <= issued <= archived:
            continue
        case = _clock_and_features(air_window, payload.get("current"), int(issued))
        if case is None or not start <= case["issue"] < midnight:
            continue
        for horizon, completion in (("arrival90", case["pointComplete"]),
                                    ("mean90to210", case["meanComplete"])):
            if completion > midnight:
                continue
            actual = _actual(buckets, case, horizon)
            if actual is not None:
                rows[horizon].append((case["features"],
                                      int(abs(actual - case["fresh"]) >= CHANGE_THRESHOLD),
                                      case["issue"], completion))
    return rows


def _fit_one(rows):
    if len(rows) < MIN_TRAINING:
        return {"available": False, "reason": "insufficient_completed_training_issues",
                "trainingCount": len(rows)}
    x = np.stack([row[0] for row in rows])
    labels = np.array([row[1] for row in rows], dtype=int)
    active = np.isfinite(x).any(axis=0)
    x = x[:, active]
    prior = float((labels.sum() + 1) / (len(labels) + 2))
    model = None
    if len(set(labels)) >= 2:
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        model.fit(x, labels)
    return {"available": True, "model": model, "active": active, "prior": prior,
            "trainingCount": len(rows), "trainingEvents": int(labels.sum()),
            "lastTrainingIssueEpoch": max(row[2] for row in rows),
            "lastTrainingOutcomeCompleteEpoch": max(row[3] for row in rows)}


def _daily_fit(db_path, midnight):
    key = (str(Path(db_path).resolve()), midnight)
    with _LOCK:
        if key in _FITS:
            _FITS.move_to_end(key)
            return _FITS[key]
    # Training is background-only. Do not hold the small reader cache lock
    # during archive reads, reconstruction, or fitting.
    with _FIT_LOCK:
        with _LOCK:
            if key in _FITS:
                return _FITS[key]
        uri = Path(db_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=2)) as conn:
            conn.execute("PRAGMA query_only=ON")
            rows = _training_rows(conn, midnight)
        fits = {h: _fit_one(rows[h]) for h in rows}
        with _LOCK:
            _FITS[key] = fits
            while len(_FITS) > 4:
                _FITS.popitem(last=False)
        return fits


def refresh_model(db_path, issue_epoch):
    """Background-only daily fitting; forecast requests never call this."""
    day = datetime.fromtimestamp(int(issue_epoch), MYT).date()
    cutoff = int(datetime(day.year, day.month, day.day, tzinfo=MYT).timestamp())
    fits = _daily_fit(db_path, cutoff)
    return {"available": any(item["available"] for item in fits.values()),
            "reason": "cached_diagnostic_fit_ready", "modelVersion": MODEL_VERSION,
            "trainingCutoffEpoch": cutoff,
            "targets": {name: {key: value for key, value in item.items()
                               if key not in ("model", "active")}
                        for name, item in fits.items()}}


def _cached_daily_fit(db_path, cutoff):
    with _LOCK:
        return _FITS.get((str(Path(db_path).resolve()), cutoff))


def predict_hazard(db_path, air_window, issue_epoch, current=None):
    """Read cached diagnostic probabilities; no fitting or DB writes."""
    base = {"available": False, "modelVersion": MODEL_VERSION,
            "experimental": True, "prospectivelyValidated": False, "calibrated": False,
            "thresholdUgM3": CHANGE_THRESHOLD, "probabilityThreshold": FLAG_THRESHOLD,
            "definition": "P(abs(target PM2.5 - as-issued fresh PM2.5) >= 20 µg/m³)",
            "directionAvailable": False, "trainingWindowDays": TRAIN_DAYS,
            "targets": {}}
    base.update(qualification_metadata(issue_epoch, training_cadence=TRAINING_CADENCE))
    case = _clock_and_features(air_window or {}, current, issue_epoch)
    if case is None:
        return {**base, "reason": "unsupported_or_mismatched_forecast_clock_or_point"}
    day = datetime.fromtimestamp(case["issue"], MYT).date()
    midnight = int(datetime(day.year, day.month, day.day, tzinfo=MYT).timestamp())
    fits = _cached_daily_fit(db_path, midnight)
    if fits is None:
        return {**base, "reason": "background_model_collecting",
                "trainingCutoffEpoch": midnight}
    targets = {}
    for horizon, clock_fields in (
        ("arrival90", {"targetEpoch": case["target"], "completionEpoch": case["pointComplete"]}),
        ("mean90to210", {"startEpoch": case["start"], "endEpoch": case["end"],
                          "completionEpoch": case["meanComplete"]}),
    ):
        fit = fits[horizon]
        if not fit["available"]:
            targets[horizon] = {**clock_fields, "available": False,
                                "reason": fit["reason"], "trainingCount": fit["trainingCount"]}
            continue
        if fit["model"] is None:
            probability = fit["prior"]
        else:
            query = case["features"][fit["active"]].reshape(1, -1)
            probability = float(fit["model"].predict_proba(query)[0, 1])
        targets[horizon] = {**clock_fields, "available": True,
                            "probability": round(probability, 4),
                            "flag": False,
                            "diagnosticFlag": bool(probability >= FLAG_THRESHOLD),
                            "flagStatus": "unqualified_experimental",
                            "operationalUseEligible": False,
                            "trainingCount": fit["trainingCount"],
                            "trainingEvents": fit["trainingEvents"],
                            "lastTrainingIssueEpoch": fit["lastTrainingIssueEpoch"],
                            "lastTrainingOutcomeCompleteEpoch": fit["lastTrainingOutcomeCompleteEpoch"]}
    return {**base, "available": any(t["available"] for t in targets.values()),
            "sourceIssueEpoch": case["issue"], "featureAnchorEpoch": case["anchor"],
            "freshPm25UgM3": case["fresh"], "trainingCutoffEpoch": midnight,
            "trainedThroughIssueDay": (day - timedelta(days=1)).isoformat(),
            "targets": targets}
