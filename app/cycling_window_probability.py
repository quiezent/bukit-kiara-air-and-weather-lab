"""Experimental probability of a personal PM2.5 ride-window planning cutoff.

Predict the clock-weighted mean over an explicit two-hour window, never the
instantaneous minimum, a first crossing, or a medical safety judgment. All model
fitting is an explicit offline operation; importing this module does not fit,
read a database, call a provider, or write a file.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION = "ttdi_ride_mean_le70_logistic_v1"
PLANNING_CUTOFF = 70.0
TIMEZONE = "Asia/Kuala_Lumpur"
LOCAL_FEATURES = (
    "fresh", "freshOffset", "delta15", "delta30", "delta60", "delta120",
    "sd60", "mean3Offset", "temp", "rh", "tempDelta30", "rhDelta30",
)
SENSOR_FEATURES = (*LOCAL_FEATURES, "sinTarget", "cosTarget",
                   "targetLeadHours", "targetDurationHours")
WEATHER_FEATURES = (
    "weatherAgeHours", "priorRainMm", "duringRainMm", "priorRainProbMean",
    "duringRainProbMean", "firstRainLeadHours", "lastRainEndLeadHours",
    "wind10UMeanKmh", "wind10VMeanKmh", "wind10UDeltaKmh", "wind10VDeltaKmh",
    "gustHourlyEnvelopeMaxKmh", "gustHourlyEnvelopeDeltaKmh",
    "ventilationMeanM2S", "ventilationDeltaM2S", "boundaryLayerMeanM",
    "targetTempDeltaC", "targetRhDeltaPct",
)
FEATURES = (*SENSOR_FEATURES, *WEATHER_FEATURES)
FEATURE_SETS = {"sensor": SENSOR_FEATURES, "weather": FEATURES}
LOGISTIC_PARAMETERS = dict(C=0.1, solver="lbfgs", max_iter=1000, tol=1e-4,
                           class_weight=None, random_state=1749)


def feature_clock_valid(frame):
    """Original archived issue availability; no future sensor/weather inputs."""
    return (frame.featureSourceMaxEpoch.le(frame.sensorWatermarkEpoch) &
            frame.freshReferenceEpoch.le(frame.sensorWatermarkEpoch) &
            frame.sensorWatermarkEpoch.le(frame.issueEpoch) &
            (frame.issueEpoch-frame.freshReferenceEpoch).between(0, 240) &
            frame.publicationAgeSeconds.between(0, 120) &
            frame.weatherFetchedEpoch.le(frame.issueEpoch) &
            (frame.issueEpoch-frame.weatherFetchedEpoch).between(0, 7200))


def target_label(actual_mean):
    """Reject missing labels rather than silently converting them to failure."""
    values = np.asarray(actual_mean, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Every target mean must be finite and nonnegative")
    return (values <= PLANNING_CUTOFF).astype(np.int64)


def feature_frame(frame, candidate="weather"):
    names = FEATURE_SETS[candidate]
    out = frame.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def fit_window_model(training, candidate="weather", *, cutoff_epoch):
    """Fit one fixed candidate using completed spaced targets before midnight.

    Caller supplies the declared four-hour training cohort. This function also
    enforces 28 days, 50 cases, 10 completed target dates, and both classes. The
    returned bundle preserves the exact feature order and daily fit cutoff.
    """
    cutoff_epoch = int(cutoff_epoch)
    stamp = pd.Timestamp(cutoff_epoch, unit="s", tz="UTC").tz_convert(TIMEZONE)
    if stamp != stamp.normalize():
        raise ValueError("Training cutoff must be Malaysia midnight")
    if not feature_clock_valid(training).all():
        raise ValueError("Training feature clocks are invalid")
    if not (training.completeEpoch.lt(cutoff_epoch) &
            training.issueEpoch.ge(cutoff_epoch-28*86400)).all():
        raise ValueError("Training outcomes cross the daily cutoff or lookback")
    if len(training) < 50 or training.targetDay.nunique() < 10:
        raise ValueError("Insufficient training support")
    if training.issueEpoch.duplicated().any():
        raise ValueError("Duplicate training issues")
    labels = target_label(training.actual)
    if len(np.unique(labels)) != 2:
        raise ValueError("Training requires both below-cutoff and above-cutoff windows")
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                       keep_empty_features=True),
                          StandardScaler(), LogisticRegression(**LOGISTIC_PARAMETERS))
    with threadpool_limits(limits=1):
        model.fit(feature_frame(training, candidate), labels)
    return {
        "pipeline": model, "candidate": candidate, "modelVersion": VERSION,
        "featureNames": list(FEATURE_SETS[candidate]), "planningCutoffUgM3": PLANNING_CUTOFF,
        "trainingCutoffEpoch": cutoff_epoch, "trainingRows": len(training),
        "trainingTargetDates": int(training.targetDay.nunique()),
        "trainingSuccesses": int(labels.sum()), "trainingFailures": int((1-labels).sum()),
        "fittedAtUtc": datetime.now(timezone.utc).isoformat(), "calibrated": False,
    }


def predict_window_probability(bundle, frame):
    """Return P(mean <=70); target bounds and issue identity stay with caller."""
    if bundle["modelVersion"] != VERSION:
        raise ValueError("Unexpected model version")
    if bundle["featureNames"] != list(FEATURE_SETS[bundle["candidate"]]):
        raise ValueError("Feature contract mismatch")
    if not feature_clock_valid(frame).all():
        raise ValueError("Prediction feature clocks are invalid")
    if not frame.issueEpoch.ge(bundle["trainingCutoffEpoch"]).all():
        raise ValueError("Model trained after query issue")
    model = bundle["pipeline"]
    index = list(model.classes_).index(1)
    with threadpool_limits(limits=1):
        result = model.predict_proba(feature_frame(frame, bundle["candidate"]))[:, index]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Invalid predicted probabilities")
    return result
