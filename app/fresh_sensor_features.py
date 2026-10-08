"""Shared, causal issue-time sensor features for newly versioned PM learners.

There are no outcomes, fits, imputers, provider reads or database writes here.
Training and inference use the same functions and the same issue-relative bins.
Known receipts after an issue are excluded. Rows whose receipt history is
unknown remain explicitly identified in provenance, rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd


VERSION = "causal_fresh_sensor_3min_sequence_v1"
LOOKBACK_MINUTES = 180
SEQUENCE_BIN_MINUTES = 3
FRESH_WINDOW_SECONDS = 300
MAX_FRESH_AGE_SECONDS = 240
LAGS_MINUTES = (5, 10, 15, 30, 60)
CHANNELS = ("pm25", "temperature", "humidity")
SOURCE_FIELDS = ("pm02", "atmp", "rhum")
BASE_FEATURE_COLUMNS = (
    "freshPm25", *(f"rawLag{m}" for m in LAGS_MINUTES),
    *(f"rawDelta{m}" for m in LAGS_MINUTES),
    "rawSd15", "rawSd30", "rawSd60", "rawRange15", "rawRange30",
    "rawSlope15PerMinute", "rawSlope30PerMinute",
    "freshTemperature", "freshHumidity", "rawTemperatureDelta30",
    "rawHumidityDelta30", "freshSampleCount", "rawSampleCount30",
    "timeSin", "timeCos", "issueOffsetSin", "issueOffsetCos",
)
SEQUENCE_FEATURE_COLUMNS = tuple(
    f"seq_{channel}_{age}_{age + SEQUENCE_BIN_MINUTES}min"
    for channel in CHANNELS
    for age in range(LOOKBACK_MINUTES - SEQUENCE_BIN_MINUTES, -1, -SEQUENCE_BIN_MINUTES)
)
FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *SEQUENCE_FEATURE_COLUMNS)


@dataclass(frozen=True)
class PreparedSensorRows:
    epochs: np.ndarray
    values: np.ndarray
    availability: np.ndarray
    receipt_known: np.ndarray


def _number(value):
    if isinstance(value, bool):
        return math.nan
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _issue_epoch(value):
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            raise ValueError("Issue timestamps require a timezone")
        value = value.timestamp()
    value = _number(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Issue epoch must be finite and nonnegative")
    return value


def _receipt(row):
    provenance = row.get("_provenance") or row.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    sources = (row, provenance)
    available = math.nan
    known = False
    explicitly_known = any(source.get("receiptKnown") is True or source.get("receipt_known") is True
                           for source in sources)
    receipt_times = {}
    invalid_confirmation = False
    for source in sources:
        if source.get("receiptKnown") is True or source.get("receipt_known") is True:
            known = True
        for key in ("availableEpoch", "available_epoch"):
            if key in source:
                number = _number(source[key])
                # Both sources are constraints; a later one cannot be ignored.
                if math.isfinite(number):
                    known = True
                    available = number if not math.isfinite(available) else max(available, number)
                elif explicitly_known:
                    invalid_confirmation = True
        for key in ("receivedEpoch", "received_epoch", "persistedEpoch", "persisted_epoch",
                    "confirmedEpoch", "confirmed_epoch"):
            if key in source:
                number = _number(source[key])
                if math.isfinite(number):
                    known = True
                    canonical = key.replace("_epoch", "").replace("Epoch", "")
                    receipt_times[canonical] = max(receipt_times.get(canonical, -math.inf), number)
                elif key.startswith("confirmed") and (explicitly_known or known):
                    invalid_confirmation = True
    if not math.isfinite(available) and "received" in receipt_times and "persisted" in receipt_times:
        available = max(receipt_times["received"], receipt_times["persisted"])
    if math.isfinite(available) and receipt_times:
        available = max(available, *receipt_times.values())
    if invalid_confirmation:
        available = math.nan
    return known, available


def prepare_rows(rows):
    """Sort once; retain revisions so each issue selects its known revision."""
    if isinstance(rows, PreparedSensorRows):
        return rows
    records = []
    for original in rows:
        row = dict(original)
        observed = _number(row.get("epoch"))
        if not math.isfinite(observed) or observed < 0:
            continue
        known, available = _receipt(row)
        values = [_number(row.get(name)) for name in SOURCE_FIELDS]
        if math.isfinite(values[0]) and values[0] < 0:
            values[0] = math.nan
        records.append((observed, available, known, values))
    # Unknown availability precedes known revisions at one measurement epoch.
    records.sort(key=lambda r: (r[0], r[1] if math.isfinite(r[1]) else -math.inf))
    epochs = np.asarray([r[0] for r in records], dtype=float)
    values = np.asarray([r[3] for r in records], dtype=float).reshape(-1, len(CHANNELS))
    availability = np.asarray([r[1] for r in records], dtype=float)
    known = np.asarray([r[2] for r in records], dtype=bool)
    for array in (epochs, values, availability, known):
        array.setflags(write=False)
    return PreparedSensorRows(epochs, values, availability, known)


def _bounded(prepared, issue, lookback_minutes=LOOKBACK_MINUTES):
    # Search only the required interval, never scan all training history per query.
    lo = np.searchsorted(prepared.epochs, issue - lookback_minutes * 60, side="right")
    hi = np.searchsorted(prepared.epochs, issue, side="right")
    epochs, values = prepared.epochs[lo:hi], prepared.values[lo:hi]
    availability, known = prepared.availability[lo:hi], prepared.receipt_known[lo:hi]
    allowed = ~known | (np.isfinite(availability) & (availability <= issue))
    epochs, values, known, availability = epochs[allowed], values[allowed], known[allowed], availability[allowed]
    if len(epochs):
        # Latest available revision at each observation time wins, without a
        # future revision displacing the older issue's still-known value.
        _, reversed_positions = np.unique(epochs[::-1], return_index=True)
        positions = np.sort(len(epochs) - 1 - reversed_positions)
        epochs, values = epochs[positions], values[positions]
        known, availability = known[positions], availability[positions]
    return epochs, values, known, availability


def _median(values):
    finite = values[np.isfinite(values)]
    if len(finite) == 1:
        return float(finite[0])
    if len(finite) == 2:
        return float((finite[0] + finite[1]) / 2)
    return float(np.median(finite)) if len(finite) else math.nan


def _reference(epochs, values, end, column=0):
    mask = (epochs > end - FRESH_WINDOW_SECONDS) & (epochs <= end) & np.isfinite(values[:, column])
    if not mask.any() or end - epochs[mask][-1] > MAX_FRESH_AGE_SECONDS:
        return math.nan
    return _median(values[mask, column])


def _sequence_values(epochs, values, issue, lookback_minutes, bin_minutes):
    if (not isinstance(lookback_minutes, int) or not isinstance(bin_minutes, int)
            or lookback_minutes <= 0 or bin_minutes <= 0 or lookback_minutes % bin_minutes):
        raise ValueError("Sequence duration must be a positive multiple of its bin duration")
    count = lookback_minutes // bin_minutes
    ends = issue - np.arange(count - 1, -1, -1, dtype=float) * bin_minutes * 60
    matrix = np.full((count, len(CHANNELS)), np.nan)
    sample_counts = np.zeros(count, dtype=int)
    indices = np.searchsorted(ends, epochs, side="left")
    valid = (epochs > issue - lookback_minutes * 60) & (epochs <= issue) & (indices < count)
    indices, values = indices[valid], values[valid]
    for index in np.unique(indices):
        subset = values[indices == index]
        sample_counts[index] = int(np.isfinite(subset[:, 0]).sum())
        if len(subset) == 1:
            matrix[index] = subset[0]
            continue
        for column in range(len(CHANNELS)):
            matrix[index, column] = _median(subset[:, column])
    return ends, matrix, sample_counts


def sequence_bins(rows, issue_epoch, *, lookback_minutes=LOOKBACK_MINUTES,
                  bin_minutes=SEQUENCE_BIN_MINUTES):
    """Right-closed bins ending at issue; missing bins remain NaN, never filled."""
    issue = _issue_epoch(issue_epoch)
    prepared = prepare_rows(rows)
    epochs, values, _, _ = _bounded(prepared, issue, lookback_minutes)
    ends, matrix, counts = _sequence_values(epochs, values, issue, lookback_minutes, bin_minutes)
    frame = pd.DataFrame(matrix, index=pd.Index(ends, name="binEndEpoch"), columns=CHANNELS)
    frame["sampleCount"] = counts
    return frame


def _features(epochs, values, issue):
    fresh = _reference(epochs, values, issue)
    result = {"freshPm25": fresh}
    for minutes in LAGS_MINUTES:
        lag = _reference(epochs, values, issue - minutes * 60)
        result[f"rawLag{minutes}"] = lag
        result[f"rawDelta{minutes}"] = fresh - lag
    for minutes in (15, 30, 60):
        mask = (epochs > issue - minutes * 60) & np.isfinite(values[:, 0])
        observed = values[mask, 0]
        result[f"rawSd{minutes}"] = float(np.std(observed, ddof=0)) if len(observed) >= 3 else math.nan
        if minutes in (15, 30):
            result[f"rawRange{minutes}"] = float(np.ptp(observed)) if len(observed) >= 2 else math.nan
            x = (epochs[mask] - issue) / 60
            centered = x - x.mean() if len(x) else x
            denominator = float(centered @ centered)
            result[f"rawSlope{minutes}PerMinute"] = (float(centered @ (observed - observed.mean()) / denominator)
                                                       if len(observed) >= 3 and denominator > 0 else math.nan)
    result["freshTemperature"] = _reference(epochs, values, issue, 1)
    result["freshHumidity"] = _reference(epochs, values, issue, 2)
    result["rawTemperatureDelta30"] = result["freshTemperature"] - _reference(epochs, values, issue - 1800, 1)
    result["rawHumidityDelta30"] = result["freshHumidity"] - _reference(epochs, values, issue - 1800, 2)
    result["freshSampleCount"] = float(((epochs > issue - 300) & np.isfinite(values[:, 0])).sum())
    result["rawSampleCount30"] = float(((epochs > issue - 1800) & np.isfinite(values[:, 0])).sum())
    # Kuala Lumpur has fixed UTC+8; no machine-local timezone is consulted.
    local_day_seconds = (issue + 8 * 3600) % 86400
    result["timeSin"] = math.sin(2 * math.pi * local_day_seconds / 86400)
    result["timeCos"] = math.cos(2 * math.pi * local_day_seconds / 86400)
    offset = issue % 900
    result["issueOffsetSin"] = math.sin(2 * math.pi * offset / 900)
    result["issueOffsetCos"] = math.cos(2 * math.pi * offset / 900)
    _, sequence, _ = _sequence_values(epochs, values, issue, LOOKBACK_MINUTES, SEQUENCE_BIN_MINUTES)
    result.update(zip(SEQUENCE_FEATURE_COLUMNS, sequence.T.ravel().tolist()))
    return {name: result[name] for name in FEATURE_COLUMNS}


def issue_features(rows, issue_epoch):
    """Numeric features only, with identical semantics for fitting and serving."""
    issue = _issue_epoch(issue_epoch)
    epochs, values, _, _ = _bounded(prepare_rows(rows), issue)
    return _features(epochs, values, issue)


def _metadata(epochs, values, known, availability, issue):
    mask = (epochs > issue - 300) & np.isfinite(values[:, 0])
    reference_epoch = float(epochs[mask][-1]) if mask.any() else None
    available = reference_epoch is not None and issue - reference_epoch <= MAX_FRESH_AGE_SECONDS
    has_unknown = bool((~known).any())
    digest = hashlib.sha256()
    digest.update(np.asarray(epochs, dtype="<f8").tobytes())
    canonical = np.asarray(values, dtype="<f8").copy()
    canonical[np.isnan(canonical)] = np.nan
    digest.update(canonical.tobytes())
    digest.update(np.asarray(known, dtype=np.uint8).tobytes())
    canonical_availability = np.asarray(availability, dtype="<f8").copy()
    canonical_availability[np.isnan(canonical_availability)] = np.nan
    digest.update(canonical_availability.tobytes())
    return {"available": bool(available),
            "reason": None if available else "fresh_sensor_reference_unavailable_or_stale",
            "featureVersion": VERSION, "forecastIssuedEpoch": issue,
            "featureSourceMaxEpoch": float(epochs[-1]) if len(epochs) else None,
            "freshReferenceEpoch": reference_epoch,
            "freshReferenceAgeSeconds": issue - reference_epoch if reference_epoch is not None else None,
            "freshReferencePm25": _reference(epochs, values, issue) if available else None,
            "inputRowCount": len(epochs), "knownReceiptRows": int(known.sum()),
            "unknownReceiptRows": int((~known).sum()), "allInputReceiptsKnown": not has_unknown,
            "availabilityPolicy": "known_receipts_at_or_before_issue_legacy_unknown_explicit",
            "legacyReceiptCaveat": ("Unknown historical receipt times cannot establish original live availability"
                                    if has_unknown else None),
            "inputValuesSha256": digest.hexdigest(), "lookbackMinutes": LOOKBACK_MINUTES,
            "sequenceBinMinutes": SEQUENCE_BIN_MINUTES, "imputationApplied": False}


def feature_metadata(rows, issue_epoch):
    """Availability and provenance stay outside the numeric learner columns."""
    issue = _issue_epoch(issue_epoch)
    return _metadata(*_bounded(prepare_rows(rows), issue), issue)


def issue_feature_frame(rows, issue_epochs):
    """Batch uses one sorted snapshot and the same bounded query as inference."""
    prepared = prepare_rows(rows)
    issues = [_issue_epoch(value) for value in issue_epochs]
    if len(set(issues)) != len(issues):
        raise ValueError("Training/query issue epochs must be unique")
    records, metadata = [], []
    for issue in issues:
        epochs, values, known, availability = _bounded(prepared, issue)
        records.append(_features(epochs, values, issue))
        metadata.append(_metadata(epochs, values, known, availability, issue))
    frame = pd.DataFrame(records, index=pd.Index(issues, name="issueEpoch"), columns=FEATURE_COLUMNS, dtype=float)
    frame.attrs["featureVersion"] = VERSION
    frame.attrs["featureProvenance"] = metadata
    frame.attrs["imputationApplied"] = False
    return frame
