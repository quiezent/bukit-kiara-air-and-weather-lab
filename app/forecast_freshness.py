"""Causal fresh-sensor references; no extrapolation, I/O, training or gap fill."""
from __future__ import annotations
import math
import numpy as np

MINIMUM_LAG_SECONDS = 0
REFERENCE_WINDOW_SECONDS = 300
VERSION = 'five_minute_sensor_refresh_v1'


def raw_arrays(rows, issue_epoch):
    finite = {}
    for row in rows:
        row = dict(row)
        try:
            observed, value = float(row['epoch']), float(row['pm02'])
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        if math.isfinite(observed) and observed <= issue_epoch and math.isfinite(value) and value >= 0:
            finite[observed] = value
    epochs = np.array(sorted(finite), dtype=float)
    return epochs, np.array([finite[e] for e in epochs], dtype=float)


def references(rows, origins, lag_seconds, issue_epoch):
    """Return medians over (historical issue-5min, historical issue], causally."""
    origins = np.asarray(origins, dtype=np.int64)
    points = np.full(len(origins), np.nan)
    stamps = np.zeros(len(origins), dtype=np.int64)
    counts = np.zeros(len(origins), dtype=np.int64)
    if not MINIMUM_LAG_SECONDS <= lag_seconds < 900:
        return points, stamps, counts
    epochs, values = raw_arrays(rows, issue_epoch)
    issues = origins + int(lag_seconds)
    left = np.searchsorted(epochs, issues-REFERENCE_WINDOW_SECONDS, side='right')
    right = np.searchsorted(epochs, issues, side='right')
    for i, (lo, hi) in enumerate(zip(left, right)):
        if issues[i] <= issue_epoch and hi > lo:
            points[i] = np.median(values[lo:hi])
            stamps[i] = epochs[hi-1]
            counts[i] = hi-lo
    return points, stamps, counts


def refresh_session_records(records, rows, lag_seconds, issue_epoch):
    """Shift a closed-feature forecast by the newly observed level difference.

    Fits, weights and exact future targets remain unchanged. Only the reference
    is refreshed. Replayed residuals must use these same refreshed predictions.
    Missing recent observations do not become filled values.
    """
    origins = [r['originEpoch'] for r in records]
    points, stamps, counts = references(rows, origins, lag_seconds, issue_epoch)
    refreshed = []
    for row, point, stamp, count in zip(records, points, stamps, counts):
        r = dict(row)
        r['closedPrediction'] = row['prediction']
        r['closedPersistence'] = row['persistence']
        r['forecastIssuedEpoch'] = row['originEpoch'] + int(lag_seconds)
        r['freshnessApplied'] = bool(np.isfinite(point))
        r['freshReferenceEpoch'] = int(stamp) if count else None
        r['freshReferenceCount'] = int(count)
        if np.isfinite(point):
            r['prediction'] = max(0., row['prediction'] + point-row['persistence'])
            r['persistence'] = float(point)
        refreshed.append(r)
    return refreshed
