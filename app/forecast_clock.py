"""Causal, decision-clock targets for a regular right-labelled sensor series.

The feature timestamp is not the forecast issue timestamp.  Fifteen-minute
medians can be closed before a decision is made, so a nominal +90-minute target
must include that lag.  These functions do not fit or change any model.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


TARGET_VERSION = "decision_clock_weighted_15min_v1"


@dataclass(frozen=True)
class ForecastClock:
    issued_epoch: int
    anchor_epoch: int
    arrival_minutes: int = 90
    window_minutes: int = 120
    bucket_seconds: int = 900

    def __post_init__(self):
        if self.bucket_seconds <= 0 or self.window_minutes <= 0:
            raise ValueError("Bucket and window durations must be positive")
        if self.arrival_minutes < 0 or self.issued_epoch < self.anchor_epoch:
            raise ValueError("A forecast must not precede its feature anchor")
        if self.anchor_epoch % self.bucket_seconds:
            raise ValueError("The feature anchor must be a closed bucket label")

    @property
    def lag_seconds(self):
        return self.issued_epoch - self.anchor_epoch

    @property
    def arrival_lead_seconds(self):
        return self.lag_seconds + self.arrival_minutes * 60

    @property
    def end_lead_seconds(self):
        return self.arrival_lead_seconds + self.window_minutes * 60

    @property
    def target_start_epoch(self):
        return self.issued_epoch + self.arrival_minutes * 60

    @property
    def target_end_epoch(self):
        return self.target_start_epoch + self.window_minutes * 60

    @property
    def outcome_complete_lead_seconds(self):
        """The last touched observed bucket must close before training/scoring."""
        return math.ceil(self.end_lead_seconds / self.bucket_seconds) * self.bucket_seconds

    def metadata(self):
        return {
            "targetVersion": TARGET_VERSION,
            "forecastIssuedEpoch": self.issued_epoch,
            "featureAnchorEpoch": self.anchor_epoch,
            "featureAnchorAgeSeconds": self.lag_seconds,
            "arrivalTargetEpoch": self.target_start_epoch,
            "windowStartEpoch": self.target_start_epoch,
            "windowEndEpoch": self.target_end_epoch,
            "arrivalLeadFromFeatureMinutes": self.arrival_lead_seconds / 60,
            "windowEndLeadFromFeatureMinutes": self.end_lead_seconds / 60,
            "arrivalTargetDefinition": "Interpolated 15-minute bucket-median proxy ending at target time, not an instantaneous measurement",
            "windowMeanTargetDefinition": "Overlap-duration weighted mean of complete 15-minute medians",
            "windowPeakTargetDefinition": "Maximum median among overlapping 15-minute buckets; edge buckets can extend outside exact window",
            "observedOutcomeCompleteAfterEpoch": self.anchor_epoch + self.outcome_complete_lead_seconds,
        }


def point_weights(lead_seconds, bucket_seconds=900):
    """Linearly interpolate the two bucket labels around an exact target."""
    if lead_seconds < 0 or bucket_seconds <= 0:
        raise ValueError("Invalid target lead or bucket duration")
    lower = int(lead_seconds // bucket_seconds)
    remainder = lead_seconds - lower * bucket_seconds
    if remainder == 0:
        return {lower: 1.0}
    fraction = remainder / bucket_seconds
    return {lower: 1 - fraction, lower + 1: fraction}


def window_weights(start_seconds, end_seconds, bucket_seconds=900):
    """Positive-overlap weights for right-closed intervals, with no imputation."""
    if start_seconds < 0 or end_seconds <= start_seconds or bucket_seconds <= 0:
        raise ValueError("Invalid window boundaries or bucket duration")
    result = {}
    first = int(start_seconds // bucket_seconds) + 1
    last = math.ceil(end_seconds / bucket_seconds)
    duration = end_seconds - start_seconds
    for offset in range(first, last + 1):
        overlap = min(end_seconds, offset * bucket_seconds) - max(
            start_seconds, (offset - 1) * bucket_seconds
        )
        if overlap > 0:
            result[offset] = overlap / duration
    return result


def _regular_series(series, bucket_seconds):
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Sensor values require a DatetimeIndex")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Sensor index must be increasing and unique")
    if len(series) > 1 and not np.all(np.diff(series.index.asi8) == bucket_seconds * 10**9):
        raise ValueError("Keep missing buckets as NaN; do not compress time gaps")
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _weighted_future(series, weights):
    parts = pd.concat({offset: series.shift(-offset) for offset in weights}, axis=1)
    # A single missing overlapping bucket invalidates the complete target.
    total = parts.mul(pd.Series(weights)).sum(axis=1, min_count=len(weights))
    return total, parts


def target_series(series, clock):
    """Construct historical targets at the SAME issue/feature offset as live.

    Future observations occur only on the response side. Caller must restrict
    model training to origins whose last required bucket had closed at issue.
    The production 330-minute embargo is sufficient only when it exceeds
    clock.outcome_complete_lead_seconds / 60.  No missing bucket is filled.
    """
    values = _regular_series(series, clock.bucket_seconds)
    arrival, _ = _weighted_future(
        values, point_weights(clock.arrival_lead_seconds, clock.bucket_seconds)
    )
    mean, parts = _weighted_future(
        values, window_weights(clock.arrival_lead_seconds, clock.end_lead_seconds, clock.bucket_seconds)
    )
    peak = parts.max(axis=1, skipna=False)
    return pd.DataFrame({"arrival": arrival, "mean": mean, "peak": peak})


def targets_match(first, second):
    """Never overlay a legacy fixed-lead estimate onto a different live target."""
    keys = ("targetVersion", "arrivalTargetEpoch", "windowStartEpoch", "windowEndEpoch")
    return bool(first and second and all(first.get(key) is not None and first.get(key) == second.get(key) for key in keys))


def score_observed_window(series, clock, as_of_epoch):
    """Score only a fully observed window, including both partial edge buckets."""
    values = _regular_series(series, clock.bucket_seconds)
    anchor = pd.Timestamp(clock.anchor_epoch, unit="s", tz="UTC")
    if values.index.tz is None:
        anchor = anchor.tz_localize(None)
    else:
        anchor = anchor.tz_convert(values.index.tz)
    last_required = clock.anchor_epoch + clock.outcome_complete_lead_seconds
    if last_required > int(as_of_epoch):
        return {"available": False, "reason": "target_buckets_not_closed", "completeAfterEpoch": last_required}
    weights = window_weights(clock.arrival_lead_seconds, clock.end_lead_seconds, clock.bucket_seconds)
    labels = pd.DatetimeIndex([anchor + pd.Timedelta(seconds=i * clock.bucket_seconds) for i in weights])
    observations = values.reindex(labels)
    covered = float(sum(weight for weight, value in zip(weights.values(), observations) if pd.notna(value)))
    if observations.isna().any():
        return {"available": False, "reason": "incomplete_observed_window", "coveredFraction": covered, "requiredBuckets": len(weights), "observedBuckets": int(observations.notna().sum())}
    return {"available": True, "mean": float(sum(value * weight for value, weight in zip(observations, weights.values()))), "peak": float(observations.max()), "coveredFraction": 1.0, "requiredBuckets": len(weights), "observedBuckets": len(weights)}
