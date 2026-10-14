"""Read-only scoring of originally published first-crossing event snapshots.

No model imports, fits, provider calls, publication or database writes occur.
The caller must supply an explicit cutoff before the sealed 2026-10-20 MYT
period. This scorer is a descriptive audit of issued predictions; a policy
written after seeing those outcomes does not make them an untouched test set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd

from collection_quality import bucket_coverage
from forecast_clock import point_weights


VERSION = "issued_first_crossing_descriptive_validation_v1"
MYT = timezone(timedelta(hours=8))
SEALED_START_EPOCH = int(datetime(2026, 10, 20, tzinfo=MYT).timestamp())
LEADS = (15, 30, 45, 60, 75, 90)
CLASSES = ("drop", "none", "rise")
PROBABILITY_KEYS = ("probabilityDrop", "probabilityNoCrossing", "probabilityRise")
MAX_LINE_BYTES = 1024 * 1024
MAX_SENSOR_ROWS = 2_000_000
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 61008
_ISSUE_PATTERN = re.compile(rb'"forecastIssuedEpoch"\s*:\s*(-?\d+)(?=\s*[,}])')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _epoch(value):
    if isinstance(value, bool):
        raise ValueError("Boolean timestamps are invalid")
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError("Timestamps must be finite integral epochs")
    return int(number)


def _validate_cutoff(cutoff_epoch, max_age_days):
    cutoff = _epoch(cutoff_epoch)
    if cutoff >= SEALED_START_EPOCH:
        raise ValueError("Cutoff must be strictly before sealed 2026-10-20 MYT")
    if cutoff > int(time.time()):
        raise ValueError("Cutoff cannot be in the future")
    days = float(max_age_days)
    if not math.isfinite(days) or not 0 < days <= 35:
        raise ValueError("max_age_days must be positive and no greater than 35")
    return cutoff, cutoff - int(days * 86400)


def _clock_regime(row):
    issue = row["forecastIssuedEpoch"]
    local = datetime.fromtimestamp(issue, MYT)
    offset = issue % 900
    four_hour_origin = local.hour % 4 == 0 and local.minute < 15
    exact = four_hour_origin and offset == 300
    band = None if exact else [int(offset // 300 * 300), int(offset // 300 * 300 + 299)]
    return {"regime": "fixed_four_hour_origin_plus300_seconds" if exact else "actual_publication_clock",
            "onLocalFourHourOrigin": four_hour_origin, "offsetBandSeconds": band,
            "archivedTrainingCadence": row.get("trainingFitCadence", "not_archived")}


def _group_metadata(row):
    policy = row.get("selectedPolicy")
    if policy is None:
        # Do not attach today's source identity or policy to legacy predictions.
        policy = {"provenance": "legacy_issue_fields_only", "target": row.get("target"),
                  "thresholdComparison": row.get("thresholdComparison", "at_least"),
                  "directionDecision": row.get("directionDecision", "archived_cause_probability_threshold"),
                  "directionCallProbability": row.get("directionCallProbability", .5),
                  "directionTieTolerance": row.get("directionTieTolerance", 0)}
    return {"modelVersion": row.get("modelVersion", "not_archived"),
            "adapterVersion": row.get("adapterVersion", "not_archived"),
            "modelIdentity": row.get("modelIdentity", {"provenance": "missing_archived_model_identity"}),
            "selectedPolicy": policy, "changeThresholdUgM3": row.get("changeThresholdUgM3"),
            "thresholdComparison": row.get("thresholdComparison", "at_least"),
            "qualificationVersion": row.get("qualificationVersion", "not_archived"),
            "qualificationState": row.get("qualification", {}).get("state", "not_archived")
                if isinstance(row.get("qualification", {}), dict) else "invalid_archived_qualification",
            "operationalDecisionRule": row.get("operationalDecisionRule", "original_archived_direction_call"),
            "actualIssueClockRegime": _clock_regime(row)}


def _lineage_reason(row, cutoff, earliest):
    try:
        issue = _epoch(row["forecastIssuedEpoch"])
        published = _epoch(row["publishedEpoch"])
        recorded = _epoch(row["recordedEpoch"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return "invalid_publication_epochs"
    if not earliest <= issue <= cutoff:
        return "issue_outside_cutoff_window"
    if issue >= SEALED_START_EPOCH:
        return "issue_in_sealed_period"
    if row.get("isOriginallyIssued") is not True:
        return "not_originally_issued"
    if row.get("predictionRole") not in ("genuinely_issued_live_event_prediction",
                                          "genuinely_issued_live_large20_prediction"):
        return "not_original_live_prediction_role"
    if not (0 <= published - issue <= 120 and 0 <= recorded - issue <= 120):
        return "publication_delay_outside_0_to120_seconds"
    if published > cutoff or recorded > cutoff:
        return "publication_not_known_at_cutoff"
    if recorded < published:
        return "recording_precedes_publication"
    if row.get("available") is not True:
        return None  # Unavailable outcomes remain in the denominator.
    for key in ("fittedAtEpoch", "trainingCutoffEpoch", "maxTrainingOutcomeEpoch", "featureOriginEpoch",
                "sensorWatermarkEpoch", "freshReferenceEpoch", "inputMaxMeasurementEpoch",
                "featureSourceMaxEpoch", "weatherFetchedEpoch"):
        if row.get(key) is not None:
            try:
                epoch = float(row[key])
            except (ValueError, TypeError, OverflowError):
                return "invalid_input_epoch_" + key
            if not math.isfinite(epoch) or epoch > issue:
                return "input_after_issue_" + key
    if row.get("trainingCutoffEpoch") is not None and row.get("maxTrainingOutcomeEpoch") is not None:
        if row["maxTrainingOutcomeEpoch"] >= row["trainingCutoffEpoch"]:
            return "training_label_not_before_training_cutoff"
    if row.get("sensorWatermarkEpoch") is not None:
        for key in ("freshReferenceEpoch", "featureSourceMaxEpoch", "inputMaxMeasurementEpoch"):
            if row.get(key) is not None and row[key] > row["sensorWatermarkEpoch"]:
                return "input_after_sensor_watermark_" + key
    return None


def _snapshot_ledgers(ledger_paths, cutoff, earliest):
    rows, manifests = [], []
    for given_path in ledger_paths:
        path = Path(given_path).resolve()
        length = path.stat().st_size  # Streamed by line: only accepted in-window rows are held.
        rejected, kept, duplicate = Counter(), {}, 0
        conflicting = set()
        bounded_hash = hashlib.sha256()
        inspected = 0
        with path.open("rb") as stream:
            remaining = length  # Appends after this snapshot are never consumed.
            while remaining:
                line = stream.readline(min(remaining, MAX_LINE_BYTES + 1))
                remaining -= len(line)
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError("Ledger line exceeds bounded snapshot size")
                if not line:
                    break
                if not line.endswith(b"\n") and remaining == 0:
                    rejected["incomplete_snapshot_tail"] += 1
                    break
                if not line.strip():
                    continue
                inspected += 1
                # Inspect only timestamp envelope before decoding a forecast.
                # Future/sealed prediction values and feature arrays are not parsed.
                match = _ISSUE_PATTERN.search(line)
                if match is None:
                    rejected["invalid_issue_epoch_envelope"] += 1
                    continue
                issue = int(match.group(1))
                if issue > cutoff or issue >= SEALED_START_EPOCH:
                    rejected["future_or_sealed_issue_not_decoded"] += 1
                    continue
                if issue < earliest:
                    rejected["issue_before_cutoff_window_not_decoded"] += 1
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected a snapshot object")
                    reason = _lineage_reason(row, cutoff, earliest)
                    fingerprint = _digest(row)
                except (ValueError, TypeError, KeyError, OverflowError):
                    rejected["malformed_snapshot"] += 1
                    continue
                if reason:
                    rejected[reason] += 1
                    continue
                bounded_hash.update(line)
                key = (issue, str(row.get("modelVersion")), str(row.get("adapterVersion")),
                       str(row.get("changeThresholdUgM3")))
                if key in conflicting:
                    rejected["conflicting_original_publication"] += 1
                    continue
                if key in kept:
                    if kept[key][0] == fingerprint:
                        duplicate += 1
                    else:
                        del kept[key]
                        conflicting.add(key)
                        rejected["conflicting_original_publication"] += 2
                    continue
                row["forecastIssuedEpoch"] = issue
                kept[key] = (fingerprint, row)
        selected = [row for _, row in kept.values()]
        rows.extend(selected)
        manifests.append({"path": str(path), "snapshotBytes": length, "inspectedTimestampEnvelopes": inspected,
                          "acceptedOriginalPublicationRows": len(selected), "duplicateRows": duplicate,
                          "rejectedRows": dict(sorted(rejected.items())),
                          "acceptedTemporalSnapshotSha256": bounded_hash.hexdigest()})
    # Passing the same ledger twice would manufacture support.
    if len({manifest["path"] for manifest in manifests}) != len(manifests):
        raise ValueError("Each ledger must appear exactly once")
    return sorted(rows, key=lambda row: row["forecastIssuedEpoch"]), manifests


def _probabilities(row):
    if row.get("target") != "first_sampled_crossing_within90_minutes":
        raise ValueError("unsupported_event_target")
    if float(row.get("changeThresholdUgM3", 0)) not in (10., 20.):
        raise ValueError("unsupported_change_threshold")
    if row.get("thresholdComparison", "at_least") != "at_least":
        raise ValueError("unsupported_threshold_comparison")
    reference = float(row["referencePm"])
    if not math.isfinite(reference) or reference < 0:
        raise ValueError("invalid_archived_reference")
    horizons = row.get("horizons", [])
    if not isinstance(horizons, list) or [h.get("leadMinutes") for h in horizons] != list(LEADS):
        raise ValueError("missing_or_invalid_six_horizons")
    probabilities = np.array([[float(h[key]) for key in PROBABILITY_KEYS] for h in horizons], dtype=float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("invalid_probability_mass")
    if (np.diff(probabilities[:, 0]) < -1e-8).any() or (np.diff(probabilities[:, 2]) < -1e-8).any():
        raise ValueError("nonmonotonic_first_cause_incidence")
    for index, key in enumerate(PROBABILITY_KEYS):
        if row.get(key) is not None and not math.isclose(float(row[key]), probabilities[-1, index], abs_tol=1e-8):
            raise ValueError("final_probability_disagrees_with_horizon")
    return probabilities


def _support_labels(issue):
    origin = issue // 900 * 900
    return [{origin + offset * 900: weight for offset, weight in point_weights(issue - origin + lead * 60).items()}
            for lead in LEADS]


def _sensor_snapshot(db_path, earliest_label, cutoff):
    path = Path(db_path).resolve()
    query_start = earliest_label - 900
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        rows = conn.execute("SELECT epoch,pm02 FROM readings WHERE epoch>=? AND epoch<=? "
                            "AND epoch<? ORDER BY epoch LIMIT ?",
                            (query_start, cutoff, SEALED_START_EPOCH, MAX_SENSOR_ROWS + 1)).fetchall()
    if len(rows) > MAX_SENSOR_ROWS:
        raise ValueError("Sensor snapshot exceeds bounded row count")
    finite = {}
    for epoch, value in rows:
        try:
            epoch, value = _epoch(epoch), float(value)
        except (ValueError, TypeError, OverflowError):
            continue
        if query_start <= epoch <= cutoff and epoch < SEALED_START_EPOCH and math.isfinite(value) and value >= 0:
            finite[epoch] = value
    epochs = sorted(finite)
    manifest = {"databasePath": str(path), "databaseReadOnly": True, "singleReadTransaction": True,
                "sensorQueryStartEpoch": query_start, "sensorQueryEndEpoch": cutoff,
                "sealedUpperExclusiveEpoch": SEALED_START_EPOCH, "sensorRowsRead": len(rows),
                "maxSensorEpochRead": max((row[0] for row in rows), default=None),
                "finiteSensorRows": len(epochs), "providerFetches": 0, "fits": 0, "databaseWrites": 0,
                "boundedSensorRowsSha256": _digest([[epoch, finite[epoch]] for epoch in epochs])}
    if not epochs:
        return {}, manifest
    index = pd.to_datetime(epochs, unit="s", utc=True).tz_convert(MYT)
    raw = pd.DataFrame({"pm02": [finite[epoch] for epoch in epochs]}, index=index)
    eligible = bucket_coverage(raw, bucket_seconds=900)
    medians = raw.pm02.resample("900s", label="right", closed="right").median()
    medians = medians.where(eligible.forecastEligible.reindex(medians.index).fillna(False))
    values = {int(stamp.timestamp()): float(value) for stamp, value in medians.items()
              if stamp.timestamp() <= cutoff and stamp.timestamp() < SEALED_START_EPOCH and pd.notna(value)}
    return values, manifest


def _observed(row, medians, cutoff):
    weights = _support_labels(row["forecastIssuedEpoch"])
    labels = sorted({label for step in weights for label in step})
    complete = max(labels)
    if complete > cutoff or complete >= SEALED_START_EPOCH:
        return None, "target_buckets_not_closed_at_cutoff"
    if any(label not in medians for label in labels):
        return None, "incomplete_covered_target_support"
    values = [sum(medians[label] * weight for label, weight in step.items()) for step in weights]
    deltas = np.array(values) - float(row["referencePm"])
    crossed = np.flatnonzero(np.abs(deltas) >= float(row["changeThresholdUgM3"]))
    first = int(crossed[0]) if len(crossed) else None
    cause = 0 if first is None else (0 if deltas[first] < 0 else 2)
    # Class index 1 denotes none; a first cause is absorbing thereafter.
    truth = np.full(len(LEADS), 1, dtype=int)
    if first is not None:
        truth[first:] = cause
    return {"truth": truth, "values": values, "firstCrossingLeadMinutes": None if first is None else LEADS[first],
            "supportStartEpoch": min(labels) - 900, "supportEndEpoch": complete,
            "targetDate": datetime.fromtimestamp(complete, MYT).date().isoformat()}, None


def _prior(row):
    counts = row.get("trainingSupport")
    if not isinstance(counts, dict):
        return None
    try:
        values = np.array([float(counts[name]) for name in CLASSES])
    except (ValueError, TypeError, KeyError, OverflowError):
        return None
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        return None
    return (values + 1) / (values.sum() + len(CLASSES))


def _raw_calls(probabilities, tolerance=0):
    calls = np.argmax(probabilities, axis=1)
    maximum = probabilities.max(axis=1, keepdims=True)
    tied = (np.abs(probabilities - maximum) <= tolerance).sum(axis=1) != 1
    calls[tied] = 1
    return calls


def _deployed_calls(row, probabilities):
    rule = row.get("directionDecision", "archived_cause_probability_threshold")
    if rule == "three_class_argmax_ties_unresolved":
        calls = _raw_calls(probabilities, float(row.get("directionTieTolerance", 0)))
    elif rule == "archived_cause_probability_threshold":
        threshold = float(row.get("directionCallProbability", .5))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("invalid_direction_call_probability")
        drop, rise = probabilities[:, 0] >= threshold, probabilities[:, 2] >= threshold
        calls = np.full(len(LEADS), 1, dtype=int)
        calls[drop & ~rise], calls[rise & ~drop] = 0, 2
    else:
        raise ValueError("unsupported_archived_direction_rule")
    mapping = {"drop": 0, "unresolved": 1, "none": 1, "rise": 2}
    raw_calls = calls.copy()
    if row.get("diagnosticDirectionAvailable") is True:
        diagnostic = mapping.get(row.get("diagnosticDirection"))
        if diagnostic is None:
            raise ValueError("invalid_archived_diagnostic_direction")
        raw_calls[-1] = diagnostic
    if row.get("operationalDecisionRule") == "no_direction_until_actual_cadence_prospective_qualification":
        calls[:] = 1
    archived = mapping.get(row.get("direction"))
    if archived is None:
        raise ValueError("invalid_archived_direction")
    # The original final call is authoritative even if an old adapter's policy
    # was incompletely archived. Report this discrepancy explicitly.
    discrepancy = int(archived != calls[-1])
    calls[-1] = archived
    return calls, discrepancy, raw_calls


def _direction_metrics(truth, called):
    event, call = truth != 1, called != 1
    correct = event & (truth == called)
    false = call & ~event
    wrong = event & call & (truth != called)
    per_cause = {}
    for code in (0, 2):
        actual, predicted = truth == code, called == code
        good = actual & predicted
        per_cause[CLASSES[code]] = {"actualEvents": int(actual.sum()), "directionCalls": int(predicted.sum()),
            "correctDirectionCalls": int(good.sum()), "falseDirectionCalls": int((predicted & ~actual).sum()),
            "wrongWayCalls": int((predicted & event & ~actual).sum()),
            "recall": float(good.sum() / actual.sum()) if actual.any() else None,
            "precision": float(good.sum() / predicted.sum()) if predicted.any() else None}
    return {"actualEvents": int(event.sum()), "directionCalls": int(call.sum()),
            "correctDirectionCalls": int(correct.sum()), "falseEventCalls": int(false.sum()),
            "falseDirectionCalls": int((call & (truth != called)).sum()), "wrongWayCalls": int(wrong.sum()),
            "falseClearingCalls": int(((called == 0) & (truth != 0)).sum()),
            "eventRecall": float((call & event).sum() / event.sum()) if event.any() else None,
            "correctDirectionRecall": float(correct.sum() / event.sum()) if event.any() else None,
            "falseEventRateAmongNone": float(false.sum() / (~event).sum()) if (~event).any() else None,
            "causes": per_cause}


def _reliability(probabilities, truth):
    result = {}
    for code, name in enumerate(CLASSES):
        assigned = np.minimum(4, np.floor(probabilities[:, code] * 5).astype(int))
        actual = truth == code
        bins = []
        for index in range(5):
            keep = assigned == index
            bins.append({"lowerProbability": index / 5, "upperProbability": (index + 1) / 5,
                         "upperInclusive": index == 4, "count": int(keep.sum()),
                         "observedCount": int((keep & actual).sum()),
                         "meanProbability": float(probabilities[keep, code].mean()) if keep.any() else None,
                         "observedFrequency": float(actual[keep].mean()) if keep.any() else None})
        result[name] = bins
    return result


def _episode_count(intervals):
    end, count = None, 0
    for start, finish in sorted(intervals):
        # Touching right-labelled bucket supports remain one connected component.
        if end is None or start > end:
            count += 1
            end = finish
        else:
            end = max(end, finish)
    return count


def _paired_interval(differences, dates):
    daily = defaultdict(lambda: [0., 0])
    for delta, date in zip(differences, dates):
        daily[date][0] += float(delta)
        daily[date][1] += 1
    ordered = [daily[date] for date in sorted(daily)]
    result = {"pairedCases": len(differences), "dayClusters": len(ordered),
              "meanModelMinusBaseline": float(np.mean(differences)),
              "confidenceLevel": .95, "descriptiveOnly": True, "lower": None, "upper": None}
    if len(ordered) < 2:
        result["reason"] = "insufficient_support_fewer_than_two_target_dates"
        return result
    sums, counts = np.array(ordered, dtype=float).T
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(ordered), size=(BOOTSTRAP_DRAWS, len(ordered)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    result["lower"], result["upper"] = map(float, np.quantile(means, [.025, .975]))
    result["bootstrapDraws"] = BOOTSTRAP_DRAWS
    result["reason"] = "overlap_and_small_day_support_limit_interpretation"
    return result


def _lead_metrics(scored, step):
    probabilities = np.stack([row["probabilities"][step] for row in scored])
    truth = np.array([row["observed"]["truth"][step] for row in scored], dtype=int)
    onehot = np.eye(3)[truth]
    losses = ((probabilities - onehot) ** 2).sum(axis=1)
    none = np.tile([0., 1., 0.], (len(scored), 1))
    none_losses = ((none - onehot) ** 2).sum(axis=1)
    dates = [row["observed"]["targetDate"] for row in scored]
    deployed = np.array([row["deployedCalls"][step] for row in scored])
    raw = np.array([row["rawCalls"][step] for row in scored])
    out = {"count": len(scored), "probabilityProvenance": "original_saved_horizon_probabilities",
           "directionProvenance": "authoritative_original_archived_final_direction" if step == len(LEADS) - 1
                                  else "derived_from_saved_probabilities_and_declared_policy_not_independently_published_calls",
           "rawDiagnosticDirectionProvenance": "original_model_policy_applied_to_saved_probabilities; archived_final_diagnostic_used_when_available",
           "multiclassBrier": float(losses.mean()),
           "causewiseBrier": {name: float(np.mean((probabilities[:, code] - onehot[:, code]) ** 2))
                              for code, name in enumerate(CLASSES)},
           "actualClassCounts": {name: int((truth == code).sum()) for code, name in enumerate(CLASSES)},
           "deployedDirection": _direction_metrics(truth, deployed),
           "diagnosticRawDirection": _direction_metrics(truth, raw),
           "reliabilityFiveBins": _reliability(probabilities, truth),
           "alwaysNone": {"count": len(scored), "multiclassBrier": float(none_losses.mean()),
                          "causewiseBrier": {name: float(np.mean((none[:, code] - onehot[:, code]) ** 2))
                                             for code, name in enumerate(CLASSES)},
                          "pairedMulticlassBrierDifference": _paired_interval(losses - none_losses, dates)}}
    prior_mask = np.array([row["prior"] is not None for row in scored])
    out["trainingSupportPriorMissingCases"] = int((~prior_mask).sum())
    if prior_mask.any():
        priors = np.stack([row["prior"] for row in scored if row["prior"] is not None])
        prior_onehot, paired_probs = onehot[prior_mask], probabilities[prior_mask]
        prior_losses = ((priors - prior_onehot) ** 2).sum(axis=1)
        model_losses = losses[prior_mask]
        prior_dates = [date for date, keep in zip(dates, prior_mask) if keep]
        out["asIssuedTrainingSupportPrior"] = {
            "count": int(prior_mask.sum()), "smoothingPseudocountPerClass": 1,
            "definition": "Archived final90-minute training class counts; same fixed prior at every lead, not a reconstructed step-specific prior",
            "multiclassBrier": float(prior_losses.mean()), "pairedModelMulticlassBrier": float(model_losses.mean()),
            "causewiseBrier": {name: float(np.mean((priors[:, code] - prior_onehot[:, code]) ** 2))
                               for code, name in enumerate(CLASSES)},
            "pairedModelCausewiseBrier": {name: float(np.mean((paired_probs[:, code] - prior_onehot[:, code]) ** 2))
                                          for code, name in enumerate(CLASSES)},
            "pairedMulticlassBrierDifference": _paired_interval(model_losses - prior_losses, prior_dates)}
    else:
        out["asIssuedTrainingSupportPrior"] = {"count": 0, "available": False, "reason": "training_support_not_archived"}
    return out


def score_issued(db_path, ledger_paths, cutoff_epoch, max_age_days=35):
    """Score bounded original publications without fitting or mutating sources.

    max_age_days is bounded at 35. All six future sampled proxies must have
    complete covered bucket support; even an early crossing cannot rescue a
    later gap. Missing/unavailable publications never disappear from counts.
    """
    cutoff, earliest = _validate_cutoff(cutoff_epoch, max_age_days)
    if isinstance(ledger_paths, (str, Path)):
        ledger_paths = [ledger_paths]
    ledger_paths = list(ledger_paths)
    if not ledger_paths:
        raise ValueError("At least one original issued ledger is required")
    snapshots, ledger_manifests = _snapshot_ledgers(ledger_paths, cutoff, earliest)
    groups = {}
    candidates = []
    for row in snapshots:
        metadata = _group_metadata(row)
        group_id = _digest(metadata)[:20]
        group = groups.setdefault(group_id, {"groupId": group_id, "metadata": metadata, "originallyIssuedCount": 0,
                "availableCount": 0, "unavailableCount": 0, "unavailableReasons": Counter(),
                "unscorableReasons": Counter(), "scored": [], "issueOffsetsSeconds": [], "issues": []})
        group["originallyIssuedCount"] += 1
        group["issueOffsetsSeconds"].append(row["forecastIssuedEpoch"] % 900)
        group["issues"].append(row["forecastIssuedEpoch"])
        if row.get("available") is not True:
            group["unavailableCount"] += 1
            group["unavailableReasons"][str(row.get("reason", "not_archived"))] += 1
            continue
        group["availableCount"] += 1
        try:
            probabilities = _probabilities(row)
            deployed, discrepancy, raw_calls = _deployed_calls(row, probabilities)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            group["unscorableReasons"][str(error)] += 1
            continue
        candidates.append({"snapshot": row, "group": group, "probabilities": probabilities,
                           "deployedCalls": deployed, "rawCalls": raw_calls,
                           "archivedDirectionRuleDiscrepancy": discrepancy, "prior": _prior(row)})
    sensor_manifest = {"databaseReadOnly": True, "sensorRowsRead": 0, "providerFetches": 0,
                       "fits": 0, "databaseWrites": 0, "reason": "no_available_valid_original_predictions"}
    if candidates:
        earliest_label = min(min(step) for candidate in candidates
                             for step in _support_labels(candidate["snapshot"]["forecastIssuedEpoch"]))
        medians, sensor_manifest = _sensor_snapshot(db_path, earliest_label, cutoff)
        for candidate in candidates:
            observed, reason = _observed(candidate["snapshot"], medians, cutoff)
            if reason:
                candidate["group"]["unscorableReasons"][reason] += 1
                continue
            candidate["observed"] = observed
            candidate["group"]["scored"].append(candidate)
    summaries = []
    for group_id in sorted(groups):
        group = groups[group_id]
        scored = group.pop("scored")
        offsets, issues = group.pop("issueOffsetsSeconds"), group.pop("issues")
        group["unavailableReasons"] = dict(sorted(group["unavailableReasons"].items()))
        group["unscorableReasons"] = dict(sorted(group["unscorableReasons"].items()))
        group.update(scoredCount=len(scored), unscorableAvailableCount=group["availableCount"] - len(scored),
                     actualIssueClock={"minimumEpoch": min(issues), "maximumEpoch": max(issues),
                        "offsetMinimumSeconds": min(offsets), "offsetMaximumSeconds": max(offsets),
                        "distinctExactOffsets": len(set(offsets))})
        group["support"] = {"distinctTargetDates": len({row["observed"]["targetDate"] for row in scored}),
                            "connectedOverlappingSupportEpisodes": _episode_count(
                                [(row["observed"]["supportStartEpoch"], row["observed"]["supportEndEpoch"]) for row in scored]),
                            "episodesAreIndependentPhysicalEvents": False}
        group["support"]["byFinalCause"] = {}
        for code, name in enumerate(CLASSES):
            cause_rows = [row for row in scored if row["observed"]["truth"][-1] == code]
            group["support"]["byFinalCause"][name] = {
                "cases": len(cause_rows),
                "distinctTargetDates": len({row["observed"]["targetDate"] for row in cause_rows}),
                "connectedOverlappingSupportEpisodes": _episode_count(
                    [(row["observed"]["supportStartEpoch"], row["observed"]["supportEndEpoch"]) for row in cause_rows])}
        group["archivedFinalDirectionRuleDiscrepancies"] = sum(row["archivedDirectionRuleDiscrepancy"] for row in scored)
        group["finalDiagnosticDirectionArchivedCases"] = sum(
            row["snapshot"].get("diagnosticDirectionAvailable") is True for row in scored)
        group["finalDiagnosticDirectionReconstructedCases"] = len(scored) - group["finalDiagnosticDirectionArchivedCases"]
        group["leads"] = {str(lead): _lead_metrics(scored, step) for step, lead in enumerate(LEADS)} if scored else {}
        group["final90Minutes"] = group["leads"].get("90", {"count": 0})
        summaries.append(group)
    policy = {"version": VERSION, "timezone": "Asia/Kuala_Lumpur", "requiredExplicitCutoff": True,
              "sealedStartExclusiveEpoch": SEALED_START_EPOCH, "sealedStartLocal": "2026-10-20T00:00:00+08:00",
              "leadMinutes": list(LEADS), "classOrder": list(CLASSES), "thresholdComparison": "at_least",
              "target": "First absorbing +/- archived10 or20 crossing among six exact-issue-clock interpolated covered15-minute median proxies",
              "coverage": "All touched right-closed15-minute buckets eligible under collection_quality; no filling; all six proxies required",
              "reference": "Original as-issued fresh sensor reference from the immutable publication snapshot",
              "publicationLineage": "isOriginallyIssued true; original-live role; publication and recording delay0..120 seconds; known at cutoff",
              "grouping": "Archived model identity, model version, adapter version, selected policy, threshold and actual issue-clock regime",
              "clockRegimeDefinition": "Exact local four-hour origin+300s separately; other clocks by local four-hour origin flag and fixed five-minute offset bands",
              "missingLegacyIdentity": "Retained as an explicit missing-identity group; current code identity is never retroactively assigned",
              "rawDirection": "Original model decision policy: archived cause probability threshold for10, archived three-class argmax for20; archived final diagnosticDirection is authoritative when available",
              "perLeadDirectionProvenance": "Only the final overall direction is actually archived; shorter-lead direction metrics are derived from saved probabilities and declared policy, not independently published calls",
              "prior": "Archived final90-minute trainingSupport with one pseudocount per class, fixed across leads; not the frozen step-specific research comparator",
              "uncertainty": "Deterministic target-date cluster bootstrap; 2000 draws, seed61008; model minus paired baseline Brier, positive means worse",
              "overlap": "Connected support intervals are descriptive episode proxies; day clusters can still be dependent across midnight",
              "prospectiveUse": "This code fixes scoring choices for later new issues but does not certify an externally timestamped preregistration",
              "interpretation": "Originally issued retrospective descriptive audit; no untouched-test, qualification, calibration or promotion claim",
              "historicalOutcomeLimitation": "Sensor medians are reconstructed from the bounded current database; legacy receipt/revision history may be incomplete",
              "providerFetches": 0, "fits": 0, "databaseWrites": 0}
    return {"version": VERSION, "cutoffEpoch": cutoff, "cutoffLocal": datetime.fromtimestamp(cutoff, MYT).isoformat(),
            "earliestIssueEpoch": earliest, "maxAgeDays": float(max_age_days), "evaluationPolicy": policy,
            "evaluationPolicySha256": _digest(policy), "scorerSourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest": {"ledgers": ledger_manifests, "sensor": sensor_manifest},
            "totals": {"originallyIssued": sum(group["originallyIssuedCount"] for group in summaries),
                       "available": sum(group["availableCount"] for group in summaries),
                       "unavailable": sum(group["unavailableCount"] for group in summaries),
                       "scored": sum(group["scoredCount"] for group in summaries),
                       "unscorableAvailable": sum(group["unscorableAvailableCount"] for group in summaries)},
            "groups": summaries}


def render_markdown(report):
    """Human-readable report alongside the complete machine-readable JSON."""
    def number(value):
        return "unavailable" if value is None else f"{value:.5f}"

    totals = report["totals"]
    lines = ["# Issued event validation", "", f"Explicit cutoff: **{report['cutoffLocal']}**.", "",
             "Originally issued retrospective descriptive audit. These outcomes are not an untouched test set. "
             "This report does not qualify, calibrate, or promote a model.", "",
             f"Original publications: {totals['originallyIssued']}; available: {totals['available']}; "
             f"unavailable: {totals['unavailable']}; fully covered and scored: {totals['scored']}; "
             f"available but unscorable: {totals['unscorableAvailable']}.", "",
             "All six +15 through +90 minute point proxies require complete coverage. First cause is absorbing. "
             "Targets use exact original issue clocks and the archived reference. No missing bucket is filled.", "",
             "Saved probabilities at every horizon are original issued estimates. Only the final overall direction "
             "is actually archived. Shorter-lead direction metrics are derived from saved probabilities and "
             "the declared rule; those direction calls were not independently published.", "",
             "The training-support comparator uses the archived final90-minute class counts with one pseudocount "
             "per class. It is a fixed prior at every lead, because legacy publications do not contain step-specific priors.", "",
             "Brier differences are model minus paired comparator; positive means worse. The intervals use a "
             "deterministic target-date cluster bootstrap and are descriptive. Connected overlapping supports "
             "are episode proxies, not independent physical events.", "",
             f"Policy SHA256: `{report['evaluationPolicySha256']}`.", ""]
    for group in report["groups"]:
        metadata = group["metadata"]
        lines += [f"## {group['groupId']}", "",
                  f"Model: `{metadata['modelVersion']}`; adapter: `{metadata['adapterVersion']}`; "
                  f"threshold: **{metadata['changeThresholdUgM3']} µg/m³**.", "",
                  f"Issue clock: `{_canonical(metadata['actualIssueClockRegime'])}`.", "",
                  f"Original publications {group['originallyIssuedCount']}; available {group['availableCount']}; "
                  f"unavailable {group['unavailableCount']}; scored {group['scoredCount']}. "
                  f"Distinct target dates {group['support']['distinctTargetDates']}; connected support episodes "
                  f"{group['support']['connectedOverlappingSupportEpisodes']}.", ""]
        if group["unavailableReasons"]:
            lines += [f"Unavailable reasons: `{_canonical(group['unavailableReasons'])}`.", ""]
        if group["unscorableReasons"]:
            lines += [f"Unscorable reasons: `{_canonical(group['unscorableReasons'])}`.", ""]
        if not group["leads"]:
            continue
        lines += ["| Lead (min) | Cases | Model Brier | Always-none Brier | Issued-support prior Brier | "
                  "Deployed correct direction recall | Raw diagnostic correct direction recall |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for lead, score in group["leads"].items():
            prior = score["asIssuedTrainingSupportPrior"]
            lines.append(f"| {lead} | {score['count']} | {number(score['multiclassBrier'])} | "
                         f"{number(score['alwaysNone']['multiclassBrier'])} | {number(prior.get('multiclassBrier'))} | "
                         f"{number(score['deployedDirection']['correctDirectionRecall'])} | "
                         f"{number(score['diagnosticRawDirection']['correctDirectionRecall'])} |")
        final = group["final90Minutes"]
        lines += ["", f"Final actual class counts: `{_canonical(final['actualClassCounts'])}`.", "",
                  f"Deployed final direction counts and rates: `{_canonical(final['deployedDirection'])}`.", "",
                  f"Diagnostic raw final direction counts and rates: `{_canonical(final['diagnosticRawDirection'])}`.", "",
                  f"Paired model minus always-none Brier: "
                  f"`{_canonical(final['alwaysNone']['pairedMulticlassBrierDifference'])}`.", ""]
        if final["asIssuedTrainingSupportPrior"].get("count"):
            lines += [f"Paired model minus as-issued support prior Brier: "
                      f"`{_canonical(final['asIssuedTrainingSupportPrior']['pairedMulticlassBrierDifference'])}`.", ""]
        lines += ["Final causewise Brier: " + _canonical(final["causewiseBrier"]) + ".", "",
                  "| Cause | Probability bin | Cases | Mean predicted probability | Observed frequency |",
                  "|---|---|---:|---:|---:|"]
        for name, bins in final["reliabilityFiveBins"].items():
            for entry in bins:
                lines.append(f"| {name} | {entry['lowerProbability']:.1f}–{entry['upperProbability']:.1f} | "
                             f"{entry['count']} | {number(entry['meanProbability'])} | {number(entry['observedFrequency'])} |")
        lines += [""]
    lines += ["## Read-only source manifest", "", "```json", json.dumps(report["manifest"], indent=2), "```", "",
              "Legacy missing model identities remain separate and explicit. Sensor outcomes use the bounded current "
              "database; original sensor receipt and revision history may be incomplete.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--ledger", required=True, action="append", type=Path)
    parser.add_argument("--cutoff-epoch", required=True, type=int)
    parser.add_argument("--max-age-days", type=float, default=35)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    for output, suffix in ((args.json_output, ".json"), (args.markdown_output, ".md")):
        if output is not None and output.suffix.lower() != suffix:
            parser.error("Output must use " + suffix + " (TXT documents are not supported)")
        if output is not None and output.resolve() in {args.db.resolve(), *(path.resolve() for path in args.ledger)}:
            parser.error("Output cannot overwrite a source")
    report = score_issued(args.db, args.ledger, args.cutoff_epoch, args.max_age_days)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"cutoffEpoch": report["cutoffEpoch"], "totals": report["totals"], "groupCount": len(report["groups"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
