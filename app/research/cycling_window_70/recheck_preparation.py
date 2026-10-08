"""Prepare causal +30-minute rechecks of fixed, originally planned ride windows.

Research only: read-only bounded archive access; no provider requests or fits.
Outcome labels are copied from the frozen initial mean90to210 artifact. Sensor
rows from this new snapshot are used only to reconstruct recheck inputs.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import afternoon_direction_features as causal
import rain_weather_features as rain
import weather_session_features as richer
from forecast_clock import window_weights

ZONE = "Asia/Kuala_Lumpur"
VERSION = "ttdi_fixed_ride_window_recheck_inputs_v1"
CUTOFF_EXCLUSIVE = int(pd.Timestamp("2026-10-06", tz=ZONE).timestamp())
SOURCE_START = int(pd.Timestamp("2026-09-07", tz=ZONE).timestamp())
SENSOR_FEATURES = ("fresh", "freshOffset", "delta15", "delta30", "delta60", "delta120",
    "sd60", "mean3Offset", "temp", "rh", "tempDelta30", "rhDelta30", "sinTarget", "cosTarget",
    "targetLeadHours", "targetDurationHours")
WEATHER_ADDITIONS = ("weatherAgeHours", "priorRainMm", "duringRainMm", "priorRainProbMean",
    "duringRainProbMean", "firstRainLeadHours", "lastRainEndLeadHours", "wind10UMeanKmh",
    "wind10VMeanKmh", "wind10UDeltaKmh", "wind10VDeltaKmh", "gustHourlyEnvelopeMaxKmh",
    "gustHourlyEnvelopeDeltaKmh", "ventilationMeanM2S", "ventilationDeltaM2S", "boundaryLayerMeanM",
    "targetTempDeltaC", "targetRhDeltaPct")
FEATURES = (*SENSOR_FEATURES, *WEATHER_ADDITIONS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_hash(rows):
    value = hashlib.sha256()
    for row in rows:
        value.update(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        value.update(b"\n")
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def snapshot_sources(db_path, cutoff):
    """One read transaction; all three source clocks bounded before Oct 6."""
    if cutoff >= CUTOFF_EXCLUSIVE:
        raise ValueError("This historical preparation excludes all October 6 sources")
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(
            "SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch",
            (SOURCE_START - 86400, cutoff))]
        publications = [tuple(row) for row in conn.execute(
            "SELECT issued_epoch,sensor_epoch FROM dashboard_forecast_issues "
            "WHERE issued_epoch>=? AND issued_epoch<=? ORDER BY issued_epoch,issue_id",
            (SOURCE_START - 86400, cutoff))]
        forecasts = [tuple(row) for row in conn.execute(
            "SELECT fetched_epoch,source,payload FROM weather_forecast_runs "
            "WHERE fetched_epoch>=? AND fetched_epoch<=? ORDER BY fetched_epoch,payload",
            (SOURCE_START - 7200, cutoff))]
    revision = records_hash(forecasts)
    weather = rain.RunArchive(tuple(rain._parse_run(float(f), str(s), str(p)) for f, s, p in forecasts),
                              tuple(float(row[0]) for row in forecasts), float(cutoff), revision, str(path))
    manifest = dict(databaseReadOnly=True, singleReadTransaction=True, cutoffEpoch=int(cutoff),
        sensorRows=len(rows), publicationRows=len(publications), weatherRuns=len(forecasts),
        sensorRowsSha256=records_hash(rows), publicationRowsSha256=records_hash(publications),
        weatherRowsSha256=revision, maxSensorEpoch=max((r["epoch"] for r in rows), default=None),
        maxPublicationEpoch=max((r[0] for r in publications), default=None),
        maxWeatherFetchedEpoch=max((r[0] for r in forecasts), default=None))
    return rows, publications, weather, manifest


def sensor_cache(rows, cutoff):
    bounded = [dict(row) for row in rows if row["epoch"] <= cutoff]
    frame = causal.strict_frame(bounded, cutoff)
    local = causal.base_features(frame)
    raw = pd.DataFrame(bounded).sort_values("epoch")
    raw.index = pd.to_datetime(raw.epoch, unit="s", utc=True).dt.tz_convert(ZONE)
    maximum = raw.epoch.resample("15min", label="right", closed="right").max().rolling(12, min_periods=1).max()
    pm = raw.loc[np.isfinite(raw.pm02) & raw.pm02.ge(0)]
    return dict(local=local, featureMax=maximum, rawEpochs=pm.epoch.to_numpy(np.int64),
                rawPm=pm.pm02.to_numpy(float), cutoffEpoch=int(cutoff))


def extract_window_features(cache, weather, issue_epoch, start_epoch, end_epoch,
                            sensor_watermark_epoch, publication_epoch):
    """Pure supplied-input extractor for an exact fixed future two-hour window.

    It never reads an outcome. Callers can reuse it with current bounded source
    snapshots; the dated historical cutoff is enforced by snapshot_sources only.
    """
    issue, start, end = map(int, (issue_epoch, start_epoch, end_epoch))
    if not issue < start < end or end - start != 7200 or issue > cache["cutoffEpoch"]:
        raise ValueError("Require a two-hour future window and an available issue snapshot")
    anchor = issue // 900 * 900
    stamp = pd.Timestamp(anchor, unit="s", tz="UTC").tz_convert(ZONE)
    local = cache["local"].reindex([stamp]).iloc[0]
    record = {name: local.get(name, np.nan) for name in SENSOR_FEATURES}
    watermark = float(sensor_watermark_epoch) if sensor_watermark_epoch is not None else np.nan
    publication = float(publication_epoch) if publication_epoch is not None else np.nan
    errors = []
    if not np.isfinite(watermark) or watermark > issue: errors.append("invalid_sensor_watermark")
    if not np.isfinite(publication) or not 0 <= issue - publication <= 120:
        errors.append("missing_or_stale_sensor_publication")
    epochs, values = cache["rawEpochs"], cache["rawPm"]
    lo = np.searchsorted(epochs, issue - 300, side="right")
    hi = np.searchsorted(epochs, min(issue, watermark), side="right") if np.isfinite(watermark) else lo
    fresh = float(np.median(values[lo:hi])) if hi > lo else np.nan
    fresh_epoch = float(epochs[hi - 1]) if hi > lo else np.nan
    feature_max = cache["featureMax"].reindex([stamp]).iloc[0]
    if not np.isfinite(fresh): errors.append("missing_fresh_five_minute_reference")
    if not np.isfinite(fresh_epoch) or not 0 <= issue - fresh_epoch <= 240:
        errors.append("stale_fresh_sensor_reference")
    if not np.isfinite(feature_max) or not np.isfinite(watermark) or feature_max > watermark:
        errors.append("closed_features_not_available_at_sensor_watermark")
    closed = local.get("closedLevel", np.nan)
    if not np.isfinite(closed): errors.append("missing_closed_sensor_level")
    middle = pd.Timestamp((start + end) / 2, unit="s", tz="UTC").tz_convert(ZONE)
    clock_hours = middle.hour + middle.minute / 60 + middle.second / 3600
    record.update(fresh=fresh, freshOffset=fresh - closed, closedLevel=closed,
                  sinTarget=np.sin(clock_hours * np.pi / 12), cosTarget=np.cos(clock_hours * np.pi / 12),
                  targetLeadHours=(start - issue) / 3600, targetDurationHours=(end - start) / 3600)
    description = richer.describe_window(weather, issue, start, end)
    record.update({name: description["featureValues"].get(name, np.nan) for name in WEATHER_ADDITIONS})
    run = rain.latest_run(weather, issue)
    fetched = description.get("fetchedEpoch")
    if not description["available"]: errors.append("weather_window_unavailable")
    if fetched is None or not 0 <= issue - fetched <= 7200:
        errors.append("weather_not_available_at_recheck")
    weights = window_weights(start - anchor, end - anchor)
    complete = int(anchor + max(weights) * 900)
    record.update(issueEpoch=issue, originEpoch=anchor, startEpoch=start, endEpoch=end,
        completeEpoch=complete, sensorWatermarkEpoch=watermark, freshReferenceEpoch=fresh_epoch,
        featureSourceMaxEpoch=feature_max, publicationEpoch=publication,
        publicationAgeSeconds=issue - publication, weatherFetchedEpoch=fetched,
        weatherPayloadHash=None if run is None else run.payload_hash,
        weatherAvailable=bool(description["available"]), valid=not errors,
        unavailableReasons=errors, issueDay=str(pd.Timestamp(issue, unit="s", tz="UTC").tz_convert(ZONE).date()),
        targetDay=str(pd.Timestamp(complete, unit="s", tz="UTC").tz_convert(ZONE).date()))
    return record


def extract_current(rows, weather_run, issue, start, end, watermark, reference):
    """Pure actual-snapshot feature extraction for staged live inference.

    Return ((one-row frame, provenance), None) or (None, reason). The supplied
    measurement snapshot is being published now; no earlier dashboard
    publication is substituted for its actual watermark or fresh reference.
    """
    try:
        issue, start, end = map(int, (issue, start, end))
        if not np.isfinite(watermark) or watermark > issue:
            return None, "invalid_sensor_watermark"
        bounded = [dict(row) for row in rows if issue - 4 * 3600 <= row["epoch"] <= issue]
        if not bounded: return None, "missing_sensor_inputs"
        if isinstance(weather_run, rain.RunArchive):
            archive = weather_run
        else:
            if weather_run is None: return None, "missing_weather_run"
            archive = rain.RunArchive((weather_run,), (weather_run.fetched_epoch,), float(issue),
                                      weather_run.payload_hash, "supplied_current_weather_run")
        cache = sensor_cache(bounded, issue)
        record = extract_window_features(cache, archive, issue, start, end, watermark, issue)
        if not record["valid"]: return None, ";".join(record["unavailableReasons"])
        if not np.isfinite(reference) or reference < 0 or not np.isclose(record["fresh"], reference, rtol=0, atol=1e-7):
            return None, "fresh_sensor_reference_mismatch"
        record["planIssuedEpoch"] = start - 5400
        values = [clean(record[name]) for name in FEATURES]
        provenance = {key: clean(record[key]) for key in ("issueEpoch", "originEpoch", "startEpoch", "endEpoch",
            "completeEpoch", "sensorWatermarkEpoch", "freshReferenceEpoch", "featureSourceMaxEpoch",
            "weatherFetchedEpoch", "weatherPayloadHash")}
        provenance.update(featureColumns=list(FEATURES), featureValues=values,
            featureValuesSha256=hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest(),
            sourceRole="current_measurement_snapshot_for_publication",
            weatherAvailabilityProvenance="legacy_archived_fetch_timestamp",
            unvalidatedActualIssueCadence=True, sourceExtractorVersion=VERSION)
        return (pd.DataFrame([record]), provenance), None
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return None, "window_feature_extraction_" + type(error).__name__


def rebuild_training(readings, publications, weather_runs, manifest):
    """Pure daily training preparation for two snapshots of each fixed plan.

    Labels use supplied historical sensor medians only after every overlapping
    bucket closes. This helper is not invoked by the frozen-label historical
    recheck preparation below. Return valid, completed frames and exclusions.
    """
    cutoff = int(manifest["cutoffEpoch"])
    fit_cutoff = cutoff + 1
    earliest = int(manifest.get("earliestTrainingIssueEpoch", max(SOURCE_START, fit_cutoff - 28 * 86400)))
    if not readings or not publications or not len(weather_runs):
        return {name: pd.DataFrame() for name in ("initial", "recheck")}, dict(
            status="missing_source_inputs", validCases={"initial": 0, "recheck": 0})
    cache = sensor_cache(readings, cutoff)
    frame = causal.strict_frame(readings, cutoff)
    origins = frame.index[(frame.index.minute == 0) &
        (((frame.index.asi8 // 10**9 - SOURCE_START) // 3600) % 4 == 0) &
        (frame.index.asi8 // 10**9 >= earliest)]
    pub_epochs = np.asarray([p[0] for p in publications], np.int64)
    output = {name: [] for name in ("initial", "recheck")}
    excluded = {name: [] for name in output}
    complete_plans = 0
    incomplete_plans = 0
    for origin in origins:
        anchor = int(origin.timestamp())
        plan_issue = anchor + 300
        start, end = plan_issue + 5400, plan_issue + 12600
        weights = window_weights(start - anchor, end - anchor)
        labels = np.asarray([anchor + offset * 900 for offset in weights], np.int64)
        complete = int(labels.max())
        if complete >= fit_cutoff:
            incomplete_plans += 1
            continue
        index = pd.to_datetime(labels, unit="s", utc=True).tz_convert(ZONE)
        values = frame.pm02.reindex(index).to_numpy(float)
        if not np.isfinite(values).all():
            incomplete_plans += 1
            continue
        actual = float(values @ np.asarray(list(weights.values())))
        complete_plans += 1
        for name, offset in (("initial", 0), ("recheck", 1800)):
            issue = plan_issue + offset
            position = np.searchsorted(pub_epochs, issue, side="right") - 1
            pub, mark = publications[position] if position >= 0 else (None, None)
            record = extract_window_features(cache, weather_runs, issue, start, end, mark, pub)
            assert record["completeEpoch"] == complete and record["startEpoch"] == start and record["endEpoch"] == end
            record.update(planIssuedEpoch=plan_issue, originalIssueEpoch=plan_issue, actual=actual,
                          snapshotKind=name, trainingCutoffEpoch=fit_cutoff)
            if record["valid"]: output[name].append(record)
            else: excluded[name].append(dict(planIssuedEpoch=plan_issue, issueEpoch=issue,
                                             reasons=record["unavailableReasons"]))
    frames = {name: pd.DataFrame(records).sort_values("issueEpoch").reset_index(drop=True)
              if records else pd.DataFrame() for name, records in output.items()}
    for part in frames.values():
        if not part.empty:
            assert part.completeEpoch.lt(fit_cutoff).all() and not part.planIssuedEpoch.duplicated().any()
    return frames, dict(status="rebuilt_from_supplied_bounded_sources", trainingCutoffEpoch=fit_cutoff,
        completeCoveredPlans=complete_plans, incompletePlansExcluded=incomplete_plans,
        validCases={name: len(part) for name, part in frames.items()}, exclusions=excluded,
        sameAbsoluteWindowForInitialAndRecheck=True, labelDefinition="overlap_weighted_complete_15min_medians",
        providerRequests=0, databaseReads=0, modelFits=0, extractorVersion=VERSION)


def prepare(db_path=None):
    base_path = ROOT / "research/evaluated_models/base_features.pkl"
    original_path = ROOT / "research/first_crossing_hazard/predictions.pkl"
    base = pd.read_pickle(base_path)
    parents = base.loc[base.targetName.eq("mean90to210")].sort_values("issueEpoch").copy()
    if parents.issueEpoch.duplicated().any(): raise ValueError("Initial plans must have unique issues")
    original_ids = set(pd.read_pickle(original_path).issueEpoch)
    assert len(original_ids) == 73
    assert (parents.startEpoch - parents.issueEpoch).eq(5400).all()
    assert (parents.endEpoch - parents.issueEpoch).eq(12600).all()
    cutoff = min(CUTOFF_EXCLUSIVE - 1, int(parents.issueEpoch.max()) + 1800)
    rows, publications, weather, sources = snapshot_sources(db_path or ROOT / "bukit_kiara_air_history.db", cutoff)
    cache = sensor_cache(rows, cutoff)
    pub_epochs = np.asarray([p[0] for p in publications], np.int64)
    results = []
    for parent in parents.itertuples(index=False):
        issue = int(parent.issueEpoch) + 1800
        position = np.searchsorted(pub_epochs, issue, side="right") - 1
        pub, mark = publications[position] if position >= 0 else (None, None)
        record = extract_window_features(cache, weather, issue, parent.startEpoch, parent.endEpoch, mark, pub)
        assert record["startEpoch"] == parent.startEpoch and record["endEpoch"] == parent.endEpoch
        assert record["completeEpoch"] == parent.completeEpoch
        assert record["targetLeadHours"] == 1.0 and record["targetDurationHours"] == 2.0
        label_available = bool(np.isfinite(parent.actual) and parent.completeEpoch < CUTOFF_EXCLUSIVE)
        record.update(originalIssueEpoch=int(parent.issueEpoch), originalOriginEpoch=int(parent.originEpoch),
            originalIssueDay=parent.issueDay, initialReferencePm=float(parent.fresh),
            initialValid=bool(parent.valid), actual=float(parent.actual), labelAvailable=label_available,
            eligiblePaired=bool(parent.valid and label_available and record["valid"]),
            isOriginal73Query=int(parent.issueEpoch) in original_ids,
            isFixedFourHour=bool(((parent.originEpoch - SOURCE_START) // 3600) % 4 == 0),
            targetName="same_planned_mean90to210_recheck30", role="paired_recheck_inputs",
            labelSource="frozen_evaluated_models_base_actual_only")
        results.append(record)
    frame = pd.DataFrame(results)
    valid = frame.loc[frame.valid]
    assert valid.featureSourceMaxEpoch.le(valid.sensorWatermarkEpoch).all()
    assert valid.freshReferenceEpoch.le(valid.sensorWatermarkEpoch).all()
    assert valid.sensorWatermarkEpoch.le(valid.issueEpoch).all()
    assert (valid.issueEpoch - valid.freshReferenceEpoch).between(0, 240).all()
    assert valid.publicationAgeSeconds.between(0, 120).all()
    assert valid.weatherFetchedEpoch.le(valid.issueEpoch).all()
    assert (valid.issueEpoch - valid.weatherFetchedEpoch).between(0, 7200).all()
    assert set(FEATURES).issubset(frame.columns)
    output = HERE / "recheck" / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    frame.to_pickle(output / "features.pkl")
    frame.to_csv(output / "features.csv", index=False)
    invalid_counts = frame.unavailableReasons.explode().dropna().value_counts().to_dict()
    counts = {}
    for name, part in (("allParentPlans", frame), ("fixedFourHourPlans", frame.loc[frame.isFixedFourHour]),
                       ("original73Queries", frame.loc[frame.isOriginal73Query])):
        eligible = part.loc[part.eligiblePaired]
        above = eligible.loc[eligible.initialReferencePm.gt(70)]
        counts[name] = dict(parentPlans=len(part), recheckInputsAvailable=int(part.valid.sum()),
            initialValidAndLabelAvailable=int((part.initialValid & part.labelAvailable).sum()),
            pairedAvailable=len(eligible), pairedInitialAbove70=len(above),
            qualifyingRideWindowsAbove70=int(above.actual.le(70).sum()),
            distinctPairedTargetDates=int(eligible.targetDay.nunique()))
    manifest = dict(version=VERSION, preparedAtUtc=datetime.now(timezone.utc).isoformat(),
        sourceSnapshot=sources, featureColumns=list(FEATURES), outcomeReadFromNewSnapshot=False,
        labelsCopiedFromFrozenInitialArtifact=True, maximumFrozenLabelCompleteEpoch=int(frame.loc[frame.labelAvailable].completeEpoch.max()),
        sensorFeaturesAvailableAtRecheckOnly=True, initialFeaturesNotReplaced=True,
        recheckOffsetSeconds=1800, originalAbsoluteWindowPreserved=True,
        modelFits=0, providerRequests=0, databaseWrites=0, productionWrites=0,
        predictionRole="reconstructed_historical_inputs_not_originally_issued_recheck_predictions",
        counts=counts, unavailableReasons=invalid_counts,
        sourceArtifacts={str(path.relative_to(ROOT)): digest(path) for path in (base_path, original_path,
            ROOT / "research/evaluated_models/preparation.json", Path(__file__),
            ROOT / "afternoon_direction_features.py", ROOT / "rain_weather_features.py",
            ROOT / "weather_session_features.py", ROOT / "forecast_clock.py",
            ROOT / "window_pm_predictor.py", ROOT / "collection_quality.py")},
        outputHashes={name: digest(output / name) for name in ("features.pkl", "features.csv")},
        caveats=["Historical sensor receipt and revision history incomplete; published sensor watermark is the availability proxy.",
            "Legacy weather fetched_epoch is the only original receipt proxy in its table; no stronger receipt guarantee is claimed.",
            "Never use these recheck features in the initial forecast. Fit only when the copied full ride-window label closes strictly before the training cutoff.",
            "The threshold70 is a personal planning cutoff, not a health-safety statement."])
    (output / "manifest.json").write_text(json.dumps(clean(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "recheck" / "LATEST.json").write_text(json.dumps(dict(path=str(output), manifestSha256=digest(output / "manifest.json")), indent=2) + "\n", encoding="utf-8")
    return output, manifest


if __name__ == "__main__":
    destination, manifest = prepare()
    print(json.dumps(clean(dict(output=str(destination), counts=manifest["counts"], unavailableReasons=manifest["unavailableReasons"])), indent=2))
