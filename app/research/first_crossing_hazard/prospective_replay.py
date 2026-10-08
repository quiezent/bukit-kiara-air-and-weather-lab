"""Offline, issue-input replay of the unchanged Oct6 first-crossing challenger.

Rebuilds guarded inputs from a bounded read-only snapshot of the live archive.
Never calls providers, starts collectors, fits in HTTP requests, or changes live
forecasts. Cached development features are NOT used as future query features.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sqlite3
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_first_crossing_hazard", HERE / "hazard_trial.py")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
import numpy as np
import pandas as pd
import sklearn

FIRST_QUERY_DAY = "2026-10-06"
MODELS = ("logistic_hazard", "step_empirical_hazard", "unconditional_cause_frequency", "always_none")
DECLARATION = {
    "version": "ttdi_first_crossing_prospective_replay_entry_point_v1",
    "startsWithIssueDay": FIRST_QUERY_DAY,
    "role": "Offline issued-input replay, NOT an originally issued forecast or live warning",
    "inputRebuild": "Bounded read-only live sensor/publication/weather archive snapshot; no provider fetches, backfill or development feature reuse",
    "fitWindowDays": 28, "issueOffsetSeconds": 300,
    "issueGrid": "Hourly origins plus300s; train/query fixed4hour grid anchored to original2026-09-07 Malaysia midnight",
    "queryCompleteness": "All six future point proxy supports closed by --cutoff; no partial label scoring",
    "strictTrainingCutoff": "Every six-proxy label support strictly closed before query day's Malaysia midnight",
    "parameters": h.LOGISTIC, "features": list(h.FEATURES),
    "supportGates": h.POLICY["minimumFixedFourHourTrainingSupport"],
    "minimumCompletedTargetDays": 10, "minimumOriginalGuardedHourlyTrainingRows": 120,
    "minimumPooledRiskRows": 120,
    "directionCallProbability": .5, "binaryEventRiskFlagProbability": .2,
    "likelihood": h.POLICY["riskLikelihood"],
    "labelRule": h.POLICY["target"],
    "duplicateRunRule": "Same issueEpoch/leadMinutes across different runs is the SAME forecast case; never sum duplicate runs as independent validation",
    "sourceLimits": [
        "Sensor measurement time/publication watermark do not prove unrevised historical row first availability; original sensor first receipts and revisions are incomplete",
        "Original dashboard weather fetched_epoch is a legacy availability proxy; per-payload first receipt/record timestamps are absent for this table",
        "This harness uses only original archived forecast payloads and does not substitute newly downloaded past runs, revised reanalysis, peers or future weather observations",
        "The reconstructed issue-feature guards reproduce the declared protocol, but receipt/revision limitations prevent calling this pristine originally issued validation",
        "Four-hour disjoint label intervals are not proof of independent physical storms",
    ],
}


def source_hashes():
    sources = (HERE / "hazard_trial.py", ROOT / "afternoon_direction_features.py",
               ROOT / "forecast_clock.py", ROOT / "rain_weather_features.py",
               ROOT / "weather_session_features.py", ROOT / "window_pm_predictor.py",
               ROOT / "forecast_freshness.py", ROOT / "collection_quality.py")
    return {p.name: h.digest(p) for p in sources}


def freeze_entry_point():
    """Fix harness/prospective policy hashes before inspecting new-date outcomes."""
    h.checked_policy()
    prospective = json.loads((HERE / "prospective_policy.json").read_text(encoding="utf-8"))
    assert prospective["startsWithIssueDay"] == FIRST_QUERY_DAY
    assert prospective["parameters"] == h.LOGISTIC
    assert prospective["features"] == list(h.FEATURES)
    assert prospective["supportGates"] == h.POLICY["minimumFixedFourHourTrainingSupport"]
    record = {**DECLARATION,
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "prospectivePolicySha256": h.digest(HERE / "prospective_policy.json"),
        "researchPolicySha256": h.digest(HERE / "policy.json"),
        "entryPointSha256": h.digest(__file__), "helperSourceSha256": source_hashes(),
        "libraryVersions": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    path = HERE / "prospective_entry_point_policy.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        assert all(prior[k] == v for k, v in record.items() if k != "frozenAtUtc"), "Frozen prospective harness changed"
        return prior
    h.save(path, record)
    return record


def checked_entry_point():
    path = HERE / "prospective_entry_point_policy.json"
    if not path.exists(): raise RuntimeError("Freeze entry point with --freeze-only before rebuilding prospective inputs")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert all(record[k] == v for k, v in DECLARATION.items())
    assert record["entryPointSha256"] == h.digest(__file__)
    assert record["helperSourceSha256"] == source_hashes()
    assert record["prospectivePolicySha256"] == h.digest(HERE / "prospective_policy.json")
    assert record["researchPolicySha256"] == h.digest(HERE / "policy.json")
    assert record["libraryVersions"] == {"python": sys.version.split()[0], "numpy": np.__version__,
                                         "pandas": pd.__version__, "sklearn": sklearn.__version__}
    return record


def parse_cutoff(value, now=None):
    now = int(time.time()) if now is None else int(now)
    if value is None: return now
    try: result = int(value)
    except (TypeError, ValueError):
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize(h.POLICY["timezone"]) if stamp.tzinfo is None else stamp
        result = int(stamp.timestamp())
    if result > now: raise ValueError("--cutoff must not request a future archive state")
    if result < h.epoch(h.POLICY["sourceStart"]): raise ValueError("--cutoff precedes declared sensor sourceStart")
    return result


def query_start_day(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError("--issue-start must be a calendar date")
    day = stamp.strftime("%Y-%m-%d")
    if day < FIRST_QUERY_DAY: raise ValueError("Prospective query issue-start cannot precede October6")
    return day


def _hash_records(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_snapshot(cutoff, issue_start):
    """One SQLite read transaction, bounded by cutoff; no source/API writes."""
    earliest_training_issue = max(h.epoch(h.POLICY["sourceStart"]), h.epoch(issue_start) - 28 * 86400)
    sensor_start = earliest_training_issue - 86400
    weather_start = earliest_training_issue - 7200
    db = ROOT / "bukit_kiara_air_history.db"
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        conn.row_factory = sqlite3.Row
        readings = [dict(row) for row in conn.execute(
            "SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch", (sensor_start, cutoff))]
        publications = [tuple(row) for row in conn.execute(
            "SELECT issued_epoch,sensor_epoch FROM dashboard_forecast_issues WHERE issued_epoch>=? AND issued_epoch<=? ORDER BY issued_epoch,issue_id",
            (sensor_start, cutoff))]
        weather_rows = [tuple(row) for row in conn.execute(
            "SELECT fetched_epoch,source,payload FROM weather_forecast_runs WHERE fetched_epoch>=? AND fetched_epoch<=? ORDER BY fetched_epoch,payload",
            (weather_start, cutoff))]
    runs = h.rain.RunArchive(tuple(h.rain._parse_run(float(f), str(s), str(p)) for f, s, p in weather_rows),
                            tuple(float(row[0]) for row in weather_rows), float(cutoff),
                            _hash_records(weather_rows), str(db))
    manifest = {"cutoffEpoch": cutoff, "capturedAtUtc": datetime.now(timezone.utc).isoformat(),
                "databaseReadOnly": True, "singleReadTransaction": True, "providerFetches": 0,
                "earliestTrainingIssueEpoch": earliest_training_issue, "sensorSourceStartEpoch": sensor_start,
                "weatherSourceStartEpoch": weather_start, "sensorRows": len(readings),
                "publicationRows": len(publications), "weatherRuns": len(weather_rows),
                "sensorRowsSha256": _hash_records(readings), "publicationRowsSha256": _hash_records(publications),
                "weatherRowsSha256": runs.revision,
                "maxSensorEpoch": max((r["epoch"] for r in readings), default=None),
                "maxPublicationEpoch": max((r[0] for r in publications), default=None),
                "maxWeatherFetchedEpoch": max((r[0] for r in weather_rows), default=None)}
    assert all(manifest[k] is None or manifest[k] <= cutoff for k in ("maxSensorEpoch", "maxPublicationEpoch", "maxWeatherFetchedEpoch"))
    return readings, publications, runs, manifest


def rebuild_inputs(readings, publications, runs, manifest):
    """Rebuild trailing local features and label proxies, not old feature caches."""
    cutoff = int(manifest["cutoffEpoch"])
    if not readings or not publications or not len(runs):
        return {"issues": pd.DataFrame(), "stepFeatures": pd.DataFrame(), "columns": list(h.FEATURES)}, {
            "status": "missing_source_inputs", "validIssues": 0, "allSixCoveredIssues": 0}
    frame = h.causal.strict_frame(readings, cutoff)
    origins = frame.index[(frame.index.minute == 0) &
                          (frame.index.asi8 // 10**9 >= manifest["earliestTrainingIssueEpoch"])]
    origin_epochs = origins.asi8 // 10**9
    issues = origin_epochs + 300
    in_cutoff = issues <= cutoff
    origins, origin_epochs, issues = origins[in_cutoff], origin_epochs[in_cutoff], issues[in_cutoff]
    local = h.causal.base_features(frame).reindex(origins).copy().reset_index(drop=True)
    raw = pd.DataFrame(readings).sort_values("epoch")
    raw.index = pd.to_datetime(raw.epoch, unit="s", utc=True).dt.tz_convert(h.POLICY["timezone"])
    max_sources = raw.epoch.resample("15min", label="right", closed="right").max().rolling(12, min_periods=1).max()
    pmraw = raw.loc[np.isfinite(raw.pm02) & raw.pm02.ge(0)]
    raw_epochs, raw_pm = pmraw.epoch.to_numpy(np.int64), pmraw.pm02.to_numpy(float)
    pub_epochs = np.array([p[0] for p in publications], dtype=np.int64)
    pub_marks = np.array([p[1] if p[1] is not None else np.nan for p in publications], dtype=float)
    positions = np.searchsorted(pub_epochs, issues, side="right") - 1
    marks, pubages, fresh, stamps = [np.full(len(issues), np.nan) for _ in range(4)]
    good = positions >= 0
    marks[good], pubages[good] = pub_marks[positions[good]], issues[good] - pub_epochs[positions[good]]
    for i, (issue, mark) in enumerate(zip(issues, marks)):
        if not np.isfinite(mark): continue
        lo = np.searchsorted(raw_epochs, issue - 300, side="right")
        hi = np.searchsorted(raw_epochs, min(issue, mark), side="right")
        if hi > lo: fresh[i], stamps[i] = np.median(raw_pm[lo:hi]), raw_epochs[hi - 1]
    descriptions = h.weather.describe_windows(runs, ((int(issue), int(issue + 5400), int(issue + 5400)) for issue in issues))
    base = local.copy()
    base["fresh"], base["freshOffset"] = fresh, fresh - base.closedLevel
    base["issueEpoch"], base["originEpoch"] = issues, origin_epochs
    base["sensorWatermarkEpoch"], base["freshReferenceEpoch"] = marks, stamps
    base["featureSourceMaxEpoch"] = max_sources.reindex(origins).to_numpy(float)
    base["publicationAgeSeconds"] = pubages
    base["weatherFetchedEpoch"] = [d.get("fetchedEpoch") for d in descriptions]
    base["weatherAvailable"] = [d["available"] for d in descriptions]
    base["valid"] = np.isfinite(fresh) & np.isfinite(base.closedLevel) & base.weatherAvailable & h.feature_clock_valid(base)
    base["issueDay"] = pd.to_datetime(base.issueEpoch, unit="s", utc=True).dt.tz_convert(h.POLICY["timezone"]).dt.date.astype(str)
    guarded = base.loc[base.valid].sort_values("issueEpoch").copy().reset_index(drop=True)
    assert h.feature_clock_valid(guarded).all()
    values, supports = h.outcome_proxies(frame.pm02, guarded)
    cause, first, complete = h.first_crossing(values, guarded.fresh, 10)
    cause20, first20, _ = h.first_crossing(values, guarded.fresh, 20)
    guarded["cause"], guarded["firstStep"] = cause, first
    guarded["cause20"], guarded["firstStep20"] = cause20, first20
    guarded["completeEpoch"] = supports[:, -1]
    guarded["allSixCovered"] = complete & guarded.completeEpoch.le(cutoff)
    guarded["targetDay"] = pd.to_datetime(guarded.completeEpoch, unit="s", utc=True).dt.tz_convert(h.POLICY["timezone"]).dt.date.astype(str)
    guarded["endpointCause"] = np.select([values[:, -1] - guarded.fresh <= -10, values[:, -1] - guarded.fresh >= 10], [-1, 1], 0)
    guarded["endpointCause20"] = np.select([values[:, -1] - guarded.fresh <= -20, values[:, -1] - guarded.fresh >= 20], [-1, 1], 0)
    for i, lead in enumerate(h.LEADS):
        guarded["outcome" + str(lead)], guarded["support" + str(lead)] = values[:, i], supports[:, i]
    features = []
    for row in guarded.itertuples(index=False):
        run = h.rain.latest_run(runs, int(row.issueEpoch))
        assert run is not None and run.fetched_epoch == row.weatherFetchedEpoch
        assert run.fetched_epoch <= row.issueEpoch and row.issueEpoch - run.fetched_epoch <= 7200
        for step, lead in enumerate(h.LEADS):
            record = {name: getattr(row, name) for name in h.LOCAL}
            record.update(h.step_weather(run, int(row.issueEpoch), step))
            record.update({name: float(i == step) for i, name in enumerate(h.INDICATORS)})
            record.update(issueEpoch=int(row.issueEpoch), step=step, leadMinutes=lead,
                          weatherFetchedEpoch=run.fetched_epoch, weatherPayloadHash=run.payload_hash)
            features.append(record)
    feature_frame = pd.DataFrame(features)
    assert len(feature_frame) == len(guarded) * 6
    return {"issues": guarded, "stepFeatures": feature_frame, "columns": list(h.FEATURES)}, {
        "status": "rebuilt_from_bounded_original_archive", "hourlyOrigins": len(base),
        "validIssues": len(guarded), "allSixCoveredIssues": int(guarded.allSixCovered.sum()),
        "incompleteSixPointIssuesExcluded": int((~guarded.allSixCovered).sum()),
        "clockGuardFailures": int((~h.feature_clock_valid(base)).sum()),
        "stepFeatureRows": len(feature_frame), "featureColumns": list(h.FEATURES),
        "developmentCachedFeaturesUsed": False, "providerBackfillUsed": False,
        "missingFraction": {name: float(feature_frame[name].isna().mean()) for name in h.FEATURES} if len(feature_frame) else {}}


def eligible_folds(bundle, issue_start, cutoff):
    """Expose availability without fitting when no completed new-date query exists."""
    issues, steps = bundle["issues"], bundle["stepFeatures"]
    if issues.empty: return [], []
    issues = issues.loc[issues.allSixCovered].copy()
    last_day = pd.Timestamp(cutoff, unit="s", tz="UTC").tz_convert(h.POLICY["timezone"]).date()
    folds, jobs = [], []
    for date in pd.date_range(issue_start, last_day):
        day = str(date.date()); fit_cutoff = h.epoch(day)
        hourly = issues.loc[issues.completeEpoch.lt(fit_cutoff) &
                            issues.issueEpoch.ge(fit_cutoff - 28 * 86400) & issues.issueEpoch.lt(fit_cutoff)]
        train = hourly.loc[h.sparse_mask(hourly)].sort_values("issueEpoch").copy()
        query = issues.loc[issues.issueDay.eq(day) & h.sparse_mask(issues) & issues.completeEpoch.le(cutoff)].sort_values("issueEpoch").copy()
        risk = h.risk_rows(train, steps)
        counts = h.support(train)
        reasons = []
        if len(hourly) < 120: reasons.append("fewer_than120_original_guarded_hourly_training_issues")
        if train.targetDay.nunique() < 10: reasons.append("fewer_than10_completed_target_dates")
        if len(risk) < 120: reasons.append("fewer_than120_pooled_at_risk_rows_not_independent_cases")
        for name, minimum in h.POLICY["minimumFixedFourHourTrainingSupport"].items():
            if counts[name] < minimum: reasons.append("insufficient_spaced_" + name + "_training_cases")
        fold = {"day": day, "fitCutoffEpoch": fit_cutoff, "originalGuardedHourlyTrainingIssues": len(hourly),
                "fixedFourHourTrainingIssues": len(train), "trainingSupport": counts,
                "completedTargetDays": int(train.targetDay.nunique()), "riskRows": len(risk),
                "queryIssues": len(query), "eligible": not reasons, "unavailableReasons": reasons,
                "maxTrainingOutcomeEpoch": int(train.completeEpoch.max()) if len(train) else None,
                "predictionRole": DECLARATION["role"]}
        folds.append(fold)
        if reasons or query.empty: continue
        assert day >= FIRST_QUERY_DAY and train.completeEpoch.lt(fit_cutoff).all()
        for cohort in (train, query):
            assert h.feature_clock_valid(cohort).all()
            if len(cohort) > 1:
                assert cohort.originEpoch.diff().dropna().ge(14400).all()
                assert (cohort.issueEpoch.to_numpy()[1:] >= cohort.completeEpoch.to_numpy()[:-1]).all()
        jobs.append((fold, train, query, risk))
    return folds, jobs


def fit_predictions(bundle, jobs):
    records, conditional_records = [], []
    steps = bundle["stepFeatures"]
    for fold, train, query, risk in jobs:
        estimator = h.fit_hazard(risk)
        query_steps = steps.loc[steps.issueEpoch.isin(query.issueEpoch)]
        fitted_h = h.predictions(estimator, query_steps)
        empirical_h, empirical_p = h.training_baselines(train, risk, len(query))
        ps = {"logistic_hazard": h.cumulative_incidence(fitted_h), "step_empirical_hazard": h.cumulative_incidence(empirical_h),
              "unconditional_cause_frequency": empirical_p,
              "always_none": np.tile(np.array([0., 1., 0.])[None, None, :], (len(query), 6, 1))}
        keep = ["issueEpoch", "originEpoch", "issueDay", "targetDay", "completeEpoch", "fresh", "delta30",
                "sensorWatermarkEpoch", "featureSourceMaxEpoch", "freshReferenceEpoch", "weatherFetchedEpoch"]
        for step, lead in enumerate(h.LEADS):
            out = query[keep].copy(); out["fitCutoffEpoch"] = fold["fitCutoffEpoch"]
            out["step"], out["leadMinutes"] = step, lead
            out["predictionRole"] = "issued_input_replay_not_originally_issued"
            for model in MODELS:
                for i, name in enumerate(h.NAMES): out[model + "_" + name] = ps[model][:, step, i]
            records.append(out)
        out = query_steps[["issueEpoch", "step", "leadMinutes", "weatherFetchedEpoch", "weatherPayloadHash"]].sort_values(["issueEpoch", "step"]).copy()
        for model, values in (("logistic_hazard", fitted_h), ("step_empirical_hazard", empirical_h)):
            for i, name in enumerate(h.NAMES): out[model + "_" + name] = values[:, :, i].reshape(-1)
        conditional_records.append(out)
    return (pd.concat(records, ignore_index=True) if records else pd.DataFrame(),
            pd.concat(conditional_records, ignore_index=True) if conditional_records else pd.DataFrame())


def score_saved(output):
    """Reload labels only after prediction-only file/hash/timestamp persistence."""
    provenance = json.loads((output / "prediction_provenance.json").read_text(encoding="utf-8"))
    assert provenance["predictionsSha256"] == h.digest(output / "predictions.pkl")
    prediction = pd.read_pickle(output / "predictions.pkl")
    bundle = pd.read_pickle(output / "prepared.pkl")
    if prediction.empty:
        return {"status": "no_completed_new_date_queries", "queryCases": 0,
                "newDateEvaluationStarts": FIRST_QUERY_DAY, "modelFits": 0,
                "predictionRole": DECLARATION["role"], "metrics": {},
                "scoringStartedUtc": datetime.now(timezone.utc).isoformat()}
    assert prediction.issueDay.ge(FIRST_QUERY_DAY).all()
    labels = bundle["issues"].loc[bundle["issues"].allSixCovered]
    all_metrics, onset_metrics = {}, {}
    for step, lead in enumerate(h.LEADS):
        p = prediction.loc[prediction.step.eq(step)].sort_values("issueEpoch")
        q = p.merge(labels[["issueEpoch", "cause", "firstStep", "cause20", "firstStep20", "endpointCause", "endpointCause20"]], on="issueEpoch", how="left", validate="one_to_one")
        assert q.cause.notna().all()
        all_metrics[str(lead)] = {name: h.metrics(q, q[[name + "_" + c for c in h.NAMES]].to_numpy(float), step) for name in MODELS}
        prior = q.loc[np.isfinite(q.delta30) & q.delta30.abs().lt(10)]
        onset_metrics[str(lead)] = {name: h.metrics(prior, prior[[name + "_" + c for c in h.NAMES]].to_numpy(float), step) for name in MODELS}
    scored = prediction.loc[prediction.step.eq(5), ["issueEpoch"]].merge(labels, on="issueEpoch", validate="one_to_one")
    assert scored.issueDay.ge(FIRST_QUERY_DAY).all()
    return {"status": "prospective_date_issued_input_replay_uncalibrated_no_deployment",
            "queryCases": len(scored), "predictionRole": DECLARATION["role"],
            "cohortSupport": h.cohort_support(scored), "metricsByHorizon": all_metrics,
            "noSubstantialPrior30minSignalMetricsByHorizon": onset_metrics,
            "large20DescriptiveSupport": h.cohort_support(scored, "cause20"),
            "large20ModelFitted": False,
            "large20FirstCrossingNotRemainingAt90": int((scored.cause20.ne(0) & scored.endpointCause20.eq(0)).sum()),
            "scoringStartedUtc": datetime.now(timezone.utc).isoformat(),
            "limitations": DECLARATION["sourceLimits"],
            "duplicateRunRule": DECLARATION["duplicateRunRule"]}


def run(cutoff, issue_start=FIRST_QUERY_DAY, output_root=None):
    declaration = checked_entry_point()
    cutoff, issue_start = parse_cutoff(cutoff), query_start_day(issue_start)
    now_day = pd.Timestamp(cutoff, unit="s", tz="UTC").tz_convert(h.POLICY["timezone"]).strftime("%Y-%m-%d")
    run_name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ") + "-cutoff-" + str(cutoff)
    output = (Path(output_root) if output_root else HERE / "prospective") / now_day / run_name
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    h.save(output / "run_policy.json", {"cutoffEpoch": cutoff, "queryIssueStart": issue_start,
        "entryPointPolicySha256": h.digest(HERE / "prospective_entry_point_policy.json"),
        "prospectivePolicySha256": declaration["prospectivePolicySha256"],
        "researchPolicySha256": declaration["researchPolicySha256"], "role": DECLARATION["role"],
        "writesRestrictedToRunDirectory": True})
    readings, publications, weather, manifest = source_snapshot(cutoff, issue_start)
    h.save(output / "source_snapshot.json", manifest)
    bundle, preparation = rebuild_inputs(readings, publications, weather, manifest)
    with (output / "prepared.pkl").open("wb") as stream: pickle.dump(bundle, stream)
    h.save(output / "preparation.json", {**preparation, "preparedSha256": h.digest(output / "prepared.pkl"),
        "databaseReadOnly": True, "developmentCachedFeaturesUsed": False,
        "firstReceiptAndRevisionLimits": DECLARATION["sourceLimits"]})
    folds, jobs = eligible_folds(bundle, issue_start, cutoff)
    h.save(output / "folds.json", folds)
    predictions, hazards = fit_predictions(bundle, jobs)
    predictions.to_pickle(output / "predictions.pkl")
    hazards.to_pickle(output / "conditional_hazards.pkl")
    h.save(output / "prediction_provenance.json", {"savedBeforeScoringUtc": datetime.now(timezone.utc).isoformat(),
        "predictionsSha256": h.digest(output / "predictions.pkl"),
        "conditionalHazardsSha256": h.digest(output / "conditional_hazards.pkl"),
        "predictionRowsContainLabels": False, "originallyIssuedForecasts": False,
        "role": DECLARATION["role"], "modelFits": len(jobs), "elapsedSecondsBeforeScoring": time.perf_counter() - started})
    result = score_saved(output)
    result.update(cutoffEpoch=cutoff, fittedFolds=len(jobs), foldCount=len(folds),
                  availableCompletedNewDateQueries=sum(f["queryIssues"] for f in folds),
                  entryPointPolicySha256=h.digest(HERE / "prospective_entry_point_policy.json"),
                  prospectivePolicySha256=declaration["prospectivePolicySha256"],
                  runtimeSeconds=time.perf_counter() - started)
    if not jobs and any(f["queryIssues"] for f in folds):
        result["status"] = "completed_new_date_queries_but_support_gates_unmet_no_model_fit"
        result["unavailableReasons"] = sorted({reason for f in folds for reason in f["unavailableReasons"]})
    h.save(output / "results.json", result)
    h.save(output / "artifact_manifest.json", {p.name: h.digest(p) for p in output.iterdir() if p.is_file()})
    return output, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true", help="Freeze harness hashes without reading live query outcomes")
    parser.add_argument("--cutoff", help="Epoch seconds or ISO timestamp; naive timestamps use Malaysia time. Default: now")
    parser.add_argument("--issue-start", default=FIRST_QUERY_DAY, help="First query issue date, >=2026-10-06")
    parser.add_argument("--output-root", type=Path, help="Optional research run directory root")
    args = parser.parse_args()
    if args.freeze_only:
        record = freeze_entry_point(); print(json.dumps({"frozen": True, "policySha256": h.digest(HERE / "prospective_entry_point_policy.json")}))
    else:
        cutoff = parse_cutoff(args.cutoff)
        output, result = run(cutoff, query_start_day(args.issue_start), args.output_root)
        print(json.dumps({"output": str(output), "status": result["status"], "queryCases": result["queryCases"],
                          "fittedFolds": result["fittedFolds"], "runtimeSeconds": result["runtimeSeconds"]}, indent=2))
