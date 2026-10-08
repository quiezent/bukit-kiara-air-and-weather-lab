"""Frozen private first-crossing research; no live fitting or database writes.

Predict the FIRST covered proxy crossing of +/-10 ug/m3 over the next 90
minutes, rather than only the concentration remaining at the +90 endpoint.
Later opposite crossings are competing events and do not generate a second
cause. All dates in this replay have already been examined: this is development.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "research/evaluated_models"
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import afternoon_direction_features as causal
from forecast_clock import point_weights
import rain_weather_features as rain
import weather_session_features as weather

CLASSES = (-1, 0, 1)
NAMES = ("drop", "none", "rise")
LEADS = (15, 30, 45, 60, 75, 90)
LOCAL = ("fresh", "freshOffset", "delta15", "delta30", "delta60", "delta120",
         "sd60", "mean3Offset", "temp", "rh", "tempDelta30", "rhDelta30")
WEATHER = ("stepRainMm", "stepRainProbabilityPct", "stepWindUKmh", "stepWindVKmh",
           "stepWindUDeltaKmh", "stepWindVDeltaKmh", "stepGustEnvelopeKmh",
           "stepTempDeltaC", "stepRhDeltaPct", "stepVentilationM2S", "weatherAgeHours")
INDICATORS = tuple("step" + str(lead) for lead in LEADS)
FEATURES = (*LOCAL, *WEATHER, *INDICATORS)
LOGISTIC = {"C": 0.1, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4,
            "class_weight": None, "random_state": 1749}
POLICY = {
    "version": "ttdi_first_crossing_competing_risk_logistic_v1",
    "status": "retrospective_development_no_promotion",
    "timezone": "Asia/Kuala_Lumpur",
    "sourceStart": "2026-09-07", "evaluationStart": "2026-09-18", "evaluationEnd": "2026-10-05",
    "target": "First +/-10 ug/m3 crossing from same-issue fresh5 raw reference among exact issue+15,+30,+45,+60,+75,+90 covered 15-minute median point proxies; first cause is absorbing",
    "thresholdUgM3": 10.0, "largeThresholdUgM3": 20.0,
    "leadMinutes": list(LEADS), "classes": dict(zip(NAMES, CLASSES)),
    "trainingLookbackDays": 28,
    "fitCutoff": "Daily Malaysia midnight; ALL six required label supports complete strictly before cutoff",
    "fitCohort": "Only fixed four-hour issue origins anchored to original sourceStart midnight, plus original hourly issue validity and publication guards",
    "queryCohort": "The identical fixed four-hour grid; all six point proxies fully covered",
    "minimumFixedFourHourTrainingSupport": {"rise": 12, "drop": 12, "none": 50},
    "minimumCompletedTargetDays": 10,
    "minimumOriginalGuardedHourlyTrainingRows": 120,
    "minimumPooledRiskRows": 120,
    "riskRowsAreNotIndependentCases": True,
    "riskLikelihood": "Unweighted conditional hazard log likelihood: one row per at-risk step, stopping after the first cause. No inverse-length weights. Six rows from one issue do not count as six independent events.",
    "localFeatures": list(LOCAL), "weatherFeatures": list(WEATHER), "horizonIndicators": list(INDICATORS),
    "weatherInputs": "Latest original dashboard archived weather fetch <= issue and age <=2h; step-specific forecasts are from that run, including temperatures/RH/winds for future steps, never future observations",
    "weatherTemporalSemantics": "Rain amount and gust envelope refer to preceding hourly forecast blocks. Rain is apportioned uniformly into each future 15-minute interval. Hourly forecasts do not resolve observed minute-level storm onset.",
    "localAvailability": "Reused frozen cached issue features; featureSourceMaxEpoch and freshReferenceEpoch <= sensorWatermarkEpoch <= issue, reading age<=240s, publication age<=120s",
    "labelAvailability": "Missing any of six future proxy supports excludes the entire query, even when a first event would occur before the gap. No imputation or endpoint substitution.",
    "candidate": "Pooled three-class multinomial logistic conditional hazard; L2 regularization",
    "logisticParameters": LOGISTIC,
    "preprocessing": "Fit training medians with missing indicators; keep all-missing numeric columns; then StandardScaler. Six horizon indicators. Training transformations only.",
    "comparators": ["Per-step conditional cause hazard with one pseudocount for each of three classes, converted to cumulative incidence", "Unconditional cumulative first-cause frequencies by each step, with one pseudocount per class", "Always no crossing"],
    "directionCallProbability": 0.5,
    "experimentalEventRiskFlagProbability": 0.2,
    "experimentalFlagDefinition": "Sum of rise/drop cumulative incidence >=0.2. Binary risk flag only; not a direction or concentration claim.",
    "metrics": ["Cumulative multiclass Brier/log-loss at each 15-minute horizon", "Causewise ROC AUC and PR AUC when both classes exist", "Frozen0.5 direction precision/recall/false clearing/wrong way", "Frozen0.2 binary event-risk precision/recall/false alarms", "Prior30-minute abs(delta)<10 subset", ">=20 crossings descriptive support, not a20 model score", "Distinct dates and connected interval episode proxies"],
    "noLargeEventModel": "The10ug/m3 model does not predict a20ug/m3 threshold probability. Large/transient20 counts are descriptive only; dedicated20 model requires separate fixed training-support gates.",
    "unavailablePolicy": "No fit if any fixed support gate is unmet; preserve all unavailable folds and counts",
    "deployment": "No live replacement or warning from retrospective results. Freeze this unchanged10ug/m3 challenger before new issues starting October6 and evaluate on previously unexamined chronological outcomes.",
    "prospectiveGate": "At least10 new distinct target dates,5 spaced rise cases and5 spaced drop cases with connected-interval episode counts reported. Improve cumulative90-minute Brier AND log-loss against both nontrivial paired baselines; direction recall must exceed them with no higher false clearing/wrong way. At least5 predicted direction calls per cause before claiming directional precision; report counts, calibration and uncertainty. No numerical forecast promotion from this probability-only model.",
    "knownHistory": "Target design and28-day lookback were informed by previously seen outcome/support audits. This is retrospective development, not untouched validation or causal proof.",
    "sourceLimitations": ["Historical TTDI sensor records may be revised; original receipt/revision history incomplete", "Legacy weather fetch is an availability proxy; original weather receipt/revision provenance is incomplete", "Six target point proxies sample15-minute medians, not instantaneous extrema; excursions between proxies can be missed", "Four-hour spacing removes overlapping target supports but does not identify independent physical storms", "Neighbor PM omitted: unresolved future-dependent historical QC, corrected/raw definition and release timing"],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def save(path, value):
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def epoch(day):
    return int(pd.Timestamp(day, tz=POLICY["timezone"]).timestamp())


def freeze():
    """Declare fixed choices and input hashes before reconstructing/fitting labels."""
    HERE.mkdir(parents=True, exist_ok=True)
    declaration = {**POLICY,
        "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "baseFeaturesSha256": digest(INPUT / "base_features.pkl"),
        "preparationSha256": digest(INPUT / "preparation.json"),
        "originalProtocolSha256": digest(INPUT / "protocol.json"),
        "supportAuditSha256": digest(ROOT / "research/event_forecast_design/support_audit.json"),
        "runnerSha256": digest(__file__)}
    path = HERE / "policy.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        assert all(prior[k] == v for k, v in declaration.items() if k != "frozenAtUtc"), "Frozen policy differs"
        return prior
    save(path, declaration)
    return declaration


def checked_policy():
    policy = json.loads((HERE / "policy.json").read_text(encoding="utf-8"))
    assert all(policy[k] == v for k, v in POLICY.items())
    assert policy["runnerSha256"] == digest(__file__)
    assert policy["baseFeaturesSha256"] == digest(INPUT / "base_features.pkl")
    assert policy["preparationSha256"] == digest(INPUT / "preparation.json")
    return policy


def sparse_mask(frame):
    return ((frame.originEpoch - epoch(POLICY["sourceStart"])) // 3600) % 4 == 0


def first_crossing(outcomes, reference, threshold=10.0):
    """Return absorbing first cause/zero-based step; incomplete rows stay invalid."""
    outcomes = np.asarray(outcomes, float)
    reference = np.asarray(reference, float)
    if outcomes.ndim != 2 or outcomes.shape[1] != len(LEADS) or len(reference) != len(outcomes):
        raise ValueError("Expected one reference and six future proxies per issue")
    complete = np.isfinite(outcomes).all(axis=1) & np.isfinite(reference)
    delta = outcomes - reference[:, None]
    crossed = np.abs(delta) >= threshold
    first = np.where(complete & crossed.any(axis=1), crossed.argmax(axis=1), -1)
    cause = np.zeros(len(outcomes), dtype=int)
    active = first >= 0
    cause[active] = np.sign(delta[np.flatnonzero(active), first[active]]).astype(int)
    return cause, first, complete


def outcome_proxies(sensor, base):
    """Strict exact-clock point weights; the later touched bucket closes the label."""
    values, supports = [], []
    for lead in LEADS:
        vals, ends = [], []
        for row in base.itertuples(index=False):
            weights = point_weights(int(row.issueEpoch - row.originEpoch + lead * 60))
            labels = [int(row.originEpoch + k * 900) for k in weights]
            stamps = pd.to_datetime(labels, unit="s", utc=True).tz_convert(POLICY["timezone"])
            required = sensor.reindex(stamps).to_numpy(float)
            vals.append(float(required @ np.array(list(weights.values()))) if np.isfinite(required).all() else np.nan)
            ends.append(max(labels))
        values.append(vals); supports.append(ends)
    return np.asarray(values, float).T, np.asarray(supports, np.int64).T


def feature_clock_valid(frame):
    return (frame.featureSourceMaxEpoch.le(frame.sensorWatermarkEpoch) &
            frame.freshReferenceEpoch.le(frame.sensorWatermarkEpoch) &
            frame.sensorWatermarkEpoch.le(frame.issueEpoch) &
            (frame.issueEpoch - frame.freshReferenceEpoch).between(0, 240) &
            frame.publicationAgeSeconds.between(0, 120) &
            frame.weatherFetchedEpoch.le(frame.issueEpoch) &
            (frame.issueEpoch - frame.weatherFetchedEpoch).between(0, 7200))


def step_weather(run, issue, step):
    """All fields are sampled from the frozen issue-vintage FORECAST run."""
    parsed = weather._parsed(run)
    end = issue + LEADS[step] * 60
    start = end - 900
    rain_stats = weather._hourly_stats(parsed, "precipitation", start, end)
    probability_stats = weather._hourly_stats(parsed, "precipitation_probability", start, end)
    probability = probability_stats["total"] * 4 if probability_stats else None
    u, v = weather._sample(parsed, "u10", end), weather._sample(parsed, "v10", end)
    iu, iv = weather._sample(parsed, "u10", issue), weather._sample(parsed, "v10", issue)
    gusts = weather._hourly_stats(parsed, "wind_gusts_10m", start, end)
    t, rh = weather._sample(parsed, "temperature_2m", end), weather._sample(parsed, "relative_humidity_2m", end)
    it, irh = weather._sample(parsed, "temperature_2m", issue), weather._sample(parsed, "relative_humidity_2m", issue)
    def difference(a, b): return a - b if a is not None and b is not None else None
    return dict(zip(WEATHER, (
        rain_stats["total"] if rain_stats else None, probability, u, v,
        difference(u, iu), difference(v, iv), gusts["max"] if gusts else None,
        difference(t, it), difference(rh, irh), weather._ventilation(parsed, start, end),
        (issue - run.fetched_epoch) / 3600)))


def prepare():
    policy = checked_policy()
    prep = json.loads((INPUT / "preparation.json").read_text(encoding="utf-8"))
    cutoff = int(prep["cutoffEpoch"])
    data = pd.read_pickle(INPUT / "base_features.pkl")
    base = data.loc[data.valid & data.targetName.eq("point90")].sort_values("issueEpoch").copy().reset_index(drop=True)
    assert not base.issueEpoch.duplicated().any()
    assert feature_clock_valid(base).all()
    start = int(base.originEpoch.min()) - 12 * 900
    with closing(sqlite3.connect((ROOT / "bukit_kiara_air_history.db").as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        raw = [dict(row) for row in conn.execute("SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch", (start, cutoff))]
        weather_rows = [tuple(row) for row in conn.execute(
            "SELECT fetched_epoch,source,payload FROM weather_forecast_runs WHERE fetched_epoch>=? AND fetched_epoch<=? ORDER BY fetched_epoch,payload",
            (epoch(POLICY["sourceStart"]) - 7200, cutoff))]
    frame = causal.strict_frame(raw, cutoff)
    values, supports = outcome_proxies(frame.pm02, base)
    final_matched = np.isfinite(base.actual) & np.isfinite(values[:, -1])
    max_difference = float(np.max(np.abs(base.actual.to_numpy()[final_matched] - values[final_matched, -1])))
    assert max_difference < 1e-8, "Label reconstruction differs from original cached +90 outcomes"
    cause, first, complete = first_crossing(values, base.fresh, POLICY["thresholdUgM3"])
    cause20, first20, complete20 = first_crossing(values, base.fresh, POLICY["largeThresholdUgM3"])
    complete &= supports[:, -1] <= cutoff
    base["completeEpoch"] = supports[:, -1]
    base["cause"], base["firstStep"] = cause, first
    base["cause20"], base["firstStep20"] = cause20, first20
    base["endpointCause"] = np.select([values[:, -1] - base.fresh <= -10, values[:, -1] - base.fresh >= 10], [-1, 1], 0)
    base["endpointCause20"] = np.select([values[:, -1] - base.fresh <= -20, values[:, -1] - base.fresh >= 20], [-1, 1], 0)
    base["allSixCovered"] = complete
    for i, lead in enumerate(LEADS):
        base["outcome" + str(lead)], base["support" + str(lead)] = values[:, i], supports[:, i]
    base["targetDay"] = pd.to_datetime(base.completeEpoch, unit="s", utc=True).dt.tz_convert(POLICY["timezone"]).dt.date.astype(str)
    # Reproduce original bounded loader's exact window and SHA256 encoding;
    # generic rain.load_runs hashes the full archive with a different algorithm.
    weather_digest = hashlib.sha256()
    for fetched, source, payload in weather_rows:
        weather_digest.update(repr((fetched, source)).encode())
        weather_digest.update(payload.encode("utf-8"))
    archive = rain.RunArchive(tuple(rain._parse_run(float(f), str(s), str(p)) for f, s, p in weather_rows),
                              tuple(float(row[0]) for row in weather_rows), float(cutoff),
                              weather_digest.hexdigest(), str(ROOT / "bukit_kiara_air_history.db"))
    assert archive.revision == prep["weatherRevision"], "Original archived weather changed"
    features = []
    for row in base.itertuples(index=False):
        run = rain.latest_run(archive, int(row.issueEpoch))
        assert run is not None and run.fetched_epoch == row.weatherFetchedEpoch
        assert run.fetched_epoch <= row.issueEpoch and row.issueEpoch - run.fetched_epoch <= 7200
        for step, lead in enumerate(LEADS):
            record = {name: getattr(row, name) for name in LOCAL}
            record.update(step_weather(run, int(row.issueEpoch), step))
            record.update({name: float(i == step) for i, name in enumerate(INDICATORS)})
            record.update(issueEpoch=int(row.issueEpoch), step=step, leadMinutes=lead,
                          weatherFetchedEpoch=run.fetched_epoch, weatherPayloadHash=run.payload_hash)
            features.append(record)
    feature_frame = pd.DataFrame(features)
    assert len(feature_frame) == 6 * len(base)
    assert not set(FEATURES).intersection({"cause", "actual", "outcome15", "outcome90", "firstStep"})
    bundle = {"issues": base, "stepFeatures": feature_frame, "columns": list(FEATURES)}
    with (HERE / "prepared.pkl").open("wb") as stream: pickle.dump(bundle, stream)
    save(HERE / "preparation.json", {
        "preparedAtUtc": datetime.now(timezone.utc).isoformat(), "policySha256": digest(HERE / "policy.json"),
        "cutoffEpoch": cutoff, "databaseReadOnly": True, "inputBaseSha256": policy["baseFeaturesSha256"],
        "weatherRevision": archive.revision, "weatherRuns": len(archive.runs),
        "preparedSha256": digest(HERE / "prepared.pkl"), "originalValidPoint90Issues": len(base),
        "allSixCoveredIssues": int(complete.sum()), "fixedFourHourAllSixIssues": int((complete & sparse_mask(base)).sum()),
        "excludedIncompleteSixPointIssues": int((~complete).sum()),
        "originalFinalLabelMatchCount": int(final_matched.sum()), "maxAbsoluteFinalLabelDifference": max_difference,
        "localFeatureClockGuardsPass": bool(feature_clock_valid(base).all()),
        "stepFeatureRows": len(feature_frame), "featureColumns": list(FEATURES),
        "missingFraction": {name: float(feature_frame[name].isna().mean()) for name in FEATURES},
        "labelsAreFeatures": False, "limitation": POLICY["sourceLimitations"]})
    return bundle


def risk_rows(issues, steps):
    """Absorbing risk sets: no later observed PM enters predictors or fit rows."""
    keys = issues[["issueEpoch", "cause", "firstStep"]]
    out = steps.merge(keys, on="issueEpoch", how="inner", validate="many_to_one")
    out = out.loc[out.firstStep.eq(-1) | out.step.le(out.firstStep)].copy()
    out["riskLabel"] = np.where(out.step.eq(out.firstStep), out.cause, 0)
    return out.sort_values(["issueEpoch", "step"]).reset_index(drop=True)


def cumulative_incidence(hazards):
    """Conditional [drop,none,rise] hazards -> absorbing cumulative probabilities."""
    hazards = np.asarray(hazards, float)
    if hazards.ndim != 3 or hazards.shape[1:] != (6, 3): raise ValueError("Expected issue x6 x3 conditional hazards")
    if not np.isfinite(hazards).all() or (hazards < 0).any() or not np.allclose(hazards.sum(axis=2), 1, atol=1e-8):
        raise ValueError("Invalid conditional probability mass")
    result = np.empty_like(hazards)
    survival = np.ones(len(hazards))
    incidence = np.zeros((len(hazards), 2))
    for i in range(6):
        incidence += survival[:, None] * hazards[:, i, [0, 2]]
        survival *= hazards[:, i, 1]
        result[:, i, 0], result[:, i, 1], result[:, i, 2] = incidence[:, 0], survival, incidence[:, 1]
    assert np.allclose(result.sum(axis=2), 1, atol=1e-8)
    return result


def cumulative_truth(issues, step):
    return np.where(issues.firstStep.between(0, step), issues.cause, 0).astype(int)


def support(issues):
    return {name: int(issues.cause.eq(code).sum()) for name, code in zip(NAMES, CLASSES)}


def cohort_support(issues, cause_column="cause"):
    out = {"cases": len(issues), "distinctIssueDays": int(issues.issueDay.nunique()), "distinctTargetDays": int(issues.targetDay.nunique())}
    for name, code in zip(NAMES, CLASSES):
        selected = issues.loc[issues[cause_column].eq(code)].sort_values("issueEpoch")
        record = {"cases": len(selected), "distinctIssueDays": int(selected.issueDay.nunique())}
        if code and len(selected):
            starts, ends = selected.issueEpoch.to_numpy(), selected.completeEpoch.to_numpy()
            connected = np.r_[True, starts[1:] > np.maximum.accumulate(ends)[:-1]]
            record["connectedIntervalEpisodeProxyCount"] = int(connected.sum())
            prior = np.isfinite(selected.delta30) & selected.delta30.abs().ge(10)
            record["noSubstantialPrior30minSensorSignal"] = int((np.isfinite(selected.delta30) & ~prior).sum())
            record["alreadyObservedSameDirection"] = int((prior & selected.delta30.mul(code).gt(0)).sum())
            record["alreadyObservedOppositeDirection"] = int((prior & selected.delta30.mul(code).lt(0)).sum())
        out[name] = record
    return out


def fit_hazard(risk):
    estimator = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), LogisticRegression(**LOGISTIC))
    x = risk[list(FEATURES)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    with threadpool_limits(limits=1): estimator.fit(x, risk.riskLabel.to_numpy(int))
    assert set(estimator.classes_) == set(CLASSES)
    return estimator


def predictions(estimator, query_steps):
    ordered = query_steps.sort_values(["issueEpoch", "step"])
    assert ordered.groupby("issueEpoch").step.apply(list).map(lambda steps: steps == list(range(6))).all()
    x = ordered[list(FEATURES)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    with threadpool_limits(limits=1): p = estimator.predict_proba(x)
    classes = list(estimator.classes_)
    return np.column_stack([p[:, classes.index(code)] for code in CLASSES]).reshape(-1, 6, 3)


def training_baselines(train, risk, count):
    per_step, unconditional = [], []
    for step in range(6):
        at_risk = risk.loc[risk.step.eq(step), "riskLabel"]
        h = np.array([(at_risk == code).sum() + 1 for code in CLASSES], float)
        per_step.append(h / h.sum())
        truth = cumulative_truth(train, step)
        p = np.array([(truth == code).sum() + 1 for code in CLASSES], float)
        unconditional.append(p / p.sum())
    hazards = np.tile(np.array(per_step)[None, :, :], (count, 1, 1))
    probabilities = np.tile(np.array(unconditional)[None, :, :], (count, 1, 1))
    return hazards, probabilities


def direction_calls(probabilities):
    p = np.asarray(probabilities)
    called = np.zeros(len(p), int)
    drop, rise = p[:, 0] >= .5, p[:, 2] >= .5
    called[drop & ~rise], called[rise & ~drop] = -1, 1
    # Exact0.5/0.5 ties have no resolved direction.
    return called


def metrics(issues, p, step):
    if issues.empty: return {"count": 0}
    truth = cumulative_truth(issues, step)
    p = np.asarray(p, float)
    assert p.shape == (len(issues), 3) and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
    onehot = np.column_stack([truth == code for code in CLASSES]).astype(float)
    selected = onehot.argmax(axis=1)
    called = direction_calls(p)
    event_flag = p[:, 0] + p[:, 2] >= .2
    actual_event = truth != 0
    out = {"count": len(issues), "distinctIssueDays": int(issues.issueDay.nunique()), "distinctTargetDays": int(issues.targetDay.nunique()),
           "multiclassBrier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
           "logLoss": float(-np.log(np.clip(p[np.arange(len(p)), selected], 1e-12, 1)).mean()),
           "actualEvents": int(actual_event.sum()), "correctDirectionCalls": int(((truth == called) & actual_event).sum()),
           "falseClearingCalls": int(((called == -1) & (truth != -1)).sum()),
           "wrongWayCalls": int((called * truth == -1).sum()), "uncalibrated": True,
           "binaryRiskFlagAt0_2": {"called": int(event_flag.sum()), "correct": int((event_flag & actual_event).sum()),
               "falseAlarmCount": int((event_flag & ~actual_event).sum()),
               "recall": float((event_flag & actual_event).sum() / actual_event.sum()) if actual_event.any() else None,
               "precision": float((event_flag & actual_event).sum() / event_flag.sum()) if event_flag.any() else None}}
    for i, (name, code) in enumerate(zip(NAMES, CLASSES)):
        actual, calls = truth == code, called == code
        good = actual & calls
        rec = {"actualCount": int(actual.sum()), "predictedDirectionCallCount": int(calls.sum()) if code else None,
               "correctDirectionCallCount": int(good.sum()) if code else None,
               "recallAt0_5": float(good.sum() / actual.sum()) if code and actual.any() else None,
               "precisionAt0_5": float(good.sum() / calls.sum()) if code and calls.any() else None,
               "classwiseBrier": float(np.mean((p[:, i] - actual) ** 2)),
               "rocAuc": float(roc_auc_score(actual, p[:, i])) if 0 < actual.sum() < len(actual) else None,
               "prAucAveragePrecision": float(average_precision_score(actual, p[:, i])) if 0 < actual.sum() < len(actual) else None,
               "eventPrevalence": float(actual.mean()), "actualDistinctIssueDays": int(issues.loc[actual, "issueDay"].nunique())}
        bins = np.minimum(9, (p[:, i] * 10).astype(int))
        rec["calibrationBins"] = [{"lower": b / 10, "upper": (b + 1) / 10,
            "count": int((bins == b).sum()), "actualCount": int((actual & (bins == b)).sum()),
            "meanProbability": float(p[bins == b, i].mean()) if (bins == b).any() else None,
            "actualFrequency": float(actual[bins == b].mean()) if (bins == b).any() else None} for b in range(10)]
        out[name] = rec
    return out


def run():
    checked_policy()
    preparation = json.loads((HERE / "preparation.json").read_text(encoding="utf-8"))
    assert preparation["preparedSha256"] == digest(HERE / "prepared.pkl")
    bundle = pd.read_pickle(HERE / "prepared.pkl")
    issues, steps = bundle["issues"], bundle["stepFeatures"]
    assert tuple(bundle["columns"]) == FEATURES
    issues = issues.loc[issues.allSixCovered].copy()
    assert feature_clock_valid(issues).all()
    started = perf_counter()
    folds, outputs, hazards_out = [], [], []
    model_names = ("logistic_hazard", "step_empirical_hazard", "unconditional_cause_frequency", "always_none")
    for date in pd.date_range(POLICY["evaluationStart"], POLICY["evaluationEnd"]):
        day = str(date.date()); cutoff = epoch(day)
        hourly = issues.loc[issues.completeEpoch.lt(cutoff) & issues.issueEpoch.ge(cutoff - 28 * 86400) & issues.issueEpoch.lt(cutoff)]
        train = hourly.loc[sparse_mask(hourly)].sort_values("issueEpoch").copy()
        query = issues.loc[issues.issueDay.eq(day) & sparse_mask(issues)].sort_values("issueEpoch").copy()
        risk = risk_rows(train, steps)
        counts = support(train)
        reasons = []
        if len(hourly) < 120: reasons.append("fewer_than120_original_guarded_hourly_training_issues")
        if train.targetDay.nunique() < 10: reasons.append("fewer_than10_completed_target_dates")
        if len(risk) < 120: reasons.append("fewer_than120_pooled_at_risk_rows_not_independent_cases")
        for name, minimum in POLICY["minimumFixedFourHourTrainingSupport"].items():
            if counts[name] < minimum: reasons.append("insufficient_spaced_" + name + "_training_cases")
        fold = {"day": day, "fitCutoffEpoch": cutoff, "originalGuardedHourlyTrainingIssues": len(hourly),
                "fixedFourHourTrainingIssues": len(train), "trainingSupport": counts,
                "completedTargetDays": int(train.targetDay.nunique()), "riskRows": len(risk),
                "queryIssues": len(query), "querySupport": support(query), "eligible": not reasons,
                "unavailableReasons": reasons, "maxTrainingOutcomeEpoch": int(train.completeEpoch.max()) if len(train) else None,
                "trainingSupportByCauseAndStep": {str(LEADS[i]): {name: int(((risk.step == i) & (risk.riskLabel == code)).sum()) for name, code in zip(NAMES, CLASSES)} for i in range(6)}}
        folds.append(fold)
        if reasons or query.empty:
            print(day, "unavailable" if reasons else "no_queries", counts, flush=True); continue
        assert train.completeEpoch.lt(cutoff).all()
        for cohort in (train, query):
            if len(cohort) > 1:
                assert cohort.originEpoch.diff().dropna().ge(14400).all()
                assert (cohort.issueEpoch.to_numpy()[1:] >= cohort.completeEpoch.to_numpy()[:-1]).all()
        estimator = fit_hazard(risk)
        query_steps = steps.loc[steps.issueEpoch.isin(query.issueEpoch)]
        fitted_h = predictions(estimator, query_steps)
        empirical_h, empirical_p = training_baselines(train, risk, len(query))
        ps = {"logistic_hazard": cumulative_incidence(fitted_h),
              "step_empirical_hazard": cumulative_incidence(empirical_h),
              "unconditional_cause_frequency": empirical_p,
              "always_none": np.tile(np.array([0., 1., 0.])[None, None, :], (len(query), 6, 1))}
        keep = ["issueEpoch", "originEpoch", "issueDay", "targetDay", "completeEpoch", "fresh", "delta30",
                "sensorWatermarkEpoch", "featureSourceMaxEpoch", "freshReferenceEpoch", "weatherFetchedEpoch"]
        # Save prediction-only rows before scorer opens or attaches query labels.
        for step, lead in enumerate(LEADS):
            out = query[keep].copy(); out["fitCutoffEpoch"] = cutoff; out["step"] = step; out["leadMinutes"] = lead
            for model in model_names:
                for i, name in enumerate(NAMES): out[model + "_" + name] = ps[model][:, step, i]
            outputs.append(out)
        h = query_steps[["issueEpoch", "step", "leadMinutes", "weatherFetchedEpoch", "weatherPayloadHash"]].sort_values(["issueEpoch", "step"]).copy()
        for model, values in (("logistic_hazard", fitted_h), ("step_empirical_hazard", empirical_h)):
            for i, name in enumerate(NAMES): h[model + "_" + name] = values[:, :, i].reshape(-1)
        hazards_out.append(h)
        print(day, "fitted", counts, "riskRows", len(risk), "queries", len(query), flush=True)
    predictions_only = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    conditional = pd.concat(hazards_out, ignore_index=True) if hazards_out else pd.DataFrame()
    predictions_only.to_pickle(HERE / "predictions.pkl")
    conditional.to_pickle(HERE / "conditional_hazards.pkl")
    save(HERE / "folds.json", folds)
    save(HERE / "prediction_provenance.json", {"savedBeforeScoringUtc": datetime.now(timezone.utc).isoformat(),
        "predictionsSha256": digest(HERE / "predictions.pkl"), "conditionalHazardsSha256": digest(HERE / "conditional_hazards.pkl"),
        "predictionRowsContainLabels": False, "fittedDailyFolds": sum(f["eligible"] and f["queryIssues"] > 0 for f in folds),
        "fitElapsedSeconds": perf_counter() - started, "policySha256": digest(HERE / "policy.json")})
    return predictions_only


def score():
    policy = checked_policy()
    provenance = json.loads((HERE / "prediction_provenance.json").read_text(encoding="utf-8"))
    assert provenance["predictionsSha256"] == digest(HERE / "predictions.pkl")
    predictions_only = pd.read_pickle(HERE / "predictions.pkl")
    # Explicit label reload occurs only AFTER prediction-only artifact was persisted.
    bundle = pd.read_pickle(HERE / "prepared.pkl")
    labels = bundle["issues"].loc[bundle["issues"].allSixCovered]
    score_started = datetime.now(timezone.utc).isoformat()
    models = ("logistic_hazard", "step_empirical_hazard", "unconditional_cause_frequency", "always_none")
    metrics_all, metrics_onset = {}, {}
    for i, lead in enumerate(LEADS):
        p = predictions_only.loc[predictions_only.step.eq(i)].sort_values("issueEpoch")
        q = p.merge(labels[["issueEpoch", "cause", "firstStep", "cause20", "firstStep20", "endpointCause", "endpointCause20"]], on="issueEpoch", how="left", validate="one_to_one")
        metrics_all[str(lead)] = {name: metrics(q, q[[name + "_" + c for c in NAMES]].to_numpy(float), i) for name in models}
        onset = q.loc[np.isfinite(q.delta30) & q.delta30.abs().lt(10)]
        metrics_onset[str(lead)] = {name: metrics(onset, onset[[name + "_" + c for c in NAMES]].to_numpy(float), i) for name in models}
    evaluated = predictions_only.loc[predictions_only.step.eq(5), ["issueEpoch"]].merge(labels, on="issueEpoch", how="left", validate="one_to_one")
    large = {"allEvaluated": cohort_support(evaluated, "cause20"),
             "notRemainingAt90Endpoint": int((evaluated.cause20.ne(0) & evaluated.endpointCause20.eq(0)).sum()),
             "oppositeEndpointCause": int((evaluated.cause20.mul(evaluated.endpointCause20) == -1).sum()),
             "firstCrossStepCounts": {str(LEADS[i]): int(evaluated.firstStep20.eq(i).sum()) for i in range(6)},
             "descriptiveOnly": True, "modelIs10Not20": True}
    folds = json.loads((HERE / "folds.json").read_text(encoding="utf-8"))
    result = {"status": POLICY["status"], "policySha256": digest(HERE / "policy.json"),
              "scoringStartedUtc": score_started, "predictionProvenance": provenance,
              "foldCount": len(folds), "eligibleFolds": sum(f["eligible"] for f in folds),
              "fittedFolds": sum(f["eligible"] and f["queryIssues"] > 0 for f in folds),
              "evaluatedCohortSupport": cohort_support(evaluated),
              "fullCoveredHistoricalSupport": cohort_support(labels.loc[sparse_mask(labels)]),
              "first10CrossingNotRemainingAt90Endpoint": int((evaluated.cause.ne(0) & evaluated.endpointCause.eq(0)).sum()),
              "large20CrossingDescriptiveSupport": large,
              "metricsByHorizon": metrics_all, "noSubstantialPrior30minSignalMetricsByHorizon": metrics_onset,
              "limitations": POLICY["sourceLimitations"], "interpretation": "Uncalibrated retrospective research probabilities; no live deployment, no20threshold probability claim, no causal rain inference"}
    save(HERE / "results.json", result)
    return result


def prospective():
    policy = checked_policy()
    save(HERE / "prospective_policy.json", {
        "version": "ttdi_first_crossing_prospective_v1", "frozenAtUtc": policy["frozenAtUtc"],
        "researchPolicySha256": digest(HERE / "policy.json"), "startsWithIssueDay": "2026-10-06",
        "candidate": "logistic_hazard", "lookbackDays": 28,
        "supportGates": POLICY["minimumFixedFourHourTrainingSupport"],
        "minimumCompletedTargetDays": 10, "minimumPooledRiskRows": 120,
        "parameters": LOGISTIC, "features": list(FEATURES),
        "targets": POLICY["target"], "directionCallProbability": .5, "binaryEventRiskFlagProbability": .2,
        "promotionGate": POLICY["prospectiveGate"],
        "availability": "Original sensor publication watermark and weather fetch guard; retain forecast payload receipt+record time before each issue when available. No future observations or backfilled run substitutions.",
        "execution": "Offline probability challenger only. Predictions computed later from archived issued inputs must be labeled issued-input replay; never label them originally issued forecasts.",
        "deployment": "No primary concentration replacement. No live warning until prospective precision/calibration and false-clearing gates are established.",
        "researchDates": "Previously examined September18-October5 remains development, excluded from prospective scoring"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "run", "score", "prospective"))
    args = parser.parse_args()
    if args.phase == "freeze":
        declaration = freeze(); print(json.dumps({"frozen": True, "policySha256": digest(HERE / "policy.json")}))
    elif args.phase == "prepare":
        bundle = prepare(); print(json.dumps({"prepared": True, "issues": len(bundle["issues"])}))
    elif args.phase == "run": print(json.dumps({"predictionRows": len(run())}))
    elif args.phase == "score":
        result = score(); print(json.dumps({"fittedFolds": result["fittedFolds"], "support": result["evaluatedCohortSupport"], "ninetyMinuteMetrics": {name: {k: v for k, v in m.items() if k in ("count", "multiclassBrier", "logLoss", "correctDirectionCalls", "falseClearingCalls", "wrongWayCalls", "binaryRiskFlagAt0_2")} for name, m in result["metricsByHorizon"]["90"].items()}}, indent=2))
    else: prospective(); print(json.dumps({"frozenProspectivePolicy": True}))
