"""Actual-issued validation ledger and bounded, explicit prospective scoring.

No fitting, network calls, or automatic confirmation looks occur here. Ordinary
writers use the caller's transaction and never commit. Explicit confirmation
consumption commits its atomic authorization marker before any label read, on a
connection with no pending caller transaction. Readiness reads ledger metadata,
not sensor labels. Old archives, protocols, and development scores are untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from forecast_backtest import covered_bucket_medians, point_weights, window_weights, weighted_target
from forecast_payload import encode_payload, loads

VERSION = "actual_issued_validation_v2"
TARGET_VERSION = "decision_clock_weighted_15min_v1"
LOCAL = timezone(timedelta(hours=8))
SEALED_EPOCH = int(datetime(2026, 10, 20, tzinfo=LOCAL).timestamp())
DEFAULT_POLICY_PATH = Path(__file__).with_name("research") / "audit_remediation_20261008" / "validation_policy.json"
FAMILIES = ("point90", "mean90_210", "morningSession", "afternoonSession")
HEADS = {"point90": ("point90",), "mean90_210": ("mean90_210",),
         "morningSession": ("morning", "nextMorning"), "afternoonSession": ("afternoon", "nextAfternoon")}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _epoch(value):
    number = _finite(value)
    return int(number) if number is not None and number == int(number) else None


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _date(epoch):
    return datetime.fromtimestamp(epoch, LOCAL).date().isoformat()


def _has_table(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def init_schema(conn):
    """Only our new namespace; the caller owns locking and commit."""
    conn.execute("""CREATE TABLE IF NOT EXISTS prospective_validation_issued(
        case_id TEXT PRIMARY KEY, recorded_epoch INTEGER NOT NULL,
        issue_epoch INTEGER NOT NULL, target_family TEXT NOT NULL, head TEXT NOT NULL,
        cohort TEXT NOT NULL, target_date TEXT, start_epoch INTEGER, end_epoch INTEGER,
        complete_epoch INTEGER, candidate_version TEXT NOT NULL, actual_model TEXT NOT NULL,
        input_hash TEXT, cohort_id TEXT NOT NULL, payload BLOB NOT NULL)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS prospective_validation_issued_lookup
        ON prospective_validation_issued(target_family,cohort,candidate_version,issue_epoch,recorded_epoch)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS prospective_validation_scores(
        case_id TEXT PRIMARY KEY, first_scored_epoch INTEGER NOT NULL,
        actual REAL NOT NULL, label_snapshot_hash TEXT NOT NULL,
        target_version TEXT NOT NULL, FOREIGN KEY(case_id) REFERENCES prospective_validation_issued(case_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS prospective_validation_reports(
        report_id TEXT PRIMARY KEY, as_of_epoch INTEGER NOT NULL, confirmation INTEGER NOT NULL,
        policy_hash TEXT NOT NULL, payload BLOB NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS prospective_validation_looks(
        target_family TEXT NOT NULL, look_number INTEGER NOT NULL,
        candidate_version TEXT NOT NULL, cohort TEXT NOT NULL, registered_epoch INTEGER NOT NULL,
        issue_start_epoch INTEGER NOT NULL, issue_end_epoch INTEGER NOT NULL,
        policy_hash TEXT NOT NULL, selection_evidence_hash TEXT NOT NULL,
        status TEXT NOT NULL, report_id TEXT, evidence_role TEXT NOT NULL DEFAULT 'operational',
        raw_model_version TEXT, raw_numerical_policy TEXT, PRIMARY KEY(target_family,look_number))""")
    columns = {r[1] for r in conn.execute("PRAGMA table_info(prospective_validation_looks)")}
    for name, declaration in (("evidence_role", "TEXT NOT NULL DEFAULT 'operational'"),
                              ("raw_model_version", "TEXT"), ("raw_numerical_policy", "TEXT")):
        if name not in columns:
            conn.execute("ALTER TABLE prospective_validation_looks ADD COLUMN " + name + " " + declaration)


def default_policy():
    """Rules are a measurement amendment, never a fitted/model-selected recipe."""
    end = int(datetime(2026, 12, 29, tzinfo=LOCAL).timestamp())
    return {"version": VERSION, "targetVersion": TARGET_VERSION,
            "sealedFromEpoch": SEALED_EPOCH, "confirmationIssueStartEpoch": SEALED_EPOCH,
            "confirmationIssueEndEpochExclusive": end,
            "familyAlpha": 0.05, "targetFamilies": list(FAMILIES), "maximumLooks": 3,
            "perTargetLookAlpha": 0.05 / (4 * 3),
            "pairedIntervalConfidence": 1 - 0.05 / (4 * 3),
            "bootstrapDraws": 10000, "bootstrapSeed": 20261008,
            "blockCalendarDays": {"point90": 1, "mean90_210": 1, "morningSession": 7, "afternoonSession": 1},
            "minimumMaeGainUgM3": {"point90": 1.3, "mean90_210": 1.4, "morningSession": 2.2, "afternoonSession": 5.3},
            "peakThresholdUgM3": {"point90": 117.04, "mean90_210": 105.93041666666666,
                                  "morningSession": 102.69625, "afternoonSession": 86.345},
            "minimumDistinctTargetDates": 70, "minimumIndependentDateBlocks": 10,
            "minimumPeakDatesPerHead": 5, "minimumLargeChangeDatesPerHead": 5,
            "minimumRecall20": 0.5, "maximumAbsoluteBiasUgM3": 5.0,
            "minimumCalibrationDates": 30, "minimumProbabilityEventDates": 5,
            "maximumPublicationLagSeconds": 120, "canonicalIssueToleranceSeconds": 120,
            "canonicalIssueClocksMYT": {"morning": "07:15", "afternoon": "12:00", "nextDay": "18:05"},
            "nearSampling": "first publication per fixed MYT four-hour issue block, unavailable first retained",
            "movingSampling": "first publication per head and fixed MYT four-hour issue block, unavailable first retained",
            "canonicalSampling": "first publication within the fixed +/-120-second canonical issue band (Morning07:15,Afternoon12:00,nextDay18:05 MYT), unavailable first retained; original clocks never snapped",
            "confirmationCandidates": "one immutable raw originally-issued candidate model and numerical policy per target per indexed look, within a versioned integrated recipe; fallback/unavailable outputs remain in cohort accounting",
            "lookAccounting": "three indexed campaign looks, at most four target tests per look; failed, insufficient, or errored begun looks count",
            "readiness": "metadata alone never establishes prediction skill or calibration",
            "labelMeaning": "covered 15-minute bucket-median proxy or exact weighted window mean, not raw instantaneous PM or first crossing"}


def load_policy(path=None):
    policy = json.loads(Path(path or DEFAULT_POLICY_PATH).read_text(encoding="utf-8"))
    _validate_policy(policy)
    return policy


def _validate_policy(policy):
    reference = default_policy()
    # These define the experiment family and source protection. A caller cannot
    # quietly loosen them by passing a custom scoring dictionary.
    for key in reference:
        if policy.get(key) != reference[key]:
            raise ValueError("Unregistered validation policy change: " + key)


def _input_manifest(payload, prediction):
    snapshot = payload.get("inputProvenance") or payload.get("materializedInputSnapshot") or {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    nested = prediction.get("patchtstModel") or prediction.get("weatherSessionModel") or {}
    if not isinstance(nested, dict):
        nested = {}
    materialized = nested.get("materializedSnapshotSha256")
    snapshot_hash = next((snapshot.get(k) for k in ("snapshotSha256", "sha256", "payloadSha256", "manifestSha256", "snapshotId")
                          if isinstance(snapshot.get(k), str) and len(snapshot[k]) == 64), None)
    input_hash = snapshot_hash or (materialized if isinstance(materialized, str) and len(materialized) == 64 else None)
    return {"inputSnapshot": snapshot, "inputHash": input_hash,
            "queryTensorSha256": nested.get("queryTensorSha256"),
            "featureValuesSha256": prediction.get("featureValuesSha256"),
            "materializedSnapshotSha256": materialized,
            "sensorWatermarkEpoch": prediction.get("sensorWatermarkEpoch"),
            "inputSnapshotReceivedEpoch": prediction.get("inputSnapshotReceivedEpoch"),
            "featureSourceMaxEpoch": prediction.get("featureSourceMaxEpoch"),
            "inputAvailabilityPolicy": prediction.get("inputAvailabilityPolicy") or nested.get("liveInputAvailabilityPolicy"),
            "originalReceiptKnown": bool(snapshot.get("allReceiptsKnown") is True and
                                          snapshot.get("reconstructionComplete") is True and
                                          snapshot.get("unknownReceiptInputs") == 0 and snapshot_hash),
            "completeAllModelInputReplay": snapshot.get("completeAllModelInputReplay") is True,
            "candidateInputReplayComplete": snapshot.get("candidateInputReplayComplete") is True}


def _session_cohort(issue, start, end, head, tolerance=120):
    if start is None or end is None or issue is None:
        return "moving"
    day = datetime.fromtimestamp(start, LOCAL)
    issue_day = datetime.fromtimestamp(issue, LOCAL).date()
    if day.date() not in (issue_day, issue_day + timedelta(days=1)):
        return "unsupported_horizon"
    same_day = _date(issue) == _date(start)
    canonical_hour = 9 if head in ("morning", "nextMorning") else 14
    canonical_start = int(day.replace(hour=canonical_hour, minute=0, second=0, microsecond=0).timestamp())
    issue_local = datetime.fromtimestamp(issue, LOCAL)
    clock_hour, clock_minute = ((7, 15) if canonical_hour == 9 else (12, 0)) if same_day else (18, 5)
    clock = int(issue_local.replace(hour=clock_hour, minute=clock_minute, second=0, microsecond=0).timestamp())
    return "canonical" if start == canonical_start and end - start == 7200 and abs(issue - clock) <= tolerance else "moving"


def _forecast_rows(payload, recorded, build, candidate_versions):
    issue = _epoch(payload.get("forecastIssuedEpoch"))
    air = payload.get("airWindow") or {}
    clock = air.get("forecastClock") or {}
    clock_ok = (clock.get("targetVersion") == TARGET_VERSION and
                _epoch(clock.get("forecastIssuedEpoch")) == issue)
    start = _epoch(clock.get("arrivalTargetEpoch")) if clock_ok else None
    end = _epoch(clock.get("windowEndEpoch")) if clock_ok else None
    policy = payload.get("forecastPolicy") or {}
    selected_sessions = policy.get("windowForecastModels") if isinstance(policy, dict) else {}
    selected_sessions = selected_sessions if isinstance(selected_sessions, dict) else {}
    definitions = [("point90", "point90", "scheduled_near", air.get("arrival") or {}, start, start),
                   ("mean90_210", "mean90_210", "scheduled_near", air.get("trail") or {}, start, end)]
    windows = payload.get("windows") or {}
    for key in ("morning", "nextMorning", "afternoon", "nextAfternoon"):
        window = windows.get(key)
        if not isinstance(window, dict):
            if key.startswith("next"):
                continue
            window = {}
        a, b = _epoch(window.get("startEpoch")), _epoch(window.get("endEpoch"))
        head = key
        if a is not None and issue is not None:
            head = ("nextMorning" if "morning" in key.lower() else "nextAfternoon") if _date(a) != _date(issue) else (
                "morning" if "morning" in key.lower() else "afternoon")
        family = "morningSession" if "morning" in head.lower() else "afternoonSession"
        definitions.append((family, head, _session_cohort(issue, a, b, head), window.get("particleForecast") or {}, a, b))
    for family, head, cohort, prediction, a, b in definitions:
        point = _finite(prediction.get("point", prediction.get("mean")))
        # Only archived numerical baseline, never a newer sensor reading or an
        # event threshold. Persistence itself has an exact identical baseline.
        baseline = _finite(prediction.get("baselinePoint", prediction.get("sensorAnchor")))
        if baseline is None and prediction.get("pointRole") == "persistence_anchor":
            baseline = point
        actual_model = str(prediction.get("modelVersion") or "unavailable")
        selection = prediction.get("modelSelection") or {}
        session_selected = (selected_sessions.get(head) or selected_sessions.get(
            "morning" if family == "morningSession" else "afternoon")) if family.endswith("Session") else None
        selected = (candidate_versions.get(family) or session_selected or
                    (selection.get("selectedModelVersion") if isinstance(selection, dict) else None) or actual_model)
        identity = _input_manifest(payload, prediction)
        fallback = {k: prediction.get(k) for k in ("experimentalXgboostFallback", "experimentalPatchtstFallback") if k in prediction}
        reasons = []
        if issue is None or not clock_ok and family in ("point90", "mean90_210"):
            reasons.append("missing_or_inconsistent_issue_clock")
        if a is None or b is None or issue is None or a <= recorded or b < a:
            reasons.append("invalid_or_already_started_target")
        if family in ("point90", "mean90_210") and issue is not None and (a != issue + 5400 or b != issue + (5400 if family == "point90" else 12600)):
            reasons.append("target_does_not_match_exact_decision_clock")
        if family in ("point90", "mean90_210") and _epoch(clock.get("windowStartEpoch")) != a:
            reasons.append("inconsistent_near_window_start")
        if family.endswith("Session") and a is not None and b is not None and b - a != 7200:
            reasons.append("session_target_not_two_hours")
        if cohort == "unsupported_horizon":
            reasons.append("session_target_outside_same_or_next_day")
        child_issue = _epoch(prediction.get("forecastIssuedEpoch"))
        if child_issue is not None and child_issue != issue:
            reasons.append("selected_output_issue_clock_mismatch")
        if issue is not None and not 0 <= recorded - issue <= 120:
            reasons.append("publication_clock_invalid")
        if not prediction.get("available", family in ("point90", "mean90_210")) or point is None or baseline is None:
            reasons.append("unavailable_point_or_original_baseline")
        if point is not None and point < 0 or baseline is not None and baseline < 0:
            reasons.append("nonphysical_negative_concentration")
        snapshot_issue = _finite(identity["inputSnapshot"].get("forecastEpoch"))
        if snapshot_issue is not None and issue is not None and snapshot_issue > issue:
            identity["originalReceiptKnown"] = False
        candidate = prediction.get("candidateForecast") or {}
        if not isinstance(candidate, dict):
            candidate = {}
        candidate_mean = _finite(candidate.get("mean", candidate.get("prediction")))
        candidate_issue = _epoch(candidate.get("forecastedAtEpoch", candidate.get("forecastIssuedEpoch")))
        candidate_start = _epoch(candidate.get("startEpoch"))
        candidate_end = _epoch(candidate.get("endEpoch"))
        candidate_model = candidate.get("modelVersion")
        candidate_policy = candidate.get("numericalPolicyIdentifier") or candidate.get("numericalPolicy")
        diagnostic_reasons = []
        if candidate_mean is None or candidate_mean < 0 or candidate.get("available") is False:
            diagnostic_reasons.append("raw_candidate_unavailable_or_nonphysical")
        if not candidate_model or not candidate_policy:
            diagnostic_reasons.append("raw_candidate_identity_missing")
        if candidate_issue != issue or candidate_start != a or candidate_end != b:
            diagnostic_reasons.append("raw_candidate_clock_or_target_mismatch")
        target_reasons = [r for r in reasons if r not in ("unavailable_point_or_original_baseline", "nonphysical_negative_concentration", "selected_output_issue_clock_mismatch")]
        if target_reasons or baseline is None or baseline < 0:
            diagnostic_reasons.append("raw_candidate_pair_or_target_unavailable")
        diagnostic = {"evidenceRole": "candidate_diagnostic", "available": not diagnostic_reasons,
            "mean": candidate_mean, "modelVersion": candidate_model, "numericalPolicyIdentifier": candidate_policy,
            "forecastIssuedEpoch": candidate_issue, "startEpoch": candidate_start, "endEpoch": candidate_end,
            "pointRole": candidate.get("pointRole") or "original_issued_candidate_mean_before_operational_gate",
            "rangeLow": _finite(candidate.get("rawRangeLow")), "rangeHigh": _finite(candidate.get("rawRangeHigh")),
            "unavailableReasons": diagnostic_reasons}
        complete = None if a is None or b is None else (math.ceil((a if family == "point90" else b) / 900) * 900)
        row = {"version": VERSION, "issueEpoch": issue, "recordedEpoch": recorded,
               "targetFamily": family, "head": head, "cohort": cohort,
               "targetVersion": TARGET_VERSION, "startEpoch": a, "endEpoch": b,
               "completeEpoch": complete, "targetDate": _date(a) if a is not None else None,
               "issueDate": _date(issue if issue is not None else recorded),
               "issueOffsetSeconds": issue % 900 if issue is not None else None,
               "modelVersion": actual_model, "candidateVersion": str(selected), "dashboardBuild": build,
               "point": point, "baselinePoint": baseline, "available": not reasons,
               "targetEligible": not target_reasons, "candidateDiagnostic": diagnostic,
               "unavailableReasons": reasons, "fallback": fallback, "modelSelection": selection,
               "pointRole": prediction.get("pointRole"), "inputManifest": identity,
               "rangeLow": _finite(prediction.get("rangeLow")), "rangeHigh": _finite(prediction.get("rangeHigh")),
               "uncertaintyMethod": prediction.get("uncertaintyMethod"), "probability": None,
               "probabilityContract": None, "probabilityModelVersion": None}
        cycling = air.get("cyclingWindowForecast") or {}
        if family == "mean90_210" and cycling.get("available") and (
            _epoch(cycling.get("startEpoch")) == a and _epoch(cycling.get("endEpoch")) == b and
            _epoch(cycling.get("forecastIssuedEpoch")) == issue and _finite(cycling.get("cutoffUgM3")) == 70):
            probability = _finite(cycling.get("chanceMeanAtOrBelowCutoff"))
            if probability is not None and 0 <= probability <= 1:
                row.update(probability=probability, probabilityContract="exact_window_mean_le70",
                           probabilityModelVersion=cycling.get("modelVersion"))
        row["cohortId"] = _hash({"version": VERSION, "target": family, "head": head, "cohort": cohort,
                                 "candidateVersion": str(selected), "targetVersion": TARGET_VERSION})
        # Publication time is deliberately excluded from identity so repeated
        # identical publications keep their first original publication time.
        identity_row = {k: v for k, v in row.items() if k != "recordedEpoch"}
        row["caseId"] = _hash(identity_row)
        yield row


def record_issued(conn, result, build=None, actual_issued_epoch=None, candidate_versions=None):
    """Archive published numeric outputs without choosing or altering a value.

    ``result`` is the native analysis payload. Call in the same transaction as
    the existing dashboard issue archive, after adding its inputProvenance.
    """
    if not isinstance(result, dict):
        raise TypeError("Issued result must be a dictionary")
    recorded = _epoch(actual_issued_epoch if actual_issued_epoch is not None else time.time())
    if recorded is None:
        # time.time() is intentionally rounded only for the ledger publication
        # clock; provided timestamps must already be integral seconds.
        if actual_issued_epoch is None:
            recorded = int(time.time())
        else:
            raise ValueError("Publication epoch must be integral")
    init_schema(conn)
    inserted = 0
    for row in _forecast_rows(result, recorded, build or result.get("dashboardBuild") or "unknown", candidate_versions or {}):
        cursor = conn.execute("INSERT OR IGNORE INTO prospective_validation_issued VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            row["caseId"], recorded, row["issueEpoch"] or recorded, row["targetFamily"], row["head"], row["cohort"],
            row["targetDate"], row["startEpoch"], row["endEpoch"], row["completeEpoch"], row["candidateVersion"],
            row["modelVersion"], row["inputManifest"]["inputHash"], row["cohortId"], encode_payload(_json(row))))
        inserted += cursor.rowcount
    return inserted


def _active_recipe_filter(candidate_versions, alias=""):
    if not candidate_versions:
        return "", []
    prefix = alias + "." if alias else ""
    clauses, parameters = [], []
    for family, recipe in candidate_versions.items():
        clauses.append("(" + prefix + "target_family=? AND " + prefix + "candidate_version=?)")
        parameters.extend((family, recipe))
    return " WHERE (" + " OR ".join(clauses) + ")", parameters


def validation_status(conn, as_of_epoch=None, candidate_versions=None):
    """Cheap readiness for serving; never accesses readings or sealed labels."""
    as_of = int(as_of_epoch if as_of_epoch is not None else time.time())
    result = {"version": VERSION, "state": "collecting_evidence", "validated": False, "calibrated": False,
              "confirmationCasesEvaluated": 0, "completedEligibleTargetDates": 0,
              "reason": "No supported completed prospective confirmation for the active model recipe.",
              "performanceUse": "informational_only_does_not_select_numeric_output",
              "pairedIntervalConfidence": default_policy()["pairedIntervalConfidence"],
              "sealedOutcomeProtectionEpoch": SEALED_EPOCH, "targets": {}}
    for family, recipe in (candidate_versions or {}).items():
        result["targets"][family] = {"candidateVersion": recipe, "issuedSnapshots": 0, "maturedByClock": 0,
                                      "validated": False, "state": "collecting_evidence"}
    if not _has_table(conn, "prospective_validation_issued"):
        return result
    issue_filter, issue_parameters = _active_recipe_filter(candidate_versions)
    counts = conn.execute("""SELECT target_family,COUNT(*),SUM(CASE WHEN complete_epoch<=? THEN 1 ELSE 0 END)
        FROM prospective_validation_issued""" + issue_filter + " GROUP BY target_family", [as_of] + issue_parameters).fetchall()
    for family, total, mature in counts:
        result["targets"].setdefault(family, {"validated": False, "state": "collecting_evidence"}).update(
            issuedSnapshots=total, maturedByClock=mature or 0)
    if _has_table(conn, "prospective_validation_scores"):
        score_filter, score_parameters = _active_recipe_filter(candidate_versions, "i")
        counts = conn.execute("""SELECT i.target_family,COUNT(*),COUNT(DISTINCT i.target_date)
            FROM prospective_validation_scores s JOIN prospective_validation_issued i ON i.case_id=s.case_id
            """ + score_filter + " GROUP BY i.target_family", score_parameters).fetchall()
        for family, scored, dates in counts:
            result["targets"].setdefault(family, {"validated": False, "state": "collecting_evidence"}).update(
                scoredCases=scored, evaluatedTargetDates=dates)
        result["completedEligibleTargetDates"] = conn.execute("""SELECT COUNT(DISTINCT i.target_date)
            FROM prospective_validation_scores s JOIN prospective_validation_issued i ON i.case_id=s.case_id""" +
            score_filter, score_parameters).fetchone()[0]
    if _has_table(conn, "prospective_validation_reports"):
        rows = conn.execute("""SELECT confirmation,payload FROM prospective_validation_reports
            WHERE confirmation=1 AND policy_hash=? ORDER BY as_of_epoch DESC LIMIT 16""", (_hash(default_policy()),)).fetchall()
        confirmation_case_ids = set()
        for confirmation, payload in rows:
            report = loads(payload)
            if confirmation and report.get("version") == VERSION and report.get("policyHash") == _hash(default_policy()):
                for target, details in report.get("groups", {}).items():
                    if candidate_versions and candidate_versions.get(target) != report.get("candidateVersion"):
                        continue
                    confirmation_case_ids.update(details.get("caseIds", []))
                    if details.get("passesConfirmation"):
                        diagnostic = report.get("evidenceRole") == "candidate_diagnostic"
                        result["targets"][target] = {"state": "candidate_confirmation_passed" if diagnostic else "confirmation_passed", "validated": not diagnostic,
                            "candidateSupportEstablished": diagnostic, "evidenceRole": report.get("evidenceRole"),
                            "candidateVersion": report.get("candidateVersion"), "cohort": report.get("cohort"),
                            "reportId": report.get("reportId"), "pairedIntervalConfidence": report["pairedIntervalConfidence"],
                            "confirmationCases": details.get("cases", 0), "targetDates": details.get("distinctTargetDates", 0)}
        result["confirmationCasesEvaluated"] = len(confirmation_case_ids)
    # Global validated remains false: one target/clock/version cannot validate
    # all targets or calibrate unrelated probability/interval outputs.
    return result


def register_confirmation_candidate(conn, target, candidate_version, look_number, *, cohort,
                                    registered_epoch, selection_evidence_hash, selection_gate_passed,
                                    raw_model_version, raw_numerical_policy, policy=None):
    """Freeze one candidate before the first sealed label exists; no labels read."""
    policy = policy or default_policy()
    _validate_policy(policy)
    if target not in FAMILIES or cohort not in ("scheduled_near", "canonical", "moving"):
        raise ValueError("Unknown target or clock cohort")
    if not isinstance(look_number, int) or not 1 <= look_number <= policy["maximumLooks"]:
        raise ValueError("Confirmation look budget exhausted")
    if not candidate_version or not selection_gate_passed or not isinstance(selection_evidence_hash, str) or len(selection_evidence_hash) != 64:
        raise ValueError("A frozen candidate and passing selection evidence hash are required")
    if not raw_model_version or not raw_numerical_policy:
        raise ValueError("The raw originally issued candidate model and numerical policy must be frozen")
    if registered_epoch >= policy["sealedFromEpoch"]:
        raise ValueError("Cannot freeze a candidate after protected outcomes start")
    init_schema(conn)
    conn.execute("""INSERT INTO prospective_validation_looks(target_family,look_number,candidate_version,cohort,
        registered_epoch,issue_start_epoch,issue_end_epoch,policy_hash,selection_evidence_hash,status,
        report_id,evidence_role,raw_model_version,raw_numerical_policy) VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?,?)""", (
        target, look_number, candidate_version, cohort, int(registered_epoch),
        policy["confirmationIssueStartEpoch"], policy["confirmationIssueEndEpochExclusive"],
        _hash(policy), selection_evidence_hash, "registered", "candidate_diagnostic", raw_model_version, raw_numerical_policy))


def claim_confirmation_look(conn, target, candidate_version, look_number, *, cohort,
                            as_of_epoch, raw_model_version, raw_numerical_policy, policy=None):
    """Durably count an authorized look BEFORE labels are opened.

    Caller MUST commit this claim separately, then call evaluate_issued. A
    crash or rollback of later score/report work cannot unspend the look.
    This function never reads labels and never commits the caller's connection.
    """
    policy = policy or default_policy()
    _validate_policy(policy)
    if as_of_epoch < policy["confirmationIssueEndEpochExclusive"] + 86400:
        raise ValueError("Full frozen confirmation period has not completed; no labels opened")
    init_schema(conn)
    cursor = conn.execute("""UPDATE prospective_validation_looks SET status='claimed_counted'
        WHERE target_family=? AND look_number=? AND candidate_version=? AND cohort=? AND policy_hash=?
        AND raw_model_version=? AND raw_numerical_policy=? AND evidence_role='candidate_diagnostic'
        AND status='registered'""", (target, look_number, candidate_version, cohort, _hash(policy),
                                    raw_model_version, raw_numerical_policy))
    if cursor.rowcount != 1:
        raise ValueError("No unused matching preregistered confirmation look")
    return {"targetFamily": target, "lookNumber": look_number, "state": "claimed_counted",
            "commitRequiredBeforeScoring": True, "labelsOpened": False}


def _selected_rows(rows):
    selected = {}
    for row in sorted(rows, key=lambda r: (r["recordedEpoch"], r["caseId"])):
        # First publication before eligibility filtering; later available
        # replacements cannot rescue an unavailable original selected case.
        issue = row["issueEpoch"] or row["recordedEpoch"]
        key = (row["targetFamily"], row["head"], row["cohort"], row["candidateVersion"],
               row["issueDate"], None if row["cohort"] == "canonical" else
               datetime.fromtimestamp(issue, LOCAL).hour // 4)
        selected.setdefault(key, row)
    return list(selected.values())


def _metrics(rows, policy, target):
    actual = np.array([r["actual"] for r in rows], float)
    pred = np.array([r["point"] for r in rows], float)
    baseline = np.array([r["baselinePoint"] for r in rows], float)
    error, baseline_error = pred - actual, baseline - actual
    truth = np.where(actual - baseline >= 20, 1, np.where(actual - baseline <= -20, -1, 0))
    calls = np.where(pred - baseline >= 20, 1, np.where(pred - baseline <= -20, -1, 0))
    changed = truth != 0
    peak = actual >= policy["peakThresholdUgM3"][target]
    event_dates = {r["targetDate"] for r, is_event in zip(rows, changed) if is_event}
    peak_dates = {r["targetDate"] for r, is_peak in zip(rows, peak) if is_peak}
    recall = float(np.mean(calls[changed] == truth[changed])) if changed.any() else None
    interval_rows = [r for r in rows if r["rangeLow"] is not None and r["rangeHigh"] is not None and r["rangeLow"] <= r["rangeHigh"]]
    probability_rows = [r for r in rows if r.get("probabilityContract") == "exact_window_mean_le70" and r["probability"] is not None]
    calibration = {"state": "uncalibrated", "validated": False, "intervalCases": len(interval_rows),
        "intervalTargetDates": len({r["targetDate"] for r in interval_rows}),
        "intervalCoverageDescriptive": float(np.mean([r["rangeLow"] <= r["actual"] <= r["rangeHigh"] for r in interval_rows])) if interval_rows else None,
        "probabilityContract": "exact_window_mean_le70" if probability_rows else None,
        "probabilityCases": len(probability_rows), "probabilityTargetDates": len({r["targetDate"] for r in probability_rows}),
        "probabilityBrierDescriptive": float(np.mean([(r["probability"] - (r["actual"] <= 70)) ** 2 for r in probability_rows])) if probability_rows else None,
        "reason": "Coverage/Brier on issued cases are descriptive; no separate frozen calibration confirmation has passed."}
    return {"cases": len(rows), "distinctTargetDates": len({r["targetDate"] for r in rows}),
        "distinctTargetWindows": len({(r["startEpoch"], r["endEpoch"]) for r in rows}),
        "mae": float(np.mean(abs(error))), "persistenceMae": float(np.mean(abs(baseline_error))),
        "bias": float(np.mean(error)), "persistenceBias": float(np.mean(baseline_error)),
        "p90AbsoluteError": float(np.quantile(abs(error), .9)),
        "largeChange20": {"events": int(changed.sum()), "distinctEventDates": len(event_dates),
            "hits": int((changed & (calls == truth)).sum()), "noCallMisses": int((changed & (calls == 0)).sum()),
            "wrongWay": int((changed & (calls != 0) & (calls != truth)).sum()),
            "quietFalseCalls": int((~changed & (calls != 0)).sum()), "recall": recall,
            "supported": len(event_dates) >= policy["minimumLargeChangeDatesPerHead"]},
        "peak": {"thresholdUgM3": policy["peakThresholdUgM3"][target], "cases": int(peak.sum()),
            "distinctDates": len(peak_dates), "supported": len(peak_dates) >= policy["minimumPeakDatesPerHead"],
            "mae": float(np.mean(abs(error[peak]))) if peak.any() else None,
            "persistenceMae": float(np.mean(abs(baseline_error[peak]))) if peak.any() else None},
        "calibration": calibration,
        "actualModelCounts": dict(Counter(r["modelVersion"] for r in rows)),
        "fallbackCases": sum(bool(r["fallback"]) for r in rows),
        "issueOffsetsSeconds": sorted({r["issueOffsetSeconds"] for r in rows}),
        "originalReceiptKnownCases": sum(r["inputManifest"]["originalReceiptKnown"] for r in rows)}


def _paired_interval(rows, target, policy):
    date_values = sorted({r["targetDate"] for r in rows})
    first = datetime.fromisoformat(date_values[0]).date()
    last = datetime.fromisoformat(date_values[-1]).date()
    dates = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
    heads = sorted({r["head"] for r in rows})
    positions = {d: i for i, d in enumerate(dates)}
    columns = {h: i for i, h in enumerate(heads)}
    sums = np.zeros((len(dates), len(heads)))
    counts = np.zeros_like(sums)
    for row in rows:
        d, h = positions[row["targetDate"]], columns[row["head"]]
        sums[d, h] += abs(row["baselinePoint"] - row["actual"]) - abs(row["point"] - row["actual"])
        counts[d, h] += 1
    length = policy["blockCalendarDays"][target]
    rng = np.random.default_rng(policy["bootstrapSeed"])
    starts = rng.integers(0, len(dates), (policy["bootstrapDraws"], math.ceil(len(dates) / length)))
    draws = ((starts[:, :, None] + np.arange(length)) % len(dates)).reshape(policy["bootstrapDraws"], -1)[:, :len(dates)]
    with np.errstate(divide="ignore", invalid="ignore"):
        sampled = (sums[draws].sum(axis=1) / counts[draws].sum(axis=1)).mean(axis=1)
        point = float((sums.sum(axis=0) / counts.sum(axis=0)).mean())
    good = sampled[np.isfinite(sampled)]
    alpha = policy["perTargetLookAlpha"]
    low, high = np.quantile(good, [alpha / 2, 1 - alpha / 2]) if len(good) else (None, None)
    return {"value": point, "confidence": policy["pairedIntervalConfidence"],
            "interval": [float(low), float(high)] if low is not None else [None, None],
            "fullWidth": float(high - low) if low is not None else None,
            "bootstrapDraws": policy["bootstrapDraws"], "validBootstrapDraws": len(good),
            "dateTimeline": len(dates), "distinctTargetDates": len(date_values),
            "blockCalendarDays": length, "effectiveDateBlocks": len(date_values) / length,
            "conditionalInterval": True, "meaning": "equal-head paired MAE gain over exact originally archived persistence"}


def _score_groups(rows, policy, confirmation, unavailable):
    # Preserved older-policy observations are outside this experiment, not
    # unavailable inputs within it. Keep their audit count without vetoing a
    # supported current-policy cohort. All actual availability failures block.
    blocking_exclusions = {reason: count for reason, count in unavailable.items()
                           if count and reason != "preserved_previous_measurement_policy"}
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["targetFamily"]].append(row)
    groups = {}
    for target, part in by_family.items():
        per_head = {h: _metrics([r for r in part if r["head"] == h], policy, target) for h in sorted({r["head"] for r in part})}
        interval = _paired_interval(part, target, policy)
        all_heads = set(per_head) == set(HEADS[target])
        minimum_gain = policy["minimumMaeGainUgM3"][target]
        support = all_heads and all(m["distinctTargetDates"] >= policy["minimumDistinctTargetDates"] and
            m["largeChange20"]["supported"] and m["peak"]["supported"] for m in per_head.values()) and (
            interval["effectiveDateBlocks"] >= policy["minimumIndependentDateBlocks"])
        operational = all(r["inputManifest"].get("inputHash") and r["inputManifest"].get("originalReceiptKnown") is True and (
            r["inputManifest"].get("completeAllModelInputReplay") is True or r.get("evidenceRole") == "candidate_diagnostic" and
            r["inputManifest"].get("candidateInputReplayComplete") is True) for r in part) and not blocking_exclusions
        gain = interval["value"] >= minimum_gain and interval["interval"][0] is not None and interval["interval"][0] > 0 and interval["fullWidth"] < minimum_gain
        head_checks = {h: {"maeNotWorseThanPersistence": m["mae"] <= m["persistenceMae"],
            "biasSupportedAndWithinLimit": m["distinctTargetDates"] >= policy["minimumDistinctTargetDates"] and abs(m["bias"]) <= policy["maximumAbsoluteBiasUgM3"],
            "peakSupportedAndNotWorse": m["peak"]["supported"] and m["peak"]["mae"] <= m["peak"]["persistenceMae"],
            "recallSupportedAndAtLeastMinimum": m["largeChange20"]["supported"] and m["largeChange20"]["recall"] >= policy["minimumRecall20"]} for h, m in per_head.items()}
        passed = bool(confirmation and support and operational and gain and all(all(check.values()) for check in head_checks.values()))
        groups[target] = {"cases": len(part), "distinctTargetDates": len({r["targetDate"] for r in part}),
            "distinctTargetWindows": len({(r["startEpoch"], r["endEpoch"]) for r in part}),
            "perHead": per_head, "pairedMaeGain": interval, "minimumMaeGainUgM3": minimum_gain,
            "supportEstablished": bool(support), "operationalInputsEstablished": bool(operational),
            "headChecks": head_checks, "passesConfirmation": passed,
            "state": "confirmation_passed" if passed else "insufficient_evidence" if not support else "confirmation_failed" if confirmation else "development_only",
            "calibrated": False, "caseIds": [r["caseId"] for r in part]}
    return groups


def evaluate_issued(conn, as_of_epoch, policy=None, *, confirmation=False, target=None,
                    candidate_version=None, look_number=None, cohort=None,
                    raw_model_version=None, raw_numerical_policy=None):
    """Explicit bounded scoring. Development can never read Oct20+ buckets.

    Confirmation additionally requires a pre-October20 registered target/look,
    full planned issue period completed, and the caller's explicit flag. A
    begun failed/insufficient/error look is consumed, never silently retried.
    """
    policy = policy or default_policy()
    _validate_policy(policy)
    as_of = int(as_of_epoch)
    init_schema(conn)
    seal = policy["sealedFromEpoch"]
    look = None
    if confirmation:
        if target not in FAMILIES or not candidate_version or look_number is None or cohort is None:
            raise ValueError("Explicit target, frozen candidate, cohort and look number required")
        look = conn.execute("SELECT * FROM prospective_validation_looks WHERE target_family=? AND look_number=?", (target, look_number)).fetchone()
        if not look or look[2] != candidate_version or look[3] != cohort or look[7] != _hash(policy) or look[9] not in ("registered", "claimed_counted") or look[11] != "candidate_diagnostic":
            raise ValueError("No unused matching preregistered confirmation look")
        if raw_model_version != look[12] or raw_numerical_policy != look[13]:
            raise ValueError("Confirmation raw model/numerical policy differs from the frozen candidate")
        # Last next-day window may finish on the following date.
        if as_of < policy["confirmationIssueEndEpochExclusive"] + 86400:
            raise ValueError("Full frozen confirmation period has not completed; no labels opened")
        if look[9] != "claimed_counted" or conn.in_transaction:
            raise ValueError("Commit a separate counted confirmation claim before opening labels")
        consumed = conn.execute("""UPDATE prospective_validation_looks SET status='begun'
            WHERE target_family=? AND look_number=? AND status='claimed_counted'""", (target, look_number))
        if consumed.rowcount != 1:
            raise ValueError("Confirmation look was already consumed by another evaluator")
        # This explicit confirmation action owns only its authorization marker.
        # There are no pending caller writes (checked above). Commit BEFORE
        # reading any labels: a later error, rollback, or process restart cannot
        # restore the right to open this protected look a second time.
        conn.commit()
    lower = policy["confirmationIssueStartEpoch"] if confirmation else 0
    upper = policy["confirmationIssueEndEpochExclusive"] if confirmation else min(as_of + 1, seal)
    try:
        clauses, args = ["issue_epoch>=?", "issue_epoch<?"], [lower, upper]
        for field, value in (("target_family", target), ("candidate_version", candidate_version), ("cohort", cohort)):
            if value is not None:
                clauses.append(field + "=?")
                args.append(value)
        raw_rows = [loads(r[0]) for r in conn.execute("SELECT payload FROM prospective_validation_issued WHERE " + " AND ".join(clauses), args)]
        # Policy v2 changes the canonical Morning issue clock. Preserve v1
        # evidence but do not pool its 07:30 question into the 07:15 study.
        previous_policy_rows = [r for r in raw_rows if r.get("version") != VERSION]
        selected = _selected_rows([r for r in raw_rows if r.get("version") == VERSION])
        eligible, excluded = [], Counter()
        if previous_policy_rows:
            excluded["preserved_previous_measurement_policy"] = len(previous_policy_rows)
        for row in selected:
            if not row["available"] and not row.get("candidateDiagnostic", {}).get("available"):
                excluded["unavailable_first_selection"] += 1
            elif row["completeEpoch"] > as_of:
                excluded["target_not_complete"] += 1
            elif not confirmation and row["completeEpoch"] >= seal:
                excluded["sealed_target_support"] += 1
            else:
                eligible.append(row)
        # Scores are stored once with their first immutable actual/label hash;
        # sensor revisions never silently replace previously scored labels.
        needed = []
        for row in eligible:
            existing = conn.execute("SELECT actual,label_snapshot_hash FROM prospective_validation_scores WHERE case_id=?", (row["caseId"],)).fetchone()
            if existing:
                row.update(actual=existing[0], labelSnapshotHash=existing[1])
            else:
                needed.append(row)
        if needed:
            lo = min(math.floor(r["startEpoch"] / 900) * 900 - 900 for r in needed)
            hi = max(r["completeEpoch"] for r in needed)
            if not confirmation and hi >= seal:
                raise ValueError("Protected label boundary reached")
            readings = [(int(e), _finite(v)) for e, v in conn.execute(
                "SELECT epoch,pm02 FROM readings WHERE epoch>? AND epoch<=? ORDER BY epoch", (lo, hi))]
            readings = [(e, v) for e, v in readings if v is not None and v >= 0]
            buckets = covered_bucket_medians(readings)
            for row in needed:
                weights = point_weights(row["startEpoch"]) if row["targetFamily"] == "point90" else window_weights(row["startEpoch"], row["endEpoch"])
                actual = weighted_target(buckets, weights, as_of)
                if actual is None:
                    excluded["incomplete_target_coverage"] += 1
                    continue
                label_hash = _hash({"targetVersion": TARGET_VERSION, "weights": weights,
                                    "values": {k: buckets[k] for k in weights}})
                conn.execute("INSERT INTO prospective_validation_scores VALUES(?,?,?,?,?)", (row["caseId"], as_of, actual, label_hash, TARGET_VERSION))
                row.update(actual=actual, labelSnapshotHash=label_hash)
        scored = [r for r in eligible if "actual" in r]
        operational = [r for r in scored if r["available"]]
        group_key = lambda r: (r["candidateVersion"], r["cohort"], r["targetFamily"])
        split = defaultdict(list)
        for row in operational:
            split[group_key(row)].append(row)
        reports = {}
        for (candidate, clock_cohort, family), rows in split.items():
            # Different candidate versions and moving/canonical clocks are
            # distinct questions; they cannot be pooled to rescue a failed gate.
            reports["|".join((candidate, clock_cohort, family))] = _score_groups(rows, policy, confirmation, excluded)
        diagnostic_rows, diagnostic_excluded = [], Counter()
        for row in selected:
            diagnostic = row.get("candidateDiagnostic") or {}
            matched = (not confirmation or (diagnostic.get("modelVersion") == raw_model_version and
                                            diagnostic.get("numericalPolicyIdentifier") == raw_numerical_policy))
            if not diagnostic.get("available") or not matched:
                diagnostic_excluded["raw_candidate_unavailable_or_frozen_identity_mismatch"] += 1
        for row in scored:
            diagnostic = row.get("candidateDiagnostic") or {}
            if diagnostic.get("available") and (not confirmation or (
                diagnostic.get("modelVersion") == raw_model_version and diagnostic.get("numericalPolicyIdentifier") == raw_numerical_policy)):
                adapted = dict(row, point=diagnostic["mean"], modelVersion=diagnostic["modelVersion"],
                    rangeLow=diagnostic["rangeLow"], rangeHigh=diagnostic["rangeHigh"],
                    probability=None, probabilityContract=None, evidenceRole="candidate_diagnostic")
                adapted["rawNumericalPolicy"] = diagnostic["numericalPolicyIdentifier"]
                diagnostic_rows.append(adapted)
        diagnostic_split = defaultdict(list)
        for row in diagnostic_rows:
            diagnostic_split[group_key(row) + (row["modelVersion"], row["rawNumericalPolicy"])].append(row)
        diagnostic_reports = {"|".join(key): _score_groups(rows, policy, confirmation, excluded + diagnostic_excluded)
                              for key, rows in diagnostic_split.items()}
        groups = next(iter(diagnostic_reports.values())) if confirmation and len(diagnostic_reports) == 1 else {}
        report = {"version": VERSION, "asOfEpoch": as_of, "confirmation": bool(confirmation),
            "candidateVersion": candidate_version, "cohort": cohort, "lookNumber": look_number,
            "policyHash": _hash(policy), "pairedIntervalConfidence": policy["pairedIntervalConfidence"],
            "perTargetLookAlpha": policy["perTargetLookAlpha"], "selectedCases": len(selected),
            "scoredCases": len(scored), "exclusions": dict(excluded), "groups": groups,
            "evidenceRole": "candidate_diagnostic" if confirmation else "separate_operational_and_candidate_diagnostic",
            "rawModelVersion": raw_model_version, "rawNumericalPolicy": raw_numerical_policy,
            "developmentGroups": reports if not confirmation else {}, "operationalGroups": reports,
            "candidateDiagnosticGroups": diagnostic_reports, "candidateDiagnosticScoredCases": len(diagnostic_rows),
            "candidateDiagnosticExclusions": dict(diagnostic_excluded),
            "confirmationCasesRead": len(scored) if confirmation else 0,
            "protectedLabelsOpened": bool(confirmation and needed), "validated": False, "calibrated": False,
            "labelSnapshotHashes": {r["caseId"]: r["labelSnapshotHash"] for r in scored},
            "previouslyExposedDevelopment": not confirmation,
            "limitations": ["Date blocks are statistical clusters, not independent meteorological storms.",
                "Conditional bootstrap precision and actual peak/event support must pass; counts alone are not skill.",
                "Numerical endpoint/mean labels never score within-window first-crossing probabilities.",
                "Interval/probability calibration has its own contract and remains unvalidated."]}
        report["reportId"] = _hash(report)
        conn.execute("INSERT OR IGNORE INTO prospective_validation_reports VALUES(?,?,?,?,?)", (
            report["reportId"], as_of, int(confirmation), _hash(policy), encode_payload(_json(report))))
        if confirmation:
            status = "passed" if groups and all(g["passesConfirmation"] for g in groups.values()) else "failed_or_insufficient"
            conn.execute("UPDATE prospective_validation_looks SET status=?,report_id=? WHERE target_family=? AND look_number=?", (status, report["reportId"], target, look_number))
        return report
    except Exception:
        if confirmation:
            conn.execute("UPDATE prospective_validation_looks SET status='errored_counted' WHERE target_family=? AND look_number=?", (target, look_number))
        raise
