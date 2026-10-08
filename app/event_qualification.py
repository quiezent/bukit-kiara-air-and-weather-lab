"""Truthful operational qualification for experimental event diagnostics.

This module does not fit, calibrate, tune a decision threshold, or infer skill
from development results. The raw numerical inference remains available for
prospective evaluation; an unqualified classifier makes no operational call.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json

MYT = timezone(timedelta(hours=8))
QUALIFICATION_VERSION = "actual_cadence_event_qualification_v1_20261008"
FIXED_EVENT_CADENCE = "fixed_four_hour_origins_plus300_seconds"


def qualification_metadata(issue_epoch, *, training_cadence=FIXED_EVENT_CADENCE):
    """Grid membership is a necessary condition, never a validation claim."""
    issue = int(issue_epoch)
    local = datetime.fromtimestamp(issue, MYT)
    offset = issue % 900
    if training_cadence == FIXED_EVENT_CADENCE:
        matched = local.hour % 4 == 0 and local.minute == 5 and local.second == 0
        cadence_reason = "live_issue_outside_fixed_four_hour_plus300_training_grid"
    elif training_cadence == "archived_halfhour_publications_with_120second_delay_guard":
        # The training selector uses the archive publication clock rather than
        # the forecast clock. This forecast-clock range is only a possible
        # member; it cannot establish receipt/selection equivalence.
        matched = 1680 <= issue % 3600 <= 1860
        cadence_reason = "live_issue_outside_archived_halfhour_training_clock_range"
    elif training_cadence == "archived_final120seconds_quarterhour_publications":
        matched = 660 <= offset < 900
        cadence_reason = "live_issue_outside_archived_quarterhour_training_clock_range"
    else:
        matched = False
        cadence_reason = "training_cadence_not_declared"
    reasons = ["no_actual_cadence_prospective_qualification"]
    reasons.append("current_input_snapshot_policy_not_original_publication_watermark"
                   if training_cadence == FIXED_EVENT_CADENCE
                   else "historical_input_receipt_and_revision_history_incomplete")
    if not matched:
        reasons.append(cadence_reason)
    return {
        "qualificationVersion": QUALIFICATION_VERSION,
        "prospectivelyValidated": False,
        "trainingFitCadence": training_cadence,
        "trainingCadenceMatched": bool(matched),
        "liveIssueOffsetSeconds": offset,
        "unvalidatedLiveCadence": True,
        "probabilityRole": "uncalibrated_experimental_diagnostic",
        "operationalDecisionRule": "no_direction_until_actual_cadence_prospective_qualification",
        "directionStatus": "unqualified_experimental",
        "qualification": {
            "state": "unqualified_experimental", "eligible": False,
            "operationalUseEligible": False, "actualCadenceValidated": False,
            "calibrationValidated": False, "trainingCadenceMatched": bool(matched),
            "reasons": reasons,
            "evidenceRole": "development_and_originally_issued_diagnostics",
        },
    }


def apply_event_qualification(result, *, training_cadence=FIXED_EVENT_CADENCE):
    """Preserve raw argmax separately and publish a stable no-direction state."""
    result.update(qualification_metadata(result["forecastIssuedEpoch"],
                                        training_cadence=training_cadence))
    result["diagnosticDirection"] = result.get("direction", "unresolved")
    result["diagnosticDirectionAvailable"] = bool(result.get("available"))
    result["direction"] = "unresolved"
    return result


def publication_metadata(result, input_lineage=None):
    """Archive cohort identity and actual lineage without inventing old receipt."""
    issue = int(result["forecastIssuedEpoch"])
    local = datetime.fromtimestamp(issue, MYT)
    model_identity = result.get("modelIdentity")
    identity_hash = (hashlib.sha256(json.dumps(model_identity, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                     if model_identity is not None else None)
    cadence = result.get("trainingFitCadence", FIXED_EVENT_CADENCE)
    metadata = qualification_metadata(issue, training_cadence=cadence)
    regime = ("fixed_training_grid" if metadata["trainingCadenceMatched"]
              else "outside_fixed_training_grid")
    values = {
        "issuedCohort": {
            "modelVersion": result.get("modelVersion"),
            "adapterVersion": result.get("adapterVersion"),
            "modelIdentitySha256": identity_hash,
            "target": result.get("target"),
            "changeThresholdUgM3": result.get("changeThresholdUgM3"),
            "qualificationVersion": result.get("qualificationVersion"),
            "trainingCadence": cadence, "issueCadenceRegime": regime,
            "issueHourMYT": local.hour, "issueOffsetSeconds": issue % 900,
        },
    }
    if input_lineage is None:
        values["inputLineage"] = {"status": "not_archived",
            "historicalReceiptTimesUnknown": True}
    else:
        # Only the publication owner can supply the immutable snapshot that
        # really produced these features. Feature hashes alone are not receipts.
        values["inputLineage"] = dict(input_lineage)
        snapshot_id = input_lineage.get("snapshotId", input_lineage.get("inputSnapshotId"))
        if snapshot_id is not None:
            values["inputSnapshotId"] = snapshot_id
    return values
