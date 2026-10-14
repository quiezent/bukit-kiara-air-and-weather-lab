"""Behavioral validation tests using only synthetic in-memory SQLite data."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import sqlite3
import unittest
from unittest.mock import patch

import prospective_validation as validation
from forecast_payload import loads

LOCAL = timezone(timedelta(hours=8))


def epoch(day=8, hour=0, minute=0, second=0, month=10):
    return int(datetime(2026, month, day, hour, minute, second, tzinfo=LOCAL).timestamp())


def payload(issue, point=12, baseline=12, snapshot=True):
    start, end = issue + 5400, issue + 12600
    near = {"available": True, "point": point, "baselinePoint": baseline,
            "modelVersion": "near_candidate_v1", "pointRole": "model_point",
            "forecastIssuedEpoch": issue, "rangeLow": 5, "rangeHigh": 20}
    result = {"available": True, "forecastIssuedEpoch": issue,
              "forecastPolicy": {"windowForecastModels": {"morning": "morning_v1", "afternoon": "patch_v1"}},
              "airWindow": {"available": True, "forecastClock": {
                  "targetVersion": validation.TARGET_VERSION, "forecastIssuedEpoch": issue,
                  "arrivalTargetEpoch": start, "windowStartEpoch": start, "windowEndEpoch": end},
                  "arrival": deepcopy(near), "trail": deepcopy(near)}, "windows": {}}
    if snapshot:
        result["inputProvenance"] = {"snapshotId": "a" * 64, "snapshotSha256": "a" * 64,
            "forecastEpoch": issue, "allReceiptsKnown": True, "unknownReceiptInputs": 0,
            "reconstructionComplete": True, "completeAllModelInputReplay": True}
    return result


def session(result, head, start, model="patch_v1", point=12):
    result["windows"][head] = {"startEpoch": start, "endEpoch": start + 7200,
        "particleForecast": {"available": True, "point": point, "baselinePoint": 12,
                             "modelVersion": model, "forecastIssuedEpoch": result["forecastIssuedEpoch"]}}


def readings(conn, first_bucket, last_bucket, value=10):
    for label in range(first_bucket, last_bucket + 1, 900):
        for offset in (1, 300, 600, 890):
            conn.execute("INSERT OR REPLACE INTO readings VALUES(?,?)", (label - 900 + offset, value))


class ProspectiveValidationTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,pm02 REAL)")
        self.sql = []
        self.conn.set_trace_callback(self.sql.append)

    def tearDown(self):
        self.conn.close()

    def rows(self):
        return [loads(r[0]) for r in self.conn.execute("SELECT payload FROM prospective_validation_issued ORDER BY head")]

    def test_actual_output_identity_baseline_and_no_cross_family_candidate(self):
        issue = epoch()
        result = payload(issue, 11, 12)
        before = deepcopy(result)
        self.assertEqual(validation.record_issued(self.conn, result, "build_v1", issue + 2), 4)
        self.assertEqual(result, before)
        near = next(r for r in self.rows() if r["head"] == "point90")
        self.assertEqual(near["point"], 11)
        self.assertEqual(near["baselinePoint"], 12)
        self.assertEqual(near["candidateVersion"], "near_candidate_v1")
        self.assertEqual(near["startEpoch"], issue + 5400)
        self.assertEqual(near["inputManifest"]["inputHash"], "a" * 64)
        self.assertTrue(near["inputManifest"]["originalReceiptKnown"])
        self.assertTrue(self.conn.in_transaction)

    def test_duplicate_outputs_keep_first_publication_and_changed_output_is_retained(self):
        issue = epoch()
        result = payload(issue)
        validation.record_issued(self.conn, result, "v1", issue + 1)
        self.assertEqual(validation.record_issued(self.conn, result, "v1", issue + 2), 0)
        self.assertEqual({r["recordedEpoch"] for r in self.rows()}, {issue + 1})
        result["airWindow"]["arrival"]["point"] = 13
        self.assertEqual(validation.record_issued(self.conn, result, "v1", issue + 3), 1)

    def test_direct_model_values_and_recipes_remain_separate_from_gated_outputs(self):
        issue = epoch(hour=12)
        old_recipes = {family: family + "_gated_v20" for family in validation.FAMILIES}
        direct_recipes = {family: family + "_direct_v21" for family in validation.FAMILIES}
        validation.record_issued(self.conn, payload(issue, 12, 12), "v20", issue + 1,
            candidate_versions=old_recipes)
        result = payload(issue, 10.37, 12)
        near = result["airWindow"]["arrival"]
        near.update(modelVersion="local_bounded_analogue_raw_point90_v1",
                    pointRole="original_issued_raw_model_output",
                    modelSelection={"skillGatePassed": False, "prospectivelyValidated": False})
        near["candidateForecast"] = {"available": True, "mean": 10.37,
            "modelVersion": near["modelVersion"], "numericalPolicyIdentifier": "direct_near_raw_model_outputs_v1",
            "forecastedAtEpoch": issue, "startEpoch": issue + 5400, "endEpoch": issue + 5400,
            "pointRole": "original_issued_raw_model_output"}
        session(result, "morning", epoch(day=9, hour=9), model="morning_hgb_basic31_model_direct_v3", point=55.67)
        morning = result["windows"]["morning"]["particleForecast"]
        morning["mean"] = morning.pop("point")
        before = deepcopy(result)
        validation.record_issued(self.conn, result, "v21", issue + 2, candidate_versions=direct_recipes)
        self.assertEqual(result, before)
        rows = self.rows()
        old = next(r for r in rows if r["head"] == "point90" and r["candidateVersion"] == old_recipes["point90"])
        direct = next(r for r in rows if r["head"] == "point90" and r["candidateVersion"] == direct_recipes["point90"])
        self.assertNotEqual(old["cohortId"], direct["cohortId"])
        self.assertEqual(direct["point"], 10.37)
        self.assertEqual(direct["baselinePoint"], 12)
        self.assertEqual(direct["candidateDiagnostic"]["mean"], 10.37)
        self.assertTrue(direct["candidateDiagnostic"]["available"])
        self.assertEqual(next(r for r in rows if r["head"] == "nextMorning" and r["candidateVersion"] == direct_recipes["morningSession"])["point"], 55.67)
        self.sql.clear()
        status = validation.validation_status(self.conn, issue + 10, candidate_versions=direct_recipes)
        self.assertEqual(status["targets"]["point90"]["issuedSnapshots"], 1)
        self.assertEqual(status["performanceUse"], "informational_only_does_not_select_numeric_output")
        self.assertEqual(result, before)
        self.assertFalse(any("readings" in sql for sql in self.sql))
        readings(self.conn, issue + 4500, issue + 6300)
        report = validation.evaluate_issued(self.conn, issue + 7200, target="point90",
            candidate_version=direct_recipes["point90"])
        self.assertEqual(report["scoredCases"], 1)
        group = next(iter(report["developmentGroups"].values()))["point90"]
        self.assertEqual(group["caseIds"], [direct["caseId"]])

    def test_active_recipe_readiness_does_not_inherit_old_counts_or_confirmation(self):
        old_issue, new_issue = epoch(day=7), epoch(day=8)
        validation.record_issued(self.conn, payload(old_issue), "v20", old_issue + 1,
            candidate_versions={"point90": "old_gated_v20"})
        validation.record_issued(self.conn, payload(new_issue, 11, 12), "v21", new_issue + 1,
            candidate_versions={"point90": "new_direct_v21"})
        near = [r for r in self.rows() if r["head"] == "point90"]
        for row in near:
            self.conn.execute("INSERT INTO prospective_validation_scores VALUES(?,?,?,?,?)",
                (row["caseId"], new_issue + 7200, 10, "b" * 64, validation.TARGET_VERSION))
        policy_hash = validation._hash(validation.default_policy())
        for recipe, version, stored_hash in (("old_gated_v20", validation.VERSION, policy_hash),
                ("new_direct_v21", "actual_issued_validation_v1", policy_hash),
                ("new_direct_v21", validation.VERSION, "c" * 64)):
            report = {"version": version, "policyHash": stored_hash, "candidateVersion": recipe,
                "groups": {"point90": {"passesConfirmation": True, "caseIds": [recipe], "cases": 1}},
                "evidenceRole": "candidate_diagnostic", "pairedIntervalConfidence": .9958333333333333}
            self.conn.execute("INSERT INTO prospective_validation_reports VALUES(?,?,?,?,?)",
                (validation._hash(report), new_issue + 7200, 1, stored_hash,
                 validation.encode_payload(validation._json(report))))
        self.sql.clear()
        status = validation.validation_status(self.conn, new_issue + 7200,
            candidate_versions={"point90": "new_direct_v21"})
        self.assertEqual(set(status["targets"]), {"point90"})
        self.assertEqual(status["targets"]["point90"]["issuedSnapshots"], 1)
        self.assertEqual(status["targets"]["point90"]["maturedByClock"], 1)
        self.assertEqual(status["targets"]["point90"]["scoredCases"], 1)
        self.assertEqual(status["targets"]["point90"]["evaluatedTargetDates"], 1)
        self.assertEqual(status["completedEligibleTargetDates"], 1)
        self.assertEqual(status["confirmationCasesEvaluated"], 0)
        self.assertFalse(status["targets"]["point90"]["validated"])
        self.assertEqual(status["targets"]["point90"]["state"], "collecting_evidence")
        self.assertFalse(any("readings" in sql for sql in self.sql))

    def test_unavailable_first_is_not_replaced_with_later_available_case(self):
        issue = epoch()
        first = payload(issue)
        first["airWindow"]["arrival"]["available"] = False
        validation.record_issued(self.conn, first, "v1", issue + 1)
        later = payload(issue + 60)
        validation.record_issued(self.conn, later, "v1", issue + 61)
        readings(self.conn, issue + 4500, issue + 6300)
        report = validation.evaluate_issued(self.conn, issue + 7200, target="point90", candidate_version="near_candidate_v1")
        self.assertEqual(report["selectedCases"], 1)
        self.assertEqual(report["scoredCases"], 0)
        self.assertEqual(report["exclusions"]["unavailable_first_selection"], 1)

    def test_fallback_keeps_integrated_candidate_and_actual_model(self):
        issue = epoch(hour=12)
        result = payload(issue)
        session(result, "afternoon", epoch(hour=14), model="former_xgboost_v1")
        result["windows"]["afternoon"]["particleForecast"]["experimentalPatchtstFallback"] = {"reason": "missing_checkpoint", "applied": False}
        validation.record_issued(self.conn, result, "v1", issue + 1)
        row = next(r for r in self.rows() if r["head"] == "afternoon")
        self.assertEqual(row["candidateVersion"], "patch_v1")
        self.assertEqual(row["modelVersion"], "former_xgboost_v1")
        self.assertEqual(row["fallback"]["experimentalPatchtstFallback"]["reason"], "missing_checkpoint")

    def test_canonical_and_moving_windows_not_pooled(self):
        issue = epoch(hour=12)
        canonical = payload(issue)
        session(canonical, "afternoon", epoch(hour=14))
        validation.record_issued(self.conn, canonical, "v1", issue + 1)
        moving = payload(issue + 300)
        session(moving, "afternoon", epoch(hour=14, minute=15))
        validation.record_issued(self.conn, moving, "v1", issue + 301)
        readings(self.conn, epoch(hour=14), epoch(hour=16, minute=15))
        report = validation.evaluate_issued(self.conn, epoch(hour=17), target="afternoonSession", candidate_version="patch_v1")
        self.assertEqual(set(report["developmentGroups"]), {"patch_v1|canonical|afternoonSession", "patch_v1|moving|afternoonSession"})
        self.assertFalse(report["validated"])

    def test_same_and_next_date_share_target_cluster(self):
        previous = payload(epoch(day=7, hour=18, minute=5))
        session(previous, "afternoon", epoch(hour=14))
        current = payload(epoch(hour=12))
        session(current, "afternoon", epoch(hour=14))
        validation.record_issued(self.conn, previous, "v1", previous["forecastIssuedEpoch"] + 1)
        validation.record_issued(self.conn, current, "v1", current["forecastIssuedEpoch"] + 1)
        readings(self.conn, epoch(hour=14), epoch(hour=16))
        report = validation.evaluate_issued(self.conn, epoch(hour=17), target="afternoonSession", candidate_version="patch_v1", cohort="canonical")
        result = report["developmentGroups"]["patch_v1|canonical|afternoonSession"]["afternoonSession"]
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["distinctTargetDates"], 1)
        self.assertEqual(result["distinctTargetWindows"], 1)
        self.assertEqual(result["pairedMaeGain"]["dateTimeline"], 1)
        self.assertEqual(set(result["perHead"]), {"afternoon", "nextAfternoon"})

    def test_development_scores_correct_paired_same_target_and_freezes_actual(self):
        issue = epoch()
        validation.record_issued(self.conn, payload(issue, 11, 12), "v1", issue + 1)
        readings(self.conn, issue + 4500, issue + 6300, 10)
        report = validation.evaluate_issued(self.conn, issue + 7200, target="point90", candidate_version="near_candidate_v1")
        group = report["developmentGroups"]["near_candidate_v1|scheduled_near|point90"]["point90"]
        self.assertEqual(group["perHead"]["point90"]["mae"], 1)
        self.assertEqual(group["perHead"]["point90"]["persistenceMae"], 2)
        self.assertEqual(group["pairedMaeGain"]["value"], 1)
        self.conn.execute("UPDATE readings SET pm02=999")
        second = validation.evaluate_issued(self.conn, issue + 7300, target="point90", candidate_version="near_candidate_v1")
        self.assertEqual(report["labelSnapshotHashes"], second["labelSnapshotHashes"])
        group2 = second["developmentGroups"]["near_candidate_v1|scheduled_near|point90"]["point90"]
        self.assertEqual(group2["perHead"]["point90"]["mae"], 1)

    def test_no_unclosed_or_missing_coverage_label_is_scored(self):
        issue = epoch()
        validation.record_issued(self.conn, payload(issue), "v1", issue + 1)
        report = validation.evaluate_issued(self.conn, issue + 5300, target="point90")
        self.assertEqual(report["scoredCases"], 0)
        self.assertEqual(report["exclusions"], {"target_not_complete": 1})
        report = validation.evaluate_issued(self.conn, issue + 6000, target="point90")
        self.assertEqual(report["scoredCases"], 0)
        self.assertEqual(report["exclusions"], {"incomplete_target_coverage": 1})

    def test_development_does_not_query_sealed_outcomes_even_at_boundary(self):
        issue = validation.SEALED_EPOCH - 6000
        validation.record_issued(self.conn, payload(issue), "v1", issue + 1)
        report = validation.evaluate_issued(self.conn, validation.SEALED_EPOCH + 100000, target="mean90_210")
        self.assertEqual(report["exclusions"], {"sealed_target_support": 1})
        self.assertFalse(any("SELECT epoch,pm02 FROM readings" in sql for sql in self.sql))

    def test_readiness_never_opens_sensor_labels_or_claims_counts_are_skill(self):
        issue = epoch()
        validation.record_issued(self.conn, payload(issue), "v1", issue + 1)
        self.sql.clear()
        status = validation.validation_status(self.conn, validation.SEALED_EPOCH + 99999)
        self.assertFalse(status["validated"])
        self.assertFalse(status["calibrated"])
        self.assertEqual(status["confirmationCasesEvaluated"], 0)
        self.assertEqual(status["completedEligibleTargetDates"], 0)
        self.assertEqual(status["state"], "collecting_evidence")
        self.assertFalse(any("readings" in sql for sql in self.sql))

    def test_adjusted_confidence_and_no_policy_loosening(self):
        policy = validation.default_policy()
        self.assertAlmostEqual(policy["perTargetLookAlpha"], .05 / 12)
        self.assertAlmostEqual(policy["pairedIntervalConfidence"], .9958333333333333)
        changed = deepcopy(policy)
        changed["minimumDistinctTargetDates"] = 1
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            validation.evaluate_issued(self.conn, epoch(), policy=changed)

    def test_confirmation_requires_unused_matching_prefrozen_candidate(self):
        with self.assertRaisesRegex(ValueError, "preregistered"):
            validation.evaluate_issued(self.conn, epoch(month=12, day=31), confirmation=True,
                target="point90", candidate_version="x", look_number=1, cohort="scheduled_near", raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.assertFalse(any("SELECT epoch,pm02 FROM readings" in sql for sql in self.sql))
        validation.register_confirmation_candidate(self.conn, "point90", "x", 1, cohort="scheduled_near",
            registered_epoch=epoch(), selection_evidence_hash="b" * 64, selection_gate_passed=True, raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        with self.assertRaises(sqlite3.IntegrityError):
            validation.register_confirmation_candidate(self.conn, "point90", "changed", 1, cohort="scheduled_near",
                registered_epoch=epoch(), selection_evidence_hash="c" * 64, selection_gate_passed=True, raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        with self.assertRaisesRegex(ValueError, "budget"):
            validation.register_confirmation_candidate(self.conn, "point90", "x", 4, cohort="scheduled_near",
                registered_epoch=epoch(), selection_evidence_hash="b" * 64, selection_gate_passed=True, raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")

    def test_premature_confirmation_is_rejected_without_consuming_or_opening_look(self):
        validation.register_confirmation_candidate(self.conn, "point90", "x", 1, cohort="scheduled_near",
            registered_epoch=epoch(), selection_evidence_hash="b" * 64, selection_gate_passed=True, raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        with self.assertRaisesRegex(ValueError, "not completed"):
            validation.evaluate_issued(self.conn, epoch(), confirmation=True,
                target="point90", candidate_version="x", look_number=1, cohort="scheduled_near", raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.assertEqual(self.conn.execute("SELECT status FROM prospective_validation_looks").fetchone()[0], "registered")
        self.assertFalse(any("SELECT epoch,pm02 FROM readings" in sql for sql in self.sql))

    def test_insufficient_completed_confirmation_counts_look_and_cannot_retry(self):
        validation.register_confirmation_candidate(self.conn, "point90", "x", 1, cohort="scheduled_near",
            registered_epoch=epoch(), selection_evidence_hash="b" * 64, selection_gate_passed=True, raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        validation.claim_confirmation_look(self.conn, "point90", "x", 1, cohort="scheduled_near",
            as_of_epoch=epoch(month=12, day=31), raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.conn.commit()
        report = validation.evaluate_issued(self.conn, epoch(month=12, day=31), confirmation=True,
            target="point90", candidate_version="x", look_number=1, cohort="scheduled_near", raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.assertEqual(report["scoredCases"], 0)
        self.assertEqual(self.conn.execute("SELECT status FROM prospective_validation_looks").fetchone()[0], "failed_or_insufficient")
        with self.assertRaisesRegex(ValueError, "preregistered"):
            validation.evaluate_issued(self.conn, epoch(month=12, day=31), confirmation=True,
                target="point90", candidate_version="x", look_number=1, cohort="scheduled_near", raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")

    def test_uncommitted_confirmation_claim_opens_no_labels_and_rollback_keeps_durable_budget(self):
        validation.register_confirmation_candidate(self.conn, "point90", "x", 1, cohort="scheduled_near",
            registered_epoch=epoch(), selection_evidence_hash="b" * 64, selection_gate_passed=True,
            raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.conn.commit()
        validation.claim_confirmation_look(self.conn, "point90", "x", 1, cohort="scheduled_near",
            as_of_epoch=epoch(month=12, day=31), raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        with self.assertRaisesRegex(ValueError, "Commit a separate"):
            validation.evaluate_issued(self.conn, epoch(month=12, day=31), confirmation=True,
                target="point90", candidate_version="x", look_number=1, cohort="scheduled_near",
                raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.assertFalse(any("SELECT epoch,pm02 FROM readings" in sql for sql in self.sql))
        self.conn.commit()
        self.conn.execute("UPDATE prospective_validation_looks SET status='begun'")
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT status FROM prospective_validation_looks").fetchone()[0], "claimed_counted")
        with self.assertRaisesRegex(ValueError, "preregistered"):
            validation.claim_confirmation_look(self.conn, "point90", "x", 1, cohort="scheduled_near",
                as_of_epoch=epoch(month=12, day=31), raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")

    def test_missing_provenance_and_child_clock_not_certified(self):
        issue = epoch()
        result = payload(issue, snapshot=False)
        result["airWindow"]["arrival"]["forecastIssuedEpoch"] = issue + 1
        validation.record_issued(self.conn, result, "v1", issue + 2)
        row = next(r for r in self.rows() if r["head"] == "point90")
        self.assertFalse(row["available"])
        self.assertIn("selected_output_issue_clock_mismatch", row["unavailableReasons"])
        self.assertIsNone(row["inputManifest"]["inputHash"])
        self.assertFalse(row["inputManifest"]["originalReceiptKnown"])

    def test_first_crossing_probability_not_credited_to_numeric_target(self):
        issue = epoch()
        result = payload(issue)
        result["airWindow"]["firstCrossingEventForecast"] = {"available": True, "riseProbability": .99, "dropProbability": .01}
        result["airWindow"]["cyclingWindowForecast"] = {"available": True,
            "forecastIssuedEpoch": issue, "startEpoch": issue + 5400, "endEpoch": issue + 12600,
            "chanceMeanAtOrBelowCutoff": .75, "cutoffUgM3": 70, "modelVersion": "ride_probability_v1"}
        validation.record_issued(self.conn, result, "v1", issue + 1)
        point = next(r for r in self.rows() if r["head"] == "point90")
        ride = next(r for r in self.rows() if r["head"] == "mean90_210")
        self.assertIsNone(point["probability"])
        self.assertEqual(ride["probability"], .75)
        self.assertEqual(ride["probabilityContract"], "exact_window_mean_le70")

    def test_original_raw_candidate_scored_separately_from_operational_persistence(self):
        issue = epoch(hour=12)
        result = payload(issue)
        session(result, "afternoon", epoch(hour=14), point=12)
        candidate = {"modelVersion": "raw_patch_v1", "numericalPolicyIdentifier": "raw_patch_recipe_v1",
            "mean": 10, "forecastedAtEpoch": issue, "startEpoch": epoch(hour=14), "endEpoch": epoch(hour=16),
            "pointRole": "raw_mean_before_operational_gate"}
        result["windows"]["afternoon"]["particleForecast"]["candidateForecast"] = candidate
        validation.record_issued(self.conn, result, "v20", issue + 1,
            candidate_versions={"afternoonSession": "afternoon_patchtst_xgb_gated_v20"})
        readings(self.conn, epoch(hour=14), epoch(hour=16), 10)
        report = validation.evaluate_issued(self.conn, epoch(hour=17), target="afternoonSession")
        operational = next(iter(report["operationalGroups"].values()))["afternoonSession"]
        raw = next(iter(report["candidateDiagnosticGroups"].values()))["afternoonSession"]
        self.assertEqual(operational["perHead"]["afternoon"]["mae"], 2)
        self.assertEqual(operational["pairedMaeGain"]["value"], 0)
        self.assertEqual(raw["perHead"]["afternoon"]["mae"], 0)
        self.assertEqual(raw["pairedMaeGain"]["value"], 2)
        self.assertEqual(report["candidateDiagnosticScoredCases"], 1)
        self.assertFalse(raw["passesConfirmation"])
        self.assertFalse(report["validated"])
        issued = next(r for r in self.rows() if r["head"] == "afternoon")
        self.assertEqual(issued["point"], 12)
        self.assertEqual(issued["candidateDiagnostic"]["mean"], 10)
        self.assertEqual(issued["candidateDiagnostic"]["modelVersion"], "raw_patch_v1")

    def test_raw_clock_mismatch_and_first_raw_unavailable_not_rescued(self):
        issue = epoch(hour=12)
        first = payload(issue)
        session(first, "afternoon", epoch(hour=14))
        first["windows"]["afternoon"]["particleForecast"]["candidateForecast"] = {
            "mean": 10, "modelVersion": "raw_v1", "numericalPolicyIdentifier": "raw_policy_v1",
            "forecastedAtEpoch": issue, "startEpoch": epoch(hour=14) + 900, "endEpoch": epoch(hour=16) + 900}
        validation.record_issued(self.conn, first, "v1", issue + 1)
        later = deepcopy(first)
        later["forecastIssuedEpoch"] = issue + 60
        later["windows"]["afternoon"]["particleForecast"]["forecastIssuedEpoch"] = issue + 60
        later["windows"]["afternoon"]["particleForecast"]["candidateForecast"].update(
            forecastedAtEpoch=issue + 60, startEpoch=epoch(hour=14), endEpoch=epoch(hour=16))
        validation.record_issued(self.conn, later, "v1", issue + 61)
        readings(self.conn, epoch(hour=14), epoch(hour=16), 10)
        report = validation.evaluate_issued(self.conn, epoch(hour=17), target="afternoonSession")
        self.assertEqual(report["selectedCases"], 1)
        self.assertEqual(report["candidateDiagnosticScoredCases"], 0)
        self.assertEqual(report["candidateDiagnosticExclusions"], {"raw_candidate_unavailable_or_frozen_identity_mismatch": 1})

    def test_confirmation_gates_require_real_support_recall_peak_bias_and_gain(self):
        start = validation.SEALED_EPOCH
        sample = list(validation._forecast_rows(payload(start), start + 1, "v1", {}))[0]
        sample["inputManifest"]["originalReceiptKnown"] = True
        rows = []
        for i in range(70):
            row = deepcopy(sample)
            row.update(caseId=str(i), targetDate=validation._date(start + i * 86400),
                actual=120 if i < 10 else 20, startEpoch=start + i * 86400 + 5400,
                endEpoch=start + i * 86400 + 5400)
            row["point"] = row["actual"]
            row["baselinePoint"] = row["actual"] + 30
            rows.append(row)
        policy = validation.default_policy()
        supported = validation._score_groups(rows, policy, True, {})["point90"]
        self.assertTrue(supported["passesConfirmation"])
        self.assertEqual(supported["pairedMaeGain"]["confidence"], policy["pairedIntervalConfidence"])
        same = deepcopy(rows)
        for row in same:
            row["point"] = row["baselinePoint"]
        self.assertFalse(validation._score_groups(same, policy, True, {})["point90"]["passesConfirmation"])
        no_peaks = deepcopy(rows)
        for row in no_peaks:
            row.update(actual=20, point=20, baselinePoint=50)
        result = validation._score_groups(no_peaks, policy, True, {})["point90"]
        self.assertFalse(result["supportEstablished"])
        self.assertFalse(result["passesConfirmation"])
        biased = deepcopy(rows)
        for row in biased:
            row["point"] = row["actual"] + 6
        result = validation._score_groups(biased, policy, True, {})["point90"]
        self.assertFalse(result["headChecks"]["point90"]["biasSupportedAndWithinLimit"])
        self.assertFalse(result["passesConfirmation"])

    def test_partial_input_manifest_cannot_establish_confirmation_inputs(self):
        issue = epoch()
        result = payload(issue)
        result["inputProvenance"]["completeAllModelInputReplay"] = False
        sample = list(validation._forecast_rows(result, issue + 1, "v1", {}))[0]
        sample.update(actual=10, point=10, baselinePoint=12)
        group = validation._score_groups([sample], validation.default_policy(), True, {})["point90"]
        self.assertTrue(sample["inputManifest"]["originalReceiptKnown"])
        self.assertFalse(group["operationalInputsEstablished"])
        self.assertFalse(group["passesConfirmation"])
        del sample["inputManifest"]["completeAllModelInputReplay"]
        del sample["inputManifest"]["candidateInputReplayComplete"]
        group = validation._score_groups([sample], validation.default_policy(), True, {})["point90"]
        self.assertFalse(group["operationalInputsEstablished"])

    def test_preserved_prior_policy_counts_do_not_veto_current_support(self):
        start = validation.SEALED_EPOCH
        sample = list(validation._forecast_rows(payload(start), start + 1, "v1", {}))[0]
        sample["inputManifest"]["originalReceiptKnown"] = True
        rows = []
        for i in range(70):
            row = deepcopy(sample)
            row.update(caseId=str(i), targetDate=validation._date(start + i * 86400),
                actual=120 if i < 10 else 20, startEpoch=start + i * 86400 + 5400,
                endEpoch=start + i * 86400 + 5400)
            row["point"] = row["actual"]
            row["baselinePoint"] = row["actual"] + 30
            rows.append(row)
        exclusions = {"preserved_previous_measurement_policy": 1}
        result = validation._score_groups(rows, validation.default_policy(), True, exclusions)["point90"]
        self.assertTrue(result["operationalInputsEstablished"])
        self.assertTrue(result["passesConfirmation"])
        self.assertEqual(result["cases"], 70)
        # An informational count cannot hide a genuine missing original input
        # or raw output when both counts are present.
        for reason in ("unavailable_first_selection", "raw_candidate_unavailable_or_frozen_identity_mismatch"):
            blocked = validation._score_groups(rows, validation.default_policy(), True,
                dict(exclusions, **{reason: 1}))["point90"]
            self.assertFalse(blocked["operationalInputsEstablished"])
            self.assertFalse(blocked["passesConfirmation"])

    def test_more_than_next_day_session_is_not_canonical_or_eligible(self):
        issue = epoch(hour=18, minute=5)
        result = payload(issue)
        session(result, "morning", epoch(day=10, hour=9), model="morning_v1")
        validation.record_issued(self.conn, result, "v1", issue + 1)
        row = next(r for r in self.rows() if r["head"] == "nextMorning")
        self.assertEqual(row["cohort"], "unsupported_horizon")
        self.assertFalse(row["available"])
        self.assertIn("session_target_outside_same_or_next_day", row["unavailableReasons"])

    def test_amended_morning_clock_is_0715_actual105_lead_old0730_is_distinct(self):
        issue = epoch(hour=7, minute=15)
        first = payload(issue)
        session(first, "morning", epoch(hour=9), model="morning_v1")
        validation.record_issued(self.conn, first, "v20", issue + 1)
        later_issue = epoch(hour=7, minute=30)
        old_clock = payload(later_issue)
        session(old_clock, "morning", epoch(hour=9), model="morning_v1")
        validation.record_issued(self.conn, old_clock, "v20", later_issue + 1)
        rows = [r for r in self.rows() if r["head"] == "morning"]
        amended = next(r for r in rows if r["issueEpoch"] == issue)
        former = next(r for r in rows if r["issueEpoch"] == later_issue)
        self.assertEqual(amended["cohort"], "canonical")
        self.assertEqual(amended["startEpoch"] - amended["issueEpoch"], 105 * 60)
        self.assertEqual(former["cohort"], "moving")
        self.assertEqual(former["startEpoch"] - former["issueEpoch"], 90 * 60)
        self.assertEqual(validation.default_policy()["canonicalIssueClocksMYT"]["morning"], "07:15")

    def test_morning_band_preserves_exact_clock_unavailable_first_and_prior_policy(self):
        issue = epoch(hour=7, minute=13)
        first = payload(issue)
        session(first, "morning", epoch(hour=9), model="morning_v1")
        validation.record_issued(self.conn, first, "v20", issue + 1)
        later_issue = epoch(hour=7, minute=15)
        later = payload(later_issue)
        session(later, "morning", epoch(hour=9), model="morning_v1")
        later["windows"]["morning"]["particleForecast"]["candidateForecast"] = {
            "mean": 10, "modelVersion": "raw_morning_v1", "numericalPolicyIdentifier": "raw_morning_policy_v1",
            "forecastedAtEpoch": later_issue, "startEpoch": epoch(hour=9), "endEpoch": epoch(hour=11)}
        validation.record_issued(self.conn, later, "v20", later_issue + 1)
        readings(self.conn, epoch(hour=9), epoch(hour=11), 10)
        report = validation.evaluate_issued(self.conn, epoch(hour=12), target="morningSession", cohort="canonical")
        self.assertEqual(report["selectedCases"], 1)
        self.assertEqual(report["candidateDiagnosticScoredCases"], 0)
        first_row = next(r for r in self.rows() if r["head"] == "morning" and r["issueEpoch"] == issue)
        self.assertEqual(first_row["issueEpoch"], epoch(hour=7, minute=13))
        old = dict(first_row, version="actual_issued_validation_v1", issueEpoch=epoch(hour=7, minute=30))
        self.conn.execute("UPDATE prospective_validation_issued SET payload=? WHERE case_id=?",
            (validation.encode_payload(validation._json(old)), first_row["caseId"]))
        report = validation.evaluate_issued(self.conn, epoch(hour=12), target="morningSession", cohort="canonical")
        self.assertEqual(report["exclusions"]["preserved_previous_measurement_policy"], 1)
        self.assertEqual(report["selectedCases"], 1)
        self.assertEqual(report["candidateDiagnosticScoredCases"], 1)
        raw_group = next(iter(report["candidateDiagnosticGroups"].values()))["morningSession"]
        self.assertTrue(raw_group["operationalInputsEstablished"])
        later_row = next(r for r in self.rows() if r["head"] == "morning" and r["issueEpoch"] == later_issue)
        self.assertEqual(raw_group["caseIds"], [later_row["caseId"]])

    def test_scoring_rollback_cannot_reopen_consumed_confirmation_labels(self):
        pre = epoch()
        validation.register_confirmation_candidate(self.conn, "point90", "x", 1, cohort="scheduled_near",
            registered_epoch=pre, selection_evidence_hash="b" * 64, selection_gate_passed=True,
            raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        issue = validation.SEALED_EPOCH
        result = payload(issue)
        result["airWindow"]["arrival"]["candidateForecast"] = {
            "modelVersion": "raw_v1", "numericalPolicyIdentifier": "raw_policy_v1", "mean": 10,
            "forecastedAtEpoch": issue, "startEpoch": issue + 5400, "endEpoch": issue + 5400}
        validation.record_issued(self.conn, result, "v1", issue + 1, candidate_versions={"point90": "x"})
        readings(self.conn, issue + 4500, issue + 6300, 10)
        validation.claim_confirmation_look(self.conn, "point90", "x", 1, cohort="scheduled_near",
            as_of_epoch=epoch(month=12, day=31), raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.conn.commit()
        report = validation.evaluate_issued(self.conn, epoch(month=12, day=31), confirmation=True,
            target="point90", candidate_version="x", look_number=1, cohort="scheduled_near",
            raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.assertEqual(report["candidateDiagnosticScoredCases"], 1)
        self.assertTrue(report["protectedLabelsOpened"])
        self.conn.rollback()
        self.assertEqual(self.conn.execute("SELECT status FROM prospective_validation_looks").fetchone()[0], "begun")
        with self.assertRaisesRegex(ValueError, "preregistered"):
            validation.evaluate_issued(self.conn, epoch(month=12, day=31), confirmation=True,
                target="point90", candidate_version="x", look_number=1, cohort="scheduled_near",
                raw_model_version="raw_v1", raw_numerical_policy="raw_policy_v1")
        self.assertEqual(sum("SELECT epoch,pm02 FROM readings" in sql for sql in self.sql), 1)


if __name__ == "__main__":
    unittest.main()
