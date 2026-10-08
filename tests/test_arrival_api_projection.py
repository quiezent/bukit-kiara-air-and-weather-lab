"""Synthetic consumer contracts for the separate learned arrival distribution.

No fits, provider requests, production database access or historical writes.
"""
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import MagicMock, patch

import compact_forecast_record as ledger
import rlcd_api
import rlcd_history
from test_rlcd_model_output import inputs
from test_weather_contracts import ANCHOR


TARGET = "exact_issue_plus90_arrival_median_proxy_delta_from_fresh5_reference"
TAIL_KEYS = ("probabilityFall20", "probabilityFall40", "probabilityRise20", "probabilityRise40")
DEVICE_TAIL_KEYS = ("fall20", "fall40", "rise20", "rise40")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def arrival_distribution():
    # Within20 wins decisively. All four minority tails must still survive.
    mass = [0.013456789012345, 0.12765432109876, 0.7, 0.138888888888895, 0.02]
    return {
        "available": True,
        "modelVersion": "receipt_visible_hgb_arrival_change20_40_distribution_v1",
        "adapterVersion": "direct_arrival_categorical_tail_adapter_v1",
        "forecastIssuedEpoch": ANCHOR,
        "arrivalEpoch": ANCHOR + 5400,
        "validUntilEpoch": ANCHOR + 5400,
        "arrivalLeadMinutes": 90,
        "target": TARGET,
        "referencePm": 159.87654321098765,
        "freshReferenceEpoch": ANCHOR - 30,
        "changeThresholdsUgM3": [20, 40],
        "classProbabilities": mass,
        "orderedClassNames": ["fall40", "fall20to40", "within20", "rise20to40", "rise40"],
        "probabilityFall20": mass[0] + mass[1],
        "probabilityFall40": mass[0],
        "probabilityRise20": mass[3] + mass[4],
        "probabilityRise40": mass[4],
        "probabilityWithin20": mass[2],
        "numericExpectationAvailable": False,
        "arrivalConcentrationExpectation": None,
        "numericalSelectionApplied": False,
        "modelIdentity": {"sha256": "synthetic-fixed-identity", "featureRevision": "test-r1"},
        "trainingCutoffEpoch": ANCHOR - 36000,
        "fittedAtEpoch": ANCHOR - 7200,
        "sensorWatermarkEpoch": ANCHOR - 30,
        "featureSourceMaxEpoch": ANCHOR - 30,
        "featureValuesSha256": "synthetic-feature-values",
        "featureValues": [159.87654321098765, None, -1.23456789012345],
        "featureProvenance": {"available": True, "rows": [{"epoch": ANCHOR - 30,
            "receivedEpoch": ANCHOR - 20, "recordedEpoch": ANCHOR - 19,
            "pm25": 159.87654321098765}], "historicalReceiptsUnknown": False},
        "selectedPolicy": {"model": "fixed-test-model", "selectionApplied": False},
    }


def rich_inputs():
    reading, analysis, weather, rows = inputs()
    analysis["current"] = deepcopy(reading)
    air = analysis["airWindow"]
    air["arrivalChangeForecast"] = arrival_distribution()
    air["arrival"]["baselinePoint"] = air["arrivalChangeForecast"]["referencePm"]
    air["firstCrossingEventForecast"]["modelVersion"] = "fresh_sensor_hgb_direct_first20_90min_v2"
    air["trail"]["rideExtrema"] = {
        "available": True, "low": 122.12345678901234, "high": 163.6543210987654,
        "forecastClock": deepcopy(air["forecastClock"]),
        "modelVersion": "fresh_sequence_ridge_delta_numeric_v3_20261008"}
    air["cyclingWindowForecast"] = {
        "available": True, "forecastIssuedEpoch": ANCHOR,
        "startEpoch": ANCHOR + 5400, "endEpoch": ANCHOR + 12600,
        "chanceMeanAtOrBelowCutoff": .043212345678901, "cutoffUgM3": 70,
        "modelVersion": "retained_mean_probability_model"}
    weather["source"] = "Synthetic hourly forecast with complete dated weather coverage"
    return reading, analysis, weather, rows


class ArrivalDistributionProjectionTests(unittest.TestCase):
    def project(self, values=None, now=ANCHOR + 20):
        reading, analysis, weather, rows = rich_inputs() if values is None else values
        return rlcd_api.build_payload(reading, analysis, weather, now, history_rows=rows)

    def assert_numeric_heads_unchanged(self, result, original):
        self.assertEqual(result["forecast"]["near90"]["pm25_ugm3"], original["airWindow"]["arrival"]["point"])
        self.assertEqual(result["forecast"]["ride90_210"]["pm25_ugm3"], original["airWindow"]["trail"]["point"])
        for session in ("morning", "afternoon"):
            self.assertEqual(result["forecast"]["sessions"][session]["pm"]["pm25_ugm3"],
                             original["windows"][session]["particleForecast"]["point"])

    def assert_distribution_unavailable(self, result):
        distribution = result["forecast"]["near90"]["arrival_change"]
        self.assertIs(distribution["available"], False)
        for key in DEVICE_TAIL_KEYS + ("reference_ugm3", "arrival_epoch"):
            self.assertIsNone(distribution[key], key)

    def test_all_four_native_tails_survive_dominant_within20_class_at_full_precision(self):
        values = rich_inputs()
        result = self.project(values)
        original = values[1]["airWindow"]["arrivalChangeForecast"]
        actual = result["forecast"]["near90"]["arrival_change"]
        self.assertIs(actual["available"], True)
        self.assertGreater(original["probabilityWithin20"], max(original[key] for key in TAIL_KEYS))
        for source_key, device_key in zip(TAIL_KEYS, DEVICE_TAIL_KEYS):
            self.assertEqual(actual[device_key].hex(), original[source_key].hex())
        self.assertEqual(actual["reference_ugm3"].hex(), original["referencePm"].hex())
        self.assertEqual(actual["arrival_epoch"], original["arrivalEpoch"])
        self.assertEqual(actual["model"], original["modelVersion"])
        self.assertEqual(set(actual), {"available", "fall20", "fall40", "rise20", "rise40",
                                      "within20", "display_text", "outcome", "outcome_probability",
                                      "issued_epoch", "fresh_reference_epoch",
                                      "reference_ugm3", "arrival_epoch", "model"})
        self.assert_numeric_heads_unchanged(result, values[1])

    def test_new_distribution_cannot_select_pm_points_or_alter_existing_first20(self):
        values = rich_inputs()
        without = deepcopy(values)
        without[1]["airWindow"].pop("arrivalChangeForecast")
        old = self.project(without)
        for mass in ([.7, .1, .15, .04, .01], [.01, .04, .15, .1, .7]):
            with self.subTest(mass=mass):
                value = values[1]["airWindow"]["arrivalChangeForecast"]
                value.update(probabilityFall20=mass[0] + mass[1], probabilityFall40=mass[0],
                             probabilityRise20=mass[3] + mass[4], probabilityRise40=mass[4],
                             probabilityWithin20=mass[2], classProbabilities=mass)
                value["selectedPolicy"] = {"selectedPolicy": "persistence", "eligible": True}
                result = self.project(values)
                self.assert_numeric_heads_unchanged(result, values[1])
                self.assertEqual(encoded(result["forecast"]["near90"]["first20"]),
                                 encoded(old["forecast"]["near90"]["first20"]))

    def test_distribution_can_remain_available_when_arrival_numeric_head_is_unavailable(self):
        values = rich_inputs()
        values[1]["airWindow"]["arrival"].update(available=False, point=None)
        result = self.project(values)
        self.assertFalse(result["forecast"]["near90"]["available"])
        self.assertIsNone(result["forecast"]["near90"]["pm25_ugm3"])
        self.assertTrue(result["forecast"]["near90"]["arrival_change"]["available"])

    def test_missing_or_explicitly_unavailable_distribution_does_not_erase_numeric_heads(self):
        for value in (None, {}, {"available": False, "modelVersion": "waiting_for_fixed_model"}):
            with self.subTest(value=value):
                values = rich_inputs()
                values[1]["airWindow"]["arrivalChangeForecast"] = value
                result = self.project(values)
                self.assert_distribution_unavailable(result)
                self.assert_numeric_heads_unchanged(result, values[1])

    def test_mismatched_old_future_issue_target_lead_or_definition_rejects_only_distribution(self):
        changes = ({"forecastIssuedEpoch": ANCHOR - 601, "arrivalEpoch": ANCHOR + 4799},
                   {"forecastIssuedEpoch": ANCHOR + 60, "arrivalEpoch": ANCHOR + 5460},
                   {"forecastIssuedEpoch": ANCHOR - 1}, {"arrivalEpoch": ANCHOR + 5401},
                   {"arrivalLeadMinutes": 89}, {"target": "first_crossing_anywhere_before_arrival"})
        for changeset in changes:
            with self.subTest(changes=changeset):
                values = rich_inputs()
                values[1]["airWindow"]["arrivalChangeForecast"].update(changeset)
                result = self.project(values)
                self.assert_distribution_unavailable(result)
                self.assertTrue(result["forecast"]["near90"]["available"])
                self.assert_numeric_heads_unchanged(result, values[1])

    def test_parent_future_expired_or_stale_issue_cannot_be_reissued_by_projection(self):
        for now, delivery in ((ANCHOR - 1, "ready"), (ANCHOR + 601, "ready"),
                              (ANCHOR + 20, "expired")):
            with self.subTest(now=now, delivery=delivery):
                values = rich_inputs()
                values[1]["delivery"]["state"] = delivery
                result = self.project(values, now)
                self.assert_distribution_unavailable(result)
                self.assertFalse(result["forecast"]["near90"]["available"])
                self.assertIsNone(result["forecast"]["near90"]["pm25_ugm3"])
                self.assertEqual(values[1]["forecastIssuedEpoch"], ANCHOR)

    def test_invalid_probability_type_bounds_or_nonfinite_value_fail_without_fallback(self):
        for key in TAIL_KEYS + ("probabilityWithin20",):
            for invalid in (None, True, False, "0.1", math.nan, math.inf, -math.inf, -.01, 1.01):
                with self.subTest(key=key, invalid=repr(invalid)):
                    values = rich_inputs()
                    values[1]["airWindow"]["arrivalChangeForecast"][key] = invalid
                    result = self.project(values)
                    self.assert_distribution_unavailable(result)
                    self.assert_numeric_heads_unchanged(result, values[1])
                    encoded(result)  # No invalid value leaks to strict JSON.

    def test_nonnested_tails_and_inconsistent_probability_sum_are_rejected(self):
        changes = ({"probabilityFall40": .5}, {"probabilityRise40": .5},
                   {"probabilityWithin20": .5}, {"probabilityFall20": .2})
        for changeset in changes:
            with self.subTest(changes=changeset):
                values = rich_inputs()
                values[1]["airWindow"]["arrivalChangeForecast"].update(changeset)
                result = self.project(values)
                self.assert_distribution_unavailable(result)
                self.assert_numeric_heads_unchanged(result, values[1])

    def test_projection_is_read_only_and_graph_budget_preserves_both_forecast_models(self):
        values = rich_inputs()
        original = encoded(values)
        result = self.project(values)
        size = len(json.dumps(result, ensure_ascii=True, allow_nan=False,
                              separators=(",", ":")).encode("utf-8"))
        self.assertLessEqual(size, 16384)
        self.assertEqual(encoded(values), original)
        self.assertTrue(result["forecast"]["near90"]["arrival_change"]["available"])
        self.assertTrue(result["forecast"]["near90"]["first20"]["available"])
        self.assertTrue(result["forecast"]["ride90_210"]["minimum_maximum"]["available"])
        self.assertTrue(result["forecast"]["ride90_210"]["mean_le70"]["available"])
        self.assert_numeric_heads_unchanged(result, values[1])
        full_history = rlcd_history.build_history(values[3], ANCHOR + 20)
        self.assertEqual(result["history"]["pm25_summary"], full_history["pm25_summary"])
        self.assertEqual(result["history"]["gaps"], full_history["gaps"])
        self.assertEqual(result["history"]["points"][0], full_history["points"][0])
        self.assertEqual(result["history"]["points"][-1], full_history["points"][-1])
        for weather in (result["forecast"]["ride90_210"]["weather"],
                        result["forecast"]["sessions"]["morning"]["weather"],
                        result["forecast"]["sessions"]["afternoon"]["weather"]):
            self.assertTrue(weather["available"])
            self.assertTrue(weather["coverage_verified"])


class ArrivalDistributionLedgerAndCoachTests(unittest.TestCase):
    def test_future_compact_issue_preserves_full_distribution_identity_clock_and_lineage(self):
        analysis = rich_inputs()[1]
        original = encoded(analysis)
        with patch("sqlite3.connect", side_effect=AssertionError("historical database write")), \
                patch("builtins.open", side_effect=AssertionError("historical file write")):
            compact = ledger.compact_dashboard_record(analysis, "synthetic-v24")
        self.assertEqual(compact["archiveRecordVersion"], "dashboard_forecast_ledger_v6_arrival_change_outputs")
        self.assertEqual(encoded(compact["airWindow"]["arrivalChangeForecast"]),
                         encoded(analysis["airWindow"]["arrivalChangeForecast"]))
        self.assertEqual(compact["forecastIssuedEpoch"], ANCHOR)
        self.assertEqual(encoded(analysis), original)
        copied = compact["airWindow"]["arrivalChangeForecast"]
        copied["modelIdentity"]["sha256"] = "changed"
        copied["featureProvenance"]["rows"][0]["receivedEpoch"] = ANCHOR + 50
        copied["classProbabilities"][0] = 1
        copied["featureValues"].append(99)
        self.assertEqual(encoded(analysis), original)

    def test_legacy_payload_has_no_invented_arrival_distribution_and_is_never_rewritten(self):
        historical = rich_inputs()[1]
        historical["airWindow"].pop("arrivalChangeForecast")
        original = encoded(historical)
        with patch("sqlite3.connect", side_effect=AssertionError("historical database write")), \
                patch("builtins.open", side_effect=AssertionError("historical file write")):
            compact = ledger.compact_dashboard_record(historical)
        self.assertNotIn("arrivalChangeForecast", compact["airWindow"])
        self.assertEqual(encoded(historical), original)
        self.assertEqual(encoded(compact["airWindow"]["firstCrossingEventForecast"]),
                         encoded(historical["airWindow"]["firstCrossingEventForecast"]))

    def test_coach_preserves_entire_native_object_and_both_existing_forecast_targets(self):
        import BukitKiara_Dashboard as dashboard
        reading, analysis, _, _ = rich_inputs()
        original = encoded(analysis)
        connection = MagicMock()
        connection.__enter__.return_value.execute.return_value.fetchone.return_value = reading
        with patch.object(dashboard, "analysis", return_value=analysis), \
                patch.object(dashboard, "db", return_value=connection), \
                patch.object(dashboard, "latest_air_quality_payload", return_value=None), \
                patch.object(dashboard, "coach_api_contract_metadata", return_value={}), \
                patch("sqlite3.connect", side_effect=AssertionError("production DB access")):
            result = dashboard.ride_conditions_api(now_epoch=ANCHOR + 20)
        self.assertEqual(encoded(result["exposureOutlook"]["arrivalChangeForecast"]),
                         encoded(analysis["airWindow"]["arrivalChangeForecast"]))
        self.assertEqual(encoded(result["exposureOutlook"]["firstCrossingEventForecast"]),
                         encoded(analysis["airWindow"]["firstCrossingEventForecast"]))
        self.assertEqual(result["exposureOutlook"]["arrival"]["projectedPm25UgM3"],
                         analysis["airWindow"]["arrival"]["point"])
        self.assertEqual(result["exposureOutlook"]["onTrail"]["projectedMeanPm25UgM3"],
                         analysis["airWindow"]["trail"]["point"])
        self.assertEqual(encoded(analysis), original)


if __name__ == "__main__":
    unittest.main()
