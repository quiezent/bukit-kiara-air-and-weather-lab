"""Behavioral contracts for publishing the fixed arrival learner's output."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
import json
import math
import unittest

import arrival_model_output as contract
from forecast_clock import ForecastClock, TARGET_VERSION
import model_output_contracts


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class DedicatedArrivalOutputContracts(unittest.TestCase):
    def setUp(self):
        self.issue = 1791450145
        self.parent_clock = ForecastClock(
            self.issue, self.issue // 900 * 900
        ).metadata()
        self.source_clock = {
            "targetVersion": TARGET_VERSION,
            "forecastIssuedEpoch": self.issue,
            "featureAnchorEpoch": self.issue,
            "featureAnchorAgeSeconds": 0,
            "arrivalTargetEpoch": self.issue + 5400,
            "arrivalTargetDefinition": "Fixed dedicated arrival learner outcome",
            "additionalSourceClockMetadata": {"recipe": "native_arrival_target"},
        }
        self.reference = 104.31234567890123
        self.source = {
            "available": True,
            "arrivalPoint": 84.12345678901234,
            "freshReferencePm25": self.reference,
            "modelVersion": "fixed_dedicated_arrival_recipe",
            "forecastClock": deepcopy(self.source_clock),
            "modelIdentity": {"name": "dedicated_arrival", "sha256": "test-fixed-identity"},
            "training": {"originCount": 817, "features": ["fresh_pm25", "wind"]},
            "inputLineage": {
                "freshReferenceEpoch": self.issue - 17,
                "sensorRows": [{"epoch": self.issue - 17, "pm25": self.reference}],
            },
            "performanceEvidence": {
                "qualified": False, "modelMae": 9999.0, "persistenceMae": 0.01,
            },
        }
        old_arrival = {
            "available": True,
            "point": 157.9,
            "baselinePoint": 154.5,
            "pointRole": "persistence_anchor",
            "modelVersion": "old_local_ridge_recipe",
            "modelSelection": {"chosen": "persistence", "eligible": True},
            "deploymentSkill": {"eligible": True},
            "qualification": {"qualified": True},
            "qualificationPolicy": {"minimumSkill": 0.1},
            "candidateForecast": {"point": 132.1, "modelVersion": "old_candidate"},
            "upperTargetCoverage": 0.99,
            "finiteSampleRankCoverage": 0.97,
            "finiteSampleUpper": True,
            "finiteSampleRank": True,
            "meanSkillEligible": True,
            "peakSkillEligible": True,
        }
        old_arrival.update({key: 187.4321 for key in model_output_contracts._NUMERIC_UNCERTAINTY_FIELDS})
        old_arrival.update({key: "old_persistence_interval" for key in model_output_contracts._UNCERTAINTY_ROLE_FIELDS})
        self.air = {
            "available": True,
            "forecastClock": deepcopy(self.parent_clock),
            "currentFast": self.reference,
            "arrival": old_arrival,
            "trail": {
                "available": True, "point": 113.12345678901234,
                "rideExtrema": {"available": True, "low": 94.22222, "high": 145.99999},
                "forecastClock": deepcopy(self.parent_clock),
            },
            "firstCrossingEventForecast": {
                "available": True, "forecastIssuedEpoch": self.issue,
                "riseProbability": 0.008, "dropProbability": 0.14,
                "noCrossingProbability": 0.852, "thresholdUgM3": 20,
            },
            "arrivalChangeForecast": {
                "available": True, "probabilities": [0.008, 0.14, 0.852],
                "forecastClock": deepcopy(self.parent_clock),
            },
            "observedMovement": {
                "eventLabel": "Rapid PM2.5 reduction detected",
                "observedThroughEpoch": self.issue,
            },
        }

    def publish(self, source=None, air=None):
        return contract.attach_numeric_output(
            self.air if air is None else air,
            self.source if source is None else source,
            self.issue,
        )

    def assert_unavailable_without_fallback(self, result):
        head = result["arrival"]
        self.assertFalse(head["available"])
        self.assertIsNone(head["point"])
        self.assertFalse(head["modelOutput"]["outputPublishedWithoutArithmetic"])
        self.assertNotIn("candidateForecast", head)
        self.assertNotIn("modelSelection", head)

    def test_full_precision_zero_negative_and_equal_current_are_literal_outputs(self):
        for value in (84.12345678901234, 0.0, -7.123456789012345, self.reference):
            with self.subTest(value=value):
                source = deepcopy(self.source)
                source["arrivalPoint"] = value
                head = self.publish(source)["arrival"]
                self.assertTrue(head["available"])
                self.assertEqual(head["point"], value)
                self.assertEqual(head["point"].hex(), value.hex())
                self.assertEqual(head["modelOutput"]["rawOutputUgM3"], value)
                self.assertTrue(head["modelOutput"]["outputPublishedWithoutArithmetic"])
                self.assertEqual(head["baselinePoint"], self.reference)

    def test_integral_model_output_is_not_coerced_or_replaced(self):
        source = deepcopy(self.source)
        source["arrivalPoint"] = 91
        head = self.publish(source)["arrival"]
        self.assertTrue(head["available"])
        self.assertEqual(head["point"], 91)
        self.assertIs(type(head["point"]), int)

    def test_invalid_outputs_are_unavailable_instead_of_old_model_or_baseline(self):
        for value in (None, math.nan, math.inf, -math.inf, True, False,
                      "84.12345678901234", [], {}, complex(80, 1), 10**400, -(10**400)):
            with self.subTest(value=repr(value)):
                source = deepcopy(self.source)
                source["arrivalPoint"] = value
                result = self.publish(source)
                self.assert_unavailable_without_fallback(result)
                self.assertEqual(result["arrival"]["reason"], "arrival_model_number_invalid")
                self.assertEqual(result["arrival"]["baselinePoint"], self.reference)

    def test_missing_output_cannot_use_an_alternative_model_field(self):
        source = deepcopy(self.source)
        source.pop("arrivalPoint")
        source.update(point=95.0, ridgePoint=121.0, persistencePoint=self.reference)
        self.assert_unavailable_without_fallback(self.publish(source))

    def test_unavailable_source_is_not_promoted_by_existing_good_point(self):
        for available in (False, None, 1, "true"):
            with self.subTest(available=available):
                source = deepcopy(self.source)
                source.update(available=available, reason="native_inputs_unavailable")
                result = self.publish(source)
                self.assert_unavailable_without_fallback(result)
                self.assertEqual(result["arrival"]["reason"], "native_inputs_unavailable")

    def test_missing_source_is_unavailable_without_erasing_other_heads(self):
        for source in ({}, None, []):
            with self.subTest(source=source):
                result = contract.attach_numeric_output(self.air, source, self.issue)
                self.assert_unavailable_without_fallback(result)
                self.assertEqual(serialized(result["trail"]), serialized(self.air["trail"]))

    def test_source_issue_anchor_version_and_target_must_match_exact_clock(self):
        invalid_fields = {
            "targetVersion": "another_target_definition",
            "forecastIssuedEpoch": self.issue - 1,
            "featureAnchorEpoch": self.issue - 1,
            "featureAnchorAgeSeconds": 1,
            "arrivalTargetEpoch": self.issue + 5401,
        }
        for field, invalid in invalid_fields.items():
            with self.subTest(field=field, mode="wrong"):
                source = deepcopy(self.source)
                source["forecastClock"][field] = invalid
                result = self.publish(source)
                self.assert_unavailable_without_fallback(result)
                self.assertEqual(result["arrival"]["reason"], "arrival_model_target_clock_mismatch")
            with self.subTest(field=field, mode="missing"):
                source = deepcopy(self.source)
                source["forecastClock"].pop(field)
                self.assert_unavailable_without_fallback(self.publish(source))

    def test_parent_issue_version_and_target_must_identify_same_target(self):
        invalid_fields = {
            "targetVersion": "old_target_definition",
            "forecastIssuedEpoch": self.issue - 1,
            "arrivalTargetEpoch": self.issue + 5401,
        }
        for field, invalid in invalid_fields.items():
            with self.subTest(field=field, mode="wrong"):
                air = deepcopy(self.air)
                air["forecastClock"][field] = invalid
                self.assert_unavailable_without_fallback(self.publish(air=air))
            with self.subTest(field=field, mode="missing"):
                air = deepcopy(self.air)
                air["forecastClock"].pop(field)
                self.assert_unavailable_without_fallback(self.publish(air=air))

    def test_source_identity_training_lineage_and_native_clock_are_retained(self):
        head = self.publish()["arrival"]
        self.assertEqual(head["modelVersion"], self.source["modelVersion"])
        self.assertEqual(head["forecastClock"], self.source_clock)
        self.assertEqual(head["forecastIssuedEpoch"], self.issue)
        self.assertEqual(head["modelFeatureAnchorEpoch"], self.issue)
        self.assertEqual(head["persistenceAnchorEpoch"], self.issue - 17)
        for name in ("expectedEpoch", "startEpoch", "endEpoch"):
            self.assertEqual(head[name], self.issue + 5400)
        metadata = head["modelOutput"]
        self.assertEqual(metadata["sourcePath"], "arrivalModelForecast.arrivalPoint")
        self.assertEqual(metadata["sourceModelVersion"], self.source["modelVersion"])
        self.assertEqual(metadata["sourceForecastClock"], self.source_clock)
        for name in ("modelIdentity", "training", "inputLineage"):
            self.assertEqual(serialized(metadata[name]), serialized(self.source[name]))

    def test_parent_clock_is_preserved_without_rebasing_other_heads(self):
        result = self.publish()
        self.assertEqual(serialized(result["forecastClock"]), serialized(self.air["forecastClock"]))
        self.assertEqual(result["arrival"]["forecastClock"]["featureAnchorEpoch"], self.issue)
        self.assertLess(result["forecastClock"]["featureAnchorEpoch"], self.issue)

    def test_superseded_uncertainty_and_selection_metadata_are_cleared(self):
        head = self.publish()["arrival"]
        numeric_fields = model_output_contracts._NUMERIC_UNCERTAINTY_FIELDS + (
            "upperTargetCoverage", "finiteSampleRankCoverage",
        )
        for field in numeric_fields:
            with self.subTest(field=field):
                self.assertIsNone(head[field])
        for field in model_output_contracts._UNCERTAINTY_ROLE_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(head[field], "raw_model_interval_unavailable")
        for field in ("finiteSampleUpper", "finiteSampleRank", "meanSkillEligible", "peakSkillEligible", "qualified", "validated",
                      "prospectivelyValidated", "calibrated", "usedForDecision", "usedForComparison"):
            with self.subTest(field=field):
                self.assertIs(head[field], False)
        for field in ("modelSelection", "deploymentSkill", "qualification", "qualificationPolicy", "candidateForecast"):
            self.assertNotIn(field, head)

    def test_performance_and_qualification_flags_cannot_select_the_number(self):
        for qualified in (False, True):
            for eligible in (False, True):
                with self.subTest(qualified=qualified, eligible=eligible):
                    source = deepcopy(self.source)
                    source.update(qualified=qualified, validated=qualified,
                                  deploymentSkill={"eligible": eligible})
                    source["performanceEvidence"].update(
                        qualified=qualified, eligible=eligible, modelMae=9999.0, persistenceMae=0.0
                    )
                    head = self.publish(source)["arrival"]
                    self.assertTrue(head["available"])
                    self.assertEqual(head["point"].hex(), source["arrivalPoint"].hex())
                    self.assertEqual(head["performanceEvidence"], source["performanceEvidence"])

    def test_other_forecasts_and_observed_labels_are_byte_preserved(self):
        for available in (True, False):
            with self.subTest(available=available):
                source = deepcopy(self.source)
                source["available"] = available
                result = self.publish(source)
                for name in ("trail", "firstCrossingEventForecast", "arrivalChangeForecast", "observedMovement"):
                    with self.subTest(head=name):
                        self.assertEqual(serialized(result[name]), serialized(self.air[name]))

    def test_inputs_are_not_mutated_and_published_nested_objects_are_independent(self):
        original_air = deepcopy(self.air)
        original_source = deepcopy(self.source)
        result = self.publish()
        self.assertEqual(serialized(self.air), serialized(original_air))
        self.assertEqual(serialized(self.source), serialized(original_source))
        result["arrival"]["forecastClock"]["additionalSourceClockMetadata"]["recipe"] = "changed"
        result["arrival"]["modelOutput"]["sourceForecastClock"]["forecastIssuedEpoch"] = 1
        result["arrival"]["modelOutput"]["modelIdentity"]["name"] = "changed"
        result["arrival"]["modelOutput"]["training"]["features"].append("changed")
        result["arrival"]["modelOutput"]["inputLineage"]["sensorRows"][0]["pm25"] = 1
        result["arrival"]["performanceEvidence"]["modelMae"] = 1
        result["trail"]["rideExtrema"]["low"] = 1
        result["firstCrossingEventForecast"]["riseProbability"] = 1
        result["arrivalChangeForecast"]["probabilities"][0] = 1
        self.assertEqual(serialized(self.air), serialized(original_air))
        self.assertEqual(serialized(self.source), serialized(original_source))

    def test_invalid_baseline_metadata_never_invalidates_or_replaces_valid_prediction(self):
        for reference in (None, True, math.nan, math.inf, "104.3"):
            with self.subTest(reference=reference):
                source = deepcopy(self.source)
                source["freshReferencePm25"] = reference
                head = self.publish(source)["arrival"]
                self.assertTrue(head["available"])
                self.assertEqual(head["point"], source["arrivalPoint"])
                self.assertIsNone(head["baselinePoint"])


if __name__ == "__main__":
    unittest.main()
