"""Independent publication checks for the fixed fresh numeric model.

Synthetic source outputs exercise stale forecast removal, literal values and
cross-consumer provenance. No fitting, service calls or database writes occur.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
import json
import unittest

import compact_forecast_record
import fresh_model_output as publication
import fresh_numeric_model
import rlcd_api
from test_rlcd_model_output import inputs
from test_weather_contracts import ANCHOR


def source(issue=ANCHOR):
    return {
        "available": True,
        "modelVersion": fresh_numeric_model.MODEL_VERSION,
        "forecastIssuedEpoch": issue,
        "forecastClock": fresh_numeric_model.forecast_clock(issue),
        "freshReferencePm25": 151.987654321,
        "arrivalPoint": 127.12345678901234,
        "trailMeanPoint": 115.98765432109876,
        "trailMinimumPoint": 90.23456789012345,
        "trailMaximumPoint": 145.87654321098765,
        "performanceEvidence": {"prospectivelyValidated": False, "mae": 999999},
        "modelIdentity": {"modelVersion": fresh_numeric_model.MODEL_VERSION,
                          "identitySha256": "immutable-fixed-model-identity"},
        "training": {"trainingCutoffEpoch": issue-86400,
                     "trainingDataHash": "immutable-causal-training-hash"},
        "inputLineage": {"forecastIssuedEpoch": issue,
                         "featureSourceMaxEpoch": issue-30,
                         "freshReferenceEpoch": issue-30,
                         "freshReferencePm25": 151.987654321,
                         "inputValuesSha256": "immutable-fresh-query-hash",
                         "allInputReceiptsKnown": False},
    }


def fixture():
    reading, analysis, weather, rows = inputs()
    analysis["current"] = deepcopy(reading)
    analysis["airWindow"]["forecastClock"] = fresh_numeric_model.forecast_clock(ANCHOR)
    for name in ("arrival", "trail"):
        head = analysis["airWindow"][name]
        head.update(point=999.87654321, baselinePoint=160.1,
                    persistenceAnchorEpoch=ANCHOR-30,
                    persistenceAnchorRole="trailing_5_minute_raw_sensor_median",
                    freshnessAdjustment={"applied": True},
                    rawUpperMean90=777., rawPeakUpper90=888.,
                    upperMean=777., decisionUpperMean=777.,
                    rawPeakRangeLow=700., rawPeakRangeHigh=888.,
                    rawPeak=888., rawProjectedPeak=888., upperPeak=888.,
                    upperPeak90=888., empiricalPeakUpper=888.,
                    finiteSampleUpper=True, finiteSampleRankCoverage=.9,
                    rawRangeLow=100., rawRangeHigh=900.)
    return reading, analysis, weather, rows


class FreshOutputPublicationTests(unittest.TestCase):
    def setUp(self):
        self.reading, self.analysis, self.weather, self.rows = fixture()
        self.air = self.analysis["airWindow"]
        self.raw = source()

    def publish(self, raw=None, air=None):
        return publication.attach_outputs(self.air if air is None else air,
                                          self.raw if raw is None else raw, ANCHOR)

    def assert_unavailable(self, actual):
        for name in ("arrival", "trail"):
            self.assertFalse(actual[name]["available"])
            self.assertIsNone(actual[name]["point"])
        extrema = actual["trail"]["rideExtrema"]
        self.assertFalse(extrema["available"])
        self.assertIsNone(extrema["low"])
        self.assertIsNone(extrema["high"])

    def test_literal_precision_and_provenance_are_copied_without_mutation(self):
        before = deepcopy((self.air, self.raw))
        actual = self.publish()
        self.assertEqual((self.air, self.raw), before)
        for name, field in (("arrival", "arrivalPoint"), ("trail", "trailMeanPoint")):
            head = actual[name]
            self.assertTrue(head["available"])
            self.assertEqual(head["point"], self.raw[field])
            self.assertEqual(head["modelVersion"], self.raw["modelVersion"])
            self.assertEqual(head["modelFeatureAnchorEpoch"], ANCHOR)
            self.assertEqual(head["modelFeatureAnchor"], self.raw["freshReferencePm25"])
            self.assertEqual(head["modelOutput"]["sourcePath"], "freshSensorForecast."+field)
            self.assertTrue(head["modelOutput"]["outputPublishedWithoutArithmetic"])
            for key in ("modelIdentity", "training", "inputLineage"):
                self.assertEqual(head["modelOutput"][key], self.raw[key])
                self.assertIsNot(head["modelOutput"][key], self.raw[key])
            self.assertFalse(head["validated"])
            self.assertFalse(head["qualified"])
            self.assertEqual(head["baselinePoint"], self.raw["freshReferencePm25"])
            self.assertEqual(head["persistenceAnchorEpoch"], self.raw["inputLineage"]["freshReferenceEpoch"])

    def test_equal_zero_negative_and_outside_range_outputs_are_not_replaced(self):
        for arrival, mean in ((160.1, 160.1), (0., -5.123456789), (-1.987654321, 999.25)):
            raw = deepcopy(self.raw)
            raw.update(arrivalPoint=arrival, trailMeanPoint=mean)
            with self.subTest(arrival=arrival, mean=mean):
                actual = self.publish(raw)
                self.assertEqual(actual["arrival"]["point"], arrival)
                self.assertEqual(actual["trail"]["point"], mean)
                self.assertEqual(actual["trail"]["rideExtrema"]["low"], raw["trailMinimumPoint"])
                self.assertEqual(actual["trail"]["rideExtrema"]["high"], raw["trailMaximumPoint"])

    def test_crossed_extrema_are_retained_and_marked_without_sorting(self):
        self.raw.update(trailMinimumPoint=170.123456789, trailMaximumPoint=80.987654321)
        extrema = self.publish()["trail"]["rideExtrema"]
        self.assertTrue(extrema["available"])
        self.assertTrue(extrema["crossing"])
        self.assertEqual(extrema["low"], 170.123456789)
        self.assertEqual(extrema["high"], 80.987654321)
        for key in ("outputSorting", "outputWidening", "meanRecentering", "persistenceFallback"):
            self.assertFalse(extrema[key])

    def test_missing_or_false_source_never_reuses_old_points(self):
        for raw in ({}, {**self.raw, "available": False},
                    {**self.raw, "available": None}, {**self.raw, "available": "true"}):
            with self.subTest(raw=raw):
                self.assert_unavailable(self.publish(raw))

    def test_malformed_numbers_are_unavailable_in_their_own_heads(self):
        for field, name in (("arrivalPoint", "arrival"), ("trailMeanPoint", "trail")):
            for value in (None, True, False, "123.4", float("nan"), float("inf"), 10**500):
                raw = deepcopy(self.raw)
                raw[field] = value
                with self.subTest(field=field, value=repr(value)):
                    actual = self.publish(raw)
                    self.assertFalse(actual[name]["available"])
                    self.assertIsNone(actual[name]["point"])
                    self.assertTrue(actual["trail"]["rideExtrema"]["available"])
        for field in ("trailMinimumPoint", "trailMaximumPoint"):
            for value in (None, True, "123.4", float("nan"), float("inf")):
                raw = deepcopy(self.raw)
                raw[field] = value
                with self.subTest(field=field, value=repr(value)):
                    actual = self.publish(raw)
                    self.assertTrue(actual["arrival"]["available"])
                    self.assertTrue(actual["trail"]["available"])
                    self.assertFalse(actual["trail"]["rideExtrema"]["available"])

    def test_missing_malformed_and_wrong_target_clocks_cannot_relabel_a_number(self):
        for clock in (None, {}, [], "clock"):
            raw = deepcopy(self.raw)
            raw["forecastClock"] = clock
            with self.subTest(clock=clock):
                self.assert_unavailable(self.publish(raw))
        for field in ("forecastIssuedEpoch", "arrivalTargetEpoch", "windowStartEpoch", "windowEndEpoch"):
            raw = deepcopy(self.raw)
            raw["forecastClock"][field] -= 1
            with self.subTest(field=field):
                self.assert_unavailable(self.publish(raw))

    def test_fresh_anchor_identity_and_parent_clock_cannot_be_relabelled(self):
        for key, value in (("featureAnchorEpoch", ANCHOR-900),
                           ("featureAnchorAgeSeconds", 900),
                           ("targetVersion", "different_target_definition")):
            raw = deepcopy(self.raw)
            raw["forecastClock"][key] = value
            with self.subTest(key=key):
                self.assert_unavailable(self.publish(raw))
        air = deepcopy(self.air)
        air["forecastClock"]["windowEndEpoch"] += 1
        self.assert_unavailable(self.publish(air=air))

    def test_valid_primary_parent_clock_records_the_new_fresh_issue_anchor(self):
        air = deepcopy(self.air)
        air["forecastClock"].update(featureAnchorEpoch=ANCHOR-137,
                                   featureAnchorAgeSeconds=137)
        actual = self.publish(air=air)
        self.assertEqual(actual["forecastClock"], self.raw["forecastClock"])
        self.assertEqual(actual["forecastClock"]["featureAnchorEpoch"], ANCHOR)
        self.assertEqual(actual["forecastClock"]["featureAnchorAgeSeconds"], 0)
        self.assertEqual(air["forecastClock"]["featureAnchorEpoch"], ANCHOR-137)

    def test_source_targets_replace_stale_head_times_for_device_projection(self):
        air = deepcopy(self.air)
        air["arrival"]["expectedEpoch"] = ANCHOR+5000
        air["trail"]["startEpoch"] = ANCHOR+5000
        air["trail"]["endEpoch"] = ANCHOR+12000
        actual = self.publish(air=air)
        self.assertEqual(actual["arrival"]["expectedEpoch"], ANCHOR+5400)
        self.assertEqual(actual["trail"]["startEpoch"], ANCHOR+5400)
        self.assertEqual(actual["trail"]["endEpoch"], ANCHOR+12600)

    def test_old_uncertainty_aliases_are_not_associated_with_a_new_model(self):
        actual = self.publish()
        aliases = ("rawUpperMean90", "upperMean", "decisionUpperMean", "rawPeakUpper90",
                   "rawPeakRangeLow", "rawPeakRangeHigh", "rawPeak", "rawProjectedPeak",
                   "upperPeak", "upperPeak90", "empiricalPeakUpper", "rawRangeLow", "rawRangeHigh")
        for name in ("arrival", "trail"):
            for field in aliases:
                with self.subTest(head=name, field=field):
                    self.assertIsNone(actual[name].get(field))

    def test_full_raw_source_and_issue_metadata_survive_future_immutable_ledger(self):
        result = deepcopy(self.analysis)
        result["airWindow"] = self.publish()
        result["freshSensorForecast"] = deepcopy(self.raw)
        before = deepcopy(result)
        archived = compact_forecast_record.compact_dashboard_record(result, build="synthetic-review-build")
        self.assertEqual(result, before)
        self.assertEqual(archived["freshSensorForecast"], self.raw)
        self.assertEqual(archived["airWindow"]["arrival"]["modelOutput"]["inputLineage"], self.raw["inputLineage"])
        self.assertEqual(archived["airWindow"]["trail"]["rideExtrema"]["modelIdentity"], self.raw["modelIdentity"])

    def test_device_preserves_literal_outputs_separate_reference_and_payload_budget(self):
        self.analysis["airWindow"] = self.publish()
        actual = rlcd_api.build_payload(self.reading, self.analysis, self.weather, ANCHOR+20,
                                        dashboard_build="synthetic-review-build", history_rows=self.rows)
        near, ride = actual["forecast"]["near90"], actual["forecast"]["ride90_210"]
        self.assertTrue(near["available"])
        self.assertTrue(ride["available"])
        self.assertEqual(near["pm25_ugm3"], self.raw["arrivalPoint"])
        self.assertEqual(ride["pm25_ugm3"], self.raw["trailMeanPoint"])
        self.assertEqual(near["reference_pm25_ugm3"], self.raw["freshReferencePm25"])
        self.assertEqual(near["model"], self.raw["modelVersion"])
        self.assertEqual(ride["model"], self.raw["modelVersion"])
        self.assertEqual(ride["minimum_maximum"]["minimum_ugm3"], self.raw["trailMinimumPoint"])
        self.assertEqual(ride["minimum_maximum"]["maximum_ugm3"], self.raw["trailMaximumPoint"])
        self.assertEqual(ride["minimum_maximum"]["model"], self.raw["modelVersion"])
        self.assertIsNone(near["range_low_ugm3"])
        self.assertIsNone(ride["range_high_ugm3"])
        self.assertLessEqual(len(json.dumps(actual, ensure_ascii=True, allow_nan=False,
                                           separators=(",", ":")).encode()), rlcd_api.MAX_PAYLOAD_BYTES)


if __name__ == "__main__":
    unittest.main()
