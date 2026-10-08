"""Exact-output/clock regression checks without I/O or model fitting."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
import unittest

from forecast_clock import ForecastClock
import model_output_contracts as contract


class DirectNearModelOutputTests(unittest.TestCase):
    def setUp(self):
        self.anchor = 1791437400
        self.issue = self.anchor + 389
        self.clock = ForecastClock(self.issue, self.anchor).metadata()
        baseline = {"available": True, "point": 161.3, "baselinePoint": 161.3,
                    "pointRole": "persistence_anchor", "forecastClock": self.clock,
                    "forecastIssuedEpoch": self.issue, "persistenceAnchorEpoch": self.issue - 180,
                    "persistenceAnchorRole": "trailing_5_minute_raw_sensor_median",
                    "freshnessAdjustment": {"applied": True, "referenceEpoch": self.issue - 180},
                    "rangeLow": 141., "rangeHigh": 170., "upper90": 170., "projectedPeak": 170.}
        self.air = {"available": True, "forecastClock": self.clock,
                    "arrival": deepcopy(baseline), "trail": deepcopy(baseline),
                    "firstCrossingEventForecast": {"referencePm": 161.3, "changeThresholdUgM3": 20}}
        self.source = {"enabled": True, "status": "replay_screened_experimental",
                       "forecastClock": deepcopy(self.clock), "featureAnchor": 157.,
                       "featureAnchorEpoch": self.anchor, "arrivalPoint": 157.9, "trailMeanPoint": 141.9,
                       "arrivalDelta": .865, "trailMeanDelta": -15.1069,
                       "trainingOriginCount": 2614, "trailMeanRidgeTrainingOriginCount": 2618,
                       "backtest": {"rawAnalogueArrivalMae": 8.05, "arrivalPersistenceMae": 7.77,
                                    "rawRidgeTrailMeanMae": 9.43, "trailMeanPersistenceMae": 9.13,
                                    "deploymentSkill": {"arrival": {"eligible": False}, "trailMean": {"eligible": False}}}}

    def project(self, source=None, air=None, issue=None):
        return contract.attach_near_model_outputs(self.air if air is None else air,
                self.source if source is None else source, self.issue if issue is None else issue)

    def test_exact_raw_values_are_published_without_rebase(self):
        result = self.project()
        self.assertEqual(result["arrival"]["point"], self.source["arrivalPoint"])
        self.assertEqual(result["trail"]["point"], self.source["trailMeanPoint"])
        for name in ("arrival", "trail"):
            self.assertEqual(result[name]["baselinePoint"], 161.3)
            self.assertEqual(result[name]["modelFeatureAnchor"], 157.)
            self.assertEqual(result[name]["forecastState"], "model_output")
            self.assertTrue(result[name]["available"])
            self.assertFalse(result[name]["validated"])
            self.assertFalse(result[name]["usedForDecision"])
            self.assertEqual(result[name]["candidateForecast"]["point"], result[name]["point"])
        self.assertEqual(result["arrival"]["candidateForecast"]["startEpoch"], self.issue + 5400)
        self.assertEqual(result["arrival"]["candidateForecast"]["endEpoch"], self.issue + 5400)
        self.assertEqual(result["trail"]["candidateForecast"]["endEpoch"], self.issue + 12600)

    def test_no_promotion_or_performance_gate_can_change_the_number(self):
        source = deepcopy(self.source)
        source["backtest"] = {"deploymentSkill": {"arrival": {"eligible": False}},
                              "rawAnalogueArrivalMae": 999999., "arrivalPersistenceMae": 0.}
        result = self.project(source)
        self.assertEqual(result["arrival"]["point"], 157.9)
        self.assertEqual(result["trail"]["point"], 141.9)
        source["backtest"]["deploymentSkill"]["arrival"]["eligible"] = True
        self.assertEqual(self.project(source)["arrival"]["point"], 157.9)

    def test_equal_and_zero_raw_values_remain_equal_and_zero(self):
        source = deepcopy(self.source)
        source.update(arrivalPoint=161.3, trailMeanPoint=0.)
        result = self.project(source)
        self.assertEqual(result["arrival"]["point"], 161.3)
        self.assertEqual(result["trail"]["point"], 0.)
        self.assertTrue(result["trail"]["available"])

    def test_event_magnitude_never_supplies_numeric_forecast(self):
        air = deepcopy(self.air)
        air["firstCrossingEventForecast"].update(changeThresholdUgM3=999., projectedPm=999.)
        result = self.project(air=air)
        self.assertEqual(result["arrival"]["point"], 157.9)
        self.assertEqual(result["trail"]["point"], 141.9)
        self.assertEqual(result["firstCrossingEventForecast"], air["firstCrossingEventForecast"])

    def test_independent_missing_head_never_falls_back_to_baseline(self):
        source = deepcopy(self.source)
        source.pop("arrivalPoint")
        result = self.project(source)
        self.assertFalse(result["arrival"]["available"])
        self.assertIsNone(result["arrival"]["point"])
        self.assertEqual(result["arrival"]["baselinePoint"], 161.3)
        self.assertTrue(result["trail"]["available"])
        self.assertEqual(result["trail"]["point"], 141.9)

    def test_malformed_numbers_are_unavailable(self):
        for value in (True, "157.9", float("nan"), float("inf"), None):
            with self.subTest(value=value):
                source = deepcopy(self.source)
                source["arrivalPoint"] = value
                result = self.project(source)
                self.assertFalse(result["arrival"]["available"])
                self.assertIsNone(result["arrival"]["point"])

    def test_finite_output_is_not_clipped_by_publication(self):
        source = deepcopy(self.source)
        source["arrivalPoint"] = -1.25
        result = self.project(source)
        self.assertTrue(result["arrival"]["available"])
        self.assertEqual(result["arrival"]["point"], -1.25)
        self.assertEqual(result["arrival"]["candidateForecast"]["mean"], -1.25)

    def test_false_source_availability_never_publishes_cached_point(self):
        source = deepcopy(self.source)
        source["available"] = False
        result = self.project(source)
        for name in ("arrival", "trail"):
            self.assertFalse(result[name]["available"])
            self.assertIsNone(result[name]["point"])

    def test_previous_issue_cannot_be_relabelled(self):
        source = deepcopy(self.source)
        source["forecastClock"] = ForecastClock(self.issue - 1, self.anchor).metadata()
        result = self.project(source)
        self.assertEqual(result["arrival"]["reason"], "model_issue_clock_mismatch")
        self.assertFalse(result["arrival"]["available"])
        self.assertEqual(result["arrival"]["modelOutput"]["sourceForecastClock"]["forecastIssuedEpoch"], self.issue - 1)

    def test_wrong_point_or_mean_targets_are_unavailable(self):
        for field in ("arrivalTargetEpoch", "windowStartEpoch", "windowEndEpoch"):
            with self.subTest(field=field):
                source = deepcopy(self.source)
                source["forecastClock"][field] += 1
                self.assertFalse(self.project(source)["arrival"]["available"])
                self.assertFalse(self.project(source)["trail"]["available"])

    def test_non_bucket_and_future_source_anchor_rejected(self):
        for anchor in (self.anchor + 1, self.anchor + 900):
            with self.subTest(anchor=anchor):
                source = deepcopy(self.source)
                source["featureAnchorEpoch"] = anchor
                source["forecastClock"]["featureAnchorEpoch"] = anchor
                result = self.project(source)
                self.assertEqual(result["arrival"]["reason"], "model_feature_anchor_invalid")
                self.assertIsNone(result["arrival"]["point"])

    def test_no_additional_stale_origin_numeric_gate(self):
        source = deepcopy(self.source)
        source["featureAnchorEpoch"] -= 900
        source["forecastClock"] = ForecastClock(self.issue, self.anchor - 900).metadata()
        self.assertEqual(self.project(source)["arrival"]["point"], 157.9)

    def test_raw_source_anchor_must_match_its_clock(self):
        source = deepcopy(self.source)
        source["featureAnchorEpoch"] -= 900
        self.assertEqual(self.project(source)["arrival"]["reason"], "raw_model_feature_anchor_mismatch")

    def test_present_parent_clock_must_match_issue_and_target(self):
        air = deepcopy(self.air)
        air["forecastClock"] = ForecastClock(self.issue - 1, self.anchor).metadata()
        self.assertFalse(self.project(air=air)["arrival"]["available"])

    def test_baseline_availability_does_not_select_raw_model_output(self):
        air = deepcopy(self.air)
        air["available"] = False
        air.pop("forecastClock")
        self.assertEqual(self.project(air=air)["arrival"]["point"], 157.9)

    def test_baseline_uncertainty_cannot_be_relabelled_as_model_interval(self):
        result = self.project()
        self.assertEqual(result["arrival"]["baselineUncertainty"]["rangeLow"], 141.)
        self.assertIsNone(result["arrival"]["rangeLow"])
        self.assertIsNone(result["arrival"]["upper90"])
        self.assertIsNone(result["trail"]["projectedPeak"])
        self.assertFalse(result["arrival"]["performanceEvidence"]["prospectivelyValidated"])
        self.assertEqual(result["arrival"]["performanceEvidence"]["rawModelMaeUgM3"], 8.05)

    def test_no_input_mutation(self):
        before_air, before_source = deepcopy(self.air), deepcopy(self.source)
        result = self.project()
        result["arrival"]["modelOutput"]["sourceForecastClock"]["forecastIssuedEpoch"] = 0
        result["arrival"]["baselineReference"]["freshnessAdjustment"]["applied"] = False
        self.assertEqual(self.air, before_air)
        self.assertEqual(self.source, before_source)

    def test_all_legacy_mean_peak_bounds_are_only_baseline_metadata(self):
        air = deepcopy(self.air)
        fields = ("rawUpperMean90", "upperMean", "decisionUpperMean", "rawPeakUpper90",
                  "rawPeakRangeLow", "rawPeakRangeHigh", "rawPeak", "rawProjectedPeak",
                  "upperPeak", "upperPeak90", "empiricalPeakUpper", "decisionPeakUpper")
        original = {field: 170.123 + i for i, field in enumerate(fields)}
        air["trail"].update(original)
        result = self.project(air=air)
        for field, value in original.items():
            with self.subTest(field=field):
                self.assertEqual(result["trail"]["baselineUncertainty"][field], value)
                self.assertIsNone(result["trail"][field])
        self.assertEqual(result["trail"]["point"], 141.9)

    def test_legacy_coverage_and_promotion_tags_cannot_qualify_raw_output(self):
        air = deepcopy(self.air)
        old = {"finiteSampleRank": True, "finiteSampleUpper": True,
               "finiteSampleRankCoverage": .92345, "upperTargetCoverage": .9,
               "calibrationState": "calibrated", "confidence": "High",
               "meanSkillEligible": True, "peakSkillEligible": True,
               "deploymentSkill": {"eligible": True}, "qualification": {"operationalUseEligible": True},
               "qualificationPolicy": {"qualified": True}}
        roles = ("rangeRole", "rawRangeRole", "upperQuantileRole", "upperMeanRole",
                 "peakUpperRole", "rawPeakUpperRole", "empiricalPeakQuantileRole",
                 "upperPeakRole", "peakRole", "peakQuantileRole", "meanQuantileRole")
        old.update({role: "calibrated_persistence_quantile" for role in roles})
        air["trail"].update(deepcopy(old))
        head = self.project(air=air)["trail"]
        for field, value in old.items():
            with self.subTest(field=field):
                self.assertEqual(head["baselineUncertainty"][field], value)
        for field in ("finiteSampleRank", "finiteSampleUpper", "meanSkillEligible", "peakSkillEligible"):
            self.assertIs(head[field], False)
        for field in ("finiteSampleRankCoverage", "upperTargetCoverage"):
            self.assertIsNone(head[field])
        for field in roles:
            self.assertEqual(head[field], "raw_model_interval_unavailable")
        for field in ("deploymentSkill", "qualification", "qualificationPolicy"):
            self.assertNotIn(field, head)
        self.assertEqual(head["calibrationState"], "not_established")
        self.assertEqual(head["confidence"], "Predictive accuracy unproven")
        self.assertFalse(head["qualified"])
        self.assertEqual(head["point"], 141.9)

    def test_invalid_issue_types_are_unavailable(self):
        for issue in (True, str(self.issue), self.issue + .5, float("nan"), -1):
            with self.subTest(issue=issue):
                self.assertEqual(self.project(issue=issue)["arrival"]["reason"], "invalid_issue_epoch")

    def test_ride_extrema_publish_literal_outputs_without_containing_mean(self):
        source = deepcopy(self.source)
        source["trailExtrema"] = {"available": True, "low": 90.123456, "high": 110.987654,
            "forecastClock": deepcopy(self.clock), "validated": False, "mae": 9999}
        air = self.project()
        result = contract.attach_ride_extrema_output(air, source, self.issue)
        extrema = result["trail"]["rideExtrema"]
        self.assertEqual((extrema["low"], extrema["high"]), (90.123456, 110.987654))
        self.assertEqual(result["trail"]["point"], 141.9)
        self.assertIsNone(result["trail"]["rangeLow"])
        self.assertIsNone(result["trail"]["rawRangeHigh"])
        self.assertFalse(extrema["isMeanPredictionInterval"])
        self.assertNotIn("rideExtrema", air["trail"])

    def test_missing_ride_extrema_never_substitutes_mean_or_persistence(self):
        result = contract.attach_ride_extrema_output(self.project(), self.source, self.issue)
        extrema = result["trail"]["rideExtrema"]
        self.assertFalse(extrema["available"])
        self.assertIsNone(extrema["low"])
        self.assertIsNone(extrema["high"])
        self.assertEqual(result["trail"]["point"], 141.9)

    def test_ride_extrema_from_another_issue_are_not_relabelled(self):
        source = deepcopy(self.source)
        source["trailExtrema"] = {"available": True, "low": 90., "high": 110.,
            "forecastClock": ForecastClock(self.issue-1, self.anchor).metadata()}
        result = contract.attach_ride_extrema_output(self.project(), source, self.issue)
        self.assertFalse(result["trail"]["rideExtrema"]["available"])
        self.assertIsNone(result["trail"]["rideExtrema"]["low"])

    def test_crossed_ride_extrema_are_kept_as_raw_model_outputs(self):
        source = deepcopy(self.source)
        source["trailExtrema"] = {"available": True, "low": 180., "high": 110.,
            "crossing": True, "forecastClock": deepcopy(self.clock)}
        result = contract.attach_ride_extrema_output(self.project(), source, self.issue)
        self.assertEqual((result["trail"]["rideExtrema"]["low"],
                          result["trail"]["rideExtrema"]["high"]), (180., 110.))

    def test_partial_malformed_extrema_do_not_replace_other_forecasts(self):
        source = deepcopy(self.source)
        source["trailExtrema"] = {"available": True, "low": None, "high": 110.,
            "forecastClock": deepcopy(self.clock)}
        result = contract.attach_ride_extrema_output(self.project(), source, self.issue)
        self.assertFalse(result["trail"]["rideExtrema"]["available"])
        self.assertEqual(result["arrival"]["point"], 157.9)
        self.assertEqual(result["trail"]["point"], 141.9)


if __name__ == "__main__":
    unittest.main()
