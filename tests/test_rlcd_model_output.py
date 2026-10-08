"""Device contracts for direct numeric model outputs; synthetic inputs only."""
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

from copy import deepcopy
import json
import unittest

import rlcd_api
import rlcd_history
import weather_contracts
import model_output_contracts
import session_model_output
from forecast_clock import ForecastClock
from test_weather_contracts import ANCHOR, HOUR, fixture


def source(point, model, role="raw_model_output"):
    return {"available": point is not None, "point": point, "baselinePoint": 160.1,
            "pointRole": role, "forecastState": "model_output", "modelVersion": model,
            "validated": False, "qualified": False, "usedForDecision": False,
            "modelSelection": {"selectedPolicy": "same_issue_persistence", "skillGatePassed": False},
            "qualificationPolicy": {"qualified": False}, "rawRangeLow": None, "rawRangeHigh": None}


def inputs():
    payload = fixture()
    template = deepcopy(payload["hourly"])
    payload["hourly"] = [{**deepcopy(template[index % len(template)]), "epoch": ANCHOR+index*HOUR}
                         for index in range(25)]
    def weather(start, end):
        return weather_contracts.guard_summary({"sourceFetchedEpoch": payload["fetchedEpoch"]},
                                               payload, start, end)
    near = {**source(150.123456, "fixed_near_raw", "experimental_model_output"),
            "forecastIssuedEpoch": ANCHOR, "expectedEpoch": ANCHOR+5400}
    ride = {**source(141.987654, "fixed_ride_raw"), "forecastIssuedEpoch": ANCHOR,
            "startEpoch": ANCHOR+5400, "endEpoch": ANCHOR+12600}
    windows = {}
    for name, lead, value, model in (("morning", 12*HOUR, 88.345678, "fixed_morning_hgb"),
                                     ("afternoon", 17*HOUR, 99.765432, "fixed_afternoon_patchtst")):
        start, end = ANCHOR+lead, ANCHOR+lead+2*HOUR
        windows[name] = {"startEpoch": start, "endEpoch": end, "forecastIssuedEpoch": ANCHOR,
                         "particleForecast": source(value, model), "weatherForecast": weather(start, end)}
    analysis = {"available": True, "forecastIssuedEpoch": ANCHOR, "delivery": {"state": "ready"},
                "airWindow": {"forecastClock": {"forecastIssuedEpoch": ANCHOR,
                    "arrivalTargetEpoch": ANCHOR+5400, "windowStartEpoch": ANCHOR+5400,
                    "windowEndEpoch": ANCHOR+12600}, "arrival": near, "trail": ride,
                    "firstCrossingEventForecast": {"available": True, "forecastIssuedEpoch": ANCHOR,
                        "validUntilEpoch": ANCHOR+5400, "changeThresholdUgM3": 20,
                        "probabilityRise": .8, "probabilityDrop": .1, "probabilityNoCrossing": .1,
                        "direction": "rise", "diagnosticDirection": "rise", "referencePm": 160.1,
                        "qualification": {"operationalUseEligible": False}}},
                "windows": windows, "weather": {"trail": weather(ANCHOR+5400, ANCHOR+12600)}}
    reading = {"epoch": ANCHOR-30, "pm02": 160.1, "atmp": 30.5, "heatindex": 40.5, "rhum": 70}
    rows = [{"epoch": ANCHOR-21600+index*180, "pm02": 100.1+index/3,
             "atmp": 30.5, "heatindex": 40.5} for index in range(121)]
    return reading, analysis, payload, rows


def projected_heads(result):
    forecast = result["forecast"]
    return [forecast["near90"], forecast["ride90_210"],
            forecast["sessions"]["morning"]["pm"], forecast["sessions"]["afternoon"]["pm"]]


class DirectModelOutputTests(unittest.TestCase):
    def project(self, values=None):
        reading, analysis, weather, rows = inputs() if values is None else values
        result = rlcd_api.build_payload(reading, analysis, weather, ANCHOR+20, history_rows=rows)
        return result, (reading, analysis, weather, rows)

    def test_all_four_heads_equal_direct_source_points_without_qualification_or_rounding(self):
        result, (_, original, _, _) = self.project()
        sources = [original["airWindow"]["arrival"], original["airWindow"]["trail"],
                   original["windows"]["morning"]["particleForecast"],
                   original["windows"]["afternoon"]["particleForecast"]]
        for actual, expected in zip(projected_heads(result), sources):
            self.assertTrue(actual["available"])
            self.assertEqual(actual["pm25_ugm3"], expected["point"])
            self.assertEqual(actual["reference_pm25_ugm3"], expected["baselinePoint"])
            self.assertEqual(actual["role"], expected["pointRole"])
            self.assertTrue(actual["experimental"])
            self.assertFalse(actual["validated"])
            self.assertFalse(actual["qualified"])
            self.assertFalse(actual["used_for_decision"])
            self.assertEqual(actual["performance_status"], "experimental_unvalidated")
            self.assertEqual(actual["range_kind"], "unavailable")

    def test_event_direction_qualification_does_not_gate_or_move_numeric_model_output(self):
        values = inputs()
        for eligible in (False, True):
            with self.subTest(eligible=eligible):
                values[1]["airWindow"]["firstCrossingEventForecast"]["qualification"]["operationalUseEligible"] = eligible
                result, _ = self.project(values)
                near = result["forecast"]["near90"]
                self.assertEqual(near["pm25_ugm3"], 150.123456)
                self.assertEqual(near["first20"]["direction"], "rise" if eligible else "unresolved")

    def test_no_model_inputs_return_null_model_value_with_separate_reference(self):
        values = inputs()
        values[1]["windows"]["afternoon"]["particleForecast"] = source(None, "fixed_afternoon_patchtst")
        result, _ = self.project(values)
        afternoon = result["forecast"]["sessions"]["afternoon"]["pm"]
        self.assertFalse(afternoon["available"])
        self.assertIsNone(afternoon["pm25_ugm3"])
        self.assertEqual(afternoon["reference_pm25_ugm3"], 160.1)
        self.assertTrue(afternoon["experimental"])

    def test_validation_metadata_can_change_without_selecting_another_numeric_point(self):
        values = inputs()
        near = values[1]["airWindow"]["arrival"]
        near.update(validated=True, qualified=True, usedForDecision=True, performanceStatus="independent_issued_evidence")
        result, _ = self.project(values)
        self.assertEqual(result["forecast"]["near90"]["pm25_ugm3"], near["point"])
        self.assertEqual(result["forecast"]["near90"]["performance_status"], "independent_issued_evidence")

    def test_real_model_equality_to_reference_keeps_model_role(self):
        values = inputs()
        values[1]["airWindow"]["arrival"]["point"] = 160.1
        result, _ = self.project(values)
        self.assertEqual(result["forecast"]["near90"]["pm25_ugm3"], 160.1)
        self.assertEqual(result["forecast"]["near90"]["role"], "experimental_model_output")

    def test_actual_near_output_contract_preserves_raw_heads_through_device_projection(self):
        values = inputs()
        clock = ForecastClock(ANCHOR, ANCHOR).metadata()
        values[1]["airWindow"]["forecastClock"] = clock
        raw = {"enabled": True, "available": True, "status": "replay_screened_experimental",
               "forecastClock": clock, "featureAnchor": 157.0, "featureAnchorEpoch": ANCHOR,
               "arrivalPoint": 156.87654321, "trailMeanPoint": 142.23456789,
               "arrivalDelta": 999.0, "trailMeanDelta": -999.0,
               "backtest": {"rawAnalogueArrivalMae": 9999.0, "arrivalPersistenceMae": 0.0,
                            "deploymentSkill": {"arrival": {"eligible": False}}}}
        values[1]["airWindow"] = model_output_contracts.attach_near_model_outputs(
            values[1]["airWindow"], raw, ANCHOR)
        result, _ = self.project(values)
        for name, key in (("near90", "arrivalPoint"), ("ride90_210", "trailMeanPoint")):
            actual = result["forecast"][name]
            self.assertTrue(actual["available"])
            self.assertEqual(actual["pm25_ugm3"], raw[key])
            self.assertEqual(actual["reference_pm25_ugm3"], 160.1)
            self.assertFalse(actual["used_for_decision"])
            self.assertEqual(actual["range_kind"], "unavailable")

    def test_actual_fixed_session_contract_cannot_be_replaced_by_old_selection_metadata(self):
        values = inputs()
        for name, raw in (("morning", 112.87654321), ("afternoon", 113.13036727905273)):
            session = values[1]["windows"][name]
            estimate = session_model_output.publish_direct_output({
                "available": True, "mean": raw, "sensorAnchor": 160.1,
                "forecastedAtEpoch": ANCHOR, "startEpoch": session["startEpoch"],
                "endEpoch": session["endEpoch"], "modelVersion": "fixed_"+name,
                "modelSelection": {"selectedPolicy": "same_issue_persistence"},
                "qualificationPolicy": {"qualified": False}}, name)
            session["particleForecast"] = {**estimate, "point": estimate["mean"],
                                          "baselinePoint": estimate["sensorAnchor"]}
        result, _ = self.project(values)
        for name, raw in (("morning", 112.87654321), ("afternoon", 113.13036727905273)):
            actual = result["forecast"]["sessions"][name]["pm"]
            self.assertTrue(actual["available"])
            self.assertEqual(actual["pm25_ugm3"], raw)
            self.assertEqual(actual["reference_pm25_ugm3"], 160.1)
            self.assertFalse(actual["used_for_decision"])
            self.assertEqual(actual["role"], "experimental_model_output")

    def test_coach_and_device_preserve_finite_unqualified_model_number_and_metadata(self):
        # A finite estimator output must not be clipped or replaced at either
        # publication adapter, even when its performance flags are false.
        import BukitKiara_Dashboard as dashboard
        value = {**source(-5.23456789, "fixed_direct_model"),
                 "numericalPolicyIdentifier": "fixed_direct_numeric_policy",
                 "modelOutputPolicy": {"fixedModel": True, "numericalSelectionApplied": False},
                 "performanceEvidence": {"prospectivelyValidated": False, "appliedToNumber": False}}
        coach = dashboard.compact_window_particle_forecast(value)
        device = rlcd_api.projection(value, True)
        self.assertEqual(coach["publishedMeanPm25UgM3"], value["point"])
        self.assertEqual(coach["projectedMeanPm25UgM3"], value["point"])
        self.assertEqual(coach["planningEstimate"]["meanPm25UgM3"], value["point"])
        self.assertEqual(device["pm25_ugm3"], value["point"])
        self.assertTrue(coach["modelPointApplied"])
        self.assertFalse(coach["usedForDecision"])
        self.assertFalse(coach["validated"])
        self.assertEqual(coach["modelOutputPolicy"], value["modelOutputPolicy"])
        self.assertEqual(coach["performanceEvidence"], value["performanceEvidence"])
        self.assertEqual(coach["numericalPolicyIdentifier"], value["numericalPolicyIdentifier"])

    def test_direct_model_payload_keeps_weather_proof_history_summary_and_byte_budget(self):
        result, (_, original, _, rows) = self.project()
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=True, allow_nan=False,
                                           separators=(",", ":")).encode()), 16384)
        forecast = result["forecast"]
        for weather in (forecast["ride90_210"]["weather"],
                        forecast["sessions"]["morning"]["weather"], forecast["sessions"]["afternoon"]["weather"]):
            self.assertTrue(weather["available"])
            self.assertTrue(weather["coverage_verified"])
        full_history = rlcd_history.build_history(rows, ANCHOR+20)
        self.assertEqual(result["history"]["pm25_summary"], full_history["pm25_summary"])
        self.assertEqual(result["history"]["gaps"], full_history["gaps"])
        self.assertEqual(result["history"]["points"][0], full_history["points"][0])
        self.assertEqual(result["history"]["points"][-1], full_history["points"][-1])
        self.assertEqual(forecast["sessions"]["afternoon"]["pm"]["pm25_ugm3"],
                         original["windows"]["afternoon"]["particleForecast"]["point"])

    def test_device_extrema_are_separate_literal_model_values(self):
        values = inputs()
        values[1]["airWindow"]["trail"]["rideExtrema"] = {
            "available": True, "low": 122.123456, "high": 163.654321,
            "forecastClock": ForecastClock(ANCHOR, ANCHOR).metadata(),
            "modelVersion": "local_extra_trees_joint_extrema90_210_v1"}
        result, _ = self.project(values)
        ride = result["forecast"]["ride90_210"]
        self.assertEqual(ride["minimum_maximum"]["minimum_ugm3"], 122.123456)
        self.assertEqual(ride["minimum_maximum"]["maximum_ugm3"], 163.654321)
        self.assertEqual(ride["pm25_ugm3"], 141.987654)
        self.assertEqual(ride["range_kind"], "unavailable")


if __name__ == "__main__":
    unittest.main()
