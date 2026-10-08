"""Full-window weather coverage contracts; synthetic hourly inputs, no I/O."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
import unittest

import rlcd_api
import weather_contracts as contract

HOUR = 3600
ANCHOR = 1790002800  # exact UTC hourly boundary


def fixture():
    points = []
    for index in range(4):
        point = {"epoch": ANCHOR + index * HOUR,
                 "temperature_2m": 28 + index, "apparent_temperature": 33 + index,
                 "relative_humidity_2m": 70, "wind_speed_10m": 2 + 4 * index,
                 "wind_direction_10m": 350 if index % 2 == 0 else 10,
                 "wind_gusts_10m": 20 + 4 * index, "precipitation": 4 * index,
                 "precipitation_probability": 0, "cloud_cover": 20,
                 "pressure_msl": 1000, "showers": 2 * index,
                 "wind_speed_180m": 15, "wind_direction_180m": 180,
                 "wind_direction_925hPa": 270}
        points.append(point)
    return {"fetchedEpoch": ANCHOR - 60, "hourly": points}


def summarize(payload, start=ANCHOR, end=ANCHOR + 2 * HOUR):
    return contract.guard_summary({"available": True, "sourceFetchedEpoch": payload["fetchedEpoch"],
                                   "ventilationUsed": False, "sourceAlignment925": 0.5},
                                  payload, start, end)


class WeatherCoverageTests(unittest.TestCase):
    def test_complete_aligned_window_preserves_hourly_semantics_and_true_dry_zero(self):
        result = summarize(fixture())
        self.assertTrue(result["available"])
        self.assertTrue(result["weatherCoverage"]["complete"])
        self.assertEqual(result["precipitationMm"], 12)
        self.assertEqual(result["showersMm"], 6)
        self.assertEqual(result["precipitationProbabilityMax"], 0)
        self.assertEqual(result["temperatureMax"], 29)
        self.assertEqual(result["windSpeed10mMean"], 4)
        self.assertEqual(result["windGust10mMax"], 28)
        self.assertIn(result["windDirection10m"], (0.0, 360.0))

    def test_shifted_window_covers_first_and_last_fractional_blocks(self):
        result = summarize(fixture(), ANCHOR + 900, ANCHOR + 8100)
        self.assertTrue(result["available"])
        self.assertEqual(result["precipitationMm"], 14)
        self.assertEqual(result["windSpeed10mMean"], 5)
        self.assertEqual(result["temperatureMax"], 30)
        self.assertEqual(result["windGust10mMax"], 32)
        for field in contract.REQUIRED_FIELDS:
            self.assertEqual(result["weatherCoverage"]["fields"][field]["coveredHours"], 2)

    def test_missing_rain_endpoint_is_unknown_not_dry(self):
        payload = fixture()
        payload["hourly"][2]["precipitation"] = None
        result = summarize(payload)
        self.assertFalse(result["available"])
        self.assertIsNone(result["precipitationMm"])
        self.assertEqual(result["precipitationProbabilityMax"], 0)
        self.assertEqual(result["rainLabel"], "Rain forecast unavailable")
        self.assertEqual(result["weatherCoverage"]["fields"]["precipitation"]["missingHours"], 1)

    def test_missing_one_instantaneous_meteorology_hour_is_not_a_partial_mean(self):
        payload = fixture()
        del payload["hourly"][0]["wind_speed_10m"]
        result = summarize(payload)
        self.assertFalse(result["available"])
        self.assertIsNone(result["windSpeed10mMean"])
        self.assertEqual(result["temperatureMax"], 29)
        self.assertEqual(result["weatherCoverage"]["fields"]["wind_speed_10m"]["missingOrInvalidEpochs"], [ANCHOR])

    def test_optional_incomplete_fields_are_null_without_erasing_proven_core_fields(self):
        payload = fixture()
        del payload["hourly"][1]["wind_speed_180m"]
        result = summarize(payload)
        self.assertTrue(result["available"])
        self.assertIsNone(result["windSpeed180mMean"])
        self.assertEqual(result["ventilationState"], "unknown")
        self.assertEqual(result["ventilationLabel"], "Airflow forecast unavailable")
        self.assertFalse(result["weatherCoverage"]["fields"]["wind_speed_180m"]["complete"])

    def test_invalid_probability_boolean_nan_and_conflicting_duplicates_are_rejected(self):
        for bad in (True, float("nan"), 101, -1):
            payload = fixture()
            payload["hourly"][2]["precipitation_probability"] = bad
            result = summarize(payload)
            self.assertFalse(result["available"])
            self.assertIsNone(result["precipitationProbabilityMax"])
        payload = fixture()
        duplicate = dict(payload["hourly"][2], precipitation=999)
        payload["hourly"].append(duplicate)
        self.assertIsNone(summarize(payload)["precipitationMm"])

    def test_invalid_empty_window_and_no_mutation(self):
        payload = fixture()
        before = deepcopy(payload)
        base = {"available": True, "precipitationMm": 999}
        result = contract.guard_summary(base, payload, ANCHOR, ANCHOR)
        self.assertFalse(result["available"])
        self.assertIsNone(result["precipitationMm"])
        self.assertEqual(base, {"available": True, "precipitationMm": 999})
        self.assertEqual(payload, before)


class RlcdWeatherCoverageTests(unittest.TestCase):
    def project(self, result, now=ANCHOR + 20, issued=ANCHOR):
        return rlcd_api.window_weather(result, now, issued, ANCHOR, ANCHOR + 2 * HOUR)

    def test_complete_proven_window_remains_available_and_fresh(self):
        result = self.project(summarize(fixture()))
        self.assertTrue(result["available"])
        self.assertTrue(result["fresh"])
        self.assertTrue(result["coverage_verified"])
        self.assertEqual(result["rain_mm"], 12)
        self.assertEqual(result["rain_chance_max_pct"], 0)
        self.assertIsNone(result["unavailable_reason"])

    def test_partial_and_legacy_summary_are_explicitly_unavailable(self):
        payload = fixture()
        payload["hourly"].pop(2)
        partial = self.project(summarize(payload))
        self.assertFalse(partial["available"])
        self.assertFalse(partial["fresh"])
        self.assertIsNone(partial["rain_mm"])
        legacy = summarize(fixture())
        del legacy["weatherCoverage"]
        old = self.project(legacy)
        self.assertFalse(old["available"])
        self.assertEqual(old["unavailable_reason"], "weather_coverage_unverified_legacy_summary")
        self.assertIsNone(old["rain_mm"])

    def test_source_age_and_future_fetch_use_response_wall_clock(self):
        complete = summarize(fixture())
        stale = self.project(complete, now=ANCHOR + 24 * HOUR)
        self.assertFalse(stale["available"])
        self.assertEqual(stale["age_s"], 24 * HOUR + 60)
        complete["sourceFetchedEpoch"] = ANCHOR + 1
        self.assertFalse(self.project(complete)["available"])

    def test_coverage_proof_for_another_target_or_missing_aggregate_is_rejected(self):
        complete = summarize(fixture())
        complete["weatherCoverage"]["endEpoch"] += HOUR
        self.assertFalse(self.project(complete)["available"])
        complete = summarize(fixture())
        complete["temperatureMax"] = None
        self.assertFalse(self.project(complete)["available"])


class RlcdEventQualificationTests(unittest.TestCase):
    def payload(self, qualification=None):
        event = {"available": True, "forecastIssuedEpoch": ANCHOR,
                 "validUntilEpoch": ANCHOR+5400, "changeThresholdUgM3": 20,
                 "probabilityRise": .8, "probabilityDrop": .1, "probabilityNoCrossing": .1,
                 "direction": "rise", "diagnosticDirection": "rise", "referencePm": 100}
        if qualification is not None:
            event["qualification"] = qualification
        analysis = {"available": True, "forecastIssuedEpoch": ANCHOR,
                    "airWindow": {"forecastClock": {"forecastIssuedEpoch": ANCHOR,
                        "arrivalTargetEpoch": ANCHOR+5400, "windowStartEpoch": ANCHOR+5400,
                        "windowEndEpoch": ANCHOR+12600},
                        "arrival": {"available": True, "forecastIssuedEpoch": ANCHOR,
                            "expectedEpoch": ANCHOR+5400, "point": 100, "pointRole": "persistence_anchor"},
                        "firstCrossingEventForecast": event}}
        result = rlcd_api.build_payload({"epoch": ANCHOR-30, "pm02": 100}, analysis, {}, ANCHOR+20)
        return result, analysis

    def test_legacy_and_unqualified_event_direction_is_diagnostic_only(self):
        for proof in (None, {}, {"operationalUseEligible": False}, {"operationalUseEligible": 1}):
            with self.subTest(proof=proof):
                result, original = self.payload(proof)
                event = result["forecast"]["near90"]["first20"]
                self.assertTrue(event["available"])
                self.assertEqual(event["direction"], "unresolved")
                self.assertEqual(event["diagnostic_direction"], "rise")
                self.assertEqual(event["rise_probability"], .8)
                self.assertFalse(event["qualification"]["operational_use_eligible"])
                self.assertEqual(original["airWindow"]["firstCrossingEventForecast"]["direction"], "rise")

    def test_explicit_operational_proof_allows_direction_without_changing_numeric_point(self):
        result, _ = self.payload({"operationalUseEligible": True, "actualCadenceValidated": True})
        near = result["forecast"]["near90"]
        self.assertEqual(near["pm25_ugm3"], 100)
        self.assertEqual(near["first20"]["direction"], "rise")
        self.assertTrue(near["first20"]["qualification"]["operational_use_eligible"])

    def test_event_projection_stays_within_device_budget(self):
        import json
        result, _ = self.payload()
        self.assertLessEqual(len(json.dumps(result, separators=(",", ":")).encode("utf-8")), 8192)

    def test_device_history_cap_preserves_observed_extrema_endpoints_gaps_and_summary(self):
        import rlcd_history
        rows = [{"epoch": ANCHOR-21600+index*180,
                 "pm02": 0 if index % 2 == 0 else 300,
                 "atmp": 30.5, "heatindex": 40.5}
                for index in range(121) if index not in (10, 11, 12, 13)]
        original = rlcd_history.build_history(rows, ANCHOR)
        device = rlcd_history.build_history(rows, ANCHOR, max_points=64)
        self.assertLessEqual(len(device["points"]), 64)
        self.assertEqual(device["pm25_summary"], original["pm25_summary"])
        self.assertEqual(device["source_rows"], original["source_rows"])
        self.assertEqual(device["gaps"], original["gaps"])
        self.assertEqual(device["points"][0], original["points"][0])
        self.assertEqual(device["points"][-1], original["points"][-1])
        self.assertEqual(min(p[1] for p in device["points"]), 0)
        self.assertEqual(max(p[1] for p in device["points"]), 300)

    def test_byte_limit_thins_only_device_graph_and_keeps_summary_metadata(self):
        import json
        import rlcd_history
        payload, _ = self.payload()
        rows = [{"epoch": ANCHOR-21600+index*180,
                 "pm02": 10.1 if index % 2 == 0 else 300.1,
                 "atmp": 30.5, "heatindex": 40.5} for index in range(121)]
        payload["history"] = rlcd_history.build_history(rows, ANCHOR)
        # The saved full real response exceeds 8KiB. Reserve comparable bytes
        # for weather/model metadata here without depending on a live file.
        size = len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        payload["dashboard_build"] = "x" * max(0, 8400-size)
        before = deepcopy(payload)
        bounded = rlcd_api.enforce_payload_budget(payload)
        self.assertLessEqual(len(json.dumps(bounded, ensure_ascii=True, separators=(",", ":")).encode("utf-8")), 8192)
        self.assertLess(len(bounded["history"]["points"]), len(before["history"]["points"]))
        self.assertEqual(bounded["history"]["pm25_summary"], before["history"]["pm25_summary"])
        self.assertEqual(bounded["history"]["gaps"], before["history"]["gaps"])
        self.assertEqual(bounded["forecast"], before["forecast"])


if __name__ == "__main__":
    unittest.main()
