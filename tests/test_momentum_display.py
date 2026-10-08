"""Shared momentum copy contracts; synthetic cached forecasts only.

The display formatter summarizes the unchanged three-class output. Its text
must not turn an unqualified diagnostic into an operational direction, mutate
the source, fit a model, or bypass the device's existing clock guards.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import momentum_display
import rlcd_api
import rlcd_history
from test_rlcd_model_output import inputs
from test_weather_contracts import ANCHOR


def event(rise=.008, drop=.14, none=.852, *, qualified=False):
    probabilities = (rise, drop, none)
    largest = max(probabilities)
    unique = probabilities.count(largest) == 1
    diagnostic = ("rise" if unique and rise == largest else
                  "drop" if unique and drop == largest else "unresolved")
    return {
        "available": True, "probabilityRise": rise,
        "probabilityDrop": drop, "probabilityNoCrossing": none,
        "diagnosticDirection": diagnostic,
        "direction": diagnostic if qualified else "unresolved",
        "directionDecision": "three_class_argmax_ties_unresolved",
        "changeThresholdUgM3": 20, "referencePm": 160.1,
        "forecastIssuedEpoch": ANCHOR, "validUntilEpoch": ANCHOR + 5400,
        "modelVersion": "synthetic_cached_event_model",
        "qualification": {"operationalUseEligible": qualified,
                          "actualCadenceValidated": qualified,
                          "calibrationValidated": qualified},
    }


class SharedMomentumFormatterTests(unittest.TestCase):
    def test_three_model_outcomes_and_ties(self):
        cases = (
            (event(), "No ≥20 µg/m³ change: 85.2%"),
            (event(.8, .1, .1), "80 % ≥20 µg/m³ rise"),
            (event(.1, .8, .1), "80 % ≥20 µg/m³ fall"),
            (event(.4, .35, .25), "40 % ≥20 µg/m³ rise"),
            (event(.35, .4, .25), "40 % ≥20 µg/m³ fall"),
            (event(.4, .4, .2), "Momentum outcomes tied"),
            (event(.5, 0, .5), "Momentum outcomes tied"),
            (event(0, .5, .5), "Momentum outcomes tied"),
            (event(1 / 3, 1 / 3, 1 / 3), "Momentum outcomes tied"),
            (event(1, 0, 0), "100 % ≥20 µg/m³ rise"),
        )
        for source, expected in cases:
            with self.subTest(probabilities=(source["probabilityRise"],
                                            source["probabilityDrop"],
                                            source["probabilityNoCrossing"])):
                self.assertEqual(momentum_display.display_text(source), expected)

    def test_display_uses_diagnostic_without_changing_operational_qualification(self):
        for qualified in (False, True):
            for rise, drop, none, expected in (
                    (.8, .1, .1, "80 % ≥20 µg/m³ rise"),
                    (.1, .8, .1, "80 % ≥20 µg/m³ fall"),
                    (.008, .14, .852, "No ≥20 µg/m³ change: 85.2%")):
                with self.subTest(qualified=qualified, expected=expected):
                    source = event(rise, drop, none, qualified=qualified)
                    before = deepcopy(source)
                    self.assertEqual(momentum_display.display_text(source), expected)
                    self.assertEqual(source, before)

    def test_close_split_no_change_includes_its_probability_without_overriding_fall(self):
        for qualified in (False, True):
            source = event(.046, .431, .523, qualified=qualified)
            before = deepcopy(source)
            self.assertEqual(momentum_display.display_text(source),
                             "No ≥20 µg/m³ change: 52.3%")
            attached = momentum_display.attach_display_text(source)
            self.assertEqual(attached["displayText"], "No ≥20 µg/m³ change: 52.3%")
            self.assertEqual(source, before)
            self.assertEqual({key: value for key, value in attached.items() if key != "displayText"},
                             before)
        # A probability lead need not exceed 50%; no new display gate is added.
        self.assertEqual(momentum_display.display_text(event(.25, .35, .4)),
                         "No ≥20 µg/m³ change: 40%")

    def test_display_rounding_matches_javascript_and_removes_trailing_point_zero(self):
        # The dashboard formats Number(p*100).toFixed(1), then strips .0.
        # 80.05's binary representation lies below the decimal half, whereas
        # 80.25 is an exact binary half and JavaScript rounds it upward.
        for rise, expected in ((.8005, "80 % ≥20 µg/m³ rise"),
                               (.7955, "79.5 % ≥20 µg/m³ rise"),
                               (.8025, "80.3 % ≥20 µg/m³ rise"),
                               (.8125, "81.3 % ≥20 µg/m³ rise"),
                               (.8725, "87.3 % ≥20 µg/m³ rise"),
                               (.80149, "80.1 % ≥20 µg/m³ rise"),
                               (.80151, "80.2 % ≥20 µg/m³ rise"),
                               (.80, "80 % ≥20 µg/m³ rise")):
            with self.subTest(rise=rise):
                self.assertEqual(momentum_display.display_text(event(rise, .1, .9-rise)), expected)

    def test_valid_legacy_metadata_is_optional_and_direction_fallback_is_checked(self):
        source = event(.8, .1, .1, qualified=True)
        source.pop("diagnosticDirection")
        source.pop("directionDecision")
        self.assertEqual(momentum_display.display_text(source), "80 % ≥20 µg/m³ rise")
        source.pop("direction")
        self.assertEqual(momentum_display.display_text(source), "80 % ≥20 µg/m³ rise")
        source.update(diagnosticDirection=None, direction="rise")
        self.assertEqual(momentum_display.display_text(source), "80 % ≥20 µg/m³ rise")
        source["direction"] = "drop"
        self.assertIsNone(momentum_display.display_text(source))

    def test_unavailable_and_malformed_probabilities_do_not_invent_text(self):
        for value in (None, {}, [], "event"):
            with self.subTest(source=value):
                self.assertIsNone(momentum_display.display_text(value))
        for available in (False, None, 0, 1, "true"):
            source = event()
            source["available"] = available
            with self.subTest(available=available):
                self.assertIsNone(momentum_display.display_text(source))
        for key in ("probabilityRise", "probabilityDrop", "probabilityNoCrossing"):
            for value in (None, True, False, "0.008", float("nan"),
                          float("inf"), float("-inf"), -.001, 1.001, 10**400):
                source = event()
                source[key] = value
                with self.subTest(key=key, value=value):
                    self.assertIsNone(momentum_display.display_text(source))
            source = event()
            source.pop(key)
            with self.subTest(missing=key):
                self.assertIsNone(momentum_display.display_text(source))

    def test_probability_mass_and_declared_model_policy_are_checked(self):
        valid = event(.8, .1, .1 + 5e-9)
        self.assertEqual(momentum_display.display_text(valid), "80 % ≥20 µg/m³ rise")
        for value in (.1 + 2e-8, .2, 0):
            source = event(.8, .1, value)
            with self.subTest(none=value):
                self.assertIsNone(momentum_display.display_text(source))
        for key, value in (("diagnosticDirection", "drop"),
                           ("diagnosticDirection", "unknown"),
                           ("diagnosticDirection", None),
                           ("directionDecision", "different_rule"),
                           ("directionDecision", None)):
            source = event(.8, .1, .1)
            source[key] = value
            with self.subTest(key=key, value=value):
                self.assertIsNone(momentum_display.display_text(source))

    def test_optional_reference_and_threshold_cannot_contradict_the_display(self):
        for value in (None, True, False, "160.1", float("nan"),
                      float("inf"), float("-inf"), -.001, 10**400):
            source = event()
            source["referencePm"] = value
            with self.subTest(reference=value):
                self.assertIsNone(momentum_display.display_text(source))
        for value in (None, True, "20", 10, 20.1):
            source = event()
            source["changeThresholdUgM3"] = value
            with self.subTest(threshold=value):
                self.assertIsNone(momentum_display.display_text(source))
        source = event()
        source.pop("referencePm")
        source.pop("changeThresholdUgM3")
        self.assertEqual(momentum_display.display_text(source), "No ≥20 µg/m³ change: 85.2%")

    def test_attachment_is_shallow_and_preserves_every_existing_model_field(self):
        source = event(.1, .8, .1)
        source["displayText"] = "Old generated firmware label"
        before = deepcopy(source)
        attached = momentum_display.attach_display_text(source)
        self.assertIsNot(attached, source)
        self.assertIs(attached["qualification"], source["qualification"])
        self.assertEqual(attached["displayText"], "80 % ≥20 µg/m³ fall")
        self.assertEqual({k: v for k, v in attached.items() if k != "displayText"},
                         {k: v for k, v in before.items() if k != "displayText"})
        self.assertEqual(source, before)
        unavailable = {**source, "available": False}
        self.assertIsNone(momentum_display.attach_display_text(unavailable)["displayText"])


class MomentumDeviceProjectionTests(unittest.TestCase):
    def project(self, source=None, *, now=ANCHOR+20, change=None):
        reading, analysis, weather, rows = inputs()
        analysis["airWindow"]["firstCrossingEventForecast"] = event() if source is None else source
        if change:
            change(reading, analysis)
        original = deepcopy((reading, analysis, weather, rows))
        result = rlcd_api.build_payload(reading, analysis, weather, now,
                                        dashboard_build="synthetic_display_test", history_rows=rows)
        self.assertEqual((reading, analysis, weather, rows), original)
        return result, original

    def test_api_uses_shared_server_text_and_keeps_original_probability_semantics(self):
        for qualified in (False, True):
            for source in (event(), event(.046, .431, .523, qualified=qualified),
                           event(.8, .1, .1, qualified=qualified),
                           event(.1, .8, .1, qualified=qualified), event(.4, .4, .2)):
                with self.subTest(source=source, qualified=qualified):
                    result, _ = self.project(source)
                    actual = result["forecast"]["near90"]["first20"]
                    self.assertTrue(actual["available"])
                    self.assertEqual(actual["display_text"], momentum_display.display_text(source))
                    self.assertEqual(actual["rise_probability"], source["probabilityRise"])
                    self.assertEqual(actual["drop_probability"], source["probabilityDrop"])
                    self.assertEqual(actual["none_probability"], source["probabilityNoCrossing"])
                    eligible = source["qualification"]["operationalUseEligible"]
                    self.assertEqual(actual["direction"], source["direction"] if eligible else "unresolved")
                    self.assertEqual(actual["diagnostic_direction"], source["diagnosticDirection"])
                    self.assertEqual(actual["reference_ugm3"], source["referencePm"])
                    self.assertEqual(actual["model"], source["modelVersion"])
                    self.assertEqual(actual["qualification"]["operational_use_eligible"], eligible)
                    self.assertFalse(actual["calibrated"])

    def test_display_is_authoritative_and_does_not_trust_a_forged_source_string(self):
        source = event(.1, .8, .1)
        source["displayText"] = "No large change expected"
        result, _ = self.project(source)
        self.assertEqual(result["forecast"]["near90"]["first20"]["display_text"],
                         "80 % ≥20 µg/m³ fall")

    def test_existing_clock_and_availability_guards_always_suppress_text(self):
        cases = (
            ("unavailable event", lambda r, a: a["airWindow"]["firstCrossingEventForecast"].update(available=False)),
            ("future parent issue", lambda r, a: a.update(forecastIssuedEpoch=ANCHOR+100)),
            ("expired delivery", lambda r, a: a["delivery"].update(state="expired")),
            ("unavailable parent", lambda r, a: a.update(available=False)),
            ("stale sensor", lambda r, a: r.update(epoch=ANCHOR-901)),
            ("future sensor", lambda r, a: r.update(epoch=ANCHOR+100)),
            ("event issue mismatch", lambda r, a: a["airWindow"]["firstCrossingEventForecast"].update(forecastIssuedEpoch=ANCHOR-1)),
            ("event expiry mismatch", lambda r, a: a["airWindow"]["firstCrossingEventForecast"].update(validUntilEpoch=ANCHOR+5401)),
            ("threshold mismatch", lambda r, a: a["airWindow"]["firstCrossingEventForecast"].update(changeThresholdUgM3=10)),
            ("clock target mismatch", lambda r, a: a["airWindow"]["forecastClock"].update(arrivalTargetEpoch=ANCHOR+5401)),
        )
        for name, mutation in cases:
            with self.subTest(case=name):
                result, _ = self.project(change=mutation)
                actual = result["forecast"]["near90"]["first20"]
                self.assertFalse(actual["available"])
                self.assertIsNone(actual["display_text"])
        result, _ = self.project(now=ANCHOR+601)
        self.assertFalse(result["forecast"]["near90"]["first20"]["available"])
        self.assertIsNone(result["forecast"]["near90"]["first20"]["display_text"])

    def test_formatter_rejection_does_not_change_the_existing_numeric_api_contract(self):
        # The device's legacy event probability tolerance is wider than the
        # display formatter's web-compatible tolerance. Reject text alone.
        for source in (event(.8, .1, .10000002),
                       {**event(.8, .1, .1), "diagnosticDirection": "drop"},
                       {**event(.8, .1, .1), "directionDecision": "different_rule"}):
            with self.subTest(source=source):
                result, _ = self.project(source)
                actual = result["forecast"]["near90"]["first20"]
                self.assertTrue(actual["available"])
                self.assertIsNone(actual["display_text"])
                self.assertEqual(actual["rise_probability"], source["probabilityRise"])
                self.assertEqual(actual["drop_probability"], source["probabilityDrop"])
                self.assertEqual(actual["none_probability"], source["probabilityNoCrossing"])

    def test_payload_stays_within_budget_and_history_statistics_survive(self):
        result, (_, analysis, _, rows) = self.project(event(.8005, .1, .0995))
        encoded = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        self.assertLessEqual(len(encoded), rlcd_api.MAX_PAYLOAD_BYTES)
        expected = rlcd_history.build_history(rows, ANCHOR+20)
        for key in ("pm25_summary", "gaps", "start_epoch", "end_epoch"):
            self.assertEqual(result["history"][key], expected[key])
        self.assertEqual(result["history"]["points"][0], expected["points"][0])
        self.assertEqual(result["history"]["points"][-1], expected["points"][-1])
        self.assertEqual(result["forecast"]["near90"]["pm25_ugm3"],
                         analysis["airWindow"]["arrival"]["point"])
        self.assertEqual(result["forecast"]["ride90_210"]["pm25_ugm3"],
                         analysis["airWindow"]["trail"]["point"])
        self.assertEqual(result["forecast"]["sessions"]["afternoon"]["pm"]["pm25_ugm3"],
                         analysis["windows"]["afternoon"]["particleForecast"]["point"])


class MomentumPublicationAttachmentTests(unittest.TestCase):
    def test_publication_wrapper_adds_text_without_fitting_or_changing_cached_model_output(self):
        import BukitKiara_Dashboard as dashboard
        air = {"arrival": {"available": True, "freshnessAdjustment": {"applied": True}}}
        rows = [{"epoch": ANCHOR, "pm02": 160.1}]
        source = event(.1, .8, .1)
        before = deepcopy(source)
        with patch.object(dashboard.forecast_freshness, "references", return_value=([160.1], None, None)), \
                patch.object(dashboard.fresh_event_model, "predict_event", return_value=source):
            result = dashboard.attach_first_crossing_event_forecast(air, rows, ANCHOR)
        actual = result["firstCrossingEventForecast"]
        self.assertEqual(actual["displayText"], "80 % ≥20 µg/m³ fall")
        self.assertEqual({k: v for k, v in actual.items() if k != "displayText"}, before)
        self.assertEqual(source, before)
        self.assertNotIn("firstCrossingEventForecast", air)

    def test_unavailable_publication_includes_nullable_text_without_trying_inference(self):
        import BukitKiara_Dashboard as dashboard
        with patch.object(dashboard.fresh_event_model, "predict_event",
                          side_effect=AssertionError("Unavailable sensor must not infer")):
            result = dashboard.attach_first_crossing_event_forecast({}, [], ANCHOR)
        actual = result["firstCrossingEventForecast"]
        self.assertFalse(actual["available"])
        self.assertIsNone(actual["displayText"])
        self.assertEqual(actual["reason"], "fresh_sensor_reference_unavailable")


if __name__ == "__main__":
    unittest.main()
