"""Synthetic formatter contracts and parity with the dashboard's actual JS."""
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unittest

import device_display

NOW = 1_800_000_000
INVALID = (None, True, False, "0.1", math.nan, math.inf, -math.inf, 10**400)


def current_case(*, gap=30, sensor_age=20, label="Rapid PM2.5 reduction detected"):
    stamp = NOW - sensor_age
    reading = {"epoch": stamp, "pm02": 159.123456789, "ageSeconds": sensor_age}
    analysis = {"current": {"epoch": stamp-gap}, "airWindow": {
        "observedMovement": {"eventLabel": label, "observedThroughEpoch": stamp-gap}}}
    return reading, analysis


def arrival_case(fall=.051, rise=.028, center=.921, *, fall40=.004, rise40=.001):
    return {"available": True, "probabilityFall20": fall, "probabilityFall40": fall40,
            "probabilityRise20": rise, "probabilityRise40": rise40,
            "probabilityWithin20": center, "forecastIssuedEpoch": NOW-20,
            "arrivalEpoch": NOW-20+5400, "arrivalLeadMinutes": 90,
            "referencePm": 160.1,
            "target": "exact_issue_plus90_arrival_median_proxy_delta_from_fresh5_reference"}


def empty_outcome():
    return {"display_text": None, "outcome": None, "outcome_probability": None}


class CurrentDisplayTests(unittest.TestCase):
    def test_observed_label_and_all_age_gap_boundaries(self):
        for gap, sensor_age, expected in ((0, 0, True), (600, 720, True),
                                          (600.001, 20, False), (30, 720.001, False),
                                          (-.001, 20, False), (30, -.001, False)):
            with self.subTest(gap=gap, sensor_age=sensor_age):
                reading, analysis = current_case(gap=gap, sensor_age=sensor_age)
                text = device_display.current_display_text(reading, analysis, NOW)
                self.assertEqual(text, "Observed: Rapid PM2.5 reduction detected" if expected
                                 else "Latest sensor reading")

    def test_nullish_observed_clock_falls_back_to_analysis_current_epoch(self):
        for explicit_null in (False, True):
            reading, analysis = current_case()
            observed = analysis["airWindow"]["observedMovement"]
            if explicit_null:
                observed["observedThroughEpoch"] = None
            else:
                observed.pop("observedThroughEpoch")
            self.assertEqual(device_display.current_display_text(reading, analysis, NOW),
                             "Observed: Rapid PM2.5 reduction detected")
        analysis["airWindow"]["observedMovement"]["observedThroughEpoch"] = 0
        self.assertEqual(device_display.current_display_text(reading, analysis, NOW),
                         "Latest sensor reading")  # Zero is invalid, not nullish.

    def test_actual_server_age_overrides_a_stale_client_age_snapshot(self):
        reading, analysis = current_case()
        reading["ageSeconds"] = 10000
        self.assertTrue(device_display.current_display_text(reading, analysis, NOW).startswith("Observed: "))
        reading["ageSeconds"] = 0
        self.assertEqual(device_display.current_display_text(reading, analysis, NOW+721),
                         "Latest sensor reading")

    def test_malformed_mappings_labels_and_clocks_fail_to_default(self):
        for reading, analysis in ((None, {}), ({}, None), ([], {}), ({}, "analysis"),
                                  ({"epoch":NOW}, {"airWindow":None}),
                                  ({"epoch":NOW}, {"airWindow":{"observedMovement":[]}})):
            self.assertEqual(device_display.current_display_text(reading, analysis, NOW),
                             "Latest sensor reading")
        for label in (None, "", "  \t", 0, True, ["label"]):
            reading, analysis = current_case(label=label)
            self.assertEqual(device_display.current_display_text(reading, analysis, NOW),
                             "Latest sensor reading")
        for field in ("reading", "through", "now"):
            for invalid in INVALID + (0, -1):
                with self.subTest(field=field, invalid=repr(invalid)):
                    reading, analysis = current_case()
                    now = NOW
                    if field == "reading":
                        reading["epoch"] = invalid
                    elif field == "through":
                        analysis["airWindow"]["observedMovement"]["observedThroughEpoch"] = invalid
                        if invalid is None:
                            analysis["current"] = None
                    else:
                        now = invalid
                    self.assertEqual(device_display.current_display_text(reading, analysis, now),
                                     "Latest sensor reading")

    def test_no_source_mutation_or_sensor_numeric_transformation(self):
        reading, analysis = current_case(label="  Dry clearing forming  ")
        original = deepcopy((reading, analysis))
        self.assertEqual(device_display.current_display_text(reading, analysis, NOW),
                         "Observed:   Dry clearing forming  ")
        self.assertEqual((reading, analysis), original)


class ArrivalOutcomeTests(unittest.TestCase):
    def test_three_outcomes_copy_the_native_winning_fraction(self):
        for source, label, outcome, field in (
            (arrival_case(), "No change on arrival", "within20", "probabilityWithin20"),
            (arrival_case(.75, .1, .15), "Fall on arrival", "fall20", "probabilityFall20"),
            (arrival_case(.1, .75, .15), "Rise on arrival", "rise20", "probabilityRise20"),
        ):
            original = deepcopy(source)
            result = device_display.arrival_outcome(source)
            self.assertEqual(result["display_text"], label)
            self.assertEqual(result["outcome"], outcome)
            self.assertIs(result["outcome_probability"], source[field])
            self.assertEqual(result["outcome_probability"].hex(), source[field].hex())
            self.assertEqual(source, original)

    def test_three_disjoint_groups_not_the_largest_individual_five_class_bin(self):
        source = arrival_case(.4, .25, .35, fall40=.1, rise40=.15)
        source["classProbabilities"] = [.1, .3, .35, .1, .15]
        self.assertEqual(max(source["classProbabilities"]), .35)
        result = device_display.arrival_outcome(source)
        self.assertEqual(result["outcome"], "fall20")
        self.assertEqual(result["outcome_probability"], .4)

    def test_exact_ties_are_uncertain_but_close_probabilities_are_not_rounded_before_selection(self):
        for fall, rise, center in ((.4, .4, .2), (.5, 0, .5), (0, .5, .5),
                                  (1/3, 1/3, 1/3)):
            source = arrival_case(fall, rise, center, fall40=0, rise40=0)
            self.assertEqual(device_display.arrival_outcome(source), {
                "display_text":"Arrival outcome uncertain", "outcome":None, "outcome_probability":None})
        source = arrival_case(.425000000001, .149999999999, .425)
        result = device_display.arrival_outcome(source)
        self.assertEqual(result["outcome"], "fall20")
        self.assertEqual(result["outcome_probability"], .425000000001)

    def test_unavailable_nonmapping_and_nonboolean_available_do_not_invent_text(self):
        for source in (None, [], "source", {}, {"available":False}):
            self.assertEqual(device_display.arrival_outcome(source), empty_outcome())
        for available in (None, False, 0, 1, "true"):
            source = arrival_case()
            source["available"] = available
            self.assertEqual(device_display.arrival_outcome(source), empty_outcome())

    def test_all_five_required_probabilities_are_finite_nonboolean_fractions(self):
        fields = ("probabilityFall20", "probabilityFall40", "probabilityRise20",
                  "probabilityRise40", "probabilityWithin20")
        for field in fields:
            for invalid in INVALID + (-.0001, 1.0001):
                with self.subTest(field=field, invalid=repr(invalid)):
                    source = arrival_case()
                    source[field] = invalid
                    self.assertEqual(device_display.arrival_outcome(source), empty_outcome())
            source = arrival_case()
            source.pop(field)
            self.assertEqual(device_display.arrival_outcome(source), empty_outcome())

    def test_nested_tail_and_disjoint_total_tolerances_reject_without_normalizing(self):
        for change in ({"probabilityFall40":.052}, {"probabilityRise40":.029},
                       {"probabilityWithin20":.5}, {"probabilityWithin20":.921+2e-8}):
            source = arrival_case()
            source.update(change)
            self.assertEqual(device_display.arrival_outcome(source), empty_outcome())
        source = arrival_case()
        source.update(probabilityFall40=.051+5e-13, probabilityWithin20=.921+5e-9)
        result = device_display.arrival_outcome(source)
        self.assertEqual(result["outcome"], "within20")
        self.assertEqual(result["outcome_probability"], .921+5e-9)

    def test_first20_operational_qualification_and_pm_points_never_choose_arrival_label(self):
        source = arrival_case()
        source.update(probabilityRise=.99, probabilityDrop=.005, probabilityNoCrossing=.005,
                      qualification={"operationalUseEligible":False},
                      modelSelection={"selectedPolicy":"persistence"}, arrivalPoint=-99.12345,
                      forecastIssuedEpoch=NOW+999999, referencePm=-1)
        original = deepcopy(source)
        self.assertEqual(device_display.arrival_outcome(source)["display_text"], "No change on arrival")
        self.assertEqual(source, original)
        # Clocks/reference are deliberately validated by the caller adapter.
        first20_only = {"available":True,"probabilityRise":.99,"probabilityDrop":.005,
                        "probabilityNoCrossing":.005}
        self.assertEqual(device_display.arrival_outcome(first20_only), empty_outcome())


def actual_web_function(html, name):
    start = html.index("function " + name + "(")
    next_function = re.search(r"\n(?:async )?function\s+", html[start+1:])
    if next_function is None:
        raise AssertionError("Could not isolate actual dashboard function " + name)
    return html[start:start+1+next_function.start()]


class ActualWebJavascriptParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        node = shutil.which("node")
        if node is None:
            raise AssertionError("Node is required for actual dashboard JS parity")
        cls.node = node
        path = Path(__file__).resolve().parents[1]/"app"/"BukitKiara_Dashboard.py"
        syntax = ast.parse(path.read_text(encoding="utf-8-sig"))
        cls.html = next(ast.literal_eval(item.value) for item in syntax.body if isinstance(item, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "HTML" for target in item.targets))

    def run_javascript(self, source, cases):
        program = """const cases=JSON.parse(process.argv[1]);
let fixedNow=0;
const NativeDate=Date;
class FixedDate extends NativeDate {static now(){return fixedNow*1000}}
globalThis.Date=FixedDate;
const elements={}; const $=id=>elements[id]??=({textContent:null,hidden:null});
let latestCurrent=null;let lastForecast=null;
""" + source
        output = subprocess.run([self.node, "-e", program, json.dumps(cases, allow_nan=False)],
                                check=True, capture_output=True, text=True, timeout=20)
        return json.loads(output.stdout)

    def test_current_helper_matches_actual_updateCurrentObservation_function(self):
        cases = []
        for gap, sensor_age in ((0, 0), (600, 720), (601, 20), (-1, 20), (30, 721), (30, -1)):
            reading, analysis = current_case(gap=gap, sensor_age=sensor_age)
            cases.append({"reading":reading,"analysis":analysis,"now":NOW})
        reading, analysis = current_case()
        analysis["airWindow"]["observedMovement"]["observedThroughEpoch"] = None
        cases.append({"reading":reading,"analysis":analysis,"now":NOW})
        for label in (None, "", "  ", "  Dry clearing forming  "):
            reading, analysis = current_case(label=label)
            cases.append({"reading":reading,"analysis":analysis,"now":NOW})
        source = actual_web_function(self.html, "updateCurrentObservation") + """
const results=cases.map(item=>{fixedNow=item.now;latestCurrent=item.reading;lastForecast=item.analysis;
 updateCurrentObservation(item.analysis);return $('particleLoad').textContent});
process.stdout.write(JSON.stringify(results));
"""
        actual = self.run_javascript(source, cases)
        expected = [device_display.current_display_text(item["reading"],item["analysis"],item["now"]) for item in cases]
        self.assertEqual(actual, expected)

    def test_arrival_helper_matches_actual_endpoint_renderer_without_first20(self):
        sources = [arrival_case(), arrival_case(.75,.1,.15), arrival_case(.1,.75,.15),
                   arrival_case(.4,.4,.2), arrival_case(.5,0,.5,fall40=0,rise40=0),
                   arrival_case(.4,.25,.35,fall40=.1,rise40=.15),
                   arrival_case(.425000000001,.149999999999,.425)]
        for item in sources:
            item.update(probabilityRise=.99,probabilityDrop=.005,probabilityNoCrossing=.005)
        cases = [{"source":item,"now":NOW} for item in sources]
        source = "\n".join(actual_web_function(self.html, name) for name in
                           ("f", "klClock", "renderArrivalChangeForecast")) + """
const results=cases.map(item=>{fixedNow=item.now;
renderArrivalChangeForecast(item.source,item.source.forecastIssuedEpoch,{baselinePoint:item.source.referencePm});
return $('arrivalChangeSignal').textContent});
process.stdout.write(JSON.stringify(results));
"""
        actual = self.run_javascript(source, cases)
        expected = [device_display.arrival_outcome(item)["display_text"] for item in sources]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
