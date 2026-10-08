"""Web/device integration contracts using only synthetic published inputs.

No model fitting, production database reads, provider calls or service writes.
"""
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

import rlcd_api
import rlcd_history
from test_arrival_api_projection import rich_inputs
from test_weather_contracts import ANCHOR


NOW = ANCHOR + 20
TAILS = (("probabilityFall20", "fall20"), ("probabilityFall40", "fall40"),
         ("probabilityRise20", "rise20"), ("probabilityRise40", "rise40"),
         ("probabilityWithin20", "within20"))


def encoded(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def set_groups(source, fall, rise, within):
    # These are the model's native disjoint aggregate probabilities, not a
    # publisher normalisation or a numeric-head selection policy.
    fall40, rise40 = fall / 4, rise / 4
    source.update(probabilityFall20=fall, probabilityFall40=fall40,
                  probabilityRise20=rise, probabilityRise40=rise40,
                  probabilityWithin20=within,
                  classProbabilities=[fall40, fall-fall40, within,
                                      rise-rise40, rise40])


def observed_rows(count=122):
    """Actual integral sample clocks, with separate channel extrema and a gap."""
    first, last = NOW-21600+50, ANCHOR-30
    rows = [{"epoch": first + i*(last-first)//(count-1),
             "pm02": 100.123456789 + i*.137 + 4*math.sin(i*.19),
             "atmp": 29.125 + 2*math.sin(i*.13),
             "heatindex": 36.625 + 3*math.cos(i*.17)} for i in range(count)]
    for index, field, value in ((count//11, "atmp", -12.34),
                                (count//11+1, "atmp", 63.27),
                                (count*7//11, "heatindex", -6.28),
                                (count*7//11+1, "heatindex", 98.36)):
        rows[index][field] = value
    if count > 128:
        # Omit original collection rows to create a real source gap; widening
        # selected-point spacing alone must never invent additional gaps.
        rows = rows[:count//3] + rows[count//3+count//20:]
    return rows


class RlcdWebContractTests(unittest.TestCase):
    def project(self, values=None, now=NOW):
        reading, analysis, weather, rows = rich_inputs() if values is None else values
        return rlcd_api.build_payload(reading, analysis, weather, now,
                                      history_rows=rows, dashboard_build="synthetic-web-contract")

    def assert_unavailable_change(self, result):
        change = result["forecast"]["near90"]["arrival_change"]
        self.assertFalse(change["available"])
        for key in ("fall20", "fall40", "rise20", "rise40", "within20",
                    "display_text", "outcome", "outcome_probability", "issued_epoch",
                    "fresh_reference_epoch", "reference_ugm3", "arrival_epoch"):
            self.assertIsNone(change[key], key)

    def test_current_text_matches_observed_web_label_and_rejects_stale_reference(self):
        values = rich_inputs()
        reading, analysis = values[:2]
        observed = {"eventLabel": "PM2.5 fell ≥20 µg/m³",
                    "observedThroughEpoch": reading["epoch"]}
        analysis["airWindow"]["observedMovement"] = observed
        result = self.project(values)
        self.assertEqual(result["current"]["display_text"], "Observed: PM2.5 fell ≥20 µg/m³")
        self.assertEqual(result["current"]["pm25_ugm3"], reading["pm02"])
        for through, now, expected in (
                (reading["epoch"]-600, NOW, "Observed: PM2.5 fell ≥20 µg/m³"),
                (reading["epoch"]-601, NOW, "Latest sensor reading"),
                (reading["epoch"]+1, NOW, "Latest sensor reading"),
                (reading["epoch"], reading["epoch"]+721, "Latest sensor reading")):
            with self.subTest(through=through, now=now):
                observed["observedThroughEpoch"] = through
                self.assertEqual(self.project(values, now)["current"]["display_text"], expected)
        observed["observedThroughEpoch"] = None
        self.assertEqual(self.project(values)["current"]["display_text"], "Observed: PM2.5 fell ≥20 µg/m³")
        analysis["current"]["epoch"] = reading["epoch"]-601
        self.assertEqual(self.project(values)["current"]["display_text"], "Latest sensor reading")

    def test_arrival_aggregate_winner_is_independent_of_first20_and_numeric_heads(self):
        values = rich_inputs()
        before = self.project(values)["forecast"]
        source = values[1]["airWindow"]["arrivalChangeForecast"]
        # The first case's largest individual five-class bin is within20,
        # while the native aggregate fall20 group is the headline winner.
        cases = ((.52, .06, .42, "Fall on arrival", "fall20", "probabilityFall20"),
                 (.05, .7, .25, "Rise on arrival", "rise20", "probabilityRise20"),
                 (.12, .18, .7, "No change on arrival", "within20", "probabilityWithin20"))
        for fall, rise, within, text, outcome, winner in cases:
            with self.subTest(outcome=outcome):
                set_groups(source, fall, rise, within)
                result = self.project(values)["forecast"]
                change = result["near90"]["arrival_change"]
                self.assertTrue(change["available"])
                self.assertEqual(change["display_text"], text)
                self.assertEqual(change["outcome"], outcome)
                self.assertEqual(change["outcome_probability"].hex(), source[winner].hex())
                for original, projected in TAILS:
                    self.assertEqual(change[projected].hex(), source[original].hex())
                self.assertEqual(encoded(result["near90"]["first20"]), encoded(before["near90"]["first20"]))
                self.assertEqual(result["near90"]["pm25_ugm3"], before["near90"]["pm25_ugm3"])
                self.assertEqual(encoded(result["ride90_210"]), encoded(before["ride90_210"]))
                self.assertEqual(encoded(result["sessions"]), encoded(before["sessions"]))

    def test_exact_aggregate_ties_are_uncertain_but_near_ties_are_not_rewritten(self):
        values = rich_inputs()
        source = values[1]["airWindow"]["arrivalChangeForecast"]
        for fall, rise, within in ((.4, .2, .4), (.5, .5, 0), (1/3, 1/3, 1/3)):
            with self.subTest(groups=(fall, rise, within)):
                set_groups(source, fall, rise, within)
                result = self.project(values)["forecast"]["near90"]["arrival_change"]
                self.assertTrue(result["available"])
                self.assertEqual(result["display_text"], "Arrival outcome uncertain")
                self.assertIsNone(result["outcome"])
                self.assertIsNone(result["outcome_probability"])
        rise = math.nextafter(.5, 0)
        set_groups(source, .5, rise, 1-.5-rise)
        result = self.project(values)["forecast"]["near90"]["arrival_change"]
        self.assertEqual(result["outcome"], "fall20")
        self.assertEqual(result["outcome_probability"], .5)
        self.assertEqual(result["rise20"].hex(), rise.hex())

    def test_new_fields_keep_native_precision_and_projection_is_read_only(self):
        values = rich_inputs()
        source = values[1]["airWindow"]["arrivalChangeForecast"]
        source["probabilityWithin20"] = math.nextafter(source["probabilityWithin20"], 1)
        source["classProbabilities"][2] = source["probabilityWithin20"]
        original = encoded(values)
        with patch("sqlite3.connect", side_effect=AssertionError("unexpected database access")):
            result = self.project(values)
        self.assertEqual(encoded(values), original)
        change = result["forecast"]["near90"]["arrival_change"]
        self.assertEqual(change["issued_epoch"], ANCHOR)
        self.assertEqual(change["fresh_reference_epoch"], source["freshReferenceEpoch"])
        self.assertEqual(change["outcome_probability"].hex(), source["probabilityWithin20"].hex())
        self.assertEqual(change["within20"].hex(), source["probabilityWithin20"].hex())
        self.assertLessEqual(len(encoded(result)), 16384)

    def test_invalid_reference_and_baseline_reject_only_arrival_change(self):
        original_values = rich_inputs()
        original_forecast = self.project(original_values)["forecast"]
        reference = original_values[1]["airWindow"]["arrivalChangeForecast"]["referencePm"]
        cases = (("freshReferenceEpoch", None), ("freshReferenceEpoch", 0),
                 ("freshReferenceEpoch", True), ("freshReferenceEpoch", "123"),
                 ("freshReferenceEpoch", ANCHOR+1), ("freshReferenceEpoch", ANCHOR-241),
                 ("referencePm", None), ("referencePm", True), ("referencePm", -1),
                 ("referencePm", math.nan), ("referencePm", math.inf),
                 ("referencePm", reference+2e-7))
        for field, invalid in cases:
            with self.subTest(field=field, value=repr(invalid)):
                values = deepcopy(original_values)
                values[1]["airWindow"]["arrivalChangeForecast"][field] = invalid
                result = self.project(values)
                self.assert_unavailable_change(result)
                near = result["forecast"]["near90"]
                self.assertTrue(near["available"])
                self.assertEqual(near["pm25_ugm3"], original_forecast["near90"]["pm25_ugm3"])
                self.assertEqual(encoded(near["first20"]), encoded(original_forecast["near90"]["first20"]))
                self.assertEqual(encoded(result["forecast"]["ride90_210"]), encoded(original_forecast["ride90_210"]))
                encoded(result)
        values = rich_inputs()
        values[1]["airWindow"]["arrival"].pop("baselinePoint")
        self.assert_unavailable_change(self.project(values))
        values = rich_inputs()
        values[1]["airWindow"]["arrivalChangeForecast"]["freshReferenceEpoch"] = ANCHOR-240
        values[1]["airWindow"]["arrival"]["baselinePoint"] = reference+5e-8
        self.assertTrue(self.project(values)["forecast"]["near90"]["arrival_change"]["available"])

    def test_fractional_future_or_mismatched_source_clocks_cannot_pass_by_truncation(self):
        for field, value in (("freshReferenceEpoch", ANCHOR+.5),
                             ("forecastIssuedEpoch", ANCHOR+.5),
                             ("arrivalEpoch", ANCHOR+5400+.5)):
            with self.subTest(field=field):
                values = rich_inputs()
                values[1]["airWindow"]["arrivalChangeForecast"][field] = value
                self.assert_unavailable_change(self.project(values))
        values = rich_inputs()
        source = values[1]["airWindow"]["arrivalChangeForecast"]
        for field in ("freshReferenceEpoch", "forecastIssuedEpoch", "arrivalEpoch"):
            source[field] = float(source[field])
        self.assertTrue(self.project(values)["forecast"]["near90"]["arrival_change"]["available"])

    def test_source_sensor_delivery_and_parent_clock_guards_do_not_reissue_a_forecast(self):
        cases = (("expired", NOW, None), ("ready", ANCHOR+601, None),
                 ("ready", ANCHOR-1, None), ("ready", NOW, NOW+1),
                 ("ready", NOW, NOW-901))
        for delivery, now, sensor_epoch in cases:
            with self.subTest(delivery=delivery, now=now, sensor_epoch=sensor_epoch):
                values = rich_inputs()
                values[1]["delivery"]["state"] = delivery
                if sensor_epoch is not None: values[0]["epoch"] = sensor_epoch
                self.assert_unavailable_change(self.project(values, now))
                self.assertEqual(values[1]["forecastIssuedEpoch"], ANCHOR)
        values = rich_inputs()
        values[1]["airWindow"]["forecastClock"]["arrivalTargetEpoch"] += 1
        self.assert_unavailable_change(self.project(values))

    def test_arrival_change_survives_unavailable_numeric_head_with_retained_reference(self):
        values = rich_inputs()
        before = self.project(values)["forecast"]["near90"]["first20"]
        values[1]["airWindow"]["arrival"].update(available=False, point=None)
        result = self.project(values)["forecast"]["near90"]
        self.assertFalse(result["available"])
        self.assertIsNone(result["pm25_ugm3"])
        self.assertTrue(result["arrival_change"]["available"])
        self.assertEqual(encoded(result["first20"]), encoded(before))

    def test_six_hour_122_observed_rows_fit_with_new_web_text_fields(self):
        reading, analysis, weather, _ = rich_inputs()
        rows = observed_rows()
        analysis["airWindow"]["observedMovement"] = {"eventLabel": "PM2.5 fell ≥20 µg/m³",
            "observedThroughEpoch": reading["epoch"]}
        result = self.project((reading, analysis, weather, rows))
        self.assertEqual(result["history"]["source_rows"], 122)
        self.assertEqual(len(result["history"]["points"]), 122)
        self.assertGreaterEqual(len(result["history"]["points"]), 100)
        self.assertEqual([point[0] for point in result["history"]["points"]], [r["epoch"] for r in rows])
        self.assertEqual(result["current"]["display_text"], "Observed: PM2.5 fell ≥20 µg/m³")
        self.assertEqual(result["forecast"]["near90"]["arrival_change"]["display_text"], "No change on arrival")
        self.assertLessEqual(len(encoded(result)), 16384)

    def test_tight_budget_rebuilds_original_rows_and_preserves_multichannel_history(self):
        reading, analysis, weather, _ = rich_inputs()
        rows = observed_rows(800)
        values = (reading, analysis, weather, rows)
        untouched = encoded(values)
        full = self.project(values)
        expected = rlcd_history.build_history(rows, NOW, max_points=24)
        expected["method"] = "device_byte_budget_" + expected["method"]
        candidate = deepcopy(full)
        candidate["history"] = expected
        budget = len(encoded(candidate))
        # A distinguishing fixture: rebuilding originals retains actual rows
        # that were not in the initial 128-point selection.
        self.assertTrue({p[0] for p in expected["points"]} - {p[0] for p in full["history"]["points"]})
        with patch.object(rlcd_api, "MAX_PAYLOAD_BYTES", budget):
            result = self.project(values)
        self.assertLessEqual(len(encoded(result)), budget)
        actual = result["history"]
        self.assertLess(len(actual["points"]), len(full["history"]["points"]))
        direct = rlcd_history.build_history(rows, NOW, max_points=len(actual["points"]))
        direct["method"] = "device_byte_budget_" + direct["method"]
        self.assertEqual(actual, direct)
        self.assertEqual(actual["pm25_summary"], full["history"]["pm25_summary"])
        self.assertEqual(actual["pm25_summary"]["sample_count"], len(rows))
        self.assertEqual(actual["gaps"], full["history"]["gaps"])
        self.assertTrue(actual["gaps"])
        self.assertEqual(actual["source_rows"], len(rows))
        selected = {point[0] for point in actual["points"]}
        for field, column in (("pm02", 1), ("atmp", 2), ("heatindex", 3)):
            for extreme in (min, max):
                original_extreme = extreme(rows, key=lambda r: r[field])
                self.assertEqual(extreme(p[column] for p in actual["points"]),
                                 round(original_extreme[field], 1))
                if field != "pm02":
                    # These temperature/HI extrema are unique even after
                    # wire rounding; their original clocks must survive.
                    self.assertIn(original_extreme["epoch"], selected)
        original_by_epoch = {row["epoch"]: row for row in rows}
        for point in actual["points"]:
            original = original_by_epoch[point[0]]
            self.assertEqual(point[1:], [round(original[field], 1)
                             for field in ("pm02", "atmp", "heatindex")])
        self.assertEqual(encoded(result["forecast"]), encoded(full["forecast"]))
        self.assertEqual(encoded(result["current"]), encoded(full["current"]))
        self.assertEqual(encoded(values), untouched)


if __name__ == "__main__":
    unittest.main()
