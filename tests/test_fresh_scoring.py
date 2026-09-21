"""Causal references and adaptive weights over small synthetic histories."""

import copy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import numpy as np

from forecast_freshness import references
import window_pm_predictor as model


def history():
    records = []
    for i in range(16):
        origin = 1_000_000 + i * 1800
        components = (10.0, 8.0, 11.0, 9.0, 12.0)
        records.append({
            "originEpoch": origin, "targetEndEpoch": origin + 600,
            "prediction": 10.0, "actual": 10.0,
            "weights": dict.fromkeys(model.COMPONENTS, 0.2),
            **dict(zip(model.COMPONENTS, components)),
        })
    return records


def readings(records, lag=0, fresh=12.0):
    return [{"epoch": row["originEpoch"] + lag, "pm02": fresh} for row in records]


class FreshScoringTests(unittest.TestCase):
    def test_reference_window_is_open_left_closed_right_and_issue_causal(self):
        rows = [
            {"epoch": 9700, "pm02": 999.0},  # Excluded left boundary.
            {"epoch": 9701, "pm02": 10.0},
            {"epoch": 10000, "pm02": 20.0},
            {"epoch": 10001, "pm02": 999.0},  # Future sample.
            {"epoch": 9999, "pm02": -1.0},
            {"epoch": 9998, "pm02": np.inf},
        ]
        values, stamps, counts = references(rows, [9900, 10800], 100, 10000)
        self.assertEqual(values[0], 15.0)
        self.assertEqual(stamps.tolist(), [10000, 0])
        self.assertEqual(counts.tolist(), [2, 0])
        self.assertTrue(np.isnan(values[1]))
        empty, _, counts = references([], [9900], 100, 10000)
        self.assertTrue(np.isnan(empty[0]))
        self.assertEqual(counts[0], 0)

    def test_weights_score_refreshed_components_and_preserve_inputs(self):
        records = history()
        before = copy.deepcopy(records)
        result = model._fresh_weighted_records(records, readings(records), 0, records[-1]["originEpoch"])
        # The +2 reference shift makes the local ridge component exactly correct.
        expected = (np.array([2.0, 0.0, 3.0, 1.0, 4.0]) + 3.0) ** -2
        expected /= expected.sum()
        np.testing.assert_allclose(list(result[-1]["weights"].values()), expected)
        self.assertEqual(max(result[-1]["weights"], key=result[-1]["weights"].get), "half_ridge_local")
        self.assertEqual(records, before)
        self.assertIsNot(result[-1]["weights"], records[-1]["weights"])

    def test_weight_eligibility_excludes_equal_end_missing_labels_and_old_origins(self):
        records = history()
        issue = records[-1]["originEpoch"]
        records[0]["targetEndEpoch"] = issue
        records[1]["targetEndEpoch"] = issue + 1
        records[2]["actual"] = None
        result = model._fresh_weighted_records(records, readings(records), 0, issue)[-1]
        self.assertEqual(result["weightScoredCount"], 12)
        self.assertLess(result["weightLatestTargetEndEpoch"], issue)
        records[3]["actual"] = None
        result = model._fresh_weighted_records(records, readings(records), 0, issue)[-1]
        self.assertEqual(result["weightScoredCount"], 11)
        self.assertEqual(list(result["weights"].values()), [0.2] * 5)
        records = history()[:3]
        for row, origin in zip(records, (issue - 3 * 86400 - 1, issue - 3 * 86400, issue)):
            row.update(originEpoch=origin, targetEndEpoch=origin + 600)
        result = model._fresh_weighted_records(records, [], 0, issue)[-1]
        self.assertEqual(result["weightScoredCount"], 1)
        self.assertFalse(result["freshnessApplied"])
        self.assertEqual(result["prediction"], result["closedPrediction"])

    def test_future_readings_and_labels_after_frozen_cutoff_cannot_change_weights(self):
        records = history()
        issue = records[-1]["originEpoch"] + 299
        rows = readings(records, lag=299)
        future = rows + [{"epoch": row["epoch"] + 1, "pm02": 999999.0} for row in rows]
        baseline = model._fresh_weighted_records(records, rows, 299, issue)
        self.assertEqual(baseline, model._fresh_weighted_records(records, future, 299, issue))
        cutoff = records[13]["targetEndEpoch"]
        original = model._fresh_weighted_records(records, rows, 299, issue, cutoff)
        poisoned = copy.deepcopy(records)
        for row in poisoned:
            if row["targetEndEpoch"] >= cutoff:
                row["actual"] = 999999.0
        changed = model._fresh_weighted_records(poisoned, rows, 299, issue, cutoff)
        self.assertEqual(original[-1]["weights"], changed[-1]["weights"])
        self.assertLess(original[-1]["weightLatestTargetEndEpoch"], cutoff)

    def test_zero_floor_is_applied_after_mixing_closed_components(self):
        records = history()
        for row in records:
            row.update(dict.fromkeys(model.COMPONENTS, 10.0))
        records[-1].update(dict(zip(model.COMPONENTS, (10.0, 0.0, 20.0, 0.0, 20.0))))
        rows = readings(records, fresh=10.0)
        rows[-1]["pm02"] = 0.0
        result = model._fresh_weighted_records(records, rows, 0, records[-1]["originEpoch"])[-1]
        self.assertAlmostEqual(result["closedPrediction"], 10.0)
        self.assertEqual(result["prediction"], 0.0)
        self.assertNotEqual(result["prediction"], np.mean([0.0, 0.0, 10.0, 0.0, 10.0]))


if __name__ == "__main__":
    unittest.main()
