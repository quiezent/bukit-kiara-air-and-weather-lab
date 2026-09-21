"""Synthetic decision-clock checks; no archived measurements are required."""

from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import numpy as np
import pandas as pd

from forecast_clock import (
    ForecastClock, score_observed_window, target_series, targets_match,
    window_weights,
)


class ForecastClockTests(unittest.TestCase):
    def setUp(self):
        self.clock = ForecastClock(issued_epoch=300, anchor_epoch=0)
        self.series = pd.Series(
            np.arange(20, dtype=float) * 10,
            index=pd.date_range("1970-01-01", periods=20, freq="15min", tz="UTC"),
        )

    def test_issue_lag_sets_exact_targets_and_partial_bucket_weights(self):
        metadata = self.clock.metadata()
        self.assertEqual(metadata["windowStartEpoch"], 5700)
        self.assertEqual(metadata["windowEndEpoch"], 12900)
        self.assertEqual(metadata["observedOutcomeCompleteAfterEpoch"], 13500)
        weights = window_weights(5700, 12900)
        self.assertEqual(set(weights), set(range(7, 16)))
        self.assertAlmostEqual(weights[7], 600 / 7200)
        self.assertAlmostEqual(weights[15], 300 / 7200)
        self.assertAlmostEqual(sum(weights.values()), 1)
        targets = target_series(self.series, self.clock).iloc[0]
        self.assertAlmostEqual(targets["arrival"], 60 * 2 / 3 + 70 / 3)
        self.assertAlmostEqual(targets["mean"], 780000 / 7200)
        self.assertEqual(targets["peak"], 150)

    def test_scoring_waits_for_final_touched_bucket_to_close(self):
        for as_of in (self.clock.target_end_epoch, 13499):
            with self.subTest(as_of=as_of):
                result = score_observed_window(self.series, self.clock, as_of)
                self.assertFalse(result["available"])
                self.assertEqual(result["reason"], "target_buckets_not_closed")
        result = score_observed_window(self.series, self.clock, 13500)
        self.assertTrue(result["available"])
        self.assertEqual(result["requiredBuckets"], 9)
        self.assertAlmostEqual(result["mean"], 780000 / 7200)

    def test_missing_partial_edge_invalidates_window_without_imputation(self):
        self.series.iloc[15] = np.nan
        targets = target_series(self.series, self.clock).iloc[0]
        self.assertTrue(np.isfinite(targets["arrival"]))
        self.assertTrue(np.isnan(targets["mean"]))
        self.assertTrue(np.isnan(targets["peak"]))
        result = score_observed_window(self.series, self.clock, 13500)
        self.assertEqual(result["reason"], "incomplete_observed_window")
        self.assertEqual(result["observedBuckets"], 8)
        self.assertAlmostEqual(result["coveredFraction"], 23 / 24)

    def test_compressed_time_gaps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Keep missing buckets as NaN"):
            target_series(self.series.drop(self.series.index[10]), self.clock)

    def test_overlay_requires_identical_target_version_and_boundaries(self):
        metadata = self.clock.metadata()
        self.assertTrue(targets_match(metadata, dict(metadata)))
        for field in ("targetVersion", "arrivalTargetEpoch", "windowStartEpoch", "windowEndEpoch"):
            with self.subTest(field=field):
                changed = dict(metadata)
                changed[field] = "different" if field == "targetVersion" else metadata[field] + 1
                self.assertFalse(targets_match(metadata, changed))
                del changed[field]
                self.assertFalse(targets_match(metadata, changed))


if __name__ == "__main__":
    unittest.main()
