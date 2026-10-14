"""Clock, coverage, and causality checks for the experimental change model."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import sqlite3
import unittest

import numpy as np

from short_horizon_change import (
    _archived_quarter_issues, _eligible_medians, _features_at, _raw_arrays,
    _target_at, _target_clock,
)


class ShortHorizonClockTests(unittest.TestCase):
    def setUp(self):
        self.anchor = 1_800_000_000 // 900 * 900
        self.issue = self.anchor + 880
        # Five observations per right-labelled 15-minute bucket.
        self.epochs = np.arange(self.anchor - 5400 + 2,
                                self.anchor + 3600, 180, dtype=np.int64)
        self.values = np.linspace(120, 50, len(self.epochs))

    def test_exact_targets_and_completion_are_issue_relative(self):
        target15, complete15, weights15 = _target_clock(self.issue, 15)
        target30, complete30, weights30 = _target_clock(self.issue, 30)
        self.assertEqual(target15, self.issue + 900)
        self.assertEqual(target30, self.issue + 1800)
        self.assertEqual(complete15, self.anchor + 1800)
        self.assertEqual(complete30, self.anchor + 2700)
        self.assertAlmostEqual(sum(weights15.values()), 1)
        self.assertAlmostEqual(sum(weights30.values()), 1)

    def test_target_requires_all_covered_buckets_and_closure(self):
        medians = _eligible_medians(self.epochs, self.values)
        self.assertIsNone(_target_at(self.issue, 15, medians, self.anchor + 1799))
        self.assertIsNotNone(_target_at(self.issue, 15, medians, self.anchor + 1800))
        # A sparse target bucket cannot silently be interpolated across.
        keep = ~((self.epochs > self.anchor + 900)
                 & (self.epochs <= self.anchor + 1800))
        sparse = _eligible_medians(self.epochs[keep], self.values[keep])
        self.assertIsNone(_target_at(self.issue, 15, sparse, self.anchor + 1800))

    def test_future_readings_do_not_change_issue_features(self):
        rows = [{"epoch": int(epoch), "pm02": float(value)}
                for epoch, value in zip(self.epochs, self.values)]
        past_epochs, past_values = _raw_arrays(rows, self.issue)
        past = _features_at(self.issue, past_epochs, past_values,
                            _eligible_medians(past_epochs, past_values))
        self.assertIsNotNone(past)
        rows.append({"epoch": self.issue + 1, "pm02": 99999.0})
        later_epochs, later_values = _raw_arrays(rows, self.issue)
        later = _features_at(self.issue, later_epochs, later_values,
                             _eligible_medians(later_epochs, later_values))
        np.testing.assert_array_equal(past["features"], later["features"])
        self.assertEqual(past["reference"], later["reference"])

    def test_archive_selection_uses_published_clock(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE dashboard_forecast_issues("
                         "issued_epoch INTEGER,forecast_epoch INTEGER,sensor_epoch INTEGER)")
            conn.executemany("INSERT INTO dashboard_forecast_issues VALUES(?,?,?)", [
                (self.anchor + 780, self.anchor + 760, self.anchor + 720),
                (self.anchor + 880, self.anchor + 850, self.anchor + 820),
                (self.anchor + 1680, self.anchor + 1670, self.anchor + 1620),
                (self.anchor + 1780, self.anchor + 1790, self.anchor + 1700),
            ])
            issues = _archived_quarter_issues(conn, self.anchor,
                                              self.anchor + 1800)
            self.assertEqual(issues, [(self.anchor + 850, self.anchor + 820),
                                      (self.anchor + 1670, self.anchor + 1620)])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
