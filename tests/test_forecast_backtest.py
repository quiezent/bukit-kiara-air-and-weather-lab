"""Focused checks for exact targets, coverage, issued values, and read-only CLI."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import forecast_backtest as bt


class ForecastBacktestTests(unittest.TestCase):
    @staticmethod
    def scored_changes(changes):
        return [{"recorded_epoch": 1000, "actual_change": actual,
                 "predicted_change": predicted, "error": predicted - actual,
                 "baseline_error": -actual, "range_hit": None}
                for actual, predicted in changes]

    def test_opposite_sign_large_forecasts_are_not_directional_hits(self):
        summary = bt.summarize(self.scored_changes([(-30, 30), (30, -30)]), 20)
        # The compatibility metric detects magnitude only; directional evidence
        # must record two misses and two false alarms rather than two successes.
        self.assertEqual(summary["large_change_hits"], 2)
        self.assertIn("Direction-agnostic", summary["legacy_large_change_metrics_definition"])
        self.assertEqual(summary["large_change_correct_sign_hits"], 0)
        self.assertEqual(summary["large_change_correct_sign_false_alarms"], 2)
        self.assertEqual(summary["large_change_correct_sign_precision"], 0)
        self.assertEqual(summary["large_change_correct_sign_recall"], 0)
        self.assertEqual(summary["large_change_wrong_sign_count"], 2)
        self.assertEqual(summary["direction_confusion"]["drop"]["rise"], 1)
        self.assertEqual(summary["direction_confusion"]["rise"]["drop"], 1)

    def test_direction_confusion_separates_misses_false_alarms_and_correct_sign(self):
        changes = [(-30, 30), (30, -30), (-30, -25), (30, 25),
                   (-30, 0), (0, 30), (0, -30), (0, 0), (30, 0)]
        summary = bt.summarize(self.scored_changes(changes), 20)
        self.assertEqual(summary["direction_confusion"], {
            "drop": {"drop": 1, "stable": 1, "rise": 1},
            "stable": {"drop": 1, "stable": 1, "rise": 1},
            "rise": {"drop": 1, "stable": 1, "rise": 1},
        })
        self.assertEqual(summary["large_actual_change_count"], 6)
        self.assertEqual(summary["large_predicted_change_count"], 6)
        self.assertEqual(summary["large_change_correct_sign_hits"], 2)
        self.assertEqual(summary["large_change_correct_sign_false_alarms"], 4)
        for label in ("drop", "rise"):
            self.assertEqual(summary["large_change_by_direction"][label], {
                "actual_count": 3, "predicted_count": 3, "correct_sign_hits": 1,
                "false_alarms": 2, "misses": 2, "precision": .333, "recall": .333,
            })

    def test_large_direction_threshold_is_inclusive_and_empty_denominators_are_unknown(self):
        summary = bt.summarize(self.scored_changes([(-20, -20), (20, 20), (19.9, -19.9)]), 20)
        self.assertEqual(summary["large_change_correct_sign_hits"], 2)
        self.assertEqual(summary["direction_confusion"]["stable"]["stable"], 1)
        stable = bt.summarize(self.scored_changes([(0, 0)]), 20)
        self.assertIsNone(stable["large_change_correct_sign_precision"])
        self.assertIsNone(stable["large_change_correct_sign_recall"])
        for label in ("drop", "rise"):
            self.assertIsNone(stable["large_change_by_direction"][label]["precision"])
            self.assertIsNone(stable["large_change_by_direction"][label]["recall"])

    def test_event_timing_uses_publication_not_earlier_internal_forecast_clock(self):
        base = {"point": 20, "baseline": 40, "low": None, "high": None,
                "kind": "arrival", "start_epoch": 2700, "end_epoch": None,
                "forecast_epoch": 1798, "event_onset_epoch": 1800}
        forecasts = [{**base, "recorded_epoch": epoch} for epoch in (1799, 1800, 1801)]
        scored, excluded = bt.score_forecasts(forecasts, {2700: 20}, 2700)
        self.assertFalse(excluded)
        self.assertEqual([row["event_timing_role"] for row in scored],
                         ["pre_onset_target_forecast", "continuation_forecast", "continuation_forecast"])
        self.assertEqual([row["recorded_seconds_relative_to_onset"] for row in scored], [-1, 0, 1])
        self.assertEqual(bt.summarize(scored, 20)["event_timing_counts"],
                         {"pre_onset_target_forecast": 1, "continuation_forecast": 2})

    def test_right_closed_coverage_rejects_internal_gap(self):
        readings = [(1, 10), (450, 20), (899, 30), (900, 40),
                    (901, 10), (1501, 20), (1799, 30)]
        medians = bt.covered_bucket_medians(readings)
        self.assertEqual(medians, {900: 25.0})
        self.assertEqual(bt.closed_label(900), 900)
        self.assertEqual(bt.closed_label(901), 1800)

    def test_exact_point_and_partial_window_require_all_buckets(self):
        buckets = {900: 10, 1800: 20, 2700: 30}
        self.assertEqual(bt.score_target(buckets, "arrival", 1350, None, 2700), 15)
        self.assertEqual(bt.score_target(buckets, "mean", 600, 2100, 2700), 20)
        self.assertIsNone(bt.score_target(buckets, "arrival", 1350, None, 900))
        self.assertIsNone(bt.score_target({900: 10, 2700: 30}, "mean", 600, 2100, 2700))

    def test_rapid_onset_is_first_bucket_and_issue_precedes_it(self):
        buckets = {900: 100, 1800: 60, 2700: 50, 3600: 20, 4500: 50}
        episodes = bt.rapid_episodes(buckets, 20)
        self.assertEqual([(e["onset_epoch"], e["direction"]) for e in episodes],
                         [(900, "drop"), (3600, "rise")])
        self.assertEqual(episodes[0]["largest_15min_change"], -40)
        rows = [("before", 899), ("at", 900), ("after", 901)]
        self.assertEqual(bt.nearest_preonset(rows, 900, 60)[0], "before")

    def test_extract_uses_frozen_point_and_policy(self):
        payload = {"forecastIssuedEpoch": 1000,
                   "forecastPolicy": {"id": "issued_policy"},
                   "airWindow": {"available": True, "current15": 40,
                                 "forecastClock": {"targetVersion": bt.TARGET_VERSION,
                                                   "arrivalTargetEpoch": 6400,
                                                   "windowStartEpoch": 6400,
                                                   "windowEndEpoch": 13600},
                                 "arrival": {"point": 50, "rangeLow": 30, "rangeHigh": 60},
                                 "trail": {"point": 45, "baselinePoint": 42,
                                           "rangeLow": 25, "rangeHigh": 65}}}
        row = ("id", 1001, 1000, "build", None)
        forecasts = bt.extract_dashboard(row, payload, "regular")
        self.assertEqual([f["point"] for f in forecasts], [50, 45])
        self.assertEqual([f["baseline"] for f in forecasts], [40, 42])
        self.assertEqual([f["policy"] for f in forecasts], ["issued_policy"] * 2)
        payload["airWindow"]["forecastClock"]["targetVersion"] = "other"
        self.assertEqual(bt.extract_dashboard(row, payload, "regular"), [])

    def test_small_database_backtest_is_read_only_and_scores_exact_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "history.db"
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,pm02 REAL)")
                conn.execute("CREATE TABLE dashboard_forecast_issues(issue_id TEXT PRIMARY KEY,issued_epoch INTEGER,forecast_epoch INTEGER,dashboard_build TEXT,payload TEXT)")
                conn.execute("CREATE TABLE window_pm_forecast_issues(model_version TEXT,origin_epoch INTEGER,start_epoch INTEGER,end_epoch INTEGER,issued_epoch INTEGER,window_key TEXT,predicted_mean REAL,anchor_pm25 REAL,payload TEXT)")
                for label in range(900, 14401, 900):
                    value = 40 if label < 6300 else 20
                    conn.executemany("INSERT INTO readings VALUES(?,?)",
                                     [(label - 899, value), (label - 450, value), (label - 1, value)])
                payload = {"forecastIssuedEpoch": 1000, "forecastPolicy": {"id": "test"},
                           "airWindow": {"available": True, "current15": 40,
                                         "forecastClock": {"targetVersion": bt.TARGET_VERSION,
                                                           "arrivalTargetEpoch": 6300,
                                                           "windowStartEpoch": 6300,
                                                           "windowEndEpoch": 13500},
                                         "arrival": {"point": 30, "rangeLow": 15, "rangeHigh": 35},
                                         "trail": {"point": 30, "rangeLow": 15, "rangeHigh": 35}}}
                conn.execute("INSERT INTO dashboard_forecast_issues VALUES(?,?,?,?,?)",
                             ("issued", 1001, 1000, "test_build", json.dumps(payload)))
            before = hashlib.sha256(db.read_bytes()).digest()
            result = bt.run_backtest(db, rapid_threshold=1000)
            after = hashlib.sha256(db.read_bytes()).digest()
            self.assertEqual(before, after)
            self.assertEqual(result["counts"]["forecasts_scored"], 2)
            self.assertEqual(result["summary_by_scope"]["regular"]["mae"], 10)
            self.assertEqual(result["summary_by_scope"]["regular"]["persistence_mae"], 20)
            self.assertEqual(result["summary_by_scope"]["regular"]["range_hit_fraction"], 1)

    def test_regular_issue_during_observed_event_is_a_continuation_check(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "history.db"
            with closing(sqlite3.connect(db)) as conn, conn:
                conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,pm02 REAL)")
                conn.execute("CREATE TABLE dashboard_forecast_issues(issue_id TEXT PRIMARY KEY,issued_epoch INTEGER,forecast_epoch INTEGER,dashboard_build TEXT,payload TEXT)")
                conn.execute("CREATE TABLE window_pm_forecast_issues(model_version TEXT,origin_epoch INTEGER,start_epoch INTEGER,end_epoch INTEGER,issued_epoch INTEGER,window_key TEXT,predicted_mean REAL,anchor_pm25 REAL,payload TEXT)")
                for label in range(900, 27001, 900):
                    value = 40 if label < 6300 else 20
                    conn.executemany("INSERT INTO readings VALUES(?,?)",
                                     [(label - 899, value), (label - 450, value), (label - 1, value)])
                payload = {"forecastIssuedEpoch": 5500,
                           "airWindow": {"available": True, "current15": 20,
                                         "forecastClock": {"targetVersion": bt.TARGET_VERSION,
                                                           "arrivalTargetEpoch": 10900,
                                                           "windowStartEpoch": 10900,
                                                           "windowEndEpoch": 18100},
                                         "arrival": {"point": 20}, "trail": {"point": 20}}}
                conn.execute("INSERT INTO dashboard_forecast_issues VALUES(?,?,?,?,?)",
                             ("reactive", 5501, 5500, "test_build", json.dumps(payload)))
            result = bt.run_backtest(db)
            self.assertEqual(result["counts"]["rapid_episodes_observed"], 1)
            self.assertEqual(result["counts"]["rapid_episodes_with_pre_onset_issue"], 0)
            self.assertEqual(result["summary_by_scope"]["regular"]["event_timing_counts"],
                             {"continuation_forecast": 2})
            self.assertIn("not anticipation", result["method"]["event_timing_semantics"])


if __name__ == "__main__":
    unittest.main()
