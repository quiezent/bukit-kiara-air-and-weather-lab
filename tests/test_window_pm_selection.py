"""Focused checks for the session forecast's as-issued skill gate."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import window_pm_predictor as predictor


KL = timezone(timedelta(hours=8))


class IssuedSessionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "issued.db"
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("""CREATE TABLE window_pm_forecast_issues(
                model_version TEXT NOT NULL, origin_epoch INTEGER NOT NULL,
                start_epoch INTEGER NOT NULL, end_epoch INTEGER NOT NULL,
                issued_epoch INTEGER NOT NULL, window_key TEXT NOT NULL,
                predicted_mean REAL NOT NULL, anchor_pm25 REAL NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(model_version,origin_epoch,start_epoch,end_epoch))""")

    def _make_history(self, count=14, bad_policy_day=None, wrong_lag_day=None, missing_day=None):
        buckets = {}
        current_origin = int(datetime(2026, 10, 1, 20, 30, tzinfo=KL).timestamp())
        lead = 9 * 3600 + 30 * 60
        current_start = current_origin + lead
        current_issue = current_origin + 25
        with closing(sqlite3.connect(self.db)) as conn, conn:
            for day in range(1, count + 1):
                origin = current_origin - day * 86400
                start = current_start - day * 86400
                end = start + 7200
                issue = origin + (200 if day == wrong_lag_day else 25)
                actual = 40. + day
                for k in range(1, 9):
                    if day != missing_day or k != 7:
                        buckets[pd.Timestamp(start + k * 900, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")] = actual
                candidate = actual + 1.
                baseline = actual + 11.
                payload = {
                    "modelVersion": predictor.CANDIDATE_MODEL_VERSION,
                    "originEpoch": origin, "startEpoch": start, "endEpoch": end,
                    "forecastedAtEpoch": issue, "mean": candidate,
                    "sensorAnchor": baseline,
                    "adaptiveWeighting": {"version": "old_policy" if day == bad_policy_day else predictor.WEIGHT_POLICY_VERSION},
                    "freshnessAdjustment": {"version": predictor.FRESHNESS_VERSION},
                }
                conn.execute("""INSERT INTO window_pm_forecast_issues VALUES(?,?,?,?,?,?,?,?,?)""",
                    (predictor.CANDIDATE_MODEL_VERSION, origin, start, end, issue + 10,
                     "morning", candidate, baseline, json.dumps(payload)))
        frame = pd.DataFrame({"pm02": pd.Series(buckets)}).sort_index()
        return frame, current_origin, current_start, current_issue

    def _evidence(self, frame, origin, start, issue):
        return predictor._issued_evidence(self.db, frame, "morning", start, start + 7200,
                                          origin, issue, issue - origin)

    def test_gate_requires_fourteen_matched_issued_days(self):
        frame, origin, start, issue = self._make_history(count=13)
        evidence, candidate_errors, baseline_errors = self._evidence(frame, origin, start, issue)
        self.assertEqual(evidence["count"], 13)
        self.assertFalse(evidence["supportSufficient"])
        self.assertFalse(evidence["skillGatePassed"])
        self.assertEqual(len(candidate_errors), len(baseline_errors))

    def test_gate_can_promote_when_matched_issued_candidate_beats_persistence(self):
        frame, origin, start, issue = self._make_history()
        evidence, _, _ = self._evidence(frame, origin, start, issue)
        self.assertEqual(evidence["distinctDays"], 14)
        self.assertTrue(evidence["supportSufficient"])
        self.assertTrue(evidence["skillGatePassed"])
        self.assertEqual(evidence["mae"], 1.)
        self.assertEqual(evidence["persistenceMae"], 11.)

    def test_wrong_policy_clock_and_incomplete_target_cannot_validate(self):
        frame, origin, start, issue = self._make_history(
            bad_policy_day=1, wrong_lag_day=2, missing_day=3)
        evidence, _, _ = self._evidence(frame, origin, start, issue)
        self.assertEqual(evidence["count"], 11)
        self.assertEqual(evidence["excludedPolicyOrClockCount"], 2)
        self.assertEqual(evidence["missingCompletedTargetCount"], 1)
        self.assertFalse(evidence["skillGatePassed"])

    def test_tail_error_blocks_promotion_despite_mean_gain(self):
        frame, origin, start, issue = self._make_history()
        with closing(sqlite3.connect(self.db)) as conn, conn:
            rows = conn.execute("""SELECT origin_epoch,start_epoch,end_epoch,payload
                FROM window_pm_forecast_issues ORDER BY origin_epoch LIMIT 2""").fetchall()
            for old_origin, old_start, old_end, raw in rows:
                payload = json.loads(raw)
                actual = float(frame.pm02.loc[pd.Timestamp(old_start + 900, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")])
                payload["mean"] = actual + 20.
                conn.execute("""UPDATE window_pm_forecast_issues
                    SET predicted_mean=?,payload=? WHERE model_version=? AND origin_epoch=? AND start_epoch=? AND end_epoch=?""",
                    (payload["mean"], json.dumps(payload), predictor.CANDIDATE_MODEL_VERSION,
                     old_origin, old_start, old_end))
        evidence, _, _ = self._evidence(frame, origin, start, issue)
        self.assertLess(evidence["mae"], evidence["persistenceMae"])
        self.assertGreater(evidence["p90AbsoluteError"], evidence["persistenceP90AbsoluteError"])
        self.assertFalse(evidence["skillGatePassed"])

    def test_published_fallback_and_candidate_have_separate_archive_versions(self):
        origin = int(datetime(2026, 10, 1, 20, 30, tzinfo=KL).timestamp())
        start = origin + 9 * 3600 + 30 * 60
        issue = origin + 25
        value = {
            "available": True, "modelVersion": predictor.MODEL_VERSION,
            "originEpoch": origin, "startEpoch": start, "endEpoch": start + 7200,
            "forecastedAtEpoch": issue, "mean": 52., "prediction": 52.,
            "sensorAnchor": 52., "candidateForecast": {
                "modelVersion": predictor.CANDIDATE_MODEL_VERSION,
                "mean": 45., "prediction": 45., "rawRangeLow": None,
                "rawRangeHigh": None},
            "adaptiveWeighting": {"version": predictor.WEIGHT_POLICY_VERSION},
            "freshnessAdjustment": {"version": predictor.FRESHNESS_VERSION},
        }
        self.assertEqual(predictor.record_issue(self.db, {"morning": value}, issue + 10), 2)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            values = conn.execute("SELECT model_version,predicted_mean FROM window_pm_forecast_issues ORDER BY model_version").fetchall()
        self.assertEqual(values, [(predictor.CANDIDATE_MODEL_VERSION, 45.), (predictor.MODEL_VERSION, 52.)])


if __name__ == "__main__":
    unittest.main()
