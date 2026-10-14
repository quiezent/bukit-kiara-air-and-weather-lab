"""Regression checks for the requested split between session and +90 policies."""

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

import BukitKiara_Dashboard as dashboard
import window_pm_predictor as predictor


KL = timezone(timedelta(hours=8))


class UserSelectedSessionPolicyTests(unittest.TestCase):
    def test_session_candidate_is_selected_without_claiming_skill(self):
        point, policy, reason, version, applied = predictor._published_session_selection(45., 52.)
        self.assertEqual(point, 45.)
        self.assertEqual(policy, "user_selected_experimental_candidate")
        self.assertEqual(reason, "explicit_user_request")
        self.assertEqual(version, predictor.CANDIDATE_MODEL_VERSION)
        self.assertTrue(applied)

        fallback = predictor._published_session_selection(float("nan"), 52.)
        self.assertEqual(fallback[0], 52.)
        self.assertEqual(fallback[1], "same_issue_persistence")

    def test_published_candidate_archive_identifies_primary_use(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "issued.db"
            origin = int(datetime(2026, 10, 1, 7, 30, tzinfo=KL).timestamp())
            start = origin + 90 * 60
            issue = origin + 25
            value = {
                "available": True, "modelVersion": predictor.MODEL_VERSION,
                "originEpoch": origin, "startEpoch": start, "endEpoch": start + 7200,
                "forecastedAtEpoch": issue, "mean": 45., "prediction": 45.,
                "sensorAnchor": 52.,
                "modelSelection": {"selectedPolicy": "user_selected_experimental_candidate"},
                "candidateForecast": {
                    "modelVersion": predictor.CANDIDATE_MODEL_VERSION,
                    "mean": 45., "prediction": 45.,
                    "rawRangeLow": None, "rawRangeHigh": None,
                },
                "adaptiveWeighting": {"version": predictor.WEIGHT_POLICY_VERSION},
                "freshnessAdjustment": {"version": predictor.FRESHNESS_VERSION},
            }
            self.assertEqual(predictor.record_issue(db, {"morning": value}, issue + 10), 2)
            with closing(sqlite3.connect(db)) as conn:
                records = conn.execute(
                    "SELECT model_version,predicted_mean,payload FROM window_pm_forecast_issues"
                ).fetchall()
            self.assertEqual({row[0] for row in records},
                             {predictor.MODEL_VERSION, predictor.CANDIDATE_MODEL_VERSION})
            self.assertTrue(all(row[1] == 45. for row in records))
            frozen = next(json.loads(row[2]) for row in records
                          if row[0] == predictor.CANDIDATE_MODEL_VERSION)
            self.assertEqual(frozen["modelSelection"]["selectedPolicy"], "published_candidate")
            self.assertTrue(frozen["modelSelection"]["appliedToPrimaryForecast"])

    def test_plus_90_remains_persistence_with_eligible_model_and_event(self):
        clock = {
            "forecastIssuedEpoch": 100000,
            "arrivalTargetEpoch": 105400,
            "windowStartEpoch": 105400,
            "windowEndEpoch": 112600,
            "targetVersion": "decision_clock_weighted_15min_v1",
        }
        baseline = {
            "available": True, "current15": 50., "forecastClock": clock,
            "arrival": {"available": True, "point": 50., "pointRole": "persistence_anchor"},
            "trail": {"available": True, "point": 50., "pointRole": "persistence_anchor"},
        }
        model = {
            "featureAnchor": 50., "forecastClock": clock,
            "arrivalPoint": 20., "arrivalDelta": -30.,
            "backtest": {"deploymentSkill": {
                "arrival": {"eligible": True, "gateOutcome": "passed"},
            }},
        }
        event = {
            "available": True, "forecastClock": clock,
            "currentRawAnchor": 50., "arrivalPoint": 15.,
            "completedEventCount": 3,
        }
        result = dashboard.aggressive_air_window_forecast(baseline, model, event)
        self.assertEqual(result["arrival"]["point"], 50.)
        self.assertEqual(result["arrival"]["pointRole"], "persistence_anchor")
        selection = result["nearTermSelectionDiagnostics"]["targets"]["arrival"]
        self.assertFalse(selection["localComponentApplied"])
        self.assertEqual(selection["selectionOutcome"], "retained_persistence_by_user")
        self.assertIn("user_selected_persistence", selection["blockedReasons"])


if __name__ == "__main__":
    unittest.main()
