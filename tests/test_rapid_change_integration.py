"""The experimental absolute-change signal stays separate from PM forecasts."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import copy
import importlib.util
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from forecast_archive import record_dashboard_issue
from forecast_clock import ForecastClock
from forecast_payload import loads


def dashboard_module():
    spec = importlib.util.spec_from_file_location(
        "dashboard_rapid_change_tests", APP / "BukitKiara_Dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RapidChangeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = dashboard_module()

    def setUp(self):
        self.issue = 1_800_000_000
        self.clock = ForecastClock(self.issue, self.issue).metadata()
        self.air_window = {
            "available": True, "forecastClock": self.clock,
            "arrival": {
                "available": True, "point": 44.0, "baselinePoint": 40.0,
                "pointRole": "aggressive_local_analogue",
                "rawRangeLow": 20.0, "rawRangeHigh": 60.0,
            },
            "trail": {
                "available": True, "point": 38.0, "baselinePoint": 40.0,
                "pointRole": "aggressive_local_ridge",
                "rawRangeLow": 18.0, "rawRangeHigh": 65.0,
            },
        }
        self.risk = {
            "available": True, "modelVersion": "test_absolute_change_v1",
            "experimental": True, "prospectivelyValidated": False,
            "calibrated": False, "directionAvailable": False,
            "thresholdUgM3": 20.0, "probabilityThreshold": 0.20,
            "definition": "P(abs(target PM2.5 - as-issued fresh PM2.5) >= 20 µg/m³)",
            "sourceIssueEpoch": self.issue, "featureAnchorEpoch": self.issue,
            "freshPm25UgM3": 40.0,
            "trainingCutoffEpoch": self.issue - 3600,
            "trainedThroughIssueDay": "2027-01-14",
            "targets": {
                "arrival90": {
                    "available": True, "probability": 0.12, "flag": False,
                    "targetEpoch": self.clock["arrivalTargetEpoch"],
                    "completionEpoch": self.clock["arrivalTargetEpoch"],
                    "trainingCount": 20,
                },
                "mean90to210": {
                    "available": True, "probability": 0.2459, "flag": True,
                    "startEpoch": self.clock["windowStartEpoch"],
                    "endEpoch": self.clock["windowEndEpoch"],
                    "completionEpoch": self.clock["observedOutcomeCompleteAfterEpoch"],
                    "trainingCount": 20,
                },
            },
        }

    def test_attachment_preserves_points_ranges_and_clock_on_success_and_failure(self):
        original = copy.deepcopy(self.air_window)
        current = {"epoch": self.issue, "pm02": 40.0, "rhum": 75.0}
        with patch.object(self.dashboard.near_term_hazard, "predict_hazard",
                          return_value=self.risk) as predict:
            attached = self.dashboard.attach_experimental_rapid_change_risk(
                self.air_window, self.issue, current
            )
        predict.assert_called_once()
        self.assertEqual(predict.call_args.args[2], self.issue)
        self.assertEqual(predict.call_args.kwargs["current"], current)
        self.assertEqual(self.air_window, original)
        self.assertEqual(attached["arrival"], original["arrival"])
        self.assertEqual(attached["trail"], original["trail"])
        self.assertEqual(attached["forecastClock"], original["forecastClock"])
        self.assertEqual(attached["experimentalRapidChangeRisk"], self.risk)
        with patch.object(self.dashboard.near_term_hazard, "predict_hazard",
                          side_effect=RuntimeError("test failure")):
            failed = self.dashboard.attach_experimental_rapid_change_risk(
                self.air_window, self.issue, current
            )
        self.assertEqual(failed["arrival"], original["arrival"])
        self.assertEqual(failed["trail"], original["trail"])
        self.assertFalse(failed["experimentalRapidChangeRisk"]["available"])
        self.assertFalse(failed["experimentalRapidChangeRisk"]["directionAvailable"])
        self.assertEqual(failed["experimentalRapidChangeRisk"]["targets"]["arrival90"]["targetEpoch"],
                         self.clock["arrivalTargetEpoch"])

    def test_published_archive_keeps_risk_with_unchanged_forecast(self):
        air_window = {**self.air_window, "experimentalRapidChangeRisk": self.risk}
        result = {
            "available": True, "forecastIssuedEpoch": self.issue,
            "current": {"epoch": self.issue}, "airWindow": air_window,
            "comparison": {}, "forecastPolicy": {}, "dataCoverage": {},
            "rainLearning": {}, "windows": {},
        }
        with sqlite3.connect(":memory:") as conn:
            inserted = record_dashboard_issue(conn, result, "test-build", self.issue)
            archived = loads(conn.execute(
                "SELECT payload FROM dashboard_forecast_issues"
            ).fetchone()[0])
        self.assertTrue(inserted)
        self.assertEqual(archived["airWindow"]["experimentalRapidChangeRisk"], self.risk)
        self.assertEqual(archived["airWindow"]["arrival"]["point"], 44.0)
        self.assertEqual(archived["airWindow"]["trail"]["rawRangeHigh"], 65.0)


if __name__ == "__main__":
    unittest.main()
