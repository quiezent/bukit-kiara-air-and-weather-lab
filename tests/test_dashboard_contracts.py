"""Pure dashboard seams imported with database/network/startup calls blocked."""

from contextlib import ExitStack, contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from forecast_clock import ForecastClock


@contextmanager
def no_runtime_io():
    with ExitStack() as stack:
        for target in (
            "sqlite3.connect", "urllib.request.urlopen", "threading.Thread.start",
            "http.server.ThreadingHTTPServer", "webbrowser.open",
            "socket.socket.bind", "socket.socket.connect",
        ):
            stack.enter_context(patch(target, side_effect=AssertionError("Unexpected runtime I/O: " + target)))
        yield


class DashboardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Importing must never initialize storage, bind a port, or start collectors.
        with no_runtime_io():
            spec = importlib.util.spec_from_file_location("dashboard_public_tests", APP / "BukitKiara_Dashboard.py")
            cls.dashboard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.dashboard)

    def test_missing_peak_candidate_does_not_block_arrival_or_mean(self):
        clock = ForecastClock(issued_epoch=1_800_000_000, anchor_epoch=1_800_000_000).metadata()
        air_window = {
            "available": True, "current15": 20.0, "forecastClock": clock,
            "arrival": {"available": True, "point": 20.0},
            "trail": {"available": True, "point": 20.0, "projectedPeak": 27.0,
                      "peakUpper": 35.0, "decisionPeakUpper": 35.0},
        }
        analogue = {
            "featureAnchor": 20.0, "forecastClock": clock,
            "arrivalPoint": 12.0, "arrivalDelta": -8.0,
            "trailMeanPoint": 8.0, "trailMeanDelta": -12.0,
            "backtest": {"deploymentSkill": {
                key: {"eligible": True} for key in ("arrival", "trailMean", "trailPeak")
            }},
        }
        before = copy.deepcopy(air_window)
        with no_runtime_io():
            result = self.dashboard.aggressive_air_window_forecast(air_window, analogue)
        self.assertEqual(air_window, before)
        self.assertEqual(result["arrival"]["point"], 18.0)
        self.assertEqual(result["trail"]["point"], 17.0)
        self.assertTrue(result["arrival"]["pointApproximate"])
        self.assertTrue(result["trail"]["pointApproximate"])
        self.assertFalse(result["trail"]["peakApproximate"])
        self.assertEqual(result["trail"]["projectedPeak"], 27.0)
        self.assertEqual(result["trail"]["decisionPeakUpper"], 35.0)

    def test_cached_window_labels_use_original_issue_not_delivery_time(self):
        windows = {"morning": {}, "afternoon": {}}
        original_issue = 1_800_000_000
        with no_runtime_io():
            returned_issue = self.dashboard.apply_cached_forecast_issue(
                windows, {"forecastIssuedEpoch": original_issue}, original_issue + 3600,
            )
        self.assertEqual(returned_issue, original_issue)
        expected_label = f"Forecasted at {self.dashboard.local_dt(original_issue):%H:%M}"
        for window in windows.values():
            self.assertEqual(window["forecastIssuedEpoch"], original_issue)
            self.assertEqual(window["forecastedLabel"], expected_label)
            self.assertEqual(window["recheckLabel"], expected_label)


if __name__ == "__main__":
    unittest.main()
