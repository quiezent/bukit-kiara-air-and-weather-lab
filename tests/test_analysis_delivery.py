"""Forecast publication stays current with live diagnostics suspended."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import importlib.util
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from analysis_delivery import AnalysisDelivery


def dashboard_module():
    spec = importlib.util.spec_from_file_location(
        "dashboard_delivery_tests", APP / "BukitKiara_Dashboard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class AnalysisDeliveryTests(unittest.TestCase):
    def test_live_worker_skips_diagnostics_across_primary_refreshes(self):
        dashboard = dashboard_module()
        now = [1000]
        delivery = AnalysisDelivery(clock=lambda: now[0])
        stop = threading.Event()

        def compute(days, *, include_trials, trial_context=None):
            self.assertFalse(include_trials)
            self.assertIsNone(trial_context)
            issued = now[0]
            return {"available": True, "forecastIssuedEpoch": issued,
                    "current": {"epoch": issued - 5}, "windows": {}}

        # The worker's session-issue writers target DB_PATH; keep them off the
        # real database.
        with patch.object(dashboard, "analysis_delivery", delivery), \
             patch.object(dashboard, "_compute_analysis", side_effect=compute), \
             patch.object(dashboard, "enrich_analysis_trials",
                          side_effect=AssertionError("live diagnostics must not run")), \
             patch.object(dashboard, "archive_published_analysis") as archive, \
             patch.object(dashboard.window_pm_predictor, "record_issue"), \
             patch.object(dashboard.weather_session_forecast, "record_issue"), \
             patch.object(dashboard.afternoon_xgboost_forecast, "record_issue"):
            primary = threading.Thread(target=dashboard.analysis_worker,
                                       args=(stop,), daemon=True)
            primary.start()
            try:
                self.assertTrue(wait_until(
                    lambda: delivery.response(28).get("forecastIssuedEpoch") == 1000
                ))
                self.assertEqual(delivery.response(28)["delivery"]["diagnostics"], "skipped")
                now[0] = 1061
                delivery.schedule(28)
                self.assertTrue(wait_until(
                    lambda: (delivery.response(28).get("forecastIssuedEpoch") == 1061)
                ))
                self.assertEqual(delivery.response(28)["delivery"]["diagnostics"], "skipped")
                self.assertEqual(archive.call_count, 2)
            finally:
                stop.set()
                delivery.wake.set()
                primary.join(3)
            self.assertFalse(primary.is_alive())


if __name__ == "__main__":
    unittest.main()
