"""Pure dashboard seams imported with database/network/startup calls blocked."""

from contextlib import ExitStack, contextmanager
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))



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
