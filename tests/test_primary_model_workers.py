"""Primary cache preparation must progress independently of optional fits.

Only refresh seams and logging are patched. Real worker loops run in bounded
threads; no model fitting, database access, provider calls or service control.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import threading
import unittest
from unittest.mock import patch

import BukitKiara_Dashboard as dashboard


class FastRetryStop(threading.Event):
    """Record the production delay while making a retry fast in a test."""
    def __init__(self):
        super().__init__()
        self.delays = []

    def wait(self, timeout=None):
        self.delays.append(timeout)
        return super().wait(.005 if timeout is not None else .2)


def start_worker(target, stop, failures):
    def run():
        try:
            target(stop)
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class PrimaryModelWorkerTests(unittest.TestCase):
    def test_blocked_optional_fit_cannot_hold_up_primary_morning_cache(self):
        diagnostic_stop = threading.Event()
        morning_stop = threading.Event()
        diagnostic_entered = threading.Event()
        release_diagnostic = threading.Event()
        diagnostic_finished = threading.Event()
        morning_ready = threading.Event()
        failures = []
        threads = []

        def blocked_diagnostic(*args, **kwargs):
            diagnostic_entered.set()
            if not release_diagnostic.wait(.6):
                raise AssertionError("The optional fit was not released during cleanup")
            diagnostic_finished.set()
            return {"state": "ready"}

        def capture_log(*args, **kwargs):
            message = str(args[0]) if args else ""
            if message.startswith("[Morning model]") and "'ready'" in message:
                morning_ready.set()

        with patch.object(dashboard.near_term_hazard, "refresh_model", side_effect=blocked_diagnostic), \
                patch.object(dashboard.short_horizon_change, "refresh_model", return_value={"state": "ready"}), \
                patch.object(dashboard.weather_session_forecast, "refresh_model",
                             return_value={"state": "ready", "trainingCutoffEpoch": 1}) as morning_refresh, \
                patch("builtins.print", side_effect=capture_log):
            try:
                threads.append(start_worker(dashboard.near_diagnostic_model_worker,
                                            diagnostic_stop, failures))
                self.assertTrue(diagnostic_entered.wait(.2), "Optional refresh did not start")
                threads.append(start_worker(dashboard.morning_model_worker, morning_stop, failures))
                self.assertTrue(morning_ready.wait(.2), "Morning preparation waited behind an optional fit")
                self.assertFalse(release_diagnostic.is_set())
                self.assertFalse(diagnostic_finished.is_set())
                self.assertEqual(morning_refresh.call_count, 1)
            finally:
                morning_stop.set()
                diagnostic_stop.set()
                release_diagnostic.set()
                for thread in threads:
                    thread.join(.2)
            self.assertTrue(diagnostic_finished.is_set())
            self.assertEqual(morning_refresh.call_count, 1,
                             "The optional worker must not also prepare the primary cache")
        self.assertEqual(failures, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))

    def test_primary_refresh_failure_retries_and_stops_cleanly(self):
        # refresh_model normally returns an unavailable status; an unexpected
        # exception must also leave the independent worker able to retry.
        for first_result in (RuntimeError("synthetic preparation failure"),
                             {"state": "unavailable", "reason": "synthetic cache unavailable"}):
            with self.subTest(first_result=first_result):
                stop = FastRetryStop()
                failures = []
                logs = []
                calls = []

                def refresh(*args, **kwargs):
                    calls.append((args, kwargs))
                    if len(calls) == 1:
                        if isinstance(first_result, Exception):
                            raise first_result
                        return first_result
                    stop.set()
                    return {"state": "ready", "trainingCutoffEpoch": 1}

                with patch.object(dashboard.weather_session_forecast, "refresh_model", side_effect=refresh), \
                        patch("builtins.print", side_effect=lambda *args, **kwargs: logs.append(str(args[0]))):
                    thread = start_worker(dashboard.morning_model_worker, stop, failures)
                    try:
                        thread.join(.2)
                    finally:
                        stop.set()
                        thread.join(.2)
                self.assertFalse(thread.is_alive(), "Stopping must interrupt the scheduled wait")
                self.assertEqual(failures, [])
                self.assertEqual(len(calls), 2)
                self.assertEqual(stop.delays, [60, 60])
                self.assertTrue(any("'ready'" in message for message in logs))
                self.assertTrue(any("synthetic" in message for message in logs))

    def test_stopped_primary_worker_never_starts_a_refresh(self):
        stop = threading.Event()
        stop.set()
        with patch.object(dashboard.weather_session_forecast, "refresh_model") as refresh:
            dashboard.morning_model_worker(stop)
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
