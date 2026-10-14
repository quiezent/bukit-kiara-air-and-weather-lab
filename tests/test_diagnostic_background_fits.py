"""Forecast requests cannot fit or block behind background model fitting."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import near_term_hazard as near
import short_horizon_change as short


class BackgroundDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.issue = int(datetime(2026, 10, 8, 12, 5, 37, tzinfo=near.MYT).timestamp())
        self.anchor = self.issue // 900 * 900
        self.cutoff = int(datetime(2026, 10, 8, tzinfo=near.MYT).timestamp())
        self.window = {"forecastClock": {"targetVersion": near.TARGET_VERSION,
            "forecastIssuedEpoch": self.issue, "featureAnchorEpoch": self.anchor,
            "arrivalTargetEpoch": self.issue + 5400, "windowStartEpoch": self.issue + 5400,
            "windowEndEpoch": self.issue + 12600},
            "arrival": {"point": 100, "pointRole": "persistence_anchor"},
            "trail": {"point": 100, "pointRole": "persistence_anchor"}}
        self.rows = [{"epoch": epoch, "pm02": 100}
                     for epoch in range(self.issue - 14400, self.issue + 1, 60)]

    def test_cold_forecast_returns_collecting_without_training_reads_or_fits(self):
        for module in (near, short):
            with self.subTest(module=module.__name__), patch.object(module, "_FITS", {}), \
                    patch.object(module, "_daily_fit", side_effect=AssertionError("request fit")), \
                    patch.object(module, "_training_rows", side_effect=AssertionError("request training read")):
                if module is near:
                    result = module.predict_hazard("unused.db", self.window, self.issue)
                else:
                    result = module.predict_short_horizon("unused.db", self.rows, self.issue)
                self.assertEqual(result["reason"], "background_model_collecting", result)
                self.assertFalse(result["available"])

    def test_cached_binary_probability_keeps_raw_flag_as_diagnostic_only(self):
        estimator = Mock()
        estimator.predict_proba.return_value = np.array([[.3, .7]])
        fit = {"available": True, "model": estimator, "active": np.ones(len(near.FEATURES), dtype=bool),
               "trainingCount": 150, "trainingEvents": 20,
               "lastTrainingIssueEpoch": self.cutoff - 7200,
               "lastTrainingOutcomeCompleteEpoch": self.cutoff - 1}
        cache = {(str(Path("unused.db").resolve()), self.cutoff):
                 {"arrival90": fit, "mean90to210": fit}}
        with patch.object(near, "_FITS", cache), \
                patch.object(near, "_daily_fit", side_effect=AssertionError("request fit")):
            result = near.predict_hazard("unused.db", self.window, self.issue)
        self.assertTrue(result["available"])
        for target in result["targets"].values():
            self.assertEqual(target["probability"], .7)
            self.assertTrue(target["diagnosticFlag"])
            self.assertFalse(target["flag"])
            self.assertFalse(target["operationalUseEligible"])

    def test_background_training_does_not_hold_reader_cache_lock(self):
        for module in (near, short):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fixture.sqlite3"
                with closing(sqlite3.connect(path)):
                    pass
                entered = threading.Event()
                release = threading.Event()
                errors = []

                def paused_rows(*_):
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError("reader blocked by background fit")
                    return ({"arrival90": [], "mean90to210": []} if module is near
                            else {15: [], 30: []})

                def run_fit():
                    try:
                        module.refresh_model(path, self.issue)
                    except Exception as error:
                        errors.append(error)

                with patch.object(module, "_FITS", {}), patch.object(module, "_training_rows", side_effect=paused_rows):
                    worker = threading.Thread(target=run_fit)
                    worker.start()
                    self.assertTrue(entered.wait(1))
                    reader = threading.Thread(target=lambda: module._cached_daily_fit(path, self.cutoff))
                    reader.start()
                    reader.join(.5)
                    was_blocked = reader.is_alive()
                    release.set()
                    worker.join(2)
                    reader.join(2)
                    self.assertFalse(was_blocked)
                    self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()
