"""CAMS snapshots must be selected at the actual issue clock without future leakage."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import cams_local_correction as correction


def run(level):
    epochs = np.arange(-3600, 18001, 900, dtype=float)
    return {"pm2_5": (epochs, np.full(len(epochs), level, dtype=float))}


class CamsIssueClockTests(unittest.TestCase):
    def test_live_run_arriving_after_closed_sensor_bucket_is_usable(self):
        origins = pd.to_datetime([0, 900], unit="s", utc=True)
        runs = [(-100, run(20)), (1000, run(30))]
        historical = correction._regional_features(origins, 90, runs)
        issued = correction._regional_features(origins, 90, runs, live_issue_epoch=1100)
        self.assertEqual(historical.camsFetchedEpoch.tolist(), [-100, -100])
        self.assertEqual(issued.camsFetchedEpoch.tolist(), [-100, 1000])
        self.assertEqual(issued.regionalMean.tolist(), [20, 30])
        self.assertEqual(issued.regionalCurrent.tolist(), [20, 30])

    def test_live_replay_does_not_rewrite_historical_source_vintages(self):
        origins = pd.to_datetime([0, 900], unit="s", utc=True)
        frame = pd.DataFrame({"pm02": [40., 41.], "atmp": [29., 29.],
                              "rhum": [60., 60.]}, index=origins)
        runs = [(-100, run(20)), (1000, run(30))]
        records = correction.replay(frame, 90, runs, live_issue_epoch=1100)
        self.assertEqual([r["camsFetchedEpoch"] for r in records], [-100, 1000])
        self.assertEqual([r["regionalMean"] for r in records], [20, 30])
        old = correction.replay(frame, 90, runs)
        self.assertEqual([r["camsFetchedEpoch"] for r in old], [-100, -100])

    def test_predict_reads_and_keys_archive_at_actual_issue_time(self):
        frame = pd.DataFrame({"pm02": [41.], "atmp": [29.], "rhum": [60.]},
                             index=pd.to_datetime([900], unit="s", utc=True))
        rows = [{"epoch": 1000, "pm02": 41.}]
        windows = {"morning": {"startEpoch": 6300, "endEpoch": 13500}}
        runs = [(-100, run(20)), (1000, run(30))]
        with patch.object(correction, "_CACHE", correction.OrderedDict()), \
             patch.object(correction.support, "_frame", return_value=frame), \
             patch.object(correction, "_archive_revision", return_value="issue-run") as revision, \
             patch.object(correction.support, "_read_runs", return_value=runs) as read_runs, \
             patch.object(correction, "replay", return_value=[]) as replay:
            result = correction.predict_windows("issue-clock-test", rows, windows, 1100)
        revision.assert_called_once_with("issue-clock-test", 1100)
        read_runs.assert_called_once_with("issue-clock-test", "air_quality_forecast_runs", 1100)
        self.assertEqual(replay.call_args.kwargs["live_issue_epoch"], 1100)
        self.assertEqual(result["morning"]["reason"], "missing_regional_sensor_overlap")

    def test_archive_retains_hour_needed_to_interpolate_closed_bucket(self):
        spec = importlib.util.spec_from_file_location(
            "cams_archive_test_dashboard", APP / "BukitKiara_Dashboard.py")
        dashboard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dashboard)
        hour = int(time.time() // 3600 * 3600)
        fetched = hour + 60
        payload = {"fetchedEpoch": fetched, "fetchedTimestamp": "test-issued-run",
                   "source": "Open-Meteo / CAMS Global",
                   "hourlyUnits": {"pm2_5": "µg/m³"},
                   "hourly": [{"epoch": hour + h * 3600, "pm2_5": float(13 + h)}
                              for h in range(-2, 5)]}
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE air_quality_forecast_runs "
                         "(fetched_epoch INTEGER PRIMARY KEY, fetched_timestamp TEXT, "
                         "source TEXT, model_version TEXT, payload TEXT)")
            # The archive write now also records an immutable source receipt.
            dashboard.observation_provenance.init_schema(conn)

            @contextmanager
            def memory_db():
                yield conn
                conn.commit()

            with patch.object(dashboard, "db", memory_db):
                dashboard.save_air_quality_forecast(payload)
            archived = json.loads(conn.execute(
                "SELECT payload FROM air_quality_forecast_runs").fetchone()[0])
            self.assertEqual(archived["archiveContextHours"], 2)
            self.assertEqual(archived["hourly"][0]["epoch"], hour - 3600)
            points = archived["hourly"]
            arrays = {"pm2_5": (np.array([p["epoch"] for p in points], dtype=float),
                                  np.array([p["pm2_5"] for p in points], dtype=float))}
            origin = pd.to_datetime([hour], unit="s", utc=True)
            features = correction._regional_features(
                origin, 90, [(fetched, arrays)], live_issue_epoch=fetched)
            self.assertTrue(np.isfinite(features.regionalCurrent.iloc[0]))
            self.assertEqual(features.camsFetchedEpoch.iloc[0], fetched)
        finally:
            conn.close()

    def test_bad_unit_refresh_preserves_last_valid_cached_run(self):
        spec = importlib.util.spec_from_file_location(
            "cams_refresh_test_dashboard", APP / "BukitKiara_Dashboard.py")
        dashboard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dashboard)
        cached = {"fetchedEpoch": int(time.time()) - 600, "hourly": [{"pm2_5": 30.}]}
        malformed = {"hourly_units": {"pm2_5": "unexpected"},
                     "hourly": {"time": ["2026-09-29T14:00"], "pm2_5": [25.]}}
        with patch.object(dashboard, "weather_status",
                          {"air_quality": cached, "air_quality_error": None}), \
             patch.object(dashboard, "fetch_json", return_value=malformed), \
             patch.object(dashboard, "save_air_quality_forecast") as save:
            self.assertIsNone(dashboard.fetch_air_quality_forecast())
            self.assertIs(dashboard.weather_status["air_quality"], cached)
            self.assertIn("unexpected units", dashboard.weather_status["air_quality_error"])
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
