"""Causal issue-vintage and interval checks for session weather features."""

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

import rain_weather_features as rain
import weather_session_features as features


KL = timezone(timedelta(hours=8))
SOURCE = "Open-Meteo Best Match"


def epoch(clock):
    return int(datetime.strptime(clock, "%Y-%m-%d %H:%M").replace(tzinfo=KL).timestamp())


def payload(fetched, *, first_hour_rain=1.0, second_hour_rain=1.0,
            wind_direction=90.0, omit_optional=False, bad_rain_unit=False,
            direction_by_hour=None):
    units = {
        "precipitation": "cm" if bad_rain_unit else "mm",
        "precipitation_probability": "%",
        "temperature_2m": "°C", "relative_humidity_2m": "%",
        "wind_speed_10m": "km/h", "wind_direction_10m": "°",
        "wind_speed_925hPa": "km/h", "wind_direction_925hPa": "°",
    }
    if not omit_optional:
        units.update({"wind_gusts_10m": "km/h", "wind_speed_180m": "km/h",
                      "boundary_layer_height": "m"})
    hourly = []
    start = epoch("2026-09-30 09:00")
    for index in range(10):
        hour = 9 + index
        direction = (direction_by_hour or {}).get(hour, wind_direction)
        row = {
            "epoch": start + index * 3600,
            "precipitation": (first_hour_rain if hour == 15 else
                              second_hour_rain if hour == 16 else 0.0),
            "precipitation_probability": 100.0 if hour in (15, 16) else 10.0,
            "temperature_2m": 30.0, "relative_humidity_2m": 70.0,
            "wind_speed_10m": 10.0, "wind_direction_10m": direction,
            "wind_speed_925hPa": 20.0, "wind_direction_925hPa": direction,
            "wind_gusts_10m": 25.0, "wind_speed_180m": 15.0,
            "boundary_layer_height": 700.0,
        }
        hourly.append(row)
    return json.dumps({"source": SOURCE, "fetchedEpoch": fetched,
                       "latitude": 3.1411, "longitude": 101.6275,
                       "hourlyUnits": units, "hourly": hourly})


class SessionWeatherFeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "weather.db"
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("CREATE TABLE weather_forecast_runs "
                               "(fetched_epoch INTEGER, source TEXT, payload TEXT)")
        self.start = epoch("2026-09-30 14:00")
        self.end = epoch("2026-09-30 16:00")

    def add_run(self, fetched, body):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO weather_forecast_runs VALUES (?,?,?)",
                               (fetched, SOURCE, body))

    def describe(self, issue, archive_cutoff=None):
        archive = rain.load_runs(self.db, archive_cutoff or issue)
        return features.describe_window(archive, issue, self.start, self.end)

    def test_latest_vintage_at_exact_issue_changes_rain_and_wind(self):
        first = epoch("2026-09-30 11:50")
        second = epoch("2026-09-30 12:15")
        self.add_run(first, payload(first, first_hour_rain=1, second_hour_rain=1,
                                    wind_direction=90))
        self.add_run(second, payload(second, first_hour_rain=3, second_hour_rain=3,
                                     wind_direction=270))
        before = self.describe(epoch("2026-09-30 12:14"), second)
        after = self.describe(second, second)
        self.assertTrue(before["available"])
        self.assertTrue(after["available"])
        self.assertEqual(before["fetchedEpoch"], first)
        self.assertEqual(after["fetchedEpoch"], second)
        self.assertEqual(before["featureValues"]["duringRainMm"], 2.0)
        self.assertEqual(after["featureValues"]["duringRainMm"], 6.0)
        self.assertLess(before["featureValues"]["wind10UMeanKmh"], -9.9)
        self.assertGreater(after["featureValues"]["wind10UMeanKmh"], 9.9)
        self.assertLessEqual(before["fetchedEpoch"], before["metadata"]["forecastIssuedEpoch"])
        self.assertLessEqual(after["fetchedEpoch"], after["metadata"]["forecastIssuedEpoch"])

    def test_latest_invalid_run_does_not_fall_back_to_older_valid_run(self):
        good = epoch("2026-09-30 11:50")
        bad = epoch("2026-09-30 11:59")
        self.add_run(good, payload(good))
        self.add_run(bad, payload(bad, bad_rain_unit=True))
        result = self.describe(epoch("2026-09-30 12:00"))
        self.assertFalse(result["available"])
        self.assertEqual(result["fetchedEpoch"], bad)
        self.assertIn("invalid_unit:precipitation", result["metadata"]["errors"])
        self.assertIsNone(result["featureValues"]["duringRainMm"])
        self.assertIsNone(result["featureValues"]["wind10UMeanKmh"])

    def test_missing_or_stale_run_does_not_invent_dry_weather(self):
        issue = epoch("2026-09-30 12:00")
        missing = self.describe(issue)
        self.assertFalse(missing["available"])
        self.assertIn("no_run_at_issue", missing["metadata"]["errors"])
        self.assertIsNone(missing["featureValues"]["duringRainMm"])
        self.assertIsNone(missing["featureValues"]["wind10SpeedMeanKmh"])
        old = issue - 7201
        self.add_run(old, payload(old))
        stale = self.describe(issue)
        self.assertFalse(stale["available"])
        self.assertEqual(stale["fetchedEpoch"], old)
        self.assertIn("stale_weather_run", stale["metadata"]["errors"])
        self.assertIsNone(stale["featureValues"]["duringRainMm"])

    def test_equal_rain_totals_retain_different_session_accumulation(self):
        first = epoch("2026-09-30 11:50")
        second = epoch("2026-09-30 12:15")
        self.add_run(first, payload(first, first_hour_rain=2, second_hour_rain=0))
        self.add_run(second, payload(second, first_hour_rain=0, second_hour_rain=2))
        early = self.describe(epoch("2026-09-30 12:00"), second)
        late = self.describe(second, second)
        self.assertEqual(early["featureValues"]["duringRainMm"], 2.0)
        self.assertEqual(late["featureValues"]["duringRainMm"], 2.0)
        self.assertAlmostEqual(early["featureValues"]["duringAccumulationMeanMm"], 1.5)
        self.assertAlmostEqual(late["featureValues"]["duringAccumulationMeanMm"], 0.5)

    def test_optional_channels_stay_missing_and_direction_wraps_as_vector(self):
        issue = epoch("2026-09-30 12:00")
        self.add_run(issue, payload(issue, omit_optional=True,
                                    direction_by_hour={14: 350, 15: 10, 16: 10}))
        result = self.describe(issue)
        self.assertTrue(result["available"])
        self.assertIsNone(result["featureValues"]["gustHourlyEnvelopeMaxKmh"])
        self.assertIsNone(result["featureValues"]["boundaryLayerMeanM"])
        self.assertIsNone(result["featureValues"]["ventilationMeanM2S"])
        self.assertIn("wind_gusts_10m", result["metadata"]["missingOptionalUnits"])
        self.assertLess(result["featureValues"]["wind10VMeanKmh"], -9)
        self.assertLess(abs(result["featureValues"]["wind10UMeanKmh"]), 1)


if __name__ == "__main__":
    unittest.main()
