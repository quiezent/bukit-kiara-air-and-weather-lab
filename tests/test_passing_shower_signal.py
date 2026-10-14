"""Regression: a rain-supported washout confirmed during a dry forming signal.

On 2026-10-10 13:15-13:17 MYT the live analysis failed five times with
"unsupported format string passed to NoneType.__format__": the forming
signal (rapid_joint, not moist cooling) carries humidityRise30=None into the
confirmed event's metrics, and the rain-supported details text formatted it.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import unittest
from unittest.mock import patch

import BukitKiara_Dashboard as app


LATEST = 1791609446  # 2026-10-10 05:17:26 UTC, 13:17 MYT (live low reading)
DROP_START = LATEST - 8 * 180

# Subang evidence as recorded for that event (sanitized to the fields used).
SUBANG = {
    "baselineEpoch": 1791604800, "currentEpoch": 1791608400,
    "windSpeedKmh": 11.16, "windCompass": "WSW", "windSpeedChange": 7.6,
    "humidityChange": 7.0, "dewpointChange": -0.4, "temperatureChange": -2.1,
    "visibilityKm": 6.0, "precipitationMm": None,
    "presentWeather": "PRECIPITATION WITHIN SIGHT, REACHING THE GROUND OR "
                      "THE SURFACE OF THE SEA, BUT DISTANT",
}


def rows():
    readings = []
    for index in range(61):
        epoch = LATEST - 180 * (60 - index)
        low = epoch >= DROP_START
        readings.append({
            "epoch": epoch, "pm02": 55.0 if low else 86.0,
            "pm10": 68.0 if low else 95.0, "pm003Count": 9000.0,
            "atmp": 29.7 if low else 31.0, "rhum": 65.0 if low else 60.0,
            "heatindex": 33.5 if low else 37.0,
        })
    return readings


def forming(humidity_rise):
    moist = humidity_rise is not None
    return {
        "active": True, "state": "forming",
        "kind": ("moist_cooling_particle_clearing_forming" if moist
                 else "dry_clearing_forming"),
        "label": "Dry clearing forming", "details": "",
        "detectedEpoch": DROP_START, "lastEvidenceEpoch": LATEST,
        "ageMinutes": 24, "rainSupport": False, "dryAirMassSupport": False,
        "metrics": {
            "detectionMode": "moist_cooling_joint" if moist else "rapid_joint",
            "earlyJointShock": True, "rapidDetectedEpoch": DROP_START,
            "rapidParticleSupport": False, "rapidGasSupport": True,
            "rapidMoistCoolingSupport": moist, "temperatureDrop30": 0.6,
            "heatIndexDrop30": 1.0, "humidityRise30": humidity_rise,
        },
        "timeline": {"onsetEpoch": DROP_START},
    }


class RainSupportedWashoutDetailsTests(unittest.TestCase):
    def signal(self, humidity_rise):
        with patch.object(app, "fast_dry_clearing_signal", return_value=forming(humidity_rise)), \
                patch.object(app, "subang_event_context", return_value=dict(SUBANG)), \
                patch.object(app, "issued_weather_point", return_value={}):
            return app.passing_shower_signal(rows())

    def test_missing_forming_humidity_rise_is_shown_as_unavailable(self):
        signal = self.signal(None)
        self.assertEqual(signal["kind"], "rain_supported_washout")
        self.assertIsNone(signal["metrics"]["humidityRise30"])
        self.assertIn("humidity rise unavailable", signal["details"])
        self.assertNotIn("None", signal["details"])

    def test_available_humidity_rise_is_unchanged(self):
        signal = self.signal(5.0)
        self.assertEqual(signal["kind"], "rain_supported_washout")
        self.assertEqual(signal["metrics"]["humidityRise30"], 5.0)
        self.assertIn("humidity ↑5%", signal["details"])


if __name__ == "__main__":
    unittest.main()
