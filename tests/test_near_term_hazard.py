"""Clock, cutoff, and cache checks for the experimental rapid-change risk."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import near_term_hazard as hazard
from forecast_clock import ForecastClock, point_weights


MYT = timezone(timedelta(hours=8))


def epoch(day, hour, minute=0, second=0):
    return int(datetime(2026, 9, day, hour, minute, second, tzinfo=MYT).timestamp())


def window(issue, anchor, fresh=50.0):
    clock = ForecastClock(issued_epoch=issue, anchor_epoch=anchor).metadata()
    return {
        "forecastClock": clock, "change30": -4.0, "change60": -6.0,
        "fastSlowGap": -1.0, "pm10ChangePct30": -8.0,
        "particleMix": {"fineShare": 89.0, "coarseParticles": 4.0},
        "arrival": {"point": fresh, "pointRole": "persistence_anchor"},
        "trail": {"point": fresh, "pointRole": "persistence_anchor"},
    }


class NearTermHazardTests(unittest.TestCase):
    def test_actual_uses_exact_issue_lag_and_partial_bucket_weights(self):
        issue = epoch(29, 20, 29, 23)
        anchor = epoch(29, 20, 15)
        case = hazard._clock_and_features(window(issue, anchor), {"rhum": 60}, issue)
        self.assertEqual(case["target"], issue + 90 * 60)
        self.assertEqual(case["pointComplete"], epoch(29, 22, 0))
        weights = point_weights(case["target"] - anchor)
        buckets = {anchor + offset * 900: value for offset, value in
                   zip(weights, (40.0, 80.0))}
        expected = sum(buckets[anchor + offset * 900] * weight
                       for offset, weight in weights.items())
        self.assertAlmostEqual(hazard._actual(buckets, case, "arrival90"), expected)
        self.assertNotEqual(expected, 80.0)

    def test_training_excludes_target_finishing_after_midnight(self):
        issue = epoch(29, 21, 30)
        anchor = epoch(29, 21, 15)
        air = window(issue, anchor)
        clock = air["forecastClock"]
        midnight = epoch(30, 0)
        self.assertLess(clock["arrivalTargetEpoch"], midnight)
        self.assertGreater(clock["observedOutcomeCompleteAfterEpoch"], midnight)
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.execute("CREATE TABLE dashboard_forecast_issues (issued_epoch INTEGER, payload TEXT)")
        payload = json.dumps({"airWindow": air, "current": {"rhum": 55}})
        conn.execute("INSERT INTO dashboard_forecast_issues VALUES (?,?)", (issue + 30, payload))
        # Even a future archive with a completed-looking response is outside
        # this fit's issue-day boundary.
        later = epoch(30, 0, 30)
        conn.execute("INSERT INTO dashboard_forecast_issues VALUES (?,?)",
                     (later + 30, json.dumps({"airWindow": window(later, epoch(30, 0, 15))})))
        buckets = {clock["arrivalTargetEpoch"]: 75.0}
        with patch.object(hazard, "_closed_buckets", return_value=buckets) as read:
            rows = hazard._training_rows(conn, midnight)
        self.assertEqual(read.call_args.args[2], midnight)
        self.assertEqual(len(rows["arrival90"]), 1)
        self.assertEqual(rows["arrival90"][0][1], 1)
        self.assertEqual(len(rows["mean90to210"]), 0)

    def test_second_issue_same_day_reuses_completed_daily_fit(self):
        hazard._FITS.clear()
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "small.db"
            sqlite3.connect(db_path).close()
            midnight = epoch(29, 0)
            with patch.object(hazard, "_training_rows", return_value={"arrival90": [],
                                                                      "mean90to210": []}) as train:
                first = hazard._daily_fit(db_path, midnight)
                second = hazard._daily_fit(db_path, midnight)
            self.assertIs(first, second)
            self.assertEqual(train.call_count, 1)


if __name__ == "__main__":
    unittest.main()
