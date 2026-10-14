"""Focused checks for rain-candidate reporting and residual cohorts."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import rain_pm_forecast as candidate


def epoch(stamp):
    return int(pd.Timestamp(stamp, tz=candidate.TZ).timestamp())


def record(issue, actual, prediction, wet):
    return {"issuedEpoch": issue, "completeAfterEpoch": issue + 3 * 3600,
            "day": str(pd.Timestamp(issue, unit="s", tz="UTC").tz_convert(candidate.TZ).date()),
            "actual": actual, "persistence": 50., "prediction": prediction,
            "highRain": wet, "wetForecast": wet}


class RainCandidateReportingTests(unittest.TestCase):
    def test_live_residuals_match_wet_or_dry_forecast(self):
        live_issue = epoch("2026-09-30 12:00")
        history = []
        for i in range(12):
            past_issue = live_issue - (i + 1) * 8 * 3600
            history.append(record(past_issue, 40., 50., True))   # residual -10
            history.append(record(past_issue, 70., 50., False))  # residual +20
        wet_summary, wet_quantiles = candidate._evidence_live(history, live_issue, True)
        dry_summary, dry_quantiles = candidate._evidence_live(history, live_issue, False)
        self.assertEqual(wet_summary["residualCohort"], "wet_forecast")
        self.assertEqual(dry_summary["residualCohort"], "dry_forecast")
        self.assertEqual(wet_summary["residualCount"], 12)
        self.assertEqual(dry_summary["residualCount"], 12)
        np.testing.assert_allclose(wet_quantiles, [-10., -10.])
        np.testing.assert_allclose(dry_quantiles, [20., 20.])

    def test_live_residuals_require_twelve_same_cohort_outcomes(self):
        live_issue = epoch("2026-09-30 12:00")
        history = [record(live_issue - (i + 1) * 8 * 3600, 70., 50., False)
                   for i in range(11)]
        history.append(record(live_issue - 8 * 3600, 40., 50., True))
        summary, quantiles = candidate._evidence_live(history, live_issue, False)
        self.assertEqual(summary["residualCount"], 11)
        self.assertIsNone(quantiles)

    def test_dry_prediction_uses_dry_residual_range(self):
        issue = epoch("2026-09-30 12:00")
        history = []
        for i in range(12):
            past_issue = issue - (i + 1) * 8 * 3600
            history.append(record(past_issue, 40., 50., True))
            history.append(record(past_issue, 70., 50., False))
        live = {**record(issue, None, 50., False), "freshReferenceCount": 0,
                "freshReferenceEpoch": None, "trainingCount": 48,
                "trainingWetForecastCount": 0, "trainingHighRainCount": 0,
                "trainingLastOutcomeCompleteEpoch": issue - 3600,
                "weatherFetchedEpoch": issue, "rainContext": {}, "rawLearnedDelta": 0.}
        records = history + [live]
        frame = pd.DataFrame({"pm02": [50.]},
                             index=pd.to_datetime([issue], unit="s", utc=True))
        rows = [{"epoch": issue, "pm02": 50., "atmp": 30., "rhum": 60.}]
        context = {"available": True, "fetchedEpoch": issue}
        candidate._CACHE.clear()
        with patch.object(candidate, "_frame", return_value=frame), \
             patch.object(candidate, "raw_arrays", return_value=(np.array([issue]), np.array([50.]))), \
             patch.object(candidate.weather, "load_runs", return_value="unused"), \
             patch.object(candidate.weather, "describe_window", return_value=context), \
             patch.object(candidate, "replay", return_value={"arrival": records, "mean": records}):
            output = candidate.predict_nearterm("unused.db", rows, issue)
        self.assertTrue(output["available"])
        self.assertFalse(output["appliedToPrimary"])
        self.assertEqual(output["arrival"]["validation"]["residualCohort"], "dry_forecast")
        self.assertEqual(output["arrival"]["rawRangeLow"], 70.)
        self.assertEqual(output["arrival"]["rawRangeHigh"], 70.)
        self.assertEqual(output["arrival"]["rangePolicyVersion"], "same_wet_dry_cohort_residual_v2")

    def test_completed_sep9_excludes_later_issues(self):
        sep9 = record(epoch("2026-09-09 12:00"), 40., 50., True)
        sep29 = record(epoch("2026-09-29 12:00"), 70., 50., False)
        asof = epoch("2026-09-30 00:00")
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "history.db"
            with closing(sqlite3.connect(db_path)) as connection:
                with connection:
                    connection.execute("CREATE TABLE readings (epoch INTEGER, pm02 REAL, atmp REAL, rhum REAL)")
                    connection.execute("INSERT INTO readings VALUES (?, 50, 30, 60)", (asof,))
            with patch.object(candidate, "_frame", return_value="unused"), \
                 patch.object(candidate.weather, "load_runs", return_value="unused"), \
                 patch.object(candidate, "replay", return_value={"arrival": [sep9, sep29],
                                                                "mean": [sep9, sep29]}), \
                 patch.object(candidate, "predict_nearterm", return_value={}):
                report = candidate.run_study(db_path)
        for block in report["offsets"].values():
            for horizon in ("arrival", "mean"):
                self.assertEqual(block[horizon]["completedSep9"]["overall"]["count"], 1)
                self.assertEqual(block[horizon]["allCompleted"]["overall"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
