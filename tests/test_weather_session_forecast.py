"""Frozen-issue evidence and archive boundaries for the session-only model."""

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
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import weather_session_forecast as forecast
import weather_session_model as learner


KL = timezone(timedelta(hours=8))


def epoch(clock):
    return int(datetime.strptime(clock, "%Y-%m-%d %H:%M").replace(tzinfo=KL).timestamp())


class IssuedWeatherSessionEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "issued.db"
        # A table without a uniqueness constraint also checks defensive
        # deduplication of imported/replayed rows for the same target date.
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("""CREATE TABLE window_pm_forecast_issues(
                model_version TEXT, origin_epoch INTEGER, start_epoch INTEGER,
                end_epoch INTEGER, issued_epoch INTEGER, window_key TEXT,
                predicted_mean REAL, anchor_pm25 REAL, payload TEXT)""")
        self.points = {}

    def insert(self, origin, start, *, point=48.0, anchor=52.0, actual=50.0,
               model_version=None, feature_version=None, training_end=None,
               training_cutoff=None, weather_fetched=None, publication_lag=10):
        issue = origin + 30
        end = start + 7200
        model_version = model_version or forecast.MODEL_VERSION
        feature_version = feature_version or learner.FEATURE_VERSION
        midnight = forecast.training_cutoff(issue)
        value = {
            "available": True, "modelVersion": model_version,
            "originEpoch": origin, "startEpoch": start, "endEpoch": end,
            "forecastedAtEpoch": issue, "mean": point,
            "sensorAnchor": anchor,
            "weatherFetchedEpoch": (issue - 60 if weather_fetched is None
                                    else weather_fetched),
            "weatherSessionModel": {
                "applied": True, "featureVersion": feature_version,
                "trainingCutoffEpoch": (midnight if training_cutoff is None
                                        else training_cutoff),
                "trainingLatestTargetEndEpoch": (midnight - 1 if training_end is None
                                                 else training_end),
            },
        }
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO window_pm_forecast_issues VALUES (?,?,?,?,?,?,?,?,?)",
                         (model_version, origin, start, end, issue + publication_lag,
                          "afternoon", point, anchor, json.dumps(value)))
        for index in range(1, 9):
            self.points[start + index * 900] = actual

    def frame(self):
        index = pd.to_datetime(sorted(self.points), unit="s", utc=True).tz_convert(
            "Asia/Kuala_Lumpur")
        return pd.DataFrame({"pm02": [self.points[int(stamp.timestamp())]
                                        for stamp in index]}, index=index)

    def evidence(self, current_origin, current_start, *, current_issue=None):
        issue = current_origin + 30 if current_issue is None else current_issue
        return forecast.issued_evidence(self.db, self.frame(), "afternoon",
                                        current_start, current_start + 7200,
                                        current_origin, issue, learner.FEATURE_VERSION)

    def test_exact_clock_scores_one_completed_target_date_once(self):
        previous = epoch("2026-09-29 07:00")
        target = epoch("2026-09-29 14:00")
        self.insert(previous, target)
        self.insert(previous, target, point=5.0)  # same target date, duplicate
        self.insert(previous, target, point=0.0, model_version="older_session_model")
        current = epoch("2026-09-30 07:00")
        evidence, errors = self.evidence(current, epoch("2026-09-30 14:00"))
        self.assertEqual(evidence["count"], 1)
        self.assertEqual(evidence["distinctDays"], 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(evidence["mae"], 2.0)
        self.assertEqual(evidence["persistenceMae"], 2.0)
        self.assertFalse(evidence["supportSufficient"])

    def test_target_ending_at_issue_is_not_training_evidence(self):
        previous = epoch("2026-09-29 00:00")
        self.insert(previous, epoch("2026-09-29 22:00"))
        current = epoch("2026-09-30 00:00")
        start = epoch("2026-09-30 22:00")
        at_completion, _ = self.evidence(current, start, current_issue=current)
        after_completion, _ = self.evidence(current, start, current_issue=current + 30)
        self.assertEqual(at_completion["count"], 0)
        self.assertEqual(after_completion["count"], 1)

    def test_invalid_vintage_training_cutoff_and_publication_are_excluded(self):
        current = epoch("2026-09-30 07:00")
        for days_back in (1, 2, 3, 4, 5):
            old_origin = current - days_back * 86400
            old_start = epoch("2026-09-30 14:00") - days_back * 86400
            issue = old_origin + 30
            extra = (
                {} if days_back == 1 else
                {"weather_fetched": issue - 7201} if days_back == 2 else
                {"training_end": issue} if days_back == 3 else
                {"publication_lag": 121} if days_back == 4 else
                {"feature_version": "previous_weather_features"}
            )
            self.insert(old_origin, old_start, **extra)
        evidence, errors = self.evidence(current, epoch("2026-09-30 14:00"))
        self.assertEqual(evidence["count"], 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(evidence["excludedPolicyOrClockCount"], 4)

    def test_post_midnight_training_label_and_wrong_daily_cutoff_are_excluded(self):
        current = epoch("2026-09-30 07:00")
        previous = epoch("2026-09-29 07:00")
        prior_midnight = forecast.training_cutoff(previous + 30)
        # This target finished before the 07:00 issue, but after the daily
        # training freeze. An issue-time-only check would incorrectly admit it.
        self.insert(previous, epoch("2026-09-29 14:00"),
                    training_end=prior_midnight + 900)
        older = epoch("2026-09-28 07:00")
        older_midnight = forecast.training_cutoff(older + 30)
        self.insert(older, epoch("2026-09-28 14:00"),
                    training_cutoff=older_midnight - 86400,
                    training_end=older_midnight - 86401)
        evidence, errors = self.evidence(current, epoch("2026-09-30 14:00"))
        self.assertEqual(evidence["count"], 0)
        self.assertEqual(evidence["excludedPolicyOrClockCount"], 2)
        self.assertEqual(len(errors), 0)


class WeatherSessionArchiveTests(unittest.TestCase):
    def test_records_only_live_session_issues_and_preserves_first_issue(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "issued.db"
            origin = epoch("2026-09-30 07:00")
            issue = origin + 30
            midnight = forecast.training_cutoff(issue)
            start = epoch("2026-09-30 14:00")
            value = {
                "available": True, "modelVersion": forecast.MODEL_VERSION,
                "originEpoch": origin, "startEpoch": start,
                "endEpoch": start + 7200, "forecastedAtEpoch": issue,
                "mean": 45.0, "sensorAnchor": 52.0,
                "weatherFetchedEpoch": issue - 60,
                "weatherSessionModel": {
                    "applied": True, "featureVersion": learner.FEATURE_VERSION,
                    "trainingCutoffEpoch": midnight,
                    "trainingLatestTargetEndEpoch": midnight - 1,
                },
            }
            # The near-term key is not archived through this session seam.
            result = {"morning": value, "arrival": value}
            self.assertEqual(forecast.record_issue(db, result, issue + 10), 1)
            self.assertEqual(forecast.record_issue(db, result, issue + 11), 0)
            invalid = dict(value)
            invalid["weatherFetchedEpoch"] = issue - 7201
            self.assertEqual(forecast.record_issue(db, {"afternoon": invalid}, issue + 10), 0)
            with closing(sqlite3.connect(db)) as conn:
                rows = conn.execute("SELECT window_key,issued_epoch,payload "
                                    "FROM window_pm_forecast_issues").fetchall()
                near_term_table = conn.execute("SELECT name FROM sqlite_master "
                                               "WHERE type='table' AND name='local_pm_forecast_issues'").fetchone()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "morning")
            self.assertEqual(rows[0][1], issue + 10)
            self.assertEqual(json.loads(rows[0][2])["recordedIssuedEpoch"], issue + 10)
            self.assertIsNone(near_term_table)

    def test_rejects_post_midnight_label_and_mismatched_training_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "issued.db"
            origin = epoch("2026-09-30 07:00")
            issue = origin + 30
            midnight = forecast.training_cutoff(issue)
            start = epoch("2026-09-30 14:00")
            base = {
                "available": True, "modelVersion": forecast.MODEL_VERSION,
                "originEpoch": origin, "startEpoch": start,
                "endEpoch": start + 7200, "forecastedAtEpoch": issue,
                "mean": 45.0, "sensorAnchor": 52.0,
                "weatherFetchedEpoch": issue - 60,
                "weatherSessionModel": {
                    "applied": True, "featureVersion": learner.FEATURE_VERSION,
                    "trainingCutoffEpoch": midnight,
                    "trainingLatestTargetEndEpoch": midnight - 1,
                },
            }
            post_midnight = {**base, "weatherSessionModel": {
                **base["weatherSessionModel"],
                "trainingLatestTargetEndEpoch": midnight + 900,
            }}
            wrong_cutoff = {**base, "startEpoch": start + 900,
                            "endEpoch": start + 8100,
                            "weatherSessionModel": {
                                **base["weatherSessionModel"],
                                "trainingCutoffEpoch": midnight - 86400,
                                "trainingLatestTargetEndEpoch": midnight - 86401,
                            }}
            self.assertLess(post_midnight["weatherSessionModel"]["trainingLatestTargetEndEpoch"], issue)
            self.assertEqual(forecast.record_issue(
                db, {"morning": post_midnight, "afternoon": wrong_cutoff}, issue + 10), 0)
            with closing(sqlite3.connect(db)) as conn:
                count = conn.execute("SELECT COUNT(*) FROM window_pm_forecast_issues").fetchone()[0]
            self.assertEqual(count, 0)


class WeatherSessionPublicationTests(unittest.TestCase):
    def setUp(self):
        self.origin = epoch("2026-09-30 06:30")
        self.issue = self.origin + 42
        self.start = epoch("2026-09-30 09:00")
        self.windows = {"morning": {"startEpoch": self.start,
                                    "endEpoch": self.start + 7200}}
        index = pd.to_datetime([self.origin], unit="s", utc=True).tz_convert(
            "Asia/Kuala_Lumpur")
        self.frame = pd.DataFrame({"pm02": [61.0]}, index=index)
        self.rows = [{"epoch": self.issue - 40, "pm02": 61.0}]
        self.archive = object()
        self.weather = {"available": True, "featureValues": {"duringRainMm": 4.0},
                        "metadata": {"forecastIssuedEpoch": self.issue - 3600}}

    def test_live_query_uses_actual_issue_and_issued_weather_vintage(self):
        query = pd.Series({field: np.nan for field in learner.FEATURES})
        query["fresh"] = 60.0
        query.attrs["valid"] = True
        query.attrs["audit"] = {"camsFetchedEpoch": None}
        fitted = {"estimator": object(), "trainingCount": 121,
                  "trainingDistinctTargetDays": 11,
                  "trainingLatestTargetEndEpoch": self.origin - 86400,
                  "trainingCutoffEpoch": forecast.training_cutoff(self.issue)}
        evidence = {"supportSufficient": False, "skillGatePassed": False}
        with (patch.object(forecast.causal, "strict_frame", return_value=self.frame),
              patch.object(forecast, "raw_arrays", return_value=(
                  np.array([self.issue - 40], dtype=np.int64), np.array([61.0]))),
              patch.object(forecast, "_trained_model", return_value=fitted) as training,
              patch.object(forecast.rain, "load_runs", return_value=self.archive) as load,
              patch.object(forecast.legacy, "_read_runs", return_value=[]) as cams,
              patch.object(forecast.weather_features, "describe_window",
                           return_value=self.weather) as weather,
              patch.object(forecast.direct, "query_features", return_value=query) as feature_query,
              patch.object(learner, "predict_model", return_value=47.5),
              patch.object(forecast, "issued_evidence", return_value=(evidence, np.array([]))),
              patch.object(forecast.rain, "latest_run", return_value=SimpleNamespace(
                  fetched_epoch=self.issue - 60)),
              patch.object(forecast.rain, "describe_window", return_value=self.weather),
              patch.object(forecast.legacy, "predict_windows") as fallback):
            result = forecast.predict_windows("history.db", self.rows, self.windows, self.issue)
        training.assert_called_once()
        self.assertEqual(training.call_args.args[2], forecast.training_cutoff(self.issue))
        load.assert_called_once_with("history.db", self.issue)
        cams.assert_called_once_with("history.db", "air_quality_forecast_runs", self.issue)
        weather.assert_called_once_with(self.archive, self.issue,
                                        self.start, self.start + 7200)
        feature_query.assert_called_once()
        self.assertEqual(feature_query.call_args.args[1:],
                         (self.issue, self.start, self.start + 7200))
        fallback.assert_not_called()
        self.assertEqual(result["morning"]["mean"], 47.5)
        self.assertEqual(result["morning"]["forecastedAtEpoch"], self.issue)
        self.assertEqual(result["morning"]["weatherFetchedEpoch"], self.issue - 60)
        self.assertEqual(result["morning"]["weatherSessionModel"]["featureVersion"],
                         learner.FEATURE_VERSION)
        self.assertEqual(result["morning"]["weatherSessionModel"]["weatherFeatures"],
                         {"duringRainMm": 4.0})

    def test_unavailable_issued_weather_never_substitutes_legacy_prediction(self):
        legacy_result = {"available": True, "modelVersion": "legacy_session",
                         "mean": 61.0}
        with (patch.object(forecast.causal, "strict_frame", return_value=self.frame),
              patch.object(forecast, "raw_arrays", return_value=(
                  np.array([self.issue - 40], dtype=np.int64), np.array([61.0]))),
              patch.object(forecast, "_trained_model", return_value={"estimator": object()}),
              patch.object(forecast.rain, "load_runs", return_value=self.archive),
              patch.object(forecast.legacy, "_read_runs", return_value=[]),
              patch.object(forecast.weather_features, "describe_window",
                           return_value={"available": False, "metadata": {
                               "errors": ["stale_weather_forecast"]}}),
              patch.object(forecast.direct, "query_features") as feature_query,
              patch.object(learner, "predict_model") as predict,
              patch.object(forecast.legacy, "predict_windows",
                           return_value={"morning": legacy_result}) as fallback):
            result = forecast.predict_windows("history.db", self.rows, self.windows, self.issue)
        feature_query.assert_not_called()
        predict.assert_not_called()
        fallback.assert_not_called()
        self.assertFalse(result["morning"]["available"])
        self.assertIsNone(result["morning"]["mean"])
        self.assertIsNone(result["morning"]["prediction"])
        self.assertEqual(result["morning"]["modelVersion"], forecast.MODEL_VERSION)
        self.assertFalse(result["morning"]["weatherSessionModel"]["applied"])
        self.assertEqual(result["morning"]["weatherSessionModel"]["reason"],
                         "missing_stale_or_incomplete_issued_weather")


class WeatherSessionDailyTrainingTests(unittest.TestCase):
    def test_daily_fit_excludes_future_and_incomplete_labels(self):
        cutoff = epoch("2026-09-30 00:00")
        complete = [cutoff - day * 86400 + 16 * 3600
                    for day in range(1, 12) for _ in range(11)]
        issues = [value - 8 * 3600 for value in complete]
        # Both target end and issue must satisfy their own as-of boundaries.
        complete += [cutoff, cutoff + 900, cutoff - 3600, cutoff - 15 * 86400]
        issues += [cutoff - 7200, cutoff - 3600, cutoff, cutoff - 15 * 86400]
        labels = np.array([50.0] * 121 + [901., 902., 903., 904.])
        valid = np.ones(len(labels), dtype=bool)
        snapshot = {"x": pd.DataFrame({"fresh": np.full(len(labels), 60.0)}),
                    "valid": valid, "labels": labels,
                    "complete": np.array(complete, dtype=np.int64),
                    "issues": np.array(issues, dtype=np.int64)}
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / "fit.db"
            with (patch.object(forecast.legacy, "_raw_runs", return_value=([], "r1")),
                  patch.object(learner, "build_dataset", return_value=snapshot) as build,
                  patch.object(learner, "fit_model", return_value=object()) as fit):
                result = forecast._trained_model(db, [], cutoff)
        build.assert_called_once_with(db, [], cutoff)
        fit.assert_called_once()
        self.assertEqual(len(fit.call_args.args[1]), 121)
        self.assertTrue(np.all(fit.call_args.args[1] == 50.0))
        self.assertEqual(result["trainingCount"], 121)
        self.assertEqual(result["trainingDistinctTargetDays"], 11)
        self.assertLess(result["trainingLatestTargetEndEpoch"], cutoff)
        self.assertEqual(result["trainingCutoffEpoch"], cutoff)


if __name__ == "__main__":
    unittest.main()
