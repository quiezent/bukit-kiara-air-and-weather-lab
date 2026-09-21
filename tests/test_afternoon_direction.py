"""Scope, issue provenance, and insufficient-support tests using made-up data."""

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import numpy as np
import pandas as pd

import afternoon_direction_forecast as wrapper
import afternoon_direction_model as model


def epoch(clock):
    return int(datetime.fromisoformat("2026-01-15T" + clock)
               .replace(tzinfo=timezone(timedelta(hours=8))).timestamp())


def inputs(issue):
    windows = {
        "morning": {"startEpoch": epoch("09:00:00"), "endEpoch": epoch("11:00:00")},
        "afternoon": {"startEpoch": epoch("14:00:00"), "endEpoch": epoch("16:00:00")},
    }
    estimates = {key: {
        "modelVersion": "synthetic_ordinary_model", "mean": 90.0, "prediction": 90.0,
        "forecastedAtEpoch": issue, "originEpoch": issue // 900 * 900,
        "nested": {"preserve": [1, 2]}, **window,
    } for key, window in windows.items()}
    return windows, estimates


def prepared(issue):
    count = model.MIN_TRAINING_ROWS
    origin = issue // 900 * 900
    training = {
        "x": pd.DataFrame({"fresh": np.full(count, 80.0)}),
        "labels": np.resize([60.0, 100.0], count),
        "issues": issue - np.arange(count, 0, -1) * 3600,
        "complete": issue - np.arange(count, 0, -1) * 1800,
    }
    query = {
        "x": pd.DataFrame([{**dict.fromkeys(model.FEATURES, np.nan),
                            "fresh": 80.0, "closedLevel": 81.0}]),
        "audit": [{"camsFetchedEpoch": None, "weatherFetchedEpoch": None}],
        "stamps": np.array([issue - 20]), "counts": np.array([2]),
    }
    return training, query, np.ones(count, dtype=bool), origin, issue - origin


class AfternoonScopeTests(unittest.TestCase):
    def apply(self, issue, estimates=None):
        windows, supplied = inputs(issue)
        supplied = supplied if estimates is None else estimates
        with patch.object(wrapper, "_training_and_query", return_value=prepared(issue)) as prepare, \
             patch.object(model, "predict_details", return_value={
                 "prediction": 60.0, "decisionReason": "event_class", "predictedClass": -1,
             }) as predict:
            result = wrapper.apply_experimental_afternoon(
                "unused-synthetic-database", [{"epoch": issue, "pm02": 80.0}],
                windows, issue, supplied,
            )
        return result, supplied, prepare, predict

    def test_scope_before_at_and_after_eight_preserves_other_windows(self):
        for clock, eligible in (("06:59:59", False), ("07:00:00", True),
                                ("07:59:59", True), ("08:00:00", True),
                                ("08:00:01", False)):
            with self.subTest(clock=clock):
                issue = epoch(clock)
                _, supplied = inputs(issue)
                before = copy.deepcopy(supplied)
                result, _, prepare, predict = self.apply(issue, supplied)
                value = result["afternoon"]
                self.assertEqual(supplied, before)
                self.assertIs(result["morning"], supplied["morning"])
                self.assertEqual(value["mean"], 60.0 if eligible else 90.0)
                self.assertEqual(value["forecastedAtEpoch"], issue)
                self.assertEqual(predict.call_count, int(eligible))
                self.assertEqual(prepare.call_count, int(eligible))
                self.assertEqual(value["modelSelection"]["scope"]["ordinaryPolicyStartsEpoch"], epoch("08:00:01"))

    def test_scope_requires_named_exact_same_day_afternoon_target(self):
        issue = epoch("07:30:00")
        for change in ("wrong_name", "wrong_day", "wrong_end"):
            with self.subTest(change=change):
                windows, estimates = inputs(issue)
                if change == "wrong_name":
                    windows["other"] = windows.pop("afternoon")
                elif change == "wrong_day":
                    windows["afternoon"] = {key: value + 86400 for key, value in windows["afternoon"].items()}
                else:
                    windows["afternoon"]["endEpoch"] += 1
                with patch.object(wrapper, "_training_and_query") as prepare:
                    result = wrapper.apply_experimental_afternoon("unused", [], windows, issue, estimates)
                self.assertEqual(result, estimates)
                prepare.assert_not_called()

    def test_cached_forecast_keeps_original_issue_point_and_model_after_handoff(self):
        cached, _, _, _ = self.apply(epoch("08:00:00"))
        before = copy.deepcopy(cached)
        result, _, prepare, predict = self.apply(epoch("08:00:01"), cached)
        value = result["afternoon"]
        for key in ("mean", "prediction", "modelVersion", "forecastIssuedEpoch", "forecastedAtEpoch",
                    "originEpoch", "startEpoch", "endEpoch", "directionalModel", "modelEvidence"):
            self.assertEqual(value[key], cached["afternoon"][key])
        self.assertEqual(cached, before)
        selection = value["modelSelection"]
        self.assertEqual(selection["forecastIssuedEpoch"], epoch("08:00:00"))
        self.assertEqual(selection["selectionIssueEpoch"], epoch("08:00:01"))
        self.assertEqual(selection["selectionReason"], "retained_previously_issued_experimental_forecast")
        self.assertEqual(selection["policyEligibilityReason"], "morning_scope_ended")
        self.assertTrue(selection["retainedIssuedForecast"])
        prepare.assert_not_called()
        predict.assert_not_called()

    def test_missing_fresh_reference_retains_ordinary_forecast_without_fitting(self):
        issue = epoch("07:30:00")
        windows, estimates = inputs(issue)
        with patch.object(wrapper, "_training_and_query") as prepare:
            result = wrapper.apply_experimental_afternoon("unused", [], windows, issue, estimates)
        value = result["afternoon"]
        self.assertEqual(value["prediction"], estimates["afternoon"]["prediction"])
        self.assertEqual(value["modelSelection"]["experimentalAttempt"], {
            "state": "not_fitted", "reason": "fresh_sensor_reference_unavailable",
        })
        self.assertNotIn("directionalModel", value)
        prepare.assert_not_called()

    def test_training_requires_last_target_bucket_complete_by_issue(self):
        issue = epoch("07:30:00")
        count = model.MIN_TRAINING_ROWS + 1
        training = {
            "x": pd.DataFrame({"fresh": np.full(count, 80.0)}),
            "labels": np.resize([60.0, 100.0], count),
            "issues": issue - np.arange(count, 0, -1) * 3600,
            "complete": np.full(count, issue - 1), "valid": np.ones(count, dtype=bool),
        }
        training["complete"][-2:] = [issue, issue + 1]
        origin = pd.Timestamp(issue, unit="s", tz="UTC")
        frame = pd.DataFrame({"pm02": [80.0]}, index=pd.DatetimeIndex([origin]))
        query = {"valid": np.array([True])}
        with patch.object(wrapper.features, "load", return_value=([], frame, [], [])), \
             patch.object(wrapper.features, "design", side_effect=[training, query]):
            result = wrapper._training_and_query("unused", [], issue, epoch("14:00:00"), epoch("16:00:00"))
        self.assertIsNotNone(result)
        usable = result[2]
        self.assertEqual(int(usable.sum()), model.MIN_TRAINING_ROWS)
        self.assertTrue(usable[-2])  # A bucket that has closed at issue is observed.
        self.assertFalse(usable[-1])


class ClassifierSupportTests(unittest.TestCase):
    def evaluate(self, count, *, issues=None, labels=None, fresh=80.0):
        issue = epoch("07:30:00")
        train = pd.DataFrame({"fresh": np.full(count, 80.0)})
        issues = issue - np.arange(count, 0, -1) * 3600 if issues is None else issues
        labels = np.resize([60.0, 100.0], count) if labels is None else labels
        with patch.object(model, "HistGradientBoostingClassifier") as estimator:
            result = model.predict_details(train, labels, {"fresh": fresh}, issues, issue, "session390")
        estimator.assert_not_called()
        return result

    def test_insufficient_input_and_missing_reference_do_not_fit(self):
        result = self.evaluate(model.MIN_TRAINING_ROWS - 1)
        self.assertEqual(result["decisionReason"], "insufficient_input_rows")
        self.assertEqual(result["prediction"], 80.0)
        missing = self.evaluate(model.MIN_TRAINING_ROWS, fresh=np.nan)
        self.assertEqual(missing["decisionReason"], "required_fresh_reference_unavailable")
        self.assertIsNone(missing["prediction"])

    def test_current_future_stale_and_nonfinite_issues_are_ineligible(self):
        count = model.MIN_TRAINING_ROWS
        issue = epoch("07:30:00")
        for rejected in (issue, issue + 1, issue - 14 * 86400 - 1, np.nan):
            with self.subTest(rejected=rejected):
                issues = (issue - np.arange(count, 0, -1) * 3600).astype(float)
                issues[-1] = rejected
                result = self.evaluate(count, issues=issues)
                self.assertEqual(result["decisionReason"], "insufficient_eligible_rows")
                self.assertEqual(result["trainingCount"], count - 1)
                self.assertEqual(result["prediction"], 80.0)

    def test_single_direction_class_is_insufficient_support(self):
        count = model.MIN_TRAINING_ROWS
        result = self.evaluate(count, labels=np.full(count, 60.0))
        self.assertEqual(result["decisionReason"], "insufficient_classes")
        self.assertEqual(result["trainingClassCounts"], {"-1": count, "0": 0, "1": 0})
        self.assertEqual(result["prediction"], 80.0)
        self.assertEqual(result["classScores"], {})


if __name__ == "__main__":
    unittest.main()
