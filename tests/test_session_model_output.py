"""Fixed model equality, extrapolation descriptions, and unavailable inputs."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import session_model_output as direct
import weather_session_forecast as morning
import weather_session_model as learner

ISSUE = int(datetime(2026, 10, 8, 12, tzinfo=timezone(timedelta(hours=8))).timestamp())
START = ISSUE + 26*3600


class DirectContractTests(unittest.TestCase):
    def value(self, raw=119.91234567890123):
        return {"available": True, "mean": raw, "modelVersion": "fixed-model-direct-v3",
                "forecastedAtEpoch": ISSUE, "startEpoch": START, "endEpoch": START+7200,
                "sensorAnchor": 100.25, "baselinePoint": 100.25,
                "numericalPolicyIdentifier": "fixed-model-numerical-recipe-v3",
                "modelEvidence": {"available": False, "skillGatePassed": False},
                "candidateForecast": {"forecastedAtEpoch": ISSUE, "startEpoch": START, "endEpoch": START+7200}}

    def test_failed_or_perfect_performance_metadata_cannot_change_literal_number(self):
        value = self.value()
        failed = direct.publish_direct_output(value, "afternoon")
        value["modelEvidence"] = {"available": True, "skillGatePassed": True, "mae": 0, "count": 100}
        perfect = direct.publish_direct_output(value, "afternoon")
        self.assertEqual(failed["mean"], value["mean"])
        self.assertEqual(perfect["mean"], value["mean"])
        self.assertEqual(failed["candidateForecast"]["mean"], value["mean"])
        self.assertNotEqual(failed["mean"], value["baselinePoint"])
        self.assertEqual(failed["modelSelection"], {})
        self.assertFalse(failed["modelOutputPolicy"]["numericalSelectionApplied"])

    def test_outside_training_domain_is_flagged_and_raw_number_is_retained(self):
        result = direct.publish_direct_output(self.value(), "afternoon")
        self.assertTrue(result["available"])
        self.assertTrue(result["modelDomain"]["canonicalTarget"])
        self.assertFalse(result["modelDomain"]["trainedLead"])
        self.assertTrue(result["modelDomain"]["extrapolation"])
        self.assertFalse(result["modelDomain"]["domainAppliedToNumber"])
        self.assertEqual(result["mean"], self.value()["mean"])
        self.assertEqual(result["forecastState"], "model_output")

    def test_contract_adds_no_floor_or_rounding_to_finite_model_output(self):
        # Current learners include their own original nonnegative transform.
        # The publication wrapper must not add a second numerical policy.
        value = self.value(raw=-0.12345678901234)
        result = direct.publish_direct_output(value, "afternoon")
        self.assertTrue(result["available"])
        self.assertEqual(result["mean"].hex(), value["mean"].hex())
        self.assertEqual(result["candidateForecast"]["mean"].hex(), value["mean"].hex())

    def test_missing_model_or_malformed_clocks_never_become_a_baseline(self):
        bad = self.value()
        bad["mean"] = None
        result = direct.publish_direct_output(bad, "afternoon")
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean"])
        bad = self.value()
        bad["candidateForecast"]["forecastedAtEpoch"] = ISSUE-1
        self.assertFalse(direct.publish_direct_output(bad, "afternoon")["available"])
        missing = direct.unavailable_model_output(None, None, None, "fixed", "missing", "morning")
        self.assertIsNone(missing["mean"])
        self.assertIsNone(missing["forecastedAtEpoch"])


class MorningDirectTests(unittest.TestCase):
    def run_model(self, scoring_failure=False):
        stamp = pd.Timestamp(ISSUE, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
        frame = pd.DataFrame({"pm02": [99.], "atmp": [30.], "rhum": [80.]}, index=[stamp])
        query = pd.Series({name: np.nan for name in learner.FEATURE_COLUMNS})
        query["fresh"] = 100.25
        query.attrs = {"valid": True, "audit": {"camsFetchedEpoch": None}}
        class FixedHead:
            def predict(self, x):
                return np.array([-26.1234567890123])
        estimator = FixedHead()
        expected = learner.predict_model(estimator, query)
        cutoff = morning.training_cutoff(ISSUE)
        fitted = {"estimator": estimator, "trainingCutoffEpoch": cutoff,
                  "trainingLatestTargetEndEpoch": cutoff-900, "trainingCount": 120,
                  "trainingDistinctTargetDays": 10, "fittedAtEpoch": ISSUE-30}
        context = {"available": True, "featureValues": {}, "metadata": {}}
        evidence = {"available": False, "count": 0, "supportSufficient": False, "skillGatePassed": False}
        score_kwargs = ({"side_effect": sqlite3.OperationalError("metadata archive locked")}
                        if scoring_failure else {"return_value": (evidence, np.array([]))})
        with patch.object(morning, "_cached_model", return_value=fitted), \
             patch.object(morning, "_trained_model", side_effect=AssertionError("no fitting")), \
             patch.object(morning.legacy, "predict_windows", side_effect=AssertionError("no alternate/baseline")), \
             patch.object(morning.causal, "strict_frame", return_value=frame), \
             patch.object(morning.rain, "load_runs", return_value=object()), \
             patch.object(morning.rain, "latest_run", return_value=SimpleNamespace(fetched_epoch=ISSUE-60)), \
             patch.object(morning.rain, "describe_window", return_value=context), \
             patch.object(morning.weather_features, "describe_window", return_value=context), \
             patch.object(morning.legacy, "_read_runs", return_value=[]), \
             patch.object(morning.direct, "query_features", return_value=query), \
             patch.object(morning, "issued_evidence", **score_kwargs):
            result = morning.predict_windows("unused", [{"epoch": ISSUE-30, "pm02": 100.25}],
                {"morning": {"startEpoch": ISSUE+21*3600, "endEpoch": ISSUE+23*3600}},
                ISSUE, allow_fit=False)["morning"]
        return result, expected

    def test_full_precision_output_equals_the_actual_prediction_helper(self):
        result, expected = self.run_model()
        self.assertTrue(result["available"])
        self.assertEqual(result["mean"], expected)
        self.assertEqual(result["prediction"], expected)
        self.assertEqual(result["candidateForecast"]["mean"], expected)
        self.assertNotEqual(expected, round(expected, 1))
        self.assertNotEqual(expected, result["sensorAnchor"])
        self.assertEqual(result["modelVersion"], morning.MODEL_VERSION)
        self.assertEqual(result["modelSelection"], {})
        self.assertEqual(result["pointRole"], "experimental_model_output")

    def test_performance_archive_failure_does_not_erase_valid_prediction(self):
        result, expected = self.run_model(scoring_failure=True)
        self.assertTrue(result["available"])
        self.assertEqual(result["mean"], expected)
        self.assertFalse(result["performanceEvidence"]["available"])
        self.assertFalse(result["performanceEvidence"]["appliedToNumber"])
        self.assertIn("metadata archive locked", result["performanceEvidence"]["reason"])


if __name__ == "__main__":
    unittest.main()
