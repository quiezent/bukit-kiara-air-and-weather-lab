"""Fresh four-head model contracts; synthetic observations and cached fits."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import fresh_numeric_model_v3 as model
import fresh_sensor_features as sensor
from forecast_clock import TARGET_VERSION

BASE = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())


def rows(days=10):
    return [{"epoch": BASE + index*120,
             "pm02": 80 + 20*np.sin(index/90) + index/10000,
             "atmp": 28 + np.sin(index/80), "rhum": 70 + 3*np.cos(index/100)}
            for index in range(days*720+1)]


class RecordingEstimator:
    def __init__(self, output=(71.123456789, 99.987654321, 120.123456789, -5.23456789)):
        self.output = output

    def fit(self, matrix, responses):
        self.training_matrix = matrix.copy()
        self.training_response = responses.copy()
        return self

    def predict(self, matrix):
        self.query_matrix = matrix.copy()
        return np.asarray([self.output])


def artifact(estimator, issue):
    return {"estimator": estimator, "metadata": {
        "trainingCutoffEpoch": model.midnight_epoch(issue),
        "latestTrainingOutcomeCompleteEpoch": model.midnight_epoch(issue)-1,
        "fittedAtEpoch": model.midnight_epoch(issue)-1,
        "modelIdentity": model.model_identity(),
        "featureColumns": list(sensor.FEATURE_COLUMNS)}}


class FreshTargetTests(unittest.TestCase):
    def series(self):
        index = pd.date_range("2026-09-01", periods=40, freq="15min", tz="UTC")
        return pd.Series(np.arange(40, dtype=float), index=index)

    def test_exact_clock_partial_edges_and_interpolation(self):
        issue = BASE+137
        targets, complete = model.targets_for_issues(self.series(), [issue])
        values = targets.iloc[0]
        self.assertAlmostEqual(values.arrivalPoint, 6+137/900)
        self.assertAlmostEqual(values.trailMeanPoint,
                               (7*(900-137)+sum(range(8, 15))*900+15*137)/7200)
        self.assertEqual(values.trailMinimumPoint, 7)
        self.assertEqual(values.trailMaximumPoint, 15)
        self.assertEqual(complete[0], BASE+15*900)

    def test_aligned_window_excludes_zero_overlap_bins(self):
        target, complete = model.targets_for_issues(self.series(), [BASE])
        self.assertEqual(target.iloc[0].tolist(), [6, 10.5, 7, 14])
        self.assertEqual(complete[0], BASE+14*900)

    def test_any_missing_required_target_invalidates_the_joint_response(self):
        for missing in (6, 7, 15):
            series = self.series()
            series.iloc[missing] = np.nan
            target, _ = model.targets_for_issues(series, [BASE+137])
            self.assertTrue(target.iloc[0].isna().all())

    def test_incomplete_sensor_buckets_are_not_medians_of_a_partial_sample(self):
        readings = [{"epoch": BASE+30, "pm02": 50},
                    {"epoch": BASE+800, "pm02": 60}]
        self.assertTrue(model.observed_series(readings, BASE+900).isna().all())
        complete = [{"epoch": BASE+stamp, "pm02": 50+stamp/100}
                    for stamp in (60, 240, 420, 600, 840)]
        self.assertEqual(model.observed_series(complete, BASE+900).iloc[0], 54.2)
        self.assertTrue(model.observed_series(complete, BASE+850).isna().all())


class FreshTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readings = rows()
        cls.cutoff = BASE+9*86400
        cls.origins = BASE+3*86400+137+np.arange(260)*900

    def test_fit_excludes_future_and_embargoed_observations_without_changing_training_hash(self):
        estimator = RecordingEstimator()
        origins = np.append(self.origins, [self.cutoff-model.EMBARGO_SECONDS,
                                          self.cutoff-model.EMBARGO_SECONDS+300])
        with patch.object(model, "_make_estimator", return_value=estimator):
            first = model.fit(self.readings, self.cutoff, issue_epochs=origins)
        original_response = estimator.training_response.copy()
        altered = deepcopy(self.readings)
        for item in altered:
            if item["epoch"] > self.cutoff:
                item.update(pm02=10000, atmp=-100, rhum=-100)
        with patch.object(model, "_make_estimator", return_value=estimator):
            second = model.fit(altered, self.cutoff, issue_epochs=origins)
        self.assertEqual(first["metadata"]["trainingDataHash"], second["metadata"]["trainingDataHash"])
        np.testing.assert_array_equal(original_response, estimator.training_response)
        self.assertLess(first["metadata"]["latestTrainingOriginEpoch"], self.cutoff-model.EMBARGO_SECONDS)
        self.assertLessEqual(first["metadata"]["latestTrainingOutcomeCompleteEpoch"], self.cutoff)
        self.assertEqual(first["metadata"]["trainingOriginCount"], len(self.origins))
        self.assertTrue(np.all(original_response[:, 2] <= original_response[:, 1]))
        self.assertTrue(np.all(original_response[:, 1] <= original_response[:, 3]))

    def test_fixed_recipe_has_training_only_imputer_scaler_and_ridge_alpha100(self):
        pipeline = model._make_estimator()
        imputer, transformed = list(pipeline.named_steps.values())
        self.assertIsInstance(transformed, model.FreshReferenceRegressor)
        scaler, ridge = list(transformed.estimator.named_steps.values())
        self.assertTrue(scaler.with_mean)
        self.assertTrue(scaler.with_std)
        self.assertEqual(imputer.strategy, "median")
        self.assertTrue(imputer.keep_empty_features)
        for key, value in model.MODEL_PARAMETERS.items():
            self.assertEqual(ridge.get_params()[key], value)
        self.assertEqual(model.TRAINING_CADENCE_SECONDS, 300)

    def test_real_ridge_has_four_native_physical_heads_without_output_sorting(self):
        matrix = np.column_stack((np.arange(40), np.sin(np.arange(40))))
        middle = 50+np.arange(40)
        response = np.column_stack((middle+2, middle, middle-5, middle+6))
        pipeline = model._make_estimator().fit(matrix, response)
        predicted = pipeline.predict(matrix[[7, 20, 32]])
        transformed = list(pipeline.named_steps.values())[-1]
        scaler, ridge = list(transformed.estimator_.named_steps.values())
        self.assertEqual(ridge.coef_.shape[0], 4)
        np.testing.assert_allclose(predicted,
                                   transformed.estimator_.predict(matrix[[7, 20, 32]])+matrix[[7, 20, 32]][:, [0]])
        self.assertTrue(np.all(predicted[:, 2] <= predicted[:, 1]))
        self.assertTrue(np.all(predicted[:, 1] <= predicted[:, 3]))

    def test_estimator_owns_the_forward_and_inverse_target_transformation(self):
        matrix = np.column_stack((100+np.arange(40), np.sin(np.arange(40))))
        response = matrix[:, [0]] + np.column_stack((np.full(40, 2), np.full(40, -3),
                                                     np.full(40, -9), np.full(40, 4)))
        fitted = model.FreshReferenceRegressor().fit(matrix, response)
        query = matrix[[7, 20]].copy()
        expected = fitted.estimator_.predict(query)+query[:, [0]]
        np.testing.assert_array_equal(fitted.predict(query), expected)
        # Constant deltas expose the inverse's translation behavior even when
        # the new absolute sensor level lies beyond all old training levels.
        translated_query = query.copy()
        translated_query[:, 0] += 100
        np.testing.assert_allclose(fitted.predict(translated_query), fitted.predict(query)+100)

    def test_joint_training_is_translation_equivariant_for_identical_shifted_data(self):
        matrix = np.column_stack((100+np.arange(40), np.arange(40)%3))
        delta = np.arange(40)%7
        response = matrix[:, [0]] + np.column_stack((delta, delta-2, delta-8, delta+5))
        shifted = matrix.copy()
        shifted[:, 0] += 200
        original = model.FreshReferenceRegressor().fit(matrix, response)
        translated = model.FreshReferenceRegressor().fit(shifted, response+200)
        np.testing.assert_allclose(translated.predict(shifted[[7, 20]]),
                                   original.predict(matrix[[7, 20]])+200)

    def test_internal_scaler_statistics_are_fitted_only_to_the_training_rows(self):
        matrix = np.column_stack((100+np.arange(40), np.arange(40)%3))
        response = matrix[:, [0]]+np.column_stack((np.ones(40), np.zeros(40),
                                                  -np.ones(40), np.full(40, 2)))
        fitted = model.FreshReferenceRegressor().fit(matrix, response)
        scaler = list(fitted.estimator_.named_steps.values())[0]
        np.testing.assert_array_equal(scaler.mean_, matrix.mean(axis=0))
        before = scaler.mean_.copy()
        fitted.predict(np.array([[10000, -10000.0]]))
        np.testing.assert_array_equal(scaler.mean_, before)

    def test_saved_asset_round_trip_preserves_identity_and_rejects_foreign_recipe(self):
        estimator = RecordingEstimator()
        with patch.object(model, "_make_estimator", return_value=estimator):
            trained = model.fit(self.readings, self.cutoff, issue_epochs=self.origins)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save_artifact(trained, Path(directory)/"model.pkl")
            loaded = model.load_artifact(path)
            self.assertEqual(loaded["metadata"], trained["metadata"])
            trained["metadata"]["modelIdentity"]["parameters"]["max_depth"] = 999
            model.save_artifact(trained, path)
            self.assertIsNone(model.load_artifact(path))


class FreshInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readings = rows()
        cls.issue = BASE+8*86400+137

    def test_outputs_are_literal_even_negative_or_crossed_no_fitting_or_fallback(self):
        estimator = RecordingEstimator()
        source = deepcopy(self.readings)
        with patch.object(model, "_make_estimator", side_effect=AssertionError("Inference cannot fit")):
            output = model.predict(source, self.issue, artifact=artifact(estimator, self.issue))
        self.assertTrue(output["available"])
        self.assertEqual(tuple(output[name] for name in model.OUTPUT_COLUMNS), estimator.output)
        self.assertEqual(output["rawOutputUgM3"], {name: estimator.output[index]
                                                  for index, name in enumerate(model.OUTPUT_COLUMNS)})
        self.assertFalse(output["rangeOrdered"])
        self.assertFalse(output["rawOutputArithmeticApplied"])
        self.assertFalse(output["numericalSelectionApplied"])
        self.assertFalse(output["persistenceFallback"])
        self.assertEqual(source, self.readings)

    def test_recent_drop_is_a_model_input_future_values_cannot_enter_query(self):
        altered = deepcopy(self.readings)
        for item in altered:
            if self.issue-1200 < item["epoch"] <= self.issue:
                item["pm02"] = 20
            elif item["epoch"] > self.issue:
                item["pm02"] = 10000
        estimator = RecordingEstimator()
        output = model.predict(altered, self.issue, artifact=artifact(estimator, self.issue))
        self.assertTrue(output["available"])
        columns = list(sensor.FEATURE_COLUMNS)
        self.assertEqual(estimator.query_matrix[0, columns.index("freshPm25")], 20)
        self.assertLess(estimator.query_matrix[0, columns.index("rawDelta30")], -20)
        self.assertLessEqual(output["featureSourceMaxEpoch"], self.issue)
        self.assertEqual(output["forecastClock"]["featureAnchorEpoch"], self.issue)
        self.assertEqual(output["forecastClock"]["targetVersion"], TARGET_VERSION)
        self.assertEqual(output["forecastClock"]["windowEndEpoch"], self.issue+12600)

    def test_missing_asset_and_stale_sensor_are_unavailable_without_old_model_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = model.predict(self.readings, self.issue,
                                   artifact_path=Path(directory)/"missing.pkl")
        self.assertFalse(output["available"])
        self.assertEqual(output["reason"], "fresh_numeric_model_asset_unavailable")
        for key in model.OUTPUT_COLUMNS:
            self.assertIsNone(output[key])
        stale = [row for row in self.readings if row["epoch"] < self.issue-600]
        estimator = RecordingEstimator()
        output = model.predict(stale, self.issue, artifact=artifact(estimator, self.issue))
        self.assertFalse(output["available"])
        self.assertFalse(hasattr(estimator, "query_matrix"))

    def test_asset_trained_after_issue_is_rejected(self):
        estimator = RecordingEstimator()
        for change in ({"trainingCutoffEpoch": self.issue+1},
                       {"latestTrainingOutcomeCompleteEpoch": self.issue+1}):
            with self.subTest(change=change):
                stored = artifact(estimator, self.issue)
                stored["metadata"].update(change)
                output = model.predict(self.readings, self.issue, artifact=stored)
                self.assertFalse(output["available"])
                self.assertFalse(hasattr(estimator, "query_matrix"))

    def test_later_fit_cannot_be_presented_as_an_original_historical_forecast(self):
        estimator = RecordingEstimator()
        stored = artifact(estimator, self.issue)
        stored["metadata"]["fittedAtEpoch"] = self.issue+1
        output = model.predict(self.readings, self.issue, artifact=stored)
        self.assertFalse(output["available"])
        self.assertEqual(output["reason"], "model_not_fitted_at_issue")
        self.assertFalse(hasattr(estimator, "query_matrix"))
        replay = model.predict(self.readings, self.issue, artifact=stored, allow_reconstruction=True)
        self.assertTrue(replay["available"])
        self.assertTrue(replay["reconstructed"])
        self.assertEqual(replay["forecastRole"], "counterfactual_reconstruction")
        stored["metadata"].pop("fittedAtEpoch")
        self.assertFalse(model.predict(self.readings, self.issue, artifact=stored)["available"])


if __name__ == "__main__":
    unittest.main()
