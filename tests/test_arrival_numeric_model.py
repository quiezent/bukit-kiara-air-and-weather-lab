"""Dedicated arrival learner: causal targets, cached fits and literal outputs."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import arrival_numeric_model as model
import fresh_sensor_features as sensor

BASE = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
COLUMNS = (*sensor.FEATURE_COLUMNS, "issuedWeatherTest")


def readings(days=13):
    return [{"epoch": BASE + index*120, "pm02": 90 + 20*np.sin(index/90),
             "atmp": 28 + np.sin(index/80), "rhum": 70 + 3*np.cos(index/100)}
            for index in range(days*720 + 1)]


def feature_frame(prepared, issues):
    frame = sensor.issue_feature_frame(prepared.sensorRows, issues)
    frame["issuedWeatherTest"] = np.nan
    return frame


INPUTS = SimpleNamespace(
    FEATURE_COLUMNS=COLUMNS, __file__=sensor.__file__,
    prepare_inputs=lambda rows: SimpleNamespace(sensorRows=tuple(rows)),
    source_sensor_rows=lambda prepared: prepared.sensorRows,
    issue_feature_frame=feature_frame,
)


class RecordingEstimator:
    def __init__(self, output=71.123456789):
        self.output = output

    def fit(self, matrix, targets):
        self.training_matrix, self.training_targets = matrix.copy(), targets.copy()
        return self

    def predict(self, matrix):
        self.query_matrix = matrix.copy()
        return np.full(len(matrix), self.output)


def artifact(estimator, issue):
    return {"estimator": estimator, "metadata": {
        "modelIdentity": model.model_identity(), "featureColumns": list(COLUMNS),
        "trainingCutoffEpoch": model.midnight_epoch(issue),
        "latestTrainingOutcomeCompleteEpoch": model.midnight_epoch(issue) - 1,
        "fittedAtEpoch": model.midnight_epoch(issue) - 1,
    }}


class InputFixture(unittest.TestCase):
    def setUp(self):
        self.input_patch = patch.object(model, "_inputs", return_value=INPUTS)
        self.input_patch.start()
        self.addCleanup(self.input_patch.stop)


class ArrivalTargetTests(unittest.TestCase):
    def series(self):
        return pd.Series(np.arange(40, dtype=float),
                         index=pd.date_range("2026-09-01", periods=40, freq="15min", tz="UTC"))

    def test_exact_issue_clock_arrival_interpolation_and_completion(self):
        for offset in (0, 137, 899):
            target, complete = model.targets_for_issues(self.series(), [BASE + offset])
            self.assertAlmostEqual(target.arrivalPoint.iloc[0], 6 + offset/900)
            self.assertEqual(complete[0], BASE + (6 if offset == 0 else 7)*900)

    def test_missing_later_ride_outcomes_do_not_remove_valid_arrival(self):
        series = self.series()
        series.iloc[8:] = np.nan
        target, complete = model.targets_for_issues(series, [BASE + 137])
        self.assertAlmostEqual(target.arrivalPoint.iloc[0], 6 + 137/900)
        self.assertEqual(complete[0], BASE + 7*900)

    def test_missing_required_arrival_bucket_invalidates_only_that_target(self):
        for bucket in (6, 7):
            series = self.series()
            series.iloc[bucket] = np.nan
            target, _ = model.targets_for_issues(series, [BASE + 137])
            self.assertTrue(np.isnan(target.arrivalPoint.iloc[0]))

    def test_incomplete_coverage_and_unfinished_observed_bins_are_missing(self):
        partial = [{"epoch": BASE + 30, "pm02": 50}, {"epoch": BASE + 800, "pm02": 60}]
        self.assertTrue(model.observed_series(partial, BASE + 900).isna().all())
        full = [{"epoch": BASE + offset, "pm02": 50 + offset/100}
                for offset in (60, 240, 420, 600, 840)]
        self.assertAlmostEqual(model.observed_series(full, BASE + 900).iloc[0], 54.2)
        self.assertTrue(model.observed_series(full, BASE + 850).isna().all())


class ArrivalEstimatorTests(InputFixture):
    def test_fixed_hgb_recipe_has_native_missing_weather_and_no_early_stop(self):
        estimator = model._make_estimator()
        self.assertEqual(estimator.reference_index, list(COLUMNS).index("freshPm25"))
        matrix = np.column_stack((100 + np.arange(40), np.sin(np.arange(40)), np.full(40, np.nan)))
        estimator.fit(matrix, matrix[:, 0] - 25)
        for key, value in model.MODEL_PARAMETERS.items():
            self.assertEqual(estimator.estimator_.get_params()[key], value)
        self.assertFalse(estimator.estimator_.early_stopping)
        query = matrix[[7, 20]].copy()
        np.testing.assert_allclose(estimator.predict(query), query[:, 0] - 25)

    def test_estimator_owns_same_reference_forward_and_inverse_transform(self):
        matrix = np.column_stack((100 + np.arange(40), np.sin(np.arange(40))))
        estimator = model.FreshReferenceArrivalRegressor().fit(matrix, matrix[:, 0] + 7.25)
        query = matrix[[7, 20]].copy()
        np.testing.assert_array_equal(estimator.predict(query),
                                      estimator.estimator_.predict(query) + query[:, 0])
        translated = query.copy()
        translated[:, 0] += 100
        np.testing.assert_allclose(estimator.predict(translated), estimator.predict(query) + 100)

    def test_missing_reference_cannot_be_imputed_or_replaced_with_current(self):
        matrix = np.column_stack((100 + np.arange(40), np.zeros(40)))
        fitted = model.FreshReferenceArrivalRegressor().fit(matrix, matrix[:, 0])
        with self.assertRaisesRegex(ValueError, "fresh_reference_unavailable"):
            fitted.predict(np.asarray([[np.nan, 10]]))
        with self.assertRaisesRegex(ValueError, "reference_or_target_unavailable"):
            model.FreshReferenceArrivalRegressor().fit(np.asarray([[np.nan, 10]]), [100])


class ArrivalTrainingTests(InputFixture):
    @classmethod
    def setUpClass(cls):
        cls.rows = readings()
        cls.cutoff = BASE + 12*86400
        cls.issues = np.concatenate([BASE + day*86400 + 137 + np.arange(31)*300
                                    for day in range(1, 11)])

    def test_future_rows_and_embargoed_origins_cannot_change_fitted_data(self):
        origins = np.append(self.issues, self.cutoff - model.EMBARGO_SECONDS)
        first_estimator = RecordingEstimator()
        with patch.object(model, "_make_estimator", return_value=first_estimator):
            first = model.fit(self.rows, self.cutoff, issue_epochs=origins)
        altered = deepcopy(self.rows)
        for row in altered:
            if row["epoch"] > self.cutoff:
                row.update(pm02=10000, atmp=999, rhum=999)
        second_estimator = RecordingEstimator()
        with patch.object(model, "_make_estimator", return_value=second_estimator):
            second = model.fit(altered, self.cutoff, issue_epochs=origins)
        self.assertEqual(first["metadata"]["trainingDataHash"], second["metadata"]["trainingDataHash"])
        np.testing.assert_array_equal(first_estimator.training_matrix, second_estimator.training_matrix)
        np.testing.assert_array_equal(first_estimator.training_targets, second_estimator.training_targets)
        self.assertLess(first["metadata"]["latestTrainingOriginEpoch"], self.cutoff - model.EMBARGO_SECONDS)
        self.assertLessEqual(first["metadata"]["latestTrainingOutcomeCompleteEpoch"], self.cutoff)
        self.assertEqual(first["metadata"]["trainingOriginCount"], 310)
        self.assertEqual(first["metadata"]["trainingDateCount"], 10)
        self.assertEqual(first["metadata"]["weatherTrainingNonemptyRowCount"], 0)
        self.assertFalse(first["metadata"]["weatherInputsLearnedFromTraining"])

    def test_prepared_training_filters_late_outcomes_and_feature_watermarks(self):
        features = feature_frame(INPUTS.prepare_inputs(self.rows), self.issues)
        target, complete = model.targets_for_issues(model.observed_series(self.rows, self.cutoff), self.issues)
        labels = pd.DataFrame({"arrivalPm": target.arrivalPoint, "completeEpoch": complete,
                               "featureSourceMaxEpoch": self.issues - 17}, index=self.issues)
        labels.loc[self.issues[0], "featureSourceMaxEpoch"] = self.issues[0] + 1
        labels.loc[self.issues[1], "completeEpoch"] = self.cutoff + 1
        labels.loc[self.issues[2], "arrivalPm"] = np.nan
        estimator = RecordingEstimator()
        with patch.object(model, "_make_estimator", return_value=estimator):
            fitted = model.fit_prepared({"features": features, "issues": labels}, self.cutoff)
        self.assertEqual(fitted["metadata"]["trainingOriginCount"], 307)
        self.assertEqual(len(estimator.training_targets), 307)

    def test_training_requires_enough_origins_and_dates(self):
        with self.assertRaisesRegex(ValueError, "insufficient_completed_training_origins"):
            model.fit(self.rows, self.cutoff, issue_epochs=self.issues[:100])
        one_day = BASE + 86400 + np.arange(310)*60
        with self.assertRaisesRegex(ValueError, "insufficient_training_dates"):
            model.fit(self.rows, self.cutoff, issue_epochs=one_day)

    def test_asset_round_trip_preserves_recipe_and_rejects_foreign_identity(self):
        with patch.object(model, "_make_estimator", return_value=RecordingEstimator()):
            stored = model.fit(self.rows, self.cutoff, issue_epochs=self.issues)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save_artifact(stored, Path(directory)/"arrival.pkl")
            self.assertEqual(model.load_artifact(path)["metadata"], stored["metadata"])
            stored["metadata"]["modelIdentity"]["parameters"]["max_iter"] = 999
            model.save_artifact(stored, path)
            self.assertIsNone(model.load_artifact(path))

    def test_real_shared_input_builder_and_hgb_keep_all_77_missing_weather_columns(self):
        import arrival_feature_inputs as inputs
        with patch.object(model, "_inputs", return_value=inputs):
            stored = model.fit(self.rows, self.cutoff, issue_epochs=self.issues)
            self.assertEqual(stored["metadata"]["featureCount"], 285)
            self.assertEqual(stored["metadata"]["weatherFeatureColumnCount"], 77)
            self.assertEqual(stored["metadata"]["weatherTrainingNonemptyValueCount"], 0)
            issue = self.cutoff + 16*3600 + 137
            output = model.predict(self.rows, issue, artifact=stored, allow_reconstruction=True)
            self.assertTrue(output["available"])
            self.assertTrue(np.isfinite(output["arrivalPoint"]))
            self.assertEqual(output["arrivalPoint"], output["rawOutputUgM3"]["arrivalPoint"])


class ArrivalPredictionTests(InputFixture):
    @classmethod
    def setUpClass(cls):
        cls.rows = readings()
        cls.issue = BASE + 11*86400 + 137

    def test_negative_and_high_precision_native_output_is_published_literally(self):
        for point in (71.123456789, -5.23456789):
            estimator = RecordingEstimator(point)
            before = deepcopy(self.rows)
            with patch.object(model, "_make_estimator", side_effect=AssertionError("predict cannot fit")):
                output = model.predict(self.rows, self.issue, artifact=artifact(estimator, self.issue))
            self.assertTrue(output["available"])
            self.assertEqual(output["arrivalPoint"], point)
            self.assertEqual(output["rawOutputUgM3"], {"arrivalPoint": point})
            self.assertFalse(output["numericalSelectionApplied"])
            self.assertFalse(output["rawOutputArithmeticApplied"])
            self.assertFalse(output["persistenceFallback"])
            self.assertEqual(output["forecastClock"]["arrivalTargetEpoch"], self.issue + 5400)
            self.assertEqual(before, self.rows)

    def test_future_measurements_cannot_enter_prediction_features(self):
        altered = deepcopy(self.rows)
        for row in altered:
            if row["epoch"] > self.issue:
                row.update(pm02=10000, atmp=999, rhum=999)
        first, second = RecordingEstimator(), RecordingEstimator()
        original = model.predict(self.rows, self.issue, artifact=artifact(first, self.issue))
        changed = model.predict(altered, self.issue, artifact=artifact(second, self.issue))
        self.assertEqual(original["queryFeaturesHash"], changed["queryFeaturesHash"])
        np.testing.assert_array_equal(first.query_matrix, second.query_matrix)
        self.assertLessEqual(original["featureSourceMaxEpoch"], self.issue)

    def test_missing_asset_and_stale_reference_are_unavailable_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = model.predict(self.rows, self.issue, artifact_path=Path(directory)/"missing.pkl")
        self.assertFalse(missing["available"])
        self.assertIsNone(missing["arrivalPoint"])
        self.assertEqual(missing["reason"], "arrival_numeric_model_asset_unavailable")
        estimator = RecordingEstimator()
        stale = [row for row in self.rows if row["epoch"] < self.issue - 600]
        missing = model.predict(stale, self.issue, artifact=artifact(estimator, self.issue))
        self.assertFalse(missing["available"])
        self.assertFalse(hasattr(estimator, "query_matrix"))

    def test_fit_and_outcome_clocks_cannot_postdate_live_issue(self):
        changes = (
            {"fittedAtEpoch": self.issue + 1},
            {"latestTrainingOutcomeCompleteEpoch": self.issue + 1},
            {"trainingCutoffEpoch": self.issue + 1},
            {"fittedAtEpoch": None},
            {"trainingCutoffEpoch": "yesterday"},
            {"latestTrainingOutcomeCompleteEpoch": None},
        )
        for change in changes:
            estimator = RecordingEstimator()
            stored = artifact(estimator, self.issue)
            stored["metadata"].update(change)
            missing = model.predict(self.rows, self.issue, artifact=stored)
            self.assertFalse(missing["available"])
            self.assertFalse(hasattr(estimator, "query_matrix"))

    def test_later_fit_only_allowed_as_explicit_counterfactual_reconstruction(self):
        stored = artifact(RecordingEstimator(), self.issue)
        stored["metadata"]["fittedAtEpoch"] = self.issue + 1
        output = model.predict(self.rows, self.issue, artifact=stored, allow_reconstruction=True)
        self.assertTrue(output["available"])
        self.assertTrue(output["reconstructed"])
        self.assertEqual(output["forecastRole"], "counterfactual_reconstruction")

    def test_cached_query_frame_selects_matching_provenance_without_refitting(self):
        issues = [self.issue - 300, self.issue, self.issue + 300]
        features = feature_frame(INPUTS.prepare_inputs(self.rows), issues)
        estimator = RecordingEstimator()
        output = model.predict(self.rows, self.issue, artifact=artifact(estimator, self.issue),
                               query_features=features)
        self.assertTrue(output["available"])
        self.assertEqual(output["inputLineage"]["forecastIssuedEpoch"], self.issue)

    def test_invalid_native_shape_or_nonfinite_output_remains_unavailable(self):
        for value in (np.asarray([[71.0]]), np.asarray([np.nan]), np.asarray([np.inf])):
            estimator = SimpleNamespace(predict=lambda matrix, value=value: value)
            output = model.predict(self.rows, self.issue, artifact=artifact(estimator, self.issue))
            self.assertFalse(output["available"])
            self.assertIsNone(output["arrivalPoint"])
            self.assertEqual(output["reason"], "arrival_numeric_estimator_output_invalid")

    def test_future_input_lineage_is_rejected_before_estimator_runs(self):
        features = feature_frame(INPUTS.prepare_inputs(self.rows), [self.issue])
        features.attrs["featureProvenance"][0]["featureSourceMaxEpoch"] = self.issue + 1
        estimator = RecordingEstimator()
        output = model.predict(self.rows, self.issue, artifact=artifact(estimator, self.issue),
                               query_features=features)
        self.assertFalse(output["available"])
        self.assertEqual(output["reason"], "arrival_input_after_issue")
        self.assertFalse(hasattr(estimator, "query_matrix"))


if __name__ == "__main__":
    unittest.main()
