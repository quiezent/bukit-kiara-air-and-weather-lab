
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast_clock import ForecastClock, window_weights
import ride_pm_extrema as extrema


class RecordingEstimator:
    def __init__(self, prediction=(71.123456789, 104.987654321)):
        self.prediction = prediction

    def fit(self, matrix, response):
        self.training_matrix = matrix.copy()
        self.training_response = response.copy()
        return self

    def predict(self, matrix):
        self.query_matrix = matrix.copy()
        return np.asarray([self.prediction])


class RideExtremaTests(unittest.TestCase):
    def fixture(self, lag=137):
        index = pd.date_range("2026-10-01", periods=240, freq="15min", tz="UTC")
        series = pd.Series(40 + np.arange(240) % 37, index=index, dtype=float)
        features = pd.DataFrame({"pmLevel": series, "clock": np.arange(240)}, index=index)
        query = index[200]
        clock = ForecastClock(int(query.timestamp()) + lag, int(query.timestamp()))
        cutoff = query - pd.Timedelta(minutes=330)
        return features, series, query, cutoff, clock

    def test_exact_partial_edge_bins_define_low_and_high(self):
        features, series, query, cutoff, clock = self.fixture()
        offsets = list(window_weights(clock.arrival_lead_seconds, clock.end_lead_seconds))
        self.assertEqual(offsets, list(range(7, 16)))
        series.loc[query + pd.Timedelta(minutes=15 * offsets[0])] = 3.125
        series.loc[query + pd.Timedelta(minutes=15 * offsets[-1])] = 209.375
        series.loc[query + pd.Timedelta(minutes=15 * (offsets[0] - 1))] = 1
        series.loc[query + pd.Timedelta(minutes=15 * (offsets[-1] + 1))] = 999
        target = extrema.window_extrema_targets(series, clock).loc[query]
        self.assertEqual(target["low"], 3.125)
        self.assertEqual(target["high"], 209.375)

    def test_missing_any_contributing_edge_bucket_invalidates_both_targets(self):
        features, series, query, cutoff, clock = self.fixture()
        series.loc[query + pd.Timedelta(minutes=105)] = np.nan
        self.assertTrue(extrema.window_extrema_targets(series, clock).loc[query].isna().all())

    def test_aligned_window_uses_eight_bins_after_arrival(self):
        features, series, query, cutoff, clock = self.fixture(lag=0)
        offsets = list(window_weights(clock.arrival_lead_seconds, clock.end_lead_seconds))
        self.assertEqual(offsets, list(range(7, 15)))
        series.loc[query + pd.Timedelta(minutes=90)] = 0
        series.loc[query + pd.Timedelta(minutes=225)] = 999
        target = extrema.window_extrema_targets(series, clock).loc[query]
        values = series.loc[[query + pd.Timedelta(minutes=15 * i) for i in offsets]]
        self.assertEqual(target["low"], values.min())
        self.assertEqual(target["high"], values.max())

    def test_predictions_are_literal_even_crossed_and_outside_a_separate_mean(self):
        features, series, query, cutoff, clock = self.fixture()
        estimator = RecordingEstimator((104.987654321, 71.123456789))
        with patch.object(extrema, "_make_estimator", return_value=estimator):
            output = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        self.assertTrue(output["available"])
        self.assertEqual(output["low"], estimator.prediction[0])
        self.assertEqual(output["high"], estimator.prediction[1])
        self.assertEqual(output["rawLow"], output["low"])
        self.assertEqual(output["rawHigh"], output["high"])
        self.assertTrue(output["crossing"])
        self.assertFalse(output["rangeOrdered"])
        self.assertFalse(output["outputSorting"])
        self.assertFalse(output["meanRecentering"])

    def test_training_hash_and_inputs_ignore_future_outcomes_and_embargoed_origins(self):
        features, series, query, cutoff, clock = self.fixture()
        estimator = RecordingEstimator()
        with patch.object(extrema, "_make_estimator", return_value=estimator):
            before = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        trained = estimator.training_response.copy()
        changed_series = series.copy()
        changed_series.loc[query + pd.Timedelta(minutes=15):] = 10000
        changed_features = features.copy()
        changed_features.loc[cutoff:query - pd.Timedelta(minutes=15), "clock"] = -999
        with patch.object(extrema, "_make_estimator", return_value=estimator):
            after = extrema.predict_ride_extrema(changed_features, changed_series, query, cutoff, clock)
        self.assertEqual(before["trainingDataHash"], after["trainingDataHash"])
        np.testing.assert_array_equal(trained, estimator.training_response)
        self.assertLess(before["latestTrainingOriginEpoch"], int(cutoff.timestamp()))
        self.assertLessEqual(before["latestTrainingOutcomeCompleteEpoch"], clock.issued_epoch)

    def test_maturity_mask_protects_against_accidental_late_cutoff(self):
        features, series, query, cutoff, clock = self.fixture()
        estimator = RecordingEstimator()
        with patch.object(extrema, "_make_estimator", return_value=estimator):
            output = extrema.predict_ride_extrema(features, series, query, query + pd.Timedelta(days=1), clock)
        self.assertLessEqual(output["latestTrainingOutcomeCompleteEpoch"], clock.issued_epoch)
        self.assertTrue(output["available"])
        self.assertFalse(output["numericalSelection"])

    def test_clock_anchor_mismatch_is_unavailable_without_fitting(self):
        features, series, query, cutoff, clock = self.fixture()
        with patch.object(extrema, "_make_estimator") as make:
            output = extrema.predict_ride_extrema(features, series, query - pd.Timedelta(minutes=15), cutoff, clock)
        self.assertFalse(output["available"])
        self.assertIsNone(output["low"])
        self.assertIsNone(output["high"])
        make.assert_not_called()

    def test_sparse_completed_history_is_unavailable_without_fitting(self):
        features, series, query, cutoff, clock = self.fixture()
        with patch.object(extrema, "_make_estimator") as make:
            output = extrema.predict_ride_extrema(features.iloc[160:], series.iloc[160:], query, cutoff, clock)
        self.assertFalse(output["available"])
        self.assertEqual(output["reason"], "insufficient_completed_training_origins")
        make.assert_not_called()

    def test_completed_training_changes_are_included_in_hash(self):
        features, series, query, cutoff, clock = self.fixture()
        estimator = RecordingEstimator()
        with patch.object(extrema, "_make_estimator", return_value=estimator):
            before = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
            series.iloc[40] += 1000
            after = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        self.assertNotEqual(before["trainingDataHash"], after["trainingDataHash"])

    def test_real_joint_model_is_ordered_without_postprocessing_and_handles_causal_nan(self):
        features, series, query, cutoff, clock = self.fixture()
        features.loc[features.index[30:45], "pmLevel"] = np.nan
        features.loc[query, "pmLevel"] = np.nan
        output = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        self.assertTrue(output["available"])
        self.assertTrue(output["rangeOrdered"])
        self.assertGreaterEqual(output["low"], series.min())
        self.assertLessEqual(output["high"], series.max())
        self.assertEqual(output["forecastClock"], clock.metadata())
        self.assertEqual(output["targetResolutionMinutes"], 15)
        self.assertFalse(output["prospectivelyValidated"])
        self.assertFalse(output["outputWidening"])

    def test_flat_history_naturally_produces_equal_learned_extrema(self):
        features, series, query, cutoff, clock = self.fixture()
        series[:] = 57.375
        output = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        self.assertTrue(output["available"])
        self.assertEqual(output["low"], 57.375)
        self.assertEqual(output["high"], 57.375)

    def test_real_training_imputation_ignores_future_and_embargoed_feature_changes(self):
        features, series, query, cutoff, clock = self.fixture()
        features.loc[query, "pmLevel"] = np.nan
        before = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        features.loc[cutoff:query - pd.Timedelta(minutes=15), "pmLevel"] = 100000
        features.loc[query + pd.Timedelta(minutes=15):, "clock"] = -100000
        after = extrema.predict_ride_extrema(features, series, query, cutoff, clock)
        self.assertEqual(before["low"], after["low"])
        self.assertEqual(before["high"], after["high"])
        self.assertEqual(before["trainingDataHash"], after["trainingDataHash"])


if __name__ == "__main__":
    unittest.main()
