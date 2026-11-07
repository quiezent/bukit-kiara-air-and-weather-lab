"""v25.2 arrival-change head: recall weights in the training loss of arrival_logistic_model (v2).

The weights (both fall classes 2, no change 1, both rise classes 1, mean-normalised) are the
frozen research choice class5_arrival_f2_r1; the published mass and its outcome stay literal.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import arrival_feature_inputs as features
import arrival_logistic_model as model
import device_display

DELTAS = (-50.0, -30.0, 0.0, 30.0, 50.0)


def prepared(cutoff, n=600, seed=7):
    rng = np.random.default_rng(seed)
    cls = rng.choice(5, size=n, p=[.05, .1, .7, .1, .05])
    issues = pd.DataFrame({"issueEpoch": cutoff - 86400 * (np.arange(n) % 12 + 1), "completeEpoch": cutoff - 1,
                           "featureSourceMaxEpoch": cutoff - 86400 * (np.arange(n) % 12 + 1),
                           "issueDay": [str(i % 12) for i in range(n)], "arrivalClass": cls,
                           "arrivalDelta": [DELTAS[c] for c in cls]})
    x = rng.normal(size=(n, len(model.FEATURE_COLUMNS)))
    x[:, 0] += 0.8 * (2 - cls)                       # an informative column; the rest is noise
    x[rng.random(x.shape) < .02] = np.nan
    return {"issues": issues, "features": x, "columns": list(model.FEATURE_COLUMNS)}


class ArrivalRecallWeightingTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = model.midnight_epoch(1791454200)
        self.prepared = prepared(self.cutoff)

    def test_version_files_and_policy_record_the_recall_weighting(self):
        self.assertEqual(model.MODEL_VERSION, "receipt_visible_logistic_arrival_change20_40_recall_f2_v2")
        self.assertEqual((model.ARTIFACT_NAME, model.STATUS_NAME, model.ISSUE_NAME),
                         ("arrival-logistic-change20-40-model-v2.pkl", "arrival-logistic-change20-40-status-v2.json",
                          "arrival-logistic-change-issued-v2.jsonl"))
        policy = model.selected_policy()
        self.assertFalse(policy["calibrated"])
        self.assertFalse(policy["publishingThresholdApplied"])
        self.assertEqual(policy["recallWeighting"]["classWeights"], {"fall": 2.0, "noChange": 1.0, "rise": 1.0})
        self.assertIsNone(policy["parameters"]["class_weight"])
        self.assertEqual(policy["parameters"]["C"], .1)
        w = model.class_sample_weights([0, 1, 2, 3, 4])
        self.assertTrue(np.allclose(w, np.array([2, 2, 1, 1, 1]) / 1.4))

    def test_fit_is_the_production_pipeline_with_mean_normalised_class_weights(self):
        artifact = model.fit_prepared(self.prepared, self.cutoff, fitted_epoch=self.cutoff)
        x = self.prepared["features"]
        y = self.prepared["issues"].arrivalClass.to_numpy(int)
        manual = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                               StandardScaler(), LogisticRegression(**model.PARAMETERS))
        weights = np.where(y <= 1, 2.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)       # sklearn 1.5 multi_class deprecation
            manual.fit(x, y, logisticregression__sample_weight=weights / weights.mean())
        self.assertTrue(np.array_equal(artifact["estimator"].predict_proba(x), manual.predict_proba(x)))
        metadata = artifact["metadata"]
        self.assertFalse(metadata["calibrated"])
        self.assertEqual(metadata["recallWeighting"], model.RECALL_WEIGHTING)
        self.assertAlmostEqual(metadata["normalisedClassWeights"]["fall40"], 2.0 / weights.mean())
        self.assertAlmostEqual(metadata["normalisedClassWeights"]["within20"], 1.0 / weights.mean())

    def test_weights_raise_fall_mass_and_the_outcome_stays_the_literal_argmax(self):
        weighted = model.fit_prepared(self.prepared, self.cutoff, fitted_epoch=self.cutoff)
        with patch.dict(model.CLASS_SAMPLE_WEIGHTS, {c: 1.0 for c in model.CLASSES}):
            plain = model.fit_prepared(self.prepared, self.cutoff, fitted_epoch=self.cutoff)
        x = self.prepared["features"]
        pw, pp = model.predict_probabilities(weighted, x), model.predict_probabilities(plain, x)
        self.assertGreater((pw[:, 0] + pw[:, 1]).mean(), (pp[:, 0] + pp[:, 1]).mean())
        self.assertLess(pw[:, 2].mean(), pp[:, 2].mean())
        for mass in pw[:20]:
            tails = model.tail_probabilities(mass)
            outcome = device_display.arrival_outcome({"available": True, **tails})
            expected = ("fall20", "rise20", "within20")[int(np.argmax([mass[0] + mass[1], mass[3] + mass[4], mass[2]]))]
            self.assertEqual(outcome["outcome"], expected)

    def test_live_result_carries_uncalibrated_recall_metadata(self):
        issue = 1791454200
        rows = [{"epoch": issue - age, "pm02": 100 - age / 300, "atmp": 29, "rhum": 65} for age in range(10800, -1, -180)]
        inputs = features.prepare_inputs(rows)
        reference = features.issue_features(inputs, issue)["freshPm25"]
        artifact = model.fit_prepared(self.prepared, self.cutoff, fitted_epoch=issue - 1)
        with patch.object(model, "_load_artifact", return_value=artifact):
            result = model.predict_arrival("unused", rows, issue, issue, reference, feature_inputs=inputs)
        self.assertTrue(result["available"], result)
        self.assertFalse(result["calibrated"])
        self.assertEqual(result["recallWeighting"], model.RECALL_WEIGHTING)
        self.assertEqual(result["probabilityRole"], "recall_weighted_uncalibrated_experimental")
        self.assertEqual(result["modelVersion"], model.MODEL_VERSION)
        self.assertNotIn("performanceEvidence", result)        # the dashboard attaches the research evidence
        self.assertEqual(model.PERFORMANCE_EVIDENCE["modelVersion"], model.MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
