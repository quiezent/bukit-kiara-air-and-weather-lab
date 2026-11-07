"""v25.2 first-crossing head: recall-weighted P1 logistic (first_crossing_logistic_model).

Recipe and label units, the nightly fit and live path on synthetic temporary inputs (built with
test_spatial_point_model.Inputs), unavailable behaviour, the issued record, the device projection,
and research parity on the 2026-10-13 snapshot (skipped where the research data is absent).
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import first_crossing_logistic_model as T
import forecast_freshness
import momentum_display
import rlcd_api
import spatial_final_models as M
import spatial_point_model as S
from test_rlcd_model_output import inputs as device_inputs
from test_spatial_point_model import HOUR, Inputs
from test_weather_contracts import ANCHOR

ROOT = Path(__file__).resolve().parents[1] / "app"
BASE = 1791561600     # 2026-10-10 00:00 MYT (Saturday)
ISSUE = BASE + 2 * HOUR + 125


def swing(epoch):
    """Calm three-hour spells alternating with 35 µg/m³ swings: all three first20 outcomes occur."""
    hour = ((epoch + 8 * HOUR) % 86400) / HOUR
    amplitude = 35.0 if int(hour) % 6 < 3 else 2.0
    return 90 + amplitude * math.sin(2 * math.pi * epoch / (2.5 * HOUR)) + 2 * math.sin(epoch / 1300.0)


def dashboard_reference(rows, issue):
    """The fresh reference attach_first_crossing_event_forecast passes in."""
    origin = issue // 900 * 900
    refs, _, _ = forecast_freshness.references(rows, [origin], issue - origin, issue)
    return float(refs[0]), max(int(r["epoch"]) for r in rows if r["epoch"] <= issue)


class RecipeTests(unittest.TestCase):
    def test_inputs_are_the_82_research_p1_columns_without_weather(self):
        self.assertEqual(len(T.INPUT_COLUMNS), 82)
        self.assertEqual(list(T.INPUT_COLUMNS[:28]), list(M.SENSOR28))
        self.assertEqual(T.INPUT_COLUMNS[28], "lin_logref")
        self.assertEqual(list(T.INPUT_COLUMNS[29:33]), list(M.CONTROLS))
        self.assertEqual(list(T.INPUT_COLUMNS[38:]), list(M.SP_STATION + M.SP_NET + M.SP_WIND))
        self.assertFalse([c for c in T.INPUT_COLUMNS if c.startswith(("w_", "c_", "t_"))])
        self.assertEqual((T.LOGISTIC_C, T.MAX_ITER, T.PROBABILITY_FLOOR, T.LOOKBACK_DAYS), (0.001, 2000, 1e-4, 28))

    def test_recall_weights_are_drop_12_none_1_rise_6_mean_normalised(self):
        w = T.class_sample_weights(np.array([-1, 0, 1, 0]))
        self.assertTrue(np.allclose(w, [12 / 5, 1 / 5, 6 / 5, 1 / 5]))
        self.assertAlmostEqual(w.mean(), 1.0)
        self.assertEqual(T.RECALL_WEIGHTING["classWeights"], {"fall": 12.0, "noChange": 1.0, "rise": 6.0})
        policy = T.selected_policy()
        self.assertFalse(policy["calibrated"])
        self.assertFalse(policy["publishingThresholdApplied"])
        self.assertEqual(policy["directionDecision"], "three_class_argmax_ties_unresolved")

    def test_clock_harmonics_and_weekend_flag(self):
        block = T.time_block([BASE + 6 * HOUR, BASE + 2 * 86400 + 18 * HOUR])   # Sat 06:00, Mon 18:00 MYT
        self.assertAlmostEqual(block.lin_h1s[0], 1.0)
        self.assertAlmostEqual(block.lin_h2c[0], -1.0)
        self.assertAlmostEqual(block.lin_h1s[1], -1.0)
        self.assertEqual(block.lin_weekend.tolist(), [1.0, 0.0])

    def test_preprocessing_fills_clips_and_standardises_from_training_rows_only(self):
        x = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 100.0], "b": [np.nan] * 5})
        prep = T.Prep(x)
        self.assertEqual(prep.med.tolist(), [3.0, 0.0])
        self.assertEqual(prep.sd[1], 1.0)
        z = prep(pd.DataFrame({"b": [7.0], "a": [1000.0]}))
        self.assertAlmostEqual(z[0, 0], (prep.hi[0] - prep.mu[0]) / prep.sd[0])
        self.assertEqual(z[0, 1], 0.0)                 # clipped to the (all-missing) training range

    def test_floor_gives_an_unseen_class_1e4_and_keeps_the_sum(self):
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(40, len(T.INPUT_COLUMNS))), columns=T.INPUT_COLUMNS)
        y = np.array([0, 1] * 20)
        lr = LogisticRegression(C=T.LOGISTIC_C, max_iter=T.MAX_ITER).fit(T.Prep(x)(x), y)
        p = T.predict_frame({"prep": T.Prep(x), "classifier": lr}, x)
        self.assertTrue(np.allclose(p[:, 0], 1e-4))
        self.assertTrue(np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-12))


class LabelTests(unittest.TestCase):
    def labels(self, issue, levels, reference=100.0, missing=()):
        origin = issue // 900 * 900
        buckets = {origin + 900 * k: level for k, level in enumerate(levels, start=1) if k not in missing}
        return T.first_crossing_labels(buckets, [issue], [reference]).iloc[0]

    def test_first_proxy_beyond_20_gives_the_sign(self):
        issue = BASE + 3 * HOUR                       # on a bucket boundary: proxies are bucket medians
        self.assertEqual(self.labels(issue, [105, 112, 121, 70, 70, 70]).y_first20, 1)
        self.assertEqual(self.labels(issue, [99, 80, 130, 130, 130, 130]).y_first20, -1)
        self.assertEqual(self.labels(issue, [119.9, 80.1, 100, 100, 100, 100]).y_first20, 0)
        self.assertEqual(self.labels(issue, [119.9] * 6).y_first20_complete, issue + 6 * 900)

    def test_interpolated_proxies_need_both_buckets_and_close_one_bucket_later(self):
        issue = BASE + 3 * HOUR + 450                 # halfway: proxy = mean of two buckets
        row = self.labels(issue, [100, 100, 100, 100, 100, 100, 141])
        self.assertEqual(row.y_first20, 1)            # (100 + 141) / 2 = 120.5 at +90
        self.assertEqual(row.y_first20_complete, issue // 900 * 900 + 7 * 900)
        self.assertTrue(math.isnan(self.labels(issue, [100] * 7, missing=(4,)).y_first20))
        self.assertTrue(math.isnan(self.labels(issue, [100] * 7, reference=float("nan")).y_first20))


class LiveModelTests(unittest.TestCase):
    """One synthetic two-day archive; the nightly fit is made once and copied per test."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.inputs = Inputs(cls.folder.name)
        start, end = BASE - 86400 - 5 * HOUR, BASE + 4 * HOUR
        cls.inputs.add_readings([(e, swing(e), 30.0 + math.sin(e / 9000.0), 70 + 10 * math.cos(e / 5000.0))
                                 for e in range(start, end + 1, 180)])
        cls.inputs.weather(start, end)
        cls.inputs.polls(start, end)
        cls.patches = [patch.object(T, "LOOKBACK_DAYS", 1), patch.object(T, "MINIMUM_TRAINING_ISSUES", 50),
                       patch.object(S, "TRAINING_LOOKBACK_DAYS", 1)]
        for patcher in cls.patches:
            patcher.start()
        cls.seed = Path(cls.folder.name) / "seed"
        cls.spatial_rows, _ = S.training_features(cls.inputs.db, BASE, archive_path=cls.inputs.archive, model_dir=cls.seed)
        cls.first = cls.refresh_into(cls.seed, ISSUE)

    @classmethod
    def tearDownClass(cls):
        for patcher in cls.patches:
            patcher.stop()
        cls.folder.cleanup()

    @classmethod
    def refresh_into(cls, model_dir, issue):
        with patch.object(T.time, "time", return_value=float(issue)):
            return T.refresh_model(cls.inputs.db, issue, archive_path=cls.inputs.archive, model_dir=model_dir)

    def setUp(self):
        self.model_dir = Path(self.folder.name) / self.id().rsplit(".", 1)[-1]
        shutil.copytree(self.seed, self.model_dir)

    def predict(self, issue=ISSUE, **kwargs):
        rows = self.inputs.rows(issue)
        reference, watermark = dashboard_reference(rows, issue)
        kwargs.setdefault("archive_path", self.inputs.archive)
        kwargs.setdefault("model_dir", self.model_dir)
        reference = kwargs.pop("reference", reference)
        return T.predict_event(self.inputs.db, rows, issue, watermark, reference, **kwargs)

    def test_nightly_fit_reuses_spatial_rows_and_trains_only_on_closed_labels(self):
        first = self.first
        self.assertTrue(first["available"], first)
        self.assertEqual(first["trainingCutoffEpoch"], BASE)
        self.assertTrue(all(count > 0 for count in first["trainingSupport"].values()), first["trainingSupport"])
        self.assertGreaterEqual(first["trainingIssueCount"], 200)
        self.assertLess(first["maxTrainingOutcomeEpoch"], BASE)
        self.assertLess(first["trainingMaxIssueEpoch"], BASE)
        self.assertFalse(first["calibrated"])
        self.assertEqual(first["recallWeighting"], T.RECALL_WEIGHTING)
        cache = first["featureCache"]
        spatial_origins = S.training_origins(BASE)
        self.assertEqual(cache["origins"], 287)          # (cutoff - 1 d, cutoff), both ends open
        self.assertEqual(cache["seededFromSpatialPointCache"], len(spatial_origins))
        self.assertEqual(cache["computedOrigins"], 287 - len(spatial_origins))
        # Seeded rows are the spatial builder's rows; recomputing them gives the same values.
        frame, again = T.training_features(self.inputs.db, BASE, archive_path=self.inputs.archive, model_dir=self.model_dir)
        self.assertEqual((again["computedOrigins"], again["seededFromSpatialPointCache"]), (0, 0))
        sample = [int(o) for o in spatial_origins[::17]]
        fresh = S.compute_training_features(self.inputs.db, sample, archive_path=self.inputs.archive)
        for column in S.CACHE_COLUMNS:
            a, b = frame.loc[sample, column].to_numpy(float), fresh[column].to_numpy(float)
            self.assertTrue(((np.isnan(a) & np.isnan(b)) | (a == b)).all(), column)
        artifact_path, status_path, cache_path = T._paths(self.model_dir)
        for path in (artifact_path, status_path, cache_path):
            self.assertTrue(path.exists(), path)
        self.assertEqual([p.name for p in self.model_dir.iterdir() if p.name.endswith(".pending")], [])
        self.assertTrue(self.refresh_into(self.model_dir, ISSUE + 600)["cacheReloaded"])

    def test_live_output_is_the_literal_weighted_logistic_and_its_argmax(self):
        result = self.predict()
        self.assertTrue(result["available"], result.get("unavailableReasons"))
        artifact = T.load_artifact(self.model_dir)
        live = S.live_issue_features(self.inputs.db, self.inputs.rows(ISSUE), ISSUE, archive_path=self.inputs.archive)
        x = T.design(pd.DataFrame([live["features"]], index=[ISSUE]))
        expected = T.predict_frame(artifact["model"], x)[0]
        got = [result["probabilityDrop"], result["probabilityNoCrossing"], result["probabilityRise"]]
        self.assertEqual(got, expected.tolist())
        self.assertAlmostEqual(sum(got), 1.0, places=12)
        self.assertTrue(min(got) >= T.PROBABILITY_FLOOR)
        winner = ("drop", "unresolved", "rise")[int(np.argmax(got))]
        self.assertEqual(result["diagnosticDirection"], winner)
        self.assertEqual(result["direction"], "unresolved")          # operational direction stays unqualified
        self.assertEqual(result["qualification"]["operationalUseEligible"], False)
        self.assertEqual((result["calibrated"], result["recallWeighting"]), (False, T.RECALL_WEIGHTING))
        self.assertEqual(result["modelVersion"], T.MODEL_VERSION)
        self.assertEqual(result["featureColumns"], list(T.INPUT_COLUMNS))
        self.assertEqual(len(result["featureValues"]), 82)
        self.assertEqual(result["validUntilEpoch"], ISSUE + 5400)
        self.assertTrue(math.isclose(result["referencePm"], live["features"]["s_freshPm25"], rel_tol=0, abs_tol=1e-7))
        self.assertEqual(result["modelIdentity"], artifact["identity"])
        self.assertIsNotNone(momentum_display.display_text(result))
        json.dumps(result, allow_nan=False)

    def test_recall_weights_move_mass_to_crossings_through_the_loss_only(self):
        history, _ = T.training_frame(self.inputs.db, BASE, archive_path=self.inputs.archive, model_dir=self.model_dir)
        weighted, _ = T.fit(history, BASE)
        with patch.dict(T.CLASS_WEIGHTS, {-1: 1.0, 0: 1.0, 1: 1.0}):
            plain, _ = T.fit(history, BASE)
        usable = history.loc[M.usable(history)]
        x = T.design(usable)
        pw, pp = T.predict_frame(weighted, x), T.predict_frame(plain, x)
        self.assertGreater(pw[:, 0].mean(), pp[:, 0].mean())
        self.assertGreater(pw[:, 2].mean(), pp[:, 2].mean())
        self.assertGreaterEqual((pw.argmax(1) != 1).sum(), (pp.argmax(1) != 1).sum())
        # Same inputs, same preprocessing: only the classifier's training weights differ.
        self.assertTrue(np.array_equal(weighted["prep"].mu, plain["prep"].mu))

    def test_missing_inputs_or_model_are_unavailable_with_a_reason(self):
        def unavailable(result, reason):
            self.assertFalse(result["available"])
            self.assertEqual(result["reason"], reason)
            self.assertNotIn("probabilityDrop", result)
            self.assertIsNone(momentum_display.display_text(result))
            self.assertFalse(result["calibrated"])
        rows = self.inputs.rows(ISSUE)
        reference, _ = dashboard_reference(rows, ISSUE)
        unavailable(self.predict(model_dir=Path(self.folder.name) / "empty"), "background_model_collecting")
        missing = Path(self.folder.name) / "absent.sqlite3"
        unavailable(self.predict(archive_path=missing), "neighbour_archive_unavailable")
        self.assertFalse(missing.exists())
        unavailable(self.predict(reference=reference + 0.5), "fresh_sensor_reference_mismatch")
        unavailable(self.predict(reference=float("nan")), "fresh_sensor_reference_unavailable")
        artifact = T.load_artifact(self.model_dir)
        later = deepcopy(artifact)
        later["metadata"]["fittedAtEpoch"] = ISSUE + 1
        unavailable(self.predict(artifact=later), "model_not_yet_fitted_at_issue")
        self.assertTrue(self.predict(artifact=later, allow_reconstruction=True)["available"])
        stale = {**artifact, "trainingCutoffEpoch": BASE - 86400}
        unavailable(self.predict(artifact=stale), "first20_model_not_fitted_for_issue_day")

    def test_recipe_change_rejects_the_artifact_and_thin_support_is_unavailable(self):
        self.assertIsNotNone(T.load_artifact(self.model_dir))
        changed = {**T.model_identity(), "identitySha256": "changed"}
        with patch.object(T, "model_identity", return_value=changed):
            self.assertIsNone(T.load_artifact(self.model_dir))
        with patch.object(T, "MINIMUM_TRAINING_ISSUES", 10 ** 6):
            status = self.refresh_into(self.model_dir, BASE + 86400 + 600)
            self.assertFalse(status["available"])
            self.assertEqual(status["reason"], "insufficient_completed_training_issues")
            self.assertEqual(status["featureCache"]["computedOrigins"], 287)

    def test_issued_record_is_written_once_for_a_fresh_publication(self):
        result = self.predict()
        with tempfile.TemporaryDirectory() as directory, patch.object(T, "PRIVATE", Path(directory)):
            with self.assertRaises(ValueError):
                T.record_issued_prediction(result, recorded_epoch=ISSUE + 121)
            self.assertTrue(T.record_issued_prediction(result, recorded_epoch=ISSUE + 20,
                                                       input_lineage={"snapshotId": "snap"}))
            self.assertFalse(T.record_issued_prediction(result, recorded_epoch=ISSUE + 30))
            path = Path(directory) / "first20-logistic-issued-v1.jsonl"
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(saved["isOriginallyIssued"])
        self.assertEqual(saved["inputSnapshotId"], "snap")
        self.assertEqual(saved["probabilityDrop"], result["probabilityDrop"])
        self.assertEqual(saved["issuedCohort"]["modelVersion"], T.MODEL_VERSION)
        self.assertEqual((saved["issuedCohort"]["calibrated"], saved["issuedCohort"]["recallWeighting"]),
                         (False, T.RECALL_WEIGHTING))
        self.assertEqual(saved["predictionRole"], "genuinely_issued_live_first20_logistic_prediction")

    def test_device_projection_keeps_field_names_text_and_budget(self):
        result = momentum_display.attach_display_text(self.predict())
        reading, analysis, weather, rows = device_inputs()
        event = {**result, "forecastIssuedEpoch": ANCHOR, "validUntilEpoch": ANCHOR + 5400}
        analysis["airWindow"]["firstCrossingEventForecast"] = event
        payload = rlcd_api.build_payload(reading, analysis, weather, ANCHOR + 20, history_rows=rows)
        first20 = payload["forecast"]["near90"]["first20"]
        self.assertTrue(first20["available"])
        self.assertEqual(first20["display_text"], momentum_display.display_text(event))
        self.assertEqual((first20["drop_probability"], first20["none_probability"], first20["rise_probability"]),
                         (event["probabilityDrop"], event["probabilityNoCrossing"], event["probabilityRise"]))
        self.assertEqual((first20["model"], first20["calibrated"], first20["direction"]), (T.MODEL_VERSION, False, "unresolved"))
        self.assertEqual(first20["diagnostic_direction"], event["diagnosticDirection"])
        self.assertLessEqual(len(json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()),
                             rlcd_api.MAX_PAYLOAD_BYTES)


class DashboardWiringTests(unittest.TestCase):
    def test_service_publishes_and_refits_the_logistic_and_keeps_hgb_v2_importable(self):
        import BukitKiara_Dashboard as app
        import fresh_event_model_v2
        source = (ROOT / "BukitKiara_Dashboard.py").read_text(encoding="utf-8")
        self.assertIs(app.first_crossing_event_model, T)
        self.assertNotIn("fresh_event_model", source)
        self.assertIn("first_crossing_event_model.refresh_model(DB_PATH", source)
        self.assertIn("first_crossing_event_model.record_issued_prediction(event", source)
        self.assertEqual(fresh_event_model_v2.MODEL_VERSION, "fresh_sensor_hgb_direct_first20_90min_v2")
        self.assertEqual(app.DASHBOARD_BUILD, "2026-10-14-eager-event-models-v25.2.0-public.1")

    def test_attach_passes_the_dashboard_reference_and_adds_evidence(self):
        import BukitKiara_Dashboard as app
        rows = [{"epoch": ISSUE - 30 - 180 * k, "pm02": 80.0} for k in range(5)][::-1]
        captured = {}

        def fake(db, rows_, issue, watermark, reference):
            captured.update(issue=issue, watermark=watermark, reference=reference)
            return T._unavailable(T._base(issue), "background_model_collecting")
        with patch.object(app.first_crossing_event_model, "predict_event", side_effect=fake):
            air = app.attach_first_crossing_event_forecast({}, rows, ISSUE)
        self.assertEqual(captured, {"issue": ISSUE, "watermark": ISSUE - 30, "reference": 80.0})
        event = air["firstCrossingEventForecast"]
        self.assertEqual((event["modelVersion"], event["calibrated"]), (T.MODEL_VERSION, False))
        self.assertIsNone(event["displayText"])
        unavailable = app.attach_first_crossing_event_forecast({}, [], ISSUE)["firstCrossingEventForecast"]
        self.assertEqual((unavailable["modelVersion"], unavailable["recallWeighting"]), (T.MODEL_VERSION, T.RECALL_WEIGHTING))


def research_data():
    needed = ("models/dataset.parquet", "models/spatial/spatial_features_lag300.parquet",
              "backtest/runs/eager_frozen__daily-28d__all.parquet")
    for candidate in (os.environ.get("TTDI_RESEARCH_DATA"), ROOT.parent / "data"):
        if candidate and all((Path(candidate) / name).exists() for name in needed):
            return Path(candidate)
    return None


DATA = research_data()


@unittest.skipIf(DATA is None, "research data (data/models, data/backtest) not present")
class ResearchParityTests(unittest.TestCase):
    def test_fit_on_the_research_rows_reproduces_the_frozen_predictions_exactly(self):
        base = [c for c in S.CACHE_COLUMNS if not c.startswith("sp_")] + ["y_first20", "y_first20_complete"]
        frame = pd.read_parquet(DATA / "models" / "dataset.parquet", columns=base).join(
            pd.read_parquet(DATA / "models" / "spatial" / "spatial_features_lag300.parquet", columns=list(T.SPATIAL_COLUMNS)))
        frozen = pd.read_parquet(DATA / "backtest" / "runs" / "eager_frozen__daily-28d__all.parquet")
        columns = [f"first20_p1_f12_r6::{c}" for c in ("p_first_drop", "p_first_none", "p_first_rise")]
        cutoff = 1791820800                          # 2026-10-13 00:00 MYT
        model, meta = T.fit(frame.loc[T.training_origins(cutoff)], cutoff)
        self.assertTrue(meta["available"])
        test = frame.loc[(frame.index >= cutoff) & (frame.index < cutoff + 86400)]
        test = test.loc[M.usable(test)]
        mine = T.predict_frame(model, T.design(test))
        theirs = frozen.reindex(test.index)[columns].to_numpy(float)
        self.assertEqual(len(test), 263)
        self.assertEqual(float(np.abs(mine - theirs).max()), 0.0)

    def test_recorded_end_to_end_parity_evidence(self):
        path = ROOT / "research" / "first20_logistic_v25_2" / "parity_check.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(evidence["days"]), {"2026-10-08", "2026-10-09", "2026-10-13"})
        clocks = 0
        for name, day in evidence["days"].items():
            first20 = day["first20"]
            self.assertEqual(first20["trainingInputParity"]["mismatchedCells"], 0)
            self.assertEqual(first20["labelParity"]["mismatchedCells"], 0)
            for block in ("gridIssues", "issuedClocks"):
                self.assertLessEqual(first20[block]["maxAbsDiff"], 1e-12)
                self.assertEqual(first20[block]["argmaxAgreement"], 1.0)
                self.assertEqual(first20[block]["serverAvailableResearchUnavailable"], 0)
                if name == "2026-10-08":     # readings before the receipt ledger (13:12 MYT): see parity_check.py
                    self.assertEqual(set(first20[block]["unavailableReasons"]), {"fresh_sensor_reference_mismatch"})
                else:
                    self.assertEqual(first20[block]["serverUnavailableResearchAvailable"], 0)
            for block in ("gridIssues", "issuedClocks"):
                clocks += first20["namedWindow"][block]["bothAvailable"]
            recipe = day["class5"]["recipeParityOnIdenticalInputs"]
            self.assertEqual((recipe["weightedMaxAbsDiff"], recipe["unweightedMaxAbsDiff"]), (0.0, 0.0))
        self.assertGreaterEqual(clocks, 30)


if __name__ == "__main__":
    unittest.main()
