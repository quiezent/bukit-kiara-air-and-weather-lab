"""v25 spatial point model: live feature builder, research parity, publication wiring,
unavailable behaviour and the atomic nightly refit.

Synthetic temporary inputs except ResearchParityTests, which reads the 2026-10-13 research
snapshot read-only (mode=ro) and is skipped where that snapshot is absent.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from contextlib import ExitStack, closing
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import fresh_model_output
import observation_provenance
import spatial_final_models as M
import spatial_live_features as SL
import spatial_point_model as S
from research.external_airgradient.neighbor_history import open_archive

ROOT = Path(__file__).resolve().parents[1] / "app"
BASE = 1791561600   # 2026-10-10 00:00 MYT
HOUR = 3600
STATIONS = list(SL.STATIONS)


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def pm_signal(epoch):
    return 45 + 15 * math.sin(2 * math.pi * epoch / 86400) + 4 * math.sin(epoch / 1700.0)


class Inputs:
    """A temporary live DB (readings, receipt ledger, issued weather) and a neighbour archive."""

    def __init__(self, folder):
        self.db = Path(folder) / "live.db"
        self.archive = Path(folder) / "neighbors.sqlite3"
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, pm01 REAL,"
                         " pm02 REAL, pm10 REAL, atmp REAL, rhum REAL, heatindex REAL)")
            conn.execute("CREATE TABLE weather_forecast_runs(fetched_epoch INTEGER PRIMARY KEY,"
                         " fetched_timestamp TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL)")
            observation_provenance.init_schema(conn)
            conn.commit()
        open_archive(self.archive).close()

    def connect(self, path):
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA synchronous=OFF")
        return closing(conn)

    def add_readings(self, rows, *, latency=30.0, tracked=True):
        with self.connect(self.db) as conn:
            for epoch, pm02, atmp, rhum in rows:
                conn.execute("INSERT INTO readings(epoch,timestamp,pm02,atmp,rhum) VALUES(?,?,?,?,?)",
                             (int(epoch), "t", pm02, atmp, rhum))
                if not tracked:
                    continue
                text = json.dumps({"epoch": int(epoch), "pm02": pm02, "atmp": atmp, "rhum": rhum})
                digest = hashlib.sha256(text.encode()).hexdigest()
                conn.execute("INSERT OR IGNORE INTO sensor_observation_revisions VALUES(?,?,?,?)",
                             (digest, int(epoch), digest, text))
                received = float(epoch) + latency
                cursor = conn.execute("INSERT INTO sensor_observation_receipts(revision_id,observed_epoch,received_epoch,"
                                      "persisted_epoch,available_epoch) VALUES(?,?,?,?,?)",
                                      (digest, int(epoch), received, received, received))
                conn.execute("INSERT INTO sensor_observation_visibility VALUES(?,?)", (cursor.lastrowid, received + 0.5))
            conn.commit()

    def readings(self, start, end, *, step=180, latency=30.0):
        self.add_readings([(e, pm_signal(e), 30.0 + math.sin(e / 9000.0), 70 + 10 * math.cos(e / 5000.0))
                           for e in range(int(start), int(end) + 1, step)], latency=latency)

    def add_weather(self, fetched, *, latency=3.0, turn=0.0):
        self.add_weather_runs([(fetched, latency, turn)])

    def add_weather_runs(self, runs):
        with self.connect(self.db) as conn:
            for fetched, latency, turn in runs:
                first = (fetched // HOUR) * HOUR - HOUR
                hourly = [{"epoch": first + HOUR * k, "time": "x", "temperature_2m": 30.0,
                           "wind_speed_10m": 6.0 + 0.4 * k, "wind_direction_10m": (200.0 + turn + 9 * k) % 360,
                           "wind_speed_925hPa": 14.0 + 0.2 * k, "wind_direction_925hPa": (230.0 + turn + 5 * k) % 360}
                          for k in range(14)]
                text = json.dumps({"fetchedEpoch": fetched, "source": S.WEATHER_SOURCE, "hourly": hourly})
                received = float(fetched) + latency
                conn.execute("INSERT INTO weather_forecast_runs VALUES(?,?,?,?)", (int(fetched), "t", S.WEATHER_SOURCE, text))
                digest = hashlib.sha256(text.encode()).hexdigest()
                conn.execute("INSERT INTO source_forecast_revisions VALUES(?,?,?,?,?)",
                             (digest, "weather_forecast_runs", int(fetched), digest, gzip.compress(text.encode())))
                cursor = conn.execute("INSERT INTO source_forecast_receipts(revision_id,source_table,source_fetched_epoch,"
                                      "received_epoch,persisted_epoch,available_epoch) VALUES(?,?,?,?,?,?)",
                                      (digest, "weather_forecast_runs", int(fetched), received, received, received))
                conn.execute("INSERT INTO source_forecast_visibility VALUES(?,?)", (cursor.lastrowid, received + 0.2))
            conn.commit()

    def weather(self, start, end):
        self.add_weather_runs([(hour + 300, 3.0, 0.0) for hour in range(int(start) // HOUR * HOUR, int(end) + 1, HOUR)
                               if start <= hour + 300 <= end])

    def add_polls(self, end_epoch, values, *, latency=20.0, kind="current", bucket_seconds=0):
        self.add_poll_batch([(end_epoch, values, latency, kind, bucket_seconds)])

    def add_poll_batch(self, batch):
        with self.connect(self.archive) as conn:
            for end_epoch, values, latency, kind, bucket_seconds in batch:
                received = float(end_epoch) + latency
                cursor = conn.execute("INSERT INTO requests(kind,url,started_epoch,received_epoch,recorded_epoch,"
                                      "http_status,sha256,raw_path,body_bytes,headers_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                      ("current_pm25", "https://example.invalid", received - 1, received, received + 0.1,
                                       200, "0" * 64, "x", 0, "{}"))
                conn.executemany("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)",
                                 [(int(location), "pm25", int(end_epoch) - bucket_seconds, bucket_seconds, float(value),
                                   "ug/m3", kind, 1, None, cursor.lastrowid) for location, value in values.items()])
            conn.commit()

    def polls(self, start, end, *, step=300, stations=STATIONS):
        self.add_poll_batch([(epoch, {location: 0.8 * pm_signal(epoch - 600 * i) + 3 * i
                                      for i, location in enumerate(stations)}, 20.0, "current", 0)
                             for epoch in range(int(start), int(end) + 1, step)])

    def fill(self, start, end, *, weather_end=None, polls_end=None, stations=STATIONS):
        self.readings(start, end)
        self.weather(start, end if weather_end is None else weather_end)
        self.polls(start, end if polls_end is None else polls_end, stations=stations)

    def rows(self, issue, lookback=S.SENSOR_WINDOW_SECONDS):
        with closing(sqlite3.connect(self.db)) as conn:
            return observation_provenance.compatible_sensor_rows(conn, int(issue), int(issue) - lookback, int(issue))

    def live(self, issue, **kwargs):
        return S.live_issue_features(self.db, self.rows(issue), issue, archive_path=self.archive, **kwargs)


class TemporaryInputs(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.inputs = Inputs(self.folder.name)
        self.model_dir = Path(self.folder.name) / "private"


class VendoredSourceTests(unittest.TestCase):
    def test_model_code_is_byte_identical_to_the_research_final_modules(self):
        for name, (research, digest) in S.VENDORED_SOURCES.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
            original = ROOT.parent / research
            if original.exists():
                self.assertEqual(original.read_bytes(), (ROOT / name).read_bytes(), research)
        self.assertEqual(M.FINAL_POINT_VARIANT, "P1_no_weather")
        self.assertEqual(M.FINAL_LE70_VARIANT, "E1_spatial_residual")
        self.assertEqual(len(S.INPUT_COLUMNS), 81)
        self.assertFalse([c for c in S.INPUT_COLUMNS if c.startswith(("w_", "c_", "rp_"))])


class FeatureBuilderTests(TemporaryInputs):
    def setUp(self):
        super().setUp()
        self.inputs.fill(BASE - 6 * HOUR, BASE + 6 * HOUR)
        self.issue = BASE + 3 * HOUR + 125

    def test_inputs_received_after_the_issue_never_change_its_features(self):
        before = self.inputs.live(self.issue)
        self.assertIsNotNone(before["features"], before["reason"])
        # A reading, a neighbour poll and a weather run, all received after the issue. The poll
        # falls in the newest neighbour bucket that is otherwise visible at the issue.
        late_poll = BASE + 2 * HOUR + 2460
        self.inputs.add_readings([(self.issue - 40, 400.0, 31.0, 60.0)], latency=100)
        self.inputs.add_polls(late_poll, {STATIONS[0]: 500.0}, latency=self.issue + 30 - late_poll)
        self.inputs.add_weather(self.issue - 20, latency=30, turn=150.0)
        after = self.inputs.live(self.issue)
        for key, value in before["features"].items():
            self.assertTrue(same(value, after["features"][key]), key)
        self.assertEqual(after["lineage"]["issuedWindInput"]["fetchedEpoch"],
                         before["lineage"]["issuedWindInput"]["fetchedEpoch"])
        # Once received they are used: the same records are visible to a later issue.
        later = self.inputs.live(self.issue + 120)
        self.assertEqual(later["lineage"]["issuedWindInput"]["fetchedEpoch"], self.issue - 20)
        table = S.live_sensor_table(self.inputs.rows(self.issue + 120), self.issue + 120)
        self.assertIn(self.issue - 40, set(table.epoch))
        self.assertNotIn(self.issue - 40, set(S.live_sensor_table(self.inputs.rows(self.issue), self.issue).epoch))
        polls = S.read_neighbour_polls(self.inputs.archive, self.issue - 3 * HOUR, self.issue)
        self.assertIn(late_poll, set(polls.end_epoch))
        self.assertTrue((polls.loc[polls.end_epoch == late_poll, "receipt_epoch"] > self.issue).all())
        self.assertGreater(before["features"]["sp_setia_age"], 0)
        self.assertLessEqual(before["features"]["sp_setia_age"], 30)

    def test_training_view_equals_live_view_when_inputs_arrived_on_time(self):
        issues = [BASE + HOUR, BASE + 2 * HOUR + 900, BASE + 3 * HOUR + 125, BASE + 5 * HOUR + 2700]
        batch = S.compute_training_features(self.inputs.db, issues, archive_path=self.inputs.archive)
        for issue in issues:
            live = self.inputs.live(issue)
            self.assertIsNotNone(live["features"], live["reason"])
            for column in S.CACHE_COLUMNS:
                self.assertTrue(same(float(batch.loc[issue, column]), float(live["features"][column])), (issue, column))
        self.assertTrue(batch["m_fresh_available"].all())
        self.assertEqual(int(batch["sp_count"].min()), 6)

    def test_spatial_block_uses_receipt_clocks_and_the_research_functions(self):
        live = self.inputs.live(self.issue)
        features = live["features"]
        polls = S.read_neighbour_polls(self.inputs.archive, self.issue - 3 * HOUR, self.issue)
        table = S.live_sensor_table(self.inputs.rows(self.issue), self.issue)
        direct = SL.spatial_features(self.issue, SL.NeighbourBuckets(polls.loc[polls.receipt_epoch <= self.issue]),
                                     SL.TTDIBuckets(table), features)
        for column in S.SPATIAL_COLUMNS:
            self.assertTrue(same(float(direct[column]), features[column]), column)
        self.assertTrue(all(np.isfinite(features[f"w_wind10_{part}_{lead}"]) for part in "uv" for lead in (0, 30, 60, 90)))
        self.assertEqual(live["lineage"]["neighbourInputs"]["stationsWithBucketWithin90Minutes"], 6)
        self.assertLessEqual(live["lineage"]["issuedWindInput"]["visibleEpoch"], self.issue)
        self.assertLessEqual(live["lineage"]["neighbourInputs"]["latestReceiptEpoch"], self.issue)

    def test_legacy_rows_are_visible_60_seconds_after_observation_and_unconfirmed_receipts_never(self):
        issue = BASE + 3 * HOUR
        rows = [{"epoch": issue - 30, "pm02": 50.0, "atmp": 30.0, "rhum": 70.0},
                {"epoch": issue - 90, "pm02": 51.0, "atmp": 30.0, "rhum": 70.0},
                {"epoch": issue - 150, "pm02": 52.0, "atmp": 30.0, "rhum": 70.0,
                 "_provenance": {"receiptKnown": True, "availableEpoch": None}}]
        table = S.live_sensor_table(rows, issue)
        self.assertEqual(list(table.epoch), [issue - 90])
        self.assertEqual(float(table.visible_epoch.iloc[0]), issue - 90 + S.LEGACY_SENSOR_VISIBILITY_SECONDS)

    def test_history_rows_received_after_the_backfill_keep_their_real_receipt(self):
        end = BASE + 4 * HOUR + 2850     # off the synthetic 5-minute poll grid
        self.inputs.add_polls(end, {STATIONS[1]: 77.0}, latency=4000, kind="history", bucket_seconds=900)
        polls = S.read_neighbour_polls(self.inputs.archive, end - HOUR, end)
        row = polls.loc[(polls.end_epoch == end) & (polls.location_id == STATIONS[1])].iloc[0]
        self.assertEqual(row.kind, "current")
        self.assertEqual(row.receipt_epoch, end + 4000 + 0.1)


def fit_artifact(history, cutoff, fitted_epoch=None):
    point, le70, support = S.fit(history, cutoff)
    return {"modelVersion": S.MODEL_VERSION, "adapterVersion": S.ADAPTER_VERSION, "identity": S.model_identity(),
            "trainingCutoffEpoch": cutoff, "point": point, "le70": le70,
            "metadata": {**support, **S._training_window(history, cutoff),
                         "fittedAtEpoch": cutoff if fitted_epoch is None else fitted_epoch}}


class RefitTests(unittest.TestCase):
    """One synthetic 3-day archive; the first nightly fit is made once and copied per test."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.inputs = Inputs(cls.folder.name)
        cls.inputs.fill(BASE - 2 * 86400 - 4 * HOUR, BASE + 86400 + 3 * HOUR)
        cls.patches = [patch.object(S, "TRAINING_LOOKBACK_DAYS", 2), patch.object(S, "MINIMUM_TRAINING_ORIGINS", 30)]
        for patcher in cls.patches:
            patcher.start()
        cls.seed = Path(cls.folder.name) / "seed"
        cls.first = cls.refresh_into(cls.seed, BASE + HOUR)

    @classmethod
    def tearDownClass(cls):
        for patcher in cls.patches:
            patcher.stop()
        cls.folder.cleanup()

    @classmethod
    def refresh_into(cls, model_dir, issue):
        # The worker fits "now"; the synthetic days lie in the past, so pin the fit clock.
        with patch.object(S.time, "time", return_value=float(issue)):
            return S.refresh_model(cls.inputs.db, issue, archive_path=cls.inputs.archive, model_dir=model_dir)

    def setUp(self):
        self.model_dir = Path(self.folder.name) / self.id().rsplit(".", 1)[-1]
        shutil.copytree(self.seed, self.model_dir)

    def refresh(self, issue):
        return self.refresh_into(self.model_dir, issue)

    def test_nightly_refit_is_atomic_incremental_and_published_literally(self):
        origins = S.training_origins(BASE)
        first = self.first
        self.assertTrue(first["available"], first)
        self.assertFalse(first["cacheReloaded"])
        self.assertEqual(first["trainingCutoffEpoch"], BASE)
        self.assertEqual(first["featureCache"], {**first["featureCache"], "origins": len(origins),
                                                 "cachedOrigins": 0, "computedOrigins": len(origins)})
        self.assertLess(first["latestTrainingOutcomeCompleteEpoch"], BASE)
        self.assertLess(first["latestTrainingOriginEpoch"], BASE - M.EMBARGO_MINUTES * 60)
        self.assertGreaterEqual(min(first["trainingOrigins"].values()), 30)
        artifact_path, status_path, cache_path = S._paths(self.model_dir)
        for path in (artifact_path, status_path, cache_path):
            self.assertTrue(path.exists(), path)
        self.assertEqual(sorted(p.name for p in self.seed.iterdir() if p.name.endswith(".pending")), [])
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(status["trainingCutoffEpoch"], BASE)
        self.assertEqual(status["modelIdentity"]["identitySha256"], S.model_identity()["identitySha256"])
        self.assertTrue(self.refresh(BASE + 2 * HOUR)["cacheReloaded"])

        issue = BASE + 2 * HOUR + 125
        result = S.predict(self.inputs.db, self.inputs.rows(issue), issue, archive_path=self.inputs.archive,
                           model_dir=self.model_dir)
        self.assertTrue(result["available"], result.get("reason"))
        artifact = S.load_artifact(self.model_dir)
        features = self.inputs.live(issue)["features"]
        expected = M.predict_point(artifact["point"], features)
        for field, head in (("arrivalPoint", "pred_arrival"), ("trailMeanPoint", "pred_ride_mean"),
                            ("trailMinimumPoint", "pred_ride_min"), ("trailMaximumPoint", "pred_ride_max")):
            self.assertEqual(result[field], expected[head])
        self.assertEqual(result["featureColumns"], list(S.INPUT_COLUMNS))
        self.assertEqual(len(result["featureValues"]), 81)
        le70 = result["le70"]
        self.assertTrue(le70["available"], le70)
        self.assertEqual(le70["chanceMeanAtOrBelowCutoff"], M.predict_le70(artifact["le70"], features))
        self.assertEqual(le70["rideMeanAnchorUgM3"], result["trailMeanPoint"])
        self.assertEqual((le70["startEpoch"], le70["endEpoch"], le70["mode"]), (issue + 5400, issue + 12600, "initial90"))
        self.assertEqual(le70["modelVersion"], S.LE70_MODEL_VERSION)

        # Next Malaysia day: only the newest day of origins is computed, the rest come from the cache.
        second = self.refresh(BASE + 86400 + 600)
        self.assertTrue(second["available"], second)
        self.assertEqual(second["trainingCutoffEpoch"], BASE + 86400)
        self.assertEqual(second["featureCache"]["computedOrigins"], 96)
        self.assertEqual(second["featureCache"]["cachedOrigins"], len(origins) - 96)
        self.assertEqual(sorted(p.name for p in self.model_dir.iterdir() if p.name.endswith(".pending")), [])
        self.assertEqual(S.predict(self.inputs.db, self.inputs.rows(issue), issue, archive_path=self.inputs.archive,
                                   model_dir=self.model_dir)["reason"], "spatial_point_model_not_fitted_for_issue_day")

    def test_failed_swap_keeps_the_previous_artifact(self):
        artifact_path, _, _ = S._paths(self.model_dir)
        before = artifact_path.read_bytes()
        original = Path.replace

        def refuse(self_path, target):
            if Path(target).name == S.ARTIFACT_NAME:
                raise OSError("simulated swap failure")
            return original(self_path, target)

        with patch.object(Path, "replace", refuse), self.assertRaises(OSError):
            self.refresh(BASE + 86400 + 600)
        self.assertEqual(artifact_path.read_bytes(), before)
        self.assertEqual(S.load_artifact(self.model_dir)["trainingCutoffEpoch"], BASE)

    def test_recipe_change_rejects_the_old_artifact_and_thin_support_is_unavailable(self):
        self.assertEqual(S.load_artifact(self.model_dir)["trainingCutoffEpoch"], BASE)
        changed = {**S.model_identity(), "identitySha256": "changed"}
        with patch.object(S, "model_identity", return_value=changed):
            self.assertIsNone(S.load_artifact(self.model_dir))
        with patch.object(S, "MINIMUM_TRAINING_ORIGINS", 10 ** 6):
            status = self.refresh(BASE + 86400 + 600)
            self.assertFalse(status["available"])
            self.assertEqual(status["reason"], "insufficient_completed_training_origins")
            self.assertTrue(self.refresh(BASE + 86400 + 900)["cacheReloaded"])
            issue = BASE + 86400 + HOUR
            result = S.predict(self.inputs.db, self.inputs.rows(issue), issue, archive_path=self.inputs.archive,
                               model_dir=self.model_dir)
        self.assertEqual(result["reason"], "insufficient_completed_training_origins")
        self.assertIsNone(result["arrivalPoint"])
        self.assertFalse(result["le70"]["available"])


class UnavailableTests(TemporaryInputs):
    def assert_unavailable(self, result, reason):
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], reason)
        for field in ("arrivalPoint", "trailMeanPoint", "trailMinimumPoint", "trailMaximumPoint", "rawOutputUgM3"):
            self.assertIsNone(result[field])
        self.assertFalse(result["le70"]["available"])
        self.assertIsNone(result["le70"]["chanceMeanAtOrBelowCutoff"])
        self.assertEqual(result["le70"]["reason"], reason)
        cycling = S.cycling_window_output(result, result["forecastIssuedEpoch"])
        self.assertIsNone(cycling["chanceMeanAtOrBelowCutoff"])
        air = fresh_model_output.attach_outputs({}, result, result["forecastIssuedEpoch"])
        for head in ("arrival", "trail"):
            self.assertFalse(air[head]["available"])
            self.assertIsNone(air[head]["point"])     # never the current reading
            self.assertEqual(air[head]["reason"], reason)
        self.assertIsNone(air["trail"]["rideExtrema"]["low"])

    def predict(self, issue, **kwargs):
        kwargs.setdefault("archive_path", self.inputs.archive)
        kwargs.setdefault("model_dir", self.model_dir)
        return S.predict(self.inputs.db, self.inputs.rows(issue), issue, **kwargs)

    def test_stale_reference_wind_neighbours_archive_and_model_each_make_every_head_unavailable(self):
        self.inputs.readings(BASE - 4 * HOUR, BASE + 8 * HOUR)
        self.inputs.weather(BASE - 4 * HOUR, BASE + 5 * HOUR)
        self.inputs.polls(BASE - 4 * HOUR, BASE + 2 * HOUR)
        self.inputs.polls(BASE + 4 * HOUR, BASE + 8 * HOUR)
        self.assert_unavailable(self.predict(BASE + 8 * HOUR + 600), "fresh_sensor_reference_unavailable_or_stale")
        self.assert_unavailable(self.predict(BASE + 7 * HOUR + 1800), "issued_wind_unavailable_or_older_than_2h")
        self.assert_unavailable(self.predict(BASE + 3 * HOUR + 1900), "neighbour_inputs_unavailable_or_stale")
        missing = Path(self.folder.name) / "absent.sqlite3"
        result = self.predict(BASE + HOUR, archive_path=missing)
        self.assert_unavailable(result, "neighbour_archive_unavailable")
        self.assertFalse(missing.exists())
        self.assert_unavailable(self.predict(BASE + HOUR), "spatial_point_model_asset_unavailable")
        stale_day = {"modelVersion": S.MODEL_VERSION, "adapterVersion": S.ADAPTER_VERSION, "identity": S.model_identity(),
                     "trainingCutoffEpoch": BASE - 86400, "point": object(), "le70": None,
                     "metadata": {"available": True, "fittedAtEpoch": BASE - 86000,
                                  "latestTrainingOutcomeCompleteEpoch": BASE - 90000}}
        self.assert_unavailable(self.predict(BASE + HOUR, artifact=stale_day), "spatial_point_model_not_fitted_for_issue_day")
        fitted_later = {**stale_day, "trainingCutoffEpoch": BASE, "metadata": {**stale_day["metadata"], "fittedAtEpoch": BASE + 2 * HOUR}}
        self.assert_unavailable(self.predict(BASE + HOUR, artifact=fitted_later), "model_not_fitted_at_issue")

    def test_one_offline_station_is_a_missing_input_not_an_unavailable_forecast(self):
        self.inputs.fill(BASE - 4 * HOUR, BASE + 4 * HOUR, stations=STATIONS[:4] + STATIONS[5:])
        live = self.inputs.live(BASE + 2 * HOUR)
        self.assertIsNotNone(live["features"], live["reason"])
        self.assertEqual(live["features"]["sp_count"], 5)
        self.assertTrue(math.isnan(live["features"]["sp_setapak_diff"]))
        self.assertTrue(math.isnan(live["features"]["sp_setapak_age"]))


def research_data():
    needed = ("snapshot_20261013.db", "neighbors_20261013.sqlite3", "models/dataset.parquet",
              "models/spatial/spatial_features_lag300.parquet", "models/final/pred_point_P1_no_weather.parquet",
              "models/final/pred_le70_E1_spatial_residual.parquet")
    for candidate in (os.environ.get("TTDI_RESEARCH_DATA"), ROOT.parent / "data"):
        if candidate and all((Path(candidate) / name).exists() for name in needed):
            return Path(candidate)
    return None


DATA = research_data()


@unittest.skipIf(DATA is None, "research snapshot (data/snapshot_20261013.db) not present")
class ResearchParityTests(unittest.TestCase):
    """Server feature builder and live path == research final model, exactly (read-only snapshot)."""

    @classmethod
    def setUpClass(cls):
        cls.db, cls.archive = DATA / "snapshot_20261013.db", DATA / "neighbors_20261013.sqlite3"
        base = [c for c in (*S.CACHE_COLUMNS, *S.TARGET_COLUMNS) if not c.startswith("sp_")]
        cls.research = pd.read_parquet(DATA / "models" / "dataset.parquet", columns=base).join(
            pd.read_parquet(DATA / "models" / "spatial" / "spatial_features_lag300.parquet", columns=list(S.SPATIAL_COLUMNS)))
        final = DATA / "models" / "final"
        cls.point = pd.read_parquet(final / "pred_point_P1_no_weather.parquet")
        cls.le70 = pd.read_parquet(final / "pred_le70_E1_spatial_residual.parquet")

    def rows(self, issue):
        with closing(sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            return observation_provenance.compatible_sensor_rows(conn, issue, issue - S.SENSOR_WINDOW_SECONDS, issue)

    def test_training_features_and_live_predictions_match_research_on_oct9_and_oct13(self):
        rng = np.random.default_rng(20261014)
        for day, cutoff in (("2026-10-09", 1791475200), ("2026-10-13", 1791820800)):
            with self.subTest(day=day):
                self.assertEqual(S.midnight_epoch(cutoff + 1), cutoff)
                origins = S.training_origins(cutoff)
                sample = np.sort(rng.choice(origins, size=10, replace=False))
                mine = S.compute_training_features(self.db, sample, archive_path=self.archive)
                theirs = self.research.loc[sample]
                for column in S.CACHE_COLUMNS:
                    a, b = mine[column].to_numpy(float), theirs[column].to_numpy(float)
                    self.assertTrue(((np.isnan(a) & np.isnan(b)) | (a == b)).all(), column)
                with closing(sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                    labels = S.training_targets(S.label_rows(conn, cutoff - 29 * 86400, cutoff), sample)
                for column in S.TARGET_COLUMNS:
                    a, b = labels[column].to_numpy(float), theirs[column].to_numpy(float)
                    self.assertTrue(((np.isnan(a) & np.isnan(b)) | (a == b)).all(), column)
                # Fit on the research rows (identical to the server's, see parity_check.json) and
                # predict through the live path from receipt-visible snapshot inputs.
                artifact = fit_artifact(self.research.loc[origins], cutoff)
                for issue in cutoff + 300 * np.array([96, 132, 150, 186, 222, 252]):
                    issue = int(issue)
                    result = S.predict(self.db, self.rows(issue), issue, archive_path=self.archive, artifact=artifact,
                                       allow_reconstruction=True)
                    if math.isnan(float(self.point.loc[issue, "pred_arrival"])):
                        self.assertFalse(result["available"], issue)     # research output was unavailable too
                        continue
                    self.assertTrue(result["available"], (issue, result.get("reason")))
                    for field, head in (("arrivalPoint", "pred_arrival"), ("trailMeanPoint", "pred_ride_mean"),
                                        ("trailMinimumPoint", "pred_ride_min"), ("trailMaximumPoint", "pred_ride_max")):
                        self.assertEqual(result[field], float(self.point.loc[issue, head]), (issue, field))
                    self.assertEqual(result["le70"]["chanceMeanAtOrBelowCutoff"], float(self.le70.loc[issue, "p_ride_le70"]))

    def test_recorded_end_to_end_parity_evidence_shows_no_difference(self):
        path = ROOT / "research" / "spatial_point_v25" / "parity_check.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(evidence["days"]), {"2026-10-09", "2026-10-13"})
        for day in evidence["days"].values():
            self.assertEqual(day["trainingFeatureParity"]["featureMismatches"], 0)
            for block in ("gridIssues", "issuedClocks"):
                for head in ("pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max", "p_ride_le70"):
                    self.assertEqual(day[block][head]["maxAbsDiff"], 0.0)
                    self.assertEqual(day[block][head]["nanMismatch"], 0)


ISSUE = BASE + 13 * HOUR + 1805   # 13:30:05 MYT


def spatial_forecast(available=True, chance=0.3125):
    base = S._base(ISSUE)
    base.update(freshReferencePm25=163.2, inputLineage={"freshReferenceEpoch": ISSUE - 30, "available": True})
    if not available:
        return S._unavailable(base, "neighbour_inputs_unavailable_or_stale")
    raw = {"arrivalPoint": 158.123456, "trailMeanPoint": 141.654321,
           "trailMinimumPoint": 120.5, "trailMaximumPoint": 160.25}
    le70 = S._le70_base(ISSUE)
    le70.update(available=True, reason="available", unavailableReasons=[], chanceMeanAtOrBelowCutoff=chance,
                referencePm=163.2, rideMeanAnchorUgM3=raw["trailMeanPoint"], featureValuesSha256="0" * 64,
                trainingCutoffEpoch=BASE)
    return {**base, **raw, "available": True, "rawOutputUgM3": dict(raw), "featureColumns": list(S.INPUT_COLUMNS),
            "featureValues": [0.0] * 81, "featureValuesSha256": "0" * 64, "le70": le70,
            "training": {"trainingCutoffEpoch": BASE}, "modelIdentity": {"identitySha256": "x"}}


class PublicationWiringTests(unittest.TestCase):
    """/api/analysis and /api/rlcd/v1 publish the v25 outputs literally."""

    def setUp(self):
        import BukitKiara_Dashboard as app
        from forecast_clock import ForecastClock
        self.app = app
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        stack = self.stack
        stack.enter_context(patch.object(app, "DB_PATH", Path(self.folder.name) / "test.db"))
        app.init_db()
        self.reading = {"epoch": ISSUE - 30, "pm02": 163.2, "pm10": 170.0, "atmp": 30.0, "rhum": 50.0, "heatindex": 39.0}
        with app.db() as conn:
            conn.execute("INSERT INTO readings(epoch,timestamp,pm02,pm10,atmp,rhum,heatindex) VALUES(?,?,?,?,?,?,?)",
                         (ISSUE - 30, "test", 163.2, 170.0, 30.0, 50.0, 39.0))
        clock = ForecastClock(ISSUE, ISSUE // 900 * 900).metadata()
        reference = {"available": True, "point": 163.2, "baselinePoint": 163.2, "pointRole": "persistence_anchor",
                     "freshnessAdjustment": {"applied": True}, "forecastClock": clock}
        air = {"available": True, "forecastClock": clock, "arrival": dict(reference), "trail": dict(reference)}
        stack.enter_context(patch.object(app.time, "time", return_value=ISSUE))
        for name in ("air_window_analysis", "refresh_near_term_persistence"):
            stack.enter_context(patch.object(app, name, return_value=copy.deepcopy(air)))
        stack.enter_context(patch.object(app, "cached_shadow_analogue_outlook", return_value={"available": False}))
        stack.enter_context(patch.object(app, "record_local_pm_forecast_issue"))
        stack.enter_context(patch.object(app, "rapid_clearance_event_outlook", return_value={}))
        stack.enter_context(patch.object(app, "shadow_dry_dispersion_watch", return_value={}))
        stack.enter_context(patch.object(app, "attach_experimental_rapid_change_risk",
                                         return_value={"experimentalRapidChangeRisk": {"available": False}}))
        for name in ("separate_observation_from_prediction", "attach_experimental_short_horizon_change",
                     "attach_first_crossing_event_forecast"):
            stack.enter_context(patch.object(app, name, side_effect=lambda value, *args: value))
        stack.enter_context(patch.object(app, "weather_outlook", return_value={"windows": {"morning": {}, "afternoon": {}}}))
        stack.enter_context(patch.object(app.regional_pm_inputs, "describe_context", return_value={}))
        stack.enter_context(patch.object(app, "regional_haze_window_outlook", return_value={}))
        stack.enter_context(patch.object(app.arrival_change_model, "predict_arrival",
                                         return_value={"available": False, "reason": "test"}))
        stack.enter_context(patch.object(app.weather_session_forecast, "predict_windows",
                                         side_effect=lambda db, rows, windows, issue, **kw: {"morning": {"available": False}}))
        stack.enter_context(patch.object(app.patchtst_session_forecast, "apply_experimental_afternoon",
                                         side_effect=lambda rows, windows, issue, estimates, **kw: {**estimates, "afternoon": {"available": False}}))
        self.predict = stack.enter_context(patch.object(app.spatial_point_model, "predict"))

    def compute(self, forecast):
        self.predict.return_value = forecast
        result = self.app._compute_analysis(28, include_trials=False)
        self.predict.assert_called_once()
        self.assertEqual(self.predict.call_args.args[2], ISSUE)
        return result

    def test_analysis_and_device_apis_publish_the_spatial_heads_and_e1_probability_literally(self):
        app = self.app
        result = self.compute(spatial_forecast())
        air = result["airWindow"]
        self.assertEqual(air["arrival"]["point"], 158.123456)
        self.assertEqual(air["arrival"]["modelVersion"], S.MODEL_VERSION)
        self.assertEqual(air["arrival"]["method"], S.METHOD)
        self.assertEqual(air["arrival"]["baselinePoint"], 163.2)
        self.assertEqual(air["trail"]["point"], 141.654321)
        self.assertEqual((air["trail"]["rideExtrema"]["low"], air["trail"]["rideExtrema"]["high"]), (120.5, 160.25))
        self.assertEqual(air["forecastClock"]["arrivalTargetEpoch"], ISSUE + 5400)
        cycling = air["cyclingWindowForecast"]
        self.assertTrue(cycling["available"])
        self.assertEqual(cycling["chanceMeanAtOrBelowCutoff"], 0.3125)
        self.assertEqual(cycling["modelVersion"], S.LE70_MODEL_VERSION)
        self.assertEqual((cycling["startEpoch"], cycling["endEpoch"], cycling["cutoffUgM3"], cycling["mode"]),
                         (ISSUE + 5400, ISSUE + 12600, 70.0, "initial90"))
        self.assertNotIn("le70", result["freshSensorForecast"])
        self.assertEqual(result["freshSensorForecast"]["modelVersion"], S.MODEL_VERSION)
        policy = result["forecastPolicy"]
        self.assertEqual(policy["id"], "fixed_direct_models_with_arrival_change_v24")
        self.assertEqual(policy["cyclingWindowProbabilityModelVersion"], S.LE70_MODEL_VERSION)
        self.assertEqual(policy["nearTermNumericalModelVersion"], S.MODEL_VERSION)
        self.assertTrue(policy["neighbourPmInputsUsedForParticleForecast"])

        device = app.rlcd_api.build_payload(self.reading, result, {}, ISSUE + 20, dashboard_build=app.DASHBOARD_BUILD)
        near, ride = device["forecast"]["near90"], device["forecast"]["ride90_210"]
        self.assertEqual((near["available"], near["pm25_ugm3"], near["model"]), (True, 158.123456, S.MODEL_VERSION))
        self.assertEqual((ride["available"], ride["pm25_ugm3"]), (True, 141.654321))
        self.assertEqual((ride["minimum_maximum"]["minimum_ugm3"], ride["minimum_maximum"]["maximum_ugm3"]), (120.5, 160.25))
        self.assertEqual((ride["mean_le70"]["available"], ride["mean_le70"]["probability"], ride["mean_le70"]["model"]),
                         (True, 0.3125, S.LE70_MODEL_VERSION))
        self.assertEqual(device["dashboard_build"], "2026-10-14-coach-api-removed-v25.1.0-public.1")
        self.assertLessEqual(len(json.dumps(device, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()), 16384)

        import jsonschema
        jsonschema.Draft202012Validator(app.cycling_window_json_schema()).validate(
            json.loads(json.dumps(cycling, allow_nan=False)))
        jsonschema.Draft202012Validator(app.cycling_window_json_schema()).validate(
            json.loads(json.dumps(spatial_forecast(available=False)["le70"])))

    def test_unavailable_spatial_forecast_is_null_in_every_api(self):
        result = self.compute(spatial_forecast(available=False))
        air = result["airWindow"]
        self.assertFalse(air["arrival"]["available"])
        self.assertIsNone(air["arrival"]["point"])
        self.assertIsNone(air["trail"]["point"])
        self.assertIsNone(air["trail"]["rideExtrema"]["low"])
        self.assertFalse(air["cyclingWindowForecast"]["available"])
        self.assertEqual(air["cyclingWindowForecast"]["reason"], "neighbour_inputs_unavailable_or_stale")
        self.assertFalse(result["forecastPolicy"]["neighbourPmInputsUsedForParticleForecast"])
        device = self.app.rlcd_api.build_payload(self.reading, result, {}, ISSUE + 20)
        self.assertFalse(device["forecast"]["near90"]["available"])
        self.assertIsNone(device["forecast"]["near90"]["pm25_ugm3"])
        self.assertFalse(device["forecast"]["ride90_210"]["mean_le70"]["available"])
        self.assertIsNone(device["forecast"]["ride90_210"]["minimum_maximum"]["minimum_ugm3"])

    def test_publication_records_validation_rows_ledger_and_the_e1_issued_jsonl(self):
        app = self.app
        result = self.compute(spatial_forecast())
        result["computation"] = {}
        output = Path(self.folder.name) / "le70-issued.jsonl"
        original = S.record_issued_le70
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        with ExitStack() as publication:
            publication.enter_context(patch.object(app, "_compute_analysis", return_value=result))
            for method, value in (("schedule", None), ("take_job", 28), ("publish_primary", None), ("finish", None)):
                publication.enter_context(patch.object(app.analysis_delivery, method, return_value=value))
            for module in (app.window_pm_predictor, app.weather_session_forecast, app.afternoon_xgboost_forecast):
                publication.enter_context(patch.object(module, "record_issue"))
            publication.enter_context(patch.object(app.fresh_event_model, "record_issued_prediction"))
            publication.enter_context(patch.object(app.arrival_change_model, "record_issued_prediction"))
            cycling_logistic = publication.enter_context(patch.object(app.cycling_window_live, "record_issued_prediction"))
            writer = publication.enter_context(patch.object(
                app.spatial_point_model, "record_issued_le70",
                side_effect=lambda value, **kw: original(value, output_path=output, **kw)))
            app.analysis_worker(stop)
        writer.assert_called_once()
        cycling_logistic.assert_not_called()
        record = json.loads(output.read_text(encoding="utf-8").strip())
        self.assertTrue(record["isOriginallyIssued"])
        self.assertEqual(record["chanceMeanAtOrBelowCutoff"], 0.3125)
        self.assertEqual(record["modelVersion"], S.LE70_MODEL_VERSION)
        self.assertEqual(record["predictionRole"], "genuinely_issued_live_spatial_le70_prediction")
        self.assertFalse(original(record, output_path=output, recorded_epoch=ISSUE + 30))
        with self.assertRaises(ValueError):
            original(result["airWindow"]["cyclingWindowForecast"], output_path=output, recorded_epoch=ISSUE + 121)
        with app.db() as conn:
            rows = conn.execute("SELECT target_family,candidate_version,actual_model FROM prospective_validation_issued "
                                "WHERE target_family IN ('point90','mean90_210') ORDER BY target_family").fetchall()
            payloads = [app.prospective_validation.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM prospective_validation_issued WHERE target_family='mean90_210'")]
            ledger = app.prospective_validation.loads(conn.execute("SELECT payload FROM dashboard_forecast_issues").fetchone()[0])
        self.assertEqual([tuple(r) for r in rows],
                         [("mean90_210", "near_spatial_hgb_mean_model_direct_v25", S.MODEL_VERSION),
                          ("point90", "near_spatial_hgb_arrival_model_direct_v25", S.MODEL_VERSION)])
        self.assertEqual(payloads[0]["probability"], 0.3125)
        self.assertEqual(payloads[0]["probabilityModelVersion"], S.LE70_MODEL_VERSION)
        self.assertEqual(ledger["freshSensorForecast"]["modelVersion"], S.MODEL_VERSION)
        self.assertEqual(ledger["airWindow"]["cyclingWindowForecast"]["chanceMeanAtOrBelowCutoff"], 0.3125)
        self.assertEqual(ledger["dashboardBuild"], "2026-10-14-coach-api-removed-v25.1.0-public.1")

    def test_service_refits_the_spatial_model_and_no_longer_the_ridge(self):
        source = (ROOT / "BukitKiara_Dashboard.py").read_text(encoding="utf-8")
        self.assertIn("target=spatial_point_model_worker", source)
        self.assertNotIn("fresh_numeric_model_worker", source)
        self.assertNotIn("import fresh_numeric_model_v4", source)
        self.assertFalse(hasattr(self.app, "fresh_numeric_model"))
        self.assertEqual(self.app.DASHBOARD_BUILD, "2026-10-14-coach-api-removed-v25.1.0-public.1")


class CoachApiRemovalTests(unittest.TestCase):
    """The retired Coach API routes are gone; the cycling-plan recheck stays described."""

    def setUp(self):
        import BukitKiara_Dashboard as app
        self.app = app

    def get(self, path):
        handler = object.__new__(self.app.Handler)
        handler.path = path
        sent = []
        handler.send_json = lambda value, *args: sent.append(("json", value, args))
        handler.send_data = lambda status, content_type, payload, **kw: sent.append(
            ("data", status, content_type, json.loads(payload)))
        handler.do_GET()
        return sent

    def test_coach_routes_are_not_found(self):
        for path in ("/api/v1/mtb/environment-evidence", "/api/v1/mtb/environment-evidence/schema",
                     "/docs/coach-api.md"):
            self.assertEqual(self.get(path), [("json", {"error": "Not found"}, (404,))])
        self.assertFalse((ROOT / "COACH_API.md").exists())

    def test_openapi_describes_only_the_cycling_plan_recheck(self):
        [(kind, status, content_type, document)] = self.get("/api/openapi.json")
        self.assertEqual((kind, status), ("data", 200))
        self.assertTrue(content_type.startswith("application/vnd.oai.openapi+json"))
        self.assertEqual(document["openapi"], "3.1.0")
        self.assertEqual(list(document["paths"]), ["/api/cycling-plan"])
        response = document["paths"]["/api/cycling-plan"]["get"]["responses"]["200"]
        self.assertEqual(response["content"]["application/json"]["schema"],
                         json.loads(json.dumps(self.app.cycling_window_json_schema())))


if __name__ == "__main__":
    unittest.main()
