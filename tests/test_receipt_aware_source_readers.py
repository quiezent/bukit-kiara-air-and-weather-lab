"""Reader integration tests: late revisions cannot rewrite earlier inputs."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from contextlib import closing
import json
import gzip
from pathlib import Path
import sqlite3
import tempfile
import unittest

import observation_provenance as provenance
import rain_weather_features as rain
import window_pm_predictor as window
import afternoon_direction_features as causal
from test_weather_session_features import epoch, payload, SOURCE
from test_observation_provenance import reading


class ReceiptAwareReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Path(self.directory.name) / "sources.sqlite"
        self.fetched = epoch("2026-09-30 11:00")
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("CREATE TABLE weather_forecast_runs(fetched_epoch INTEGER PRIMARY KEY,source TEXT,payload TEXT)")
            conn.execute("CREATE TABLE air_quality_forecast_runs(fetched_epoch INTEGER PRIMARY KEY,source TEXT,model_version TEXT,payload BLOB)")
            conn.execute("CREATE TABLE readings(epoch INTEGER PRIMARY KEY,timestamp TEXT,pm02 REAL)")
            provenance.init_schema(conn)
        window._RUN_CACHE.clear()

    def insert(self, fetched, text, received, persisted):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO weather_forecast_runs VALUES(?,?,?)", (fetched, SOURCE, text))
            provenance.record_source_receipt(conn, "weather_forecast_runs", text, fetched, received, clock=lambda: persisted)
            conn.commit()
            provenance.confirm_receipts_visible(conn, clock=lambda: persisted+0.25)
            conn.commit()

    def test_in_place_overwrite_keeps_original_revision_at_previous_issue(self):
        original = payload(self.fetched, first_hour_rain=1, second_hour_rain=1)
        corrected = payload(self.fetched, first_hour_rain=9, second_hour_rain=9)
        self.insert(self.fetched, original, self.fetched+5, self.fetched+10)
        self.insert(self.fetched, corrected, self.fetched+2400, self.fetched+2700)
        early = self.fetched+1800
        final = self.fetched+3000
        archive = rain.load_runs(self.db, final)
        self.assertEqual(rain.latest_run(archive, early).payload_json, original)
        self.assertEqual(rain.latest_run(archive, final).payload_json, corrected)
        self.assertTrue(rain.latest_run(archive, early).receipt_provenance['receiptKnown'])
        raw, _ = window._raw_runs(self.db, "weather_forecast_runs", early)
        self.assertEqual([text for _, text in raw], [original])
        self.assertTrue(raw.receipt_provenance[0]['receiptKnown'])
        runs = window._read_runs(self.db, "weather_forecast_runs", final)
        first_arrays, _ = window._source_at_issue(runs, early)
        last_arrays, _ = window._source_at_issue(runs, final)
        target = epoch("2026-09-30 15:00")
        self.assertEqual(window._sample(first_arrays, [target], "precipitation")[0], 1)
        self.assertEqual(window._sample(last_arrays, [target], "precipitation")[0], 9)

    def test_late_only_tracked_epoch_is_not_replaced_with_compatibility_bytes(self):
        fetched = self.fetched+1200
        text = payload(fetched)
        self.insert(fetched, text, self.fetched+2400, self.fetched+2700)
        early = self.fetched+1800
        raw, _ = window._raw_runs(self.db, "weather_forecast_runs", early)
        self.assertEqual(raw, [])
        archive = rain.load_runs(self.db, self.fetched+3000)
        self.assertIsNone(rain.latest_run(archive, early))
        self.assertIsNotNone(rain.latest_run(archive, self.fetched+3000))

    def test_legacy_training_history_is_explicitly_unknown_and_strict_option_excludes_it(self):
        text = payload(self.fetched)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO weather_forecast_runs VALUES(?,?,?)", (self.fetched, SOURCE, text))
        raw, _ = window._raw_runs(self.db, "weather_forecast_runs", self.fetched+1800)
        self.assertEqual(len(raw), 1)
        self.assertFalse(raw.receipt_provenance[0]['receiptKnown'])
        self.assertIsNone(raw.receipt_provenance[0]['availableEpoch'])
        strict, _ = window._raw_runs(self.db, "weather_forecast_runs", self.fetched+1800, strict_receipts=True)
        self.assertEqual(strict, [])
        self.assertEqual(len(rain.load_runs(self.db, self.fetched+1800)), 1)
        self.assertEqual(len(rain.load_runs(self.db, self.fetched+1800, strict_receipts=True)), 0)

    def test_compatible_sensor_reconstructs_old_values_and_excludes_late_only_rows(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            old = reading(pm=1.5)
            new = reading(pm=9.5)
            conn.execute("INSERT INTO readings VALUES(?,?,?)", (old['epoch'], old['timestamp'], new['pm02']))
            provenance.record_sensor_receipt(conn, old, 1001, clock=lambda: 1100)
            conn.commit()
            provenance.confirm_receipts_visible(conn, clock=lambda: 1100.25)
            conn.commit()
            provenance.record_sensor_receipt(conn, new, 1200, clock=lambda: 1300)
            conn.commit()
            provenance.confirm_receipts_visible(conn, clock=lambda: 1300.25)
            conn.commit()
            legacy = reading(epoch=900, pm=2.5)
            conn.execute("INSERT INTO readings VALUES(?,?,?)", (legacy['epoch'], legacy['timestamp'], legacy['pm02']))
            late = reading(epoch=1100, pm=3.5)
            conn.execute("INSERT INTO readings VALUES(?,?,?)", (late['epoch'], late['timestamp'], late['pm02']))
            provenance.record_sensor_receipt(conn, late, 1150, clock=lambda: 1400)
            conn.commit()
            provenance.confirm_receipts_visible(conn, clock=lambda: 1400.25)
            conn.commit()
            rows = provenance.compatible_sensor_rows(conn, 1250, 800)
            self.assertEqual([(r['epoch'], r['pm02']) for r in rows], [(900, 2.5), (1000, 1.5)])
            self.assertFalse(rows[0]['_provenance']['receiptKnown'])
            self.assertTrue(rows[1]['_provenance']['receiptKnown'])
            strict = provenance.compatible_sensor_rows(conn, 1250, 800, strict=True)
            self.assertEqual([(r['epoch'], r['pm02']) for r in strict], [(1000, 1.5)])

    def test_legacy_compressed_weather_payload_is_decoded_without_changing_bytes(self):
        text = payload(self.fetched)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO weather_forecast_runs VALUES(?,?,?)", (self.fetched, SOURCE, gzip.compress(text.encode('utf-8'))))
        raw, _ = window._raw_runs(self.db, "weather_forecast_runs", self.fetched+1800)
        self.assertEqual(raw[0][1], text)
        self.assertEqual(rain.latest_run(rain.load_runs(self.db, self.fetched+1800), self.fetched+1800).payload_json, text)

    def test_cams_historical_regional_features_use_receipt_available_revision(self):
        def cams(pm):
            return json.dumps({'source':'Open-Meteo / CAMS Global', 'modelVersion':'cams_anchor_v2',
                'fetchedEpoch':self.fetched,'hourlyUnits':{'pm2_5':'µg/m³'},
                'hourly':[{'epoch':self.fetched+i*3600,'pm2_5':pm} for i in range(5)]})
        with closing(sqlite3.connect(self.db)) as conn, conn:
            for text, offset in ((cams(10.5),10),(cams(80.5),2700)):
                conn.execute("INSERT OR REPLACE INTO air_quality_forecast_runs VALUES(?,?,?,?)",
                    (self.fetched,'Open-Meteo / CAMS Global','cams_anchor_v2',text))
                provenance.record_source_receipt(conn,'air_quality_forecast_runs',text,
                    self.fetched,self.fetched+offset-5,clock=lambda:self.fetched+offset)
                conn.commit()
                provenance.confirm_receipts_visible(conn, clock=lambda: self.fetched+offset+0.25)
                conn.commit()
        runs = window._read_runs(self.db,'air_quality_forecast_runs',self.fetched+3000)
        early = self.fetched+1800
        current,target,fetched = causal.regional(runs,early,early+300,early+1500)
        self.assertEqual((current,target,fetched),(10.5,10.5,self.fetched))
        late = self.fetched+3000
        current,target,fetched = causal.regional(runs,late,late+300,late+1500)
        self.assertEqual((current,target,fetched),(80.5,80.5,self.fetched))

    def test_unconfirmed_commit_cannot_certify_pre_commit_source_availability(self):
        text = payload(self.fetched)
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute("INSERT INTO weather_forecast_runs VALUES(?,?,?)", (self.fetched, SOURCE, text))
            provenance.record_source_receipt(conn, "weather_forecast_runs", text,
                self.fetched,self.fetched+5,clock=lambda:self.fetched+10)
            conn.commit()  # The first transaction may have been held for minutes.
            raw, _ = window._raw_runs(self.db, "weather_forecast_runs", self.fetched+2000)
            self.assertEqual(raw, [])
            provenance.confirm_receipts_visible(conn, clock=lambda:self.fetched+2200)
            conn.commit()
        early, _ = window._raw_runs(self.db, "weather_forecast_runs", self.fetched+2000)
        late, _ = window._raw_runs(self.db, "weather_forecast_runs", self.fetched+2300)
        self.assertEqual(early, [])
        self.assertEqual(len(late), 1)
        archive = rain.load_runs(self.db, self.fetched+2300)
        self.assertIsNone(rain.latest_run(archive, self.fetched+2000))
        self.assertEqual(rain.latest_run(archive, self.fetched+2300).receipt_provenance['confirmedEpoch'], self.fetched+2200)


if __name__ == "__main__":
    unittest.main()
