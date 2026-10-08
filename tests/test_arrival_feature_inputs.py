"""Meaningful issue-time, receipt, revision and native sensor feature checks."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import arrival_feature_inputs as inputs
import fresh_sensor_features as sensor
import observation_provenance as receipts


ISSUE=100000.0


def sensor_rows(issue=ISSUE):
    return [{"epoch":issue-age,"pm02":150-age/600,"atmp":30,"rhum":60}
            for age in range(60,10800,180)]


def weather_row(fetched=ISSUE-600,available=ISSUE-500,temperature=25,*,confirmed=None):
    lower=int((ISSUE-7200)//3600*3600)
    points=[]
    for stamp in range(lower,lower+8*3600,3600):
        points.append({"epoch":stamp, "temperature_2m":temperature,"relative_humidity_2m":70,
                       "precipitation_probability":80,"precipitation":2,"wind_speed_10m":5,
                       "wind_direction_10m":350,"wind_gusts_10m":10})
    return {"fetched_epoch":fetched,"payload":{"fetchedEpoch":fetched,"hourly":points},
            "_provenance":{"receiptId":1,"receiptKnown":True,"receivedEpoch":available-2,
                           "persistedEpoch":available-1,"availableEpoch":available-1,
                           "confirmedEpoch":available if confirmed is None else confirmed}}


class ArrivalFeatureTests(unittest.TestCase):
    def test_shared_sensor_reference_and_columns_unchanged(self):
        rows=sensor_rows()
        actual=inputs.issue_features(inputs.prepare_inputs(rows),ISSUE)
        expected=sensor.issue_features(rows,ISSUE)
        self.assertEqual(tuple(actual),inputs.FEATURE_COLUMNS)
        self.assertEqual(tuple(inputs.FEATURE_COLUMNS[:208]),sensor.FEATURE_COLUMNS)
        np.testing.assert_equal([actual[name] for name in sensor.FEATURE_COLUMNS],
                                [expected[name] for name in sensor.FEATURE_COLUMNS])
        self.assertEqual(len(inputs.FEATURE_COLUMNS),285)

    def test_future_valid_forecast_is_allowed_only_after_confirmed_receipt(self):
        prepared=inputs.prepare_inputs(sensor_rows(),[weather_row()])
        result=inputs.issue_features(prepared,ISSUE)
        self.assertEqual(result["issued_weather_temperature_2m_90min"],25)
        metadata=inputs.feature_metadata(prepared,ISSUE)
        self.assertTrue(metadata["weatherFeaturesAvailable"])
        self.assertLess(metadata["weatherAvailableEpoch"],ISSUE)
        self.assertTrue(metadata["weatherForecastValidTimesMayFollowIssue"])

    def test_future_confirmation_and_persistence_are_excluded(self):
        for field in ("confirmedEpoch","persistedEpoch","availableEpoch","receivedEpoch"):
            row=weather_row()
            row["_provenance"][field]=ISSUE+1
            result=inputs.issue_features(inputs.prepare_inputs(sensor_rows(),[row]),ISSUE)
            self.assertTrue(math.isnan(result["issued_weather_temperature_2m_90min"]),field)

    def test_missing_confirmation_is_not_a_fetch_clock_fallback(self):
        row=weather_row()
        row["_provenance"].pop("confirmedEpoch")
        prepared=inputs.prepare_inputs(sensor_rows(),[row])
        self.assertFalse(inputs.feature_metadata(prepared,ISSUE)["weatherFeaturesAvailable"])
        self.assertEqual(prepared.archive_metadata["weatherUnknownOrInvalidRowsExcluded"],1)

    def test_explicit_null_or_contradictory_confirmation_is_excluded(self):
        for value in (None,ISSUE-900):
            row=weather_row()
            row["_provenance"]["confirmedEpoch"]=value
            result=inputs.issue_features(inputs.prepare_inputs(sensor_rows(),[row]),ISSUE)
            self.assertTrue(math.isnan(result["issued_weather_temperature_2m_90min"]))

    def test_latest_visible_revision_does_not_rewrite_older_issue(self):
        first=weather_row(temperature=21)
        second=weather_row(available=ISSUE+30,temperature=40)
        second["_provenance"]["receiptId"]=2
        prepared=inputs.prepare_inputs(sensor_rows(),[first,second])
        self.assertEqual(inputs.issue_features(prepared,ISSUE)["issued_weather_temperature_2m_90min"],21)
        self.assertEqual(inputs.issue_features(prepared,ISSUE+60)["issued_weather_temperature_2m_90min"],40)

    def test_newer_unavailable_fetch_does_not_displace_previous_vintage(self):
        first=weather_row(temperature=21)
        second=weather_row(fetched=ISSUE-100,available=ISSUE+100,temperature=40)
        prepared=inputs.prepare_inputs(sensor_rows(),[first,second])
        self.assertEqual(inputs.issue_features(prepared,ISSUE)["issued_weather_temperature_2m_90min"],21)

    def test_unknown_legacy_weather_never_enters_prior_features(self):
        legacy=weather_row()
        legacy.pop("_provenance")
        prepared=inputs.prepare_inputs(sensor_rows(),[legacy])
        actual=inputs.issue_features(prepared,ISSUE)
        self.assertTrue(all(math.isnan(actual[name]) for name in inputs.WEATHER_FEATURE_COLUMNS))

    def test_forecast_gap_and_extrapolation_stay_missing(self):
        row=weather_row()
        points=row["payload"]["hourly"]
        row["payload"]["hourly"]=[points[0],points[-1]]
        result=inputs.issue_features(inputs.prepare_inputs(sensor_rows(),[row]),ISSUE)
        self.assertTrue(math.isnan(result["issued_weather_temperature_2m_90min"]))
        prepared=inputs.prepare_inputs(sensor_rows(),[weather_row()])
        run=prepared.weather[0]
        self.assertTrue(np.isnan(inputs._weather_values(run,run.epochs[-1]+1)).all())

    def test_missing_field_is_not_imputed(self):
        result=inputs.issue_features(inputs.prepare_inputs(sensor_rows(),[weather_row()]),ISSUE)
        self.assertTrue(math.isnan(result["issued_weather_cape_90min"]))
        self.assertFalse(inputs.feature_metadata(inputs.prepare_inputs(sensor_rows()),ISSUE)["imputationApplied"])

    def test_direction_wrap_and_precipitation_interval_contract(self):
        row=weather_row()
        for index,point in enumerate(row["payload"]["hourly"]):
            point["wind_direction_10m"]=350 if index%2==0 else 10
            point["precipitation"]=index
        prepared=inputs.prepare_inputs(sensor_rows(),[row])
        run=prepared.weather[0]
        middle=(run.epochs[0]+run.epochs[1])/2
        values=inputs._weather_values(run,middle)
        self.assertAlmostEqual(values[inputs.WEATHER_POINT_FIELDS.index("wind_direction_10m_sin")],0,places=12)
        self.assertGreater(values[inputs.WEATHER_POINT_FIELDS.index("wind_direction_10m_cos")],.9)
        self.assertEqual(values[inputs.WEATHER_POINT_FIELDS.index("precipitation")],1)

    def test_weather_revision_uses_same_absolute_arrival_and_older_visible_vintage(self):
        old=weather_row(fetched=ISSUE-3600,available=ISSUE-3500,temperature=20)
        new=weather_row(temperature=30)
        prepared=inputs.prepare_inputs(sensor_rows(),[old,new])
        result=inputs.issue_features(prepared,ISSUE)
        self.assertEqual(result["issued_weather_arrival_revision_temperature_2m_30min"],10)
        self.assertTrue(math.isnan(result["issued_weather_arrival_revision_temperature_2m_60min"]))

    def test_stale_weather_is_missing(self):
        row=weather_row(fetched=ISSUE-7201,available=ISSUE-7200)
        result=inputs.issue_features(inputs.prepare_inputs(sensor_rows(),[row]),ISSUE)
        self.assertTrue(all(math.isnan(result[name]) for name in inputs.WEATHER_FEATURE_COLUMNS))

    def test_neighbor_request_clocks_are_diagnostic_and_never_model_features(self):
        neighbor={"location_id":99667152,"bucket_start_epoch":ISSUE-300,"bucket_seconds":0,
                  "pm25":90,"received_epoch":ISSUE-240,"recorded_epoch":ISSUE-239,
                  "kind":"current","valid":1}
        baseline=inputs.issue_features(inputs.prepare_inputs(sensor_rows()),ISSUE)
        prepared=inputs.prepare_inputs(sensor_rows(),[],[neighbor])
        np.testing.assert_equal(list(baseline.values()),list(inputs.issue_features(prepared,ISSUE).values()))
        diagnostics=inputs.feature_metadata(prepared,ISSUE)["neighborDiagnostics"]
        self.assertEqual(diagnostics["requestClockEligibleStationCount"],1)
        self.assertFalse(diagnostics["usedAsLearnerFeatures"])
        self.assertFalse(diagnostics["observationCommitConfirmationAvailable"])

    def test_neighbors_future_receipts_and_target_station_are_excluded_from_diagnostics(self):
        rows=[{"location_id":location,"bucket_start_epoch":ISSUE-300,"bucket_seconds":0,"pm25":90,
               "received_epoch":received,"recorded_epoch":received,"valid":1}
              for location,received in ((99667152,ISSUE+1),(956,ISSUE-1))]
        metadata=inputs.feature_metadata(inputs.prepare_inputs(sensor_rows(),[],rows),ISSUE)
        self.assertEqual(metadata["neighborDiagnostics"]["requestClockEligibleStationCount"],0)

    def test_batch_and_single_query_have_identical_features_and_missing_values(self):
        prepared=inputs.prepare_inputs(sensor_rows(),[weather_row()])
        issues=[ISSUE,ISSUE+60]
        frame=inputs.issue_feature_frame(prepared,issues)
        self.assertEqual(tuple(frame.columns),inputs.FEATURE_COLUMNS)
        for index,issue in enumerate(issues):
            np.testing.assert_equal(frame.iloc[index].to_numpy(),list(inputs.issue_features(prepared,issue).values()))
            self.assertEqual(frame.attrs["featureProvenance"][index],inputs.feature_metadata(prepared,issue))
        with self.assertRaises(ValueError):
            inputs.issue_feature_frame(prepared,[ISSUE,ISSUE])

    def test_original_input_and_accessor_mutation_do_not_change_prepared_features(self):
        original=sensor_rows()
        prepared=inputs.prepare_inputs(original)
        before=inputs.issue_features(prepared,ISSUE)
        original[0]["pm02"]=999
        detached=inputs.source_sensor_rows(prepared)
        detached[0]["pm02"]=998
        np.testing.assert_equal(list(before.values()),list(inputs.issue_features(prepared,ISSUE).values()))

    def test_cached_sensor_augmentation_preserves_values_and_rejects_wrong_snapshot(self):
        rows=sensor_rows()
        cached=sensor.issue_feature_frame(rows,[ISSUE])
        prepared=inputs.prepare_inputs(rows,[weather_row()])
        augmented=inputs.augment_sensor_frame(prepared,cached)
        np.testing.assert_equal(augmented.iloc[0].to_numpy(),list(inputs.issue_features(prepared,ISSUE).values()))
        rows[0]["pm02"]=10
        with self.assertRaisesRegex(ValueError,"different snapshot"):
            inputs.augment_sensor_frame(inputs.prepare_inputs(rows),cached)

    def test_receipt_hash_mismatch_cannot_certify_modified_weather(self):
        row=weather_row()
        row["_provenance"]["payloadSha256"]="0"*64
        prepared=inputs.prepare_inputs(sensor_rows(),[row])
        self.assertFalse(inputs.feature_metadata(prepared,ISSUE)["weatherFeaturesAvailable"])

    def test_readonly_loader_retains_legacy_sensor_but_not_legacy_weather(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"inputs.db"
            connection=sqlite3.connect(path)
            connection.executescript("CREATE TABLE readings(epoch INTEGER,pm02 REAL,atmp REAL,rhum REAL);"
                                     "CREATE TABLE weather_forecast_runs(fetched_epoch INTEGER,payload TEXT,source TEXT);")
            connection.executemany("INSERT INTO readings VALUES(?,?,?,?)",[(row["epoch"],row["pm02"],row["atmp"],row["rhum"]) for row in sensor_rows()])
            connection.execute("INSERT INTO weather_forecast_runs VALUES(?,?,?)",(ISSUE-600,'{}','test'))
            connection.commit()
            receipts.init_schema(connection)
            connection.commit()
            connection.close()
            before=path.read_bytes()
            prepared=inputs.load_inputs(path,ISSUE-1800,ISSUE)
            self.assertTrue(inputs.feature_metadata(prepared,ISSUE)["available"])
            self.assertFalse(inputs.feature_metadata(prepared,ISSUE)["weatherFeaturesAvailable"])
            self.assertEqual(prepared.archive_metadata["legacyWeatherRunsInRange"],1)
            self.assertEqual(path.read_bytes(),before)


if __name__=="__main__":
    unittest.main()
