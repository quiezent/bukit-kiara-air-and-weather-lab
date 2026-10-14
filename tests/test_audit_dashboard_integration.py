"""Regressions through the real dashboard seams; isolated SQLite, no fitting/network."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime
from contextlib import closing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import BukitKiara_Dashboard as app
from test_weather_contracts import fixture, ANCHOR, HOUR


class DashboardIntegrationTests(unittest.TestCase):
    def issue(self, clock):
        return int(datetime.fromisoformat('2026-10-08T'+clock+'+08:00').timestamp())

    def test_session_target_stays_canonical_across_roll_boundary(self):
        config = next(c for c in app.RIDE_WINDOWS if c['key']=='afternoon')
        for clock, day in [('12:00:00',8),('12:15:01',8),('12:30:00',8),('12:30:01',9),('15:00:00',9)]:
            target = app.next_feasible_ride_period(self.issue(clock),config)
            self.assertEqual((target[3].day,target[3].hour,target[3].minute),(day,14,0))
            self.assertEqual((target[4]-target[3]).total_seconds(),7200)
            self.assertGreaterEqual(target[3].timestamp()-self.issue(clock),5400)

    def test_morning_roll_keeps_nine_to_eleven(self):
        config = next(c for c in app.RIDE_WINDOWS if c['key']=='morning')
        target = app.next_feasible_ride_period(self.issue('07:30:01'),config)
        self.assertEqual((target[3].day,target[3].hour,target[4].hour),(9,9,11))

    def test_legacy_fallback_never_bypasses_qualification(self):
        issue=self.issue('12:00:00'); start=self.issue('14:00:00')
        rows=[{'epoch':issue-30,'pm02':100}, {'epoch':issue-180,'pm02':104}]
        value={'available':True,'modelVersion':'old_uncertified_model','mean':60,'sensorAnchor':150,
               'forecastedAtEpoch':issue,'startEpoch':start,'endEpoch':start+7200}
        result=app.qualify_session_outputs(rows,{'afternoon':{'startEpoch':start,'endEpoch':start+7200}},issue,{'afternoon':value})['afternoon']
        self.assertEqual(result['mean'],102)
        self.assertEqual(result['candidateForecast']['mean'],60)
        self.assertFalse(result['modelSelection']['skillGatePassed'])
        self.assertTrue(result['modelSelection']['gatesAppliedToSelection'])

    def test_previous_issue_candidate_cannot_be_relabelled(self):
        issue=self.issue('12:00:00');start=self.issue('14:00:00')
        value={'available':True,'modelVersion':'old_candidate','mean':60,'sensorAnchor':100,
               'forecastedAtEpoch':issue-60,'startEpoch':start,'endEpoch':start+7200}
        result=app.qualify_session_outputs([{'epoch':issue-30,'pm02':100}],{'afternoon':{'startEpoch':start,'endEpoch':start+7200}},issue,{'afternoon':value})['afternoon']
        self.assertEqual(result['mean'],100)
        self.assertFalse(result['candidateForecast']['available'])

    def test_partial_weather_is_unavailable_through_dashboard(self):
        payload=fixture();payload['hourly']=payload['hourly'][:2]
        result=app.summarize_weather_range(payload,ANCHOR,ANCHOR+2*HOUR,{})
        self.assertFalse(result['available'])
        self.assertIsNone(result['precipitationMm'])
        self.assertFalse(result['weatherCoverage']['complete'])

    def test_complete_weather_true_zero_stays_zero(self):
        payload=fixture()
        for point in payload['hourly']:point['precipitation']=0
        result=app.summarize_weather_range(payload,ANCHOR,ANCHOR+2*HOUR,{})
        self.assertTrue(result['available'])
        self.assertEqual(result['precipitationMm'],0)

    def test_malformed_weather_becomes_unknown_through_actual_dashboard(self):
        from copy import deepcopy
        cases = [('precipitation_probability', 2, float('inf'), 'precipitationProbabilityMax'),
                 ('relative_humidity_2m', 0, float('nan'), 'relativeHumidityMean'),
                 ('precipitation', 2, True, 'precipitationMm'),
                 ('wind_speed_10m', 0, 'bad-value', 'windSpeed10mMean'),
                 ('temperature_2m', 0, 10**1000, 'temperatureMax')]
        for field, index, invalid, aggregate in cases:
            with self.subTest(field=field):
                payload=fixture();payload['hourly'][index][field]=invalid
                before=deepcopy(payload)
                result=app.summarize_weather_range(payload,ANCHOR,ANCHOR+2*HOUR,{})
                self.assertFalse(result['available'])
                self.assertIsNone(result[aggregate])
                self.assertFalse(result['weatherCoverage']['fields'][field]['complete'])
                self.assertEqual(payload['hourly'][index]['epoch'],before['hourly'][index]['epoch'])
                self.assertIs(payload['hourly'][index][field],invalid)

    def test_invalid_optional_direction_preserves_complete_core_weather(self):
        payload=fixture();payload['hourly'][0]['wind_direction_10m']=float('inf')
        result=app.summarize_weather_range(payload,ANCHOR,ANCHOR+2*HOUR,{})
        self.assertTrue(result['available'])
        self.assertIsNone(result['windDirection10m'])
        self.assertIsNone(result['windDirection10mCompass'])
        self.assertFalse(result['weatherCoverage']['fields']['wind_direction_10m']['complete'])

    def test_invalid_hourly_clock_is_omitted_and_support_remains_missing(self):
        payload=fixture();payload['hourly'][0]['epoch']='malformed-clock'
        result=app.summarize_weather_range(payload,ANCHOR,ANCHOR+2*HOUR,{})
        self.assertFalse(result['available'])
        self.assertIsNone(result['temperatureMax'])
        self.assertEqual(result['weatherCoverage']['fields']['temperature_2m']['missingOrInvalidEpochs'],[ANCHOR])

    def test_collector_keeps_same_timestamp_revisions_and_receipts(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'DB_PATH',Path(folder)/'test.db'):
            app.init_db();epoch=int(time.time())-15
            data={'timestamp':datetime.fromtimestamp(epoch,app.timezone.utc).isoformat(),'pm02':100}
            app.save_reading(data,received_epoch=time.time())
            data['pm02']=80;app.save_reading(data,received_epoch=time.time())
            with app.db() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM sensor_observation_receipts').fetchone()[0],2)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM sensor_observation_revisions').fetchone()[0],2)
                self.assertEqual(conn.execute('SELECT pm02 FROM readings').fetchone()[0],80)

    def test_weather_api_clock_remains_current_when_sensor_is_old(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'DB_PATH',Path(folder)/'test.db'):
            app.init_db();epoch=int(time.time())-86400
            with app.db() as conn:conn.execute('INSERT INTO readings(epoch,timestamp,pm02) VALUES(?,?,?)',(epoch,'old',100))
            handler=object.__new__(app.Handler);handler.path='/api/weather';captured=[]
            handler.send_json=lambda payload,*args:captured.append(payload)
            with patch.object(app,'weather_outlook',side_effect=lambda rows,reference:{'reference':reference}) as weather:
                before=int(time.time());handler.do_GET();after=int(time.time())
            self.assertTrue(before<=captured[0]['weather']['reference']<=after)
            self.assertEqual(weather.call_args.args[0][-1]['epoch'],epoch)

    def test_future_source_revision_does_not_change_earlier_dashboard_weather(self):
        import json
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'DB_PATH',Path(folder)/'test.db'):
            app.init_db();epoch=int(time.time())-300
            old={'fetchedEpoch':epoch,'source':'test','hourly':[],'revision':'original'}
            revised={**old,'revision':'late-correction'}
            import sqlite3
            with closing(sqlite3.connect(app.DB_PATH)) as conn:
                app.observation_provenance.record_source_receipt(conn,'weather_forecast_runs',json.dumps(old),epoch,epoch+1,clock=lambda:epoch+2)
                conn.commit()
                app.observation_provenance.confirm_receipts_visible(conn,clock=lambda:epoch+3)
                conn.commit()
                app.observation_provenance.record_source_receipt(conn,'weather_forecast_runs',json.dumps(revised),epoch,epoch+200,clock=lambda:epoch+201)
                conn.commit()
                app.observation_provenance.confirm_receipts_visible(conn,clock=lambda:epoch+202)
                conn.commit()
                conn.execute('INSERT INTO weather_forecast_runs VALUES(?,?,?,?)',(epoch,'test','test',json.dumps(revised)))
                conn.commit()
            self.assertEqual(app.latest_weather_payload(epoch+100)['revision'],'original')
            self.assertEqual(app.latest_weather_payload(epoch+250)['revision'],'late-correction')

    def test_publication_ledger_retains_input_hash_and_unavailable_heads(self):
        import json
        from forecast_payload import loads
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'DB_PATH',Path(folder)/'test.db'):
            app.init_db();issue=int(time.time());manifest=app.observation_provenance.build_input_manifest([],[],issue)
            result={'available':True,'forecastIssuedEpoch':issue,'current':{},'windows':{},'computation':{},'_materializedInputManifest':manifest}
            app.archive_published_analysis(result)
            self.assertTrue(result['computation']['publishedForecastArchived'])
            self.assertFalse(result['inputProvenance']['allReceiptsKnown'])
            with app.db() as conn:
                saved=loads(conn.execute('SELECT payload FROM dashboard_forecast_issues').fetchone()[0])
                self.assertEqual(saved['inputProvenance']['snapshotSha256'],result['inputProvenance']['snapshotSha256'])
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM forecast_input_snapshots').fetchone()[0],1)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM prospective_validation_issued').fetchone()[0],4)

    def test_open_read_snapshot_cannot_block_publication_archive(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'DB_PATH',Path(folder)/'test.db'):
            app.init_db();issue=int(time.time())
            with closing(sqlite3.connect(app.DB_PATH.as_uri()+'?mode=ro',uri=True,timeout=1)) as reader:
                # A nightly refit holds a read snapshot like this one while the writer archives.
                reader.execute('BEGIN');reader.execute('SELECT COUNT(*) FROM readings').fetchone()
                result={'available':True,'forecastIssuedEpoch':issue,'current':{},'windows':{},'computation':{},
                        '_materializedInputManifest':app.observation_provenance.build_input_manifest([],[],issue)}
                started=time.monotonic();app.archive_published_analysis(result)
                self.assertLess(time.monotonic()-started,5)
                self.assertTrue(result['computation']['publishedForecastArchived'])
                self.assertEqual(reader.execute('PRAGMA journal_mode').fetchone()[0],'wal')
                self.assertEqual(reader.execute('SELECT COUNT(*) FROM dashboard_forecast_issues').fetchone()[0],0)
                reader.commit()
                self.assertEqual(reader.execute('SELECT COUNT(*) FROM dashboard_forecast_issues').fetchone()[0],1)

    def test_failed_archive_cannot_publish_phantom_snapshot(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as folder, patch.object(app,'DB_PATH',Path(folder)/'test.db'):
            app.init_db();issue=int(time.time())
            result={'available':True,'forecastIssuedEpoch':issue,'current':{},'windows':{},'computation':{},
                    '_materializedInputManifest':app.observation_provenance.build_input_manifest([],[],issue)}
            with patch.object(app,'record_dashboard_issue',side_effect=sqlite3.OperationalError('injected archive failure')):
                with self.assertRaises(sqlite3.OperationalError):app.archive_published_analysis(result)
            self.assertNotIn('inputProvenance',result)
            with app.db() as conn:self.assertEqual(conn.execute('SELECT COUNT(*) FROM forecast_input_snapshots').fetchone()[0],0)

    def test_transient_background_lock_retries_without_blocking_other_diagnostics(self):
        from unittest.mock import Mock
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        import sqlite3
        with patch.object(app.near_term_hazard,'refresh_model',side_effect=sqlite3.OperationalError('database is locked')), \
                patch.object(app.short_horizon_change,'refresh_model',return_value={'available':True}) as short, \
                patch.object(app.weather_session_forecast,'refresh_model',return_value={'state':'ready'}) as weather:
            app.near_diagnostic_model_worker(stop)
        short.assert_called_once()
        # Primary Morning preparation belongs to its independent worker.
        weather.assert_not_called()
        stop.wait.assert_called_once_with(60)

    def test_patch_runtime_load_survives_background_step_failure(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        cache = object()
        runtime = SimpleNamespace(get_cache=Mock(return_value=cache),
            background_step=Mock(side_effect=RuntimeError('injected collection failure')))
        blend = SimpleNamespace(step=Mock(return_value={'state':'cached'}))
        with patch.dict('sys.modules', {'patchtst_live_runtime':runtime,'afternoon_wind61':blend}), \
                patch.object(app.patchtst_session_forecast,'prepare_runtime',return_value={'available':True}) as prepare:
            app.patchtst_research_worker(stop)
        prepare.assert_called_once()
        self.assertIs(prepare.call_args.args[0],cache)
        runtime.background_step.assert_called_once_with(app.DB_PATH)
        blend.step.assert_not_called()
        stop.wait.assert_called_once_with(30)

    def test_fresh_baseline_cannot_qualify_stale_candidate_inputs(self):
        from test_session_prediction_policy import value,ISSUE,START,END
        candidate=value();candidate.update(sensorAnchor=180,sensorReferenceEpoch=ISSUE-86400)
        candidate['modelEvidence']={"asIssuedBeforeOutcome":True,"independentOrigins":True,
            "incompleteTargetsScored":False,"candidateModelVersion":"candidate_v2",
            "numericalPolicyIdentifier":"raw_recipe_v2","scope":"same_day_afternoon",
            "targetClockMatched":True,"issueClockMatched":True,"matchedIssueLocalSeconds":43200,
            "matchedTargetStartLocalSeconds":50400,"matchedDurationSeconds":7200,
            "matchedActualLeadSeconds":7200,"issueClockToleranceSeconds":0,
            "validationMode":"as_issued_same_session_target_lead_and_local_issue_clock",
            "count":14,"distinctDays":14,"mae":10,"persistenceMae":15,
            "rmse":12,"persistenceRmse":17,"p90AbsoluteError":19,
            "persistenceP90AbsoluteError":25,"positiveDayFraction":.8}
        result=app.qualify_session_outputs([{'epoch':ISSUE-30,'pm02':100}],
            {'afternoon':{'startEpoch':START,'endEpoch':END}},ISSUE,{'afternoon':candidate})['afternoon']
        self.assertEqual(result['mean'],100)
        self.assertFalse(result['qualificationPolicy']['qualified'])
        self.assertEqual(result['candidateForecast']['originalSensorReference']['sensorAnchor'],180)


if __name__=='__main__':unittest.main()
