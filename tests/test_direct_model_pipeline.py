"""Exercise real publication wiring with fixed fake estimators and a temp DB."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import BukitKiara_Dashboard as app
from forecast_clock import ForecastClock

ISSUE = int(datetime.fromisoformat('2026-10-08T13:30:05+08:00').timestamp())
NEAR_POINT = 157.731234
NEAR_MEAN = 141.312345
MORNING = 133.612345
AFTERNOON = 113.13036727905273


def session(value, model, target, issue):
    return {'available':True,'mean':value,'prediction':value,'modelVersion':model,
        'pointRole':'experimental_model_output','forecastState':'model_output',
        'forecastedAtEpoch':issue,**target,'sensorAnchor':163.2,'baselinePoint':163.2,
        'weatherAvailable':True,'weatherFetchedEpoch':issue-60,
        'numericalPolicyIdentifier':'fixture_direct_model_output','prospectivelyValidated':False,
        'usedForDecision':False,'validated':False,
        'modelOutputPolicy':{'fixedModel':True,'numericalPerformanceSelection':False},
        'candidateForecast':{'available':True,'mean':value,'prediction':value,
            'modelVersion':model,'forecastedAtEpoch':issue,**target},
        'modelEvidence':{'mae':999,'persistenceMae':1,'skillGatePassed':False}}


class DirectPipelineTests(unittest.TestCase):
    def test_real_pipeline_publishes_estimator_outputs_and_never_calls_selectors(self):
        clock=ForecastClock(ISSUE,ISSUE//900*900).metadata()
        reference={'available':True,'point':163.2,'baselinePoint':163.2,
            'pointRole':'persistence_anchor','freshnessAdjustment':{'applied':True},
            'forecastClock':clock}
        air={'available':True,'forecastClock':clock,'arrival':dict(reference),'trail':dict(reference)}
        shadow={'enabled':True,'available':True,'status':'available','forecastClock':clock,
            'featureAnchor':157.0,'featureAnchorEpoch':clock['featureAnchorEpoch'],
            'arrivalPoint':NEAR_POINT,'trailMeanPoint':NEAR_MEAN,
            'trailExtrema':{'available':True,'low':128.123456,'high':139.987654,
                'forecastClock':clock,'modelVersion':'local_extra_trees_joint_extrema90_210_v1'},
            'backtest':{'deploymentSkill':{'arrival':{'eligible':False},'trailMean':{'eligible':False}}}}
        fresh={'available':True,'arrivalPoint':NEAR_POINT,'trailMeanPoint':NEAR_MEAN,
            'trailMinimumPoint':128.123456,'trailMaximumPoint':139.987654,
            'modelVersion':app.fresh_numeric_model.MODEL_VERSION,
            'forecastClock':app.fresh_numeric_model.forecast_clock(ISSUE),
            'freshReferencePm25':163.2,
            'performanceEvidence':{'mae':999,'baselineMae':1,'qualified':False}}
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(app,'DB_PATH',Path(folder)/'test.db'))
            app.init_db()
            with app.db() as conn:
                conn.execute('INSERT INTO readings(epoch,timestamp,pm02,pm10,atmp,rhum,heatindex) VALUES(?,?,?,?,?,?,?)',
                    (ISSUE-30,'test',163.2,170,30,50,39))
            stack.enter_context(patch.object(app.time,'time',return_value=ISSUE))
            stack.enter_context(patch.object(app,'air_window_analysis',return_value=air))
            stack.enter_context(patch.object(app,'refresh_near_term_persistence',return_value=air))
            stack.enter_context(patch.object(app,'cached_shadow_analogue_outlook',return_value=shadow))
            stack.enter_context(patch.object(app.fresh_numeric_model,'predict',return_value=fresh))
            stack.enter_context(patch.object(app,'record_local_pm_forecast_issue'))
            stack.enter_context(patch.object(app,'rapid_clearance_event_outlook',return_value={}))
            stack.enter_context(patch.object(app,'shadow_dry_dispersion_watch',return_value={}))
            stack.enter_context(patch.object(app,'attach_experimental_rapid_change_risk',return_value={'experimentalRapidChangeRisk':{}}))
            for name in ('separate_observation_from_prediction','attach_experimental_short_horizon_change',
                         'attach_first_crossing_event_forecast','attach_cycling_window_forecast'):
                stack.enter_context(patch.object(app,name,side_effect=lambda value,*args:value))
            stack.enter_context(patch.object(app,'weather_outlook',return_value={'windows':{'morning':{},'afternoon':{}}}))
            stack.enter_context(patch.object(app.regional_pm_inputs,'describe_context',return_value={}))
            stack.enter_context(patch.object(app,'regional_haze_window_outlook',return_value={}))
            def morning(db,rows,windows,issue,**kwargs):
                self.assertEqual(set(windows),{'morning'})
                return {'morning':session(MORNING,app.weather_session_forecast.MODEL_VERSION,windows['morning'],issue)}
            def afternoon(rows,windows,issue,estimates,**kwargs):
                return {**estimates,'afternoon':session(AFTERNOON,app.patchtst_session_forecast.MODEL_VERSION,windows['afternoon'],issue)}
            stack.enter_context(patch.object(app.weather_session_forecast,'predict_windows',side_effect=morning))
            stack.enter_context(patch.object(app.patchtst_session_forecast,'apply_experimental_afternoon',side_effect=afternoon))
            forbidden=[]
            for owner,name in ((app,'aggressive_air_window_forecast'),(app,'qualify_session_outputs'),
                (app,'cached_ride_window_particle_forecast'),(app.afternoon_xgboost_forecast,'predict_windows'),
                (app.afternoon_direction_forecast,'apply_experimental_afternoon')):
                forbidden.append(stack.enter_context(patch.object(owner,name,side_effect=AssertionError('numeric selector called'))))
            result=app._compute_analysis(28,include_trials=False)
            self.assertEqual(result['airWindow']['arrival']['point'],NEAR_POINT)
            self.assertEqual(result['airWindow']['trail']['point'],NEAR_MEAN)
            self.assertEqual(result['airWindow']['trail']['rideExtrema']['low'],128.123456)
            self.assertEqual(result['airWindow']['trail']['rideExtrema']['high'],139.987654)
            # The publication seam never recenters model outputs around a mean.
            self.assertGreater(NEAR_MEAN,result['airWindow']['trail']['rideExtrema']['high'])
            for key,expected in (('morning',MORNING),('afternoon',AFTERNOON)):
                self.assertEqual(result['windowPrediction']['predictions'][key]['mean'],expected)
                self.assertEqual(result['windows'][key]['particleForecast']['point'],expected)
                self.assertEqual(result['rideForecast']['windows'][key]['point'],expected)
                compact=app.compact_window_particle_forecast(result['windows'][key]['particleForecast'])
                self.assertTrue(compact['modelPointApplied'])
                self.assertEqual(compact['numericalPolicyIdentifier'],'fixture_direct_model_output')
                self.assertEqual(compact['baselineMeanPm25UgM3'],163.2)
            for method in forbidden:method.assert_not_called()
            result['computation']={'seconds':0.1}
            app.archive_published_analysis(result)
            with app.db() as conn:
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM prospective_validation_issued').fetchone()[0],4)


if __name__=='__main__':unittest.main()
