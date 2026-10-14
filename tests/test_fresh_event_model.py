
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

import fresh_event_model as m
import fresh_sensor_features as sensor


class ConstantHazards:
    classes_ = np.array([-1,0,1])
    def predict_proba(self,x):
        return np.tile([0.1,0.85,0.05],(len(x),1))


def rows_before(issue):
    return [{'epoch':issue-age,'pm02':100-age/120,'atmp':29.0,'rhum':65.0}
            for age in range(10800,-1,-180)]


class FreshEventTests(unittest.TestCase):
    def setUp(self):
        self.issue = 1791449400
        self.rows = rows_before(self.issue)
        self.features = sensor.issue_features(self.rows,self.issue)
        self.reference = self.features['freshPm25']
        self.artifact = {'modelVersion':m.MODEL_VERSION,'columns':list(sensor.FEATURE_COLUMNS),
            'identity':m._identity(),'estimator':ConstantHazards(),
            'trainingCutoffEpoch':m.midnight_epoch(self.issue),
            'metadata':{'fittedAtEpoch':self.issue-1,'maxTrainingOutcomeEpoch':m.midnight_epoch(self.issue)-900,
                        'trainingCutoffEpoch':m.midnight_epoch(self.issue),'available':True}}

    def test_dense_cadence_covers_every_quarter_hour_minute(self):
        times=m.dense_origins(0,86400)
        self.assertEqual(set((times//60)%15),set(range(15)))
        self.assertEqual(set(times%60),{30})
        self.assertGreaterEqual(np.diff(times).min(),60)
        self.assertLessEqual(np.diff(times).max(),360)

    def test_absorbing_probabilities_keep_mass_and_monotonic_causes(self):
        result=m.cumulative_incidence(np.tile([.1,.8,.1],(6,1)))
        np.testing.assert_allclose(result.sum(axis=1),1)
        self.assertTrue((np.diff(result[:,0])>0).all())
        self.assertTrue((np.diff(result[:,2])>0).all())
        self.assertTrue((np.diff(result[:,1])<0).all())
        self.assertAlmostEqual(result[-1,1],.8**6)

    def test_invalid_hazard_distribution_is_rejected(self):
        with self.assertRaises(ValueError): m.cumulative_incidence(np.zeros((6,3)))
        with self.assertRaises(ValueError): m.cumulative_incidence(np.tile([.1,1,.1],(6,1)))

    def test_runtime_is_literal_cached_inference_and_never_trains(self):
        expected=m.cumulative_incidence(np.tile([.1,.85,.05],(6,1)))[-1]
        with patch.object(m,'_load_artifact',return_value=self.artifact), \
             patch.object(m,'refresh_model',side_effect=AssertionError('request-path fitting')), \
             patch.object(m,'read_training_rows',side_effect=AssertionError('request-path training sources')):
            result=m.predict_event('unused',self.rows,self.issue,self.issue,self.reference)
        self.assertTrue(result['available'],result)
        np.testing.assert_allclose([result['probabilityDrop'],result['probabilityNoCrossing'],result['probabilityRise']],expected)
        self.assertFalse(result['qualification']['operationalUseEligible'])
        self.assertEqual(result['freshReferenceEpoch'],self.issue)
        self.assertEqual(result['featureSourceMaxEpoch'],self.issue)

    def test_future_observation_cannot_change_runtime_predictions(self):
        with patch.object(m,'_load_artifact',return_value=self.artifact):
            before=m.predict_event('unused',self.rows,self.issue,self.issue,self.reference)
            after=m.predict_event('unused',self.rows+[{'epoch':self.issue+1,'pm02':9999,'atmp':90,'rhum':0}],self.issue,self.issue,self.reference)
        self.assertEqual(before['featureValuesSha256'],after['featureValuesSha256'])
        self.assertEqual(before['probabilityDrop'],after['probabilityDrop'])

    def test_mismatched_reference_returns_unavailable_without_substitute(self):
        with patch.object(m,'_load_artifact',return_value=self.artifact):
            result=m.predict_event('unused',self.rows,self.issue,self.issue,self.reference+1)
        self.assertFalse(result['available'])
        self.assertNotIn('probabilityDrop',result)
        self.assertIsNone(result['displayText'])

    def test_low_reference_tail_probability_remains_literal(self):
        rows=[{**r,'pm02':r['pm02']/10} for r in self.rows]
        reference=sensor.issue_features(rows,self.issue)['freshPm25']
        with patch.object(m,'_load_artifact',return_value=self.artifact):
            result=m.predict_event('unused',rows,self.issue,self.issue,reference)
        self.assertTrue(result['available'],result)
        self.assertTrue(result['impossibleDropSupport'])
        self.assertAlmostEqual(result['probabilityDrop'],m.cumulative_incidence(np.tile([.1,.85,.05],(6,1)))[-1,0])

    def test_unique_near_tie_is_preserved_as_model_argmax(self):
        class NearTie:
            def predict_proba(self,x):
                return np.asarray([[.1,.45-5e-14,.45+5e-14]]+[[0.,1.,0.]]*5)
        self.artifact['estimator']=NearTie()
        with patch.object(m,'_load_artifact',return_value=self.artifact):
            result=m.predict_event('unused',self.rows,self.issue,self.issue,self.reference)
        self.assertTrue(result['available'])
        self.assertEqual(result['diagnosticDirection'],'rise')
        self.assertEqual(result['directionTieTolerance'],0.0)

    def test_model_fitted_after_issue_cannot_claim_historical_availability(self):
        self.artifact['metadata']['fittedAtEpoch']=self.issue+1
        with patch.object(m,'_load_artifact',return_value=self.artifact):
            result=m.predict_event('unused',self.rows,self.issue,self.issue,self.reference)
        self.assertFalse(result['available'])
        self.assertIn('model_not_yet_fitted_at_issue',result['unavailableReasons'])

    def test_training_target_purge_and_no_future_row_influence(self):
        cutoff=m.midnight_epoch(self.issue)
        rows=[{'epoch':cutoff-4*3600+i*180,'pm02':100+15*np.sin(i/5),'atmp':29,'rhum':65} for i in range(80)]
        times=m.dense_origins(cutoff-3*3600,cutoff)
        data=m.prepare_training_data(rows,times,cutoff)
        changed=m.prepare_training_data(rows+[{'epoch':cutoff,'pm02':9999,'atmp':90,'rhum':0}],times,cutoff)
        pd.testing.assert_frame_equal(data['issues'],changed['issues'])
        np.testing.assert_allclose(data['features'],changed['features'],equal_nan=True)
        self.assertTrue(data['issues'].completeEpoch.lt(cutoff).all())
        self.assertTrue((data['issues'].issueEpoch+5400<cutoff).all())

    def test_risk_rows_stop_at_first_crossing(self):
        data={'issues':pd.DataFrame([{'firstStep':2,'cause':-1},{'firstStep':-1,'cause':0}]),
              'features':np.zeros((2,3))}
        x,y=m.risk_arrays(data)
        self.assertEqual(len(y),9)
        self.assertEqual(list(y),[0,0,-1,0,0,0,0,0,0])

    def test_publication_requires_timely_clock_and_preserves_raw_outputs(self):
        result={'forecastIssuedEpoch':self.issue,'modelIdentity':{'one':'model'},
                'probabilityDrop':.123456789,'featureValuesSha256':'hash','reason':'available'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'issued.jsonl'
            with self.assertRaises(ValueError):
                m.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+121)
            self.assertTrue(m.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+35,input_lineage={'snapshotId':'real'}))
            self.assertFalse(m.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+35))
            archived=json.loads(path.read_text())
            self.assertEqual(archived['probabilityDrop'],.123456789)
            self.assertEqual(archived['inputSnapshotId'],'real')
            self.assertTrue(archived['isOriginallyIssued'])


if __name__ == '__main__': unittest.main()
