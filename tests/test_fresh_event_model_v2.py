
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
import fresh_event_model_v2 as m
import fresh_sensor_features as sensor


class LiteralClassifier:
    classes_=np.array([-1,0,1])
    def predict_proba(self,x): return np.tile([.6123456789,.3,.0876543211],(len(x),1))


class DirectEventTests(unittest.TestCase):
    def setUp(self):
        self.issue=1791449400
        self.rows=[{'epoch':self.issue-age,'pm02':100-age/120,'atmp':29,'rhum':65} for age in range(10800,-1,-180)]
        self.ref=sensor.issue_features(self.rows,self.issue)['freshPm25']
        self.artifact={'modelVersion':m.MODEL_VERSION,'columns':list(sensor.FEATURE_COLUMNS),
            'identity':m._identity(),'estimator':LiteralClassifier(),'trainingCutoffEpoch':m.midnight_epoch(self.issue),
            'metadata':{'available':True,'fittedAtEpoch':self.issue-1,'maxTrainingOutcomeEpoch':m.midnight_epoch(self.issue)-900}}

    def infer(self,rows=None,reference=None):
        with patch.object(m,'_load_artifact',return_value=self.artifact):
            return m.predict_event('unused',self.rows if rows is None else rows,self.issue,self.issue,self.ref if reference is None else reference)

    def test_native_direct_probabilities_and_single_true_learned_horizon(self):
        with patch.object(m,'refresh_model',side_effect=AssertionError('request fit')):
            result=self.infer()
        self.assertTrue(result['available'],result)
        self.assertEqual(result['probabilityDrop'],.6123456789)
        self.assertEqual(result['probabilityNoCrossing'],.3)
        self.assertEqual(result['probabilityRise'],.0876543211)
        self.assertEqual([h['leadMinutes'] for h in result['horizons']],[90])
        self.assertEqual(result['diagnosticDirection'],'drop')
        self.assertFalse(result['qualification']['operationalUseEligible'])

    def test_future_sensor_values_cannot_change_features_or_prediction(self):
        old=self.infer()
        changed=self.infer(self.rows+[{'epoch':self.issue+1,'pm02':9999,'atmp':90,'rhum':0}])
        self.assertEqual(old['featureValuesSha256'],changed['featureValuesSha256'])
        self.assertEqual(old['probabilityDrop'],changed['probabilityDrop'])

    def test_no_reference_substitution_when_clock_or_reference_invalid(self):
        result=self.infer(reference=self.ref+1)
        self.assertFalse(result['available'])
        self.assertNotIn('probabilityDrop',result)
        self.assertIsNone(result['displayText'])
        self.artifact['metadata']['fittedAtEpoch']=self.issue+1
        self.assertFalse(self.infer()['available'])

    def test_low_reference_model_tail_is_preserved(self):
        rows=[{**r,'pm02':r['pm02']/10} for r in self.rows]
        reference=sensor.issue_features(rows,self.issue)['freshPm25']
        result=self.infer(rows,reference)
        self.assertTrue(result['available'])
        self.assertTrue(result['impossibleDropSupport'])
        self.assertEqual(result['probabilityDrop'],.6123456789)

    def test_exact_argmax_keeps_unique_near_tie(self):
        class NearTie:
            def predict_proba(self,x): return np.tile([.1,.45-5e-14,.45+5e-14],(len(x),1))
        self.artifact['estimator']=NearTie()
        result=self.infer()
        self.assertEqual(result['diagnosticDirection'],'rise')
        self.assertEqual(result['directionTieTolerance'],0)

    def test_malformed_native_mass_is_unavailable(self):
        class Invalid:
            def predict_proba(self,x): return np.zeros((len(x),3))
        self.artifact['estimator']=Invalid()
        self.assertFalse(self.infer()['available'])

    def test_training_feature_and_target_cutoffs_checked_before_estimator_fit(self):
        cutoff=m.midnight_epoch(self.issue)
        issues=pd.DataFrame([{'issueEpoch':cutoff-86400*(i%12+1),'completeEpoch':cutoff-1,
            'featureSourceMaxEpoch':cutoff-86400*(i%12+1),'issueDay':str(i%12),'cause':[-1,0,1][i%3]} for i in range(300)])
        prepared={'issues':issues,'features':np.zeros((300,208)),'columns':list(sensor.FEATURE_COLUMNS)}
        prepared['issues'].loc[0,'completeEpoch']=cutoff
        with self.assertRaisesRegex(ValueError,'label_cutoff'): m.fit_prepared(prepared,cutoff)
        prepared['issues'].loc[0,'completeEpoch']=cutoff-1
        prepared['issues'].loc[0,'featureSourceMaxEpoch']=cutoff
        with self.assertRaisesRegex(ValueError,'feature_clock'): m.fit_prepared(prepared,cutoff)

    def test_publication_uses_new_model_identity_and_preserves_literal_fields(self):
        result=self.infer()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v2.jsonl'
            with self.assertRaises(ValueError): m.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+121)
            self.assertTrue(m.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+25,input_lineage={'snapshotId':'actual'}))
            self.assertFalse(m.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+25))
            record=json.loads(path.read_text())
            self.assertEqual(record['issuedCohort']['modelVersion'],m.MODEL_VERSION)
            self.assertEqual(record['probabilityDrop'],.6123456789)
            self.assertTrue(record['isOriginallyIssued'])
            self.assertEqual(record['inputSnapshotId'],'actual')


if __name__=='__main__': unittest.main()
