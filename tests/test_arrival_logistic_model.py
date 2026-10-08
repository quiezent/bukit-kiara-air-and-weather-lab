
from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import arrival_logistic_model as model
import arrival_feature_inputs as features
from forecast_clock import point_weights


class LiteralClassifier:
    classes_=np.asarray([0,1,2,3,4])
    def predict_proba(self,x):return np.tile([.123456789,.2,.4,.15,.126543211],(len(x),1))


class LogisticArrivalDistributionTests(unittest.TestCase):
    def setUp(self):
        self.issue=1791454200
        self.rows=[{'epoch':self.issue-age,'pm02':100-age/300,'atmp':29,'rhum':65} for age in range(10800,-1,-180)]
        self.prepared=features.prepare_inputs(self.rows)
        self.reference=features.issue_features(self.prepared,self.issue)['freshPm25']
        self.artifact={'modelVersion':model.MODEL_VERSION,'columns':list(model.FEATURE_COLUMNS),
            'identity':model._identity(),'estimator':LiteralClassifier(),'trainingCutoffEpoch':model.midnight_epoch(self.issue),
            'metadata':{'available':True,'fittedAtEpoch':self.issue-1,'maxTrainingOutcomeEpoch':model.midnight_epoch(self.issue)-900}}

    def infer(self,*,rows=None,reference=None,watermark=None,prepared=None):
        with patch.object(model,'_load_artifact',return_value=self.artifact):
            return model.predict_arrival('unused',self.rows if rows is None else rows,self.issue,
                self.issue if watermark is None else watermark,self.reference if reference is None else reference,
                feature_inputs=prepared)

    def test_target_boundaries_are_inclusive_at_20_and_40(self):
        values=[-41,-40,-39.999,-20,-19.999,0,19.999,20,39.999,40,41]
        self.assertEqual(model.label_classes(values).tolist(),[0,0,1,1,2,2,2,3,3,4,4])
        with self.assertRaises(ValueError):model.label_classes([float('nan')])

    def test_training_origins_match_frozen_aligned_five_minute_cadence(self):
        epochs=model.dense_origins(1791388801,1791390300)
        self.assertTrue(np.array_equal(epochs,[1791389100,1791389400,1791389700,1791390000]))
        self.assertTrue((epochs%300==0).all())
        self.assertEqual(model.selected_policy()['trainingGridOffsetSeconds'],0)

    def test_native_five_masses_and_cdf_tails_are_literal_no_argmax(self):
        with (patch.object(model,'refresh_model',side_effect=AssertionError('request fit')),
              patch.object(features,'load_inputs',side_effect=AssertionError('request DB read'))):
            result=self.infer()
        self.assertTrue(result['available'],result)
        expected=LiteralClassifier().predict_proba(np.zeros((1,1)))[0]
        self.assertTrue(np.array_equal(result['classProbabilities'],expected))
        self.assertEqual(result['probabilityFall20'],expected[0]+expected[1])
        self.assertEqual(result['probabilityFall40'],expected[0])
        self.assertEqual(result['probabilityRise20'],expected[3]+expected[4])
        self.assertEqual(result['probabilityRise40'],expected[4])
        self.assertEqual(result['probabilityWithin20'],expected[2])
        self.assertLessEqual(result['probabilityFall40'],result['probabilityFall20'])
        self.assertLessEqual(result['probabilityRise40'],result['probabilityRise20'])
        self.assertEqual(result['arrivalEpoch'],self.issue+5400)
        self.assertNotIn('direction',result)
        self.assertIsNone(result['arrivalConcentrationExpectation'])

    def test_absent_training_classes_receive_only_native_zero_not_synthetic_mass(self):
        class Partial:
            classes_=np.array([1,2,3])
            def predict_proba(self,x):return np.tile([.2,.5,.3],(len(x),1))
        self.artifact['estimator']=Partial()
        p=model.predict_probabilities(self.artifact,np.zeros(len(self.artifact['columns'])))[0]
        self.assertTrue(np.array_equal(p,[0,.2,.5,.3,0]))
        result=self.infer()
        self.assertEqual(result['probabilityFall40'],0)
        self.assertEqual(result['probabilityRise40'],0)

    def test_future_sensor_values_and_receipts_cannot_change_prediction(self):
        original=self.infer()
        changed=self.infer(rows=self.rows+[{'epoch':self.issue+1,'pm02':9999,'atmp':90,'rhum':0}])
        self.assertEqual(original['featureValuesSha256'],changed['featureValuesSha256'])
        self.assertEqual(original['classProbabilities'],changed['classProbabilities'])
        hidden=self.infer(rows=self.rows+[{'epoch':self.issue,'pm02':9999,'atmp':90,'rhum':0,'available_epoch':self.issue+1}])
        self.assertEqual(original['featureValuesSha256'],hidden['featureValuesSha256'])

    def test_weather_context_cannot_change_sensor_only_native_output(self):
        original=self.infer()
        changed=features.issue_features(self.prepared,self.issue)
        changed.update({name:999999 for name in features.WEATHER_FEATURE_COLUMNS})
        with patch.object(features,'issue_features',return_value=changed):result=self.infer()
        self.assertEqual(original['featureValuesSha256'],result['featureValuesSha256'])
        self.assertEqual(original['classProbabilities'],result['classProbabilities'])
        self.assertFalse(result['weatherFeaturesUsed'])
        self.assertFalse(result['actualWeatherLearningSupported'])

    def test_low_reference_learned_fall_mass_is_preserved(self):
        rows=[{**row,'pm02':row['pm02']/10} for row in self.rows]
        reference=features.issue_features(features.prepare_inputs(rows),self.issue)['freshPm25']
        result=self.infer(rows=rows,reference=reference)
        self.assertTrue(result['available'],result)
        self.assertEqual(result['probabilityFall40'],.123456789)

    def test_reference_watermark_and_fitted_clocks_fail_without_output_substitution(self):
        self.assertFalse(self.infer(reference=self.reference+1)['available'])
        self.assertFalse(self.infer(watermark=self.issue+1)['available'])
        self.assertFalse(self.infer(watermark=self.issue-180,prepared=self.prepared)['available'])
        self.artifact['metadata']['fittedAtEpoch']=self.issue+1
        result=self.infer()
        self.assertFalse(result['available'])
        self.assertNotIn('classProbabilities',result)
        self.artifact['metadata']['fittedAtEpoch']=self.issue-1
        self.artifact['trainingCutoffEpoch']-=86400
        self.assertFalse(self.infer()['available'])

    def test_malformed_native_mass_is_rejected_not_repaired(self):
        class Invalid:
            classes_=np.array([0,1,2,3,4])
            def predict_proba(self,x):return np.tile([.2,.2,.2,.2,-.1],(len(x),1))
        self.artifact['estimator']=Invalid()
        self.assertFalse(self.infer()['available'])
        with self.assertRaises(ValueError):model.tail_probabilities([.2,.2,.2,.2,-.1])

    def test_training_origin_feature_label_cutoff_and_embargo_precede_fit(self):
        cutoff=model.midnight_epoch(self.issue)
        issues=pd.DataFrame([{'issueEpoch':cutoff-86400*(i%12+1),'completeEpoch':cutoff-1,
            'featureSourceMaxEpoch':cutoff-86400*(i%12+1),'issueDay':str(i%12),
            'arrivalClass':[1,2,3][i%3],'arrivalDelta':[-25,0,25][i%3]} for i in range(300)])
        prepared={'issues':issues,'features':np.zeros((300,len(model.FEATURE_COLUMNS))),'columns':list(model.FEATURE_COLUMNS)}
        issues.loc[0,'completeEpoch']=cutoff
        with self.assertRaisesRegex(ValueError,'label_cutoff'):model.fit_prepared(prepared,cutoff)
        issues.loc[0,'completeEpoch']=cutoff-1;issues.loc[0,'issueEpoch']=cutoff-7199
        with self.assertRaisesRegex(ValueError,'origin_embargo'):model.fit_prepared(prepared,cutoff)
        issues.loc[0,'issueEpoch']=cutoff-86400;issues.loc[0,'featureSourceMaxEpoch']=cutoff
        with self.assertRaisesRegex(ValueError,'feature_clock'):model.fit_prepared(prepared,cutoff)
        issues.loc[0,'featureSourceMaxEpoch']=cutoff-86400;issues.loc[0,'issueEpoch']=cutoff-86400+1
        with self.assertRaisesRegex(ValueError,'grid_alignment'):model.fit_prepared(prepared,cutoff)

    def test_arrival_label_needs_exact90_support_not_every_intermediate_event_bucket(self):
        cutoff=model.midnight_epoch(self.issue);issue=cutoff-10800+137;origin=issue//900*900
        rows=[{'epoch':issue-age,'pm02':100,'atmp':29,'rhum':65} for age in range(10800,-1,-180)]
        weights=point_weights(issue-origin+5400)
        labels=[origin+offset*900 for offset in weights]
        levels=[65,55]
        for label,level in zip(labels,levels):
            rows.extend({'epoch':label-age,'pm02':level,'atmp':29,'rhum':65} for age in [720,540,360,180,0])
        source=features.prepare_inputs(rows)
        prepared=model.prepare_training_data(source,[issue],cutoff)
        self.assertEqual(len(prepared['issues']),1)
        expected=sum(level*weight for level,weight in zip(levels,weights.values()))
        row=prepared['issues'].iloc[0]
        self.assertAlmostEqual(row.arrivalPm,expected)
        self.assertAlmostEqual(row.arrivalDelta,expected-100)
        self.assertEqual(row.completeEpoch,max(labels))
        self.assertEqual(row.arrivalClass,int(model.label_classes(expected-100)))
        excluded=model.prepare_training_data(source,[issue],max(labels))
        self.assertTrue(excluded['issues'].empty)
        revisions=[]
        for row in rows:
            if row['epoch']>issue:
                revisions.extend([{**row,'pm02':row['pm02']+10,'available_epoch':cutoff-2},
                                  {**row,'pm02':9999,'available_epoch':cutoff+1}])
        revised=model.prepare_training_data(features.prepare_inputs(rows+revisions),[issue],cutoff)
        self.assertAlmostEqual(revised['issues'].iloc[0].arrivalPm,expected+10)

    def test_cached_artifact_source_identity_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/model.ARTIFACT_NAME
            path.write_bytes(pickle.dumps(self.artifact))
            self.assertIsNotNone(model._load_artifact(directory))
            corrupted={**self.artifact,'identity':{'selectedPolicy':'different'}}
            path.write_bytes(pickle.dumps(corrupted))
            self.assertIsNone(model._load_artifact(directory))

    def test_fitted_sparse_class_distribution_and_missing_weather_support_remain_literal(self):
        cutoff=model.midnight_epoch(self.issue)
        records=[{'issueEpoch':cutoff-86400*(i%12+1),'completeEpoch':cutoff-1,
            'featureSourceMaxEpoch':cutoff-86400*(i%12+1),'issueDay':str(i%12),
            'arrivalClass':[1,2,3][i%3],'arrivalDelta':[-25,0,25][i%3]} for i in range(300)]
        x=np.full((300,len(model.FEATURE_COLUMNS)),np.nan)
        x[:,0]=[100+row['arrivalDelta'] for row in records]
        prepared={'issues':pd.DataFrame(records),'features':x,'columns':list(model.FEATURE_COLUMNS)}
        artifact=model.fit_prepared(prepared,cutoff,fitted_epoch=self.issue-1)
        native=artifact['estimator'].predict_proba(x[:2])
        ordered=model.predict_probabilities(artifact,x[:2])
        self.assertTrue(np.array_equal(ordered[:,[1,2,3]],native))
        self.assertTrue(np.array_equal(ordered[:,[0,4]],np.zeros((2,2))))
        self.assertEqual(artifact['metadata']['unsupportedClassIds'],[0,4])
        self.assertFalse(artifact['metadata']['actualWeatherLearningSupported'])
        self.assertTrue(all(count==0 for count in artifact['metadata']['externalFeatureNonmissingTrainingCounts'].values()))

    def test_actual_publication_keeps_reference_arrival_and_literal_mass(self):
        result=self.infer()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'issued.jsonl'
            with self.assertRaises(ValueError):model.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+121)
            self.assertTrue(model.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+25,input_lineage={'snapshotId':'actual'}))
            self.assertFalse(model.record_issued_prediction(result,output_path=path,recorded_epoch=self.issue+25))
            saved=json.loads(path.read_text())
            self.assertEqual(saved['classProbabilities'],result['classProbabilities'])
            self.assertEqual(saved['referencePm'],self.reference)
            self.assertEqual(saved['arrivalEpoch'],self.issue+5400)
            self.assertTrue(saved['isOriginallyIssued'])

    def test_default_publication_path_is_an_isolated_logistic_cohort(self):
        result=self.infer()
        with tempfile.TemporaryDirectory() as directory,patch.object(model,'PRIVATE',Path(directory)):
            self.assertTrue(model.record_issued_prediction(result,recorded_epoch=self.issue+25))
            path=Path(directory)/'arrival-logistic-change-issued-v1.jsonl'
            saved=json.loads(path.read_text())
            self.assertEqual(saved['issuedCohort']['modelVersion'],model.MODEL_VERSION)
            self.assertEqual(saved['issuedCohort']['trainingCadence'],model.TRAINING_CADENCE)
            self.assertFalse((Path(directory)/'arrival-change-issued-v1.jsonl').exists())


if __name__=='__main__':unittest.main()
