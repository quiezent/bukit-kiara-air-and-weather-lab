"""Direct learned first90-minute three-class event probabilities.

One fixed HGB learns the displayed outcome directly. Background-only fitting;
request inference never trains, tunes a decision rule, or substitutes outputs.
"""
from __future__ import annotations
import hashlib
import json
import math
import pickle
import threading
import time
from pathlib import Path
from collections import OrderedDict
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import fresh_sensor_features as sensor
import fresh_event_model as data_source

ROOT=Path(__file__).resolve().parent
PRIVATE=ROOT/'private'
MODEL_VERSION='fresh_sensor_hgb_direct_first20_90min_v2'
ADAPTER_VERSION='fresh_sensor_direct_event_adapter_v2'
ARTIFACT_NAME='fresh-sensor-first20-model-v2.pkl'
STATUS_NAME='fresh-sensor-first20-status-v2.json'
TRAINING_CADENCE=data_source.TRAINING_CADENCE
MYT=data_source.MYT
CLASSES=(-1,0,1)
THRESHOLD=20.0
PARAMETERS=dict(max_iter=150,max_leaf_nodes=15,min_samples_leaf=20,
                l2_regularization=1.0,learning_rate=0.05,random_state=20261008,
                early_stopping=False,class_weight=None)
_LOCK=threading.RLock()
_FIT_LOCK=threading.Lock()
_RECORD_LOCK=threading.Lock()
_CACHE={}
_RECORDED=OrderedDict()
_READ_PATHS=set()
midnight_epoch=data_source.midnight_epoch
dense_origins=data_source.dense_origins
prepare_training_data=data_source.prepare_training_data
read_training_rows=data_source.read_training_rows


def selected_policy():
    return {'modelVersion':MODEL_VERSION,'parameters':dict(PARAMETERS),
            'target':'first_sampled_crossing_within90_minutes','changeThresholdUgM3':THRESHOLD,
            'thresholdComparison':'at_least','targetSampleLeadsMinutes':[15,30,45,60,75,90],
            'learnedProbabilityHorizonsMinutes':[90],
            'probabilityMethod':'direct_three_class_first90_outcome_predict_proba',
            'trainingCadence':TRAINING_CADENCE,'trainingLookbackDays':28,
            'trainingCutoff':'strict_completed_labels_before_issue_day_midnight_Asia_Kuala_Lumpur',
            'directionDecision':'three_class_argmax_ties_unresolved','calibrated':False,
            'numericalSelectionApplied':False}


def _identity():
    files=[Path(__file__),ROOT/'fresh_event_model.py',ROOT/'fresh_sensor_features.py',
           ROOT/'forecast_clock.py',ROOT/'collection_quality.py']
    return {'sourceSha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'selectedPolicy':selected_policy()}


def fit_prepared(prepared,cutoff,*,fitted_epoch=None):
    issues=prepared['issues']
    if issues.empty or len(issues)<300 or issues.issueDay.nunique()<10:
        raise ValueError('insufficient_completed_training_issues')
    if not issues.completeEpoch.lt(cutoff).all():
        raise ValueError('training_label_cutoff_violation')
    if not issues.featureSourceMaxEpoch.le(issues.issueEpoch).all():
        raise ValueError('training_feature_clock_violation')
    y=issues.cause.to_numpy(int)
    if set(y)!=set(CLASSES): raise ValueError('all_three_outcome_classes_required')
    estimator=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                            HistGradientBoostingClassifier(**PARAMETERS))
    with threadpool_limits(limits=1): estimator.fit(prepared['features'],y)
    if tuple(estimator.classes_)!=CLASSES: raise ValueError('class_order_mismatch')
    support={name:int((y==cause).sum()) for cause,name in zip(CLASSES,('drop','none','rise'))}
    metadata={'available':True,'trainingCutoffEpoch':int(cutoff),
              'fittedAtEpoch':int(time.time()) if fitted_epoch is None else int(fitted_epoch),
              'trainingIssueCount':len(issues),'trainingRows':len(issues),
              'trainingSupport':support,'trainingDistinctIssueDays':int(issues.issueDay.nunique()),
              'maxTrainingOutcomeEpoch':int(issues.completeEpoch.max()),
              'trainingMinIssueEpoch':int(issues.issueEpoch.min()),'trainingMaxIssueEpoch':int(issues.issueEpoch.max()),
              'independentIssues':False,'overlappingTrainingIssues':True,
              'historicalReceiptTimesUnknown':True,'calibrated':False,
              'trainingFitCadence':TRAINING_CADENCE,'weatherFeaturesUsed':False,
              'featureCount':len(prepared['columns']),'learnedProbabilityHorizonsMinutes':[90],
              'objective':'direct_first90_three_class_outcome'}
    return {'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,'identity':_identity(),
            'columns':list(prepared['columns']),'trainingCutoffEpoch':int(cutoff),
            'metadata':metadata,'estimator':estimator}


def predict_probabilities(artifact,features):
    x=np.asarray(features,float)
    if x.ndim==1: x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=len(artifact['columns']):
        raise ValueError('feature_shape_mismatch')
    with threadpool_limits(limits=1): p=artifact['estimator'].predict_proba(x)
    p=np.asarray(p,float)
    if p.shape!=(len(x),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('invalid_learned_probability_distribution')
    return p


def _paths(model_dir=None):
    directory=Path(model_dir) if model_dir is not None else PRIVATE
    return directory/ARTIFACT_NAME,directory/STATUS_NAME


def _load_artifact(model_dir=None):
    path,_=_paths(model_dir)
    if not path.exists(): return None
    signature=(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
    with _LOCK:
        if signature not in _CACHE:
            artifact=pickle.loads(path.read_bytes())
            if artifact.get('modelVersion')!=MODEL_VERSION or artifact.get('identity')!=_identity():
                return None
            _CACHE.clear(); _CACHE[signature]=artifact
        return _CACHE[signature]


def refresh_model(db_path,issue_epoch,*,model_dir=None):
    cutoff=midnight_epoch(issue_epoch)
    with _FIT_LOCK:
        try:
            prior=_load_artifact(model_dir)
            if prior is not None and prior['trainingCutoffEpoch']==cutoff:
                return {**prior['metadata'],'reason':'cached_fit_current'}
            rows=read_training_rows(db_path,cutoff)
            prepared=prepare_training_data(rows,dense_origins(cutoff-28*86400,cutoff),cutoff)
            artifact=fit_prepared(prepared,cutoff)
            path,status_path=_paths(model_dir)
            data_source._atomic(path,pickle.dumps(artifact,protocol=pickle.HIGHEST_PROTOCOL))
            status={**artifact['metadata'],'modelVersion':MODEL_VERSION,'reason':'available','identity':artifact['identity']}
            data_source._atomic(status_path,json.dumps(status,indent=2,allow_nan=False).encode('utf-8'))
            return status
        except Exception as error:
            return {'available':False,'modelVersion':MODEL_VERSION,'trainingCutoffEpoch':cutoff,
                    'reason':'background_fit_unavailable','unavailableReasons':[type(error).__name__,str(error)]}


def predict_event(db_path,rows,issue_epoch,sensor_watermark_epoch,reference_pm,*,model_dir=None):
    issue=int(issue_epoch)
    result={'available':False,'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,
            'experimental':True,'calibrated':False,'forecastIssuedEpoch':issue,'validUntilEpoch':issue+5400,
            'changeThresholdUgM3':THRESHOLD,'thresholdComparison':'at_least',
            'target':'first_sampled_crossing_within90_minutes','direction':'unresolved',
            'directionDecision':'three_class_argmax_ties_unresolved','directionTieTolerance':0.0,
            'horizons':[],'magnitudeEstimates':{},'selectedPolicy':selected_policy(),'displayText':None}
    try:
        ref=float(reference_pm)
        if not math.isfinite(ref) or ref<0: raise ValueError('invalid_reference')
        result['referencePm']=ref
        artifact=_load_artifact(model_dir)
        if artifact is None:
            result.update(reason='background_model_collecting',unavailableReasons=['no_current_cached_estimator'])
            return data_source._qualification(result)
        if artifact['trainingCutoffEpoch']!=midnight_epoch(issue): raise ValueError('background_model_cutoff_mismatch')
        if artifact['metadata']['fittedAtEpoch']>issue: raise ValueError('model_not_yet_fitted_at_issue')
        if artifact['metadata']['maxTrainingOutcomeEpoch']>=artifact['trainingCutoffEpoch']:
            raise ValueError('training_label_cutoff_violation')
        watermark=int(sensor_watermark_epoch)
        if watermark>issue: raise ValueError('invalid_sensor_watermark')
        prepared=sensor.prepare_rows([dict(r) for r in rows if issue-10800<=int(r['epoch'])<=watermark])
        provenance=sensor.feature_metadata(prepared,issue)
        if not provenance.get('available'): raise ValueError(provenance.get('reason','fresh_features_unavailable'))
        features=sensor.issue_features(prepared,issue)
        if not math.isclose(features['freshPm25'],ref,rel_tol=0,abs_tol=1e-7):
            raise ValueError('fresh_sensor_reference_mismatch')
        vector=np.asarray([features[n] for n in artifact['columns']],float)
        p=predict_probabilities(artifact,vector)[0]
        ties=np.flatnonzero(p==p.max())
        direction='drop' if len(ties)==1 and ties[0]==0 else 'rise' if len(ties)==1 and ties[0]==2 else 'unresolved'
        values=[float(v) if math.isfinite(v) else None for v in vector]
        result.update(artifact['metadata'],available=True,reason='available',unavailableReasons=[],
                      direction=direction,modelIdentity=artifact['identity'],
                      probabilityDrop=float(p[0]),probabilityNoCrossing=float(p[1]),probabilityRise=float(p[2]),
                      probabilityAnyCrossing=float(p[0]+p[2]),
                      horizons=[{'leadMinutes':90,'probabilityDrop':float(p[0]),'probabilityNoCrossing':float(p[1]),'probabilityRise':float(p[2])}],
                      featureOriginEpoch=issue//900*900,sensorWatermarkEpoch=watermark,
                      freshReferenceEpoch=provenance['freshReferenceEpoch'],
                      featureSourceMaxEpoch=provenance['featureSourceMaxEpoch'],inputMaxMeasurementEpoch=provenance['featureSourceMaxEpoch'],
                      sourceRole='fresh_sensor_snapshot_for_publication',weatherFeaturesUsed=False,
                      impossibleDropSupport=bool(ref<THRESHOLD and p[0]>0),
                      featureColumns=artifact['columns'],featureValues=values,
                      featureValuesSha256=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest(),
                      freshFeatureProvenance=provenance)
    except Exception as error:
        result.update(available=False,reason='fresh_event_inference_unavailable',unavailableReasons=[type(error).__name__,str(error)])
    return data_source._qualification(result)


def record_issued_prediction(result,*,output_path=None,recorded_epoch=None,input_lineage=None):
    recorded=int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue=int(result['forecastIssuedEpoch'])
    if not 0<=recorded-issue<=120: raise ValueError('Only fresh actual publication may be recorded as originally issued')
    path=Path(output_path) if output_path is not None else PRIVATE/'fresh-event-issued-v2.jsonl'
    identity_hash=hashlib.sha256(json.dumps(result.get('modelIdentity'),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    resolved=str(path.resolve())
    key=(resolved,issue,identity_hash,result.get('featureValuesSha256'),result.get('reason'))
    lineage=dict(input_lineage) if input_lineage is not None else {'status':'not_archived','historicalReceiptTimesUnknown':True}
    record={**result,'recordedEpoch':recorded,'publishedEpoch':recorded,'isOriginallyIssued':True,
            'predictionRole':'genuinely_issued_live_fresh_direct_first20_prediction','inputLineage':lineage,
            'issuedCohort':{'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,
              'modelIdentitySha256':identity_hash,'target':result.get('target'),'changeThresholdUgM3':THRESHOLD,
              'trainingCadence':TRAINING_CADENCE,'issueCadenceRegime':'dense_training_actual_cadence_not_prospectively_validated',
              'issueHourMYT':data_source.datetime.fromtimestamp(issue,data_source.MYT).hour,'issueOffsetSeconds':issue%900}}
    snapshot_id=lineage.get('snapshotId',lineage.get('inputSnapshotId'))
    if snapshot_id is not None: record['inputSnapshotId']=snapshot_id
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        if resolved not in _READ_PATHS:
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-2_000_000))
                    for line in stream.read().splitlines():
                        try:
                            old=json.loads(line)
                            _RECORDED[(resolved,int(old['forecastIssuedEpoch']),old['issuedCohort']['modelIdentitySha256'],old.get('featureValuesSha256'),old.get('reason'))]=None
                        except (ValueError,KeyError,TypeError): pass
            _READ_PATHS.add(resolved)
        if key in _RECORDED: return False
        with path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        _RECORDED[key]=None
        while len(_RECORDED)>4096: _RECORDED.popitem(last=False)
    return True
