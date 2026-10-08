"""One learned five-bin distribution of exact +90 minute arrival change.

Only refresh_model fits. Request inference preserves native categorical mass;
20/40 tails are sums over that same distribution, never separate selectors.
"""
from __future__ import annotations
import hashlib
import json
import math
import pickle
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import arrival_feature_inputs as inputs
import fresh_sensor_features as sensor
from collection_quality import bucket_coverage
from forecast_clock import point_weights

ROOT=Path(__file__).resolve().parent
PRIVATE=ROOT/'private'
MYT=timezone(timedelta(hours=8))
MODEL_VERSION='receipt_visible_hgb_arrival_change20_40_distribution_v1'
ADAPTER_VERSION='direct_arrival_categorical_tail_adapter_v1'
ARTIFACT_NAME='arrival-change20-40-model-v1.pkl'
STATUS_NAME='arrival-change20-40-status-v1.json'
TRAINING_CADENCE='dense_five_minute_variable_offset_arrival_origins_v1'
CLASSES=(0,1,2,3,4)
CLASS_NAMES=('fall40','fall20to40','within20','rise20to40','rise40')
PARAMETERS=dict(max_iter=150,max_leaf_nodes=15,min_samples_leaf=20,
                l2_regularization=1.0,learning_rate=0.05,random_state=20261008,
                early_stopping=False,class_weight=None)
_LOCK=threading.RLock()
_FIT_LOCK=threading.Lock()
_RECORD_LOCK=threading.Lock()
_CACHE={}
_RECORDED=OrderedDict()
_READ_PATHS=set()


def midnight_epoch(issue_epoch):
    return int(datetime.fromtimestamp(int(issue_epoch),MYT).replace(hour=0,minute=0,second=0,microsecond=0).timestamp())


def dense_origins(start,end):
    bases=np.arange(int(start)//300*300,int(end),300,dtype=np.int64)
    epochs=bases+30+((bases//300)%5)*60
    return epochs[(epochs>=start)&(epochs<end)]


def label_classes(delta):
    """Inclusive 20/40 boundaries; finite labels required, no probability rule."""
    values=np.asarray(delta,float)
    if not np.isfinite(values).all(): raise ValueError('nonfinite_arrival_label')
    return np.select([values<=-40,values<=-20,values<20,values<40],[0,1,2,3],default=4).astype(int)


def selected_policy():
    return {'modelVersion':MODEL_VERSION,'parameters':dict(PARAMETERS),
            'target':'exact_issue_plus90_arrival_median_proxy_delta_from_fresh5_reference',
            'arrivalLeadMinutes':90,'changeThresholdsUgM3':[20,40],
            'orderedClassNames':list(CLASS_NAMES),
            'classBoundaries':['delta<=-40','-40<delta<=-20','-20<delta<20','20<=delta<40','delta>=40'],
            'probabilityMethod':'native_categorical_predict_proba_and_disjoint_tail_sums',
            'missingTrainingClass':'native_zero_mass_with_unsupported_class_metadata',
            'trainingCadence':TRAINING_CADENCE,'trainingLookbackDays':28,
            'trainingOriginEmbargoMinutes':120,
            'trainingCutoff':'all_label_support_strictly_before_issue_day_midnight_Asia_Kuala_Lumpur',
            'numericExpectation':'not_identified_by_five_bin_distribution',
            'calibrated':False,'numericalSelectionApplied':False}


def _identity():
    files=[Path(__file__),ROOT/'arrival_feature_inputs.py',ROOT/'fresh_sensor_features.py',
           ROOT/'observation_provenance.py',ROOT/'forecast_clock.py',ROOT/'collection_quality.py']
    return {'sourceSha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'selectedPolicy':selected_policy()}


def prepare_training_data(prepared_inputs,issue_epochs,cutoff,*,lookback_days=28):
    """Causal feature snapshots and only the fully mature +90 arrival target."""
    columns=list(inputs.FEATURE_COLUMNS)
    # The archive retains revisions for earlier issue features. Target medians
    # instead use each observation's latest revision actually visible at cutoff.
    raw_sensor=sensor.prepare_rows(inputs.source_sensor_rows(prepared_inputs))
    epochs,observations,_,_=sensor._bounded(raw_sensor,np.nextafter(float(cutoff),-np.inf),
                                           lookback_minutes=int(lookback_days)*1440+360)
    if not len(epochs):return {'features':np.empty((0,len(columns))),'issues':pd.DataFrame(),'columns':columns}
    raw=pd.DataFrame({'epoch':epochs,'pm02':observations[:,0],'atmp':observations[:,1],'rhum':observations[:,2]})
    raw.index=pd.to_datetime(raw.epoch,unit='s',utc=True)
    raw.pm02=pd.to_numeric(raw.pm02,errors='coerce').where(lambda x:np.isfinite(x)&x.ge(0))
    medians=raw.pm02.resample('900s',label='right',closed='right').median().where(bucket_coverage(raw).forecastEligible)
    buckets={int(t.timestamp()):float(v) for t,v in medians.items() if pd.notna(v)}
    times=np.asarray(issue_epochs,dtype=np.int64)
    times=times[(times>=cutoff-int(lookback_days)*86400)&(times<=cutoff-7200)]
    frame=inputs.issue_feature_frame(prepared_inputs,times)
    values=frame[columns].to_numpy(float)
    provenance=frame.attrs.get('featureProvenance',[])
    if len(provenance)!=len(times): raise ValueError('missing_training_feature_provenance')
    records,vectors,kept_provenance=[],[],[]
    for index,issue in enumerate(times):
        ref=values[index,columns.index('freshPm25')];meta=provenance[index]
        if not math.isfinite(ref) or ref<0 or not meta.get('available'):continue
        origin=int(issue)//900*900
        weights=point_weights(int(issue-origin+5400));labels=[origin+offset*900 for offset in weights]
        complete=max(labels)
        if complete>=cutoff or any(label not in buckets for label in labels):continue
        arrival=sum(buckets[label]*weight for label,weight in zip(labels,weights.values()))
        delta=arrival-ref
        records.append({'issueEpoch':int(issue),'referencePm':float(ref),'arrivalPm':float(arrival),
                        'arrivalDelta':float(delta),'arrivalClass':int(label_classes(delta)),
                        'completeEpoch':int(complete),'issueDay':datetime.fromtimestamp(int(issue),MYT).date().isoformat(),
                        'featureSourceMaxEpoch':meta['featureSourceMaxEpoch']})
        vectors.append(values[index]);kept_provenance.append(meta)
    return {'features':np.asarray(vectors,float).reshape(-1,len(columns)),
            'issues':pd.DataFrame(records),'columns':columns,'featureProvenance':kept_provenance,
            'trainingCadence':TRAINING_CADENCE}


def fit_prepared(prepared,cutoff,*,fitted_epoch=None):
    issues=prepared['issues'];columns=list(prepared['columns']);x=np.asarray(prepared['features'],float)
    if issues.empty or len(issues)<300 or issues.issueDay.nunique()<10:
        raise ValueError('insufficient_completed_training_issues')
    if not issues.completeEpoch.lt(cutoff).all():raise ValueError('training_label_cutoff_violation')
    if not issues.issueEpoch.lt(cutoff).all():raise ValueError('training_origin_cutoff_violation')
    if not issues.issueEpoch.le(cutoff-7200).all():raise ValueError('training_origin_embargo_violation')
    if not issues.featureSourceMaxEpoch.le(issues.issueEpoch).all():raise ValueError('training_feature_clock_violation')
    if x.shape!=(len(issues),len(columns)) or np.isinf(x).any():raise ValueError('invalid_training_feature_shape_or_values')
    y=issues.arrivalClass.to_numpy(int)
    if not set(y).issubset(CLASSES) or len(set(y))<2:raise ValueError('insufficient_observed_arrival_classes')
    if not np.array_equal(y,label_classes(issues.arrivalDelta.to_numpy(float))):raise ValueError('arrival_label_class_mismatch')
    estimator=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                            HistGradientBoostingClassifier(**PARAMETERS))
    with threadpool_limits(limits=1):estimator.fit(x,y)
    observed=tuple(int(c) for c in estimator.classes_)
    if observed!=tuple(sorted(set(y))):raise ValueError('class_order_mismatch')
    external=[i for i,name in enumerate(columns) if name not in inputs.SENSOR_FEATURE_COLUMNS]
    weather=[i for i in external if columns[i] in inputs.WEATHER_FEATURE_COLUMNS]
    metadata={'available':True,'trainingCutoffEpoch':int(cutoff),
              'fittedAtEpoch':int(time.time()) if fitted_epoch is None else int(fitted_epoch),
              'trainingIssueCount':len(issues),'trainingRows':len(issues),
              'trainingDistinctIssueDays':int(issues.issueDay.nunique()),
              'trainingSupport':{name:int((y==c).sum()) for c,name in zip(CLASSES,CLASS_NAMES)},
              'observedTrainingClasses':list(observed),'unsupportedClassIds':[c for c in CLASSES if c not in observed],
              'maxTrainingOutcomeEpoch':int(issues.completeEpoch.max()),
              'trainingMinIssueEpoch':int(issues.issueEpoch.min()),'trainingMaxIssueEpoch':int(issues.issueEpoch.max()),
              'independentIssues':False,'overlappingTrainingIssues':True,
              'historicalReceiptTimesUnknown':True,'calibrated':False,'prospectivelyValidated':False,
              'trainingFitCadence':prepared.get('trainingCadence',TRAINING_CADENCE),
              'trainingOriginEmbargoMinutes':120,
              'featureCount':len(columns),'learnedProbabilityHorizonsMinutes':[90],
              'externalFeatureNonmissingTrainingCounts':{columns[i]:int(np.isfinite(x[:,i]).sum()) for i in external},
              'actualWeatherLearningSupported':bool(weather and any(np.isfinite(x[:,i]).any() for i in weather)),
              'neighborNumericFeaturesUsed':False,
              'objective':'direct_ordered_five_bin_plus90_arrival_change_distribution'}
    return {'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,'identity':_identity(),
            'columns':columns,'trainingCutoffEpoch':int(cutoff),'metadata':metadata,'estimator':estimator}


def predict_probabilities(artifact,features):
    """Return ordered native mass; unobserved estimator classes receive zero."""
    x=np.asarray(features,float)
    if x.ndim==1:x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=len(artifact['columns']) or np.isinf(x).any():raise ValueError('feature_shape_or_values_mismatch')
    with threadpool_limits(limits=1):native=np.asarray(artifact['estimator'].predict_proba(x),float)
    observed=tuple(int(c) for c in artifact['estimator'].classes_)
    if len(set(observed))!=len(observed) or not set(observed).issubset(CLASSES):raise ValueError('invalid_native_class_order')
    if native.shape!=(len(x),len(observed)) or not np.isfinite(native).all() or (native<0).any() or (native>1).any() or not np.allclose(native.sum(axis=1),1,rtol=1e-10,atol=1e-12):
        raise ValueError('invalid_learned_probability_distribution')
    ordered=np.zeros((len(x),5),float)
    ordered[:,list(observed)]=native
    return ordered


def tail_probabilities(mass):
    """CDF/tails of one native ordered distribution, without clipping or masks."""
    p=np.asarray(mass,float)
    if p.shape!=(5,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.isclose(p.sum(),1,rtol=1e-10,atol=1e-12):
        raise ValueError('invalid_arrival_mass')
    return {'probabilityFall20':float(p[0]+p[1]),'probabilityFall40':float(p[0]),
            'probabilityRise20':float(p[3]+p[4]),'probabilityRise40':float(p[4]),
            'probabilityWithin20':float(p[2]),'probabilityWithin40':float(p[1]+p[2]+p[3])}


def _paths(model_dir=None):
    directory=Path(model_dir) if model_dir is not None else PRIVATE
    return directory/ARTIFACT_NAME,directory/STATUS_NAME


def _atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending');temporary.write_bytes(data);temporary.replace(path)


def _load_artifact(model_dir=None):
    path,_=_paths(model_dir)
    if not path.exists():return None
    signature=(str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
    with _LOCK:
        if signature not in _CACHE:
            artifact=pickle.loads(path.read_bytes())
            if artifact.get('modelVersion')!=MODEL_VERSION or artifact.get('identity')!=_identity():return None
            _CACHE.clear();_CACHE[signature]=artifact
        return _CACHE[signature]


def refresh_model(db_path,issue_epoch,*,neighbor_db_path=None,model_dir=None):
    """The sole background-fitting entry point; never called by inference."""
    cutoff=midnight_epoch(issue_epoch)
    with _FIT_LOCK:
        try:
            prior=_load_artifact(model_dir)
            if prior is not None and prior['trainingCutoffEpoch']==cutoff:return {**prior['metadata'],'reason':'cached_fit_current'}
            source=inputs.load_inputs(db_path,cutoff-28*86400-10800,cutoff,neighbor_db_path=neighbor_db_path)
            prepared=prepare_training_data(source,dense_origins(cutoff-28*86400,cutoff-7200),cutoff)
            artifact=fit_prepared(prepared,cutoff)
            path,status_path=_paths(model_dir);_atomic(path,pickle.dumps(artifact,protocol=pickle.HIGHEST_PROTOCOL))
            status={**artifact['metadata'],'modelVersion':MODEL_VERSION,'reason':'available','identity':artifact['identity']}
            _atomic(status_path,json.dumps(status,indent=2,allow_nan=False).encode('utf-8'))
            return status
        except Exception as error:
            return {'available':False,'modelVersion':MODEL_VERSION,'trainingCutoffEpoch':cutoff,
                    'reason':'background_fit_unavailable','unavailableReasons':[type(error).__name__,str(error)]}


def predict_arrival(db_path,rows,issue_epoch,sensor_watermark_epoch,reference_pm,*,
                    source_weather_rows=(),neighbor_rows=(),neighbor_metadata=(),feature_inputs=None,model_dir=None):
    """Infer one cache on issue-visible inputs; db_path is deliberately unused."""
    issue=int(issue_epoch)
    result={'available':False,'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,
            'forecastIssuedEpoch':issue,'arrivalEpoch':issue+5400,'validUntilEpoch':issue+5400,
            'target':'exact_issue_plus90_arrival_median_proxy_delta_from_fresh5_reference',
            'arrivalLeadMinutes':90,'changeThresholdsUgM3':[20,40],
            'orderedClassNames':list(CLASS_NAMES),'selectedPolicy':selected_policy(),
            'prospectivelyValidated':False,'calibrated':False,'numericalSelectionApplied':False,
            'arrivalConcentrationExpectation':None,'numericExpectationAvailable':False,
            'numericExpectationUnavailableReason':'categorical_bins_do_not_identify_concentration_expectation'}
    try:
        ref=float(reference_pm)
        if not math.isfinite(ref) or ref<0:raise ValueError('invalid_reference')
        result['referencePm']=ref
        artifact=_load_artifact(model_dir)
        if artifact is None:
            result.update(reason='background_model_collecting',unavailableReasons=['no_current_cached_estimator']);return result
        if artifact['trainingCutoffEpoch']!=midnight_epoch(issue):raise ValueError('background_model_cutoff_mismatch')
        if artifact['metadata']['fittedAtEpoch']>issue:raise ValueError('model_not_yet_fitted_at_issue')
        if artifact['metadata']['maxTrainingOutcomeEpoch']>=artifact['trainingCutoffEpoch']:raise ValueError('training_label_cutoff_violation')
        watermark=int(sensor_watermark_epoch)
        if watermark>issue:raise ValueError('invalid_sensor_watermark')
        if feature_inputs is None:
            bounded=[dict(r) for r in rows if issue-10800<=int(r['epoch'])<=watermark]
            feature_inputs=inputs.prepare_inputs(bounded,source_weather_rows,neighbor_rows,neighbor_metadata)
        provenance=inputs.feature_metadata(feature_inputs,issue)
        if not provenance.get('available'):raise ValueError(provenance.get('reason','fresh_features_unavailable'))
        if provenance['featureSourceMaxEpoch']>watermark:raise ValueError('feature_sensor_watermark_violation')
        features=inputs.issue_features(feature_inputs,issue)
        if not math.isclose(features['freshPm25'],ref,rel_tol=0,abs_tol=1e-7):raise ValueError('fresh_sensor_reference_mismatch')
        vector=np.asarray([features[n] for n in artifact['columns']],float)
        mass=predict_probabilities(artifact,vector)[0]
        finite_values=[float(v) if math.isfinite(v) else None for v in vector]
        result.update(artifact['metadata'],available=True,reason='available',unavailableReasons=[],
                      modelIdentity=artifact['identity'],classProbabilities=[float(p) for p in mass],
                      **tail_probabilities(mass),sensorWatermarkEpoch=watermark,
                      freshReferenceEpoch=provenance['freshReferenceEpoch'],
                      featureSourceMaxEpoch=provenance['featureSourceMaxEpoch'],
                      featureColumns=artifact['columns'],featureValues=finite_values,
                      featureValuesSha256=hashlib.sha256(json.dumps(finite_values,separators=(',',':')).encode()).hexdigest(),
                      featureProvenance=provenance,
                      probabilityThresholds=[{'changeThresholdUgM3':20,'probabilityFall':float(mass[0]+mass[1]),'probabilityRise':float(mass[3]+mass[4]),'probabilityNeither':float(mass[2])},
                                            {'changeThresholdUgM3':40,'probabilityFall':float(mass[0]),'probabilityRise':float(mass[4]),'probabilityNeither':float(mass[1]+mass[2]+mass[3])}])
    except Exception as error:
        result.update(available=False,reason='arrival_distribution_inference_unavailable',unavailableReasons=[type(error).__name__,str(error)])
    return result


def record_issued_prediction(result,*,output_path=None,recorded_epoch=None,input_lineage=None):
    """Archive actual publication only, with the fixed absolute arrival clock."""
    recorded=int(time.time()) if recorded_epoch is None else int(recorded_epoch);issue=int(result['forecastIssuedEpoch'])
    if not 0<=recorded-issue<=120:raise ValueError('Only fresh actual publication may be recorded as originally issued')
    path=Path(output_path) if output_path is not None else PRIVATE/'arrival-change-issued-v1.jsonl'
    identity_hash=hashlib.sha256(json.dumps(result.get('modelIdentity'),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    resolved=str(path.resolve());key=(resolved,issue,identity_hash,result.get('featureValuesSha256'),result.get('reason'))
    lineage=dict(input_lineage) if input_lineage is not None else {'status':'not_archived','historicalReceiptTimesUnknown':True}
    record={**result,'recordedEpoch':recorded,'publishedEpoch':recorded,'isOriginallyIssued':True,
            'predictionRole':'genuinely_issued_arrival_change_distribution','inputLineage':lineage,
            'issuedCohort':{'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,
                           'modelIdentitySha256':identity_hash,'target':result['target'],
                           'arrivalLeadMinutes':90,'changeThresholdsUgM3':[20,40],
                           'trainingCadence':result.get('trainingFitCadence',TRAINING_CADENCE)}}
    if 'snapshotId' in lineage:record['inputSnapshotId']=lineage['snapshotId']
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        if resolved not in _READ_PATHS:
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:
                        old=json.loads(line)
                        old_key=(resolved,old['forecastIssuedEpoch'],old['issuedCohort']['modelIdentitySha256'],old.get('featureValuesSha256'),old.get('reason'))
                        _RECORDED[old_key]=None
                    except (ValueError,KeyError):continue
            _READ_PATHS.add(resolved)
        if key in _RECORDED:return False
        with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        _RECORDED[key]=None
        while len(_RECORDED)>20000:_RECORDED.popitem(last=False)
    return True
