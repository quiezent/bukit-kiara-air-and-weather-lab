"""Fresh sensor ExtraTrees competing-risk inference and background-only fitting.

Targets retain the six exact-clock >=20 first median-proxy crossing semantics.
Only refresh_model may fit; HTTP prediction reads one completed cached artifact.
"""
from __future__ import annotations
import hashlib
import json
import math
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import fresh_sensor_features as sensor
from collection_quality import bucket_coverage
from forecast_clock import point_weights

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / 'private'
MODEL_VERSION = 'fresh_sensor_extra_trees_first20_competing_risk_v1'
ADAPTER_VERSION = 'fresh_sensor_extra_trees_event_adapter_v1'
ARTIFACT_NAME = 'fresh-sensor-first20-model-v1.pkl'
STATUS_NAME = 'fresh-sensor-first20-status-v1.json'
TRAINING_CADENCE = 'dense_five_minute_variable_offset_sensor_origins_v1'
THRESHOLD = 20.0
LEADS = (15, 30, 45, 60, 75, 90)
CLASSES = (-1, 0, 1)
TREE_PARAMETERS = dict(n_estimators=128, max_depth=12, min_samples_leaf=10,
                       max_features='sqrt', random_state=20261008, n_jobs=1,
                       class_weight=None)
TRAIN_DAYS = 28
MYT = timezone(timedelta(hours=8))
_CACHE = {}
_LOCK = threading.RLock()
_FIT_LOCK = threading.Lock()
_RECORD_LOCK = threading.Lock()
_RECORDED = OrderedDict()
_READ_PUBLICATION_PATHS = set()


def midnight_epoch(issue_epoch):
    return int(datetime.fromtimestamp(int(issue_epoch), MYT).replace(hour=0, minute=0,
                second=0, microsecond=0).timestamp())


def selected_policy():
    return {'modelVersion': MODEL_VERSION, 'parameters': dict(TREE_PARAMETERS),
            'changeThresholdUgM3': THRESHOLD, 'thresholdComparison': 'at_least',
            'target': 'first_sampled_crossing_within90_minutes',
            'sampleLeadsMinutes': list(LEADS), 'trainingCadence': TRAINING_CADENCE,
            'trainingLookbackDays': TRAIN_DAYS,
            'trainingCutoff': 'strict_completed_labels_before_issue_day_midnight_Asia_Kuala_Lumpur',
            'probabilityMethod': 'learned_conditional_hazards_absorbing_cumulative_incidence',
            'directionDecision': 'three_class_argmax_ties_unresolved',
            'calibrated': False, 'numericalSelectionApplied': False}


def _qualification(result):
    """Keep learned diagnostic outputs without claiming prospective skill."""
    result.update(qualificationVersion='fresh_actual_cadence_event_qualification_v1',
                  prospectivelyValidated=False, trainingFitCadence=TRAINING_CADENCE,
                  trainingCadenceMatched=False, unvalidatedLiveCadence=True,
                  liveIssueOffsetSeconds=int(result['forecastIssuedEpoch'])%900,
                  probabilityRole='uncalibrated_experimental_diagnostic',
                  operationalDecisionRule='no_direction_until_actual_cadence_prospective_qualification',
                  directionStatus='unqualified_experimental',
                  qualification={'state':'unqualified_experimental','eligible':False,
                    'operationalUseEligible':False,'actualCadenceValidated':False,
                    'calibrationValidated':False,'trainingCadenceMatched':False,
                    'reasons':['no_actual_cadence_prospective_qualification',
                               'historical_input_receipt_and_revision_history_incomplete',
                               'dense_overlapping_training_issues_are_not_independent_events'],
                    'evidenceRole':'chronological_development_and_originally_issued_diagnostics'})
    result['diagnosticDirection'] = result.get('direction','unresolved')
    result['diagnosticDirectionAvailable'] = bool(result.get('available'))
    result['direction'] = 'unresolved'
    return result


def _identity():
    files = [Path(__file__), ROOT/'fresh_sensor_features.py', ROOT/'forecast_clock.py',
             ROOT/'collection_quality.py']
    return {'sourceSha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'selectedPolicy': selected_policy()}


def dense_origins(start, end):
    """Fixed 1--6 minute gaps cover all minute offsets without target peeking."""
    bases = np.arange(int(start)//300*300, int(end), 300, dtype=np.int64)
    epochs = bases + 30 + ((bases//300) % 5)*60
    return epochs[(epochs >= start) & (epochs < end)]


def read_training_rows(db_path, cutoff):
    """Never read any source observations at or after the fit's cutoff."""
    uri = Path(db_path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            'SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<? ORDER BY epoch',
            (int(cutoff - TRAIN_DAYS*86400 - 10800), int(cutoff)))]
    return rows


def prepare_training_data(rows, issue_epochs, cutoff, *, lookback_days=TRAIN_DAYS):
    """Causal features plus fully mature labels; no estimator invocation."""
    bounded = [dict(r) for r in rows if int(r['epoch']) < int(cutoff)]
    if not bounded:
        return {'features': np.empty((0, len(sensor.FEATURE_COLUMNS))),
                'issues': pd.DataFrame(), 'columns': list(sensor.FEATURE_COLUMNS)}
    raw = pd.DataFrame(bounded)
    raw.index = pd.to_datetime(raw.epoch, unit='s', utc=True)
    raw.pm02 = pd.to_numeric(raw.pm02, errors='coerce')
    raw.pm02 = raw.pm02.where(raw.pm02.ge(0) & np.isfinite(raw.pm02))
    medians = raw.pm02.resample('900s', label='right', closed='right').median()
    coverage = bucket_coverage(raw)
    medians = medians.where(coverage.forecastEligible)
    buckets = {int(t.timestamp()): float(v) for t,v in medians.items() if pd.notna(v)}
    times = np.asarray(issue_epochs, dtype=np.int64)
    times = times[(times >= cutoff-int(lookback_days)*86400) & (times < cutoff)]
    features = sensor.issue_feature_frame(bounded, times)
    columns = list(sensor.FEATURE_COLUMNS)
    feature_array = features[columns].to_numpy(float)
    provenance = features.attrs.get('featureProvenance', [])
    records, vectors = [], []
    for index, issue in enumerate(times):
        ref = feature_array[index, columns.index('freshPm25')]
        if not math.isfinite(ref) or ref < 0:
            continue
        if provenance and not provenance[index].get('available', False):
            continue
        origin = int(issue)//900*900
        values, completions = [], []
        for lead in LEADS:
            weights = point_weights(int(issue-origin+lead*60))
            labels = [origin+offset*900 for offset in weights]
            complete = max(labels)
            completions.append(complete)
            if complete >= cutoff or any(label not in buckets for label in labels):
                values.append(math.nan)
            else:
                values.append(sum(buckets[label]*weight for label,weight in zip(labels, weights.values())))
        if not np.isfinite(values).all():
            continue
        delta = np.asarray(values)-ref
        crossed = np.flatnonzero(np.abs(delta)>=THRESHOLD)
        first = int(crossed[0]) if len(crossed) else -1
        cause = int(np.sign(delta[first])) if first >= 0 else 0
        records.append({'issueEpoch': int(issue), 'referencePm': float(ref),
                        'completeEpoch': int(max(completions)), 'cause': cause,
                        'firstStep': first, 'issueDay': datetime.fromtimestamp(int(issue), MYT).date().isoformat(),
                        'featureSourceMaxEpoch': (provenance[index].get('featureSourceMaxEpoch') if provenance else None),
                        **{'outcome'+str(lead):float(v) for lead,v in zip(LEADS,values)}})
        vectors.append(feature_array[index])
    return {'features': np.asarray(vectors, dtype=float).reshape(-1,len(columns)),
            'issues': pd.DataFrame(records), 'columns': columns}


def risk_arrays(prepared):
    """At-risk rows are correlated steps, never independent issue counts."""
    x, y = [], []
    for index, row in enumerate(prepared['issues'].itertuples(index=False)):
        for step in range(6 if row.firstStep < 0 else row.firstStep+1):
            x.append(np.r_[prepared['features'][index], np.eye(6)[step]])
            y.append(row.cause if step == row.firstStep else 0)
    return np.asarray(x, float), np.asarray(y, int)


def fit_prepared(prepared, cutoff, *, fitted_epoch=None):
    """One preregistered estimator; scores never select a replacement."""
    issues = prepared['issues']
    if issues.empty or len(issues)<300 or issues.issueDay.nunique()<10:
        raise ValueError('insufficient_completed_training_issues')
    if not issues.completeEpoch.lt(cutoff).all():
        raise ValueError('training_label_cutoff_violation')
    x,y = risk_arrays(prepared)
    if set(y) != set(CLASSES):
        raise ValueError('all_three_risk_classes_required')
    estimator = make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                            keep_empty_features=True),
                              ExtraTreesClassifier(**TREE_PARAMETERS))
    with threadpool_limits(limits=1):
        estimator.fit(x,y)
    if tuple(estimator.classes_) != CLASSES:
        raise ValueError('class_order_mismatch')
    support = {name:int((issues.cause==cause).sum()) for cause,name in zip(CLASSES,('drop','none','rise'))}
    metadata = {'available':True, 'trainingCutoffEpoch':int(cutoff),
                'fittedAtEpoch':int(time.time()) if fitted_epoch is None else int(fitted_epoch),
                'trainingIssueCount':len(issues), 'riskRowCount':len(y),
                'trainingSupport':support, 'trainingDistinctIssueDays':int(issues.issueDay.nunique()),
                'maxTrainingOutcomeEpoch':int(issues.completeEpoch.max()),
                'trainingMinIssueEpoch':int(issues.issueEpoch.min()),
                'trainingMaxIssueEpoch':int(issues.issueEpoch.max()),
                'independentIssues':False, 'overlappingTrainingIssues':True,
                'historicalReceiptTimesUnknown':True, 'calibrated':False,
                'trainingFitCadence':TRAINING_CADENCE,
                'weatherFeaturesUsed':False, 'featureCount':len(prepared['columns'])}
    return {'modelVersion':MODEL_VERSION, 'adapterVersion':ADAPTER_VERSION,
            'identity':_identity(), 'columns':list(prepared['columns']),
            'trainingCutoffEpoch':int(cutoff), 'metadata':metadata, 'estimator':estimator}


def cumulative_incidence(hazards):
    h = np.asarray(hazards,float)
    if h.shape != (6,3) or not np.isfinite(h).all() or (h<0).any() or not np.allclose(h.sum(axis=1),1):
        raise ValueError('invalid_hazard_distribution')
    survival, drop, rise = 1.0, 0.0, 0.0
    result = []
    for row in h:
        drop += survival*row[0]
        rise += survival*row[2]
        survival *= row[1]
        result.append([drop,survival,rise])
    return np.asarray(result)


def predict_probabilities(artifact, features):
    feature_array = np.asarray(features,float)
    if feature_array.shape != (len(artifact['columns']),):
        raise ValueError('feature_shape_mismatch')
    steps = np.asarray([np.r_[feature_array,np.eye(6)[step]] for step in range(6)])
    with threadpool_limits(limits=1):
        hazards = artifact['estimator'].predict_proba(steps)
    return cumulative_incidence(hazards)


def _paths(model_dir):
    directory = Path(model_dir) if model_dir is not None else PRIVATE
    return directory/ARTIFACT_NAME, directory/STATUS_NAME


def _atomic(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    temporary.write_bytes(data)
    temporary.replace(path)


def _load_artifact(model_dir=None):
    path,_ = _paths(model_dir)
    if not path.exists(): return None
    signature = (str(path.resolve()),path.stat().st_mtime_ns,path.stat().st_size)
    with _LOCK:
        if signature not in _CACHE:
            artifact = pickle.loads(path.read_bytes())
            if artifact.get('modelVersion') != MODEL_VERSION or artifact.get('identity') != _identity():
                return None
            _CACHE.clear(); _CACHE[signature] = artifact
        return _CACHE[signature]


def refresh_model(db_path, issue_epoch, *, model_dir=None):
    """Background-only fit with a strict midnight maturity purge."""
    cutoff = midnight_epoch(issue_epoch)
    with _FIT_LOCK:
        prior = _load_artifact(model_dir)
        if prior is not None and prior['trainingCutoffEpoch']==cutoff:
            return {**prior['metadata'],'reason':'cached_fit_current'}
        try:
            rows = read_training_rows(db_path,cutoff)
            epochs = dense_origins(cutoff-TRAIN_DAYS*86400,cutoff)
            prepared = prepare_training_data(rows,epochs,cutoff)
            artifact = fit_prepared(prepared,cutoff)
            path,status_path = _paths(model_dir)
            _atomic(path,pickle.dumps(artifact,protocol=pickle.HIGHEST_PROTOCOL))
            status = {**artifact['metadata'],'modelVersion':MODEL_VERSION,'reason':'available',
                      'identity':artifact['identity']}
            _atomic(status_path,json.dumps(status,indent=2,allow_nan=False).encode('utf-8'))
            return status
        except Exception as error:
            return {'available':False,'modelVersion':MODEL_VERSION,'trainingCutoffEpoch':cutoff,
                    'reason':'background_fit_unavailable','unavailableReasons':[type(error).__name__,str(error)]}


def predict_event(db_path, rows, issue_epoch, sensor_watermark_epoch, reference_pm, *, model_dir=None):
    """Literal model inference; never train or fetch external data here."""
    issue = int(issue_epoch)
    result = {'available':False,'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,
              'experimental':True,'calibrated':False,'forecastIssuedEpoch':issue,
              'validUntilEpoch':issue+5400,'changeThresholdUgM3':THRESHOLD,
              'thresholdComparison':'at_least','target':'first_sampled_crossing_within90_minutes',
              'direction':'unresolved','directionDecision':'three_class_argmax_ties_unresolved',
              'directionTieTolerance':0.0,'horizons':[], 'magnitudeEstimates':{},
              'selectedPolicy':selected_policy(), 'displayText':None}
    try:
        reference = float(reference_pm)
        if not math.isfinite(reference) or reference<0:
            raise ValueError('invalid_reference')
        result['referencePm'] = reference
        artifact = _load_artifact(model_dir)
        if artifact is None:
            result.update(reason='background_model_collecting',unavailableReasons=['no_current_cached_estimator'])
            return _qualification(result)
        if artifact['trainingCutoffEpoch'] != midnight_epoch(issue):
            raise ValueError('background_model_cutoff_mismatch')
        if artifact['metadata']['fittedAtEpoch'] > issue:
            raise ValueError('model_not_yet_fitted_at_issue')
        if artifact['metadata']['maxTrainingOutcomeEpoch'] >= artifact['trainingCutoffEpoch']:
            raise ValueError('training_label_cutoff_violation')
        watermark = int(sensor_watermark_epoch)
        if watermark > issue:
            raise ValueError('invalid_sensor_watermark')
        bounded = [dict(r) for r in rows if issue-10800 <= int(r['epoch']) <= watermark]
        prepared = sensor.prepare_rows(bounded)
        provenance = sensor.feature_metadata(prepared,issue)
        if not provenance.get('available',False):
            raise ValueError(provenance.get('reason','fresh_sensor_features_unavailable'))
        features = sensor.issue_features(prepared,issue)
        if not math.isclose(float(features['freshPm25']),reference,rel_tol=0,abs_tol=1e-7):
            raise ValueError('fresh_sensor_reference_mismatch')
        vector = np.asarray([features[name] for name in artifact['columns']],float)
        probabilities = predict_probabilities(artifact,vector)
        final = probabilities[-1]
        ties = np.flatnonzero(final==final.max())
        direction = 'drop' if len(ties)==1 and ties[0]==0 else 'rise' if len(ties)==1 and ties[0]==2 else 'unresolved'
        normalized = [float(v) if math.isfinite(v) else None for v in vector]
        result.update(artifact['metadata'],available=True,reason='available',unavailableReasons=[],
                      direction=direction,modelIdentity=artifact['identity'],
                      probabilityDrop=float(final[0]),probabilityNoCrossing=float(final[1]),probabilityRise=float(final[2]),
                      probabilityAnyCrossing=float(final[0]+final[2]),
                      horizons=[{'leadMinutes':lead,'probabilityDrop':float(v[0]),'probabilityNoCrossing':float(v[1]),'probabilityRise':float(v[2])} for lead,v in zip(LEADS,probabilities)],
                      featureOriginEpoch=issue//900*900,sensorWatermarkEpoch=watermark,
                      freshReferenceEpoch=provenance.get('freshReferenceEpoch'),
                      featureSourceMaxEpoch=provenance.get('featureSourceMaxEpoch'),inputMaxMeasurementEpoch=provenance.get('featureSourceMaxEpoch'),
                      sourceRole='fresh_sensor_snapshot_for_publication',weatherFeaturesUsed=False,
                      impossibleDropSupport=bool(reference<THRESHOLD and final[0]>0),
                      featureColumns=artifact['columns'],featureValues=normalized,
                      featureValuesSha256=hashlib.sha256(json.dumps(normalized,separators=(',',':')).encode()).hexdigest(),
                      freshFeatureProvenance=provenance)
    except Exception as error:
        result.update(available=False,reason='fresh_event_inference_unavailable',unavailableReasons=[type(error).__name__,str(error)])
    return _qualification(result)


def record_issued_prediction(result, *, output_path=None, recorded_epoch=None, input_lineage=None):
    """Archive actual newly published forecasts, never offline counterfactuals."""
    recorded = int(time.time()) if recorded_epoch is None else int(recorded_epoch)
    issue = int(result['forecastIssuedEpoch'])
    if not 0 <= recorded-issue <= 120:
        raise ValueError('Only fresh actual publication may be recorded as originally issued')
    path = Path(output_path) if output_path is not None else PRIVATE/'fresh-event-issued-v1.jsonl'
    identity_hash = hashlib.sha256(json.dumps(result.get('modelIdentity'),sort_keys=True,
                      separators=(',',':'),allow_nan=False).encode()).hexdigest()
    key = (str(path.resolve()),issue,identity_hash,result.get('featureValuesSha256'),result.get('reason'))
    lineage = dict(input_lineage) if input_lineage is not None else {
        'status':'not_archived','historicalReceiptTimesUnknown':True}
    record = {**result,'recordedEpoch':recorded,'publishedEpoch':recorded,'isOriginallyIssued':True,
              'predictionRole':'genuinely_issued_live_fresh_first20_prediction', 'inputLineage':lineage,
              'issuedCohort':{'modelVersion':MODEL_VERSION,'adapterVersion':ADAPTER_VERSION,
                'modelIdentitySha256':identity_hash,'target':result.get('target'),
                'changeThresholdUgM3':THRESHOLD,'trainingCadence':TRAINING_CADENCE,
                'issueCadenceRegime':'dense_training_actual_cadence_not_prospectively_validated',
                'issueHourMYT':datetime.fromtimestamp(issue,MYT).hour,'issueOffsetSeconds':issue%900}}
    snapshot_id = lineage.get('snapshotId',lineage.get('inputSnapshotId'))
    if snapshot_id is not None: record['inputSnapshotId'] = snapshot_id
    with _RECORD_LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        resolved = str(path.resolve())
        if resolved not in _READ_PUBLICATION_PATHS:
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-2_000_000))
                    for line in stream.read().splitlines():
                        try:
                            old = json.loads(line)
                            old_key = (resolved,int(old['forecastIssuedEpoch']),
                                old['issuedCohort']['modelIdentitySha256'],old.get('featureValuesSha256'),old.get('reason'))
                            _RECORDED[old_key] = None
                        except (ValueError,KeyError,TypeError): pass
            _READ_PUBLICATION_PATHS.add(resolved)
        if key in _RECORDED: return False
        with path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        _RECORDED[key] = None
        while len(_RECORDED)>4096: _RECORDED.popitem(last=False)
    return True
