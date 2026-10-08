"""One frozen same-day Afternoon challenger; fitting is background-only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import daily_cache as cache

HERE = Path(__file__).resolve().parent
ROOT = cache.original.ROOT
POLICY = HERE / 'AFTERNOON_BLEND_POLICY.json'
FREEZE = HERE / 'AFTERNOON_BLEND_FREEZE.json'
SVR_SOURCE = ROOT / 'research/svr_context_20261008/svr_trainer.py'
spec = importlib.util.spec_from_file_location('forward_afternoon_context_svr', SVR_SOURCE)
svr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svr)
PARAMETERS = dict(max_iter=100, max_depth=3, max_leaf_nodes=7,
                  min_samples_leaf=80, l2_regularization=50., learning_rate=.05,
                  loss='absolute_error', early_stopping=False, random_state=1749)
_LOCK = threading.RLock()
_MODELS = {}


def _sources():
    files = [Path(__file__), HERE / 'daily_cache.py', HERE.parent / 'prospective_worker.py',
             SVR_SOURCE, svr.ORIGINAL, ROOT / 'weather_session_model.py',
             HERE / 'DAILY_POLICY.json', HERE / 'DAILY_FREEZE.json',
             HERE / 'DAILY_AFTERNOON_BLEND_REVISION.json',
             HERE / 'DAILY_AFTERNOON_BLEND_REVISION_FREEZE.json',
             HERE / 'DAILY_AFTERNOON_RAW_RETRY_REVISION.json',
             HERE / 'DAILY_AFTERNOON_RAW_RETRY_REVISION_FREEZE.json',
             ROOT / 'research/afternoon_svr_hgb_mix_20261008/POLICY.json',
             ROOT / 'research/afternoon_svr_hgb_mix_20261008/FREEZE.json']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): cache.original.sha(p) for p in files}


def freeze_policy():
    cache.verify_policy()
    if POLICY.exists():
        return verify_policy()
    policy = {
        'version': 'same_day_afternoon_equal_svr_gate20_hgb_mean_raw31_v1',
        'frozenAtEpoch': time.time(), 'head': 'same-day Afternoon only',
        'canonical': {'clockMYT': '12:00', 'captureGraceSeconds': 120,
                      'target': 'Today14:00-16:00 exact dated two-hour mean',
                      'selection': 'First genuinely received valid snapshot in[12:00,12:02]; never backfill or replace'},
        'training': {'lookbackDays': 14, 'cutoff': 'MYTmidnight; issue and label completion strictly before cutoff',
                     'minimumRows': 120, 'minimumDistinctTargetDates': 10,
                     'membership': 'Same completed pooled seven-lead prefix/prospective rows for BOTH components. Exclude prospective rows lacking genuinely captured raw float64 basic31, disclose count; original Patch fit remains unchanged.',
                     'precision': 'Certified prefix basic31 float64 from original metadata; newly captured basic31 float64 from sessions.query_features against the SAME materialized snapshot. Never promote old float32 inputs as exact raw64.'},
        'svr': svr.specification(),
        'hgb': {'parameters': PARAMETERS, 'features': 'Unscaled original basic31 float64, same order as original context',
                'imputer': {'strategy': 'mean', 'add_indicator': True, 'keep_empty_features': True},
                'target': 'actual-fresh', 'output': 'max(0,fresh+predicted signed change)', 'threads': 1},
        'output': '0.5*(fresh if abs(SVRraw-fresh)<20 else SVRraw)+0.5*HGBraw; NO final deadband',
        'sourceClocks': 'Actual API receipt and read-only DB snapshot receipt are recorded separately. Sources fetched<=DB snapshot open<=DB receipt<=issue. Sensor<=original API watermark<=publication<=API receipt<=issue.',
        'legacyReceiptLimitation': 'Historical weather/CAMS fetched_epoch is available; their independent original HTTP/archive receipt is not. Current API and DB snapshot receipt does not invent those legacy clocks.',
        'freshness': {'sensorMaximumAgeSeconds': 240, 'apiPublicationMaximumAgeSeconds': 120,
                      'weatherMaximumAgeSeconds': 7200, 'artifact': 'State AND readiness JSON must physically exist before actual candidate issue'},
        'inputs': 'Label-free compact original tensors plus raw float64 context; actual issue/target/source clocks and genuine issued incumbent exact value/reference retained',
        'protectedLabels': 'Never label/read/score Oct20-Nov2 target outcomes; reuse existing exclusion before label DB access',
        'promotion': {'newDistinctTargetDates': 10, 'independentRises': 5, 'independentDrops': 5,
                      'minimumMAEGainOverIssuedIncumbentAndPersistence': .5,
                      'maximumP90Increase': .5, 'maximumQuietMAEIncrease': 1.,
                      'direction': 'Improved sign-correct>=10 recall, no worse wrongway/false-clearing; count distinct events',
                      'oneChallengerPerHead': True, 'automaticPromotion': False},
        'exposure': 'Sep18-Oct7 development outcomes already inspected; original SVR nextAfternoon primary untouched. No fresh qualification from development or current noncanonical runtime diagnostic.',
        'deployment': 'Genuine forward shadow only. Offline CPU1 daily fit in existing independent service worker; no HTTP fitting or selected dashboard numerical/UI changes.',
        'diagnostic': 'One genuine current snapshot may be saved as explicitly NONCANONICAL software diagnostic; it never enters canonical qualification',
        'featureOrder': list(cache.original.prepare.CONTEXT), 'sklearnVersion': sklearn.__version__,
    }
    cache.original.write(POLICY, policy)
    cache.original.write(FREEZE, {'policySha256': cache.original.sha(POLICY), 'sources': _sources(),
                                'frozenBeforeNewPredictions': True})
    return policy


def verify_policy():
    cache.verify_policy()
    frozen = cache._json(FREEZE)
    if frozen['policySha256'] != cache.original.sha(POLICY) or frozen['sources'] != _sources():
        raise RuntimeError('Frozen Afternoon blend source/policy changed')
    policy = cache._json(POLICY)
    if policy['sklearnVersion'] != sklearn.__version__ or len(policy['featureOrder']) != 31:
        raise RuntimeError('Frozen Afternoon blend environment/feature order differs')
    return policy


def raw_context(snapshot, start, end):
    """Recompute raw31 using the same received sources; no label/API reads."""
    issue = int(snapshot['issueEpoch'])
    source = cache.original.prepare.Sources(snapshot)
    raw = [(f, p) for f, _, p in snapshot['cams']]
    digest = hashlib.sha256(json.dumps(raw).encode()).hexdigest()
    cams = cache.original.regional._read_runs(ROOT / 'bukit_kiara_air_history.db',
                                            'air_quality_forecast_runs', issue, raw_snapshot=(raw, digest))
    values = cache.original.sessions.query_features(
        {'frame': source.frame, 'rows': snapshot['sensor'], 'weather': source.archive,
         'cams': cams, 'cutoff': issue}, issue, int(start), int(end))
    if values is None or not values.attrs['valid']:
        raise RuntimeError('Genuine raw float64 basic31 unavailable')
    array = values.reindex(cache.original.prepare.CONTEXT).to_numpy(dtype=np.float64)
    array[~np.isfinite(array)] = np.nan
    if array.shape != (31,) or not np.isfinite(array[0]):
        raise RuntimeError('Invalid raw float64 reference/context shape')
    return array


def _paths(cutoff):
    path = cache.CACHE / 'models' / f'{cutoff}_afternoon_blend_r2.joblib'
    return path, path.with_suffix('.json')


def _load(cutoff):
    verify_policy()
    path, info_path = _paths(cutoff)
    if not path.exists() or not info_path.exists():
        raise RuntimeError('Afternoon background artifact not ready; prediction never fits')
    info = cache._json(info_path)
    if (info['policySha256'] != cache.original.sha(POLICY) or info['sourceHashes'] != _sources() or
            cache.original.sha(path) != info['checkpointSha256'] or
            int(info['trainingCutoffEpoch']) != int(cutoff)):
        raise RuntimeError('Afternoon completed artifact/source differs')
    if cutoff not in _MODELS:
        saved = joblib.load(path)
        if saved['sourceHashes'] != _sources() or saved['sklearnVersion'] != sklearn.__version__:
            raise RuntimeError('Afternoon saved learner source/version differs')
        s = svr.ContextSVR()
        s.model = saved['svr']['model']; s.normalizer = saved['svr']['normalizer']
        s.training_metadata = saved['svr']['training_metadata']
        _MODELS.clear(); _MODELS[cutoff] = (s, saved['hgb'])
    info = dict(info, physicallyReadyEpoch=max(path.stat().st_mtime, info_path.stat().st_mtime,
                                              info['artifactReadyEpoch']))
    return _MODELS[cutoff], info


def refresh_daily_cache(db_path=ROOT / 'bukit_kiara_air_history.db', now=None):
    """Only the independent worker/explicit offline command may call this."""
    freeze_policy()
    cutoff = cache._midnight(time.time() if now is None else now)
    path, info_path = _paths(cutoff)
    with _LOCK:
        if path.exists() and info_path.exists():
            return _load(cutoff)[1]
        if path.exists() or info_path.exists():
            raise RuntimeError('Incomplete Afternoon artifact; do not silently replace')
        attached = cache.attach_completed_labels(db_path, cutoff)
        bundle, audit = cache._eligible_training(cutoff, include_raw_context=True)
        positions = np.flatnonzero(bundle['raw_context_available'])
        dates = set(bundle['target_days'][positions].astype(str))
        if len(positions) < 120 or len(dates) < 10:
            raise RuntimeError('Insufficient common raw64 rolling14d support for Afternoon blend')
        started = time.time()
        s = svr.ContextSVR().fit(bundle, positions)
        if s.model.fit_status_ != 0:
            raise RuntimeError('Fixed SVR did not converge')
        h = make_pipeline(SimpleImputer(strategy='mean', add_indicator=True, keep_empty_features=True),
                          HistGradientBoostingRegressor(**PARAMETERS))
        x = np.asarray(bundle['context_raw64'][positions], dtype=np.float64)
        y = np.asarray(bundle['y'][positions], dtype=np.float64) - bundle['fresh'][positions]
        with threadpool_limits(limits=1):
            h.fit(x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        joblib.dump({'svr': {'model': s.model, 'normalizer': s.normalizer,
                            'training_metadata': s.training_metadata}, 'hgb': h,
                     'sourceHashes': _sources(), 'sklearnVersion': sklearn.__version__}, temporary, compress=3)
        temporary.replace(path)
        info = {**audit, 'commonTrainingRows': len(positions), 'commonDistinctTargetDates': len(dates),
                'excludedProspectiveRowsWithoutGenuineRaw64': int((~bundle['raw_context_available']).sum()),
                'commonBundleRowPositionsSha256': hashlib.sha256(positions.astype(np.int64).tobytes()).hexdigest(),
                'raw64TrainingFeatureSha256': hashlib.sha256(x.tobytes()).hexdigest(),
                'hgbTrainingTargetSha256': hashlib.sha256(y.tobytes()).hexdigest(),
                'checkpoint': str(path.relative_to(HERE)), 'checkpointSha256': cache.original.sha(path),
                'policySha256': cache.original.sha(POLICY), 'sourceHashes': _sources(),
                'fitStartedEpoch': started, 'fitFinishedEpoch': time.time(), 'artifactReadyEpoch': time.time(),
                'labelsNewlyAttached': len(attached), 'svrLearnerMetadata': s.training_metadata,
                'hgbParameters': PARAMETERS, 'hgbImputerStrategy': 'mean', 'threads': 1,
                'bothComponentsSameTrainingMembership': True, 'inferenceDevice': 'cpu'}
        cache._atomic(info_path, info)
        return _load(cutoff)[1]


def _components(models, bundle, index=0):
    s, h = models
    raw_svr = float(s.predict(bundle, np.array([index], dtype=np.int64))[0])
    fresh = float(bundle['fresh'][index])
    with threadpool_limits(limits=1):
        delta_hgb = float(h.predict(np.asarray(bundle['context_raw64'][[index]], dtype=np.float64))[0])
    raw_hgb = max(0., fresh + delta_hgb)
    gated_svr = fresh if abs(raw_svr - fresh) < 20 else raw_svr
    point = .5 * gated_svr + .5 * raw_hgb
    if not np.isfinite([raw_svr, gated_svr, raw_hgb, point]).all():
        raise RuntimeError('Nonfinite Afternoon component/output')
    return {'svrRawMeanUgM3': raw_svr, 'svrGate20MeanUgM3': gated_svr,
            'hgbRawMeanUgM3': raw_hgb, 'predictedMeanUgM3': point,
            'finalDeadbandApplied': False, 'predictedChangeFromReferenceUgM3': point - fresh}


def _predict_group(path, canonical):
    verify_policy()
    group = cache._json(path); row = group['rows'][0]
    cutoff = cache._midnight(row['issueEpoch'])
    if canonical and (group['kind'] != 'afternoon' or row['startEpoch'] != cutoff + 14 * 3600 or
                      row['endEpoch'] != cutoff + 16 * 3600):
        raise RuntimeError('Afternoon canonical exact dated target mismatch')
    if not row.get('rawContextAvailable'):
        raise RuntimeError('Afternoon query lacks genuinely captured raw64')
    issue = row['issueEpoch']
    if not (row['sensorWatermarkEpoch'] <= row['apiPublishedForecastIssueEpoch'] <=
            row['apiSnapshotReceivedEpoch'] <= row['inputSnapshotReceivedEpoch'] <= issue and
            row['dbSnapshotOpenedEpoch'] <= row['inputSnapshotReceivedEpoch'] and
            row['featureSourceMaxEpoch'] <= row['sensorWatermarkEpoch'] and
            row['sequenceSourceMaxEpoch'] <= row['sensorWatermarkEpoch'] and
            row['sensorReferenceEpoch'] <= row['sensorWatermarkEpoch'] and
            0 <= issue - row['sensorReferenceEpoch'] <= 240 and
            0 <= issue - row['apiPublishedForecastIssueEpoch'] <= 120 and
            0 <= issue - row['weatherFetchedEpoch'] <= 7200 and
            row['weatherFetchedEpoch'] <= row['dbSnapshotOpenedEpoch']):
        raise RuntimeError('Afternoon input source/receipt clock guard')
    cams_fetched = row.get('camsFetchedEpoch')
    if cams_fetched is not None and np.isfinite(cams_fetched) and not (
            cams_fetched <= row['dbSnapshotOpenedEpoch'] and 0 <= issue - cams_fetched <= 7200):
        raise RuntimeError('Afternoon CAMS source-after-issue/age guard')
    models, fitted = _load(cutoff)
    if fitted['physicallyReadyEpoch'] > row['issueEpoch']:
        raise RuntimeError('Afternoon artifact was not ready before actual issue')
    tensor_path = HERE / group['arrayPath']
    if cache.original.sha(tensor_path) != group['tensorSha256']:
        raise RuntimeError('Genuine Afternoon input tensor changed')
    with np.load(tensor_path, allow_pickle=False) as arrays:
        bundle = {k: arrays[k] for k in arrays.files}
    if not np.array_equal(bundle['context_raw64'][0].astype(np.float32), bundle['context'][0], equal_nan=True):
        raise RuntimeError('Raw64 context differs from same-snapshot original context')
    if bundle['context_raw64'].dtype != np.float64 or float(bundle['context_raw64'][0, 0]) != float(row['fresh']):
        raise RuntimeError('Genuine raw64 query/reference differs')
    points = _components(models, bundle)
    generated = time.time()
    if not (row['issueEpoch'] <= generated < row['startEpoch'] and
            0 <= generated - row['apiPublishedForecastIssueEpoch'] <= 120 and
            0 <= generated - row['sensorReferenceEpoch'] <= 240):
        raise RuntimeError('Afternoon current input is overdue/stale; never backdate')
    if canonical and not 0 <= generated - row['cohortTickEpoch'] <= 120:
        raise RuntimeError('Afternoon canonical issue missed noon120second grace')
    result = {**row, **points, 'forecastIssuedEpoch': row['issueEpoch'],
              'forecastGeneratedEpoch': generated, 'canonicalEvaluationClock': canonical,
              'candidateVersion': 'same_day_afternoon_equal_svr_gate20_hgb_mean_raw31_v1',
              'pairedPersistenceMeanUgM3': float(row['fresh']), 'researchShadowOnly': True,
              'trainingCutoffEpoch': cutoff, 'artifactPhysicallyReadyEpoch': fitted['physicallyReadyEpoch'],
              'checkpointSha256': fitted['checkpointSha256'], 'policySha256': cache.original.sha(POLICY),
              'inputGroupSha256': cache.original.sha(path), 'queryTensorSha256': group['tensorSha256'],
              'futureOutcomeLoaded': False, 'legacyWeatherArchiveReceiptEpoch': None,
              'legacyCamsArchiveReceiptEpoch': None, 'recordedEpoch': time.time()}
    forecast_path = cache.CACHE / 'forecasts' / (row['caseId'].replace(':', '_') + '_afternoon_blend.json')
    if forecast_path.exists():
        raise RuntimeError('Do not replace an existing genuine Afternoon forecast')
    cache._atomic(forecast_path, result)
    return result


def predict_canonical(now=None):
    verify_policy()
    cutoff = cache._midnight(time.time() if now is None else now)
    tick = cutoff + 12 * 3600
    path = cache.CACHE / 'inputs' / cache._day(tick) / f'{tick}_afternoon.json'
    if not path.exists():
        return None
    group = cache._json(path); row = group['rows'][0]
    forecast_path = cache.CACHE / 'forecasts' / (row['caseId'].replace(':', '_') + '_afternoon_blend.json')
    if forecast_path.exists():
        return cache._json(forecast_path)
    return _predict_group(path, True)


def diagnostic(db_path=ROOT / 'bukit_kiara_air_history.db'):
    """Actual current inputs, explicitly noncanonical; never fits here."""
    verify_policy()
    snapshot = cache.original.capture(db_path)
    start, end = cache.original.dated_target(snapshot['issueEpoch'])
    bundle, row = cache.original.query(snapshot, start=start, end=end)
    raw = raw_context(snapshot, start, end)
    bundle['context_raw64'] = raw[None, :]
    bundle['raw_context_available'] = np.array([True])
    issue = int(row['issueEpoch'])
    row.update(caseId=f'prospective-afternoon-blend-diagnostic:{issue}:{start}:{end}',
               targetName='afternoonBlendNoncanonicalDiagnostic', canonicalEvaluationClock=False,
               rawContextAvailable=True, completeEpoch=int(np.ceil(end / 900) * 900),
               targetDay=cache._day(start), cohortTickEpoch=issue,
               pairedIssuedIncumbent=cache._paired_incumbent(snapshot, start, end))
    path = cache.CACHE / 'inputs' / cache._day(issue) / f'{issue}_afternoon_blend_diagnostic.json'
    if path.exists():
        raise RuntimeError('Never replace a genuine diagnostic input')
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = path.with_suffix('.npz')
    np.savez_compressed(tensor_path, **bundle)
    cache.original.write(path, {'kind': 'afternoonBlendNoncanonicalDiagnostic', 'actualIssueEpoch': issue,
                               'arrayPath': str(tensor_path.relative_to(HERE)),
                               'tensorSha256': cache.original.sha(tensor_path), 'rows': [row],
                               'outcomeLabelsPresent': False, 'storedRawSnapshots': False})
    return _predict_group(path, False)


def step(db_path=ROOT / 'bukit_kiara_air_history.db', now=None):
    """Independent from Patch fit success; never used by HTTP handlers."""
    freeze_policy()
    now = time.time() if now is None else now
    status = {'checkedAtEpoch': time.time(), 'trainingCutoffEpoch': cache._midnight(now),
              'state': 'collecting', 'reason': None}
    try:
        fitted = refresh_daily_cache(db_path, now)
        status['commonTrainingRows'] = fitted['commonTrainingRows']
    except Exception as error:
        status.update(state='model_unavailable', reason=f'{type(error).__name__}: {error}')
    # Fit failure must not discard a genuinely received noon/training input.
    try:
        collection = cache.capture_cases(db_path, time.time())
        status['capturedInputGroups'] = collection['captured']
        if collection.get('requiredInputFailures'):
            status.update(state='awaiting_valid_noon_input',
                          reason='required_raw64_context_unavailable_retry_within120second_grace',
                          requiredInputFailures=collection['requiredInputFailures'])
        if 'commonTrainingRows' in status:
            forecast = predict_canonical(time.time())
            if forecast is not None:
                status['latestCanonicalForecastCaseId'] = forecast['caseId']
                status['state'] = 'forecast_recorded'
    except Exception as error:
        status.update(state='skipped', reason=f'{type(error).__name__}: {error}')
    cache._atomic(cache.CACHE / 'AFTERNOON_BLEND_STATUS.json', status)
    return status
