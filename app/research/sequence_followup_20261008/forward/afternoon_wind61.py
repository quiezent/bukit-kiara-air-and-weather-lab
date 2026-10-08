"""One standalone wind61 SVR forward challenger; no fitting in HTTP handling."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time

import numpy as np
import sklearn

import daily_cache as cache

HERE = Path(__file__).resolve().parent
ROOT = cache.original.ROOT
STUDY = ROOT / 'research/svr_wind61_blend_20261008'
WIND = ROOT / 'research/ordered_weather_20261008'
POLICY = HERE / 'AFTERNOON_WIND61_POLICY.json'
FREEZE = HERE / 'AFTERNOON_WIND61_FREEZE.json'
SCHEMA = HERE / 'AFTERNOON_WIND61_SCHEMA.json'
SUPERSESSION = HERE / 'AFTERNOON_CHALLENGER_SUPERSESSION.json'

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

trainer = _module('forward_fixed_wind61_svr_trainer', STUDY / 'wind_svr_trainer.py')
features = _module('forward_fixed_issued_wind30', WIND / 'features.py')
_LOCK = threading.RLock()
_MODELS = {}


def _sources():
    paths = [Path(__file__), HERE / 'daily_cache.py', HERE.parent / 'prospective_worker.py',
             STUDY / 'wind_svr_trainer.py', STUDY / 'POLICY.json', STUDY / 'FREEZE.json',
             STUDY / 'SOURCE_SCHEMA.json', trainer.ORIGINAL,
             WIND / 'features.py', WIND / 'FEATURE_ARRAYS.npz', WIND / 'POLICY.json', WIND / 'FREEZE.json',
             HERE / 'DAILY_WIND61_SUPERSESSION_REVISION.json',
             HERE / 'DAILY_WIND61_SUPERSESSION_REVISION_FREEZE.json', SUPERSESSION]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): cache.original.sha(p) for p in paths}


def freeze_policy():
    cache.verify_policy()
    if POLICY.exists(): return verify_policy()
    columns = list(cache.original.prepare.CONTEXT) + list(features.EXTRA_COLUMNS)
    assert len(columns) == 61
    cache.original.write(SCHEMA, {'columns': columns, 'original31': 'EXACT original float32 context values promoted losslessly tofloat64',
                                'extra30': 'Frozen features.row on SAME originally captured future26x15/future_epochs/issue/start/end; optional missing values preserved',
                                'prefix': 'Frozen FEATURE_ARRAYS train_wind61 extras selected by exact original eligible row positions',
                                'transformedDimension': 122, 'units': 'Wind/gust km/h,BLH m,ventilation m2/s;925hPa pressure level differs from180m height',
                                'rain': 'Keep certified original basic31 native-hour rain; no extra shifted rain summaries',
                                'raw64HgbPrerequisite': False})
    policy = {'version': 'same_day_afternoon_standalone_wind61_svr_gate20_v1',
              'frozenAtEpoch': time.time(), 'head': 'same-day Afternoon only',
              'canonical': {'clockMYT': '12:00', 'captureGraceSeconds': 120, 'target': 'Today14:00-16:00 exact dated mean',
                            'selection': 'First valid actually received input in[12:00,12:02]; invalid required wind extraction does not finalize group; no backfill/replacement'},
              'learner': trainer.specification(), 'output': 'max0(fresh+20*SVRhead), then abs(raw-fresh)<20 returns fresh; exactly20 retained. NO HGB or blend.',
              'training': {'lookbackDays': 14, 'cutoff': 'MYTmidnight; original issue and completeEpoch strictly beforecutoff',
                           'minimumRows': 120, 'minimumDistinctTargetDates': 10,
                           'membership': 'Full original certified prefix plus genuine completed pooled seven-lead inputs; no raw64-HGB exclusions. Original Patch membership unchanged.',
                           'missingness': 'Training-only61 observed means/std and61 missing flags; float32 transformed122; target float32(actual-fresh)/20'},
              'sourceGuards': 'Sourcefetch<=DBsnapshotopen<=DBreceipt<=issue; publishedsensorwatermark<=APIpublication<=APIreceipt<=DBreceipt<=issue; all usedsensor/buckets<=watermark. Same original query/source ages.',
              'freshness': {'sensorMaximumAgeSeconds': 240, 'apiPublicationMaximumAgeSeconds': 120,
                            'weatherMaximumAgeSeconds': 7200, 'artifact': 'Completed state AND readinessJSON physically exist beforeactual candidateissue'},
              'legacyReceiptLimitation': 'Retain sourcefetched_epoch separately from actual API/DBsnapshotreceipt; independent original weather/CAMS HTTP/archive receipt unavailable, never invented',
              'selectionExposure': 'Explicit development selection from already exposed standalone diagnostic outputs. AllSep18-Oct7 inspected. Original wind61_equal_mix primary remains rejected; no primary redefinition/fresh qualification.',
              'supersession': 'Retire earlier basic31SVR/HGBequalmix BEFORE firstnoon canonical; preserve its policies/artifacts/diagnostics. Only standalone wind61 may publish same-day Afternoon challenger forecasts.',
              'protectedLabels': 'Never read/label/score Oct20-Nov2 target outcomes; original skip beforeDBlabelaccess retained',
              'promotion': {'newDistinctTargetDates': 10, 'independentRises': 5, 'independentDrops': 5,
                            'minimumMAEGainOverIssuedIncumbentAndPersistence': .5, 'maximumP90Increase': .5,
                            'maximumQuietMAEIncrease': 1., 'direction': 'Improved sign-correct>=10recall; no worse wrongway/false-clearing; distinct events',
                            'oneChallengerPerHead': True, 'automaticPromotion': False},
              'runtime': 'One CPU1 daily fit in existing independent Windowsservice worker, cached prediction only; no HTTPfitting/mainnumerical/modelselection/UIchange',
              'diagnostic': 'Actual current NONCANONICAL snapshot proves software path only, never canonical evidence',
              'schemaSha256': cache.original.sha(SCHEMA), 'sklearnVersion': sklearn.__version__}
    cache.original.write(POLICY, policy)
    cache.original.write(FREEZE, {'policySha256': cache.original.sha(POLICY), 'schemaSha256': cache.original.sha(SCHEMA),
                                'sources': _sources(), 'frozenBeforeDailyFitOrNewPrediction': True})
    return policy


def verify_policy():
    cache.verify_policy()
    f = cache._json(FREEZE)
    if (f['policySha256'] != cache.original.sha(POLICY) or f['schemaSha256'] != cache.original.sha(SCHEMA)
            or f['sources'] != _sources()):
        raise RuntimeError('Frozen standalone wind61 policy/source changed')
    p = cache._json(POLICY)
    if p['sklearnVersion'] != sklearn.__version__: raise RuntimeError('Frozen wind61 sklearn differs')
    return p


def wind_extra(bundle, row, index=0):
    """Pure causal transform; missing optional winds stay missing, not invalid."""
    return features.row(bundle['future'][index], bundle['future_epochs'][index],
                        int(row['issueEpoch']), int(row['startEpoch']), int(row['endEpoch']))


def prefix_extra(indices, original_context):
    with np.load(WIND / 'FEATURE_ARRAYS.npz', allow_pickle=False) as stored:
        base = stored['train_basic31'][indices]; rich = stored['train_wind61'][indices]
    if not np.array_equal(base, original_context, equal_nan=True) or not np.array_equal(rich[:, :31], base, equal_nan=True):
        raise RuntimeError('Original wind61 prefix row/precision alignment differs')
    return rich[:, 31:]


def _paths(cutoff):
    path = cache.CACHE / 'models' / f'{cutoff}_afternoon_wind61_v1.joblib'
    return path, path.with_suffix('.json')


def _load(cutoff):
    verify_policy(); path, info_path = _paths(cutoff)
    if not path.exists() or not info_path.exists(): raise RuntimeError('Wind61 background artifact not ready; prediction never fits')
    info = cache._json(info_path)
    if (info['policySha256'] != cache.original.sha(POLICY) or info['sourceHashes'] != _sources()
            or cache.original.sha(path) != info['checkpointSha256'] or info['trainingCutoffEpoch'] != cutoff):
        raise RuntimeError('Completed wind61 artifact/source differs')
    if cutoff not in _MODELS:
        _MODELS.clear(); _MODELS[cutoff] = trainer.ContextSVR.load(path)
    info = dict(info, physicallyReadyEpoch=max(path.stat().st_mtime, info_path.stat().st_mtime, info['artifactReadyEpoch']))
    return _MODELS[cutoff], info


def refresh_daily_cache(db_path=ROOT / 'bukit_kiara_air_history.db', now=None):
    """Background-only daily CPU1 fit; public HTTP never calls this."""
    freeze_policy(); cutoff = cache._midnight(time.time() if now is None else now)
    path, info_path = _paths(cutoff)
    with _LOCK:
        if path.exists() and info_path.exists(): return _load(cutoff)[1]
        if path.exists() or info_path.exists(): raise RuntimeError('Incomplete wind61 artifact; no silent replacement')
        attached = cache.attach_completed_labels(db_path, cutoff)
        bundle, audit = cache._eligible_training(cutoff, include_wind_context=True)
        assert bundle['context'].dtype == np.float32
        prepared = dict(bundle, context=np.concatenate([bundle['context'].astype(np.float64), bundle['wind_extra30']], axis=1))
        assert np.array_equal(prepared['context'][:, :31], bundle['context'], equal_nan=True)
        positions = np.arange(len(bundle['y']), dtype=np.int64)
        model = trainer.ContextSVR().fit(prepared, positions)
        if model.model.fit_status_ != 0: raise RuntimeError('Fixed wind61 SVR failed to converge')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp'); model.save(temporary); temporary.replace(path)
        info = {**audit, 'checkpoint': str(path.relative_to(HERE)), 'checkpointSha256': cache.original.sha(path),
                'policySha256': cache.original.sha(POLICY), 'sourceHashes': _sources(), 'artifactReadyEpoch': time.time(),
                'fittedAtEpoch': time.time(), 'labelsNewlyAttached': len(attached), 'learnerMetadata': model.training_metadata,
                'windExtraFeatureSha256': hashlib.sha256(bundle['wind_extra30'].tobytes()).hexdigest(),
                'originalFirst31Float32Exact': True, 'noRaw64HgbPrerequisite': True, 'inferenceDevice': 'cpu'}
        cache._atomic(info_path, info)
        return _load(cutoff)[1]


def _points(model, bundle, row, index=0):
    extra = wind_extra(bundle, row, index)
    if 'wind_extra30' in bundle and not np.array_equal(extra, bundle['wind_extra30'][index], equal_nan=True):
        raise RuntimeError('Saved wind summaries differ from exact originally captured tensors')
    context = np.concatenate([bundle['context'].astype(np.float64),
                              np.asarray(bundle.get('wind_extra30', extra[None, :]), dtype=np.float64)], axis=1)
    prepared = dict(bundle, context=context)
    raw = float(model.predict(prepared, np.array([index], np.int64))[0])
    fresh = float(bundle['fresh'][index]); point = fresh if abs(raw-fresh) < 20 else raw
    if not np.isfinite([raw, fresh, point]).all(): raise RuntimeError('Nonfinite wind61 prediction')
    return {'svrWind61RawMeanUgM3': raw, 'predictedMeanUgM3': point, 'deadband20Applied': point != raw,
            'predictedChangeFromReferenceUgM3': point-fresh, 'hgbOrBlendUsed': False}


def _predict_group(path, canonical):
    verify_policy(); group = cache._json(path); row = group['rows'][0]; issue = row['issueEpoch']
    cutoff = cache._midnight(issue)
    if canonical and (group['kind'] != 'afternoon' or row['startEpoch'] != cutoff+14*3600 or row['endEpoch'] != cutoff+16*3600):
        raise RuntimeError('Wind61 canonical exact dated target mismatch')
    if not row.get('windContextAvailable'): raise RuntimeError('Required wind61 tensor summaries unavailable')
    if not (row['sensorWatermarkEpoch'] <= row['apiPublishedForecastIssueEpoch'] <= row['apiSnapshotReceivedEpoch'] <=
            row['inputSnapshotReceivedEpoch'] <= issue and row['dbSnapshotOpenedEpoch'] <= row['inputSnapshotReceivedEpoch'] and
            row['featureSourceMaxEpoch'] <= row['sensorWatermarkEpoch'] and row['sequenceSourceMaxEpoch'] <= row['sensorWatermarkEpoch'] and
            row['sensorReferenceEpoch'] <= row['sensorWatermarkEpoch'] and 0 <= issue-row['sensorReferenceEpoch'] <= 240 and
            0 <= issue-row['apiPublishedForecastIssueEpoch'] <= 120 and 0 <= issue-row['weatherFetchedEpoch'] <= 7200 and
            row['weatherFetchedEpoch'] <= row['dbSnapshotOpenedEpoch']):
        raise RuntimeError('Wind61 source/receipt clock guard')
    cams = row.get('camsFetchedEpoch')
    if cams is not None and np.isfinite(cams) and not(cams <= row['dbSnapshotOpenedEpoch'] and 0 <= issue-cams <= 7200):
        raise RuntimeError('Wind61 CAMS source-afterissue/age guard')
    model, fitted = _load(cutoff)
    if fitted['physicallyReadyEpoch'] > issue: raise RuntimeError('Wind61 artifact not ready before actual issue')
    tensor_path = HERE/group['arrayPath']
    if cache.original.sha(tensor_path) != group['tensorSha256']: raise RuntimeError('Genuine wind61 tensor changed')
    with np.load(tensor_path, allow_pickle=False) as stored: bundle = {k: stored[k] for k in stored.files}
    if bundle['context'].dtype != np.float32 or bundle['context'].shape != (1, 31):
        raise RuntimeError('Original first31 precision/shape differs')
    points = _points(model, bundle, row); generated = time.time()
    if not (issue <= generated < row['startEpoch'] and 0 <= generated-row['apiPublishedForecastIssueEpoch'] <= 120
            and 0 <= generated-row['sensorReferenceEpoch'] <= 240): raise RuntimeError('Wind61 overdue/stale input; never backdate')
    if canonical and not 0 <= generated-row['cohortTickEpoch'] <= 120: raise RuntimeError('Wind61 noon120second grace missed')
    result = {**row, **points, 'candidateVersion': 'same_day_afternoon_standalone_wind61_svr_gate20_v1',
              'forecastIssuedEpoch': issue, 'forecastGeneratedEpoch': generated, 'canonicalEvaluationClock': canonical,
              'pairedPersistenceMeanUgM3': float(row['fresh']), 'researchShadowOnly': True, 'trainingCutoffEpoch': cutoff,
              'artifactPhysicallyReadyEpoch': fitted['physicallyReadyEpoch'], 'checkpointSha256': fitted['checkpointSha256'],
              'policySha256': cache.original.sha(POLICY), 'inputGroupSha256': cache.original.sha(path),
              'queryTensorSha256': group['tensorSha256'], 'futureOutcomeLoaded': False,
              'legacyWeatherArchiveReceiptEpoch': None, 'legacyCamsArchiveReceiptEpoch': None, 'recordedEpoch': time.time()}
    out = cache.CACHE/'forecasts'/(row['caseId'].replace(':', '_')+'_afternoon_wind61.json')
    if out.exists(): raise RuntimeError('Do not replace genuine wind61 forecast')
    cache._atomic(out, result); return result


def predict_canonical(now=None):
    verify_policy(); cutoff = cache._midnight(time.time() if now is None else now); tick = cutoff+12*3600
    path = cache.CACHE/'inputs'/cache._day(tick)/f'{tick}_afternoon.json'
    if not path.exists(): return None
    row = cache._json(path)['rows'][0]
    out = cache.CACHE/'forecasts'/(row['caseId'].replace(':', '_')+'_afternoon_wind61.json')
    if out.exists(): return cache._json(out)
    return _predict_group(path, True)


def diagnostic(db_path=ROOT/'bukit_kiara_air_history.db'):
    verify_policy(); snapshot = cache.original.capture(db_path)
    start, end = cache.original.dated_target(snapshot['issueEpoch'])
    bundle, row = cache.original.query(snapshot, start=start, end=end)
    bundle['wind_extra30'] = wind_extra(bundle, row)[None, :]
    issue = row['issueEpoch']
    row.update(caseId=f'prospective-afternoon-wind61-diagnostic:{issue}:{start}:{end}',
               targetName='afternoonWind61NoncanonicalDiagnostic', canonicalEvaluationClock=False,
               windContextAvailable=True, cohortTickEpoch=issue, targetDay=cache._day(start),
               completeEpoch=int(np.ceil(end/900)*900), pairedIssuedIncumbent=cache._paired_incumbent(snapshot, start, end))
    path = cache.CACHE/'inputs'/cache._day(issue)/f'{issue}_afternoon_wind61_diagnostic.json'
    if path.exists(): raise RuntimeError('Never replace genuine wind61 diagnostic input')
    path.parent.mkdir(parents=True, exist_ok=True); tensor = path.with_suffix('.npz')
    np.savez_compressed(tensor, **bundle)
    cache.original.write(path, {'kind': 'afternoonWind61NoncanonicalDiagnostic', 'actualIssueEpoch': issue,
                               'arrayPath': str(tensor.relative_to(HERE)), 'tensorSha256': cache.original.sha(tensor),
                               'rows': [row], 'outcomeLabelsPresent': False, 'storedRawSnapshots': False})
    return _predict_group(path, False)


def step(db_path=ROOT/'bukit_kiara_air_history.db', now=None):
    freeze_policy(); now = time.time() if now is None else now
    status = {'checkedAtEpoch': time.time(), 'trainingCutoffEpoch': cache._midnight(now), 'state': 'collecting', 'reason': None}
    try:
        fit = refresh_daily_cache(db_path, now); status['trainingRows'] = fit['trainingRows']
    except Exception as error:
        status.update(state='model_unavailable', reason=f'{type(error).__name__}: {error}')
    try:
        collected = cache.capture_cases(db_path, time.time()); status['capturedInputGroups'] = collected['captured']
        if collected.get('requiredInputFailures'):
            status.update(state='awaiting_valid_noon_input', reason='required_wind61_inputs_unavailable_retry_within120second_grace',
                          requiredInputFailures=collected['requiredInputFailures'])
        if 'trainingRows' in status:
            forecast = predict_canonical(time.time())
            if forecast is not None: status.update(state='forecast_recorded', latestCanonicalForecastCaseId=forecast['caseId'])
    except Exception as error:
        status.update(state='skipped', reason=f'{type(error).__name__}: {error}')
    cache._atomic(cache.CACHE/'AFTERNOON_WIND61_STATUS.json', status); return status
