"""Small prospective PatchTST input collector and daily offline cache.

No HTTP handler should call refresh_daily_cache: it may fit. Prediction only
loads a completed CPU artifact. Production selection is never changed here.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import urllib.request

import numpy as np
import pandas as pd
import torch

import forward_patchtst as original
from forecast_clock import window_weights

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'runtime'
LEADS = (105, 120, 240, 420, 750, 1080, 1380)
_LOCK = threading.RLock()
_MODELS = {}
SEALED_START = int(pd.Timestamp('2026-10-20', tz='Asia/Kuala_Lumpur').timestamp())
SEALED_END = int(pd.Timestamp('2026-11-03', tz='Asia/Kuala_Lumpur').timestamp())


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(original.safe(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def freeze_policy():
    """Freeze before any new outcome-label access; preserve original adapter."""
    path = HERE / 'DAILY_POLICY.json'
    if path.exists():
        return verify_policy()
    original.verify()
    policy = {
        'version': 'patchtst_prospective_rolling_daily_cache_v1',
        'frozenAtEpoch': time.time(), 'architecture': 'Original fixed PatchTST weather fusion; no architecture/parameter changes',
        'prefix': 'Original certified6328 session_train rows; final55min12sec before Oct8midnight absent',
        'collection': 'Only newly received original-input snapshots; no reconstructed forecast cases or historical backfill',
        'trainingCaseClocks': {'gridSeconds': 1800, 'captureGraceSeconds': 120, 'leadsMinutes': list(LEADS),
                               'durationMinutes': 120, 'target': 'tick+lead to tick+lead+120min; issue remains actual snapshot receipt clock'},
        'canonicalCase': {'clockMYT': '18:05', 'captureGraceSeconds': 120,
                          'selection': 'First valid actually captured case in [18:05,18:07]; no later replacement by a more favorable prediction',
                          'target': 'Next local date14:00-16:00; exact actual issue/target retained'},
        'inputs': 'Original96x3past,26x15issuedweather,31causalcontext; sensor<=published API watermark; external fetch<=DB snapshot open; actual snapshot receipt<=candidate issue',
        'freshness': {'sensorReferenceMaximumAgeSeconds': 240, 'apiPublicationMaximumAgeSeconds': 120,
                      'weatherMaximumAgeSeconds': 7200, 'minimumCoveredPastPmSteps': 48},
        'storedInputs': 'Compressed tensors and compact source-clock/hash metadata outside SQLite; no full raw sensor/weather snapshots retained',
        'labels': 'Only strict covered right-closed15minPM medians over the exact target; all touched buckets must be complete; no filling; attach only after target complete; fit requires completeEpoch strictly before local midnight',
        'protectedLabels': 'Do not read or score Oct20-Nov2 target labels; skip overlapping target intervals in label materialization and training',
        'fit': {'lookbackDays': 14, 'minimumRows': 120, 'minimumTargetDates': 10, 'dailyCutoff': 'MYTmidnight',
                'epochs': 30, 'device': 'CUDA if torch.cuda.is_available at daily fit, otherwise CPU; same fixed recipe and recorded backend',
                'eligible': 'Prefix and newly collected input cases with issue in [midnight-14d,midnight), completeEpoch<midnight, finite completed label, original source guards passed',
                'once': 'Reuse completed dated model artifact; Oct8originalartifact may be adopted if its certified eligible-prefix row positions match exactly'},
        'outputs': 'Save raw originalPatchTST and fixed20deadband from same fit: abs(raw-fresh)<20 returns fresh, otherwise raw',
        'inference': 'Cached CPU state; no fitting in live request handling',
        'promotion': 'Research shadow only; unchanged declared improvement/support criteria, no automatic selected numerical promotion',
        'evaluation': 'No scoring implemented here; canonical first receipt and exact dated targets retained for independent prospective comparison',
        'producerWarmup': 'Cheap APIanalysis request in final60seconds before halfhour or18:05 tick; retry only within120second capture grace, never backdate',
    }
    original.write(path, policy)
    original.write(HERE / 'DAILY_FREEZE.json', {
        'policySha256': original.sha(path), 'dailyCacheSourceSha256': original.sha(Path(__file__)),
        'originalAdapterSha256': original.sha(HERE / 'forward_patchtst.py'),
        'originalFreezeSha256': original.sha(HERE / 'FREEZE.json'),
        'prefixArraySha256': original.sha(original.DATA / 'session_train.npz'),
        'prefixMetadataSha256': original.sha(original.DATA / 'session_train_metadata.pkl'),
        'modelSources': original.trainer.source_hashes()})
    return policy


def verify_policy():
    frozen = _json(HERE / 'DAILY_FREEZE.json')
    source_hash = original.sha(Path(__file__))
    source_matches = frozen['dailyCacheSourceSha256'] == source_hash
    if not source_matches and (HERE / 'DAILY_RUNTIME_REVISION.json').exists():
        revision_path = HERE / 'DAILY_RUNTIME_REVISION.json'
        revision = _json(revision_path)
        revision_freeze = _json(HERE / 'DAILY_RUNTIME_REVISION_FREEZE.json')
        source_matches = (revision_freeze['revisionSha256'] == original.sha(revision_path) and
                          revision['sourceSha256Before'] == frozen['dailyCacheSourceSha256'] and
                          revision['sourceSha256After'] == source_hash and
                          revision['originalDailyFreezeSha256'] == original.sha(HERE / 'DAILY_FREEZE.json'))
    if (frozen['policySha256'] != original.sha(HERE / 'DAILY_POLICY.json') or
        not source_matches or
        frozen['originalAdapterSha256'] != original.sha(HERE / 'forward_patchtst.py') or
        frozen['modelSources'] != original.trainer.source_hashes()):
        raise RuntimeError('Daily prospective source/policy changed')
    return _json(HERE / 'DAILY_POLICY.json')


def _day(epoch):
    return str(pd.Timestamp(epoch, unit='s', tz='UTC').tz_convert('Asia/Kuala_Lumpur').date())


def _midnight(epoch):
    return original.sessions.local_midnight(int(epoch))


def _unsealed(start, end):
    return end <= SEALED_START or start >= SEALED_END


def _groups():
    return sorted((CACHE / 'inputs').glob('*/*.json')) if (CACHE / 'inputs').exists() else []


def capture_cases(db_path=original.ROOT / 'bukit_kiara_air_history.db', now=None):
    """Capture genuine cases near the fixed tick; never backdate missed ticks."""
    freeze_policy()
    now = time.time() if now is None else float(now)
    tick = int(now) // 1800 * 1800
    midnight = _midnight(now)
    canonical_tick = midnight + (18 * 3600 + 5 * 60)
    wanted = []
    if 0 <= now - tick <= 120:
        wanted.append(('training', tick))
    if 0 <= now - canonical_tick <= 120:
        wanted.append(('nextAfternoon', canonical_tick))
    wanted = [(kind, anchor) for kind, anchor in wanted
              if not (CACHE / 'inputs' / _day(anchor) / f'{anchor}_{kind}.json').exists()]
    if not wanted:
        return {'state': 'outside_or_already_captured', 'captured': []}
    snapshot = original.capture(db_path)
    issue = int(snapshot['issueEpoch'])
    if issue - snapshot['apiPublishedForecastIssueEpoch'] > 120:
        raise RuntimeError('Published API forecast exceeds original120second freshness guard')
    captured = []
    with _LOCK:
        for kind, anchor in wanted:
            if not 0 <= issue - anchor <= 120:
                continue  # Snapshot latency cannot silently backdate a case.
            path = CACHE / 'inputs' / _day(anchor) / f'{anchor}_{kind}.json'
            if path.exists():
                continue
            queries, metadata = [], []
            targets = [(f'pooled_lead{lead}', anchor + lead * 60, anchor + (lead + 120) * 60)
                       for lead in LEADS] if kind == 'training' else [
                           ('nextAfternoon', midnight + 86400 + 14 * 3600, midnight + 86400 + 16 * 3600)]
            for name, start, end in targets:
                bundle, audit = original.query(snapshot, start=start, end=end)
                audit['caseId'] = f'prospective-{name}:{issue}:{start}:{end}'
                audit['targetName'] = name
                audit['targetDefinition'] = 'Exact two-hour mean of strict covered right-closed15minPM2.5 medians'
                audit['canonicalEvaluationClock'] = kind == 'nextAfternoon'
                audit['cohortTickEpoch'] = anchor
                audit['issueOffsetFromCohortTickSeconds'] = issue - anchor
                audit['completeEpoch'] = math.ceil(end / 900) * 900
                audit['targetDay'] = _day(start)
                audit['weatherPayloadHash'] = audit.pop('weatherPayloadSha256')
                audit['weatherPayloadHashAlgorithm'] = 'blake2b_160'
                audit['sourceSensorSha256'] = snapshot['sourceSensorSha256']
                audit['collectionReceiptEpoch'] = time.time()
                audit['futureOutcomeLoaded'] = False
                queries.append(bundle)
                metadata.append(audit)
            arrays = {key: np.concatenate([q[key] for q in queries], axis=0) for key in queries[0]}
            path.parent.mkdir(parents=True, exist_ok=True)
            tensor_path = path.with_suffix('.npz')
            temporary = tensor_path.with_name(tensor_path.stem + '.tmp.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(tensor_path)
            original.write(path, {'version': 'genuine_sequence_input_group_v1', 'kind': kind,
                                  'cohortTickEpoch': anchor, 'actualIssueEpoch': issue,
                                  'arrayPath': str(tensor_path.relative_to(HERE)),
                                  'tensorSha256': original.sha(tensor_path), 'rows': metadata,
                                  'dailyPolicySha256': original.sha(HERE / 'DAILY_POLICY.json'),
                                  'storedRawSnapshots': False, 'outcomeLabelsPresent': False})
            captured.append(str(path))
    return {'state': 'captured' if captured else 'snapshot_missed_tick', 'captured': captured,
            'actualIssueEpoch': issue}


def attach_completed_labels(db_path, cutoff):
    """Materialize only unsealed complete labels before the requested cutoff."""
    verify_policy()
    cutoff = int(cutoff)
    if cutoff > int(time.time()):
        raise ValueError('Cannot label future observations')
    added = []
    for group_path in _groups():
        group = _json(group_path)
        for i, row in enumerate(group['rows']):
            case_path = CACHE / 'labels' / (row['caseId'].replace(':', '_') + '.json')
            if case_path.exists() or row['completeEpoch'] >= cutoff or not _unsealed(row['startEpoch'], row['endEpoch']):
                continue
            # Only label buckets touched by this target are loaded. The edge
            # margin supplies raw observations of the first full native bucket.
            start = int(row['startEpoch']) // 900 * 900
            end = int(row['completeEpoch'])
            path = Path(db_path).resolve()
            opened = time.time()
            with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)) as conn:
                conn.execute('PRAGMA query_only=ON')
                conn.execute('BEGIN')
                records = [dict(zip(['epoch', 'pm02', 'atmp', 'rhum'], r)) for r in conn.execute(
                    'SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>? AND epoch<=? ORDER BY epoch', (start, end))]
            received = time.time()
            frame = original.prepare.causal.strict_frame(records, end)
            weights = window_weights(row['startEpoch'] - start, row['endEpoch'] - start)
            epochs = [start + offset * 900 for offset in weights]
            index = pd.to_datetime(epochs, unit='s', utc=True).tz_convert('Asia/Kuala_Lumpur')
            observed = frame.pm02.reindex(index).to_numpy(float) if not frame.empty else np.full(len(index), np.nan)
            if not np.isfinite(observed).all():
                continue
            actual = float(np.dot(observed, list(weights.values())))
            if actual < 0:
                continue
            label = {'caseId': row['caseId'], 'actual': actual, 'startEpoch': row['startEpoch'],
                     'endEpoch': row['endEpoch'], 'completeEpoch': row['completeEpoch'],
                     'observedLabelCutoffEpoch': cutoff, 'labelSnapshotOpenedEpoch': opened,
                     'labelSnapshotReceivedEpoch': received, 'labelCreatedEpoch': time.time(),
                     'bucketEpochs': epochs, 'bucketValues': observed.tolist(), 'bucketWeights': list(weights.values()),
                     'sourceRowsSha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                     'coveredEveryTouchedBucket': True, 'sealedLabel': False}
            original.write(case_path, label)
            added.append(str(case_path))
    return added


def _eligible_training(cutoff):
    """Combine fixed prefix and completed genuinely collected input cases."""
    original.verify()
    with np.load(original.DATA / 'session_train.npz', allow_pickle=False) as stored:
        prefix = {key: stored[key] for key in stored.files}
    metadata = pd.read_pickle(original.DATA / 'session_train_metadata.pkl').reset_index(drop=True)
    if not np.array_equal(metadata.arrayIndex.to_numpy(), np.arange(len(metadata))):
        raise RuntimeError('Certified prefix row order differs')
    eligible = (metadata.issueEpoch.ge(cutoff - 14 * 86400) & metadata.issueEpoch.lt(cutoff) &
                metadata.completeEpoch.lt(cutoff) & np.isfinite(metadata.actual) &
                metadata.sequenceValid & prefix['valid'])
    if 'valid' in metadata:
        eligible &= metadata.valid
    prefix_indices = np.flatnonzero(eligible.to_numpy())
    keys = ('past', 'future', 'context', 'fresh', 'y', 'valid')
    chunks = {key: [prefix[key][prefix_indices]] for key in keys}
    prefix_meta = metadata.iloc[prefix_indices]
    target_days = set(prefix_meta.targetDay.astype(str))
    cases = []
    for group_path in _groups():
        group = _json(group_path)
        # Only seven fixed training leads join the pooled fit. The canonical
        # evaluation query stays a query; do not enrich the fit with that head.
        if group['kind'] != 'training':
            continue
        eligible = []
        labels = []
        for i, row in enumerate(group['rows']):
            if not (cutoff - 14 * 86400 <= row['issueEpoch'] < cutoff and row['completeEpoch'] < cutoff
                    and _unsealed(row['startEpoch'], row['endEpoch'])):
                continue
            label_path = CACHE / 'labels' / (row['caseId'].replace(':', '_') + '.json')
            if not label_path.exists():
                continue
            label = _json(label_path)
            if label['caseId'] != row['caseId'] or not np.isfinite(label['actual']):
                raise RuntimeError('Saved prospective label does not match its case')
            eligible.append(i)
            labels.append(float(label['actual']))
            target_days.add(row['targetDay'])
            cases.append({'caseId': row['caseId'], 'inputGroupSha256': original.sha(group_path),
                          'labelSha256': original.sha(label_path), 'completeEpoch': row['completeEpoch']})
        if not eligible:
            continue
        array_path = HERE / group['arrayPath']
        if original.sha(array_path) != group['tensorSha256']:
            raise RuntimeError('Prospective input tensor changed')
        with np.load(array_path, allow_pickle=False) as arrays:
            for key in keys:
                chunks[key].append(np.asarray(labels, dtype=float) if key == 'y' else arrays[key][eligible])
    bundle = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
    if len(bundle['y']) < 120 or len(target_days) < 10:
        raise RuntimeError('Insufficient unsealed completed rolling14d training support')
    digest = hashlib.sha256(b''.join(np.ascontiguousarray(bundle[k]).tobytes() for k in keys)).hexdigest()
    audit = {'trainingCutoffEpoch': cutoff, 'trainingCutoffMYT': original.local(cutoff),
             'trainingRows': len(bundle['y']), 'distinctTrainingTargetDates': len(target_days),
             'certifiedPrefixRows': len(prefix_indices), 'prospectiveRows': len(cases),
             'prefixRowPositionsSha256': hashlib.sha256(prefix_indices.astype(np.int64).tobytes()).hexdigest(),
             'trainingTensorLabelSha256': digest,
             'maximumTrainingCompleteEpoch': max(([int(prefix_meta.completeEpoch.max())] if len(prefix_meta) else []) + [r['completeEpoch'] for r in cases]),
             'prospectiveCases': cases}
    return bundle, audit


def refresh_daily_cache(db_path=original.ROOT / 'bukit_kiara_air_history.db', now=None):
    """Background-only daily fit or adopt/reuse a completed CPU artifact."""
    freeze_policy()
    now = time.time() if now is None else float(now)
    cutoff = _midnight(now)
    path = CACHE / 'models' / f'{cutoff}_patchtst.pt'
    info_path = path.with_suffix('.json')
    with _LOCK:
        if path.exists() and info_path.exists():
            info = _json(info_path)
            if original.sha(path) != info['checkpointSha256']:
                raise RuntimeError('Completed daily artifact differs')
            if cutoff not in _MODELS:
                _MODELS[cutoff] = original.trainer.SequenceRegressor.load(path)
            return info
        if path.exists() or info_path.exists():
            raise RuntimeError('Incomplete daily artifact; no silent replacement')
        attached = attach_completed_labels(db_path, cutoff)
        bundle, audit = _eligible_training(cutoff)
        started = time.time()
        original_fit = _json(HERE / 'FIT.json')
        adopt = (cutoff == original_fit['trainingCutoffEpoch'] and audit['prospectiveRows'] == 0 and
                 audit['prefixRowPositionsSha256'] == original_fit['trainingRowPositionsSha256'])
        if adopt:
            source_path = HERE / original_fit['checkpoint']
            if original.sha(source_path) != original_fit['checkpointSha256']:
                raise RuntimeError('Original cached artifact changed')
            model = original.trainer.SequenceRegressor.load(source_path)
            training = original_fit['learnerMetadata']
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = original.trainer.SequenceRegressor('patchtst', device=device).fit(bundle, np.arange(len(bundle['y'])))
            training = model.training_metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        model.save(temporary)
        temporary.replace(path)
        info = {**audit, 'checkpoint': str(path.relative_to(HERE)), 'checkpointSha256': original.sha(path),
                'dailyPolicySha256': original.sha(HERE / 'DAILY_POLICY.json'),
                'fittedAtEpoch': time.time(), 'refreshStartedEpoch': started,
                'adoptedOriginalDailyFit': adopt, 'labelsNewlyAttached': len(attached),
                'learnerMetadata': training, 'inferenceDevice': 'cpu'}
        original.write(info_path, info)
        _MODELS[cutoff] = model
        for old in list(_MODELS):
            if old != cutoff:
                del _MODELS[old]
        return info


def _cached(cutoff):
    path = CACHE / 'models' / f'{cutoff}_patchtst.pt'
    info_path = path.with_suffix('.json')
    if not path.exists() or not info_path.exists():
        raise RuntimeError('Daily background model is not ready; prediction never fits')
    info = _json(info_path)
    if cutoff not in _MODELS:
        if original.sha(path) != info['checkpointSha256']:
            raise RuntimeError('Saved daily model hash differs')
        _MODELS[cutoff] = original.trainer.SequenceRegressor.load(path)
    return _MODELS[cutoff], info


def predict_canonical(now=None):
    """Predict the captured current-day18:05 case using CPU cache; no fitting."""
    verify_policy()
    now = time.time() if now is None else float(now)
    cutoff = _midnight(now)
    tick = cutoff + 18 * 3600 + 5 * 60
    path = CACHE / 'inputs' / _day(tick) / f'{tick}_nextAfternoon.json'
    if not path.exists():
        return None
    group = _json(path)
    row = group['rows'][0]
    forecast_path = CACHE / 'forecasts' / (row['caseId'].replace(':', '_') + '.json')
    if forecast_path.exists():
        return _json(forecast_path)
    model, fitted = _cached(cutoff)
    if fitted['fittedAtEpoch'] > row['issueEpoch']:
        raise RuntimeError('Daily artifact was not ready at captured canonical issue')
    tensor_path = HERE / group['arrayPath']
    if original.sha(tensor_path) != group['tensorSha256']:
        raise RuntimeError('Captured canonical tensor changed')
    with np.load(tensor_path, allow_pickle=False) as stored:
        bundle = {key: stored[key] for key in stored.files}
    raw = float(model.predict(bundle, np.array([0], dtype=np.int64))[0])
    generated = time.time()
    if not (np.isfinite(raw) and row['issueEpoch'] <= generated < row['startEpoch'] and
            generated <= row['cohortTickEpoch'] + 120 and
            0 <= generated - row['apiPublishedForecastIssueEpoch'] <= 120 and
            0 <= generated - row['sensorReferenceEpoch'] <= 240):
        raise RuntimeError('Overdue or stale canonical input cannot create a backdated forecast')
    gate = row['fresh'] if abs(raw - row['fresh']) < 20 else raw
    result = {**row, 'forecastIssuedEpoch': row['issueEpoch'], 'forecastGeneratedEpoch': generated,
              'predictedMeanUgM3': raw, 'predictedChangeFromReferenceUgM3': raw - row['fresh'],
              'deadband20MeanUgM3': float(gate), 'deadband20Applied': gate != raw,
              'pairedPersistenceMeanUgM3': row['fresh'], 'researchShadowOnly': True,
              'dailyPolicySha256': original.sha(HERE / 'DAILY_POLICY.json'),
              'checkpointSha256': fitted['checkpointSha256'], 'trainingCutoffEpoch': cutoff,
              'inputGroupSha256': original.sha(path), 'queryTensorSha256': group['tensorSha256'],
              'recordedEpoch': time.time(), 'futureOutcomeLoaded': False}
    original.write(forecast_path, result)
    return result


def step(db_path=original.ROOT / 'bukit_kiara_air_history.db', now=None):
    """One independent service-worker step; never invoke from HTTP handlers."""
    freeze_policy()
    now = time.time() if now is None else float(now)
    cutoff = _midnight(now)
    status = {'state': 'collecting', 'trainingCutoffEpoch': cutoff,
              'checkedAtEpoch': time.time(), 'reason': None}
    try:
        fitted = refresh_daily_cache(db_path, now)
        status['trainingRows'] = fitted['trainingRows']
        status['prospectiveTrainingRows'] = fitted['prospectiveRows']
        next_halfhour = (int(now) // 1800 + 1) * 1800
        canonical_tick = cutoff + 18 * 3600 + 5 * 60
        warm = (0 < next_halfhour - now <= 60 or 0 < canonical_tick - now <= 60)
        if warm:
            # A request wakes the existing demand-driven producer; its result
            # is not reused as a candidate issue or falsely backdated input.
            with urllib.request.urlopen('http://127.0.0.1:8765/api/analysis?days=28', timeout=30) as response:
                response.read()
            status['state'] = 'preparing'
        collection = capture_cases(db_path, time.time())
        status['capturedInputGroups'] = collection['captured']
        forecast = predict_canonical(time.time())
        if forecast is not None:
            status['latestCanonicalForecastCaseId'] = forecast['caseId']
        if collection['captured']:
            status['state'] = 'captured'
        elif collection['state'] == 'snapshot_missed_tick':
            status['state'] = 'skipped'
            status['reason'] = 'snapshot_receipt_missed_declared_tick_grace'
    except (RuntimeError, ValueError, OSError, KeyError, AssertionError) as error:
        status['state'] = 'skipped'
        status['reason'] = f'{type(error).__name__}: {error}'
    _atomic(CACHE / 'STATUS.json', status)
    return status


if __name__ == '__main__':
    print(json.dumps({'policy': freeze_policy()['version'], 'state': 'policy_frozen_no_label_read'}))
