"""Original fixed PatchTST: one daily offline fit and real forward queries.

Only research artifacts are written. A query reads sensor observations no later
than the API's published watermark, and already received issued weather/CAMS.
No future observed sensor rows or query outcomes are loaded.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = ROOT / 'research/sequence_models_20261008'
DATA = EXPERIMENT / 'data'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT))
import sequence_trainer as trainer
import weather_session_model as sessions
import window_pm_predictor as regional
from forecast_payload import decode_payload_text

_spec = importlib.util.spec_from_file_location('forward_sequence_sources', DATA / 'prepare.py')
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(safe(value), indent=2, allow_nan=False) + '\n'
    # Do not overwrite an issued record. Complete the file before making it visible.
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def local(epoch):
    return str(pd.Timestamp(epoch, unit='s', tz='UTC').tz_convert('Asia/Kuala_Lumpur'))


def cutoff_now():
    return int(pd.Timestamp.now(tz='Asia/Kuala_Lumpur').normalize().timestamp())


def sources():
    paths = [Path(__file__), DATA / 'prepare.py', DATA / 'session_train.npz',
             DATA / 'session_train_metadata.pkl', DATA / 'PREPARATION.json',
             DATA / 'POLICY.json', EXPERIMENT / 'MODEL_POLICY.json',
             EXPERIMENT / 'MODEL_FREEZE.json', ROOT / 'weather_session_model.py',
             ROOT / 'weather_session_features.py', ROOT / 'afternoon_direction_features.py',
             ROOT / 'rain_weather_features.py', ROOT / 'window_pm_predictor.py',
             ROOT / 'forecast_freshness.py', ROOT / 'collection_quality.py',
             ROOT / 'forecast_clock.py', ROOT / 'forecast_payload.py']
    paths += [EXPERIMENT / name for name in trainer.source_hashes()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze():
    frozen = json.loads((EXPERIMENT / 'MODEL_FREEZE.json').read_text(encoding='utf-8'))
    if frozen['sourceHashes'] != trainer.source_hashes():
        raise RuntimeError('Original architecture/trainer changed')
    if sha(DATA / 'PREPARATION.json') != frozen['dataPreparationSha256']:
        raise RuntimeError('Original prepared source changed')
    cutoff = cutoff_now()
    policy = {'version': 'original_patchtst_actual_forward_diagnostic_v1',
              'frozenAtEpoch': time.time(), 'trainingCutoffEpoch': cutoff,
              'trainingCutoffMYT': local(cutoff), 'variant': 'patchtst',
              'fit': 'Original fixed30epoch CUDA daily fit; issue in [midnight-14d,midnight), completed label strictly before midnight; original frozen session_train rows only',
              'fitCutoff': 'Original Oct7 23:04:48 source means final55min12sec before current midnight absent; no new outcome labels read',
              'target': 'One dated Afternoon14:00-16:00 two-hour mean; first eligible future date with >=90min lead; no first-crossing or probability claim',
              'queries': 'Actual API publication watermark bounds a read-only materialized DB transaction; candidate issue freezes after API and DB snapshot receipt',
              'sourceClocks': 'Network fetch <=DB snapshot opened <=DB receipt <=candidate issue; published API watermark <=original API forecast issue <=API receipt <=candidate issue',
              'maximumReferenceAgeSeconds': 240, 'maximumApiPublicationAgeSeconds': 180,
              'maximumWeatherAgeSeconds': 7200, 'minimumCoveredPastPmSteps': 48,
              'featureAdapter': 'Unmodified original data.prepare.Sources.past/future plus weather_session_model.query_features basic31; original train-only normalizers',
              'deployment': 'Research shadow only; no production selection/service changes',
              'evaluation': 'Noncanonical current issue is a genuine forward diagnostic; cannot be counted as canonical18:05 skill evidence',
              'uncertainty': 'Scalar estimate only; no calibrated interval, event chance, or cycling-clearance claim'}
    policy['secondSavedOutput'] = 'Same fitted raw point with fixed20ug/m3deadband: if abs(raw-fresh)<20 return fresh, otherwise raw; no separate fit or outcome-based threshold choice'
    write(HERE / 'POLICY.json', policy)
    write(HERE / 'FREEZE.json', {'policySha256': sha(HERE / 'POLICY.json'), 'sourceSha256': sources()})
    print(json.dumps({'stage': 'frozen', 'cutoff': cutoff, 'policy': str(HERE / 'POLICY.json')}), flush=True)


def verify():
    freeze = json.loads((HERE / 'FREEZE.json').read_text(encoding='utf-8'))
    if freeze['policySha256'] != sha(HERE / 'POLICY.json'):
        raise RuntimeError('Forward policy changed')
    for relative, digest in freeze['sourceSha256'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Pinned source changed: ' + relative)
    return json.loads((HERE / 'POLICY.json').read_text(encoding='utf-8'))


def training_indices(policy):
    meta = pd.read_pickle(DATA / 'session_train_metadata.pkl').reset_index(drop=True)
    with np.load(DATA / 'session_train.npz', allow_pickle=False) as saved:
        bundle = {k: saved[k] for k in saved.files}
    if not np.array_equal(meta.arrayIndex, np.arange(len(meta))):
        raise RuntimeError('Training arrays and metadata disagree')
    cutoff = policy['trainingCutoffEpoch']
    eligible = (meta.issueEpoch.ge(cutoff - 14 * 86400) & meta.issueEpoch.lt(cutoff) &
                meta.completeEpoch.lt(cutoff) & np.isfinite(meta.actual) &
                meta.sequenceValid & bundle['valid'])
    if 'valid' in meta:
        eligible &= meta.valid
    rows = np.flatnonzero(eligible.to_numpy())
    days = int(meta.iloc[rows].targetDay.nunique())
    if len(rows) < 120 or days < 10:
        raise RuntimeError('Insufficient original completed daily training slice')
    audit = {'trainingCutoffEpoch': cutoff, 'trainingCutoffMYT': local(cutoff),
             'trainingRows': len(rows), 'distinctTrainingTargetDates': days,
             'trainingIssueMinEpoch': int(meta.iloc[rows].issueEpoch.min()),
             'trainingIssueMaxEpoch': int(meta.iloc[rows].issueEpoch.max()),
             'maximumTrainingCompleteEpoch': int(meta.iloc[rows].completeEpoch.max()),
             'trainingRowPositionsSha256': hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),
             'frozenSourceCutoffEpoch': json.loads((DATA / 'PREPARATION.json').read_text())['sessionCutoffEpoch'],
             'trainingTensorFileSha256': sha(DATA / 'session_train.npz')}
    return bundle, rows, audit


def capture(db_path=ROOT / 'bukit_kiara_air_history.db', api='http://127.0.0.1:8765'):
    """Return a materialized read-only input snapshot with actual receipt clocks."""
    requested = time.time()
    with urllib.request.urlopen(api + '/api/analysis?days=28', timeout=30) as response:
        analysis = json.load(response)
    api_received = time.time()
    publication = int(analysis['forecastIssuedEpoch'])
    mark = int(analysis['current']['epoch'])
    if mark > publication or publication > api_received or api_received - publication > 180:
        raise RuntimeError('API publication is stale or has invalid source clocks')
    opened = time.time()
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30)) as conn:
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        sensor = [dict(zip(['epoch', 'pm02', 'atmp', 'rhum'], r)) for r in conn.execute(
            'SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch',
            (mark - 25 * 3600, mark))]
        weather = [(int(f), str(s), decode_payload_text(p)) for f, s, p in conn.execute(
            'SELECT fetched_epoch,source,payload FROM weather_forecast_runs WHERE fetched_epoch<=? ORDER BY fetched_epoch DESC LIMIT 1', (opened,))]
        cams = [(int(f), str(s), decode_payload_text(p)) for f, s, p in conn.execute(
            "SELECT fetched_epoch,source,payload FROM air_quality_forecast_runs WHERE fetched_epoch<=? AND model_version IN ('cams_anchor_v1','cams_anchor_v2') AND source='Open-Meteo / CAMS Global' ORDER BY fetched_epoch DESC LIMIT 1", (opened,))]
    received = time.time()
    issue = math.ceil(received)
    if issue > time.time():
        time.sleep(issue - time.time())
    if not sensor or int(sensor[-1]['epoch']) != mark:
        raise RuntimeError('DB snapshot does not contain the published sensor watermark')
    if not weather or not 0 <= issue - weather[0][0] <= 7200:
        raise RuntimeError('Original weather unavailable or stale for this query')
    if not 0 <= issue - mark <= 240:
        raise RuntimeError('Published reference is older than240seconds; recapture next publication')
    assert mark <= publication <= api_received <= received <= issue
    assert all(r['epoch'] <= mark for r in sensor)
    assert all(f <= opened <= received <= issue for f, _, _ in weather + cams)
    return {'cutoffEpoch': issue, 'sensor': sensor, 'weather': sorted(weather), 'cams': sorted(cams),
            'sensorWatermarkEpoch': mark, 'apiRequestStartedEpoch': requested,
            'apiSnapshotReceivedEpoch': api_received, 'apiPublishedForecastIssueEpoch': publication,
            'dbSnapshotOpenedEpoch': opened, 'inputSnapshotReceivedEpoch': received,
            'issueEpoch': issue, 'readOnly': True, 'futureObservedSensorRowsLoaded': 0,
            'dashboardBuild': analysis.get('dashboardBuild'),
            'incumbentAtOriginalApiIssue': {k: (analysis.get('windows') or {}).get(k) for k in ('morning', 'afternoon')},
            'sourceSensorSha256': hashlib.sha256(json.dumps(sensor, sort_keys=True).encode()).hexdigest()}


def dated_target(issue):
    day = pd.Timestamp(issue, unit='s', tz='UTC').tz_convert('Asia/Kuala_Lumpur').normalize()
    start = int((day + pd.Timedelta(hours=14)).timestamp())
    if start - issue < 90 * 60:
        start += 86400
    if start - issue > 24 * 3600:
        raise RuntimeError('Next Afternoon exceeds original24hour target lead; recapture at eligible issue')
    return start, start + 7200


def query(snapshot, start=None, end=None):
    """Reuse original sequence and causal summary computations; labels unused."""
    issue, mark = int(snapshot['issueEpoch']), int(snapshot['sensorWatermarkEpoch'])
    if start is None:
        start, end = dated_target(issue)
    start, end = int(start), int(end)
    if start <= issue or end - start != 7200:
        raise ValueError('Need an exact future two-hour session')
    if any(r['epoch'] > mark for r in snapshot['sensor']) or mark > issue:
        raise ValueError('Future or unpublished sensor input')
    source = prepare.Sources(snapshot)
    raw = [(f, p) for f, _, p in snapshot['cams']]
    cams_digest = hashlib.sha256(json.dumps(raw).encode()).hexdigest()
    cams = regional._read_runs(ROOT / 'bukit_kiara_air_history.db', 'air_quality_forecast_runs', issue,
                               raw_snapshot=(raw, cams_digest))
    context_source = {'frame': source.frame, 'rows': snapshot['sensor'],
                      'weather': source.archive, 'cams': cams, 'cutoff': issue}
    context = sessions.query_features(context_source, issue, start, end)
    if context is None or not context.attrs['valid']:
        raise RuntimeError('Current exact causal context unavailable; no fabricated bucket')
    fetched = context.attrs['audit']['weatherFetchedEpoch']
    past, past_grid, maximum, count = source.past(issue, mark)
    future, future_grid, weather_hash, units = source.future(issue, start, end, fetched)
    ref_epoch = int(context.attrs['referenceEpoch'])
    origin = int(context.attrs['originEpoch'])
    basic_source = source.bucket_max.loc[source.bucket_max.index <= pd.Timestamp(origin, unit='s', tz='UTC').tz_convert('Asia/Kuala_Lumpur')].iloc[-12:]
    basic_max = float(basic_source.max())
    if count < 48 or not maximum <= mark <= issue or not basic_max <= mark:
        raise RuntimeError('Insufficient covered past history or source exceeds watermark')
    if not 0 <= issue - ref_epoch <= 240 or ref_epoch > mark:
        raise RuntimeError('Fresh context reference stale or unavailable')
    features = context.reindex(prepare.CONTEXT).to_numpy(dtype=np.float32)
    bundle = {'past': past[None, ...], 'future': future[None, ...], 'context': features[None, ...],
              'fresh': np.array([float(context['fresh'])]), 'valid': np.array([True]),
              'past_epochs': past_grid[None, ...], 'future_epochs': future_grid[None, ...]}
    audit = {'caseId': f'actual-forward-afternoon:{issue}:{start}:{end}',
             'issueEpoch': issue, 'issueMYT': local(issue), 'startEpoch': start, 'endEpoch': end,
             'targetStartMYT': local(start), 'targetEndMYT': local(end),
             'targetName': 'afternoon', 'targetDefinition': 'Overlap-duration weighted mean of covered15minPM2.5 medians over the exact dated14:00-16:00 session',
             'canonicalEvaluationClock': False, 'sensorWatermarkEpoch': mark,
             'sensorReferenceEpoch': ref_epoch, 'sensorReferenceCount': context.attrs['referenceCount'],
             'fresh': float(context['fresh']), 'closedLevel': float(context['closedLevel']),
             'featureSourceMaxEpoch': basic_max, 'sequenceSourceMaxEpoch': maximum,
             'pastPmValidSteps': count, 'weatherFetchedEpoch': fetched,
             'weatherPayloadSha256': weather_hash, 'camsFetchedEpoch': context.attrs['audit']['camsFetchedEpoch'],
             'inputSnapshotReceivedEpoch': snapshot['inputSnapshotReceivedEpoch'],
             'apiSnapshotReceivedEpoch': snapshot['apiSnapshotReceivedEpoch'],
             'apiPublishedForecastIssueEpoch': snapshot['apiPublishedForecastIssueEpoch'],
             'dbSnapshotOpenedEpoch': snapshot['dbSnapshotOpenedEpoch'],
             'futureOutcomeLoaded': False, 'futureKnownMetValues': int(np.isfinite(future[:, :12]).sum()),
             'tensorShapes': {k: list(bundle[k].shape) for k in ('past', 'future', 'context')},
             'contextFeatures': dict(zip(prepare.CONTEXT, features.astype(float))),
             'weatherUnits': units}
    return bundle, audit


def save_query(snapshot, bundle, audit, role):
    directory = HERE / 'queries' / f"{audit['issueEpoch']}_{role}"
    directory.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(safe(snapshot), separators=(',', ':'), allow_nan=False).encode()
    (directory / 'inputs.json.gz').write_bytes(gzip.compress(encoded, compresslevel=6, mtime=0))
    np.savez_compressed(directory / 'query.npz', **bundle)
    write(directory / 'QUERY.json', {**audit, 'inputsSha256': sha(directory / 'inputs.json.gz'),
                                   'tensorSha256': sha(directory / 'query.npz')})
    return directory


def setup():
    policy = verify()
    _, _, training = training_indices(policy)
    snapshot = capture()
    bundle, audit = query(snapshot)
    directory = save_query(snapshot, bundle, audit, 'software')
    write(HERE / 'SETUP.json', {'training': training, 'query': audit, 'queryPath': str(directory),
                             'softwareChecks': {'sourceWatermarkBounded': True, 'exactFutureTarget': True,
                                                'originalShapes': True, 'futureOutcomeNotLoaded': True},
                             'fitStarted': False, 'gpuGrantRequired': True})
    print(json.dumps({'stage': 'query_ready_no_fit', 'trainingRows': training['trainingRows'],
                      'pastPmSteps': audit['pastPmValidSteps'], 'target': audit['targetStartMYT'],
                      'query': str(directory)}), flush=True)


def fit():
    policy = verify()
    model_path = HERE / 'models' / f"{policy['trainingCutoffEpoch']}_patchtst.pt"
    info_path = HERE / 'FIT.json'
    if model_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text())
        if info['checkpointSha256'] != sha(model_path):
            raise RuntimeError('Saved forward artifact differs')
        print('ALREADY_FITTED', model_path, flush=True)
        return
    if model_path.exists() or info_path.exists():
        raise FileExistsError('Partial daily fit artifacts; do not overwrite')
    bundle, rows, training = training_indices(policy)
    started = time.time()
    model = trainer.SequenceRegressor('patchtst', device='cuda').fit(bundle, rows)
    model.save(model_path)
    write(info_path, {**training, 'fitStartedEpoch': started, 'fittedAtEpoch': time.time(),
                    'checkpoint': str(model_path.relative_to(HERE)), 'checkpointSha256': sha(model_path),
                    'policySha256': sha(HERE / 'POLICY.json'), 'learnerMetadata': model.training_metadata})
    print(json.dumps({'stage': 'fitted_cpu_artifact', 'rows': len(rows), 'checkpoint': str(model_path),
                      'seconds': model.training_metadata['trainingSeconds']}), flush=True)


def predict():
    policy = verify()
    fitted = json.loads((HERE / 'FIT.json').read_text(encoding='utf-8'))
    path = HERE / fitted['checkpoint']
    if fitted['checkpointSha256'] != sha(path):
        raise RuntimeError('Model hash mismatch')
    if cutoff_now() != policy['trainingCutoffEpoch']:
        raise RuntimeError('This original daily fit is for a different issue date')
    # Load model before capturing: inputs are recaptured after the real fit.
    model = trainer.SequenceRegressor.load(path)
    snapshot = capture()
    bundle, audit = query(snapshot)
    point = float(model.predict(bundle, np.array([0], dtype=np.int64))[0])
    generated = time.time()
    if not math.isfinite(point) or point < 0 or generated >= audit['startEpoch']:
        raise RuntimeError('Invalid or retrospectively generated forward forecast')
    directory = save_query(snapshot, bundle, audit, 'issued')
    record = {**audit, 'version': policy['version'], 'model': 'original_fixed_patchtst_weather_fusion',
              'forecastGeneratedEpoch': generated, 'recordedEpoch': time.time(),
              'predictedMeanUgM3': point, 'predictedChangeFromReferenceUgM3': point - audit['fresh'],
              'deadband20MeanUgM3': audit['fresh'] if abs(point - audit['fresh']) < 20 else point,
              'deadband20Applied': abs(point - audit['fresh']) < 20,
              'pairedPersistenceMeanUgM3': audit['fresh'], 'researchShadowOnly': True,
              'policySha256': sha(HERE / 'POLICY.json'), 'checkpointSha256': sha(path),
              'modelSourceHashes': trainer.source_hashes(), 'trainingCutoffEpoch': policy['trainingCutoffEpoch'],
              'trainingMetadataSha256': sha(HERE / 'FIT.json'),
              'inputSnapshotSha256': sha(directory / 'inputs.json.gz'),
              'queryTensorSha256': sha(directory / 'query.npz')}
    assert record['inputSnapshotReceivedEpoch'] <= record['issueEpoch'] <= record['forecastGeneratedEpoch'] < record['startEpoch']
    write(directory / 'FORECAST.json', record)
    write(HERE / 'issued' / (audit['caseId'].replace(':', '_') + '.json'), record)
    print(json.dumps({'stage': 'actual_forward_forecast_saved', 'issue': audit['issueMYT'],
                      'target': [audit['targetStartMYT'], audit['targetEndMYT']], 'mean': point,
                      'persistence': audit['fresh'], 'record': str(directory / 'FORECAST.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'setup', 'fit', 'predict'])
    args = parser.parse_args()
    {'freeze': freeze, 'setup': setup, 'fit': fit, 'predict': predict}[args.stage]()
