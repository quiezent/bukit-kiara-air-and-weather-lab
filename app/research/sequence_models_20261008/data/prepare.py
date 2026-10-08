"""Frozen, read-only sequence inputs; no model fitting or outcome scoring.

All scalar features, labels and genuine issued comparators are copied from
previous certified cohorts. This adds ordered past sensor and original-issued
future weather tensors without looking for new validation outcomes.
"""
from __future__ import annotations
from bisect import bisect_left, bisect_right
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import afternoon_direction_features as causal
import rain_weather_features as rain
from forecast_payload import decode_payload_text

SESSION = ROOT / 'research/daily/2026-10-07_weather_tests/extratrees'
NEAR = ROOT / 'research/autoencoder_20261007/near_term'
CONTEXT = list(causal.FEATURES)
PAST = ['pm02', 'atmp', 'rhum']
FUTURE = ['rainMmNativeHour', 'rainProbabilityPct', 'temperatureC', 'humidityPct',
          'wind10UKmh', 'wind10VKmh', 'wind925UKmh', 'wind925VKmh',
          'wind180UKmh', 'wind180VKmh', 'gustKmh', 'boundaryLayerM',
          'relativeLeadHours', 'targetIntervalOverlapFraction', 'targetPointProximity']
VERSION = 'sequence96_sensor3_future26_weather12_context31_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def freeze():
    policy = {'version': VERSION, 'frozenAtUtc': datetime.now(timezone.utc).isoformat(),
              'noFits': True, 'noOutcomeScoring': True, 'noNewQueryLabelsRead': True,
              'past': {'steps': 96, 'stepSeconds': 900, 'channels': PAST,
                       'anchor': 'Last fully closed 15-min boundary <=min(issue,published sensor watermark)',
                       'coverage': 'Each channel >=3 finite readings, both bucket edges <=240s, internal gaps<=480s; no filling',
                       'minimalValidPmSteps': 48, 'missing': 'NaN values plus explicit boolean masks',
                       'sourceGuard': 'Every used reading and bucket end <=published watermark<=issue; audit entire 96-step window'},
              'future': {'steps': 26, 'stepSeconds': 3600, 'channels': FUTURE,
                         'anchor': 'Exact issue + j hours for j=0..25',
                         'vintage': 'ONE latest original network fetch <=issue, <=7200s age; must equal previously certified weatherFetchedEpoch',
                         'rain': 'Native amount/probability on (hourEnd-3600,hourEnd]; row at grid instant belongs to its native right-closed hourly interval. Never interpolate or turn probability into event probability.',
                         'continuous': 'Adjacent hourly interpolation only, maximum bracket gap3600s; convert wind from-direction to signed transport U/V before interpolation',
                         'optional': '180m wind, gust and BLH require validated units; missing channels/extra-horizon steps remain NaN',
                         'targetOverlap': 'Fraction of (gridInstant-3600,gridInstant] overlapping exact mean target',
                         'pointProximity': 'For point targets max(0,1-abs(gridInstant-target)/3600), else0',
                         'CAPE': 'Excluded: varying historical availability'},
              'context': {'channels': CONTEXT, 'copy': 'Existing causal basic31 including guarded CAMS/reference/time, no recomputation'},
              'validity': 'Original valid row plus >=48 covered PM history steps and original issued-weather clock/coverage guards. Optional feature missingness does not invalidate.',
              'targets': 'Copy certified exact scalar labels; never relabel point90, (90,210] mean or dated two-hour sessions',
              'cohorts': 'Original valid session training, exactly certified69 session queries, original valid near training/queries; all rows retain IDs/clocks and extra sequence validity',
              'exposure': 'Previously inspected historical cohorts; reconstructed candidates, no independent new holdout claim',
              'limitations': ['Sensor first-receipt/revision histories absent; original published watermark is conservative availability guard.',
                             'Legacy weather network fetch retained; independent archive receipt absent.',
                             'Partial/gappy sensor histories are explicit missingness; no invented observations.',
                             'Hourly weather resolution remains hourly despite off-hour grid alignment.'],
              'sourceFiles': {str(p.relative_to(ROOT)): sha(p) for p in [
                  SESSION/'prepared/prepared.pkl', SESSION/'predictions.pkl',
                  SESSION/'prepared/source_digest.json', NEAR/'source_snapshot.json.gz',
                  NEAR/'training_features.csv', NEAR/'exact_issue_query_features.csv',
                  ROOT/'afternoon_direction_features.py', ROOT/'rain_weather_features.py',
                  ROOT/'collection_quality.py', ROOT/'window_pm_predictor.py', Path(__file__)]}}
    path = HERE/'POLICY.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        assert old['sourceFiles'] == policy['sourceFiles'], 'Preparation sources changed after freeze'
        return old
    write_json('POLICY.json', policy)
    return policy


def session_snapshot(cutoff, start):
    path = HERE/'session_source_snapshot.json.gz'
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    with closing(sqlite3.connect((ROOT/'bukit_kiara_air_history.db').as_uri()+'?mode=ro', uri=True, timeout=30)) as conn:
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        sensor = [dict(zip(['epoch', *PAST], row)) for row in conn.execute(
            'SELECT epoch,pm02,atmp,rhum FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch', (start-86400, cutoff))]
        weather = [(f, s, decode_payload_text(payload)) for f, s, payload in conn.execute(
            'SELECT fetched_epoch,source,payload FROM weather_forecast_runs WHERE fetched_epoch>=? AND fetched_epoch<=? ORDER BY fetched_epoch,payload', (start-7200, cutoff))]
    digest = json.loads((SESSION/'prepared/source_digest.json').read_text())
    sensor_digest = hashlib.sha256(json.dumps(sensor, sort_keys=True, default=str).encode()).hexdigest()
    wh = hashlib.sha256()
    for f, s, payload in weather:
        wh.update(repr((f,s)).encode()); wh.update(payload.encode('utf-8'))
    assert sensor_digest == digest['sensorRowsSha256'], 'Historical sensor source changed; do not silently reuse old labels'
    assert wh.hexdigest() == digest['weatherRevision'], 'Original issued weather source changed'
    value = {'cutoffEpoch': cutoff, 'sourceStartEpoch': start, 'sensor': sensor, 'weather': weather,
             'sensorRowsSha256': sensor_digest, 'weatherRevision': wh.hexdigest(), 'readOnly': True}
    payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return value


def numeric(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


class Sources:
    def __init__(self, snapshot):
        rows = snapshot['sensor']
        self.frame = causal.strict_frame(rows, snapshot['cutoffEpoch'])
        raw = pd.DataFrame(rows)
        raw.index = pd.to_datetime(raw.epoch, unit='s', utc=True).dt.tz_convert('Asia/Kuala_Lumpur')
        self.bucket_max = raw.epoch.resample('15min', label='right', closed='right').max()
        self.runs = tuple(rain._parse_run(float(f), str(s), payload) for f,s,payload in snapshot['weather'])
        self.archive = rain.RunArchive(self.runs, tuple(r.fetched_epoch for r in self.runs),
                                      snapshot['cutoffEpoch'], hashlib.sha256(json.dumps(snapshot['weather']).encode()).hexdigest(), 'frozen_sequence_source')
        self.parsed = {}
        self.past_cache = {}

    def past(self, issue, mark):
        anchor = int(min(issue,mark))//900*900
        key = (anchor,float(mark))
        if key in self.past_cache:
            return self.past_cache[key]
        grid = np.arange(anchor-95*900, anchor+1, 900, dtype=np.int64)
        ix = pd.to_datetime(grid, unit='s', utc=True).tz_convert('Asia/Kuala_Lumpur')
        source = self.bucket_max.reindex(ix).to_numpy(float)
        values = self.frame.reindex(ix)[PAST].to_numpy(dtype=np.float32)
        unprovable = ~np.isfinite(source) | (source>mark) | (grid>mark)
        values[unprovable,:] = np.nan
        used = np.isfinite(values).any(axis=1)
        maximum = float(source[used].max()) if used.any() else np.nan
        assert not used.any() or maximum<=mark<=issue
        result = (values, grid, maximum, int(np.isfinite(values[:,0]).sum()))
        self.past_cache[key] = result
        return result

    def parse(self, run):
        key = (run.fetched_epoch,run.payload_hash)
        if key in self.parsed:
            return self.parsed[key]
        data = json.loads(run.payload_json)
        units = data.get('hourlyUnits') or data.get('hourly_units') or {}
        points = {float(p['epoch']):p for p in data['hourly']}
        vals = {field:np.array([numeric(v) for v in value],dtype=float) for field,value in run.values.items()}
        allowed = {'wind_speed_180m': {'km/h'}, 'wind_direction_180m': {'°','degree','degrees'},
                   'wind_gusts_10m': {'km/h'}, 'boundary_layer_height': {'m'}}
        for field, expected in allowed.items():
            vals[field] = np.array([numeric(points[e].get(field)) if units.get(field) in expected else np.nan for e in run.epochs])
            if field.startswith('wind_direction'):
                vals[field][(vals[field]<0)|(vals[field]>360)] = np.nan
            else:
                vals[field][vals[field]<0] = np.nan
        for suffix, height in [('10m','10'),('925hPa','925'),('180m','180')]:
            speed, direction = vals['wind_speed_'+suffix], vals['wind_direction_'+suffix]
            vals['u'+height] = -speed*np.sin(np.deg2rad(direction))
            vals['v'+height] = -speed*np.cos(np.deg2rad(direction))
        result = (np.asarray(run.epochs), vals, units)
        self.parsed[key] = result
        return result

    @staticmethod
    def sample(epochs, values, instant, block=False):
        if block:
            pos = bisect_left(epochs, instant)
            return values[pos] if pos<len(epochs) and epochs[pos]-3600<instant<=epochs[pos] else np.nan
        right = bisect_left(epochs,instant)
        if right<len(epochs) and epochs[right]==instant:
            return values[right]
        if right==0 or right==len(epochs) or epochs[right]-epochs[right-1]>3600:
            return np.nan
        left = right-1
        if not np.isfinite(values[left:right+1]).all():
            return np.nan
        weight = (instant-epochs[left])/(epochs[right]-epochs[left])
        return values[left]+weight*(values[right]-values[left])

    def future(self, issue, start, end, expected_fetch):
        run = rain.latest_run(self.archive, issue)
        assert run is not None and run.fetched_epoch==expected_fetch and 0<=issue-run.fetched_epoch<=7200
        description = rain.describe_window(self.archive,issue,start,end)
        assert description['available'] and not description['metadata']['errors'] and not run.validation_errors
        epochs, vals, units = self.parse(run)
        fields = ['precipitation','precipitation_probability','temperature_2m','relative_humidity_2m',
                  'u10','v10','u925','v925','u180','v180','wind_gusts_10m','boundary_layer_height']
        grid = issue+np.arange(26,dtype=np.int64)*3600
        result = np.full((26,15),np.nan,dtype=np.float32)
        for j, instant in enumerate(grid):
            for k, field in enumerate(fields):
                result[j,k] = self.sample(epochs,vals[field],instant,block=k<2)
            result[j,12] = j
            result[j,13] = max(0,min(instant,end)-max(instant-3600,start))/3600 if end>start else 0
            result[j,14] = max(0,1-abs(instant-start)/3600) if end==start else 0
        return result, grid, run.payload_hash, units


def prepare_bundle(name, rows, sources):
    rows = rows.reset_index(drop=True).copy()
    rows['arrayIndex'] = np.arange(len(rows))
    if 'caseId' not in rows:
        rows['caseId'] = [f'{r.targetName}:{int(r.issueEpoch)}:{int(r.startEpoch)}:{int(r.endEpoch)}' for r in rows.itertuples()]
    past = np.empty((len(rows),96,3),dtype=np.float32)
    future = np.empty((len(rows),26,15),dtype=np.float32)
    pg = np.empty((len(rows),96),dtype=np.int64); fg = np.empty((len(rows),26),dtype=np.int64)
    audits = []
    unit_variants = {}
    for i,row in enumerate(rows.itertuples(index=False)):
        past[i],pg[i],smax,count = sources.past(int(row.issueEpoch),float(row.sensorWatermarkEpoch))
        future[i],fg[i],phash,units = sources.future(int(row.issueEpoch),int(row.startEpoch),int(row.endEpoch),float(row.weatherFetchedEpoch))
        unit_id = hashlib.sha256(json.dumps(units,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        unit_variants[unit_id] = units
        audits.append({'sequenceSourceMaxEpoch':smax,'pastPmValidSteps':count,'pastAnchorEpoch':int(pg[i,-1]),
                       'sequenceWeatherPayloadHash':phash,'sequenceWeatherUnitId':unit_id,
                       'sequenceValid':bool(count>=48),'futureKnownMetValues':int(np.isfinite(future[i,:,:12]).sum())})
    for column in audits[0] if audits else []:
        rows[column] = [a[column] for a in audits]
    context = rows[CONTEXT].to_numpy(dtype=np.float32)
    np.savez_compressed(HERE/f'{name}.npz',past=past,past_missing=~np.isfinite(past),future=future,future_missing=~np.isfinite(future),
                        context=context,context_missing=~np.isfinite(context),past_epochs=pg,future_epochs=fg,
                        y=rows.actual.to_numpy(dtype=np.float64),fresh=rows.fresh.to_numpy(dtype=np.float64),valid=rows.sequenceValid.to_numpy(bool))
    rows.to_pickle(HERE/f'{name}_metadata.pkl'); rows.to_csv(HERE/f'{name}_metadata.csv',index=False)
    return {'rows':len(rows),'sequenceValidRows':int(rows.sequenceValid.sum()),'pastShape':list(past.shape),'futureShape':list(future.shape),
            'contextShape':list(context.shape),'minimalPmSteps':int(rows.pastPmValidSteps.min()),
            'maxPmSteps':int(rows.pastPmValidSteps.max()),'futureChannelKnownValues':dict(zip(FUTURE,np.isfinite(future).sum(axis=(0,1)).astype(int).tolist())),
            'unitVariants':unit_variants,'npzSha256':sha(HERE/f'{name}.npz'),'metadataSha256':sha(HERE/f'{name}_metadata.pkl')}


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    policy = freeze()
    preparation = json.loads((SESSION/'prepared/preparation.json').read_text())
    protocol = json.loads((ROOT/'research/session_longterm_20261006/protocol.json').read_text())
    start = int(pd.Timestamp(protocol['sourceStart'],tz='Asia/Kuala_Lumpur').timestamp())
    ss = session_snapshot(preparation['sourceCutoffEpoch'],start)
    ns = json.loads(gzip.decompress((NEAR/'source_snapshot.json.gz').read_bytes()))
    session_sources,near_sources = Sources(ss),Sources(ns)
    prepared = pd.read_pickle(SESSION/'prepared/prepared.pkl')
    certified_queries = pd.read_pickle(SESSION/'predictions.pkl')
    assert len(certified_queries)==69 and certified_queries.valid.all()
    session_train = prepared.loc[prepared.role.eq('train')&prepared.valid].copy()
    near_train = pd.read_csv(NEAR/'training_features.csv'); near_queries = pd.read_csv(NEAR/'exact_issue_query_features.csv')
    for frame in [near_train,near_queries]:
        frame.rename(columns={'targetStartEpoch':'startEpoch','targetEndEpoch':'endEpoch'},inplace=True)
    output = {}
    for name, frame, sources in [('session_train',session_train,session_sources),('session_query',certified_queries,session_sources),
                                  ('near_train',near_train.loc[near_train.valid].copy(),near_sources),('near_query',near_queries.loc[near_queries.valid].copy(),near_sources)]:
        print('prepare',name,len(frame),flush=True)
        output[name] = prepare_bundle(name,frame,sources)
        print('saved',name,output[name]['sequenceValidRows'],flush=True)
    write_json('PREPARATION.json',{'version':VERSION,'preparedAtUtc':datetime.now(timezone.utc).isoformat(),'policySha256':sha(HERE/'POLICY.json'),
                                  'prepareScriptSha256':sha(Path(__file__)),'bundles':output,
                                  'sessionCutoffEpoch':ss['cutoffEpoch'],'nearCutoffEpoch':ns['cutoffEpoch'],
                                  'sessionSensorSha256':ss['sensorRowsSha256'],'sessionWeatherRevision':ss['weatherRevision'],
                                  'sessionSnapshotSha256':sha(HERE/'session_source_snapshot.json.gz'),
                                  'nearSnapshotSha256':sha(NEAR/'source_snapshot.json.gz'),
                                  'noFits':True,'noOutcomeScoring':True,'sourceRowsNotUsedBeyondOriginalPublishedWatermark':True})
    print('completed common frozen tensors without fitting or scoring',flush=True)


if __name__=='__main__':
    main()
