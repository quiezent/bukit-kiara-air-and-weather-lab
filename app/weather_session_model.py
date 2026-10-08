"""Causal pooled two-hour session features and fixed experimental regression.

Only this independent session model uses these enriched weather features. It
does not fetch weather, write archives, or modify the near-term estimators.
Fit targets are absolute covered session means; the estimator internally learns
their signed change from that issue's fresh five-minute sensor reference.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

import afternoon_direction_features as causal
import weather_session_features as weather_features

LEAD_MINUTES=(105,120,240,420,750,1080,1380)
LOOKBACK_DAYS=14
MIN_TRAINING_ROWS=120
MIN_TARGET_DAYS=10
FEATURE_VERSION='causal_pooled_session_weather_features_v1'
PARAMETERS={'max_iter':100,'max_depth':3,'max_leaf_nodes':7,
            'min_samples_leaf':80,'l2_regularization':50.,'learning_rate':.05,
            'loss':'absolute_error','early_stopping':False,'random_state':1749}
RAIN_INTERACTION_COLUMNS=('priorRainMm','duringRainMm','duringAccumulationMeanMm','prior3hRainMm')
COMPACT_ADDITIONS=('duringAccumulationMeanMm','prior3hRainMm',
                   'priorRainMaxHourlyMm','duringRainMaxHourlyMm',
                   'wind10UMeanKmh','wind10VMeanKmh','wind925UMeanKmh','wind925VMeanKmh',
                   'wind10UDeltaKmh','wind10VDeltaKmh','wind925UDeltaKmh','wind925VDeltaKmh',
                   'priorRainMmLevelInteraction','duringRainMmLevelInteraction')
ALL_COLUMNS=tuple(dict.fromkeys((*causal.FEATURES,
    *(c for c in weather_features.FEATURE_NAMES if c!='priorExposureMm'),
    *(c+'LevelInteraction' for c in RAIN_INTERACTION_COLUMNS),
    *(c+'Log' for c in RAIN_INTERACTION_COLUMNS))))
# Frozen after the development-period comparison, before held-out evaluation.
FEATURE_COLUMNS=tuple(causal.FEATURES)
MODEL_POLICY='hgb_basic_delta_14d'
NUMERICAL_POLICY='pooled_14d_daily_midnight_hgb_signed_change_basic31_v1'
LEARNER_NAME='100-iteration gradient-boosted signed session-change regression'
FEATURES=FEATURE_COLUMNS


def local_midnight(issue):
    value=pd.Timestamp(int(issue),unit='s',tz='UTC').tz_convert('Asia/Kuala_Lumpur')
    return int(value.normalize().timestamp())


def snapshot(db_path,rows,issue):
    """Return the bounded sensor frame and external archives known by issue."""
    issue=int(issue)
    rows,frame,cams,weather=causal.load(db_path,issue,rows)
    return {'rows':rows,'frame':frame,'cams':cams,'weather':weather,'cutoff':issue}


def _enrich(data,weather):
    x=data['x'].copy()
    descriptions=weather_features.describe_windows(weather,zip(data['issues'],data['start'],data['end']))
    extras=pd.DataFrame([r.get('featureValues') or {} for r in descriptions],index=x.index)
    for col in weather_features.FEATURE_NAMES:
        if col=='priorExposureMm':continue
        x[col]=pd.to_numeric(extras.reindex(columns=[col])[col],errors='coerce').to_numpy(float)
    for col in RAIN_INTERACTION_COLUMNS:
        x[col+'LevelInteraction']=x[col]*x.fresh/50.
        x[col+'Log']=np.log1p(x[col].clip(lower=0))
    # Core rain coverage defines usable weather. Optional wind/gust/mixing
    # channels can remain absent and are represented by fitted missingness.
    valid=np.asarray(data['valid'],dtype=bool)&np.array([r['available'] for r in descriptions],dtype=bool)
    audit=[]
    for previous,desc,issued in zip(data['audit'],descriptions,data['issues']):
        fetched=desc.get('fetchedEpoch')
        if fetched is not None and fetched>issued:
            raise ValueError('future weather vintage in session features')
        audit.append({**previous,'weatherFetchedEpoch':fetched,
                      'weatherFeatureVersion':weather_features.VERSION,
                      'weatherAvailable':bool(desc['available']),
                      'weatherMetadata':desc['metadata']})
    return {**data,'x':x.reindex(columns=ALL_COLUMNS),'valid':valid,
            'audit':audit,'weatherDescriptions':descriptions}


def build_dataset(db_path,rows,issue_epoch):
    """Build pooled historical rows; unfinished labels remain NaN and invalid.

    The wrapper should call this at local midnight and cache that frozen daily
    fit. The extra warmup day is used only for past local feature windows.
    Historical weather is selected independently at each actual issue clock.
    """
    issue_epoch=int(issue_epoch)
    source=snapshot(db_path,rows,issue_epoch)
    frame=source['frame']
    if frame.empty:
        return {**source,'x':pd.DataFrame(columns=ALL_COLUMNS),
                **{c:np.array([]) for c in ('labels','issues','complete','start','end')},
                'origins':pd.DatetimeIndex([]),'valid':np.array([],dtype=bool),'audit':[]}
    origins=frame.index[frame.index.minute.isin([0,30])]
    datasets=[]
    for lead in LEAD_MINUTES:
        part=causal.design(frame,source['rows'],source['cams'],source['weather'],issue_epoch,
                           lead,120,lag_seconds=0,only_origins=origins)
        datasets.append(_enrich(part,source['weather']))
    result={**source,'x':pd.concat([p['x'] for p in datasets]),
            'origins':pd.DatetimeIndex(np.concatenate([p['origins'] for p in datasets])),
            'audit':[a for p in datasets for a in p['audit']]}
    for col in ('labels','issues','complete','start','end','valid'):
        result[col]=np.concatenate([p[col] for p in datasets])
    completed=np.isfinite(result['labels'])&(result['complete']<issue_epoch)
    within=(result['issues']>=issue_epoch-LOOKBACK_DAYS*86400)&(result['issues']<issue_epoch)
    result['valid'] &= completed&within
    result['featureVersion']=FEATURE_VERSION
    return result


def query_features(source,issue,start,end):
    """Return one exact issued session row from a fresh independent snapshot."""
    issue,start,end=int(issue),int(start),int(end)
    origin=issue//900*900
    if (end-start!=7200 or not 90*60<=start-origin<=1440*60
            or (start-origin)%900 or start<=issue):
        return None
    frame=source['frame']
    if frame.empty:return None
    stamp=pd.Timestamp(origin,unit='s',tz='UTC').tz_convert('Asia/Kuala_Lumpur')
    if stamp not in frame.index:return None
    data=causal.design(frame,source['rows'],source['cams'],source['weather'],issue,
                       (start-issue)/60,120,lag_seconds=issue-origin,
                       only_origins=pd.DatetimeIndex([stamp]))
    data=_enrich(data,source['weather'])
    fetched=data['audit'][0]['weatherFetchedEpoch']
    if fetched is not None and fetched>issue:
        raise ValueError('future weather vintage in query')
    query=data['x'].iloc[0].copy()
    query.attrs.update(valid=bool(data['valid'][0]),audit=data['audit'][0],
                       weatherDescription=data['weatherDescriptions'][0],
                       issueEpoch=issue,startEpoch=start,endEpoch=end,
                       originEpoch=origin,referenceEpoch=int(data['stamps'][0]),
                       referenceCount=int(data['counts'][0]))
    return query


def fit_model(train_x,absolute_y):
    """Fit the fixed signed-change model from absolute completed target means."""
    x=train_x.reindex(columns=FEATURE_COLUMNS).apply(pd.to_numeric,errors='coerce').to_numpy(float)
    x[~np.isfinite(x)]=np.nan
    y=np.asarray(absolute_y,dtype=float)
    if len(x)<MIN_TRAINING_ROWS or len(y)!=len(x) or not np.isfinite(y).all():
        return None
    anchor=x[:,FEATURE_COLUMNS.index('fresh')]
    if not np.isfinite(anchor).all():return None
    estimator=make_pipeline(SimpleImputer(add_indicator=True),HistGradientBoostingRegressor(**PARAMETERS))
    with threadpool_limits(limits=1):estimator.fit(x,y-anchor)
    return estimator


def predict_model(estimator,query_x):
    """Return the absolute session point, or None when no finite reference exists."""
    if isinstance(query_x,pd.Series):query_x=query_x.to_frame().T
    if estimator is None or len(query_x)!=1:return None
    x=query_x.reindex(columns=FEATURE_COLUMNS).apply(pd.to_numeric,errors='coerce').to_numpy(float)
    x[~np.isfinite(x)]=np.nan
    anchor=x[0,FEATURE_COLUMNS.index('fresh')]
    if not np.isfinite(anchor):return None
    with threadpool_limits(limits=1):delta=float(estimator.predict(x)[0])
    if not math.isfinite(delta):return None
    result=max(0.,float(anchor)+delta)
    return result if math.isfinite(result) else None
