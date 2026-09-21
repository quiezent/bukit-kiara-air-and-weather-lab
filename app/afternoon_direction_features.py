"""Frozen experimental-afternoon feature calculations from the model challenge.

No model may read labels outside the completed training slice handed to it.
Weather and CAMS are forecast vintages, not observations or later reanalysis.
"""
from bisect import bisect_right
import math
import numpy as np
import pandas as pd
import window_pm_predictor as production
import rain_weather_features as rain
from collection_quality import bucket_coverage
from forecast_freshness import references
from forecast_clock import point_weights, window_weights

FEATURE_VERSION='causal_challenge_features_v1'
WEATHER=('priorRainMm','duringRainMm','priorRainProbMean','duringRainProbMean',
         'firstWetLeadHours','hoursUntilExpectedWetEnd','wetTempMinDeltaC',
         'targetTempDeltaC','targetRhDeltaPct','targetWind10DeltaKmh','targetWind925MeanKmh')
FEATURES=('fresh','closedLevel','freshOffset','delta15','delta30','delta60','delta120',
          'sd60','mean3Offset','temp','rh','tempDelta30','rhDelta30','sinTarget','cosTarget',
          'camsCurrent','camsTarget','camsDelta',*WEATHER,'targetLeadHours','targetDurationHours')

def load(db,cutoff,rows):
    """Use at most 15 days of supplied sensor rows and validated as-of archives.

    The extra day supplies rolling-feature warmup before the 14-day fit slice.
    The rain archive is the same validated object attached by the shared
    challenge loader; unused legacy weather arrays need not be parsed here.
    No telemetry writes, collector calls or historical model replay occur.
    """
    bounded=[]
    for source in rows:
        row=dict(source)
        try:epoch=float(row.get('epoch'))
        except (TypeError,ValueError,OverflowError):continue
        if math.isfinite(epoch) and cutoff-15*86400<=epoch<=cutoff:
            bounded.append({'epoch':epoch,**{name:row.get(name) for name in ('pm02','atmp','rhum')}})
    rows=sorted(bounded,key=lambda row:row['epoch'])
    frame=strict_frame(rows,cutoff)
    cams=production._read_runs(db,'air_quality_forecast_runs',cutoff)
    weather=rain.load_runs(db,cutoff)
    return rows,frame,cams,weather

def strict_frame(rows,cutoff):
    frame=production._frame(rows,cutoff)
    if frame.empty:return frame
    raw=pd.DataFrame([r for r in rows if r['epoch']<=cutoff])
    raw.index=pd.to_datetime(raw.pop('epoch'),unit='s',utc=True).dt.tz_convert('Asia/Kuala_Lumpur')
    raw=raw.sort_index().apply(pd.to_numeric,errors='coerce')
    for col in ('atmp','rhum'):
        eligible=bucket_coverage(raw,column=col).forecastEligible
        frame.loc[~eligible.reindex(frame.index,fill_value=False),col]=np.nan
    return frame

def base_features(frame):
    pm=frame.pm02
    x=pd.DataFrame(index=frame.index)
    x['closedLevel']=pm
    for n,m in ((1,15),(2,30),(4,60),(8,120)):
        complete=pm.notna().rolling(n+1,min_periods=n+1).sum().eq(n+1)
        x[f'delta{m}']=pm.diff(n).where(complete)
    x['sd60']=pm.rolling(4,min_periods=4).std(ddof=0)
    x['mean3Offset']=pm.rolling(12,min_periods=12).mean()-pm
    for src,name in (('atmp','temp'),('rhum','rh')):
        x[name]=frame[src]
        complete=frame[src].notna().rolling(3,min_periods=3).sum().eq(3)
        x[name+'Delta30']=frame[src].diff(2).where(complete)
    return x

def regional(cams,issued,start,end):
    ix=bisect_right([r[0] for r in cams],issued)-1
    if ix<0:return np.nan,np.nan,None
    fetched,run=cams[ix]
    if issued-fetched>7200 or 'pm2_5' not in run:return np.nan,np.nan,fetched
    grid=run['pm2_5'][0]
    times=np.unique(np.r_[start,grid[(grid>start)&(grid<end)],end])
    current=production._sample(run,np.array([issued]),'pm2_5',max_gap_seconds=10800)[0]
    vals=production._sample(run,times,'pm2_5',max_gap_seconds=10800)
    if not np.isfinite(current) or not np.isfinite(vals).all():return np.nan,np.nan,fetched
    target=vals[0] if start==end else np.trapz(vals,times)/(end-start)
    return float(current),float(target),fetched

def design(frame,rows,cams,weather,cutoff,lead_minutes,duration_minutes,lag_seconds=300,origin_minute=0,extra_origins=None,only_origins=None):
    origins=(frame.index[frame.index.minute==origin_minute] if only_origins is None
             else pd.DatetimeIndex(only_origins))
    if extra_origins is not None:origins=origins.union(extra_origins).sort_values()
    epochs=origins.asi8//10**9
    issues=epochs+lag_seconds
    fresh,stamps,counts=references(rows,epochs,lag_seconds,cutoff)
    x=base_features(frame).reindex(origins)
    x['fresh']=fresh;x['freshOffset']=fresh-x.closedLevel
    start=issues+lead_minutes*60;end=start+duration_minutes*60
    middle=pd.to_datetime((start+end)/2,unit='s',utc=True).tz_convert('Asia/Kuala_Lumpur')
    hour=middle.hour+middle.minute/60+middle.second/3600
    x['sinTarget']=np.sin(hour*np.pi/12);x['cosTarget']=np.cos(hour*np.pi/12)
    x['targetLeadHours']=lead_minutes/60;x['targetDurationHours']=duration_minutes/60
    external=[];audit=[]
    archive=getattr(weather,'rain_archive',weather)
    for issued,a,b in zip(issues,start,end):
        current,target,cf=regional(cams,int(issued),int(a),int(b))
        desc=rain.describe_window(archive,int(issued),int(a),int(b))
        values=desc['featureValues'];errors=desc.get('metadata',{}).get('errors',[])
        record={key:(np.nan if errors else values.get(key,np.nan)) for key in WEATHER}
        record.update(camsCurrent=current,camsTarget=target,camsDelta=target-current)
        external.append(record)
        wf=desc.get('fetchedEpoch')
        assert cf is None or cf<=issued
        assert wf is None or wf<=issued
        audit.append({'camsFetchedEpoch':cf,'weatherFetchedEpoch':wf,'weatherErrors':errors})
    x=x.join(pd.DataFrame(external,index=origins,dtype=float)).reindex(columns=FEATURES)
    weights=(point_weights(lag_seconds+lead_minutes*60) if not duration_minutes else
             window_weights(lag_seconds+lead_minutes*60,lag_seconds+(lead_minutes+duration_minutes)*60))
    parts=pd.concat({n:frame.pm02.shift(-n) for n in weights},axis=1)
    labels=parts.mul(pd.Series(weights)).sum(axis=1,min_count=len(weights)).reindex(origins).to_numpy(float)
    complete=epochs+max(weights)*900
    labels[complete>cutoff]=np.nan
    return {'x':x,'labels':labels,'issues':issues,'complete':complete,'origins':origins,
            'start':start,'end':end,'stamps':stamps,'counts':counts,'audit':audit,
            'valid':np.isfinite(fresh)&np.isfinite(x.closedLevel.to_numpy(float))}
