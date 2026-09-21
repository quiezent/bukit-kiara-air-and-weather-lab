"""Experimental, horizon-specific TTDI PM2.5 session-mean prediction.

Read-only prediction seam: predict_windows(db_path, rows, windows, issue_epoch).
No collectors, web calls, public ontology or coaching decisions. Future sensor
targets never enter a fit or adaptive weights before their complete end time.
Meteorological and CAMS inputs must have fetched_epoch <= each actual issue.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import closing
import copy
import hashlib
import json
import math
import sqlite3
import threading
import time
import numpy as np
import pandas as pd
from collection_quality import bucket_coverage
from forecast_freshness import refresh_session_records, raw_arrays, VERSION as FRESHNESS_VERSION, REFERENCE_WINDOW_SECONDS
import rain_weather_features as rain_features

MODEL_VERSION="local_session_adaptive_v5_fresh_component_scoring"
RAIN_MODEL_VERSION="local_session_adaptive_v5_rain_fresh_component_scoring_experimental"
WEIGHT_POLICY_VERSION="fresh_component_mae_completed_only_v1"
BUCKET_SECONDS=900
MAX_LEAD_MINUTES=1440
COMPONENTS=("persistence","half_ridge_local","half_ridge_weather","half_cams","mean_revert")
LOCAL_COLS=("level","delta30","delta60","delta120","mean3Offset","mean24Offset","sd60","tempDelta30","rhDelta30","sinTarget","cosTarget")
WEATHER_COLS=("camsDelta","camsLevel","windU","windV","weatherTempDelta","weatherRain")
# Declared before the paired evaluation. Rain probability describes weather,
# never probability of PM removal. All coefficients, including signs, are fit.
RAIN_COLS=("priorRainMm","duringRainMm","priorRainProbMean","priorRainProbMax",
           "duringRainProbMean","duringRainProbMax","priorWetSignalHours","duringWetSignalHours",
           "firstWetLeadHours","hoursUntilExpectedWetEnd","wetTempMinDeltaC",
           "priorAmountLevelInteraction","duringAmountLevelInteraction")
RAIN_RIDGE_PENALTY=200.
RAIN_INTERACTION_PENALTY=400.
_WEATHER_ARRAY_CACHE=OrderedDict()
_CACHE=OrderedDict()
_FIT_CACHE=OrderedDict()
_LOCK=threading.RLock()
_RUN_CACHE=OrderedDict()
_RUN_LOCK=threading.RLock()


class WeatherRuns(list):
    """Keep the legacy list-of-(fetched,arrays) API used by CAMS companions."""
    rain_archive=None


def _raw_runs(db_path,table,issue_epoch):
    """Read and hash actual issued payloads, including same-count corrections."""
    if table not in ("air_quality_forecast_runs","weather_forecast_runs"):
        raise ValueError("unsupported archive table")
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro",uri=True,timeout=20)) as conn:
        version_filter=" AND model_version IN ('cams_anchor_v1','cams_anchor_v2')" if table=="air_quality_forecast_runs" else ""
        raw=conn.execute(f"SELECT fetched_epoch,payload,source FROM {table} WHERE fetched_epoch<=?{version_filter} ORDER BY fetched_epoch,payload",(issue_epoch,)).fetchall()
    digest=hashlib.blake2b(digest_size=16)
    rows=[]
    for fetched,payload,source in raw:
        digest.update(repr(fetched).encode("ascii"));digest.update(b"\0")
        digest.update(payload.encode("utf-8"));digest.update(b"\0")
        digest.update(str(source).encode("utf-8"));digest.update(b"\0")
        # Keep source corrections in the revision hash, even when rejected.
        if table=="air_quality_forecast_runs" and source!="Open-Meteo / CAMS Global":continue
        rows.append((fetched,payload))
    return rows,digest.hexdigest()


def _frame(rows,issue_epoch):
    records=[]
    for row in rows:
        r=dict(row)
        try:epoch=float(r.get("epoch"))
        except (TypeError,ValueError,OverflowError):continue
        if math.isfinite(epoch) and epoch<=issue_epoch:
            records.append({"epoch":epoch,**{c:r.get(c) for c in ("pm02","atmp","rhum")}})
    if not records:return pd.DataFrame()
    f=pd.DataFrame(records)
    for col in f:f[col]=pd.to_numeric(f[col],errors="coerce")
    f.index=pd.to_datetime(f.pop("epoch"),unit="s",utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
    f=f.sort_index()
    # A minimum count alone accepts interrupted edge buckets around laptop
    # sleep. Both edges must also be sampled within four minutes. This is known
    # at bucket close; no later gap endpoint can rewrite an earlier feature.
    coverage=bucket_coverage(f)
    grouped=f.resample("15min",label="right",closed="right")
    f=grouped.median()
    f.loc[~coverage.forecastEligible.reindex(f.index,fill_value=False),"pm02"]=np.nan
    cutoff=pd.Timestamp(issue_epoch,unit="s",tz="UTC").tz_convert("Asia/Kuala_Lumpur").floor("15min")
    f=f.loc[f.index<=cutoff]
    f.attrs["targetCoveragePolicy"]="eight_closed_15_minute_buckets_each_with_at_least_three_finite_samples_both_edges_within_240_seconds_internal_gaps_at_most_480_seconds"
    return f


def _read_runs(db_path,table,issue_epoch,*,raw_snapshot=None):
    result=WeatherRuns() if table=="weather_forecast_runs" else []
    # Read-only SQLite connection: this module never changes the telemetry DB.
    # A caller may reuse the exact rows and content hash already read for this
    # table/cutoff. Standalone callers still reread bytes to detect corrections.
    rows,revision=_raw_runs(db_path,table,issue_epoch) if raw_snapshot is None else raw_snapshot
    # The payload set can be unchanged across adjacent issue times, but the
    # attached rain archive must retain this call's exact availability cutoff.
    cache_key=(str(db_path),table,revision,float(issue_epoch)) if table=="weather_forecast_runs" else (str(db_path),table,revision)
    with _RUN_LOCK:
        if cache_key in _RUN_CACHE:
            _RUN_CACHE.move_to_end(cache_key)
            return _RUN_CACHE[cache_key]
        for fetched,payload in rows:
            try:
                data=json.loads(payload)
                source=str(data.get("source", ""))
                if table=="air_quality_forecast_runs" and source!="Open-Meteo / CAMS Global":continue
                if table=="weather_forecast_runs" and source!="Open-Meteo Best Match":continue
                units=data.get("hourlyUnits") or {}
                if table=="air_quality_forecast_runs":
                    fetched_value=float(fetched)
                    if not math.isfinite(fetched_value):continue
                    # Older payloads may omit this field; a present timestamp
                    # must agree with the archive's as-of selection clock.
                    if "fetchedEpoch" in data:
                        payload_fetched=float(data["fetchedEpoch"])
                        if not math.isfinite(payload_fetched) or payload_fetched!=fetched_value:continue
                    unit=str(units.get("pm2_5", "")).replace("\u03bc","u").replace("\u00b5","u").replace("\u00b3","3")
                    if unit!="ug/m3":continue
                if "wind_speed_925hPa" in units and units["wind_speed_925hPa"]!="km/h":continue
                if "latitude" in data and "longitude" in data:
                    if abs(float(data["latitude"])-3.1411)>.3 or abs(float(data["longitude"])-101.6275)>.3:continue
                points={int(v["epoch"]):v for v in data.get("hourly",[]) if v.get("epoch") is not None}
                if not points:continue
                epochs=np.array(sorted(points),dtype=float)
                fields=("pm2_5",) if table=="air_quality_forecast_runs" else ("temperature_2m","wind_speed_925hPa","wind_direction_925hPa","precipitation")
                values={c:np.array([float(points[int(e)][c]) if points[int(e)].get(c) is not None else np.nan for e in epochs]) for c in fields}
                if "wind_speed_925hPa" in values:
                    speed=values["wind_speed_925hPa"];direction=values["wind_direction_925hPa"]
                    values["windU925"]=-speed*np.sin(np.deg2rad(direction))
                    values["windV925"]=-speed*np.cos(np.deg2rad(direction))
                arrays={c:(epochs[np.isfinite(v)],v[np.isfinite(v)]) for c,v in values.items() if np.isfinite(v).any()}
                result.append((float(fetched),arrays))
            except (ValueError,TypeError,KeyError,OverflowError):continue
        if table=="weather_forecast_runs":
            result.rain_archive=rain_features.load_runs(db_path,issue_epoch)
        _RUN_CACHE[cache_key]=result
        while len(_RUN_CACHE)>6:_RUN_CACHE.popitem(last=False)
    return result


def _sample(run,times,column,max_gap_seconds=3600):
    if run is None or column not in run:return np.full(len(times),np.nan)
    x,y=run[column]
    values=np.interp(times,x,y,left=np.nan,right=np.nan)
    left=np.searchsorted(x,times,side="right")-1
    right=np.searchsorted(x,times,side="left")
    valid=(left>=0)&(right<len(x))
    bounded=np.where(valid)[0]
    values[bounded[(x[right[bounded]]-x[left[bounded]])>max_gap_seconds]]=np.nan
    return values


def _validated_weather_arrays(archive,issued):
    selected=rain_features.latest_run(archive,issued)
    if selected is None:return None,None
    if selected.validation_errors or issued-selected.fetched_epoch>7200:
        return None,selected.fetched_epoch
    key=(selected.fetched_epoch,selected.payload_hash)
    if key in _WEATHER_ARRAY_CACHE:
        _WEATHER_ARRAY_CACHE.move_to_end(key)
        return _WEATHER_ARRAY_CACHE[key],selected.fetched_epoch
    epochs=np.asarray(selected.epochs,dtype=float)
    values={c:np.array(v,dtype=float) for c,v in selected.values.items()}
    speed=values["wind_speed_925hPa"];direction=values["wind_direction_925hPa"]
    values["windU925"]=-speed*np.sin(np.deg2rad(direction))
    values["windV925"]=-speed*np.cos(np.deg2rad(direction))
    arrays={c:(epochs[np.isfinite(v)],v[np.isfinite(v)]) for c,v in values.items() if np.isfinite(v).any()}
    _WEATHER_ARRAY_CACHE[key]=arrays
    while len(_WEATHER_ARRAY_CACHE)>2048:_WEATHER_ARRAY_CACHE.popitem(last=False)
    return arrays,selected.fetched_epoch


def _archive_features(origins,lead_minutes,cams_runs,weather_runs,issue_lag_seconds=0,use_shared_source=False):
    records=[]
    ce=np.array([r[0] for r in cams_runs]);we=np.array([r[0] for r in weather_runs])
    for origin in origins:
        epoch=int(origin.timestamp())
        issued=epoch+float(issue_lag_seconds)
        ci=int(np.searchsorted(ce,issued,side="right"))-1
        wi=int(np.searchsorted(we,issued,side="right"))-1
        cams=cams_runs[ci][1] if ci>=0 and issued-ce[ci]<=2*3600 else None
        weather=weather_runs[wi][1] if wi>=0 and issued-we[wi]<=2*3600 else None
        weather_fetched=int(we[wi]) if weather is not None else None
        if use_shared_source:
            weather,weather_fetched=_validated_weather_arrays(weather_runs.rain_archive,issued)
        centres=epoch+lead_minutes*60+np.arange(8)*900+450
        pm=_sample(cams,np.r_[epoch,centres],"pm2_5",max_gap_seconds=10800)
        temp=_sample(weather,np.r_[epoch,centres],"temperature_2m")
        wind_u=_sample(weather,centres,"windU925")
        wind_v=_sample(weather,centres,"windV925")
        rain=_sample(weather,centres,"precipitation")
        mean=lambda x:float(np.mean(x)) if np.isfinite(x).all() else np.nan
        records.append({"camsDelta":mean(pm[1:])-pm[0],"camsLevel":pm[0],
                        "windU":mean(wind_u),"windV":mean(wind_v),
                        "weatherTempDelta":mean(temp[1:])-temp[0],"weatherRain":mean(rain),
                        "camsFetchedEpoch":int(ce[ci]) if cams is not None else None,
                        "weatherFetchedEpoch":weather_fetched})
    return pd.DataFrame(records,index=origins)


def _rain_frame(origins,lead_minutes,archive,issue_lag_seconds=0):
    """One shared description per exact actual-issue/session clock."""
    result=[];descriptions=[]
    for origin in origins:
        epoch=int(origin.timestamp())
        description=rain_features.describe_window(archive,epoch+float(issue_lag_seconds),
                                                  epoch+lead_minutes*60,epoch+(lead_minutes+120)*60)
        descriptions.append(description)
        values=description.get("featureValues") or {}
        if (description.get("metadata",{}).get("wetTiming",{}).get("status")=="no_wet_signal"):
            values={**values,"firstWetLeadHours":0.,"hoursUntilExpectedWetEnd":0.,"wetTempMinDeltaC":0.}
        result.append({c:values.get(c) for c in RAIN_COLS if not c.endswith("Interaction")})
    return pd.DataFrame(result,index=origins,dtype=float),descriptions


def _features(frame,lead_minutes):
    pm=frame.pm02;f=pd.DataFrame(index=frame.index);f["level"]=pm
    for count,minutes in ((2,30),(4,60),(8,120)):
        f[f"delta{minutes}"]=pm.diff(count).where(pm.notna().rolling(count+1).sum()==count+1)
    f["mean3Offset"]=pm.rolling(12,min_periods=9).mean()-pm
    f["mean24Offset"]=pm.rolling(96,min_periods=72).mean()-pm
    f["sd60"]=pm.rolling(4,min_periods=3).std()
    f["tempDelta30"]=frame.atmp.diff(2);f["rhDelta30"]=frame.rhum.diff(2)
    hour=frame.index.hour+(frame.index.minute+lead_minutes+60)/60
    f["sinTarget"]=np.sin(2*np.pi*hour/24);f["cosTarget"]=np.cos(2*np.pi*hour/24)
    return f


def _targets(frame,lead_minutes):
    lead=lead_minutes//15
    target_pm=frame.pm02
    values=pd.concat([target_pm.shift(-(lead+k)) for k in range(1,9)],axis=1)
    return values.mean(axis=1).where(values.notna().all(axis=1))


def _column_quartiles(x):
    """Vectorized linear nan-quartiles; equivalent to numpy's default method.

    Sorting all feature columns together avoids numpy.nanquantile's per-column
    Python dispatch in each small rolling Ridge fit. Only completed training
    rows enter x; this optimization cannot change the training cutoff.
    """
    finite=np.isfinite(x)
    ordered=np.sort(np.where(finite,x,np.nan),axis=0)
    count=finite.sum(axis=0)
    ranks=(count[None,:]-1)*np.array([.25,.5,.75])[:,None]
    lower=np.floor(ranks).astype(int);upper=np.ceil(ranks).astype(int)
    cols=np.arange(x.shape[1])[None,:]
    a=ordered[lower,cols];b=ordered[upper,cols]
    return a+(b-a)*(ranks-lower)


def _ridge(x,y,q,penalties=None):
    usable=np.isfinite(q)&(np.isfinite(x).mean(axis=0)>=.6)
    x=x[:,usable];q=q[usable]
    if x.shape[1]<5:return None
    lower,med,upper=_column_quartiles(x)
    scale=upper-lower
    scale=np.where(scale>1e-5,scale,1.)
    x=np.clip((np.where(np.isfinite(x),x,med)-med)/scale,-4,4)
    q=np.clip((q-med)/scale,-4,4)
    x=np.column_stack([np.ones(len(x)),x]);q=np.r_[1.,q]
    if penalties is None:penalties=np.full(len(usable),50.)
    penalty=np.diag(np.r_[0.,np.asarray(penalties)[usable]])
    try:coef=np.linalg.solve(x.T@x+penalty,x.T@y)
    except np.linalg.LinAlgError:coef=np.linalg.lstsq(x.T@x+penalty,x.T@y,rcond=None)[0]
    return float(q@coef)


def replay(frame,lead_minutes,cams_runs,weather_runs,include_live=True,issue_lag_seconds=0,
           alignment_only=False,training_cutoff_epoch=None,prior_records=None,corrected_weather_only=False,
           shared_source_alignment=False):
    """Prequential fits with weather known by origin plus the exact issue lag.

    These component fits and provisional closed-reference weights are independent
    of raw five-minute references. predict_windows re-scores weights and refreshes
    the point with _fresh_weighted_records using raw readings and the same lag.
    A fixed optional training cutoff supports blocked-date evaluation; the actual
    target values remain available to score.
    """
    if not 0<=issue_lag_seconds<900:raise ValueError("issue lag must lie in [0,900)")
    if shared_source_alignment:
        alignment_only=False
        corrected_weather_only=True
    features=_features(frame,lead_minutes)
    origins=features.index[features.index.minute.isin([0,30])]
    if include_live:origins=origins.union(frame.index[-1:]).sort_values()
    f=features.loc[origins].join(_archive_features(origins,lead_minutes,cams_runs,weather_runs,issue_lag_seconds,
                                                  use_shared_source=not alignment_only))
    rain_archive=getattr(weather_runs,"rain_archive",None)
    descriptions=[None]*len(origins)
    weather_cols=WEATHER_COLS
    penalties=None
    if not alignment_only:
        if rain_archive is None:raise ValueError("rain replay requires weather runs returned by this module's _read_runs")
        rf,descriptions=_rain_frame(origins,lead_minutes,rain_archive,issue_lag_seconds)
        f=f.join(rf)
        # Previous weatherRain sampled hourly sums as if they were point rates.
        # The shared overlap integral yields true target-window mm; /2 is mean
        # mm/hour for this retained original weather column.
        f["weatherRain"]=f["duringRainMm"]/2.
        f["priorAmountLevelInteraction"]=f["priorRainMm"]*f.level/50.
        f["duringAmountLevelInteraction"]=f["duringRainMm"]*f.level/50.
        if not corrected_weather_only:
            weather_cols=WEATHER_COLS+RAIN_COLS
            penalties=np.r_[np.full(len(LOCAL_COLS+WEATHER_COLS),50.),
                            np.full(len(RAIN_COLS)-2,RAIN_RIDGE_PENALTY),
                            np.full(2,RAIN_INTERACTION_PENALTY)]
    target=_targets(frame,lead_minutes).reindex(origins)
    ends=origins+pd.Timedelta(minutes=lead_minutes+120)
    xl=f[list(LOCAL_COLS)].to_numpy(float);xw=f[list(LOCAL_COLS+weather_cols)].to_numpy(float)
    weather_usable=np.isfinite(f[list(WEATHER_COLS[2:])].to_numpy(float)).any(axis=1)
    y=(target-f.level).to_numpy(float);records=list(prior_records or [])
    prior_last=records[-1]["originEpoch"] if records else -1
    for i,origin in enumerate(origins):
        epoch=int(origin.timestamp());issued_epoch=epoch+float(issue_lag_seconds)
        if epoch<=prior_last:continue
        if origin<origins.min()+pd.Timedelta(days=3) or not np.isfinite(f.level.iloc[i]):continue
        # Forecast issuance must not depend on whether future observations will
        # be collected. Missing/unfinished targets remain None and never score.
        known_cutoff=issued_epoch if training_cutoff_epoch is None else min(issued_epoch,int(training_cutoff_epoch))
        mask=(ends.asi8//10**9<known_cutoff)&(origins>=origin-pd.Timedelta(days=7))&np.isfinite(y)&np.isfinite(f.level)
        if mask.sum()<48:continue
        anchor=float(f.level.iloc[i]);local=_ridge(xl[mask],y[mask],xl[i]);weather=_ridge(xw[mask],y[mask],xw[i],penalties)
        cams_delta=float(f.camsDelta.iloc[i]);revert=float(np.nan_to_num(f.mean24Offset.iloc[i]))*min(.5,(lead_minutes/60+1)/24)
        candidates={"persistence":anchor,"half_ridge_local":max(0,anchor+.5*local) if local is not None else anchor,
                    "half_ridge_weather":max(0,anchor+.5*weather) if weather is not None else anchor,
                    "half_cams":max(0,anchor+.5*cams_delta) if np.isfinite(cams_delta) else anchor,"mean_revert":max(0,anchor+revert)}
        minimum=epoch-3*86400
        known=[r for r in records if r["targetEndEpoch"]<known_cutoff and r["originEpoch"]>=minimum and r["actual"] is not None]
        if len(known)>=12:
            errors=np.array([np.mean([abs(r[c]-r["actual"]) for r in known]) for c in COMPONENTS])
            weights=1/(errors+3.)**2
        else:weights=np.ones(len(COMPONENTS))
        weights/=weights.sum()
        pred=float(sum(w*candidates[c] for w,c in zip(weights,COMPONENTS)))
        records.append({"originEpoch":epoch,"forecastIssuedEpoch":issued_epoch,"targetStartEpoch":epoch+lead_minutes*60,
                        "targetEndEpoch":int(ends[i].timestamp()),"leadMinutes":lead_minutes,
                        "actual":float(target.iloc[i]) if np.isfinite(target.iloc[i]) else None,"prediction":pred,
                        "trainingCount":int(mask.sum()),"weightScoredCount":len(known),"weights":dict(zip(COMPONENTS,weights.tolist())),
                        "camsFetchedEpoch":int(f.camsFetchedEpoch.iloc[i]) if pd.notna(f.camsFetchedEpoch.iloc[i]) else None,
                        "weatherFetchedEpoch":float(f.weatherFetchedEpoch.iloc[i]) if pd.notna(f.weatherFetchedEpoch.iloc[i]) else None,
                        "camsUsable":bool(np.isfinite(cams_delta)),
                        "weatherUsable":bool(weather_usable[i]),
                        "rainFeatures":descriptions[i],
                        "trainingLatestTargetEndEpoch":int((ends[mask].asi8//10**9).max()),
                        "weightLatestTargetEndEpoch":max((r["targetEndEpoch"] for r in known),default=None),
                        "weatherFeatureColumns":list(weather_cols),
                        **candidates})
    return records


def _fresh_weighted_records(records,rows,lag_seconds,issue_epoch,training_cutoff_epoch=None):
    """Score standalone components under the published fresh-reference policy.

    Each historical component is scored as max(0, closed component + that issue's
    fresh-minus-closed reference), using only strictly completed outcomes. Missing
    fresh references retain the closed component. Fits and the 3-day, 12-outcome,
    inverse-square weighting rule are unchanged.

    The published estimator remains max(0, sum(w * closed components) + shift).
    Clipping does not commute with averaging: do not average individually floored
    fresh components. Component MAE weights are a heuristic, not optimization of
    the mixture's MAE. Re-score outside the fit cache so corrected raw references
    change weights and residuals without refitting unchanged components.
    Input records must be the closed replay output, not previously refreshed
    records. Frozen research replays must repeat their training cutoff here.
    """
    refreshed=refresh_session_records(records,rows,lag_seconds,issue_epoch)
    if not records:return refreshed
    components=np.array([[r[c] for c in COMPONENTS] for r in records],dtype=float)
    origins=np.array([r["originEpoch"] for r in records])
    ends=np.array([r["targetEndEpoch"] for r in records])
    actual=np.array([r["actual"] if r["actual"] is not None else np.nan for r in records])
    shifts=np.array([r["persistence"]-r["closedPersistence"] for r in refreshed])
    errors=np.abs(np.maximum(0.,components+shifts[:,None])-actual[:,None])
    for i,row in enumerate(refreshed):
        cutoff=row["forecastIssuedEpoch"]
        if training_cutoff_epoch is not None:cutoff=min(cutoff,training_cutoff_epoch)
        known=(ends[:i]<cutoff)&(origins[:i]>=origins[i]-3*86400)&np.isfinite(actual[:i])
        count=int(known.sum())
        weights=1/(errors[:i][known].mean(axis=0)+3.)**2 if count>=12 else np.ones(len(COMPONENTS))
        weights/=weights.sum()
        closed=float(sum(w*c for w,c in zip(weights,components[i])))
        row["weights"]=dict(zip(COMPONENTS,weights.tolist()))
        row["adaptiveWeightPolicyVersion"]=WEIGHT_POLICY_VERSION
        row["weightScoredCount"]=count
        row["weightLatestTargetEndEpoch"]=int(ends[:i][known].max()) if count else None
        row["closedPrediction"]=closed
        row["prediction"]=max(0.,closed+shifts[i])
    return refreshed


def _evidence(records,epoch):
    scored=[r for r in records if r["actual"] is not None and r["targetEndEpoch"]<epoch]
    if not scored:return {"count":0,"distinctDays":0,"calibrated":False},None
    error=np.array([r["actual"]-r["prediction"] for r in scored])
    base=np.array([r["actual"]-r["persistence"] for r in scored])
    dates={pd.Timestamp(r["originEpoch"],unit="s",tz="UTC").tz_convert("Asia/Kuala_Lumpur").date() for r in scored}
    evidence={"count":len(scored),"distinctDays":len(dates),"mae":round(float(np.abs(error).mean()),2),
              "persistenceMae":round(float(np.abs(base).mean()),2),"rmse":round(float(np.sqrt((error**2).mean())),2),
              "closedMainReferenceMae":round(float(np.mean([abs(r["actual"]-r.get("closedPrediction",r["prediction"])) for r in scored])),2),
              "freshnessAppliedCount":sum(bool(r.get("freshnessApplied")) for r in scored),
              "persistenceRole":"same_issue_five_minute_reference_with_closed_anchor_fallback",
              "validationMode":"exact_target_same_issue_lag_fresh_reference_replay","originSpacingMinutes":30,
              "independentOrigins":False,"calibrated":False,"prospectivelyValidated":False,
              "limitation":"Experimental reactive reference policy; mixed retrospective day and tail-error results. Overlapping historical outcomes are not independent or proof of future accuracy."}
    recent=[r for r in scored if r["targetEndEpoch"]>=epoch-3*86400]
    evidence["recentCompleted72Hours"]={
        "count":len(recent),
        "mae":round(float(np.mean([abs(r["actual"]-r["prediction"]) for r in recent])),2) if recent else None,
        "persistenceMae":round(float(np.mean([abs(r["actual"]-r["persistence"]) for r in recent])),2) if recent else None,
        "closedMainReferenceMae":round(float(np.mean([abs(r["actual"]-r.get("closedPrediction",r["prediction"])) for r in recent])),2) if recent else None,
        "meanErrorPredictionMinusActual":round(float(np.mean([r["prediction"]-r["actual"] for r in recent])),2) if recent else None,
        "selectionBasis":"target_end_in_latest_72_hours_and_strictly_before_issue",
        "independentOrigins":False,
    }
    evidence["missingCompletedTargetCount"]=sum(r["actual"] is None and r["targetEndEpoch"]<epoch for r in records)
    evidence["incompleteTargetsScored"]=False
    return evidence,np.quantile(error,[.1,.9])


def predict_windows(db_path,rows,windows,issue_epoch,*,rain_learning=False):
    """Return PM mean points for exact selected sessions, separately from weather.

    windows maps key -> {startEpoch, endEpoch}. Only 120-minute sessions whose
    start lies 90..1440 minutes after the latest complete sensor bucket are used.
    Returned ranges describe empirical mean-forecast residuals, not trail min/max.
    Default: original weather ridge with corrected source clock, validation and
    interval amount. New rain-response coefficients remain opt-in research.
    """
    start_clock=time.perf_counter()
    model_version=RAIN_MODEL_VERSION if rain_learning else MODEL_VERSION
    unavailable=lambda reason:{"available":False,"modelVersion":model_version,"reason":reason}
    try:
        issue_number=float(issue_epoch)
        if not math.isfinite(issue_number):raise ValueError("nonfinite issue")
        issue_epoch=int(issue_number)
    except (ValueError,TypeError,OverflowError):
        return {key:unavailable("invalid_issue_epoch") for key in windows}
    valid_rows=[]
    for source in rows:
        row=dict(source)
        try:epoch=float(row.get("epoch"))
        except (ValueError,TypeError,OverflowError):continue
        if math.isfinite(epoch) and epoch<=issue_epoch:
            valid_rows.append(row)
    rows=valid_rows
    frame=_frame(rows,issue_epoch)
    if frame.empty:return {key:unavailable("missing_sensor_data") for key in windows}
    raw_epochs,raw_values=raw_arrays(rows,issue_epoch)
    latest_raw=int(raw_epochs[-1]) if len(raw_epochs) else 0
    if issue_epoch-latest_raw>600:
        return {key:unavailable("stale_latest_sensor_reading") for key in windows}
    origin=frame.index[-1];origin_epoch=int(origin.timestamp())
    if not np.isfinite(frame.pm02.iloc[-1]) or issue_epoch-origin_epoch>1800:
        return {key:unavailable("stale_or_missing_closed_sensor_bucket") for key in windows}
    hasher=hashlib.blake2b(pd.util.hash_pandas_object(frame,index=True).values.tobytes(),digest_size=12)
    digest=hasher.hexdigest()
    window_key=tuple((k,w.get("startEpoch"),w.get("endEpoch")) for k,w in sorted(windows.items()))
    lag_seconds=issue_epoch-origin_epoch
    if not 0<=lag_seconds<900:
        return {key:unavailable("invalid_issue_lag") for key in windows}
    # Historical issues all precede the live closed origin. New live weather
    # therefore invalidates the query, without invalidating historical fits.
    # Exact lag is part of the fit key: each replay uses the same issue clock.
    tables=("air_quality_forecast_runs","weather_forecast_runs")
    revision=tuple(_raw_runs(db_path,table,origin_epoch)[1] for table in tables) if issue_epoch!=origin_epoch else None
    issue_snapshots=tuple(_raw_runs(db_path,table,issue_epoch) for table in tables)
    live_revision=tuple(snapshot[1] for snapshot in issue_snapshots)
    if revision is None:revision=live_revision
    fit_key=(str(db_path),model_version,digest,window_key,revision,lag_seconds)
    cams=_read_runs(db_path,"air_quality_forecast_runs",issue_epoch,raw_snapshot=issue_snapshots[0])
    weather=_read_runs(db_path,"weather_forecast_runs",issue_epoch,raw_snapshot=issue_snapshots[1])
    del issue_snapshots
    # Full causal raw history participates in the final key because historical
    # five-minute references determine weights and this policy's residuals.
    raw_hash=hashlib.blake2b(digest_size=12)
    raw_hash.update(raw_epochs.tobytes());raw_hash.update(raw_values.tobytes())
    key=(fit_key,live_revision,FRESHNESS_VERSION,WEIGHT_POLICY_VERSION,raw_hash.hexdigest(),issue_epoch)
    with _LOCK:
        if key in _CACHE:
            result=copy.deepcopy(_CACHE[key]);_CACHE.move_to_end(key)
            return result
        live_records={}
        if fit_key in _FIT_CACHE:
            fitted=_FIT_CACHE[fit_key];_FIT_CACHE.move_to_end(fit_key)
        else:
            fitted={}
            for name,window in windows.items():
                start=window.get("startEpoch");end=window.get("endEpoch")
                try:
                    if start is None or end is None or not math.isfinite(float(start)) or not math.isfinite(float(end)) or end-start!=7200 or (start-origin_epoch)%900:
                        raise ValueError("unsupported window")
                except (ValueError,TypeError,OverflowError):
                    fitted[name]=unavailable("unsupported_session_alignment_or_duration");continue
                lead=(start-origin_epoch)//60
                if not 90<=lead<=MAX_LEAD_MINUTES:
                    fitted[name]=unavailable("outside_90_min_to_24_hour_model_horizon");continue
                records=replay(frame,int(lead),cams,weather,issue_lag_seconds=lag_seconds,corrected_weather_only=not rain_learning)
                if not records or records[-1]["originEpoch"]!=origin_epoch:
                    fitted[name]=unavailable("insufficient_complete_training_history");continue
                fitted[name]={"records":records[:-1],"lead":int(lead)}
                # This full replay already contains the live origin. Keep it
                # only for this call; cached fits omit the live row so newer
                # actual-issue weather still triggers a query on a fit hit.
                live_records[name]=records
            _FIT_CACHE[fit_key]=fitted
            while len(_FIT_CACHE)>4:_FIT_CACHE.popitem(last=False)
        result={}
        for name,window in windows.items():
            start=window.get("startEpoch");end=window.get("endEpoch")
            fit=fitted[name]
            if "records" not in fit:
                result[name]=copy.deepcopy(fit);continue
            lead=fit["lead"]
            queried=live_records.get(name)
            if queried is None:
                queried=replay(frame,lead,cams,weather,issue_lag_seconds=lag_seconds,prior_records=fit["records"],corrected_weather_only=not rain_learning)
            if not queried or queried[-1]["originEpoch"]!=origin_epoch:
                result[name]=unavailable("insufficient_complete_training_history");continue
            records=_fresh_weighted_records(queried,rows,lag_seconds,issue_epoch)
            row=records[-1];evidence,interval=_evidence(records,issue_epoch)
            evidence["issueLagSeconds"]=lag_seconds
            evidence["freshnessPolicyVersion"]=FRESHNESS_VERSION
            evidence["adaptiveWeightPolicyVersion"]=WEIGHT_POLICY_VERSION
            evidence["weatherAsOfPolicy"]="latest_valid_fetched_at_or_before_each_actual_issue_origin_plus_exact_lag"
            evidence["rainFeatureVersion"]=getattr(rain_features,"VERSION","rain_weather_features_v1")
            evidence["validationMode"]="exact_target_same_issue_lag_rain_feature_and_fresh_reference_replay"
            mean=row["prediction"]
            lo=max(0,mean+interval[0]) if interval is not None else None
            hi=max(0,mean+interval[1]) if interval is not None else None
            result[name]={"available":True,"modelVersion":model_version,"source":"local_session_adaptive_model",
                          "pointRole":"experimental_session_mean","mean":round(mean,1),"prediction":round(mean,1),
                          "sensorAnchor":round(row["persistence"],1),"closedSensorAnchor":round(row["closedPersistence"],1),
                          "closedMainPrediction":round(row["closedPrediction"],1),
                          "sensorReferenceEpoch":row["freshReferenceEpoch"],"sensorReferenceCount":row["freshReferenceCount"],
                          "anchorRole":"fresh_five_minute_sensor_median_reference_not_forecast" if row["freshnessApplied"] else "closed_15_minute_median_reference_not_forecast",
                          "freshnessAdjustment":{"applied":row["freshnessApplied"],"version":FRESHNESS_VERSION,
                              "amountUgM3":round(mean-row["closedPrediction"],4),
                              "closedReferencePm25UgM3":round(row["closedPersistence"],1),
                              "prospectivelyValidated":False,
                              "referenceDifference":round(row["persistence"]-row["closedPersistence"],4),
                              "windowSeconds":REFERENCE_WINDOW_SECONDS,"featureAnchorEpoch":origin_epoch,
                              "referenceEpoch":row["freshReferenceEpoch"],"referenceCount":row["freshReferenceCount"],
                              "lagSeconds":lag_seconds,"pointPolicy":"experimental_observed_level_refresh_not_validated_model_promotion",
                              "limitations":"Reacts after local changes are observed; mixed retrospective day and tail accuracy; sparse five-minute samples; no anticipation claim."},
                          "originEpoch":origin_epoch,"forecastedAtEpoch":int(issue_epoch),"startEpoch":int(start),"endEpoch":int(end),
                          "leadHours":round(lead/60,2),"target":"mean_of_eight_closed_15_minute_sensor_medians",
                          "remainingLeadHours":round((start-issue_epoch)/3600,2),
                          "rawRangeLow":None if lo is None else round(lo,1),"rawRangeHigh":None if hi is None else round(hi,1),
                          "rangeRole":"empirical_q10_q90_session_mean_error_span","calibrated":False,"confidence":"low",
                          "experimental":True,"usedForDecision":False,"usedForComparison":True,"trainingCount":row["trainingCount"],
                          "adaptiveWeightScoredCount":row["weightScoredCount"],"components":{c:round(row["closedPersistence"] if c=="persistence" else row[c],2) for c in COMPONENTS},
                          "componentsRole":"closed_sensor_features_actual_issue_weather_components_before_fresh_reference_adjustment",
                          "adaptiveWeighting":{"version":WEIGHT_POLICY_VERSION,
                              "scoringPolicy":"standalone_component_after_same_issue_fresh_shift_and_zero_floor",
                              "historyDays":3,"minimumCompletedOutcomes":12,
                              "completionRule":"target_end_strictly_before_each_issue",
                              "missingFreshPolicy":"closed_reference_fallback",
                              "pointFormula":"max(0, sum(weights * closed_components) + fresh_reference - closed_reference)",
                              "limitation":"Component MAE weighting is a heuristic; individually floored components are not averaged and accuracy improvement is not established."},
                          "weights":{c:round(w,4) for c,w in row["weights"].items()},"modelEvidence":evidence,
                          "camsFetchedEpoch":row["camsFetchedEpoch"],"weatherFetchedEpoch":row["weatherFetchedEpoch"],
                          "camsAvailable":row["camsUsable"],"weatherAvailable":row["weatherUsable"],
                          "rainFeatures":copy.deepcopy(row["rainFeatures"]),
                          "rainContext":copy.deepcopy(row["rainFeatures"]),
                          "weatherSourceClock":{"asOfEpoch":issue_epoch,"sensorFeatureOriginEpoch":origin_epoch,
                              "fetchedEpoch":row["weatherFetchedEpoch"],"issueLagSeconds":lag_seconds,
                              "availabilityRule":"fetched_epoch_at_or_before_actual_issue","maximumAgeSeconds":7200},
                          "rainLearning":{"enabled":bool(rain_learning),"appliedToPrimary":False,
                              "candidateOnly":bool(rain_learning),"featureColumns":list(RAIN_COLS) if rain_learning else [],
                              "ridgePenalty":RAIN_RIDGE_PENALTY,"interactionPenalty":RAIN_INTERACTION_PENALTY,
                              "probabilityRole":"forecast_precipitation_probability_not_PM_washout_probability",
                              "effectPolicy":"learned_coefficients_with_unconstrained_signs_and_shrinkage" if rain_learning else "expanded_rain_response_candidate_not_promoted_after_paired_evaluation",
                              "trainingLatestTargetEndEpoch":row["trainingLatestTargetEndEpoch"],
                              "weightLatestTargetEndEpoch":row["weightLatestTargetEndEpoch"]},
                          "sourceCaveat":"Single TTDI sensor; local plume changes and new regimes can exceed historical forecast errors."}
        elapsed=round(time.perf_counter()-start_clock,3)
        for value in result.values():value["computeSeconds"]=elapsed
        _CACHE[key]=copy.deepcopy(result)
        while len(_CACHE)>4:_CACHE.popitem(last=False)
        return result


def record_issue(db_path,result,recorded_epoch):
    """Archive genuinely live issued points once; never call for a replay.

    Deduplication intentionally preserves the first actual publication for an
    origin/session/model. Late starts retain their true recorded issue time.
    """
    with closing(sqlite3.connect(str(db_path),timeout=20)) as conn,conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS window_pm_forecast_issues(
            model_version TEXT NOT NULL, origin_epoch INTEGER NOT NULL,
            start_epoch INTEGER NOT NULL, end_epoch INTEGER NOT NULL,
            issued_epoch INTEGER NOT NULL, window_key TEXT NOT NULL,
            predicted_mean REAL NOT NULL, anchor_pm25 REAL NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY(model_version,origin_epoch,start_epoch,end_epoch))""")
        count=0
        for name,value in result.items():
            if not value.get("available") or value.get("modelVersion") not in (MODEL_VERSION,RAIN_MODEL_VERSION,"afternoon_direction_hgb_v1"):continue
            if int(recorded_epoch)<int(value["originEpoch"]):continue
            if (value.get("modelVersion") == "afternoon_direction_hgb_v1"
                    and int(recorded_epoch) < int(value["forecastedAtEpoch"])):continue
            payload=copy.deepcopy(value)
            payload["recordedIssuedEpoch"]=int(recorded_epoch)
            cur=conn.execute("""INSERT OR IGNORE INTO window_pm_forecast_issues
                (model_version,origin_epoch,start_epoch,end_epoch,issued_epoch,window_key,predicted_mean,anchor_pm25,payload)
                VALUES(?,?,?,?,?,?,?,?,?)""",(value["modelVersion"],value["originEpoch"],value["startEpoch"],value["endEpoch"],
                    int(recorded_epoch),name,value["mean"],value["sensorAnchor"],json.dumps(payload,separators=(",",":"))))
            count+=cur.rowcount
        return count
