"""Pure causal sensor transforms; no labels, imputation, rules, or file access."""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOWS = ((0,15),(15,30),(30,60),(60,90))
CHANNELS = ('temperature','humidity')
EXTRA_COLUMNS = tuple(
    name for channel in CHANNELS for name in (
        *(f'{channel}Mean{lo}_{hi}' for lo,hi in WINDOWS),
        *(f'{channel}RecentMinus{lo}_{hi}' for lo,hi in WINDOWS[1:]),
        *(f'{channel}Slope{minutes}PerMinute' for minutes in (15,30,60)),
        *(f'{channel}Sd{minutes}' for minutes in (15,30)),
    )
)
INTERACTIONS = (
    ('temperatureHumidityRecentChangeProduct','temperatureRecentMinus15_30','humidityRecentMinus15_30'),
    ('temperatureHumiditySlope15Product','temperatureSlope15PerMinute','humiditySlope15PerMinute'),
    ('pmDelta15TemperatureChangeProduct','rawDelta15','temperatureRecentMinus15_30'),
    ('pmDelta15HumidityChangeProduct','rawDelta15','humidityRecentMinus15_30'),
    ('pmLevelTemperatureChangeProduct','freshPm25','temperatureRecentMinus15_30'),
    ('pmLevelHumidityChangeProduct','freshPm25','humidityRecentMinus15_30'),
    ('pmSlopeTemperatureSlope15Product','rawSlope15PerMinute','temperatureSlope15PerMinute'),
    ('pmSlopeHumiditySlope15Product','rawSlope15PerMinute','humiditySlope15PerMinute'),
)


def window(frame, channel, lo, hi):
    columns=[f'seq_{channel}_{age}_{age+3}min' for age in range(lo,hi,3)]
    values=frame[columns].to_numpy(float)
    values[~np.isfinite(values)]=np.nan
    valid=np.isfinite(values)
    count=valid.sum(axis=1)
    usable=count>=max(2,int(np.ceil(len(columns)/2)))
    total=np.nansum(values,axis=1)
    mean=np.divide(total,count,out=np.full(len(values),np.nan),where=count>0)
    mean[~usable]=np.nan
    return values, valid, usable, mean


def transform(frame, base_columns):
    """Feature windows are issue-relative bins already constrained by receipts."""
    output=frame[list(base_columns)].copy()
    for channel in CHANNELS:
        for lo,hi in WINDOWS:
            output[f'{channel}Mean{lo}_{hi}']=window(frame,channel,lo,hi)[3]
        for lo,hi in WINDOWS[1:]:
            output[f'{channel}RecentMinus{lo}_{hi}']=output[f'{channel}Mean0_15']-output[f'{channel}Mean{lo}_{hi}']
        for minutes in (15,30,60):
            values,valid,usable,mean=window(frame,channel,0,minutes)
            # Ages are bin-left distances from issue, so right-bin ends are -age.
            t=-np.arange(0,minutes,3,dtype=float)
            count=valid.sum(axis=1)
            mean_t=np.divide((valid*t).sum(axis=1),count,out=np.full(len(values),np.nan),where=count>0)
            centered_t=t[None,:]-mean_t[:,None]
            centered_y=values-mean[:,None]
            denominator=np.where(valid,centered_t**2,0.).sum(axis=1)
            numerator=np.nansum(centered_t*centered_y,axis=1)
            slope=np.divide(numerator,denominator,out=np.full(len(values),np.nan),where=usable&(denominator>0))
            output[f'{channel}Slope{minutes}PerMinute']=slope
            if minutes in (15,30):
                variance=np.divide(np.nansum(centered_y**2,axis=1),count,out=np.full(len(values),np.nan),where=usable)
                output[f'{channel}Sd{minutes}']=np.sqrt(variance)
    for name,left,right in INTERACTIONS:
        output[name]=output[left]*output[right]
    output=output.reindex(columns=[*base_columns,*EXTRA_COLUMNS,*(x[0] for x in INTERACTIONS)])
    assert list(output.columns)==[*base_columns,*EXTRA_COLUMNS,*(x[0] for x in INTERACTIONS)]
    assert not np.isinf(output.to_numpy(float)).any()
    return output
