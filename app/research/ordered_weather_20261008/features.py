"""Compact forecast-wind summaries from the original issue-available tensors.

No labels, model fits, DB reads or new forecasts are used here. Rain retains
the certified basic31 native-hour features; issue-shifted tensor rain is unused.
Wind summaries approximate hourly trajectories, not observed wind or causation.
"""
from __future__ import annotations
import numpy as np

EXTRA_COLUMNS = tuple(
    name for height in ('10m', '925hPa', '180m')
    for name in (f'target_{height}_u', f'target_{height}_v',
                 f'target_{height}_mean_speed', f'target_{height}_coherence',
                 f'issue_to_target_{height}_rotation_cos',
                 f'issue_to_target_{height}_rotation_sin')
) + ('target_gust_mean', 'target_gust_max', 'target_blh_mean',
     'target_10m_925hPa_vector_difference', 'target_10m_180m_vector_difference',
     'target_10m_925hPa_alignment_cos', 'target_ventilation_m2s',
     'pretarget_ventilation_m2s', 'issue_to_target_ventilation_change_m2s',
     'path_10m_max_vector_step', 'path_925hPa_max_vector_step', 'pretarget_gust_max')


def mean(values, weights):
    """Strict coverage: a missing contributing sample makes its mean missing."""
    values = np.asarray(values, float)
    use = np.asarray(weights, float) > 0
    if not use.any() or not np.isfinite(values[use]).all():
        return np.nan
    return float(np.dot(values[use], weights[use]) / weights[use].sum())


def maximum(values, weights):
    use = np.asarray(weights, float) > 0
    if not use.any() or not np.isfinite(values[use]).all():
        return np.nan
    return float(np.max(values[use]))


def angle(u0, v0, u1, v1):
    denominator = np.hypot(u0, v0) * np.hypot(u1, v1)
    if not np.isfinite(denominator) or denominator <= 0.01:
        return np.nan, np.nan
    return (float(np.clip((u0*u1 + v0*v1)/denominator, -1, 1)),
            float(np.clip((u0*v1 - v0*u1)/denominator, -1, 1)))


def row(future, epochs, issue, start, end):
    f = np.asarray(future, float)
    epochs = np.asarray(epochs, float)
    assert f.shape == (26, 15) and len(epochs) == 26 and end > start
    assert np.array_equal(epochs, issue + np.arange(26)*3600)
    # Continuous wind/gust/BLH hourly endpoint approximations, not native rain.
    target = np.maximum(0, np.minimum(epochs, end) - np.maximum(epochs-3600, start))/3600
    before = np.maximum(0, np.minimum(epochs, start) - np.maximum(epochs-3600, issue))/3600
    assert np.allclose(target, f[:, 13], atol=1e-6, rtol=0)
    assert np.isclose(target.sum(), (end-start)/3600)
    features = []
    speeds, vectors = [], []
    for u_col, v_col in ((4,5), (6,7), (8,9)):
        u, v = f[:, u_col], f[:, v_col]
        speed = np.hypot(u, v)
        u_mean, v_mean, speed_mean = mean(u,target), mean(v,target), mean(speed,target)
        coherence = (float(np.clip(np.hypot(u_mean,v_mean)/speed_mean,0,1))
                     if np.isfinite(speed_mean) and speed_mean > .1 else np.nan)
        cosine, sine = angle(u[0],v[0],u_mean,v_mean)
        features += [u_mean,v_mean,speed_mean,coherence,cosine,sine]
        speeds.append(speed); vectors.append((u,v))
    gust, blh = f[:,10], f[:,11]
    diff_925 = np.hypot(vectors[0][0]-vectors[1][0], vectors[0][1]-vectors[1][1])
    diff_180 = np.hypot(vectors[0][0]-vectors[2][0], vectors[0][1]-vectors[2][1])
    align = np.array([angle(a,b,c,d)[0] for a,b,c,d in zip(*vectors[0],*vectors[1])])
    ventilation = speeds[0]/3.6 * blh
    target_ventilation = mean(ventilation,target)
    path = (epochs >= issue) & (epochs <= end)
    steps = []
    for u,v in vectors[:2]:
        uv_change = np.hypot(np.diff(u),np.diff(v))
        available = path[1:] & path[:-1]
        steps.append(maximum(uv_change,available.astype(float)))
    features += [mean(gust,target),maximum(gust,target),mean(blh,target),
                 mean(diff_925,target),mean(diff_180,target),mean(align,target),
                 target_ventilation,mean(ventilation,before),
                 target_ventilation-ventilation[0],*steps,maximum(gust,before)]
    assert len(features) == len(EXTRA_COLUMNS)
    result = np.asarray(features,float)
    result[~np.isfinite(result)] = np.nan
    return result


def transform(bundle, metadata):
    assert len(metadata) == len(bundle['context'])
    extra = np.array([row(bundle['future'][i],bundle['future_epochs'][i],
                          r.issueEpoch,r.startEpoch,r.endEpoch)
                      for i,r in enumerate(metadata.itertuples(index=False))])
    assert extra.shape == (len(metadata),len(EXTRA_COLUMNS))
    base = np.asarray(bundle['context'],float)
    return base, np.concatenate((base,extra),axis=1)
