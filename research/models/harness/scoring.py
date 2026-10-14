"""Scoring: identical targets/cutoffs for every candidate, persistence and incumbents.

Conventions
* Persistence = the fresh 5-min reference at the issue (``y_ref``) for every
  concentration target; as a probability it is degenerate (0/1).
* Event truth (>=T, T in {20, 40}): delta = y_arrival - y_ref; fall if delta <= -T,
  rise if delta >= T (production label_classes boundaries).
* Event calls: from a point forecast, sign(pred - ref) when |pred - ref| >= T;
  from probabilities, the most probable of (fall, neither, rise) ("argmax",
  the primary, untuned rule) and, as a fixed sensitivity, any direction with
  probability >= 0.2 (larger one wins).
* Issue-level confusion: hit (right direction), miss (event, no call),
  wrong-way (event, opposite call), false (no event, call).
* Episodes: consecutive issues with the same true direction, a gap <= 30 min
  continuing the episode. Caught = at least one right-direction call inside it;
  lead = minutes from the episode's first issue to the first right call.
  False clusters: runs of false calls merged the same way.
* Probabilities: Brier and log loss against the walk-forward climatological
  prior (class frequencies of labels completed before each day's cutoff within
  28 days), i.e. the same information persistence of the base rate would have.
  The frequencies come from the full labelled history (``history=``), never
  from the evaluation subset alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EPISODE_GAP_SECONDS = 30 * 60
CLASS5 = ("fall40", "fall20to40", "within20", "rise20to40", "rise40")


# ----------------------------------------------------------------------------- points
def point_metrics(y, pred, ref=None) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if ref is not None:
        ref = np.asarray(ref, float)
        mask &= np.isfinite(ref)
    if not mask.any():
        return {"n": 0}
    error = pred[mask] - y[mask]
    out = {"n": int(mask.sum()), "mae": float(np.mean(np.abs(error))),
           "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error))}
    if ref is not None:
        base = np.abs(ref[mask] - y[mask])
        out["persistence_mae"] = float(base.mean())
        out["skill_vs_persistence"] = float(1 - out["mae"] / out["persistence_mae"]) if base.mean() > 0 else np.nan
        big = np.abs(y[mask] - ref[mask]) >= 20
        out["n_change20"] = int(big.sum())
        out["mae_change20"] = float(np.mean(np.abs(error[big]))) if big.any() else np.nan
        out["persistence_mae_change20"] = float(base[big].mean()) if big.any() else np.nan
    return out


def pinball(y, q, tau) -> float:
    y, q = np.asarray(y, float), np.asarray(q, float)
    mask = np.isfinite(y) & np.isfinite(q)
    diff = y[mask] - q[mask]
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff))) if mask.any() else np.nan


# ----------------------------------------------------------------------------- events
def truth_direction(delta, threshold) -> np.ndarray:
    delta = np.asarray(delta, float)
    out = np.where(delta <= -threshold, -1.0, np.where(delta >= threshold, 1.0, 0.0))
    out[~np.isfinite(delta)] = np.nan
    return out


def calls_from_point(pred, ref, threshold) -> np.ndarray:
    change = np.asarray(pred, float) - np.asarray(ref, float)
    out = np.where(change <= -threshold, -1.0, np.where(change >= threshold, 1.0, 0.0))
    out[~np.isfinite(change)] = np.nan
    return out


def calls_from_probs(p_fall, p_rise, rule="argmax") -> np.ndarray:
    p_fall, p_rise = np.asarray(p_fall, float), np.asarray(p_rise, float)
    neither = 1 - p_fall - p_rise
    if rule == "argmax":
        stack = np.vstack([p_fall, neither, p_rise])
        out = np.argmax(np.nan_to_num(stack, nan=-1), axis=0).astype(float) - 1
    else:
        tau = float(rule)
        out = np.where((p_fall >= tau) | (p_rise >= tau), np.where(p_fall >= p_rise, -1.0, 1.0), 0.0)
    out[~(np.isfinite(p_fall) & np.isfinite(p_rise))] = np.nan
    return out


def _runs(issues, flags, gap=EPISODE_GAP_SECONDS):
    """Group sorted issue epochs where flags is True into runs (gap <= gap continues)."""
    epochs = np.asarray(issues, np.int64)[np.asarray(flags, bool)]
    if not len(epochs):
        return []
    breaks = np.flatnonzero(np.diff(epochs) > gap) + 1
    return np.split(epochs, breaks)


def event_confusion(issues, truth, call) -> dict:
    issues = np.asarray(issues, np.int64)
    truth, call = np.asarray(truth, float), np.asarray(call, float)
    mask = np.isfinite(truth) & np.isfinite(call)
    issues, truth, call = issues[mask], truth[mask], call[mask]
    order = np.argsort(issues)
    issues, truth, call = issues[order], truth[order], call[order]
    event = truth != 0
    out = {"n_issues": int(len(issues)), "n_event_issues": int(event.sum()),
           "hits": int((event & (call == truth)).sum()),
           "misses": int((event & (call == 0)).sum()),
           "wrong_way": int((event & (call == -truth)).sum()),
           "false_calls": int((~event & (call != 0)).sum())}
    out["recall"] = out["hits"] / out["n_event_issues"] if out["n_event_issues"] else np.nan
    calls = int((call != 0).sum())
    out["precision"] = out["hits"] / calls if calls else np.nan
    lookup = dict(zip(issues, zip(truth, call)))
    caught, leads, episodes = 0, [], 0
    for direction in (-1.0, 1.0):
        for run in _runs(issues, truth == direction):
            episodes += 1
            right = [e for e in run if lookup[e][1] == direction]
            if right:
                caught += 1
                leads.append((right[0] - run[0]) / 60)
    out["episodes"] = episodes
    out["episodes_caught"] = caught
    out["median_minutes_to_first_hit"] = float(np.median(leads)) if leads else np.nan
    out["false_clusters"] = len(_runs(issues, (~event) & (call != 0)))
    wrong = len(_runs(issues, event & (call == -truth)))
    out["wrong_way_clusters"] = wrong
    return out


def episodes_table(issues, truth) -> pd.DataFrame:
    issues = np.asarray(issues, np.int64)
    truth = np.asarray(truth, float)
    order = np.argsort(issues)
    issues, truth = issues[order], truth[order]
    rows = []
    for direction in (-1.0, 1.0):
        for run in _runs(issues, truth == direction):
            rows.append({"direction": "fall" if direction < 0 else "rise", "start": int(run[0]), "end": int(run[-1]),
                         "start_myt": str(config.myt(run[0]))[:16], "issues": len(run)})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------- probabilities
def walk_forward_prior(ds: pd.DataFrame, column: str, values, complete_col: str, lookback_days=28,
                       history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-issue prior = label frequencies completed before the issue day's cutoff.

    ``history`` holds the labelled rows the frequencies come from. Pass the
    full dataset when ``ds`` is an evaluation subset: a subset starting on the
    first evaluation day has no completed labels for that day and only a
    truncated lookback for the following weeks.
    """
    history = ds if history is None else history
    days = np.array([config.midnight(i) for i in ds.index.to_numpy(np.int64)])
    index = history.index.to_numpy(np.int64)
    labels = history[column].to_numpy(float)
    complete = history[complete_col].to_numpy(np.int64)
    out = np.full((len(ds), len(values)), np.nan)
    for cutoff in np.unique(days):
        # A label counts only if it is one of ``values``: y_class5 uses -1 for
        # 'no label', but y_first20 uses -1 for a drop, which must be counted.
        train = (index >= cutoff - lookback_days * 86400) & (complete < cutoff) & np.isin(labels, values)
        if not train.any():
            continue
        counts = np.array([(labels[train] == v).sum() for v in values], float)
        out[days == cutoff] = counts / counts.sum()
    return pd.DataFrame(out, index=ds.index, columns=[f"prior_{v}" for v in values])


def multiclass_scores(p, y, classes) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    # Score every row whose label is one of ``classes`` (first crossing: -1 is a drop, not 'unlabelled').
    mask = np.isfinite(p).all(axis=1) & np.isin(y, np.asarray(classes, float))
    if not mask.any():
        return {"n": 0}
    onehot = (y[mask, None] == np.asarray(classes)[None, :]).astype(float)
    pm = p[mask]
    brier = float(np.mean(np.sum((pm - onehot) ** 2, axis=1)))
    picked = np.clip(np.sum(pm * onehot, axis=1), 1e-15, 1)
    return {"n": int(mask.sum()), "brier": brier, "log_loss": float(-np.mean(np.log(picked)))}


def binary_brier(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, float)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return {"n": 0}
    pc = np.clip(p[mask], 1e-15, 1 - 1e-15)
    return {"n": int(mask.sum()), "brier": float(np.mean((p[mask] - y[mask]) ** 2)),
            "log_loss": float(-np.mean(y[mask] * np.log(pc) + (1 - y[mask]) * np.log(1 - pc))),
            "base_rate": float(y[mask].mean())}


# ----------------------------------------------------------------------------- one-call summary
def score_candidate(ds: pd.DataFrame, pred: pd.DataFrame, name="candidate", mask=None) -> dict:
    """Score whatever standard prediction columns ``pred`` has, on identical rows.

    Standard columns (all optional): pred_arrival, pred_ride_mean, pred_ride_min,
    pred_ride_max, p_fall20, p_rise20, p_fall40, p_rise40, p_class5_0..4,
    p_first_drop/p_first_none/p_first_rise, p_ride_le70, pred_morning,
    pred_afternoon. Missing predictions count as unavailable, never as persistence.
    """
    frame = ds.join(pred, how="inner", rsuffix="_pred")
    if mask is not None:
        frame = frame.loc[mask.reindex(frame.index).fillna(False).to_numpy(bool)]
    ref = frame.y_ref.to_numpy(float)
    issues = frame.index.to_numpy(np.int64)
    result = {"name": name, "rows": len(frame)}
    if "pred_arrival" in frame:
        result["arrival"] = point_metrics(frame.y_arrival, frame.pred_arrival, ref)
        result["arrival_coverage"] = float(np.isfinite(frame.pred_arrival).mean())
        for t in config.THRESHOLDS:
            result[f"events{int(t)}_point"] = event_confusion(
                issues, truth_direction(frame.y_delta, t), calls_from_point(frame.pred_arrival, ref, t))
    for head in ("ride_mean", "ride_min", "ride_max"):
        if f"pred_{head}" in frame:
            result[head] = point_metrics(frame[f"y_{head}"], frame[f"pred_{head}"], ref)
    if "pred_ride_min" in frame:
        result["ride_min_pinball10"] = pinball(frame.y_ride_min, frame.pred_ride_min, 0.1)
    if "pred_ride_max" in frame:
        result["ride_max_pinball90"] = pinball(frame.y_ride_max, frame.pred_ride_max, 0.9)
    for t in config.THRESHOLDS:
        fall, rise = f"p_fall{int(t)}", f"p_rise{int(t)}"
        if fall in frame and rise in frame:
            truth = truth_direction(frame.y_delta, t)
            for rule in ("argmax", "0.2"):
                result[f"events{int(t)}_prob_{rule}"] = event_confusion(
                    issues, truth, calls_from_probs(frame[fall], frame[rise], rule))
            result[f"brier_fall{int(t)}"] = binary_brier(frame[fall], frame[f"y_fall{int(t)}"])
            result[f"brier_rise{int(t)}"] = binary_brier(frame[rise], frame[f"y_rise{int(t)}"])
    p5 = [f"p_class5_{k}" for k in range(5)]
    if all(c in frame for c in p5):
        result["class5"] = multiclass_scores(frame[p5].to_numpy(float), frame.y_class5, range(5))
    if "p_ride_le70" in frame:
        result["ride_le70"] = binary_brier(frame.p_ride_le70, frame.y_ride_le70)
    first = ["p_first_drop", "p_first_none", "p_first_rise"]
    if all(c in frame for c in first):
        result["first20"] = multiclass_scores(frame[first].to_numpy(float), frame.y_first20, (-1, 0, 1))
        argmax = np.argmax(np.nan_to_num(frame[first].to_numpy(float), nan=-1), axis=1) - 1.0
        argmax[~np.isfinite(frame[first].to_numpy(float)).all(axis=1)] = np.nan
        result["first20_events"] = event_confusion(issues, frame.y_first20.to_numpy(float), argmax)
    for key in ("morning", "afternoon"):
        if f"pred_{key}" in frame:
            result[key] = point_metrics(frame[f"y_{key}_mean"], frame[f"pred_{key}"], ref)
    return result


def persistence_predictions(ds: pd.DataFrame) -> pd.DataFrame:
    ref = ds.y_ref
    return pd.DataFrame({"pred_arrival": ref, "pred_ride_mean": ref, "pred_ride_min": ref, "pred_ride_max": ref,
                         "pred_morning": ref, "pred_afternoon": ref,
                         "p_ride_le70": (ref <= config.PLANNING_CUTOFF).astype(float).where(ref.notna()),
                         "p_fall20": 0.0, "p_rise20": 0.0, "p_fall40": 0.0, "p_rise40": 0.0}, index=ds.index)
