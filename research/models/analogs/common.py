"""Shared plumbing for the analog experiments: data, split, walk-forward run, identical-row scoring."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from harness import config, dataset, report, scoring, walkforward

from . import analog

OUT = config.OUT / "analogs"
SLIM_PATH = OUT / "slim.parquet"
# Tuning split: issue days 2026-09-10 .. 09-27 (18 days). Holdout (never used for
# any choice): 2026-09-28 .. 10-13 (16 days, all eight big-event cases).
TUNE_START = config.myt_epoch("2026-09-10")
HOLDOUT_START = config.myt_epoch("2026-09-28")
TARGETS = ("arrival", "ride_mean", "first20", "ride_le70")
EMBARGO_MINUTES = 330          # the stricter production embargo (Ridge v4) for every head


def load_slim(rebuild=False) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    if SLIM_PATH.exists() and not rebuild:
        return pd.read_parquet(SLIM_PATH)
    slim = analog.slim_dataset(dataset.load())
    slim.to_parquet(SLIM_PATH)
    return slim


def days(period: str) -> list[int]:
    all_days = walkforward.eval_days(load_slim())
    if period == "tune":
        return [d for d in all_days if d < HOLDOUT_START]
    if period == "holdout":
        return [d for d in all_days if d >= HOLDOUT_START]
    return all_days


def run(cfg: analog.AnalogConfig, period="tune", heads=("arrival", "class", "first", "ride"),
        lookback_days=60, slim=None, quiet=True) -> pd.DataFrame:
    slim = load_slim() if slim is None else slim
    return walkforward.walk_forward(slim, analog.make_fit_predict(cfg, heads), targets=TARGETS,
                                    lookback_days=lookback_days, embargo_minutes=EMBARGO_MINUTES,
                                    days=days(period), log=(lambda *_: None) if quiet else print)


_CACHE = {}


def references():
    """Evaluation frame, incumbent replay, persistence and walk-forward priors (full-history)."""
    if "ref" not in _CACHE:
        ds = dataset.load()
        ev = ds.loc[ds["eval"] & ds.m_fresh_available]
        replay = report.standard_replay(ds).reindex(ev.index)
        persist = scoring.persistence_predictions(ev)
        prior5 = scoring.walk_forward_prior(ev, "y_class5", range(5), "y_arrival_complete", history=ds)
        prior5.columns = [f"p_class5_{k}" for k in range(5)]
        prior5["p_fall20"] = prior5.p_class5_0 + prior5.p_class5_1
        prior5["p_rise20"] = prior5.p_class5_3 + prior5.p_class5_4
        prior5["p_fall40"], prior5["p_rise40"] = prior5.p_class5_0, prior5.p_class5_4
        priorf = scoring.walk_forward_prior(ev, "y_first20", (-1, 0, 1), "y_first20_complete", history=ds)
        priorf.columns = ["p_first_drop", "p_first_none", "p_first_rise"]
        prior70 = scoring.walk_forward_prior(ev, "y_ride_le70", (0.0, 1.0), "y_ride_complete", history=ds)
        prior = prior5.join(priorf).join(prior70[["prior_1.0"]].rename(columns={"prior_1.0": "p_ride_le70"}))
        _CACHE["ref"] = (ds, ev, replay, persist, prior)
    return _CACHE["ref"]


def masks(ev, replay, period):
    idx = ev.index.to_numpy(np.int64)
    if period == "tune":
        rows = (idx >= TUNE_START) & (idx < HOLDOUT_START)
    elif period == "holdout":
        rows = idx >= HOLDOUT_START
    else:
        rows = np.ones(len(ev), bool)
    rows = pd.Series(rows, index=ev.index)
    return {
        "point": rows & replay.pred_arrival.notna() & ev.y_arrival.notna(),
        "ride": rows & replay.pred_ride_mean.notna() & ev.y_ride_mean.notna(),
        "class": rows & replay.p_fall20.notna() & ev.y_delta.notna(),
        "first": rows & replay.p_first_drop.notna() & ev.y_first20.notna(),
        "le70": rows & replay.p_ride_le70.notna() & ev.y_ride_le70.notna(),
    }


POINT_COLS = ["pred_arrival"]
RIDE_COLS = ["pred_ride_mean", "pred_ride_min", "pred_ride_max"]
CLASS_COLS = ["p_fall20", "p_rise20", "p_fall40", "p_rise40"] + [f"p_class5_{k}" for k in range(5)]
FIRST_COLS = ["p_first_drop", "p_first_none", "p_first_rise"]


def score_all(pred: pd.DataFrame, period="tune", others=True) -> dict:
    """Score candidate, incumbent replay, persistence and prior on identical rows per head group.

    The rows of each head group are those where the incumbent produced an output,
    the label exists and the candidate produced an output (it does wherever a
    fresh reference exists, i.e. on every evaluation row).
    """
    ds, ev, replay, persist, prior = references()
    m = masks(ev, replay, period)
    out = {}
    models = {"candidate": pred, "prior": prior, "persistence": persist}
    if others:
        models["incumbent"] = replay
    groups = {"point": POINT_COLS, "ride": RIDE_COLS, "class": CLASS_COLS, "first": FIRST_COLS, "le70": ["p_ride_le70"]}
    for group, cols in groups.items():
        mask = m[group]
        if group in ("point", "ride", "class", "first", "le70"):
            have = [c for c in cols if c in pred]
            if have:
                mask = mask & pred[have].reindex(ev.index).notna().all(axis=1)   # candidate must also have output
        for name, frame in models.items():
            have = [c for c in cols if c in frame]
            if not have or (name == "prior" and group in ("point", "ride")) \
                    or (name == "persistence" and group in ("class", "first")):
                continue
            out[(group, name)] = scoring.score_candidate(ev, frame[have].reindex(ev.index), name=name, mask=mask)
    return out


def summary(scores: dict) -> dict:
    """Flat key numbers for tuning tables."""
    s = {}
    c = scores.get(("point", "candidate"))
    if c:
        s["arr_mae"] = c["arrival"]["mae"]
        s["arr_mae_pers"] = c["arrival"]["persistence_mae"]
        s["arr_skill"] = c["arrival"]["skill_vs_persistence"]
        e = c["events20_point"]
        s["pt20"] = f"{e['episodes_caught']}/{e['episodes']} f{e['false_calls']}"
    c = scores.get(("ride", "candidate"))
    if c:
        for h in ("ride_mean", "ride_min", "ride_max"):
            s[h] = c[h]["mae"]
    c, q = scores.get(("class", "candidate")), scores.get(("class", "prior"))
    if c:
        s["c5_brier"] = c["class5"]["brier"]
        s["c5_skill"] = 1 - c["class5"]["brier"] / q["class5"]["brier"]
        s["c5_ll"] = c["class5"]["log_loss"]
        s["c5_ll_prior"] = q["class5"]["log_loss"]
        for t in (20, 40):
            for side in ("fall", "rise"):
                s[f"bs_{side}{t}"] = 1 - c[f"brier_{side}{t}"]["brier"] / q[f"brier_{side}{t}"]["brier"]
            for rule in ("argmax", "0.2"):
                e = c[f"events{t}_prob_{rule}"]
                s[f"ev{t}_{rule}"] = (f"{e['episodes_caught']}/{e['episodes']} h{e['hits']} w{e['wrong_way']} "
                                      f"f{e['false_calls']}({e['false_clusters']})")
                s[f"ep{t}_{rule}"] = e["episodes_caught"]
                s[f"fc{t}_{rule}"] = e["false_clusters"]
                s[f"wc{t}_{rule}"] = e["wrong_way_clusters"]
    c, q = scores.get(("first", "candidate")), scores.get(("first", "prior"))
    if c:
        s["f_skill"] = 1 - c["first20"]["brier"] / q["first20"]["brier"]
        e = c["first20_events"]
        s["first_ev"] = f"{e['episodes_caught']}/{e['episodes']} h{e['hits']} w{e['wrong_way']} f{e['false_calls']}({e['false_clusters']})"
    c, q = scores.get(("le70", "candidate")), scores.get(("le70", "persistence"))
    if c:
        s["le70_brier"] = c["ride_le70"]["brier"]
        s["le70_pers"] = q["ride_le70"]["brier"]
    return s


def timed_run(cfg, period="tune", **kw):
    started = time.time()
    pred = run(cfg, period, **kw)
    return pred, time.time() - started
