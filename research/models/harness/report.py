"""Baseline table (persistence vs incumbent replay vs genuinely issued) and big-event cases.

    python -B -m harness.report        # writes research/models/BASELINES.md and data/models/baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, dataset, issued, scoring
from .incumbents import run as runs

REPORT_PATH = Path(__file__).resolve().parents[1] / "BASELINES.md"
JSON_PATH = config.OUT / "baselines.json"
ISSUED_START = {"ridge": config.myt_epoch("2026-10-09 17:44"),
                "arrival_logistic": config.myt_epoch("2026-10-08 19:11"),
                "first_crossing": config.myt_epoch("2026-10-08 17:45"),
                "cycling70": config.myt_epoch("2026-10-08")}

CASES = [  # (label, MYT window start, window end, direction)
    ("Sep 28 rise", "2026-09-28 11:00", "2026-09-28 15:00", 1),
    ("Sep 29 clearing", "2026-09-29 18:00", "2026-09-29 21:30", -1),
    ("Oct 1 storm fall", "2026-10-01 15:30", "2026-10-01 19:00", -1),
    ("Oct 6 surge, start (ref ~65)", "2026-10-06 03:00", "2026-10-06 06:30", 1),
    ("Oct 6 surge, late (to ~160)", "2026-10-06 09:00", "2026-10-06 11:30", 1),
    ("Oct 8 fall", "2026-10-08 14:30", "2026-10-08 18:30", -1),
    ("Oct 9 rain drop", "2026-10-09 13:30", "2026-10-09 16:00", -1),
    ("Oct 13 rain drop", "2026-10-13 16:00", "2026-10-13 19:00", -1),
]


# ----------------------------------------------------------------------------- standard columns
def standard_replay(ds: pd.DataFrame) -> pd.DataFrame:
    """All grid replays mapped to the scorer's standard prediction columns."""
    out = pd.DataFrame(index=ds.index)
    try:
        ridge = runs.load("ridge_v4")
        for c in ("pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max"):
            out[c] = ridge[f"ridge_v4__{c}"]
    except FileNotFoundError:
        pass
    try:
        logistic = runs.load("arrival_logistic")
        names = ["p_fall40", "p_fall20to40", "p_within20", "p_rise20to40", "p_rise40"]
        for k, name in enumerate(names):
            out[f"p_class5_{k}"] = logistic[f"arrival_logistic__{name}"]
        out["p_fall20"] = logistic["arrival_logistic__p_fall20"]
        out["p_rise20"] = logistic["arrival_logistic__p_rise20"]
        out["p_fall40"] = logistic["arrival_logistic__p_fall40"]
        out["p_rise40"] = logistic["arrival_logistic__p_rise40"]
    except FileNotFoundError:
        pass
    try:
        first = runs.load("first_crossing")
        for a, b in (("p_drop", "p_first_drop"), ("p_none", "p_first_none"), ("p_rise", "p_first_rise")):
            out[b] = first[f"first_crossing__{a}"]
    except FileNotFoundError:
        pass
    try:
        out["p_ride_le70"] = runs.load("cycling70")["cycling70__p_ride_le70"]
    except FileNotFoundError:
        pass
    try:
        out["pred_morning"] = runs.load("morning_hgb")["morning_hgb__pred_morning"]
    except FileNotFoundError:
        pass
    try:
        patch = runs.load("afternoon_patchtst")
        grid = patch.copy()
        local = pd.to_datetime(grid.issue_epoch, unit="s", utc=True).dt.tz_convert("Asia/Kuala_Lumpur")
        grid["grid_epoch"] = (local.dt.floor("5min").astype("int64") // 10**9).to_numpy()
        series = grid.drop_duplicates("grid_epoch").set_index("grid_epoch")["pred"]
        out["pred_afternoon"] = series.reindex(out.index)
    except FileNotFoundError:
        pass
    return out.reindex(ds.index)


def _fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}f}"


def _events_cell(result):
    if not result or not result.get("n_issues"):
        return "–"
    return (f"{result['episodes_caught']}/{result['episodes']} ep; hits {result['hits']}, miss {result['misses']}, "
            f"wrong {result['wrong_way']}, false {result['false_calls']} ({result['false_clusters']} cl)")


# ----------------------------------------------------------------------------- grid table
def grid_scores(ds, replay):
    evaluate = ds.loc[ds["eval"] & ds.m_fresh_available]
    persistence = scoring.persistence_predictions(evaluate)
    rows = []
    # identical rows: where the replay produced an output for that head
    heads = [("+90 arrival", "pred_arrival", "arrival", "y_arrival"),
             ("ride mean +90..210", "pred_ride_mean", "ride_mean", "y_ride_mean"),
             ("ride min", "pred_ride_min", "ride_min", "y_ride_min"),
             ("ride max", "pred_ride_max", "ride_max", "y_ride_max")]
    for label, column, key, target in heads:
        if column not in replay:
            continue
        mask = replay[column].notna() & evaluate[target].notna()
        p = scoring.score_candidate(evaluate, persistence[[column]], mask=mask)
        r = scoring.score_candidate(evaluate, replay[[column]], mask=mask)
        rows.append((label, "MAE", _fmt(p[key]["mae"]), _fmt(r[key]["mae"]), r[key]["n"]))
        rows.append((label, "skill vs persistence", "0", _fmt(r[key]["skill_vs_persistence"], 3), r[key]["n"]))
        rows.append((label, "MAE where |true change| >= 20", _fmt(p[key]["mae_change20"]),
                     _fmt(r[key]["mae_change20"]), r[key]["n_change20"]))
        if key == "ride_min":
            rows.append((label, "pinball q0.1", _fmt(scoring.pinball(evaluate.y_ride_min[mask], persistence.pred_ride_min[mask], .1)),
                         _fmt(scoring.pinball(evaluate.y_ride_min[mask], replay.pred_ride_min[mask], .1)), int(mask.sum())))
        if key == "ride_max":
            rows.append((label, "pinball q0.9", _fmt(scoring.pinball(evaluate.y_ride_max[mask], persistence.pred_ride_max[mask], .9)),
                         _fmt(scoring.pinball(evaluate.y_ride_max[mask], replay.pred_ride_max[mask], .9)), int(mask.sum())))
        if key == "arrival":
            for t in (20, 40):
                rows.append((f"arrival change >={t} (Ridge point)", "events",
                             _events_cell(p[f"events{t}_point"]), _events_cell(r[f"events{t}_point"]),
                             r[f"events{t}_point"]["n_issues"]))
    if "p_fall20" in replay:
        mask = replay.p_fall20.notna() & evaluate.y_delta.notna()
        p = scoring.score_candidate(evaluate, persistence[["p_fall20", "p_rise20", "p_fall40", "p_rise40"]], mask=mask)
        cols = ["p_fall20", "p_rise20", "p_fall40", "p_rise40"] + [f"p_class5_{k}" for k in range(5)]
        r = scoring.score_candidate(evaluate, replay[cols], mask=mask)
        prior = scoring.walk_forward_prior(evaluate, "y_class5", range(5), "y_arrival_complete", history=ds)
        prior.columns = [f"p_class5_{k}" for k in range(5)]
        prior["p_fall20"] = prior.p_class5_0 + prior.p_class5_1
        prior["p_rise20"] = prior.p_class5_3 + prior.p_class5_4
        prior["p_fall40"], prior["p_rise40"] = prior.p_class5_0, prior.p_class5_4
        q = scoring.score_candidate(evaluate, prior, mask=mask)
        for t in (20, 40):
            for rule in ("argmax", "0.2"):
                rows.append((f"arrival change >={t} (logistic, {rule})", "events",
                             _events_cell(p[f"events{t}_prob_{rule}"]), _events_cell(r[f"events{t}_prob_{rule}"]),
                             r[f"events{t}_prob_{rule}"]["n_issues"]))
            for side in ("fall", "rise"):
                key = f"brier_{side}{t}"
                rows.append((f"P({side} >={t})", "Brier (prior | model)", _fmt(q[key]["brier"], 4),
                             _fmt(r[key]["brier"], 4), r[key]["n"]))
        rows.append(("5-class arrival change", "Brier (prior | model)", _fmt(q["class5"]["brier"], 4),
                     _fmt(r["class5"]["brier"], 4), r["class5"]["n"]))
        rows.append(("5-class arrival change", "log loss (prior | model)", _fmt(q["class5"]["log_loss"], 4),
                     _fmt(r["class5"]["log_loss"], 4), r["class5"]["n"]))
    if "p_first_drop" in replay:
        mask = replay.p_first_drop.notna() & evaluate.y_first20.notna()
        r = scoring.score_candidate(evaluate, replay[["p_first_drop", "p_first_none", "p_first_rise"]], mask=mask)
        prior = scoring.walk_forward_prior(evaluate, "y_first20", (-1, 0, 1), "y_first20_complete", history=ds)
        prior.columns = ["p_first_drop", "p_first_none", "p_first_rise"]
        q = scoring.score_candidate(evaluate, prior, mask=mask)
        none_call = pd.DataFrame({"p_first_drop": 0.0, "p_first_none": 1.0, "p_first_rise": 0.0}, index=evaluate.index)
        z = scoring.score_candidate(evaluate, none_call, mask=mask)
        rows.append(("first crossing >=20 in 90 min", "events (argmax)", _events_cell(z["first20_events"]),
                     _events_cell(r["first20_events"]), r["first20_events"]["n_issues"]))
        rows.append(("first crossing >=20 in 90 min", "Brier (prior | model)", _fmt(q["first20"]["brier"], 4),
                     _fmt(r["first20"]["brier"], 4), r["first20"]["n"]))
        rows.append(("first crossing >=20 in 90 min", "log loss (prior | model)", _fmt(q["first20"]["log_loss"], 4),
                     _fmt(r["first20"]["log_loss"], 4), r["first20"]["n"]))
    if "p_ride_le70" in replay:
        mask = replay.p_ride_le70.notna() & evaluate.y_ride_le70.notna()
        p = scoring.score_candidate(evaluate, persistence[["p_ride_le70"]], mask=mask)
        r = scoring.score_candidate(evaluate, replay[["p_ride_le70"]], mask=mask)
        prior = scoring.walk_forward_prior(evaluate, "y_ride_le70", (0.0, 1.0), "y_ride_complete", history=ds)
        q = scoring.score_candidate(evaluate, prior[["prior_1.0"]].rename(columns={"prior_1.0": "p_ride_le70"}), mask=mask)
        rows.append(("P(ride mean <=70)", "Brier (persistence 0/1 | model)", _fmt(p["ride_le70"]["brier"], 4),
                     _fmt(r["ride_le70"]["brier"], 4), r["ride_le70"]["n"]))
        rows.append(("P(ride mean <=70)", "Brier (prior | model)", _fmt(q["ride_le70"]["brier"], 4),
                     _fmt(r["ride_le70"]["brier"], 4), r["ride_le70"]["n"]))
    return rows


def session_rows(ds, replay):
    rows = []
    heads = [("Morning same day @07:15", "07:15", "pred_morning", "morning", True),
             ("Morning next day @18:05", "18:05", "pred_morning", "morning", False),
             ("Morning next day @12:00", "12:00", "pred_morning", "morning", False),
             ("Afternoon same day @12:00", "12:00", "pred_afternoon", "afternoon", True),
             ("Afternoon next day @18:05 (Oct 8+)", "18:05", "pred_afternoon", "afternoon", False),
             ("Afternoon next day @20:00 (prefix head)", "20:00", "pred_afternoon", "afternoon", False)]
    evaluate = ds.loc[ds["eval"]]
    clock = pd.to_datetime(evaluate.index, unit="s", utc=True).tz_convert("Asia/Kuala_Lumpur").strftime("%H:%M")
    for label, at, column, key, same in heads:
        if column not in replay:
            continue
        mask = pd.Series((clock == at), index=evaluate.index) & replay[column].reindex(evaluate.index).notna() \
            & evaluate[f"y_{key}_mean"].notna() & (evaluate[f"y_{key}_same_day"] == same)
        if not mask.any():
            continue
        frame = evaluate.loc[mask]
        r = scoring.point_metrics(frame[f"y_{key}_mean"], replay.loc[frame.index, column], frame.y_ref)
        rows.append((label, "MAE", _fmt(r["persistence_mae"]), _fmt(r["mae"]), r["n"]))
    return rows


# ----------------------------------------------------------------------------- issued period
def _daily_prior(ds, column, values, complete):
    prior = scoring.walk_forward_prior(ds, column, values, complete)
    prior["day"] = [config.midnight(i) for i in prior.index]
    return prior.groupby("day").first()


def issued_rows(ds):
    frame = runs.issued_clock_frame()
    day = pd.Series([config.midnight(i) for i in frame.index], index=frame.index)
    class_prior = _daily_prior(ds, "y_class5", range(5), "y_arrival_complete")
    replay = runs.load("issued_clocks")
    rows, fidelity = [], []
    standard = pd.DataFrame(index=frame.index)
    for c in ("pred_arrival", "pred_ride_mean", "pred_ride_min", "pred_ride_max"):
        standard[c] = replay[f"ridge_v4__{c}"]
    issued_std = pd.DataFrame({"pred_arrival": frame.issued__arrival_point, "pred_ride_mean": frame.issued__ride_mean_point,
                               "pred_ride_min": frame.issued__ride_min_point, "pred_ride_max": frame.issued__ride_max_point},
                              index=frame.index)
    ridge_ok = (frame.issued__arrival_model == issued.INCUMBENT_VERSIONS["ridge"]) & (frame.index >= ISSUED_START["ridge"])
    for label, column, key, target in (("+90 arrival", "pred_arrival", "arrival", "y_arrival"),
                                       ("ride mean", "pred_ride_mean", "ride_mean", "y_ride_mean"),
                                       ("ride min", "pred_ride_min", "ride_min", "y_ride_min"),
                                       ("ride max", "pred_ride_max", "ride_max", "y_ride_max")):
        mask = ridge_ok & issued_std[column].notna() & standard[column].notna() & frame[target].notna()
        sub = frame.loc[mask]
        i = scoring.point_metrics(sub[target], issued_std.loc[mask, column], sub.y_ref)
        r = scoring.point_metrics(sub[target], standard.loc[mask, column], sub.y_ref)
        rows.append((label, "MAE", _fmt(i["persistence_mae"]), _fmt(r["mae"]), _fmt(i["mae"]), i["n"]))
        both = ridge_ok & issued_std[column].notna() & standard[column].notna()
        diff = (issued_std.loc[both, column] - standard.loc[both, column]).abs()
        fidelity.append((f"Ridge v4 {label}", int(both.sum()), _fmt(float((diff < 1e-6).mean()), 3),
                         _fmt(diff.median(), 3), _fmt(diff.quantile(.95), 3), _fmt(diff.max(), 3)))
    for t in (20, 40):
        mask = ridge_ok & issued_std.pred_arrival.notna() & standard.pred_arrival.notna() & frame.y_delta.notna()
        sub = frame.loc[mask]
        truth = scoring.truth_direction(sub.y_delta, t)
        issues = sub.index.to_numpy(np.int64)
        e_p = scoring.event_confusion(issues, truth, np.zeros(len(sub)))
        e_r = scoring.event_confusion(issues, truth, scoring.calls_from_point(standard.loc[mask, "pred_arrival"], sub.y_ref, t))
        e_i = scoring.event_confusion(issues, truth, scoring.calls_from_point(issued_std.loc[mask, "pred_arrival"], sub.y_ref, t))
        rows.append((f"change >={t} (Ridge point)", "events", _events_cell(e_p), _events_cell(e_r), _events_cell(e_i), len(sub)))
    ac_ok = (frame.issued__ac_model == issued.INCUMBENT_VERSIONS["arrival_logistic"]) & frame.issued__ac_p_fall20.notna()
    mask = ac_ok & replay["arrival_logistic__p_fall20"].notna() & frame.y_delta.notna()
    sub = frame.loc[mask]
    issues = sub.index.to_numpy(np.int64)
    for t in (20, 40):
        truth = scoring.truth_direction(sub.y_delta, t)
        fall_i, rise_i = sub[f"issued__ac_p_fall{t}"], sub[f"issued__ac_p_rise{t}"]
        fall_r, rise_r = replay.loc[mask, f"arrival_logistic__p_fall{t}"], replay.loc[mask, f"arrival_logistic__p_rise{t}"]
        e_p = scoring.event_confusion(issues, truth, np.zeros(len(sub)))
        e_r = scoring.event_confusion(issues, truth, scoring.calls_from_probs(fall_r, rise_r))
        e_i = scoring.event_confusion(issues, truth, scoring.calls_from_probs(fall_i, rise_i))
        rows.append((f"change >={t} (logistic argmax)", "events", _events_cell(e_p), _events_cell(e_r), _events_cell(e_i), len(sub)))
        classes = {("fall", 20): [0, 1], ("rise", 20): [3, 4], ("fall", 40): [0], ("rise", 40): [4]}
        for side, fi, fr in (("fall", fall_i, fall_r), ("rise", rise_i, rise_r)):
            prior = class_prior[[f"prior_{k}" for k in classes[(side, t)]]].sum(axis=1)
            bp = scoring.binary_brier(day.loc[sub.index].map(prior), sub[f"y_{side}{t}"])
            bi = scoring.binary_brier(fi, sub[f"y_{side}{t}"])
            br = scoring.binary_brier(fr, sub[f"y_{side}{t}"])
            rows.append((f"P({side} >={t})", "Brier (prior | replay | issued)", _fmt(bp["brier"], 4),
                         _fmt(br["brier"], 4), _fmt(bi["brier"], 4), bi["n"]))
    for name in ("p_fall40", "p_fall20to40", "p_within20", "p_rise20to40", "p_rise40"):
        both = ac_ok & replay[f"arrival_logistic__{name}"].notna()
        diff = (frame.loc[both, f"issued__ac_{name}"] - replay.loc[both, f"arrival_logistic__{name}"]).abs()
        fidelity.append((f"logistic {name}", int(both.sum()), _fmt(float((diff < 1e-6).mean()), 3),
                         _fmt(diff.median(), 4), _fmt(diff.quantile(.95), 4), _fmt(diff.max(), 4)))
    fc_ok = (frame.issued__fc_model == issued.INCUMBENT_VERSIONS["first_crossing"]) & frame.issued__fc_p_drop.notna()
    for a, b in (("p_drop", "fc_p_drop"), ("p_rise", "fc_p_rise")):
        both = fc_ok & replay[f"first_crossing__{a}"].notna()
        diff = (frame.loc[both, f"issued__{b}"] - replay.loc[both, f"first_crossing__{a}"]).abs()
        fidelity.append((f"first-crossing {a}", int(both.sum()), _fmt(float((diff < 1e-6).mean()), 3),
                         _fmt(diff.median(), 4), _fmt(diff.quantile(.95), 4), _fmt(diff.max(), 4)))
    mask = fc_ok & replay["first_crossing__p_drop"].notna() & frame.y_first20.notna()
    sub = frame.loc[mask]
    if len(sub):
        def calls(p):
            arr = np.argmax(np.nan_to_num(p, nan=-1), axis=1) - 1.0
            return arr
        p_i = sub[["issued__fc_p_drop", "issued__fc_p_none", "issued__fc_p_rise"]].to_numpy(float)
        p_r = replay.loc[mask, ["first_crossing__p_drop", "first_crossing__p_none", "first_crossing__p_rise"]].to_numpy(float)
        issues = sub.index.to_numpy(np.int64)
        e_p = scoring.event_confusion(issues, sub.y_first20, np.zeros(len(sub)))
        rows.append(("first crossing >=20", "events (argmax)", _events_cell(e_p),
                     _events_cell(scoring.event_confusion(issues, sub.y_first20, calls(p_r))),
                     _events_cell(scoring.event_confusion(issues, sub.y_first20, calls(p_i))), len(sub)))
    cy_ok = (frame.issued__cy_model == issued.INCUMBENT_VERSIONS["cycling70"]) & frame.issued__cy_p_le70.notna()
    mask = cy_ok & replay["cycling70__p_ride_le70"].notna() & frame.y_ride_le70.notna()
    sub = frame.loc[mask]
    if len(sub):
        persistence = (sub.y_ref <= 70).astype(float)
        rows.append(("P(ride mean <=70)", "Brier", _fmt(scoring.binary_brier(persistence, sub.y_ride_le70)["brier"], 4),
                     _fmt(scoring.binary_brier(replay.loc[mask, "cycling70__p_ride_le70"], sub.y_ride_le70)["brier"], 4),
                     _fmt(scoring.binary_brier(sub.issued__cy_p_le70, sub.y_ride_le70)["brier"], 4), len(sub)))
    both = cy_ok & replay["cycling70__p_ride_le70"].notna()
    diff = (frame.loc[both, "issued__cy_p_le70"] - replay.loc[both, "cycling70__p_ride_le70"]).abs()
    fidelity.append(("cycling P(mean<=70)", int(both.sum()), _fmt(float((diff < 1e-6).mean()), 3),
                     _fmt(diff.median(), 4), _fmt(diff.quantile(.95), 4), _fmt(diff.max(), 4)))
    return rows, fidelity


def issued_session_rows(ds, replay):
    ledger = issued.ledger()
    canonical = issued.canonical_sample(ledger.loc[ledger.issue_epoch >= config.myt_epoch("2026-10-08")])
    rows, fidelity = [], []
    for label, clock, key, same in (("Morning same day @07:15", "07:15", "morning", True),
                                    ("Morning next day @18:05", "18:05", "morning", False),
                                    ("Morning next day @12:00", "12:00", "morning", False),
                                    ("Afternoon same day @12:00", "12:00", "afternoon", True),
                                    ("Afternoon next day @18:05", "18:05", "afternoon", False)):
        part = canonical.loc[(canonical.canonical == clock) & (canonical[f"{key}_model"] == issued.INCUMBENT_VERSIONS[key])]
        records = []
        for _, row in part.iterrows():
            tick = int(row.canonical_tick)
            if tick not in ds.index:
                continue
            target = ds.loc[tick]
            if int(target[f"y_{key}_start"]) != int(row[f"{key}_start"]) or bool(target[f"y_{key}_same_day"]) != same:
                continue
            records.append((tick, target[f"y_{key}_mean"], row[f"{key}_point"], row[f"{key}_baseline"],
                            replay.get(f"pred_{key}", pd.Series(dtype=float)).get(tick, np.nan)))
        frame = pd.DataFrame(records, columns=["tick", "y", "issued", "baseline", "replay"]).dropna(subset=["y", "issued"])
        if not len(frame):
            continue
        i = scoring.point_metrics(frame.y, frame.issued, frame.baseline)
        r = scoring.point_metrics(frame.y, frame.replay, frame.baseline) if frame.replay.notna().any() else {}
        rows.append((label, "MAE", _fmt(i["persistence_mae"]), _fmt(r.get("mae")) + (f" (n={r['n']})" if r.get("n") else ""),
                     _fmt(i["mae"]), i["n"]))
        both = frame.dropna(subset=["replay"])
        if len(both):
            diff = (both.issued - both.replay).abs()
            fidelity.append((f"{label}", len(both), _fmt(float((diff < 1e-6).mean()), 3),
                             _fmt(diff.median(), 2), _fmt(diff.quantile(.95), 2), _fmt(diff.max(), 2)))
    return rows, fidelity


def jsonl_consistency():
    """Ledger (published payload) vs the private issued JSONL records at the same issue clocks."""
    ledger = issued.ledger().drop_duplicates("issue_epoch").set_index("issue_epoch")
    tables = issued.jsonl_tables()
    rows = []
    logistic = tables["arrival_logistic"].dropna(subset=["classProbabilities"]).drop_duplicates("forecastIssuedEpoch")
    logistic = logistic.set_index("forecastIssuedEpoch").join(ledger, how="inner")
    if len(logistic):
        jl = np.vstack(logistic.classProbabilities.to_numpy())
        lg = logistic[["ac_p_fall40", "ac_p_fall20to40", "ac_p_within20", "ac_p_rise20to40", "ac_p_rise40"]].to_numpy(float)
        ok = np.isfinite(lg).all(axis=1)
        rows.append(("logistic class probabilities", int(ok.sum()), _fmt(float(np.nanmax(np.abs(jl[ok] - lg[ok]))) if ok.any() else np.nan, 6)))
    first = tables["first_crossing"].dropna(subset=["probabilityDrop"]).drop_duplicates("forecastIssuedEpoch")
    first = first.set_index("forecastIssuedEpoch").join(ledger, how="inner")
    ok = first.fc_p_drop.notna()
    if ok.any():
        rows.append(("first-crossing P(drop)", int(ok.sum()), _fmt(float((first.probabilityDrop[ok] - first.fc_p_drop[ok]).abs().max()), 6)))
    cyc = tables["cycling70"]
    cyc = cyc.loc[cyc["mode"] == "initial90"].dropna(subset=["chanceMeanAtOrBelowCutoff"]).drop_duplicates("forecastIssuedEpoch")
    cyc = cyc.set_index("forecastIssuedEpoch").join(ledger, how="inner")
    ok = cyc.cy_p_le70.notna()
    if ok.any():
        rows.append(("cycling P(mean<=70)", int(ok.sum()), _fmt(float((cyc.chanceMeanAtOrBelowCutoff[ok] - cyc.cy_p_le70[ok]).abs().max()), 6)))
    cases = issued.prospective_cases()
    for family, column in (("point90", "arrival_point"), ("mean90_210", "ride_mean_point")):
        part = cases.loc[cases.family == family].dropna(subset=["point"]).drop_duplicates("issue_epoch")
        part = part.set_index("issue_epoch").join(ledger[[column]], how="inner").dropna(subset=[column])
        if len(part):
            rows.append((f"prospective_validation_issued {family}", len(part),
                         _fmt(float((part.point.astype(float) - part[column]).abs().max()), 6)))
    return rows


# ----------------------------------------------------------------------------- cases
def case_table(ds, replay):
    ledger = issued.ledger()
    lookup = ledger.sort_values("issue_epoch").drop_duplicates("issue_epoch").set_index("issue_epoch")
    lines = []
    for label, start, end, direction in CASES:
        a, b = config.myt_epoch(start), config.myt_epoch(end)
        part = ds.loc[(ds.index >= a) & (ds.index <= b)]
        truth = scoring.truth_direction(part.y_delta, 20)
        hits = part.index[truth == direction]
        if not len(hits):
            lines.append({"case": label, "onset": "no >=20 episode in window"})
            continue
        onset = int(hits[0])
        peak = int(part.loc[hits].y_delta.abs().idxmax())
        window = ds.loc[(ds.index >= onset - 3600) & (ds.index <= int(hits[-1]))]
        rp = replay.reindex(window.index)
        point_call = scoring.calls_from_point(rp.get("pred_arrival"), window.y_ref, 20) if "pred_arrival" in rp else np.full(len(window), np.nan)
        prob_call = (scoring.calls_from_probs(rp.p_fall20, rp.p_rise20) if "p_fall20" in rp else np.full(len(window), np.nan))
        first_call = (np.argmax(np.nan_to_num(rp[["p_first_drop", "p_first_none", "p_first_rise"]].to_numpy(float), nan=-1), axis=1) - 1.0
                      if "p_first_drop" in rp else np.full(len(window), np.nan))

        def first_right(calls):
            idx = np.flatnonzero(np.asarray(calls) == direction)
            return f"{(int(window.index[idx[0]]) - onset) / 60:+.0f}" if len(idx) else "never"

        def at(column, epoch):
            return replay[column].get(epoch, np.nan) if column in replay else np.nan

        side = "fall" if direction < 0 else "rise"
        prob = f"p_{side}20"
        first = "p_first_drop" if direction < 0 else "p_first_rise"
        nearest = lookup.index[(lookup.index >= onset) & (lookup.index < onset + 300)]
        issued_cell = "–"
        if len(nearest):
            row = lookup.loc[nearest[0]]
            parts = []
            if row.arrival_model == issued.INCUMBENT_VERSIONS["ridge"] and np.isfinite(row.arrival_point):
                parts.append(f"Ridge {row.arrival_point:.0f}")
            if row.ac_model == issued.INCUMBENT_VERSIONS["arrival_logistic"] and np.isfinite(row[f"ac_p_{side}20"]):
                parts.append(f"P{side}20 {row[f'ac_p_{side}20']:.2f}")
            fc_column = "fc_p_drop" if direction < 0 else "fc_p_rise"
            if row.fc_model == issued.INCUMBENT_VERSIONS["first_crossing"] and np.isfinite(row[fc_column]):
                parts.append(f"first {row[fc_column]:.2f}")
            issued_cell = ", ".join(parts) if parts else "older model version"
        o = ds.loc[onset]
        lines.append({
            "case": label, "onset": str(config.myt(onset))[5:16],
            "ref -> +90 at onset": f"{o.y_ref:.0f} -> {o.y_arrival:.0f}",
            "peak change": f"{ds.loc[peak].y_delta:+.0f} @{str(config.myt(peak))[11:16]}",
            "episode issues": int(len(hits)),
            "Ridge replay at onset": f"{at('pred_arrival', onset):.0f}" if np.isfinite(at('pred_arrival', onset)) else "–",
            "Ridge first right call (min vs onset, from -60)": first_right(point_call),
            "logistic P(dir>=20) onset-30 / onset": f"{side}: {at(prob, onset - 1800):.2f} / {at(prob, onset):.2f}",
            "logistic first argmax call": first_right(prob_call),
            "first-crossing P at onset": f"{at(first, onset):.2f}",
            "first-crossing first argmax call": first_right(first_call),
            "issued at onset (Oct 8+)": issued_cell,
        })
    return pd.DataFrame(lines)


# ----------------------------------------------------------------------------- render
def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def build():
    ds = dataset.load()
    replay = standard_replay(ds)
    grid = grid_scores(ds, replay)
    sessions = session_rows(ds, replay)
    issued_period, fidelity = issued_rows(ds)
    issued_sessions, session_fidelity = issued_session_rows(ds, replay)
    cases = case_table(ds, replay)
    evaluate = ds.loc[ds["eval"]]
    from . import prod
    drift = [m for m in prod.drift()["changedTopLevelModules"] if not m.startswith("test_")]
    text = [
        "# Baselines: persistence vs incumbent replay vs issued incumbent",
        "",
        f"Generated by `python -B -m harness.report` from `{config.DATASET_PATH.name}` "
        f"({len(ds)} issues, 5-min grid {ds.issue_myt.iloc[0]} .. {ds.issue_myt.iloc[-1]} MYT). "
        f"Evaluation period: {evaluate.issue_myt.iloc[0][:10]} .. {evaluate.issue_myt.iloc[-1][:10]} "
        f"({evaluate.date_myt.nunique()} days, {len(evaluate)} issues); training exists for >= 14 days before it.",
        "",
        "Incumbents are production code as of the snapshot (frozen mirror `data/models/prod_src_20261013`). "
        + ("Live server modules changed since the mirror (check whether model math changed before relying on the "
           "replays): " + ", ".join(drift) + "." if drift else "The live server sources still match the mirror."),
        "",
        "Events: `k/n ep` = episodes with at least one right-direction call / true episodes; issue counts of hits, "
        "misses (no call), wrong-way and false calls; `cl` = false-call clusters. Persistence never calls a change.",
        "",
        "## Grid: persistence vs incumbent replay (identical rows: where the replay produced an output)",
        "",
        _table(["Target", "Metric", "Persistence / prior", "Incumbent replay", "n"], grid),
        "",
        "## Sessions at canonical clocks (eval period)",
        "",
        _table(["Session", "Metric", "Persistence", "Incumbent replay", "n"], sessions),
        "",
        "## Genuinely issued incumbent (exact issued clocks, Oct 8+; one per 5-min slot)",
        "",
        "Rows where the published model version equals the current incumbent. The replay is the walk-forward refit "
        "evaluated at the same exact issue clock. Persistence = the issued reference (sessions: issued baselinePoint); "
        "for probabilities the first column is the walk-forward class prior, for P(mean<=70) the 0/1 persistence call.",
        "",
        _table(["Target", "Metric", "Persistence", "Replay (same clocks)", "Issued", "n"], issued_period + issued_sessions),
        "",
        "## Replay fidelity (|issued - replay| at identical clocks)",
        "",
        _table(["Output", "n", "share exact (<1e-6)", "median", "p95", "max"], fidelity + session_fidelity),
        "",
        "Ledger (published payload) vs private issued JSONL and the prospective_validation_issued table at the same clocks (max |diff|):",
        "",
        _table(["Output", "n", "max abs diff"], jsonl_consistency()),
        "",
        "## Big-event cases (>=20 at +90; onset = first grid issue whose +90 change qualifies)",
        "",
        "First right call = minutes from onset to the first issue (searching from onset-60 to the episode end) where the "
        "model calls the right direction (point: |pred-ref|>=20; probabilities: argmax). Negative = before the onset issue. "
        "Every call at or after onset still precedes the change at arrival by up to 90 min.",
        "",
        _table(list(cases.columns), cases.astype(str).values.tolist()),
        "",
    ]
    REPORT_PATH.write_text("\n".join(text), encoding="utf-8")
    JSON_PATH.write_text(json.dumps({"grid": grid, "sessions": sessions, "issued": issued_period + issued_sessions,
                                     "fidelity": fidelity + session_fidelity,
                                     "cases": cases.astype(str).to_dict("records")}, indent=1, default=str),
                         encoding="utf-8")
    return "\n".join(text)


if __name__ == "__main__":
    print(build())
