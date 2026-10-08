"""Deterministic experimental 20 ug/m3 first-crossing probability helpers.

These functions only fit supplied historical frames or predict from supplied
feature frames. They do not load data, fetch providers, activate a live model,
or estimate excursion magnitude. Classes are always [drop, none, rise].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASSES = (-1, 0, 1)
NAMES = ("drop", "none", "rise")
LEADS = (15, 30, 45, 60, 75, 90)
LOCAL = ("fresh", "freshOffset", "delta15", "delta30", "delta60", "delta120",
         "sd60", "mean3Offset", "temp", "rh", "tempDelta30", "rhDelta30")
WEATHER = ("stepRainMm", "stepRainProbabilityPct", "stepWindUKmh", "stepWindVKmh",
           "stepWindUDeltaKmh", "stepWindVDeltaKmh", "stepGustEnvelopeKmh",
           "stepTempDeltaC", "stepRhDeltaPct", "stepVentilationM2S", "weatherAgeHours")
INDICATORS = tuple("step" + str(lead) for lead in LEADS)
FEATURES = (*LOCAL, *WEATHER, *INDICATORS)
POOLED_FEATURES = (*FEATURES, "threshold20", *("threshold20_" + x for x in INDICATORS))
LOGISTIC = {"C": 0.1, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4,
            "class_weight": None, "random_state": 1749}
MODEL_VERSION = "ttdi_large20_first_crossing_logistic_development_v1"


def risk_rows(issues, step_features, threshold=20):
    """One row per still-at-risk step; stop at the first threshold cause."""
    if threshold not in (10, 20):
        raise ValueError("Only the frozen 10 and 20 thresholds are supported")
    cause, first = ("cause", "firstStep") if threshold == 10 else ("cause20", "firstStep20")
    if issues.issueEpoch.duplicated().any():
        raise ValueError("Training issues must be unique")
    labels = issues[["issueEpoch", cause, first]].rename(columns={cause: "targetCause", first: "targetFirstStep"})
    out = step_features.merge(labels, on="issueEpoch", how="inner", validate="many_to_one")
    if not out.groupby("issueEpoch").step.apply(lambda x: sorted(x.tolist()) == list(range(6))).all():
        raise ValueError("Every training issue requires six unique feature steps")
    if set(out.issueEpoch) != set(issues.issueEpoch):
        raise ValueError("Missing training feature issues")
    out = out.loc[out.targetFirstStep.eq(-1) | out.step.le(out.targetFirstStep)].copy()
    out["riskLabel"] = np.where(out.step.eq(out.targetFirstStep), out.targetCause, 0).astype(int)
    if not set(out.riskLabel).issubset(CLASSES):
        raise ValueError("Unknown training cause")
    out["threshold20"] = float(threshold == 20)
    for name in INDICATORS:
        out["threshold20_" + name] = out["threshold20"] * out[name]
    return out.sort_values(["issueEpoch", "step"]).reset_index(drop=True)


def fit_hazard(issues, step_features, kind="dedicated20"):
    """Fit one fixed candidate; callers enforce chronological source guards.

    Pooled copies use weight 0.5 per at-risk row. Copies and steps from an
    issue never count as independent training cases. No class upweighting.
    """
    if kind not in ("dedicated20", "pooled10_20"):
        raise ValueError("Unknown frozen candidate")
    risk20 = risk_rows(issues, step_features, 20)
    if set(risk20.riskLabel) != set(CLASSES):
        raise ValueError("Dedicated 20 target needs all three classes represented")
    risk = risk20
    columns = FEATURES
    if kind == "pooled10_20":
        risk = pd.concat([risk_rows(issues, step_features, 10), risk20], ignore_index=True)
        columns = POOLED_FEATURES
    x = risk[list(columns)].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    weights = np.full(len(risk), 0.5 if kind == "pooled10_20" else 1.0)
    estimator = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), LogisticRegression(**LOGISTIC))
    with threadpool_limits(limits=1):
        estimator.fit(x, risk.riskLabel.to_numpy(int), logisticregression__sample_weight=weights)
    if tuple(estimator.classes_) != CLASSES:
        raise ValueError("Fitted cause ordering mismatch")
    return {"version": MODEL_VERSION, "kind": kind, "estimator": estimator,
            "columns": tuple(columns), "riskRows": len(risk),
            "weightedRiskRows": float(weights.sum()), "uniqueTrainingIssues": len(issues),
            "thresholds": (10, 20) if kind == "pooled10_20" else (20,)}


def predict_hazards(model, query_steps, threshold=20):
    """Return issue-sorted n x 6 x 3 conditional hazards for one threshold.

    A pooled model produces separately absorbing 10 and 20 distributions.
    A first direction at one threshold need not match that at the other;
    callers must not impose directional probability nesting across thresholds.
    """
    if threshold not in model["thresholds"]:
        raise ValueError("Threshold was not included in the fitted model")
    ordered = query_steps.sort_values(["issueEpoch", "step"]).copy()
    if ordered.empty:
        return np.empty((0, 6, 3), float)
    if not ordered.groupby("issueEpoch").step.apply(lambda x: x.tolist() == list(range(6))).all():
        raise ValueError("Every query requires exactly six ordered feature steps")
    ordered["threshold20"] = float(threshold == 20)
    for name in INDICATORS:
        ordered["threshold20_" + name] = ordered["threshold20"] * ordered[name]
    x = ordered[list(model["columns"])].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    with threadpool_limits(limits=1):
        raw = model["estimator"].predict_proba(x)
    classes = list(model["estimator"].classes_)
    hazards = np.column_stack([raw[:, classes.index(code)] for code in CLASSES]).reshape(-1, 6, 3)
    references = ordered.fresh.to_numpy(float).reshape(-1, 6)
    if not np.isfinite(references).all() or (references < 0).any() or not (references == references[:, :1]).all():
        raise ValueError("Every query requires one finite nonnegative exact reference")
    impossible_drop = references[:, 0] < threshold
    hazards[impossible_drop, :, 1] += hazards[impossible_drop, :, 0]
    hazards[impossible_drop, :, 0] = 0.0
    return hazards


def cumulative_incidence(hazards):
    hazards = np.asarray(hazards, float)
    if hazards.ndim != 3 or hazards.shape[1:] != (6, 3):
        raise ValueError("Expected issue x six steps x three causes")
    if not np.isfinite(hazards).all() or (hazards < 0).any() or not np.allclose(hazards.sum(axis=2), 1, atol=1e-8):
        raise ValueError("Invalid conditional probability mass")
    out = np.empty_like(hazards)
    survival = np.ones(len(hazards))
    incidence = np.zeros((len(hazards), 2))
    for i in range(6):
        incidence += survival[:, None] * hazards[:, i, [0, 2]]
        survival *= hazards[:, i, 1]
        out[:, i, 0], out[:, i, 1], out[:, i, 2] = incidence[:, 0], survival, incidence[:, 1]
    if not np.allclose(out.sum(axis=2), 1, atol=1e-8):
        raise ValueError("Invalid cumulative probability mass")
    return out


def direction_calls(probabilities, rule="argmax"):
    """Equal error cost: choose largest of three classes, ties resolve none."""
    p = np.asarray(probabilities, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Expected valid cumulative drop/none/rise probabilities")
    called = np.zeros(len(p), int)
    if rule == "argmax":
        unique = (p == p.max(axis=1, keepdims=True)).sum(axis=1) == 1
        called[unique] = np.asarray(CLASSES)[p[unique].argmax(axis=1)]
    elif rule == "threshold_0_5":
        drop, rise = p[:, 0] >= 0.5, p[:, 2] >= 0.5
        called[drop & ~rise], called[rise & ~drop] = -1, 1
    else:
        raise ValueError("Unknown direction decision rule")
    return called
