"""Configurations chosen on the tuning half (2026-09-10..09-26 MYT) only.

Tuning evidence: data\\models\\diurnal\\tune_*.json (summarised in REPORT.md).
Each entry maps a head group to (model factory, lookback days). Embargoes are
fixed per group in models.GROUPS (330 min; 120 min for the event heads).
"""
from __future__ import annotations

import numpy as np

from .models import DiurnalHGB, KernelClimatology

INF = np.inf


def _hgb(feature_set):
    return lambda: DiurnalHGB(feature_set=feature_set, min_samples_leaf=1000, learning_rate=0.05, max_iter=100,
                              loss="absolute_error")


CHOSEN = {
    # The hypothesis in its pure form: clock of day only (best hour-only bandwidths on the tuning half).
    "pure_diurnal": {
        "near": (lambda: KernelClimatology(bw_hour=2.0, bw_level=INF, bw_trend=INF, alpha=5.0), 28),
        "events": (lambda: KernelClimatology(bw_hour=2.0, bw_level=INF, bw_trend=INF, alpha=5.0), 28),
        "le70": (lambda: KernelClimatology(bw_hour=2.0, bw_level=INF, bw_trend=INF, alpha=5.0), 28),
        "morning": (lambda: KernelClimatology(bw_level=INF, bw_trend=INF, bw_hour_session=1.0), 28),
        "afternoon": (lambda: KernelClimatology(bw_level=INF, bw_trend=INF, bw_hour_session=1.0), 28),
    },
    # Best kernel per head group (tuning chose an infinite clock bandwidth for the near heads).
    "kernel_best": {
        "near": (lambda: KernelClimatology(bw_hour=INF, bw_level=0.3, bw_trend=3.0, alpha=5.0), 28),
        "events": (lambda: KernelClimatology(bw_hour=6.0, bw_level=0.3, bw_trend=5.0, alpha=5.0), 28),
        "le70": (lambda: KernelClimatology(bw_hour=INF, bw_level=0.3, bw_trend=3.0, alpha=5.0), 28),
        "morning": (lambda: KernelClimatology(bw_level=0.15, bw_trend=INF, bw_hour_session=0.25), 28),
        "afternoon": (lambda: KernelClimatology(bw_level=0.3, bw_trend=INF, bw_hour_session=2.0), 28),
    },
    # Best HGB per head group (time + sensor + diurnal analogs + Open-Meteo mixing-layer/rain drivers).
    "hgb_best": {
        "near": (_hgb("diurnal_nwp"), 45),
        "events": (_hgb("diurnal_nwp"), 45),
        "le70": (_hgb("diurnal_nwp"), 45),
        "morning": (_hgb("diurnal_nwp"), 45),
        "afternoon": (_hgb("time"), 45),
    },
    # Ablations (not selected, run after the final configs were fixed): the same HGB with clock +
    # sensor level/trend only, and with clock + sensor + Open-Meteo drivers but no diurnal analogs.
    "hgb_time": {
        "near": (_hgb("time"), 45),
        "events": (_hgb("time"), 45),
    },
    "hgb_time_nwp": {
        "near": (_hgb("time_nwp"), 45),
        "events": (_hgb("time_nwp"), 45),
    },
}

# The candidate: per head group, the family that won on the tuning half.
CANDIDATE = "diurnal_v1"
COMPOSE = {"near": "kernel_best", "events": "hgb_best", "le70": "hgb_best",
           "morning": "kernel_best", "afternoon": "kernel_best"}
