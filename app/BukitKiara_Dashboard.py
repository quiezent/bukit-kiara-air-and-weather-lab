from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, parse_qs, urlencode
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager
import csv
import copy
from collections import OrderedDict
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import threading
import time
import webbrowser

import numpy as np
import pandas as pd

# Also support snapshot audit imports from a different working directory.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import haze_transport
import window_pm_predictor
import afternoon_direction_forecast
import cams_local_correction
from collection_quality import bucket_coverage, collection_summary
from forecast_archive import record_dashboard_issue
from forecast_clock import ForecastClock, target_series, score_observed_window, targets_match
import forecast_freshness
import rain_pm_forecast
from analysis_delivery import AnalysisDelivery

# ============================================================
# Bukit Kiara / TTDI AirGradient Ride Dashboard
# Pandas provides time-bucketed particle and arrival-error analysis.
# ============================================================

# Public distribution is loopback-only unless the operator opts into LAN access.
HOST = os.environ.get("BUKIT_KIARA_HOST", "127.0.0.1")
PORT = int(os.environ.get("BUKIT_KIARA_PORT", "8765"))
LOCATION_ID = 86311
API_URL = f"https://api.airgradient.com/public/api/v1/world/locations/{LOCATION_ID}/measures/current"

WEATHER_LATITUDE = 3.1411106257487
WEATHER_LONGITUDE = 101.62749852676
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_POLL_SECONDS = 900  # Refresh the model cache every 15 minutes, independently of browsers.
WEATHER_PAST_HOURS = 72
WEATHER_FORECAST_HOURS = 36
AIR_QUALITY_POLL_SECONDS = 3600  # CAMS Global normally updates only twice daily.
AIR_QUALITY_DEGRADED_SECONDS = 2 * 3600
AIR_QUALITY_MAX_DECISION_AGE_SECONDS = 90 * 60
AIR_QUALITY_PAST_HOURS = 120
AIR_QUALITY_FORECAST_HOURS = 48
AIR_QUALITY_MODEL_VERSION = "cams_anchor_v2"
AIR_QUALITY_COMPATIBLE_ARCHIVES = ("cams_anchor_v1", "cams_anchor_v2")
AIR_QUALITY_FORECAST_METHOD_VERSION = "cams_interval_aligned_v4"
# The rider has explicitly selected a responsive operating mode.  These fixed
# weights are an explicitly aggressive operating choice informed by the current
# horizon-embargoed replay.  They are not prospectively validated; the local
# local models are most useful for the two-hour trail mean and especially its peak.
# Each target is independently replay-screened against persistence before it can
# alter a live point. The mean uses a strongly shrunk robust Ridge candidate;
# the peak retains the bounded matched-analogue outcome.
AGGRESSIVE_FORECAST_ENABLED = True
AGGRESSIVE_ARRIVAL_ANALOGUE_WEIGHT = 0.25
AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT = 0.25
AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT = 0.50
AGGRESSIVE_ANALOGUE_MIN_REPLAY_ORIGINS = 96
AGGRESSIVE_ANALOGUE_MIN_REPLAY_DAYS = 5
AGGRESSIVE_ANALOGUE_MIN_INDEPENDENT_ORIGINS = 24
AGGRESSIVE_ANALOGUE_INDEPENDENT_SEPARATION_MINUTES = 210
AGGRESSIVE_ANALOGUE_MIN_SKILL_GAIN = 0.05
AGGRESSIVE_ANALOGUE_MIN_WIN_FRACTION = 0.55
# A fixed shrinkage is deliberately predeclared so later issued-forecast tests
# measure one unchanged candidate instead of repeatedly fitting the short record.
AIR_QUALITY_DELTA_SHRINK = 0.50
AIR_QUALITY_MIN_SCORED_WINDOWS = 60
AIR_QUALITY_MIN_ISSUED_DAYS = 30
AIR_QUALITY_MIN_TASK_WINDOWS = 20
AIR_QUALITY_MIN_TASK_DAYS = 20
AIR_QUALITY_MIN_PAIR_DAYS = 30
AIR_QUALITY_MIN_PAIR_CALLS = 20
AIR_QUALITY_MIN_PAIR_WILSON_LOWER = 0.50
AIR_QUALITY_MIN_ORIGIN_READINGS = 3
AIR_QUALITY_MIN_WINDOW_BUCKETS = 8  # A full two-hour target requires all eight buckets.
AIR_QUALITY_MIN_SKILL_GAIN = 0.10
RIDE_WINDOW_MATCHES = 24
RIDE_WINDOW_MIN_MATCHES = 12
RIDE_WINDOW_MATCH_SEPARATION_MINUTES = 240
ISSUED_WEATHER_MIN_ORIGINS = 100
ISSUED_WEATHER_MIN_OFFSET_ORIGINS = 20
SUBANG_WIGOS_ID = "0-20000-0-48647"
SUBANG_WIS_URL = (
    "http://wis2node.met.gov.my/oapi/collections/"
    "urn:wmo:md:my-metmalaysia:synop-hourly/items"
)
HAZE_SOURCE_BEARING = 184.0  # Approximate TTDI-to-Pekanbaru bearing; used only in shadow analysis.
TRANSPORT_SHADOW_MODEL_VERSION = "regional_transport_shadow_v1"
TRANSPORT_SHADOW_LAGS_HOURS = (12, 18, 24)
TRANSPORT_SHADOW_ARCHIVE_DAYS = 90
TRANSPORT_SHADOW_MIN_ISSUED_DAYS = 30
TRANSPORT_SHADOW_MIN_COUNTERFACTUAL_ORIGINS = 20
TRANSPORT_SHADOW_MIN_PROSPECTIVE_ORIGINS = 30
TRANSPORT_SHADOW_MIN_STABLE_LAGS = 2
TRANSPORT_SHADOW_MIN_SKILL_GAIN = 0.05
TRANSPORT_SHADOW_MIN_TRAIN_ORIGINS = 16
TRANSPORT_SHADOW_RIDGE_ALPHA = 10.0
DRY_CLEARING_WIND_SPEED = 10.0
DRY_CLEARING_WIND_JUMP = 8.0
DRY_CLEARING_MAX_RAIN_PROBABILITY = 20.0
DRY_CLEARING_MAX_PRECIPITATION = 0.1

POLL_SECONDS = 180       # collect every 3 minutes
RETENTION_DAYS = 400     # Preserve more than one year for future seasonal validation.
RIDE_API_SCHEMA_VERSION = "1.24.0"
DASHBOARD_BUILD = "2026-09-21-reviewed-provenance-v18.3-public.1"
RIDE_API_ANALYSIS_DAYS = 28.0
SENSOR_DEGRADED_SECONDS = POLL_SECONDS * 2 + 60
SENSOR_UNAVAILABLE_SECONDS = 15 * 60
WEATHER_DEGRADED_SECONDS = 30 * 60
KL_TZ = timezone(timedelta(hours=8))
TREND_LOOKBACK_HOURS = 2
MIN_WINDOW_BUCKETS = 12  # 75% of the sixteen 15-minute buckets in four hours
HISTORY_HALF_LIFE_DAYS = 30  # Secondary descriptive context, never a forecast gate.
AIR_BUCKET_MINUTES = 15
ARRIVAL_MINUTES = 90
ARRIVAL_BUCKETS = ARRIVAL_MINUTES // AIR_BUCKET_MINUTES
ARRIVAL_NEIGHBORS = 24
INTERVAL_MATCH_SEPARATION_MINUTES = 60
TRAIL_MINUTES = 120
# With right-labelled 15-minute medians, the +105 label covers (+90,+105]:
# the first on-trail interval after arrival. Shifts 7..14 cover (+90,+210].
TRAIL_FIRST_BUCKET = ARRIVAL_BUCKETS + 1
TRAIL_LAST_BUCKET = (ARRIVAL_MINUTES + TRAIL_MINUTES) // AIR_BUCKET_MINUTES
FAST_RISE_PM25_CHANGE = 15.0
FAST_RISE_PM10_FRACTION = 0.15
MIX_FAST_RISE_PM25_CHANGE = 10.0
MIX_FAST_RISE_PM10_FRACTION = 0.08
MIX_FINE_SHARE_CHANGE = 3.0
MIX_COARSE_CHANGE = 1.5
MIX_COARSE_ELEVATED = 10.0
FAST_RISE_HISTORY_MIN = 9
WASHOUT_HISTORY_MIN = 9
EVENT_HISTORY_MIN_DAYS = 5
SHADOW_ANALOG_MIN_ORIGINS = 48
SHADOW_ANALOG_MATCHES = 16
SHADOW_ANALOG_MIN_MATCHES = 12
SHADOW_ANALOG_SEPARATION_MINUTES = 180
SHADOW_TRAINING_EMBARGO_MINUTES = ARRIVAL_MINUTES + TRAIL_MINUTES + 120
SHADOW_RIDGE_ALPHA = 100.0
LOCAL_RIDGE_MODEL_VERSION = "ridge_particle_met_decision_clock_v2"
# Seven days of fixed-phase hourly origins retain the current full eligible
# record while bounding request-time replay cost as history grows.
SHADOW_BACKTEST_MAX_ORIGINS = 7 * 24
LOWER_PARTICLE_MARKER = 35.0  # Dashboard comparison marker, not a safety threshold.
WASHOUT_PM25_DROP = 20.0
WASHOUT_PM25_FRACTION = 0.25
WASHOUT_PM10_FRACTION = 0.20
DRY_FORMING_BUCKET_MINUTES = 3
DRY_FORMING_LOOKBACK_BUCKETS = 5  # Fifteen-minute comparison at the sensor cadence.
DRY_FORMING_PM25_DROP = 10.0
DRY_FORMING_PM25_FRACTION = 0.12
DRY_FORMING_PM10_FRACTION = 0.10
DRY_FORMING_COUNT_FRACTION = 0.15
DRY_FORMING_RH_DROP = 1.0
# Internal relationship rule: moist daytime cooling is a corroborating sign of
# a shared environmental transition (for example a rain-cooled boundary), not
# a claim that lower temperature itself removes particles. It can classify a
# joint clearing already underway, but it never activates a numeric forecast.
MOIST_COOLING_TEMP_DROP_30 = 1.5
MOIST_COOLING_RH_RISE_30 = 2.0
MOIST_COOLING_PM25_FRACTION_30 = 0.20
MOIST_COOLING_PM10_FRACTION_30 = 0.20
MOIST_COOLING_COUNT_FRACTION_30 = 0.20
DRY_FORMING_HOLD_MINUTES = 36  # Thirty minutes plus two sensor-cadence buckets.
DRY_SHOCK_LOOKBACK_BUCKETS = 3  # Nine minutes at the sensor cadence.
DRY_SHOCK_PM25_DROP = 10.0
DRY_SHOCK_PM25_FRACTION = 0.12
DRY_SHOCK_PM10_FRACTION = 0.10
DRY_SHOCK_COUNT_FRACTION = 0.12
DRY_SHOCK_RH_DROP = 1.0
DRY_FORMING_TVOC_FRACTION = 0.30
DRY_FORMING_CO2_FRACTION = 0.01
RAPID_EVENT_MIN_COMPLETED = 2
RAPID_EVENT_MIN_DISTINCT_DAYS = 2
RAPID_EVENT_OVERLAY_MINUTES = 30
RAPID_EVENT_CONFIRM_MINUTES = 2.5
RAPID_EVENT_CONFIRM_SAMPLES = 2
RAPID_EVENT_FULL_WEIGHT_END_MINUTES = 8.5
RAPID_EVENT_PARTIAL_WEIGHT_END_MINUTES = 15.5
RAPID_EVENT_PARTIAL_WEIGHT = 0.25
DRY_FORMING_REARM_MINUTES = 60
# A second branch recognizes a slower, joint decline without pretending it was
# visible at the beginning of the event.  The thresholds were sensitivity-
# checked over the stored history; they are a provisional nowcast, not a
# 90-minute PM forecast.
DRY_SUSTAINED_LOOKBACK_BUCKETS = 20  # Sixty minutes at the 3-minute cadence.
DRY_SUSTAINED_PM25_DROP = 10.0
DRY_SUSTAINED_PM25_FRACTION = 0.08
DRY_SUSTAINED_PM10_FRACTION = 0.06
DRY_SUSTAINED_COUNT_FRACTION = 0.06
DRY_SUSTAINED_RH_DROP = 1.0
DRY_SUSTAINED_FALLS_45 = 9
DRY_SUSTAINED_FALLS_15 = 3
DRY_SUSTAINED_MAX_FROM_RECENT_LOW = 3.0
# This weather-assisted rule is deliberately isolated in shadow output.  It
# matched only two independent high-haze morning events and cannot yet drive a
# ride recommendation or the production exposure estimate.
DRY_DISPERSION_MIN_PM25 = 85.0
DRY_DISPERSION_START_HOUR = 5
DRY_DISPERSION_END_HOUR = 10
DRY_DISPERSION_PM25_DROP30 = 2.0
DRY_DISPERSION_PM10_FRACTION30 = 0.015
DRY_DISPERSION_COUNT_FRACTION30 = 0.03
DRY_DISPERSION_MAX_RAIN_PROBABILITY = 20.0
DRY_DISPERSION_MIN_BOUNDARY_LAYER_M = 600.0
DRY_DISPERSION_MIN_WIND_180M_KMH = 15.0
RIDE_WINDOWS = [
    {"key": "morning", "label": "Morning", "start": 9, "end": 13,
     "decision_start": 7, "decision_end": 8},
    {"key": "afternoon", "label": "Afternoon", "start": 14, "end": 18,
     "decision_start": 12, "decision_end": 13},
]

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "bukit_kiara_air_history.db"
COACH_API_DOC_PATH = BASE_DIR.parent / "docs" / "COACH_API.md"

FIELDS = [
    "pm01", "pm02", "pm10", "pm003Count", "atmp", "rhum", "rco2",
    "tvoc", "tvocIndex", "noxIndex", "heatindex"
]

status_lock = threading.Lock()
collector_status = {
    "last_attempt": None,
    "last_success": None,
    "error": None,
}

weather_lock = threading.Lock()
weather_status = {
    "last_attempt": None,
    "last_success": None,
    "error": None,
    "air_quality_error": None,
    "subang_error": None,
    "forecast": None,
    "air_quality": None,
    "subang": None,
}

shadow_lock = threading.Lock()
shadow_cache = {"cacheKey": None, "value": None}
ride_forecast_lock = threading.Lock()
ride_forecast_cache = {"cacheKey": None, "value": None}
# Per-key completion events keep concurrent browser/API requests from running
# the same relatively expensive forecast more than once.  Waiters sleep without
# holding ride_forecast_lock, so unrelated forecast keys can still progress.
ride_forecast_flights = {}
air_quality_evidence_lock = threading.Lock()
air_quality_evidence_cache = {"cacheKey": None, "value": None}
transport_evidence_lock = threading.Lock()
transport_evidence_cache = {"cacheKey": None, "value": None}
local_replay_lock = threading.Lock()
local_replay_cache = OrderedDict()
analysis_delivery = AnalysisDelivery(refresh_seconds=60, maximum_age_seconds=600)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bukit Kiara Air &amp; Weather Outlook</title>
<style>
:root{
  color-scheme:dark;
  --bg:#0f1113;--panel:#1a1d20;--panel2:#22262a;--border:#343a40;
  --text:#f4f6f8;--muted:#a7afb7;--green:#7ee787;--red:#ff7b72;
  --blue:#79c0ff;--orange:#ffa657;--yellow:#f2cc60;--cyan:#56d4dd;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif
}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--text)}
main{max-width:1180px;margin:auto;padding:26px 18px 60px}
h1{margin:0 0 5px;font-size:clamp(1.8rem,4vw,2.55rem);letter-spacing:-.04em}
h2{font-size:1.08rem;margin:0}
.sub{color:var(--muted);margin-bottom:18px}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:18px}
.pill{display:inline-flex;gap:7px;align-items:center;padding:7px 10px;border:1px solid var(--border);
  border-radius:999px;background:var(--panel);color:var(--muted);font-size:.85rem}
.dot{width:8px;height:8px;border-radius:50%;background:var(--yellow)}
.dot.ok{background:var(--green)} .dot.bad{background:var(--red)}
.panel,.now-card,.window-card{background:var(--panel);border:1px solid var(--border);border-radius:16px}
.panel{padding:17px;margin-top:13px}
.section-head{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}
.buttons{display:flex;gap:7px;flex-wrap:wrap}
button,.button{font:inherit;font-size:.83rem;color:var(--text);background:var(--panel2);
  border:1px solid var(--border);border-radius:9px;padding:8px 11px;cursor:pointer;text-decoration:none}
button.active{background:#333940;border-color:#65707a}
button:hover,.button:hover{background:#2b3035}
.now-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}
.now-card{padding:16px;min-height:138px}
.label,.k{font-size:.8rem;color:var(--muted)}
.value{font-size:clamp(1.65rem,3vw,2.15rem);font-weight:780;letter-spacing:-.04em;line-height:1;margin:10px 0 8px}
.value.range-value{font-size:clamp(1.35rem,2.25vw,1.85rem);white-space:nowrap}
.unit{font-size:.82rem;color:var(--muted);font-weight:650;margin-left:4px}
.signal{font-weight:730;line-height:1.25}
.note,.signal-note{font-size:.78rem;color:var(--muted);line-height:1.42}
.signal-note{margin-top:6px}
.decision-panel{border-top:3px solid var(--cyan);padding-top:15px}
.eyebrow{font-size:.72rem;font-weight:760;letter-spacing:.09em;text-transform:uppercase;color:var(--cyan);margin-bottom:6px}
.verdict{font-size:clamp(1.55rem,3.2vw,2.25rem);font-weight:800;letter-spacing:-.035em;line-height:1.05;margin:0 0 7px}
.verdict-reason{max-width:850px;color:var(--text);font-size:.92rem;line-height:1.45;margin:0}
.verdict-meta{color:var(--muted);font-size:.76rem;margin-top:8px}
.decision-grid{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-top:15px}
.window-card{background:var(--panel2);padding:15px}
.window-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;margin-bottom:11px}
.window-title{font-size:1.03rem;font-weight:760}.window-target{font-size:.76rem;color:var(--muted);margin-top:3px}
.badge{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:5px 8px;font-size:.72rem;color:var(--muted);white-space:nowrap}
.metric{border-top:1px solid #33393e;padding:9px 0 7px}
.metric:first-of-type{border-top:0}
.metric-row{display:grid;grid-template-columns:62px 1fr;gap:9px;align-items:start}
.metric-label{font-size:.72rem;font-weight:760;letter-spacing:.05em;color:var(--muted)}
.metric-main{font-weight:710;line-height:1.3}
.pm-window-value{font-size:30px;letter-spacing:-.6px}
.window-weather .metric-main{font-size:15px;font-weight:500;color:var(--muted)}
.metric-detail{grid-column:2;color:var(--muted);font-size:.73rem;line-height:1.35;margin-top:2px}
.risk-details{margin:7px 0 1px 71px;color:var(--muted);font-size:.72rem}
.risk-details summary{cursor:pointer;color:var(--muted)}
.risk-copy{line-height:1.42;margin-top:5px;padding-right:4px}
.window-foot{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;border-top:1px solid #33393e;padding-top:10px;margin-top:3px;font-size:.75rem}
.confidence{font-weight:740}.recheck{color:var(--muted)}
.comparison-progress{font-size:.76rem;color:var(--muted);border:1px solid var(--border);border-radius:999px;padding:6px 9px}
.method-details{margin-top:12px;color:var(--muted);font-size:.76rem}
.method-details summary{cursor:pointer;color:var(--muted)}
.method-details .note{margin-top:7px;max-width:850px}
.recent-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:13px}
.mini{background:var(--panel2);border:1px solid var(--border);border-radius:12px;padding:12px}
.v{font-weight:720;margin-top:5px}
.big{font-size:1.55rem;font-weight:780;margin:6px 0 3px}
.direction{color:var(--cyan)}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-top:13px}
.chartbox{position:relative;background:var(--panel2);border:1px solid var(--border);border-radius:13px;padding:14px;min-width:0}
.charthead{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:4px}
.charthead span{font-size:.78rem;color:var(--muted)}
canvas{display:block;width:100%;height:250px;touch-action:pan-y}
.chart-tooltip{position:absolute;z-index:4;pointer-events:none;min-width:150px;padding:8px 10px;
  border:1px solid #56616b;border-radius:8px;background:#111518ee;color:var(--text);
  box-shadow:0 8px 22px #0008;font-size:.75rem;line-height:1.45;white-space:nowrap}
.chart-tooltip strong{display:block;margin-bottom:3px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.78rem;margin-top:6px}
.sw{display:inline-block;width:10px;height:3px;border-radius:4px;margin-right:5px;vertical-align:middle}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.stat{background:var(--panel2);border-radius:10px;padding:10px;text-align:center}
.slabel{color:var(--muted);font-size:.73rem}.svalue{font-weight:720;margin-top:4px}
details.panel{padding:0;overflow:hidden}
details.panel summary{cursor:pointer;padding:16px 17px;font-weight:760;list-style:none}
details.panel summary::-webkit-details-marker{display:none}
details.panel summary::after{content:"＋";float:right;color:var(--muted)}
details.panel[open] summary::after{content:"−"}
.detail-content{padding:0 17px 17px;border-top:1px solid var(--border)}
.secondary{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:13px}
.error{color:var(--red)}
@media(max-width:850px){
  .now-grid{grid-template-columns:1fr 1fr}
  .decision-grid,.charts{grid-template-columns:1fr}
}
@media(max-width:600px){
  main{padding:20px 12px 48px}
  .now-grid{grid-template-columns:1fr 1fr;gap:9px}
  .now-card{min-height:122px;padding:14px}
  .recent-grid{grid-template-columns:1fr}
  .secondary{grid-template-columns:1fr 1fr}
  .window-head{align-items:flex-start}
  .risk-details{margin-left:0}
  .stats{grid-template-columns:1fr 1fr}.stat:last-child{grid-column:span 2}
  canvas{height:215px}
}
@media(max-width:420px){
  .now-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<main>
  <h1>Bukit Kiara Air &amp; Weather Outlook</h1>
  <div class="sub">TTDI AirGradient PM2.5 · local weather model · refreshes every 3 minutes</div>

  <div class="row">
    <div class="pill"><span id="dot" class="dot"></span><span id="status">Starting…</span></div>
    <div class="pill">Next dashboard refresh <strong id="countdown">3:00</strong></div>
    <div class="pill">Stored history <strong id="historyStatus">loading…</strong></div>
    <div class="pill" role="status"><span id="forecastStatus">Preparing forecast…</span><span id="forecastAge"></span></div>
  </div>

  <section class="now-grid" aria-label="Current air quality and weather">
    <div class="now-card">
      <div class="label">Current PM2.5</div>
      <div class="value"><span id="pm02">—</span><span class="unit">µg/m³</span></div>
      <div class="signal" id="particleLoad">Raw sensor concentration</div>
      <div class="signal-note" id="currentTrend">Recent movement collecting…</div>
      <div class="signal-note" id="particleMixSignal" hidden></div>
    </div>
    <div class="now-card" id="arrivalCard">
      <div class="label">PM2.5 forecast · 90 min after issue</div>
      <div class="value"><span id="arrivalValue">—</span><span class="unit">µg/m³</span></div>
      <div class="signal" id="nowcastSignal">Preparing forecast…</div>
      <div class="signal-note" id="arrivalBand">—</div>
    </div>
    <div class="now-card" id="trailForecastCard">
      <div class="label">PM2.5 2-hour mean range · issue +90 to +210 min</div>
      <div class="value range-value"><span id="trailMean">—</span><span class="unit">µg/m³</span></div>
      <div class="signal" id="trailSignal">Preparing forecast…</div>
      <div class="signal-note" id="trailBand">—</div>
    </div>
    <div class="now-card" id="showerCard">
      <div class="label">Current heat and weather</div>
      <div class="value"><span id="heatindex">—</span><span class="unit">°C</span></div>
      <div class="signal" id="heatLoad">Apparent temperature</div>
      <div class="signal-note" id="heatContext">Temperature — · humidity —</div>
      <div class="signal-note" id="trailWeatherSignal">Forecast collecting…</div>
      <div class="signal-note" id="showerSignal" hidden></div>
    </div>
  </section>

  <section class="panel">
    <div class="section-head">
      <div><h2>Sensor history</h2><div class="note">Hover or tap the graphs for exact readings.</div><div class="note" id="historyLoadStatus"></div><div class="note" id="collectionCoverage"></div></div>
      <div class="buttons"><span id="ranges"><button data-h="6" class="active">6 h</button><button data-h="24">24 h</button><button data-h="168">7 d</button></span><button onclick="forceRefresh()">Fetch now</button></div>
    </div>
    <div class="charts">
      <div class="chartbox">
        <div class="charthead"><strong>Particles</strong><span id="particleSummary">—</span></div>
        <canvas id="particleChart"></canvas><div class="chart-tooltip" id="particleTooltip" hidden></div>
        <div class="legend"><span><i class="sw" style="background:#79c0ff"></i>PM2.5</span><span><i class="sw" style="background:#ffa657"></i>PM10</span></div>
      </div>
      <div class="chartbox">
        <div class="charthead"><strong>Temperature and heat index</strong><span id="heatSummary">—</span></div>
        <canvas id="heatChart"></canvas><div class="chart-tooltip" id="heatTooltip" hidden></div>
        <div class="legend"><span><i class="sw" style="background:#7ee787"></i>Temperature</span><span><i class="sw" style="background:#ff7b72"></i>Heat index</span></div>
      </div>
    </div>
    <div class="stats">
      <div class="stat"><div class="slabel">Selected-period PM2.5 average</div><div id="pmAvg" class="svalue">—</div></div>
      <div class="stat"><div class="slabel">Lowest</div><div id="pmMin" class="svalue">—</div></div>
      <div class="stat"><div class="slabel">Highest</div><div id="pmMax" class="svalue">—</div></div>
    </div>
  </section>

  <section class="panel decision-panel" id="comparisonPanel">
    <div class="section-head">
      <div><h2>Morning and afternoon · PM2.5 forecast</h2><div class="note">Estimated local concentration during each 2-hour session</div></div>
      <div class="comparison-progress" id="comparisonProgress">PM model starting…</div>
    </div>
    <div class="verdict" id="comparisonVerdict" style="margin-top:13px">Preparing forecast…</div>
    <p class="verdict-reason" id="comparisonReason">Waiting for both issued ride-window forecasts.</p>

    <div class="decision-grid" id="comparisonGrid" hidden>
      <article class="window-card" id="morningCard">
        <div class="window-head">
          <div><div class="window-title" id="morningTitle">Morning · 09:00–13:00</div><div class="window-target" id="morningTarget">—</div></div>
          <div class="badge" id="morningBadge">PM forecast collecting</div>
        </div>
        <div class="metric"><div class="metric-row"><div class="metric-label">PM2.5</div><div class="metric-main pm-window-value" id="morningAir">—</div><div class="metric-detail" id="morningAirDetail">—</div></div><details class="risk-details" id="morningRiskDetails"><summary>Prediction error &amp; model</summary><div class="risk-copy" id="morningTrial" hidden></div><div class="risk-copy" id="morningRisk">—</div></details></div>
        <div class="metric window-weather"><div class="metric-row"><div class="metric-label">WEATHER</div><div class="metric-main" id="morningHeat">—</div><div class="metric-detail" id="morningWeatherDetail">—</div></div></div>
        <div class="window-foot"><span class="confidence" id="morningConfidence">Low confidence</span><span class="recheck" id="morningRecheck">Forecasted at —</span></div>
      </article>

      <article class="window-card" id="afternoonCard">
        <div class="window-head">
          <div><div class="window-title" id="afternoonTitle">Afternoon · 14:00–18:00</div><div class="window-target" id="afternoonTarget">—</div></div>
          <div class="badge" id="afternoonBadge">PM forecast collecting</div>
        </div>
        <div class="metric"><div class="metric-row"><div class="metric-label">PM2.5</div><div class="metric-main pm-window-value" id="afternoonAir">—</div><div class="metric-detail" id="afternoonAirDetail">—</div></div><details class="risk-details" id="afternoonRiskDetails"><summary>Prediction error &amp; model</summary><div class="risk-copy" id="afternoonTrial" hidden></div><div class="risk-copy" id="afternoonRisk">—</div></details></div>
        <div class="metric window-weather"><div class="metric-row"><div class="metric-label">WEATHER</div><div class="metric-main" id="afternoonHeat">—</div><div class="metric-detail" id="afternoonWeatherDetail">—</div></div></div>
        <div class="window-foot"><span class="confidence" id="afternoonConfidence">Low confidence</span><span class="recheck" id="afternoonRecheck">Forecasted at —</span></div>
      </article>
    </div>
    <details class="method-details"><summary>Forecast method and validation</summary><div class="note" id="modelReview">21 Sep: experimental direction model for the 14:00–16:00 forecast issued 07:00–08:00. Other forecasts unchanged. It caught more historical changes but also falsely predicted clearing; not prospectively validated.</div><div class="note" id="windowModelValidation">PM model checks collecting…</div><div class="note" id="regionalModelProvenance">Forecast sources collecting…</div><div class="note" id="localModelValidation">Near-term issued forecast validation collecting…</div></details>
  </section>

  <details class="panel">
    <summary>History and sensor details</summary>
    <div class="detail-content">
      <div class="note" style="margin-top:13px">AirGradient readings and issued weather runs are stored locally in <strong>bukit_kiara_air_history.db</strong> for ongoing forecast checks.</div>
      <div class="note" id="weatherProvenance" style="margin-top:8px">Weather sources collecting…</div>
      <div class="note" id="transportStatus" style="margin-top:5px">Regional wind context is collecting; it is not used in PM2.5 forecasts.</div>
      <div class="secondary">
        <div class="mini"><div class="k">PM10</div><div class="v"><span id="pm10">—</span> µg/m³</div></div>
        <div class="mini"><div class="k">Temperature</div><div class="v"><span id="atmp">—</span> °C</div></div>
        <div class="mini"><div class="k">Humidity</div><div class="v"><span id="rhum">—</span>%</div></div>
        <div class="mini"><div class="k">PM1.0</div><div class="v"><span id="pm01">—</span> µg/m³</div></div>
        <div class="mini"><div class="k">Particles ≥0.3 µm</div><div class="v"><span id="pm003Count">—</span> #/dL</div></div>
        <div class="mini"><div class="k">NOx Index</div><div id="noxIndex" class="v">—</div></div>
        <div class="mini"><div class="k">TVOC Index</div><div id="tvocIndex" class="v">—</div></div>
        <div class="mini"><div class="k">CO₂</div><div class="v"><span id="rco2">—</span> ppm</div></div>
      </div>
      <div class="row" style="margin:13px 0 0"><a class="button" href="/api/export.csv?days=30">Export 30-day CSV</a><button onclick="loadAll()">Refresh dashboard</button></div>
    </div>
  </details>
</main>

<script>
const REFRESH=180;
const DASHBOARD_BUILD="__DASHBOARD_BUILD__";
let nextRefreshAt=Date.now()+REFRESH*1000;
let refreshPromise=null;
let currentPromise=null;
let latestCurrent=null;
let historyRequest=0;
let storedHistoryLoaded=false;
let forecastPromise=null;
let nextForecastAt=Date.now();
let lastForecast=null;
let forecastDelivery={state:"initializing",updating:true};
let forecastReceivedAt=0;
let forecastAgeAtReceipt=null;
let forecastIssuedEpoch=null;
let forecastMaximumAge=600;
let forecastLoadError=null;
let selectedHours=6;
let hist=[];
let historyGaps=[];
const PARTICLE_SERIES=[
  {field:"pm02",label:"PM2.5",unit:"µg/m³",color:"#79c0ff"},
  {field:"pm10",label:"PM10",unit:"µg/m³",color:"#ffa657"}
];
const HEAT_SERIES=[
  {field:"atmp",label:"Temperature",unit:"°C",color:"#7ee787"},
  {field:"heatindex",label:"Heat index",unit:"°C",color:"#ff7b72"}
];

const $=id=>document.getElementById(id);
function f(v,d=1){
  if(v===null||v===undefined||!Number.isFinite(Number(v)))return "—";
  return Number(v).toFixed(d).replace(/\.0$/,"");
}
function countf(v){
  if(v===null||v===undefined||!Number.isFinite(Number(v)))return "—";
  return Math.round(Number(v)).toLocaleString();
}
function klClock(epoch){
  return new Date(epoch*1000).toLocaleTimeString([], {timeZone:"Asia/Kuala_Lumpur",hour:"2-digit",minute:"2-digit",hour12:false});
}
function stats(a){
  a=a.map(Number).filter(Number.isFinite);
  if(!a.length)return {avg:null,min:null,max:null};
  return {avg:a.reduce((x,y)=>x+y,0)/a.length,min:Math.min(...a),max:Math.max(...a)}
}
function hourLabel(h){
  return String(h).padStart(2,"0")+":00–"+String((h+1)%24).padStart(2,"0")+":00"
}
async function getJSON(url){
  const sep=url.includes("?")?"&":"?";
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),url.startsWith("/api/analysis")?10000:30000);
  try{
    const r=await fetch(url+sep+"ts="+Date.now(),{cache:"no-store",signal:controller.signal});
    const txt=await r.text();
    let data;
    try{data=JSON.parse(txt)}catch{throw new Error("Server returned non-JSON: "+txt.slice(0,120))}
    if(!r.ok)throw new Error(data.error||("HTTP "+r.status));
    return data
  }finally{clearTimeout(timeout)}
}
function previous(field,minutes=30){
  if(hist.length<2)return null;
  const target=hist[hist.length-1].epoch-minutes*60;
  let best=null,diff=1e99;
  for(const r of hist){
    if(r[field]==null)continue;
    const d=Math.abs(r.epoch-target);
    if(d<diff){diff=d;best=r[field]}
  }
  return diff<=900?best:null
}
function trend(now,old,unit){
  if(now==null||old==null)return "trend collecting…";
  const d=Number(now)-Number(old);
  const arrow=d>.05?"↑":d<-.05?"↓":"→";
  return arrow+" "+f(Math.abs(d))+unit+" vs ~30 min ago"
}
function showStoredHistory(hours){
  if(hours==null||!Number.isFinite(Number(hours)))return;
  const hrs=Math.max(0,Number(hours));
  $("historyStatus").textContent=hrs<48?hrs.toFixed(1)+" h":(hrs/24).toFixed(1)+" d";
}
function updateCurrentTrend(){
  if(latestCurrent)$("currentTrend").textContent=trend(latestCurrent.pm02,previous("pm02")," µg/m³");
}
function loadCurrent(){
  if(currentPromise)return currentPromise;
  currentPromise=fetchCurrent().catch(e=>{
    console.error(e);
    $("dot").className="dot bad";
    $("status").textContent="Sensor refresh failed · "+e.message;
    if(!storedHistoryLoaded)$("historyStatus").textContent="unavailable";
    throw e;
  }).finally(()=>{currentPromise=null});
  return currentPromise;
}
async function fetchCurrent(){
  const d=await getJSON("/api/current");
  if(d.dashboardBuild&&d.dashboardBuild!==DASHBOARD_BUILD){
    window.location.reload();
    return
  }
  if(d.historyHours!=null){showStoredHistory(d.historyHours);storedHistoryLoaded=true}
  const r=d.reading;
  if(!r){
    $("status").textContent=d.collector?.error||"Waiting for first reading…";
    $("dot").className="dot bad";
    return
  }
  if(latestCurrent&&Number(r.epoch)<Number(latestCurrent.epoch))return;
  latestCurrent=r;
  ["pm01","pm02","pm10","heatindex","atmp","rhum","rco2","tvocIndex","noxIndex"].forEach(id=>$(id).textContent=f(r[id]));
  $("pm003Count").textContent=countf(r.pm003Count);
  $("particleLoad").textContent="Raw sensor concentration";
  $("heatLoad").textContent="Apparent temperature";
  $("heatContext").textContent="Temperature "+f(r.atmp)+" °C · humidity "+f(r.rhum,0)+"%";
  updateCurrentTrend();
  $("particleMixSignal").hidden=true;
  $("showerSignal").hidden=true;
  $("dot").className=r.ageSeconds>720?"dot bad":"dot ok";
  $("status").textContent=r.ageSeconds>720
    ? "Sensor data is "+Math.round(r.ageSeconds/60)+" min old"
    : "Sensor online · "+new Date(r.epoch*1000).toLocaleTimeString();
}

async function loadHistory(){
  const request=++historyRequest;
  const hours=selectedHours;
  let d;
  try{d=await getJSON("/api/history?hours="+hours)}catch(e){
    if(request===historyRequest){$("historyLoadStatus").textContent="History refresh failed · "+e.message}
    throw e;
  }
  if(request!==historyRequest)return;
  $("historyLoadStatus").textContent="Showing "+(hours<48?hours+" h":hours/24+" d")+" of sensor history";
  hist=d.readings||[];
  historyGaps=d.gaps||[];
  updateCurrentTrend();
  const s=d.summary?.pm02||stats(hist.map(x=>x.pm02));
  $("pmAvg").textContent=s.avg==null?"—":f(s.avg)+" µg/m³";
  $("pmMin").textContent=s.min==null?"—":f(s.min)+" µg/m³";
  $("pmMax").textContent=s.max==null?"—":f(s.max)+" µg/m³";
  $("particleSummary").textContent=s.avg==null?"collecting…":"PM2.5 avg "+f(s.avg)+" µg/m³";
  const hs=d.summary?.heatindex||stats(hist.map(x=>x.heatindex));
  $("heatSummary").textContent=hs.avg==null?"collecting…":"heat index avg "+f(hs.avg)+" °C";
  draw($("particleChart"),PARTICLE_SERIES);
  draw($("heatChart"),HEAT_SERIES);
}

function forecastAgeSeconds(){
  return forecastAgeAtReceipt==null?null:Math.max(0,forecastAgeAtReceipt+(Date.now()-forecastReceivedAt)/1000);
}
function clearForecast(message,detail=""){
  lastForecast=null;
  ["arrivalValue","trailMean","morningAir","afternoonAir"].forEach(id=>$(id).textContent="—");
  ["arrivalBand","trailBand","morningTrial","afternoonTrial","morningRisk","afternoonRisk","morningAirDetail","afternoonAirDetail","morningWeatherDetail","afternoonWeatherDetail"].forEach(id=>$(id).textContent="");
  ["morningHeat","afternoonHeat"].forEach(id=>$(id).textContent="Forecast unavailable");
  ["morningTarget","afternoonTarget"].forEach(id=>$(id).textContent="—");
  ["morningRecheck","afternoonRecheck"].forEach(id=>$(id).textContent="Forecasted at —");
  $("nowcastSignal").textContent=message;
  $("trailSignal").hidden=false;
  $("trailSignal").textContent=message;
  $("trailWeatherSignal").textContent=message;
  $("showerSignal").hidden=true;
  $("comparisonGrid").hidden=true;
  $("comparisonVerdict").textContent=message;
  $("comparisonReason").textContent=detail;
  $("comparisonProgress").textContent=message;
}
function updateForecastStatus(){
  const age=forecastAgeSeconds();
  if(lastForecast&&age!=null&&age>forecastMaximumAge){
    forecastDelivery={...forecastDelivery,state:"expired"};
    clearForecast("Forecast expired", "Preparing an updated forecast.");
    nextForecastAt=Math.min(nextForecastAt,Date.now());
  }
  let label=forecastDelivery.state==="expired"?"Forecast expired":
    forecastDelivery.state==="error"?"Forecast unavailable":
    lastForecast?(forecastDelivery.updating||forecastDelivery.state==="initializing"||forecastDelivery.state==="updating"||forecastPromise?"Updating forecast…":"Forecast ready"):
    "Preparing forecast…";
  if((forecastLoadError||forecastDelivery.state==="error")&&forecastDelivery.state!=="expired")label=lastForecast?"Forecast refresh failed · showing last forecast":"Forecast unavailable · retrying…";
  $("forecastStatus").textContent=label;
  $("forecastStatus").className=forecastDelivery.state==="error"||forecastDelivery.state==="expired"||forecastLoadError?"error":"";
  $("forecastStatus").title=forecastLoadError||forecastDelivery.error||"";
  $("forecastAge").textContent=forecastIssuedEpoch&&age!=null
    ? "· issued "+klClock(forecastIssuedEpoch)+" · "+(age<60?"just now":Math.floor(age/60)+" min old")
    : "";
}
function loadAnalysis(){
  if(forecastPromise)return forecastPromise;
  nextForecastAt=Infinity;
  forecastPromise=(async()=>{
    try{
      const a=await getJSON("/api/analysis?days=28");
      forecastLoadError=null;
      forecastDelivery=a.delivery||{state:a.available===false?"initializing":"ready",updating:false};
      const maximum=Number(forecastDelivery.maximumAgeSeconds);
      forecastMaximumAge=Number.isFinite(maximum)&&maximum>0?Math.min(600,maximum):600;
      const issued=Number(a.forecastIssuedEpoch||a.airWindow?.forecastClock?.forecastIssuedEpoch);
      const age=forecastDelivery.forecastAgeSeconds;
      if(issued>0){
        forecastIssuedEpoch=issued;
        forecastAgeAtReceipt=age!=null&&Number.isFinite(Number(age))?Math.max(0,Number(age)):
          Math.max(0,Number(forecastDelivery.servedAtEpoch||Date.now()/1000)-issued);
        forecastReceivedAt=Date.now();
      }
      const expired=forecastDelivery.state==="expired"||(issued>0&&forecastAgeSeconds()>forecastMaximumAge);
      if(a.available===false||expired){
        if(expired)forecastDelivery={...forecastDelivery,state:"expired"};
        const message=expired?"Forecast expired":forecastDelivery.state==="error"?"Forecast unavailable":"Preparing forecast…";
        // A server warming up may have no result yet while this page still has
        // an unexpired issued forecast. Explicit expiry/error always removes it.
        if(expired||forecastDelivery.state==="error"||!lastForecast||forecastAgeSeconds()==null){
          clearForecast(message,a.message||"");
        }
      }else{
        lastForecast=a;
        renderAnalysis(a);
      }
      const pending=forecastDelivery.updating||forecastDelivery.state==="initializing"||forecastDelivery.state==="updating"||forecastDelivery.diagnostics==="pending";
      nextForecastAt=Date.now()+(pending?5:30)*1000;
    }catch(e){
      console.error(e);
      forecastLoadError=e.name==="AbortError"?"Forecast request timed out":e.message;
      if(!lastForecast)clearForecast("Forecast unavailable", "Retrying shortly.");
      nextForecastAt=Date.now()+30000;
    }finally{
      forecastPromise=null;
      updateForecastStatus();
    }
  })();
  updateForecastStatus();
  return forecastPromise;
}
function renderAnalysis(a){
  const currentIsFresh=!latestCurrent||Number(a.current?.epoch)>=Number(latestCurrent.epoch);
  const o=a.outlook||{};
  if(currentIsFresh&&o.available){
    const arrow=o.momentumDirection==="Rising"?"↑":o.momentumDirection==="Falling"?"↓":"→";
    const momentum=arrow+" "+f(Math.abs(o.momentumChange))+" µg/m³ over "+o.momentumMinutes+" min";
    $("currentTrend").textContent=momentum;
  }else if(currentIsFresh){
    $("currentTrend").textContent=o.message||a.message||"Recent movement collecting…";
  }

  const c=a.comparison||{};
  const model=c.model||{};
  $("comparisonVerdict").textContent=c.headline||"Outlook collecting";
  $("comparisonReason").textContent=c.summary||"Both windows are still collecting.";
  $("comparisonProgress").textContent=model.displayLabel||"Experimental PM forecast";
  const weather=a.weather||{};
  const weatherWindows=weather.windows||{};
  const hasWindowOutlook=Boolean(
    weatherWindows.morning?.available||weatherWindows.afternoon?.available||
    a.windows?.morning?.particleForecast?.available||a.windows?.afternoon?.particleForecast?.available
  );
  $("comparisonGrid").hidden=!hasWindowOutlook;

  function renderForecast(key,prefix,label){
    const w=a.windows?.[key]||{};
    const pf=w.particleForecast||{};
    const wf=w.weatherForecast||{};
    $(prefix+"Title").textContent=label+" · "+(w.modeledSessionLabel||wf.modeledSession||"—");
    const targetDate=w.startEpoch
      ? new Date(w.startEpoch*1000).toLocaleDateString([], {timeZone:"Asia/Kuala_Lumpur",weekday:"short",day:"numeric",month:"short"})
      : (w.targetLabel||"Upcoming");
    const logistics=w.decisionLabel?" · "+w.decisionLabel:"";
    $(prefix+"Target").textContent=targetDate+" · within "+(w.rideLabel||wf.window||"—")+logistics;
    const pointAvailable=Boolean(pf.available&&pf.point!=null);
    const referenceOnly=pf.pointRole==="persistence_anchor";
    const directional=Boolean(pf.directionalModel?.applied);
    $(prefix+"Badge").textContent=pointAvailable?(directional?"Experimental direction":referenceOnly?"Persistence baseline":"Adaptive local model"):"Unavailable";
    $(prefix+"Air").textContent=pointAvailable?(referenceOnly?"":"≈")+f(pf.point,0)+" µg/m³":"—";
    const anchor=pf.baselinePoint;
    const change=anchor==null||pf.point==null?null:pf.point-anchor;
    $(prefix+"AirDetail").textContent=pointAvailable
      ? (referenceOnly?"No skill-supported change estimate":("Estimated 2-hour mean"+(change==null?"":" · "+(change>=0?"+":"")+f(change,0)+(directional||pf.freshnessAdjustment?.applied?" vs recent sensor reference":" vs latest completed reading"))))
      : pf.message||"Insufficient recent data for this session";
    const risk=[];
    const trial=pf.regionalCorrectionTrial||{},correction=trial.regionalCorrection||{};
    let trialText="";
    if(trial.available&&trial.meanPm25UgM3!=null){
      const offset=correction.learnedCorrectionPm25UgM3;
      trialText="CAMS + local trial ≈"+f(trial.meanPm25UgM3,1)+" µg/m³: regional "+f(correction.regionalMeanPm25UgM3,1)+(offset<0?" − ":" + ")+f(Math.abs(offset),1)+" local correction"+(correction.nonnegativeFloorApplied?" · zero floor applied":"")+" · not applied to main forecast";
    }else if(trial.modelVersion){
      trialText="CAMS + local trial unavailable: "+String(trial.reason||"missing inputs").replaceAll("_"," ");
    }
    $(prefix+"Trial").textContent=trialText;
    $(prefix+"Trial").hidden=!trialText;
    if(pf.method){risk.push("Main forecast: "+pf.method)}
    const selection=pf.modelSelection||{};
    if(selection.selectionReasonText){risk.push(selection.selectionReasonText)}
    if(selection.handoff?.mayChangePointWithoutNewObservations){risk.push("A scheduled model change can move the estimate without a measured PM change")}
    const refresh=pf.freshnessAdjustment||{};
    if(refresh.applied){risk.push("Recent-sensor correction "+(refresh.amountUgM3>=0?"+":"")+f(refresh.amountUgM3)+" µg/m³; updates the level after a change, not advance detection")}
    const rainInput=pf.rainContext||{};
    const weatherEpoch=rainInput.fetchedEpoch;
    if(weatherEpoch){risk.push("Weather input "+new Date(weatherEpoch*1000).toLocaleTimeString([], {timeZone:"Asia/Kuala_Lumpur",hour:"2-digit",minute:"2-digit",hour12:false})+" · aligned with this PM calculation")}
    if(pf.rainLearning?.trialAvailable){risk.push("Rain timing/clearing model is being tested separately; no demonstrated accuracy gain yet.")}
    const error=pf.modelEvidence||{};
    if(error.mae!=null){risk.push("Historical replay average error "+f(error.mae,1)+" µg/m³ vs persistence "+f(error.persistenceMae,1)+" · "+Number(error.count||0)+" forecasts across "+Number(error.distinctDays||0)+" days")}
    const recent=error.recentCompleted72Hours||{};
    if(recent.mae!=null){risk.push("Latest 72 h completed targets: error "+f(recent.mae,1)+" vs persistence "+f(recent.persistenceMae,1)+" µg/m³ ("+recent.count+" overlapping forecasts)")}
    if(pf.rawRangeLow!=null&&pf.rawRangeHigh!=null){risk.push("Historical forecast-error span "+f(pf.rawRangeLow,0)+"–"+f(pf.rawRangeHigh,0)+" µg/m³ · uncalibrated, not within-session min/max")}
    if(pf.modelNote){risk.push(pf.modelNote)}
    $(prefix+"RiskDetails").hidden=!risk.length&&!trialText;
    $(prefix+"Risk").textContent=risk.join(" · ");
    if(wf.available){
      $(prefix+"Heat").textContent="Feels like "+f(wf.apparentTemperatureMax,0)+" °C · "+(wf.skyLabel||"Sky unavailable");
      const wind=wf.windSpeed10mMean==null?"wind —":"wind "+f(wf.windSpeed10mMean)+" km/h";
      const direction=wf.windDirection10mCompass?" "+wf.windDirection10mCompass:"";
      const gust=wf.windGust10mMax==null?"":" · gusts "+f(wf.windGust10mMax)+" km/h";
      const rain=wf.precipitationProbabilityMax==null?"rain —":"rain "+f(wf.precipitationProbabilityMax,0)+"%";
      const amount=wf.precipitationMm==null||Number(wf.precipitationMm)<=0
        ? ""
        : " · "+f(wf.precipitationMm)+" mm";
      $(prefix+"WeatherDetail").textContent=wind+direction+gust+" · "+rain+amount;
    }else{
      $(prefix+"Heat").textContent="Forecast unavailable";
      $(prefix+"WeatherDetail").textContent="No modeled weather values for this session";
    }
    $(prefix+"Confidence").textContent=pf.confidence||"Low confidence";
    $(prefix+"Recheck").textContent=w.forecastedLabel||"Forecasted at —";
  }
  if(hasWindowOutlook){
    renderForecast("morning","morning","Morning");
    renderForecast("afternoon","afternoon","Afternoon");
  }
  const provenance=a.rideForecast?.sourceStatus||{};
  const localTime=value=>value?new Date(value).toLocaleString([], {timeZone:"Asia/Kuala_Lumpur"}):"unavailable";
  $("regionalModelProvenance").textContent="Local sensor history and archived model forecasts are evaluated without future observations. Weather is secondary; no wind-direction or seasonal PM penalty is applied.";
  $("windowModelValidation").textContent=a.windowPrediction?.note||"Experimental local PM estimates; model checks are specific to the session lead time.";

  const shower=a.showerSignal||{};
  const showWeatherEvent=(shower.active||shower.state==="recent")
    &&(shower.rainSupport||shower.dryAirMassSupport);
  if(currentIsFresh){
    $("showerSignal").hidden=!showWeatherEvent;
    $("showerSignal").textContent=showWeatherEvent?shower.label:"";
  }
  $("showerCard").style.borderLeft="";

  const trailWeather=weather.trail||{};
  if(trailWeather.available){
    const wind=trailWeather.windSpeed10mMean==null
      ? "wind —"
      : "wind "+f(trailWeather.windSpeed10mMean)+" km/h";
    const gust=trailWeather.windGust10mMax==null
      ? "gust forecast —"
      : "gusts "+f(trailWeather.windGust10mMax)+" km/h";
    const rain=trailWeather.precipitationProbabilityMax==null
      ? "rain —"
      : "rain "+f(trailWeather.precipitationProbabilityMax,0)+"%";
    $("trailWeatherSignal").textContent=(trailWeather.skyLabel||"Sky forecast unavailable")+" · "+wind+" · "+gust+" · "+rain+(forecastIssuedEpoch?" · forecast issued "+klClock(forecastIssuedEpoch):"");
  }else{
    $("trailWeatherSignal").textContent=weather.message||"Weather forecast collecting…";
  }

  const station=weather.subang||{};
  const evidence=weather.evidence||{};
  const match=evidence.stationMatch||{};
  const modelAge=weather.ageMinutes==null?"":" · model "+weather.ageMinutes+" min old";
  const stationVisibility=station.visibility_km==null?"":" · "+f(station.visibility_km,1)+" km visibility";
  const stationText=station.report_epoch
    ? " · MET Malaysia Subang observation "+station.ageMinutes+" min old, "+f(station.atmp)+" °C, "+f(station.rhum,0)+"% RH, "+f(station.wind_speed_kmh)+" km/h "+(station.windCompass||"")+stationVisibility
    : " · Subang observation collecting";
  const checkText=evidence.originCount
    ? " · issued wind check "+evidence.originCount+"/"+evidence.minimumOrigins+(evidence.windPm3hSpearman==null?"":" · r="+f(evidence.windPm3hSpearman,2))
    : "";
  const camsEvidence=a.rideForecast?.issuedEvidence||{};
  const camsStatus=a.rideForecast?.sourceStatus||{};
  const camsText=" · CAMS ride windows "+Number(camsEvidence.scoredWindowCount||0)+"/"+Number(camsEvidence.minimumScoredWindows||60);
  const camsAge=camsStatus.ageMinutes==null?"":" · CAMS "+camsStatus.ageMinutes+" min old";
  const camsIssue=camsStatus.error?" · CAMS unavailable":"";
  $("weatherProvenance").textContent="Open-Meteo hourly weather and Copernicus Atmosphere Monitoring Service (CAMS) Global particles"+modelAge+camsAge+camsIssue+camsText+stationText+checkText+". Source ages are as of forecast issue. Subang validates conditions; it is not a forecast.";
  const transport=weather.transport||{};
  $("transportStatus").textContent=(transport.label||"Regional haze transport collecting")+". "+(transport.reason||"");

  const aw=a.airWindow||{};
  const gap=(a.dataCoverage?.recentGaps||[]).at(-1);
  $("collectionCoverage").textContent=gap
    ? "Last collection gap: "+new Date(gap.startEpoch*1000).toLocaleString([], {timeZone:"Asia/Kuala_Lumpur",day:"numeric",month:"short",hour:"2-digit",minute:"2-digit"})+"–"+new Date(gap.endEpoch*1000).toLocaleTimeString([], {timeZone:"Asia/Kuala_Lumpur",hour:"2-digit",minute:"2-digit"})+" · "+f(gap.durationMinutes,0)+" min"+(gap.cause==="laptop_sleep_user_confirmed"?" · laptop asleep":"")+" · not filled"
    : "";
  const issuedCheck=a.shadowForecast?.prospectiveEvidence||{};
  const issuedMetrics=issuedCheck.metrics||{};
  const replay=a.shadowForecast?.backtest||{};
  const skill=replay.deploymentSkill||{};
  const checks=[];
  if(replay.testOriginCount){
    checks.push("Near-term candidate replay: "+replay.testOriginCount+" overlapping origins across "+replay.distinctDays+" days.");
    checks.push("+90 min average error "+f(replay.arrivalMae)+" vs closed-15-min reference "+f(replay.arrivalPersistenceMae)+" µg/m³; "+(skill.arrival?.eligible?"candidate eligible.":"candidate not applied."));
    checks.push("+90–210 min mean average error "+f(replay.trailMeanMae)+" vs closed-15-min reference "+f(replay.trailMeanPersistenceMae)+" µg/m³; "+(skill.trailMean?.eligible?"candidate eligible.":"candidate not applied."));
  }
  checks.push(issuedCheck.scoredCount
    ? "Separate issued mean-model check: "+issuedCheck.scoredCount+" forecasts across "+issuedCheck.distinctDays+
      " days ("+issuedCheck.independentOriginCount+" separated windows). Average error "+f(issuedMetrics.mae)+
      " vs persistence "+f(issuedMetrics.persistenceMae)+" µg/m³. "+
      (issuedCheck.eligible?"Issued-outcome checks passed.":"Still under evaluation.")
    : "Awaiting completed issued forecasts for the mean-model check.");
  $("localModelValidation").textContent=a.delivery?.diagnostics==="pending"
    ? "Additional model checks are updating."
    : a.delivery?.diagnostics==="error"?"Additional model checks are currently unavailable.":checks.join(" ");
  if(!storedHistoryLoaded)showStoredHistory(a.historyHours);
  const arrival=aw.arrival||{};
  const trail=aw.trail||{};
  const arrivalEvent=arrival.pointRole==="aggressive_rapid_clearance_event";
  const trailEvent=trail.pointRole==="aggressive_rapid_clearance_event";
  const mix=aw.particleMix||{};
  const mixSignal=$("particleMixSignal");
  if(currentIsFresh){
    mixSignal.hidden=!mix.relevant;
    mixSignal.textContent=mix.relevant
      ? mix.label+(mix.details?" · "+mix.details:"")
      : "";
  }
  const arrivalRain=Boolean(arrival.rainModel?.applied);
  const arrivalAggressive=String(arrival.forecastState||"").startsWith("aggressive_")||arrivalRain;
  $("arrivalValue").textContent=arrival.available?(arrivalAggressive?"≈":"")+f(arrival.point):"—";
  const observed=aw.observedMovement||{};
  if(currentIsFresh)$("particleLoad").textContent=observed.eventLabel
    ? "Observed: "+observed.eventLabel : "Raw sensor concentration";
  $("nowcastSignal").textContent=arrival.headline||"Collecting PM2.5 forecast…";
  if(arrival.available){
    $("arrivalBand").textContent=arrivalRain
      ? "Rain-aware model · empirical q10–q90 span "+f(arrival.rawRangeLow)+"–"+f(arrival.rawRangeHigh)+" · uncalibrated"
      : arrivalAggressive
      ? (arrivalEvent
          ? "Live anchor "+f(arrival.baselinePoint)+" · "+Number(arrival.rapidEventSupport?.completedEventCount||0)+" completed clearing events · envelope "+f(arrival.rangeLow)+"–"+f(arrival.rangeHigh)+" · experimental"
          : (arrival.persistenceAnchorRole==="latest_closed_15_minute_bucket_median"?"Closed 15-min reference ":"Current reference ")+f(arrival.baselinePoint)+" · empirical q10–q90 span "+f(arrival.rawRangeLow)+"–"+f(arrival.rawRangeHigh)+" · experimental")
      : (arrival.persistenceAnchorRole==="latest_closed_15_minute_bucket_median"
          ? "Last complete 15-min median"
          : "Recent 5-min sensor median")+" · empirical q10–q90 span "+f(arrival.rawRangeLow)+"–"+f(arrival.rawRangeHigh)+" · uncalibrated";
  }else{
    $("arrivalBand").textContent=arrival.headline||"History band collecting…";
  }
  const clock=aw.forecastClock||{};
  if(clock.forecastIssuedEpoch){
    $("arrivalBand").textContent+=" · Forecasted "+klClock(clock.forecastIssuedEpoch)+" for "+klClock(clock.arrivalTargetEpoch);
  }
  const trailMeanExperimental=Boolean(
    trailEvent||trail.meanSkillEligible||
    (trail.pointApproximate&&trail.pointRole!=="persistence_anchor")
  );
  const trailPeakExperimental=Boolean(
    trailEvent||trail.peakSkillEligible||trail.peakApproximate
  );
  const trailEnvelopeLow=Number(trail.rawRangeLow);
  const trailEnvelopeHigh=Number(trail.rawRangeHigh);
  const trailEnvelopeAvailable=trail.available&&trail.rawRangeLow!=null&&trail.rawRangeHigh!=null&&Number.isFinite(trailEnvelopeLow)&&Number.isFinite(trailEnvelopeHigh);
  $("trailMean").textContent=trailEnvelopeAvailable
    ? f(trailEnvelopeLow)+"–"+f(trailEnvelopeHigh)
    : trail.available?f(trail.point):"—";
  const trailSignal=$("trailSignal");
  if(!trail.available){
    trailSignal.hidden=false;
    trailSignal.textContent=trail.headline||"Collecting trail outlook…";
  }else if(trailMeanExperimental&&trailPeakExperimental){
    trailSignal.hidden=false;
    trailSignal.textContent="Experimental mean ≈"+f(trail.point)+" · peak ≈"+f(trail.projectedPeak)+" µg/m³";
  }else if(trailMeanExperimental){
    trailSignal.hidden=false;
    trailSignal.textContent="Experimental mean ≈"+f(trail.point)+" µg/m³";
  }else{
    trailSignal.hidden=true;
    trailSignal.textContent="";
  }
  if(trail.available){
    const experimentalPeakText=trailPeakExperimental
      ? " · experimental peak ≈"+f(trail.projectedPeak)+" µg/m³"
      : "";
    $("trailBand").textContent=(trail.rainModel?.applied?"Rain-aware model · ":"")+"Empirical q10–q90 for the window mean · uncalibrated"+
      experimentalPeakText+(trail.rawPeakUpper90==null?"":" · window-peak q90 "+f(trail.rawPeakUpper90)+" µg/m³");
  }else{
    $("trailBand").textContent=trail.confidence||"History collecting";
  }
  if(clock.forecastIssuedEpoch){
    $("trailBand").textContent+=" · "+klClock(clock.windowStartEpoch)+"–"+klClock(clock.windowEndEpoch);
  }
  $("arrivalCard").style.borderLeft="";
  $("trailForecastCard").style.borderLeft="";
}

function draw(canvas,series){
  const rect=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1);
  canvas.width=Math.max(300,Math.round(rect.width*dpr));
  canvas.height=Math.round(rect.height*dpr);
  const c=canvas.getContext("2d");
  c.setTransform(dpr,0,0,dpr,0,0);
  const w=rect.width,h=rect.height,p={l:43,r:10,t:15,b:28},pw=w-p.l-p.r,ph=h-p.t-p.b;
  const vals=[];
  hist.forEach(r=>series.forEach(s=>{const v=r[s.field];if(v!=null&&Number.isFinite(Number(v)))vals.push(Number(v))}));
  c.clearRect(0,0,w,h);
  if(hist.length<2||!vals.length){
    canvas._chartMeta=null;
    c.fillStyle="#a7afb7";c.font="13px system-ui";c.fillText("Collecting history…",p.l,h/2);return
  }
  let lo=Math.min(...vals),hi=Math.max(...vals);
  if(hi<=lo)hi=lo+1;
  const m=Math.max((hi-lo)*.12,1);lo=Math.max(0,lo-m);hi+=m;
  const t0=hist[0].epoch,t1=hist[hist.length-1].epoch;
  const x=t=>p.l+((t-t0)/(t1-t0||1))*pw,y=v=>p.t+(1-(v-lo)/(hi-lo))*ph;
  c.strokeStyle="#30363b";c.fillStyle="#8f979f";c.lineWidth=1;c.font="11px system-ui";
  for(let i=0;i<=4;i++){
    const yy=p.t+ph*i/4;c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke();
    c.fillText(f(hi-(hi-lo)*i/4),3,yy+4)
  }
  series.forEach(s=>{
    c.strokeStyle=s.color;c.lineWidth=2;c.beginPath();let started=false,segment=null;
    hist.forEach(r=>{
      const v=r[s.field];
      if(v==null||!Number.isFinite(Number(v))){started=false;return}
      const xx=x(r.epoch),yy=y(Number(v));
      if(!started||r.segment!==segment){c.moveTo(xx,yy);started=true}else c.lineTo(xx,yy);
      segment=r.segment
    });
    c.stroke()
  });
  c.fillStyle="#8f979f";
  const labels=[t0,(t0+t1)/2,t1];
  ["left","center","right"].forEach((align,i)=>{
    c.textAlign=align;const d=new Date(labels[i]*1000);
    const txt=selectedHours>48?d.toLocaleDateString([],{month:"short",day:"numeric"}):d.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});
    c.fillText(txt,x(labels[i]),h-6)
  });
  c.textAlign="left";
  canvas._chartMeta={t0,t1,p,pw,w,h,series}
}

function setupChartHover(canvas,tooltip){
  const show=event=>{
    const m=canvas._chartMeta;
    if(!m||!hist.length||event.offsetX<m.p.l||event.offsetX>m.w-m.p.r){tooltip.hidden=true;return}
    const target=m.t0+((event.offsetX-m.p.l)/m.pw)*(m.t1-m.t0);
    if(historyGaps.some(g=>target>g.startEpoch&&target<g.endEpoch)){tooltip.hidden=true;return}
    let reading=hist[0],distance=Math.abs(hist[0].epoch-target);
    for(const r of hist){
      const d=Math.abs(r.epoch-target);
      if(d<distance){reading=r;distance=d}
    }
    const when=new Date(reading.epoch*1000).toLocaleString([],{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"});
    const values=m.series.map(s=>s.label+": "+f(reading[s.field])+" "+s.unit).join("<br>");
    tooltip.innerHTML="<strong>"+when+"</strong>"+values;
    tooltip.hidden=false;
    const box=canvas.parentElement;
    const left=Math.min(canvas.offsetLeft+event.offsetX+12,box.clientWidth-tooltip.offsetWidth-8);
    const top=Math.max(38,canvas.offsetTop+event.offsetY-tooltip.offsetHeight-10);
    tooltip.style.left=Math.max(8,left)+"px";
    tooltip.style.top=top+"px"
  };
  canvas.addEventListener("pointermove",show);
  canvas.addEventListener("pointerdown",show);
  canvas.addEventListener("mouseleave",()=>{tooltip.hidden=true})
}

async function forceRefresh(){
  $("status").textContent="Fetching now…";
  try{await getJSON("/api/refresh")}catch(e){$("status").textContent=e.message}
  await loadAll()
}
function loadAll(){
  if(refreshPromise)return refreshPromise;
  loadAnalysis();
  refreshPromise=Promise.allSettled([loadCurrent(),loadHistory()]).then(results=>{
    nextRefreshAt=Date.now()+(results.some(r=>r.status==="rejected")?30:REFRESH)*1000;
  }).finally(()=>{refreshPromise=null});
  return refreshPromise
}
document.querySelectorAll("#ranges button").forEach(b=>b.onclick=async()=>{
  document.querySelectorAll("#ranges button").forEach(x=>x.classList.remove("active"));
  b.classList.add("active");selectedHours=Number(b.dataset.h);
  try{await loadHistory()}catch(e){console.error(e)}
});
setInterval(()=>{
  const left=Math.max(0,Math.ceil((nextRefreshAt-Date.now())/1000));
  if(left<=0)loadAll();
  updateForecastStatus();
  if(Date.now()>=nextForecastAt)loadAnalysis();
  $("countdown").textContent=Math.floor(left/60)+":"+String(left%60).padStart(2,"0")
},1000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden)loadAll()});
window.addEventListener("focus",()=>loadAll());
window.addEventListener("resize",()=>{draw($("particleChart"),PARTICLE_SERIES);draw($("heatChart"),HEAT_SERIES)});
setupChartHover($("particleChart"),$("particleTooltip"));
setupChartHover($("heatChart"),$("heatTooltip"));
loadAll();
</script>
</body>
</html>
"""
HTML = HTML.replace("__DASHBOARD_BUILD__", DASHBOARD_BUILD)


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value):
    """Convert Pandas/NumPy scalars and non-finite values to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS readings(
            epoch INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            pm01 REAL, pm02 REAL, pm10 REAL, pm003Count REAL,
            atmp REAL, rhum REAL, rco2 REAL,
            tvoc REAL, tvocIndex REAL, noxIndex REAL, heatindex REAL
        )
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(readings)")}
        if "pm003Count" not in columns:
            conn.execute("ALTER TABLE readings ADD COLUMN pm003Count REAL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_epoch ON readings(epoch)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_forecast_runs(
            fetched_epoch INTEGER PRIMARY KEY,
            fetched_timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS air_quality_forecast_runs(
            fetched_epoch INTEGER PRIMARY KEY,
            fetched_timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            model_version TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS local_pm_forecast_issues(
            origin_epoch INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            issued_epoch INTEGER NOT NULL,
            anchor_pm25 REAL NOT NULL,
            arrival_delta REAL,
            trail_mean_delta REAL,
            trail_peak_delta REAL,
            payload TEXT NOT NULL,
            PRIMARY KEY(origin_epoch, model_version)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS subang_observations(
            report_epoch INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            atmp REAL, rhum REAL, wind_speed_kmh REAL, wind_direction REAL,
            dewpoint REAL, visibility_km REAL, present_weather TEXT,
            precipitation_mm REAL, precipitation_period_hours REAL,
            first_fetched_epoch INTEGER, last_fetched_epoch INTEGER,
            fetched_epoch INTEGER NOT NULL
        )
        """)
        subang_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(subang_observations)")
        }
        subang_migrations = {
            "dewpoint": "REAL",
            "visibility_km": "REAL",
            "present_weather": "TEXT",
            "precipitation_mm": "REAL",
            "precipitation_period_hours": "REAL",
            "first_fetched_epoch": "INTEGER",
            "last_fetched_epoch": "INTEGER",
        }
        for column, column_type in subang_migrations.items():
            if column not in subang_columns:
                conn.execute(
                    f"ALTER TABLE subang_observations ADD COLUMN {column} {column_type}"
                )
        conn.execute(
            "UPDATE subang_observations SET "
            "first_fetched_epoch=COALESCE(first_fetched_epoch,fetched_epoch), "
            "last_fetched_epoch=COALESCE(last_fetched_epoch,fetched_epoch)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_weather_fetched ON weather_forecast_runs(fetched_epoch)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_air_quality_fetched "
            "ON air_quality_forecast_runs(fetched_epoch)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_local_pm_issue "
            "ON local_pm_forecast_issues(issued_epoch)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subang_report ON subang_observations(report_epoch)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subang_first_fetched "
            "ON subang_observations(first_fetched_epoch)"
        )
        conn.commit()


def num(v):
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def parse_timestamp(value):
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return int(d.timestamp()), d.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return int(time.time()), iso_now()


def save_reading(data):
    # Observation time must come from the sensor, including after host resume.
    # An invalid timestamp must never turn an old response into a fresh sample.
    try:
        observed = datetime.fromisoformat(str(data.get("timestamp")).replace("Z", "+00:00"))
        if observed.tzinfo is None:
            raise ValueError("timezone missing")
        epoch = int(observed.timestamp())
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("AirGradient returned a missing or invalid sensor timestamp") from error
    fetched_epoch = int(time.time())
    if epoch > fetched_epoch + 60:
        raise ValueError("AirGradient sensor timestamp is more than 60 seconds in the future")
    timestamp = observed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    vals = [num(data.get(f)) for f in FIELDS]
    vals = [value if value is None or math.isfinite(value) else None for value in vals]
    placeholders = ",".join("?" for _ in range(2 + len(FIELDS)))
    cols = ",".join(["epoch", "timestamp"] + FIELDS)

    with db() as conn:
        previous_epoch = conn.execute("SELECT MAX(epoch) FROM readings").fetchone()[0]
        conn.execute(
            f"INSERT OR REPLACE INTO readings({cols}) VALUES({placeholders})",
            [epoch, timestamp] + vals,
        )
        cutoff = int(time.time()) - RETENTION_DAYS * 86400
        conn.execute("DELETE FROM readings WHERE epoch < ?", (cutoff,))
        conn.commit()
    return {
        "epoch": epoch,
        "timestamp": timestamp,
        "newObservation": previous_epoch is None or epoch > previous_epoch,
        "previousEpoch": previous_epoch,
        "gapSeconds": epoch - previous_epoch if previous_epoch is not None and epoch > previous_epoch else 0,
    }


def fetch_reading():
    with status_lock:
        collector_status["last_attempt"] = iso_now()

    try:
        req = Request(API_URL, headers={
            "User-Agent": "BukitKiara-Ride-Dashboard/1.0",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        })
        with urlopen(req, timeout=20) as r:
            raw = r.read()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("AirGradient returned unexpected JSON")
        observation = save_reading(data)

        with status_lock:
            collector_status["last_success"] = iso_now()
            collector_status["last_observation_epoch"] = observation["epoch"]
            collector_status["last_observation_timestamp"] = observation["timestamp"]
            collector_status["sensor_age_seconds"] = max(0, int(time.time()) - observation["epoch"])
            collector_status["new_observation"] = observation["newObservation"]
            if observation["gapSeconds"] > SENSOR_DEGRADED_SECONDS:
                collector_status["last_gap_start_epoch"] = observation["previousEpoch"]
                collector_status["last_gap_end_epoch"] = observation["epoch"]
                collector_status["last_gap_seconds"] = observation["gapSeconds"]
            collector_status["error"] = None

        print(
            f"[AirGradient] {data.get('timestamp')} | "
            f"PM2.5={data.get('pm02')} | temp={data.get('atmp')} | RH={data.get('rhum')}"
        )
        return data

    except Exception as e:
        with status_lock:
            collector_status["error"] = str(e)
        print("[AirGradient] ERROR:", e)
        return None


def collector():
    next_attempt = 0.0
    previous_tick = time.time()
    while True:
        now = time.time()
        pause_seconds = now - previous_tick
        if pause_seconds > SENSOR_DEGRADED_SECONDS:
            with status_lock:
                collector_status["last_runtime_pause_seconds"] = round(pause_seconds, 1)
                collector_status["last_runtime_pause_detected"] = iso_now()
            next_attempt = now
        elif now < previous_tick - 5:
            # A corrected system clock must not defer the next attempt forever.
            next_attempt = now
        previous_tick = now
        if now < next_attempt:
            time.sleep(min(5.0, next_attempt - now))
            continue
        data = fetch_reading()
        with status_lock:
            observation_age = collector_status.get("sensor_age_seconds")
        fresh = data is not None and observation_age is not None and observation_age <= SENSOR_DEGRADED_SECONDS
        interval = POLL_SECONDS if fresh else 30
        finished = time.time()
        next_attempt = max(finished + 5, now + interval)
        previous_tick = finished


def fetch_json(url, timeout=25):
    req = Request(url, headers={
        "User-Agent": "BukitKiara-Ride-Dashboard/1.1",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
    })
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def local_weather_epoch(value):
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KL_TZ)
    return int(parsed.timestamp())


def save_weather_forecast(payload):
    fetched_epoch = int(payload["fetchedEpoch"])
    # Retain a small amount of modelled pre-issue context as it was available
    # in this vintage. This is not a rain-gauge observation or a reconstruction
    # of older forecast runs. It supports future rain-persistence checks.
    archived = dict(payload)
    archived["hourly"] = [
        point for point in payload.get("hourly", [])
        if point["epoch"] >= fetched_epoch - 4 * 3600
    ]
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO weather_forecast_runs"
            "(fetched_epoch,fetched_timestamp,source,payload) VALUES(?,?,?,?)",
            (fetched_epoch, payload["fetchedTimestamp"], payload["source"],
             json.dumps(archived, separators=(",", ":"))),
        )
        cutoff = int(time.time()) - RETENTION_DAYS * 86400
        conn.execute("DELETE FROM weather_forecast_runs WHERE fetched_epoch < ?", (cutoff,))


def fetch_weather_forecast():
    with weather_lock:
        weather_status["last_attempt"] = iso_now()
    try:
        fields = [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation_probability", "precipitation", "showers",
            "cloud_cover", "pressure_msl", "wind_speed_10m",
            "wind_direction_10m", "wind_gusts_10m",
            "wind_speed_180m", "wind_direction_180m",
            "boundary_layer_height", "wind_speed_925hPa", "wind_direction_925hPa",
        ]
        params = {
            "latitude": WEATHER_LATITUDE,
            "longitude": WEATHER_LONGITUDE,
            "hourly": ",".join(fields),
            "timezone": "Asia/Kuala_Lumpur",
            "past_hours": WEATHER_PAST_HOURS,
            "forecast_hours": WEATHER_FORECAST_HOURS,
        }
        raw = fetch_json(WEATHER_API_URL + "?" + urlencode(params))
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            raise ValueError("Open-Meteo returned no hourly forecast")

        points = []
        for index, valid_time in enumerate(times):
            point = {"epoch": local_weather_epoch(valid_time), "time": valid_time}
            for field in fields:
                values = hourly.get(field, [])
                point[field] = num(values[index]) if index < len(values) else None
            points.append(point)

        fetched_epoch = int(time.time())
        payload = {
            "fetchedEpoch": fetched_epoch,
            "fetchedTimestamp": iso_now(),
            "source": "Open-Meteo Best Match",
            "latitude": num(raw.get("latitude")),
            "longitude": num(raw.get("longitude")),
            "elevation": num(raw.get("elevation")),
            "timezone": raw.get("timezone"),
            "hourlyUnits": raw.get("hourly_units", {}),
            "resolution": "hourly",
            "hourly": points,
        }
        save_weather_forecast(payload)
        with weather_lock:
            weather_status["forecast"] = payload
            weather_status["last_success"] = payload["fetchedTimestamp"]
            weather_status["error"] = None
        print(
            f'[Weather] {payload["fetchedTimestamp"]} | '
            f'{len(points)} model hours ({WEATHER_PAST_HOURS} past / {WEATHER_FORECAST_HOURS} future)'
        )
        return payload
    except Exception as error:
        with weather_lock:
            weather_status["error"] = str(error)
        print("[Weather] ERROR:", error)
        return None


def save_air_quality_forecast(payload):
    """Archive only forecast-time CAMS points for prospective skill checks."""
    fetched_epoch = int(payload["fetchedEpoch"])
    archived = dict(payload)
    archived["hourly"] = [
        point for point in payload.get("hourly", [])
        if int(point["epoch"]) >= fetched_epoch - 3600
    ]
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO air_quality_forecast_runs"
            "(fetched_epoch,fetched_timestamp,source,model_version,payload) "
            "VALUES(?,?,?,?,?)",
            (fetched_epoch, payload["fetchedTimestamp"], payload["source"],
             AIR_QUALITY_MODEL_VERSION,
             json.dumps(archived, separators=(",", ":"))),
        )
        cutoff = int(time.time()) - RETENTION_DAYS * 86400
        conn.execute(
            "DELETE FROM air_quality_forecast_runs WHERE fetched_epoch < ?",
            (cutoff,),
        )


def fetch_air_quality_forecast():
    """Fetch Open-Meteo's direct CAMS Global particle forecast."""
    try:
        fields = ["pm2_5", "pm10", "aerosol_optical_depth", "dust"]
        params = {
            "latitude": WEATHER_LATITUDE,
            "longitude": WEATHER_LONGITUDE,
            "hourly": ",".join(fields),
            "domains": "cams_global",
            "cell_selection": "nearest",
            "timezone": "Asia/Kuala_Lumpur",
            "past_hours": AIR_QUALITY_PAST_HOURS,
            "forecast_hours": AIR_QUALITY_FORECAST_HOURS,
        }
        raw = fetch_json(AIR_QUALITY_API_URL + "?" + urlencode(params))
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            raise ValueError("Open-Meteo returned no CAMS particle forecast")

        points = []
        for index, valid_time in enumerate(times):
            point = {"epoch": local_weather_epoch(valid_time), "time": valid_time}
            for field in fields:
                values = hourly.get(field, [])
                point[field] = num(values[index]) if index < len(values) else None
            points.append(point)

        fetched_epoch = int(time.time())
        payload = {
            "fetchedEpoch": fetched_epoch,
            "fetchedTimestamp": iso_now(),
            "source": "Open-Meteo / CAMS Global",
            "domain": "cams_global",
            "modelVersion": AIR_QUALITY_MODEL_VERSION,
            "attribution": (
                "Open-Meteo; Copernicus Atmosphere Monitoring Service (CAMS)"
            ),
            "attributionUrl": "https://open-meteo.com/en/docs/air-quality-api",
            "modelNativeResolutionHours": 3,
            "modelNominalUpdateHours": 12,
            "latitude": num(raw.get("latitude")),
            "longitude": num(raw.get("longitude")),
            "elevation": num(raw.get("elevation")),
            "timezone": raw.get("timezone"),
            "hourlyUnits": raw.get("hourly_units", {}),
            "resolution": "hourly interpolation of CAMS Global",
            "hourly": points,
        }
        save_air_quality_forecast(payload)
        with weather_lock:
            weather_status["air_quality"] = payload
            weather_status["air_quality_error"] = None
        current_point = min(
            points, key=lambda point: abs(int(point["epoch"]) - fetched_epoch)
        )
        print(
            f'[CAMS] {payload["fetchedTimestamp"]} | '
            f'{len(points)} model hours | PM2.5={current_point.get("pm2_5")}'
        )
        return payload
    except Exception as error:
        with weather_lock:
            weather_status["air_quality_error"] = str(error)
        print("[CAMS] ERROR:", error)
        return None


def subang_history_hours():
    with db() as conn:
        row = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN dewpoint IS NOT NULL "
            "AND visibility_km IS NOT NULL AND present_weather IS NOT NULL "
            "THEN 1 ELSE 0 END) FROM subang_observations"
        ).fetchone()
    count, diagnostic_count = int(row[0] or 0), int(row[1] or 0)
    return 72 if count < 24 or diagnostic_count < 24 else 8


def fetch_subang_observations():
    try:
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=subang_history_hours())
        params = {
            "bbox": "101.50,3.08,101.60,3.18",
            "datetime": (
                start.isoformat().replace("+00:00", "Z") + "/" +
                now.isoformat().replace("+00:00", "Z")
            ),
            "limit": 2000,
            "f": "json",
        }
        # The public WIS2 endpoint currently presents an invalid HTTPS certificate.
        # HTTP is used only for non-sensitive, read-only station observations; these
        # observations never change the PM forecast on their own.
        raw = fetch_json(SUBANG_WIS_URL + "?" + urlencode(params), timeout=25)
        names = {
            "air_temperature": "atmp",
            "dewpoint_temperature": "dewpoint",
            "relative_humidity": "rhum",
            "horizontal_visibility": "visibility_km",
            "present_weather": "present_weather",
            "wind_speed": "wind_speed_kmh",
            "wind_direction": "wind_direction",
            "total_precipitation_or_total_water_equivalent": "precipitation_mm",
        }
        reports = {}
        for feature in raw.get("features", []):
            prop = feature.get("properties", {})
            if prop.get("wigos_station_identifier") != SUBANG_WIGOS_ID:
                continue
            column = names.get(prop.get("name"))
            if not column:
                continue
            epoch, timestamp = parse_timestamp(prop.get("reportTime"))
            report = reports.setdefault(epoch, {
                "report_epoch": epoch, "timestamp": timestamp,
                "atmp": None, "rhum": None, "wind_speed_kmh": None,
                "wind_direction": None, "dewpoint": None,
                "visibility_km": None, "present_weather": None,
                "precipitation_mm": None, "precipitation_period_hours": None,
            })
            if column == "present_weather":
                report[column] = prop.get("description") or prop.get("value")
                continue

            value = num(prop.get("value"))
            if column == "wind_speed_kmh" and value is not None:
                value *= 3.6
            elif column == "visibility_km" and value is not None:
                if str(prop.get("units", "")).lower() == "m":
                    value /= 1000.0
            elif column == "precipitation_mm":
                period_hours = None
                phenomenon_time = str(prop.get("phenomenonTime") or "")
                if "/" in phenomenon_time:
                    try:
                        start_text, end_text = phenomenon_time.split("/", 1)
                        start_epoch, _ = parse_timestamp(start_text)
                        end_epoch, _ = parse_timestamp(end_text)
                        period_hours = max(0.0, (end_epoch - start_epoch) / 3600.0)
                    except Exception:
                        period_hours = None
                existing_period = report.get("precipitation_period_hours")
                if (existing_period is not None and period_hours is not None
                        and period_hours < existing_period):
                    continue
                report["precipitation_period_hours"] = period_hours
            report[column] = value

        fetched_epoch = int(time.time())
        with db() as conn:
            for report in reports.values():
                conn.execute(
                    "INSERT INTO subang_observations"
                    "(report_epoch,timestamp,atmp,rhum,wind_speed_kmh,wind_direction,"
                    "dewpoint,visibility_km,present_weather,precipitation_mm,"
                    "precipitation_period_hours,first_fetched_epoch,"
                    "last_fetched_epoch,fetched_epoch) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(report_epoch) DO UPDATE SET "
                    # Preserve the first non-null value seen for replay.  Later
                    # SYNOP revisions remain represented by last_fetched_epoch
                    # but cannot rewrite an earlier event's evidence.
                    "timestamp=subang_observations.timestamp, "
                    "atmp=COALESCE(subang_observations.atmp,excluded.atmp), "
                    "rhum=COALESCE(subang_observations.rhum,excluded.rhum), "
                    "wind_speed_kmh=COALESCE(subang_observations.wind_speed_kmh,excluded.wind_speed_kmh), "
                    "wind_direction=COALESCE(subang_observations.wind_direction,excluded.wind_direction), "
                    "dewpoint=COALESCE(subang_observations.dewpoint,excluded.dewpoint), "
                    "visibility_km=COALESCE(subang_observations.visibility_km,excluded.visibility_km), "
                    "present_weather=COALESCE(subang_observations.present_weather,excluded.present_weather), "
                    "precipitation_mm=COALESCE(subang_observations.precipitation_mm,excluded.precipitation_mm), "
                    "precipitation_period_hours=COALESCE(subang_observations.precipitation_period_hours,excluded.precipitation_period_hours), "
                    "first_fetched_epoch=MIN("
                    "COALESCE(subang_observations.first_fetched_epoch,excluded.first_fetched_epoch),"
                    "excluded.first_fetched_epoch), "
                    "last_fetched_epoch=MAX("
                    "COALESCE(subang_observations.last_fetched_epoch,excluded.last_fetched_epoch),"
                    "excluded.last_fetched_epoch), "
                    "fetched_epoch=excluded.fetched_epoch",
                    (report["report_epoch"], report["timestamp"], report["atmp"],
                     report["rhum"], report["wind_speed_kmh"],
                     report["wind_direction"], report["dewpoint"],
                     report["visibility_km"], report["present_weather"],
                     report["precipitation_mm"],
                     report["precipitation_period_hours"], fetched_epoch,
                     fetched_epoch, fetched_epoch),
                )
            cutoff = int(time.time()) - RETENTION_DAYS * 86400
            conn.execute("DELETE FROM subang_observations WHERE report_epoch < ?", (cutoff,))

        latest_report = max(reports.values(), key=lambda item: item["report_epoch"]) if reports else None
        with weather_lock:
            weather_status["subang"] = latest_report
            weather_status["subang_error"] = None
        if latest_report:
            print(
                f'[Subang] {latest_report["timestamp"]} | '
                f'temp={latest_report["atmp"]} | RH={latest_report["rhum"]} | '
                f'wind={latest_report["wind_speed_kmh"]}'
            )
        return latest_report
    except Exception as error:
        with weather_lock:
            weather_status["subang_error"] = str(error)
        print("[Subang] ERROR:", error)
        return None


def weather_collector():
    last_air_quality_success = 0.0
    next_attempt = 0.0
    previous_tick = time.time()
    while True:
        started = time.time()
        if started - previous_tick > SENSOR_DEGRADED_SECONDS or started < previous_tick - 5:
            next_attempt = started
        previous_tick = started
        if started < next_attempt:
            time.sleep(min(5.0, next_attempt - started))
            continue
        weather = fetch_weather_forecast()
        air_quality_failed = False
        if (last_air_quality_success == 0.0
                or started < last_air_quality_success
                or started - last_air_quality_success >= AIR_QUALITY_POLL_SECONDS):
            if fetch_air_quality_forecast() is not None:
                last_air_quality_success = time.time()
            else:
                air_quality_failed = True
        fetch_subang_observations()
        interval = 60 if weather is None or air_quality_failed else WEATHER_POLL_SECONDS
        finished = time.time()
        next_attempt = max(finished + 30, started + interval)
        previous_tick = finished


def latest_weather_payload(as_of_epoch=None):
    with weather_lock:
        payload = weather_status.get("forecast")
    if payload and (
            as_of_epoch is None
            or int(payload.get("fetchedEpoch") or 0) <= int(as_of_epoch)):
        return payload
    with db() as conn:
        if as_of_epoch is None:
            row = conn.execute(
                "SELECT payload FROM weather_forecast_runs "
                "ORDER BY fetched_epoch DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT payload FROM weather_forecast_runs WHERE fetched_epoch<=? "
                "ORDER BY fetched_epoch DESC LIMIT 1", (int(as_of_epoch),)
            ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError):
        return None


def latest_air_quality_payload(as_of_epoch=None):
    with weather_lock:
        payload = weather_status.get("air_quality")
    if payload and (
            as_of_epoch is None
            or int(payload.get("fetchedEpoch") or 0) <= int(as_of_epoch)):
        return payload
    with db() as conn:
        compatible_slots = ",".join("?" for _ in AIR_QUALITY_COMPATIBLE_ARCHIVES)
        if as_of_epoch is None:
            row = conn.execute(
                "SELECT payload FROM air_quality_forecast_runs "
                f"WHERE model_version IN ({compatible_slots}) "
                "ORDER BY fetched_epoch DESC LIMIT 1",
                AIR_QUALITY_COMPATIBLE_ARCHIVES,
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT payload FROM air_quality_forecast_runs "
                f"WHERE model_version IN ({compatible_slots}) AND fetched_epoch<=? "
                "ORDER BY fetched_epoch DESC LIMIT 1",
                (*AIR_QUALITY_COMPATIBLE_ARCHIVES, int(as_of_epoch)),
            ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload"])
    except (TypeError, json.JSONDecodeError):
        return None


def latest_subang_observation(as_of_epoch=None):
    with weather_lock:
        observation = weather_status.get("subang")
    if observation and as_of_epoch is None:
        return observation
    with db() as conn:
        query = (
            "SELECT report_epoch,timestamp,atmp,rhum,wind_speed_kmh,wind_direction,"
            "dewpoint,visibility_km,present_weather,precipitation_mm,"
            "precipitation_period_hours FROM subang_observations "
        )
        if as_of_epoch is None:
            row = conn.execute(query + "ORDER BY report_epoch DESC LIMIT 1").fetchone()
        else:
            row = conn.execute(
                query
                + "WHERE report_epoch<=? AND COALESCE(first_fetched_epoch,fetched_epoch)<=? "
                + "ORDER BY report_epoch DESC LIMIT 1",
                (int(as_of_epoch), int(as_of_epoch)),
            ).fetchone()
    return rowdict(row)


def rowdict(row):
    return dict(row) if row else None


def latest():
    with db() as conn:
        return rowdict(conn.execute("SELECT * FROM readings ORDER BY epoch DESC LIMIT 1").fetchone())


def history(hours):
    hours = max(1.0, min(float(hours), RETENTION_DAYS * 24.0))
    cutoff = int(time.time() - hours * 3600)
    with db() as conn:
        full_rows = conn.execute(
            "SELECT * FROM readings WHERE epoch >= ? ORDER BY epoch ASC", (cutoff,)
        ).fetchall()
    full_rows = [dict(r) for r in full_rows]

    # Segment the original observations before thinning the browser payload.
    # A long plotted interval caused only by thinning is not a sensor outage.
    gaps = []
    boundaries = set()
    segment = 0
    chart_fields = ("pm02", "pm10", "atmp", "heatindex")
    for index, row in enumerate(full_rows):
        if index:
            previous = full_rows[index - 1]
            seconds = row["epoch"] - previous["epoch"]
            if seconds > SENSOR_DEGRADED_SECONDS:
                gaps.append({
                    "startEpoch": previous["epoch"],
                    "endEpoch": row["epoch"],
                    "durationSeconds": seconds,
                })
                segment += 1
                boundaries.update((index - 1, index))
            # Retain missing-value boundaries as well as timestamp gaps.
            if any((previous.get(field) is None) != (row.get(field) is None)
                   for field in chart_fields):
                boundaries.update((index - 1, index))
        row["segment"] = segment

    summary = {
        "pm02": statv(full_rows, "pm02"),
        "pm10": statv(full_rows, "pm10"),
        "atmp": statv(full_rows, "atmp"),
        "heatindex": statv(full_rows, "heatindex"),
        "rhum": statv(full_rows, "rhum"),
    }
    coverage_hours = 0.0
    if len(full_rows) >= 2:
        coverage_hours = round((full_rows[-1]["epoch"] - full_rows[0]["epoch"]) / 3600, 2)

    # Keep browser payload reasonable when history gets large.
    rows = full_rows
    if len(rows) > 1500:
        step = math.ceil(len(rows) / 1500)
        selected = set(range(0, len(rows), step)) | boundaries | {len(rows) - 1}
        rows = [rows[index] for index in sorted(selected)]
    return {
        "sourceCount": len(full_rows),
        "count": len(rows),
        "coverageHours": coverage_hours,
        "summary": summary,
        "readings": rows,
        "gaps": gaps,
        "gapThresholdSeconds": SENSOR_DEGRADED_SECONDS,
    }


def avgv(rows, key, fallback=None):
    vals = []
    for r in rows:
        v = r[key] if key in r.keys() else None
        if v is None and fallback:
            v = r[fallback]
        if v is not None:
            vals.append(float(v))
    return round(sum(vals) / len(vals), 1) if vals else None


def statv(rows, key):
    vals = []
    for r in rows:
        v = r[key] if key in r.keys() else None
        if v is not None:
            vals.append(float(v))
    if not vals:
        return {"avg": None, "min": None, "max": None, "count": 0}
    return {
        "avg": round(sum(vals) / len(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "count": len(vals),
    }


def local_dt(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).astimezone(KL_TZ)


def trend_outlook(rows, shower_signal=None):
    pm_rows = [r for r in rows if r["pm02"] is not None]
    if len(pm_rows) < 4:
        return {"available": False, "message": "At least four recent PM2.5 readings are needed."}

    latest_epoch = int(pm_rows[-1]["epoch"])
    recent_cutoff = latest_epoch - TREND_LOOKBACK_HOURS * 3600
    recent = [r for r in pm_rows if r["epoch"] >= recent_cutoff]

    # Keep the latest ~30-minute movement as short-term context. The smoothed
    # two-hour regression below is the headline direction because it is less
    # sensitive to one short sensor movement.
    target_epoch = latest_epoch - 30 * 60
    momentum_candidates = [r for r in recent if latest_epoch - 45 * 60 <= r["epoch"] <= latest_epoch - 15 * 60]
    if not momentum_candidates:
        return {"available": False, "message": "About 30 minutes of PM2.5 readings are needed."}
    momentum_start = min(momentum_candidates, key=lambda r: abs(r["epoch"] - target_epoch))
    current = float(pm_rows[-1]["pm02"])
    momentum_minutes = max(1, round((latest_epoch - momentum_start["epoch"]) / 60))
    momentum_change = current - float(momentum_start["pm02"])
    momentum_rate = momentum_change / (momentum_minutes / 60)
    if momentum_change >= 2:
        momentum_direction = "Rising"
    elif momentum_change <= -2:
        momentum_direction = "Falling"
    else:
        momentum_direction = "Mostly steady"

    # Average each 15-minute bucket before regression so irregular polling does
    # not give a densely sampled part of the period disproportionate weight.
    buckets = {}
    for r in recent:
        bucket = (int(r["epoch"]) // 900) * 900
        buckets.setdefault(bucket, []).append(float(r["pm02"]))
    points = [(epoch, sum(vals) / len(vals)) for epoch, vals in sorted(buckets.items())]
    if len(points) < 3:
        return {"available": False, "message": "About 30–45 minutes of readings are needed for the outlook."}

    x0 = points[0][0]
    xs = [(p[0] - x0) / 3600 for p in points]
    ys = [p[1] for p in points]
    xavg = sum(xs) / len(xs)
    yavg = sum(ys) / len(ys)
    denom = sum((x - xavg) ** 2 for x in xs)
    background_slope = 0.0 if denom == 0 else sum((x - xavg) * (y - yavg) for x, y in zip(xs, ys)) / denom
    background_slope = max(-25.0, min(25.0, background_slope))
    if background_slope <= -2:
        direction = "Falling"
    elif background_slope >= 2:
        direction = "Rising"
    else:
        direction = "Mostly steady"

    recent_stats = statv(recent, "pm02")
    if current <= 15:
        level = ""
    elif current <= 35:
        level = "watch"
    elif current <= 55:
        level = "high"
    else:
        level = "very-high"

    shower_state = (shower_signal or {}).get("state", "none")
    if shower_state in ("active", "recent", "forming"):
        text = "Observed movement may include a recent particle washout or air-mass change."
    else:
        text = "Observed movement only; it is not a forecast of a later ride window."

    return {
        "available": True,
        "current": round(current, 1),
        "recentLow": recent_stats["min"],
        "recentHigh": recent_stats["max"],
        "momentumChange": round(momentum_change, 1),
        "momentumMinutes": momentum_minutes,
        "momentumRate": round(momentum_rate, 1),
        "momentumDirection": momentum_direction,
        "direction": direction,
        "slope": round(background_slope, 1),
        "sampleCount": len(recent),
        "guidance": {"level": level, "text": text},
    }


def sensor_frame(rows):
    """Return evenly spaced, local-time Pandas medians for sensor analysis."""
    columns = ["epoch", "pm02", "pm10", "pm003Count", "atmp", "rhum", "heatindex"]
    if not rows:
        return pd.DataFrame(columns=columns[1:])

    records = []
    for row in rows:
        keys = set(row.keys())
        records.append({key: row[key] if key in keys else None for key in columns})
    frame = pd.DataFrame.from_records(records)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["epoch"]).sort_values("epoch")
    if frame.empty:
        return pd.DataFrame(columns=columns[1:])

    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("epoch"), unit="s", utc=True)
    ).tz_convert("Asia/Kuala_Lumpur")
    result = frame.resample(
        f"{AIR_BUCKET_MINUTES}min", label="right", closed="right"
    ).median(numeric_only=True)
    result["_forecastEligible"] = bucket_coverage(frame)["forecastEligible"]
    return result


def fully_closed_sensor_frame(frame, issue_epoch):
    """Exclude the right-labelled bucket that has not finished at issue time.

    Live movement may use the current partial bucket, but a forecast origin or
    outcome must not.  For example, an 08:31 reading belongs to the bucket
    labelled 08:45; only buckets through 08:30 are causally complete.
    """
    if frame.empty or issue_epoch is None:
        return frame.iloc[0:0].copy()
    cutoff = pd.Timestamp(
        int(issue_epoch), unit="s", tz="UTC"
    ).tz_convert("Asia/Kuala_Lumpur").floor(f"{AIR_BUCKET_MINUTES}min")
    result = frame.loc[frame.index <= cutoff].copy()
    if "_forecastEligible" in result:
        invalid = ~result["_forecastEligible"].fillna(False).astype(bool)
        result.loc[invalid, [c for c in result if not c.startswith("_")]] = np.nan
    return result


def segmented_ewm(series, span):
    """EMA that restarts after each missing time bucket."""
    groups = series.isna().cumsum()
    return series.groupby(groups).transform(
        lambda segment: segment.ewm(span=span, adjust=False).mean()
    ).where(series.notna())


def compass_direction(degrees):
    if degrees is None or pd.isna(degrees):
        return None
    labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return labels[int((float(degrees) + 11.25) // 22.5) % 16]


def circular_mean(values):
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not clean:
        return None
    sin_mean = sum(math.sin(math.radians(value)) for value in clean) / len(clean)
    cos_mean = sum(math.cos(math.radians(value)) for value in clean) / len(clean)
    return (math.degrees(math.atan2(sin_mean, cos_mean)) + 360) % 360


def source_alignment(direction):
    if direction is None or pd.isna(direction):
        return None
    difference = ((float(direction) - HAZE_SOURCE_BEARING + 180) % 360) - 180
    return max(0.0, math.cos(math.radians(difference)))


def source_wind_component(speed, direction):
    """Signed wind-from-source component for shadow transport analysis."""
    if speed is None or direction is None or pd.isna(speed) or pd.isna(direction):
        return None
    difference = ((float(direction) - HAZE_SOURCE_BEARING + 180) % 360) - 180
    return float(speed) * math.cos(math.radians(difference))


def model_frame(payload):
    if not payload or not payload.get("hourly"):
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(payload["hourly"])
    if frame.empty or "epoch" not in frame:
        return pd.DataFrame()
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("epoch"), unit="s", utc=True)
    ).tz_convert("Asia/Kuala_Lumpur")
    return frame.sort_index()


def interpolated_model_values(model, timestamps, column="pm2_5"):
    """Linearly sample an hourly model at exact, timezone-aware instants.

    Open-Meteo's hourly CAMS values are instantaneous.  Treating the preceding
    hour as a step function shifts the model horizon by as much as 59 minutes,
    so both live forecasts and their scorer use this one interpolation seam.
    Values outside the issued trajectory are intentionally left missing.
    """
    target = pd.DatetimeIndex(timestamps)
    if model.empty or column not in model or target.empty:
        return pd.Series(index=target, dtype="float64")
    source = pd.to_numeric(model[column], errors="coerce").dropna()
    if source.empty:
        return pd.Series(index=target, dtype="float64")
    source = source[~source.index.duplicated(keep="last")].sort_index()
    combined_index = source.index.union(target).sort_values()
    interpolated = source.reindex(combined_index).interpolate(
        method="time", limit_area="inside"
    )
    return interpolated.reindex(target).astype("float64")


def model_window_values(model, start, end, column="pm2_5"):
    """Sample instantaneous model values at matching sensor-bucket centres.

    Sensor outcomes are right-labelled medians for ``(label - bucket, label]``.
    Sampling the model at each interval centre aligns the instantaneous CAMS
    trajectory to those same intervals.  Building the labels from an explicit
    period count makes the session endpoint part of both live and issued paths.
    """
    bucket = pd.Timedelta(minutes=AIR_BUCKET_MINUTES)
    duration = end - start
    if duration <= pd.Timedelta(0) or duration % bucket:
        return pd.Series(dtype="float64")
    half_bucket = bucket / 2
    period_count = int(duration / bucket)
    labels = pd.date_range(start + bucket, periods=period_count, freq=bucket)
    centres = labels - half_bucket
    values = interpolated_model_values(model, centres, column)
    values.index = labels
    return values


def aligned_model_projection(model, origin_time, start, end, column="pm2_5"):
    """Return one exact-time model origin and its aligned outcome intervals.

    All values come from one forecast payload that was already available at the
    decision origin.  Interpolation therefore uses future *valid times* from an
    issued trajectory, never a forecast retrieved after the origin.
    """
    origin_values = interpolated_model_values(
        model, pd.DatetimeIndex([origin_time]), column
    )
    window_values = model_window_values(model, start, end, column)
    if (origin_values.empty or origin_values.isna().any()
            or window_values.empty or window_values.isna().any()):
        return None, window_values
    return float(origin_values.iloc[0]), window_values.astype("float64")


def archived_issued_weather(sensor_index, as_of_epoch=None):
    """Return weather features from the latest forecast available at each origin."""
    if len(sensor_index) == 0:
        return pd.DataFrame()
    origins = pd.DatetimeIndex(sensor_index).sort_values()
    if as_of_epoch is not None:
        # A right-labelled sensor bucket can end after a historical replay
        # instant.  Remove it before choosing an issued run so a retrieval made
        # between the last raw observation and that future label cannot leak
        # into the replay.
        replay_time = pd.Timestamp(int(as_of_epoch), unit="s", tz="UTC")
        if origins.tz is None:
            replay_time = replay_time.tz_localize(None)
        else:
            replay_time = replay_time.tz_convert(origins.tz)
        origins = origins[origins <= replay_time]
    if origins.empty:
        return pd.DataFrame()
    first_epoch = int(origins.min().timestamp())
    last_epoch = int(origins.max().timestamp())
    with db() as conn:
        run_rows = conn.execute(
            "SELECT fetched_epoch,payload FROM weather_forecast_runs "
            "WHERE fetched_epoch<=? ORDER BY fetched_epoch",
            (last_epoch,),
        ).fetchall()
    runs = []
    for row in run_rows:
        try:
            payload = json.loads(row["payload"])
            points = {
                int(point["epoch"]): point for point in payload.get("hourly", [])
                if point.get("epoch") is not None
            }
            if points:
                runs.append((int(row["fetched_epoch"]), points))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not runs:
        return pd.DataFrame()

    records = []
    run_position = -1
    current_run = None
    for origin in origins:
        origin_epoch = int(origin.timestamp())
        if origin_epoch < first_epoch or origin_epoch > last_epoch:
            continue
        while (run_position + 1 < len(runs)
               and runs[run_position + 1][0] <= origin_epoch):
            run_position += 1
            current_run = runs[run_position]
        if current_run is None:
            continue
        fetched_epoch, point_map = current_run
        forecast_points = [
            point_map.get(origin_epoch + lead * 3600) for lead in (1, 2, 3)
        ]
        if any(point is None for point in forecast_points):
            continue

        def mean_field(field):
            values = [num(point.get(field)) for point in forecast_points]
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        def max_field(field):
            values = [num(point.get(field)) for point in forecast_points]
            values = [value for value in values if value is not None]
            return max(values) if values else None

        first_point = forecast_points[0]
        wind_speed_925 = mean_field("wind_speed_925hPa")
        wind_direction_925 = circular_mean([
            point.get("wind_direction_925hPa") for point in forecast_points
        ])
        records.append({
            "origin": origin,
            "issuedAgeMinutes": (origin_epoch - fetched_epoch) / 60.0,
            "windSpeed10m": mean_field("wind_speed_10m"),
            "windSpeed180m": mean_field("wind_speed_180m"),
            "windSpeed925hPa": wind_speed_925,
            "windDirection925hPa": wind_direction_925,
            "sourceAlignment925": source_alignment(wind_direction_925),
            "sourceComponent925": source_wind_component(
                wind_speed_925, wind_direction_925
            ),
            "boundaryLayerHeight": mean_field("boundary_layer_height"),
            "precipitationMax": max_field("precipitation"),
            "rainProbabilityMax": max_field("precipitation_probability"),
            "valid1Epoch": origin_epoch + 3600,
            "temperature1": num(first_point.get("temperature_2m")),
            "humidity1": num(first_point.get("relative_humidity_2m")),
            "windSpeed10m1": num(first_point.get("wind_speed_10m")),
            "windDirection10m1": num(first_point.get("wind_direction_10m")),
        })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("origin").sort_index()


def transport_spearman(frame, left, right):
    pair = frame[[left, right]].dropna()
    if len(pair) < 5 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return None
    value = pair[left].corr(pair[right], method="spearman")
    return None if pd.isna(value) else round(float(value), 2)


def transport_walk_forward(pair, lag_hours):
    """Causal shadow score for one fixed lag against PM persistence.

    Six-hour origin spacing limits repeated outcomes.  Each candidate is fitted
    only with outcomes that had fully completed before that scored origin.
    This score never alters a live PM forecast.
    """
    sampled = pair.loc[pair.index.hour % 6 == 0].copy()
    records = []
    outcome_delay = pd.Timedelta(hours=lag_hours + 2)
    features = ["sourceComponent925", "windSpeed925hPa"]
    for origin, row in sampled.iterrows():
        training = sampled.loc[sampled.index + outcome_delay <= origin].dropna(
            subset=features + ["futureDelta"]
        )
        if (len(training) < TRANSPORT_SHADOW_MIN_TRAIN_ORIGINS
                or training.index.normalize().nunique() < 5):
            continue

        train_x = training[features].astype(float)
        centre = train_x.mean(axis=0)
        scale = train_x.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
        standardized = ((train_x - centre) / scale).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(training)), standardized])
        penalty = np.diag([0.0, TRANSPORT_SHADOW_RIDGE_ALPHA,
                           TRANSPORT_SHADOW_RIDGE_ALPHA])
        try:
            coefficients = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ training["futureDelta"].to_numpy(dtype=float),
            )
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                design.T @ design + penalty,
                design.T @ training["futureDelta"].to_numpy(dtype=float),
                rcond=None,
            )[0]

        candidate_x = ((row[features].astype(float) - centre) / scale).to_numpy()
        candidate_delta = float(
            np.dot(np.r_[1.0, candidate_x], coefficients)
        )
        actual_delta = float(row["futureDelta"])
        records.append({
            "origin": origin,
            "persistenceAbsError": abs(actual_delta),
            "candidateAbsError": abs(actual_delta - candidate_delta),
        })

    if not records:
        return {
            "originSpacingHours": 6,
            "scoredOrigins": 0,
            "distinctDays": 0,
            "persistenceMae": None,
            "candidateMae": None,
            "skillGainPct": None,
            "dayBlockWins": 0,
        }

    scored = pd.DataFrame.from_records(records).set_index("origin")
    persistence_mae = float(scored["persistenceAbsError"].mean())
    candidate_mae = float(scored["candidateAbsError"].mean())
    skill = None
    if persistence_mae > 0:
        skill = 100.0 * (1.0 - candidate_mae / persistence_mae)
    day_scores = scored.groupby(scored.index.normalize()).mean(numeric_only=True)
    return {
        "originSpacingHours": 6,
        "scoredOrigins": int(len(scored)),
        "distinctDays": int(scored.index.normalize().nunique()),
        "persistenceMae": round(persistence_mae, 1),
        "candidateMae": round(candidate_mae, 1),
        "skillGainPct": None if skill is None else round(skill, 1),
        "dayBlockWins": int((
            day_scores["candidateAbsError"] < day_scores["persistenceAbsError"]
        ).sum()),
    }


def _compute_regional_transport_evidence(as_of_epoch):
    """Evaluate regional wind only from forecasts archived before each origin."""
    cutoff = int(as_of_epoch) - TRANSPORT_SHADOW_ARCHIVE_DAYS * 86400
    with db() as conn:
        rows = conn.execute(
            "SELECT epoch,pm02 FROM readings "
            "WHERE epoch BETWEEN ? AND ? ORDER BY epoch",
            (cutoff, int(as_of_epoch)),
        ).fetchall()

    base = {
        "state": "shadow_collecting",
        "usedInForecast": False,
        "modelVersion": TRANSPORT_SHADOW_MODEL_VERSION,
        "validationMode": "causal_issued_forecasts",
        "sourceBearing": HAZE_SOURCE_BEARING,
        "sourceAssumption": "Pekanbaru / central Sumatra",
        "preRegisteredLagHours": list(TRANSPORT_SHADOW_LAGS_HOURS),
        "label": "Regional wind context · shadow-only",
    }
    if not rows:
        return {
            **base,
            "reason": "Issued wind evidence is collecting. Not used in PM2.5 forecasts.",
            "current": {},
            "evidence": {
                "issuedOriginHours": 0, "issuedForecastDays": 0,
                "counterfactualOrigins": 0, "byLag": {},
                "validation": {"eligible": False, "reasons": ["No paired sensor history"]},
            },
        }

    sensor = sensor_frame(rows).resample(
        "1h", label="right", closed="right"
    ).median(numeric_only=True)
    as_of_time = pd.Timestamp(int(as_of_epoch), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    sensor = sensor.loc[sensor.index <= as_of_time]
    issued = archived_issued_weather(sensor.index, int(as_of_epoch))
    required = {
        "windSpeed925hPa", "windDirection925hPa", "sourceAlignment925",
        "sourceComponent925",
    }
    if sensor.empty or issued.empty or not required.issubset(issued.columns):
        return {
            **base,
            "reason": "Issued wind evidence is collecting. Not used in PM2.5 forecasts.",
            "current": {},
            "evidence": {
                "issuedOriginHours": 0, "issuedForecastDays": 0,
                "counterfactualOrigins": 0, "byLag": {},
                "validation": {"eligible": False, "reasons": ["No issued wind pairs"]},
            },
        }

    valid_issued = issued.dropna(subset=list(required)).copy()
    joined = sensor[["pm02"]].join(valid_issued, how="inner")
    joined = joined.dropna(subset=["pm02", "sourceComponent925"])
    issued_days = int(valid_issued.index.normalize().nunique())
    counterfactual_origins = int(
        valid_issued["sourceAlignment925"].lt(0.5).sum()
    )
    latest = valid_issued.iloc[-1] if not valid_issued.empty else None
    current = {} if latest is None else {
        "originEpoch": int(valid_issued.index[-1].timestamp()),
        "windSpeed925hPa": round(float(latest["windSpeed925hPa"]), 1),
        "windDirection925hPa": round(float(latest["windDirection925hPa"])),
        "windDirection925hPaCompass": compass_direction(
            latest["windDirection925hPa"]
        ),
        "sourceAlignment925": round(float(latest["sourceAlignment925"]), 2),
        "sourceComponent925": round(float(latest["sourceComponent925"]), 1),
    }

    by_lag = {}
    stable_lags = 0
    pm = sensor["pm02"]
    for lag_hours in TRANSPORT_SHADOW_LAGS_HOURS:
        future_parts = pd.concat(
            [pm.shift(-(lag_hours + offset)) for offset in range(3)], axis=1
        )
        future_mean = future_parts.mean(axis=1).where(
            future_parts.notna().sum(axis=1).eq(3)
        )
        pair = joined.copy()
        pair["futureMeanPm25"] = future_mean.reindex(pair.index)
        pair["futureDelta"] = pair["futureMeanPm25"] - pair["pm02"]
        pair = pair.dropna(subset=[
            "pm02", "futureMeanPm25", "futureDelta", "sourceComponent925",
            "windSpeed925hPa",
        ])

        offsets = []
        for offset in range(3):
            subset = pair.iloc[offset::3]
            offsets.append({
                "offset": offset,
                "pairs": int(len(subset)),
                "spearman": transport_spearman(
                    subset, "sourceComponent925", "futureDelta"
                ),
            })
        midpoint = len(pair) // 2
        halves = [pair.iloc[:midpoint], pair.iloc[midpoint:]] if midpoint else []
        half_correlations = [
            transport_spearman(part, "sourceComponent925", "futureDelta")
            for part in halves
        ]
        prospective = transport_walk_forward(pair, lag_hours)
        offset_values = [item["spearman"] for item in offsets]
        prospective_skill = prospective.get("skillGainPct")
        stable = bool(
            prospective.get("scoredOrigins", 0)
            >= TRANSPORT_SHADOW_MIN_PROSPECTIVE_ORIGINS
            and prospective_skill is not None
            and prospective_skill >= 100 * TRANSPORT_SHADOW_MIN_SKILL_GAIN
            and half_correlations
            and all(value is not None and value > 0 for value in half_correlations)
            and all(value is not None and value > 0 for value in offset_values)
        )
        stable_lags += int(stable)
        by_lag[str(lag_hours)] = {
            "lagHours": lag_hours,
            "completedPairs": int(len(pair)),
            "originDays": int(pair.index.normalize().nunique()),
            "outcomeDays": int(
                (pair.index + pd.Timedelta(hours=lag_hours + 2))
                .normalize().nunique()
            ),
            "sourceComponentDeltaSpearman": transport_spearman(
                pair, "sourceComponent925", "futureDelta"
            ),
            "offsetCorrelations": offsets,
            "halfCorrelations": half_correlations,
            "walkForward": prospective,
            "stable": stable,
        }

    validation_reasons = []
    if issued_days < TRANSPORT_SHADOW_MIN_ISSUED_DAYS:
        validation_reasons.append(
            f"{issued_days}/{TRANSPORT_SHADOW_MIN_ISSUED_DAYS} issued-forecast days"
        )
    if counterfactual_origins < TRANSPORT_SHADOW_MIN_COUNTERFACTUAL_ORIGINS:
        validation_reasons.append(
            f"{counterfactual_origins}/{TRANSPORT_SHADOW_MIN_COUNTERFACTUAL_ORIGINS} "
            "low-alignment 925 hPa origins"
        )
    for lag_hours, result in by_lag.items():
        scored = result["walkForward"]["scoredOrigins"]
        if scored < TRANSPORT_SHADOW_MIN_PROSPECTIVE_ORIGINS:
            validation_reasons.append(
                f"{lag_hours} h walk-forward {scored}/"
                f"{TRANSPORT_SHADOW_MIN_PROSPECTIVE_ORIGINS} origins"
            )
    if stable_lags < TRANSPORT_SHADOW_MIN_STABLE_LAGS:
        validation_reasons.append(
            f"{stable_lags}/{TRANSPORT_SHADOW_MIN_STABLE_LAGS} stable pre-registered lags"
        )

    eligible = bool(
        issued_days >= TRANSPORT_SHADOW_MIN_ISSUED_DAYS
        and counterfactual_origins >= TRANSPORT_SHADOW_MIN_COUNTERFACTUAL_ORIGINS
        and stable_lags >= TRANSPORT_SHADOW_MIN_STABLE_LAGS
    )
    if eligible:
        state = "shadow_supported"
        reason = (
            f"{issued_days} issued-forecast days; the pre-registered 12/18/24 h "
            "shadow gate passed. Still not used in PM2.5 forecasts."
        )
    else:
        state = "shadow_collecting" if (
            issued_days < TRANSPORT_SHADOW_MIN_ISSUED_DAYS
            or counterfactual_origins
            < TRANSPORT_SHADOW_MIN_COUNTERFACTUAL_ORIGINS
        ) else "shadow_inconclusive"
        reason = (
            f"{issued_days} issued-forecast days · "
            f"{counterfactual_origins}/"
            f"{TRANSPORT_SHADOW_MIN_COUNTERFACTUAL_ORIGINS} low-alignment "
            "925 hPa origins. "
            "Prospective 12/18/24 h scoring is collecting. "
            "Not used in PM2.5 forecasts."
        )

    return {
        **base,
        "state": state,
        "reason": reason,
        "current": current,
        "evidence": {
            "issuedOriginHours": int(len(valid_issued)),
            "issuedForecastDays": issued_days,
            "counterfactualDefinition": "925 hPa source alignment below 0.5",
            "counterfactualOrigins": counterfactual_origins,
            "byLag": by_lag,
            "validation": {
                "eligible": eligible,
                "stableLagCount": stable_lags,
                "requiredStableLags": TRANSPORT_SHADOW_MIN_STABLE_LAGS,
                "reasons": validation_reasons,
            },
        },
    }


def regional_transport_evidence(as_of_epoch):
    """Hourly single-flight cache for bounded, shadow-only transport evidence."""
    sensor_hour = (int(as_of_epoch) // 3600) * 3600
    with db() as conn:
        latest_sensor = conn.execute(
            "SELECT MAX(epoch) FROM readings WHERE epoch<=?", (int(as_of_epoch),)
        ).fetchone()[0]
        latest_issued_run = conn.execute(
            "SELECT MAX(fetched_epoch) FROM weather_forecast_runs "
            "WHERE fetched_epoch<=?", (sensor_hour,)
        ).fetchone()[0]
    cache_key = (
        TRANSPORT_SHADOW_MODEL_VERSION,
        int(latest_sensor or 0) // 3600,
        int(latest_issued_run or 0),
    )
    with transport_evidence_lock:
        if transport_evidence_cache.get("cacheKey") == cache_key:
            return transport_evidence_cache["value"]
        value = _compute_regional_transport_evidence(int(as_of_epoch))
        transport_evidence_cache["cacheKey"] = cache_key
        transport_evidence_cache["value"] = value
        return value


def issued_station_match(issued, as_of_epoch=None):
    if issued.empty or "valid1Epoch" not in issued:
        return {}
    bounded_issued = issued
    if as_of_epoch is not None:
        # Station observations are outcomes, so both their valid time and the
        # time they first became available must precede a replay boundary.
        bounded_issued = issued.loc[
            pd.to_numeric(issued["valid1Epoch"], errors="coerce")
            .le(int(as_of_epoch))
        ]
    if bounded_issued.empty:
        return {}
    first_epoch = int(bounded_issued["valid1Epoch"].min())
    last_epoch = int(bounded_issued["valid1Epoch"].max())
    with db() as conn:
        rows = conn.execute(
            "SELECT report_epoch,atmp,rhum,wind_speed_kmh,wind_direction "
            "FROM subang_observations WHERE report_epoch BETWEEN ? AND ? "
            "AND COALESCE(first_fetched_epoch,fetched_epoch)<=?",
            (first_epoch, last_epoch, int(as_of_epoch or last_epoch)),
        ).fetchall()
    station = {int(row["report_epoch"]): dict(row) for row in rows}
    comparisons = []
    for _, forecast in bounded_issued.iterrows():
        observation = station.get(int(forecast["valid1Epoch"]))
        if observation:
            comparisons.append({**forecast.to_dict(), **{
                "stationTemperature": observation.get("atmp"),
                "stationHumidity": observation.get("rhum"),
                "stationWindSpeed": observation.get("wind_speed_kmh"),
                "stationWindDirection": observation.get("wind_direction"),
            }})
    if not comparisons:
        return {}
    frame = pd.DataFrame.from_records(comparisons)
    result = {}
    checks = [
        ("temperature1", "stationTemperature", "temperatureMae"),
        ("humidity1", "stationHumidity", "humidityMae"),
        ("windSpeed10m1", "stationWindSpeed", "windSpeedMae"),
    ]
    for forecast_column, observed_column, result_key in checks:
        pair = frame[[forecast_column, observed_column]].dropna()
        if not pair.empty:
            result[result_key] = round(
                float((pair[forecast_column] - pair[observed_column]).abs().mean()), 1
            )
            result["hours"] = max(result.get("hours", 0), len(pair))
    pair = frame[["windDirection10m1", "stationWindDirection"]].dropna()
    if not pair.empty:
        error = (
            (pair["windDirection10m1"] - pair["stationWindDirection"] + 180) % 360 - 180
        ).abs()
        result["windDirectionMae"] = round(float(error.mean()))
    return result


def weather_evidence(rows, payload=None, as_of_epoch=None):
    """Validate weather predictors only against forecasts archived before outcomes."""
    if not rows:
        return {
            "supported": False, "hours": 0, "originCount": 0,
            "minimumOrigins": ISSUED_WEATHER_MIN_ORIGINS,
            "validationMode": "issued_forecasts",
            "note": "Issued-forecast validation is still collecting.",
        }
    as_of_epoch = int(as_of_epoch or rows[-1]["epoch"])
    sensor = sensor_frame(rows).resample(
        "1h", label="right", closed="right"
    ).median(numeric_only=True)
    as_of_time = pd.Timestamp(as_of_epoch, unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    sensor = sensor.loc[sensor.index <= as_of_time]
    issued = archived_issued_weather(sensor.index, as_of_epoch)
    if sensor.empty or issued.empty:
        return {
            "supported": False, "hours": 0, "originCount": 0,
            "minimumOrigins": ISSUED_WEATHER_MIN_ORIGINS,
            "validationMode": "issued_forecasts",
            "note": "Issued-forecast validation is still collecting.",
        }

    # Derive elapsed-time deltas on the complete regular sensor grid before
    # selecting issued-forecast origins.  Shifting after the inner join could
    # silently span several hours whenever an issued run was missing.
    sensor_targets = sensor[["pm02"]].copy()
    sensor_targets["pastDelta1"] = (
        sensor_targets["pm02"] - sensor_targets["pm02"].shift(1)
    )
    sensor_targets["delta3"] = (
        sensor_targets["pm02"].shift(-3) - sensor_targets["pm02"]
    )
    joined = sensor_targets.join(issued, how="inner")
    base = joined[["windSpeed180m", "delta3"]].dropna()
    sensitivity = joined.loc[
        joined["pastDelta1"].abs().le(30), ["windSpeed180m", "delta3"]
    ].dropna()

    def correlation(frame):
        if len(frame) < 5:
            return None
        return float(frame.corr(method="spearman").iloc[0, 1])

    overall_correlation = correlation(base)
    sensitivity_correlation = correlation(sensitivity)
    offsets = []
    for offset in range(3):
        subset = base.iloc[offset::3]
        value = correlation(subset)
        offsets.append({
            "offset": offset,
            "origins": int(len(subset)),
            "spearman": None if value is None else round(value, 2),
        })
    stable_offsets = bool(offsets) and all(
        item["origins"] >= ISSUED_WEATHER_MIN_OFFSET_ORIGINS
        and item["spearman"] is not None and item["spearman"] <= -0.10
        for item in offsets
    )
    supported = bool(
        len(base) >= ISSUED_WEATHER_MIN_ORIGINS
        and overall_correlation is not None and overall_correlation <= -0.25
        and sensitivity_correlation is not None and sensitivity_correlation <= -0.20
        and stable_offsets
    )
    remaining = max(0, ISSUED_WEATHER_MIN_ORIGINS - len(base))
    return {
        "supported": supported,
        "hours": int(len(base)),
        "originCount": int(len(base)),
        "minimumOrigins": ISSUED_WEATHER_MIN_ORIGINS,
        "remainingOrigins": int(remaining),
        "validationMode": "issued_forecasts",
        "windPm3hSpearman": (
            None if overall_correlation is None else round(overall_correlation, 2)
        ),
        "sensitivitySpearman": (
            None if sensitivity_correlation is None else round(sensitivity_correlation, 2)
        ),
        "offsetCorrelations": offsets,
        "stationMatch": issued_station_match(issued, as_of_epoch),
        "note": (
            "Issued wind forecasts pass the stability checks and may support dispersion context."
            if supported else
            f'Issued wind forecast check: {len(base)}/{ISSUED_WEATHER_MIN_ORIGINS} origins; '
            "no PM adjustment is validated."
        ),
    }


def classify_pm_window_pair(morning_mean, afternoon_mean,
                            morning_peak, afternoon_peak):
    """Apply one material-separation rule to scoring and live comparison."""
    values = [morning_mean, afternoon_mean, morning_peak, afternoon_peak]
    if any(num(value) is None for value in values):
        return {
            "called": False, "winner": None,
            "reason": "complete_mean_and_peak_required",
        }
    morning_mean, afternoon_mean, morning_peak, afternoon_peak = map(
        float, values
    )
    mean_gap = morning_mean - afternoon_mean
    peak_gap = morning_peak - afternoon_peak
    mean_threshold = max(5.0, 0.10 * min(morning_mean, afternoon_mean))
    mean_winner = "morning" if mean_gap < 0 else "afternoon"
    peak_winner = (
        "morning" if peak_gap < 0 else "afternoon"
    ) if abs(peak_gap) >= 5.0 else None
    called = bool(
        abs(mean_gap) >= mean_threshold and peak_winner == mean_winner
    )
    return {
        "called": called,
        "winner": mean_winner if called else None,
        "meanGap": mean_gap,
        "peakGap": peak_gap,
        "meanThreshold": mean_threshold,
        "peakThreshold": 5.0,
        "peakAgrees": peak_winner == mean_winner,
        "reason": "material_mean_and_agreeing_peak" if called
        else "not_materially_separated",
    }


def wilson_lower_bound(successes, total, z=1.96):
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def _compute_air_quality_issued_evidence(rows, as_of_epoch=None):
    """Score CAMS against the exact issued ride-window means and peaks.

    A retrieved forecast can only earn permission for the operational task it
    was tested on: the 07:30 morning decision, the 07:30 provisional afternoon
    comparison, or the 12:30 final afternoon check.  Each record is one fixed
    two-hour modeled session on one day; isolated hourly targets cannot activate this
    model.
    """
    as_of_epoch = int(as_of_epoch or rows[-1]["epoch"])
    with db() as conn:
        run_rows = conn.execute(
            "SELECT fetched_epoch,payload FROM air_quality_forecast_runs "
            "WHERE model_version IN (?,?) AND fetched_epoch<=? "
            "ORDER BY fetched_epoch"
            , (*AIR_QUALITY_COMPATIBLE_ARCHIVES, as_of_epoch)
        ).fetchall()

    runs = []
    for row in run_rows:
        try:
            payload = json.loads(row["payload"])
            points = {
                int(point["epoch"]): point for point in payload.get("hourly", [])
                if point.get("epoch") is not None
            }
            if points:
                runs.append((int(row["fetched_epoch"]), points))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    base = {
        "supported": False,
        "usedForDecision": False,
        "validationMode": "causal_issued_model_archive_replay",
        "modelVersion": AIR_QUALITY_FORECAST_METHOD_VERSION,
        "sourceArchiveVersion": AIR_QUALITY_MODEL_VERSION,
        "compatibleSourceArchives": list(AIR_QUALITY_COMPATIBLE_ARCHIVES),
        "retrievedSnapshotCount": len(runs),
        "scoredWindowCount": 0,
        "distinctDays": 0,
        "minimumScoredWindows": AIR_QUALITY_MIN_SCORED_WINDOWS,
        "minimumDistinctDays": AIR_QUALITY_MIN_ISSUED_DAYS,
        "minimumTaskWindows": AIR_QUALITY_MIN_TASK_WINDOWS,
        "minimumTaskDays": AIR_QUALITY_MIN_TASK_DAYS,
        "deltaShrink": AIR_QUALITY_DELTA_SHRINK,
        "target": "Two-hour PM2.5 mean and 15-minute-median peak",
        "modelSampling": (
            "Linear interpolation at the exact raw-sensor anchor and centres "
            "of right-labelled 15-minute outcome intervals, including the endpoint"
        ),
    }
    if not runs:
        base["status"] = "collecting"
        base["note"] = "Issued CAMS particle forecasts are now being archived for local scoring."
        return base

    first_run_epoch = runs[0][0]
    with db() as conn:
        sensor_rows = conn.execute(
            "SELECT epoch,pm02 FROM readings WHERE epoch>=? AND epoch<=? ORDER BY epoch",
            (first_run_epoch - 86400, as_of_epoch),
        ).fetchall()
    if not sensor_rows:
        base["status"] = "collecting"
        base["note"] = "Issued CAMS forecasts are archived; matching sensor outcomes are collecting."
        return base

    raw = pd.DataFrame.from_records([dict(row) for row in sensor_rows])
    raw["epoch"] = pd.to_numeric(raw["epoch"], errors="coerce")
    raw["pm02"] = pd.to_numeric(raw["pm02"], errors="coerce")
    raw = raw.dropna(subset=["epoch"]).sort_values("epoch")
    raw.index = pd.DatetimeIndex(
        pd.to_datetime(raw.pop("epoch"), unit="s", utc=True)
    ).tz_convert("Asia/Kuala_Lumpur")
    raw = raw.dropna(subset=["pm02"])
    if raw.empty:
        base["status"] = "collecting"
        base["note"] = "Issued CAMS forecasts are archived; matching sensor outcomes are collecting."
        return base

    sensor = pd.DataFrame({
        "pm02": raw["pm02"].resample(
            f"{AIR_BUCKET_MINUTES}min", label="right", closed="right"
        ).median(),
        "readingCount": raw["pm02"].resample(
            f"{AIR_BUCKET_MINUTES}min", label="right", closed="right"
        ).count(),
    })
    quality = bucket_coverage(raw)
    sensor.loc[~quality["forecastEligible"], "pm02"] = np.nan
    latest_observed = raw.index[-1]
    run_epochs = np.array([run[0] for run in runs], dtype="int64")
    tasks = (
        ("morning", 7, 30, 9, 11),
        ("afternoon_provisional", 7, 30, 14, 16),
        ("afternoon_final", 12, 30, 14, 16),
    )
    records = []
    first_day = datetime.fromtimestamp(first_run_epoch, KL_TZ).date()
    last_day = latest_observed.date()
    day = first_day
    while day <= last_day:
        for task, decision_hour, decision_minute, start_hour, end_hour in tasks:
            decision = pd.Timestamp(datetime(
                day.year, day.month, day.day, decision_hour, decision_minute,
                tzinfo=KL_TZ,
            ))
            start = pd.Timestamp(datetime(
                day.year, day.month, day.day, start_hour, tzinfo=KL_TZ,
            ))
            end = pd.Timestamp(datetime(
                day.year, day.month, day.day, end_hour, tzinfo=KL_TZ,
            ))
            if end > latest_observed:
                continue

            decision_epoch = int(decision.timestamp())
            run_index = int(np.searchsorted(run_epochs, decision_epoch, side="right") - 1)
            if run_index < 0:
                continue
            fetched_epoch, points = runs[run_index]
            if decision_epoch - fetched_epoch > AIR_QUALITY_MAX_DECISION_AGE_SECONDS:
                continue

            anchor_candidates = sensor.loc[
                (sensor.index > decision - pd.Timedelta(minutes=AIR_BUCKET_MINUTES))
                & (sensor.index <= decision)
                & sensor["readingCount"].ge(AIR_QUALITY_MIN_ORIGIN_READINGS)
            ]
            if anchor_candidates.empty:
                continue
            raw_anchor = raw.loc[
                (raw.index > decision - pd.Timedelta(minutes=AIR_BUCKET_MINUTES))
                & (raw.index <= decision), "pm02"
            ].dropna()
            if raw_anchor.empty:
                continue
            anchor_time = raw_anchor.index[-1]
            anchor_pm = float(raw_anchor.iloc[-1])

            target_index = pd.date_range(
                start + pd.Timedelta(minutes=AIR_BUCKET_MINUTES), end,
                freq=f"{AIR_BUCKET_MINUTES}min",
            )
            observed = sensor.reindex(target_index)
            observed = observed.loc[
                observed["readingCount"].ge(AIR_QUALITY_MIN_ORIGIN_READINGS),
                "pm02",
            ].dropna()
            if len(observed) < AIR_QUALITY_MIN_WINDOW_BUCKETS:
                continue

            issued_model = model_frame({"hourly": list(points.values())})
            model_origin, model_target = aligned_model_projection(
                issued_model, anchor_time, start, end
            )
            if model_origin is None or len(model_target) != len(target_index):
                continue
            # A complete target is required; sleep gaps are never partial scores.
            scored_model_target = model_target.reindex(observed.index)
            if (len(scored_model_target) != len(observed)
                    or scored_model_target.isna().any()):
                continue
            model_mean = float(scored_model_target.mean())
            model_peak = float(scored_model_target.max())
            candidate_mean = max(
                0.0, anchor_pm + AIR_QUALITY_DELTA_SHRINK * (model_mean - model_origin)
            )
            candidate_peak = max(
                candidate_mean,
                anchor_pm + AIR_QUALITY_DELTA_SHRINK * (model_peak - model_origin),
            )
            records.append({
                "day": day,
                "task": task,
                "decision": decision,
                "fetchedEpoch": fetched_epoch,
                "forecastAgeMinutes": (decision_epoch - fetched_epoch) / 60,
                "anchorEpoch": int(anchor_time.timestamp()),
                "anchorAgeMinutes": (decision - anchor_time).total_seconds() / 60,
                "observedBucketCount": int(len(observed)),
                "modelBucketCount": int(len(scored_model_target)),
                "fullModelBucketCount": int(len(model_target)),
                "actualMean": float(observed.mean()),
                "actualPeak": float(observed.max()),
                "persistenceMean": anchor_pm,
                "persistencePeak": anchor_pm,
                "candidateMean": candidate_mean,
                "candidatePeak": candidate_peak,
            })
        day += timedelta(days=1)

    if not records:
        base["status"] = "collecting"
        base["note"] = "No issued CAMS forecast has completed a full scored ride session yet."
        return base

    scored = pd.DataFrame.from_records(records)
    scored_window_count = int(len(scored))
    distinct_days = int(scored["day"].nunique())

    def metrics(frame):
        mean_persistence_error = (frame["actualMean"] - frame["persistenceMean"]).abs()
        mean_candidate_error = (frame["actualMean"] - frame["candidateMean"]).abs()
        peak_persistence_error = (frame["actualPeak"] - frame["persistencePeak"]).abs()
        peak_candidate_error = (frame["actualPeak"] - frame["candidatePeak"]).abs()
        mean_persistence_mae = float(mean_persistence_error.mean())
        mean_candidate_mae = float(mean_candidate_error.mean())
        peak_persistence_mae = float(peak_persistence_error.mean())
        peak_candidate_mae = float(peak_candidate_error.mean())
        mean_gain = (
            0.0 if mean_persistence_mae <= 1e-9
            else 1.0 - mean_candidate_mae / mean_persistence_mae
        )
        peak_gain = (
            0.0 if peak_persistence_mae <= 1e-9
            else 1.0 - peak_candidate_mae / peak_persistence_mae
        )
        actual_delta = frame["actualMean"] - frame["persistenceMean"]
        candidate_delta = frame["candidateMean"] - frame["persistenceMean"]
        direction = float((np.sign(actual_delta) == np.sign(candidate_delta)).mean())
        mean_win_fraction = float((mean_candidate_error < mean_persistence_error).mean())
        peak_win_fraction = float((peak_candidate_error < peak_persistence_error).mean())
        mean_residual = frame["actualMean"] - frame["candidateMean"]
        peak_residual = frame["actualPeak"] - frame["candidatePeak"]
        raw = {
            "meanGain": mean_gain,
            "peakGain": peak_gain,
            "direction": direction,
            "meanWinFraction": mean_win_fraction,
            "peakWinFraction": peak_win_fraction,
        }
        result = {
            "testCount": int(len(frame)),
            "distinctDays": int(frame["day"].nunique()),
            "meanPersistenceMae": round(mean_persistence_mae, 2),
            "meanCandidateMae": round(mean_candidate_mae, 2),
            "meanSkillGainPct": round(100 * mean_gain, 1),
            "meanCandidateSignedBias": round(float((frame["actualMean"] - frame["candidateMean"]).mean()), 2),
            "meanPersistenceSignedBias": round(float((frame["actualMean"] - frame["persistenceMean"]).mean()), 2),
            "peakPersistenceMae": round(peak_persistence_mae, 2),
            "peakCandidateMae": round(peak_candidate_mae, 2),
            "peakSkillGainPct": round(100 * peak_gain, 1),
            "peakCandidateSignedBias": round(float((frame["actualPeak"] - frame["candidatePeak"]).mean()), 2),
            "peakPersistenceSignedBias": round(float((frame["actualPeak"] - frame["persistencePeak"]).mean()), 2),
            "meanDirectionAccuracyPct": round(100 * direction),
            "meanPositiveDayPct": round(100 * mean_win_fraction),
            "peakPositiveDayPct": round(100 * peak_win_fraction),
            "residualCalibration": {
                "meanLow": round(empirical_quantile(mean_residual, 0.10), 2),
                "meanHigh": round(empirical_quantile(mean_residual, 0.90, upper=True), 2),
                "peakUpper": round(empirical_quantile(peak_residual, 0.90, upper=True), 2),
                "mode": (
                    "raw task-specific empirical residual quantiles; not unioned "
                    "with persistence bounds"
                ),
            },
        }
        return result, raw

    overall, overall_raw = metrics(scored)
    global_sample_gate = bool(
        scored_window_count >= AIR_QUALITY_MIN_SCORED_WINDOWS
        and distinct_days >= AIR_QUALITY_MIN_ISSUED_DAYS
    )
    by_task = {}
    for task, group in scored.groupby("task"):
        task_result, task_raw = metrics(group)
        task_supported = bool(
            global_sample_gate
            and task_result["testCount"] >= AIR_QUALITY_MIN_TASK_WINDOWS
            and task_result["distinctDays"] >= AIR_QUALITY_MIN_TASK_DAYS
            and task_raw["meanGain"] >= AIR_QUALITY_MIN_SKILL_GAIN
            and task_raw["peakGain"] >= AIR_QUALITY_MIN_SKILL_GAIN
            and task_raw["direction"] >= 0.65
            and task_raw["meanWinFraction"] >= 0.60
            and task_raw["peakWinFraction"] >= 0.60
        )
        task_result["supported"] = task_supported
        by_task[task] = task_result

    # Score the actual 07:30 choice as a pair.  A model that improves two
    # isolated MAEs can still rank Morning versus Afternoon incorrectly, so an
    # abstention is recorded separately from a directional call.
    pair_records = []
    for scored_day, group in scored.groupby("day"):
        task_rows = {row["task"]: row for _, row in group.iterrows()}
        morning_row = task_rows.get("morning")
        afternoon_row = task_rows.get("afternoon_provisional")
        if morning_row is None or afternoon_row is None:
            continue
        candidate_pair = classify_pm_window_pair(
            morning_row["candidateMean"], afternoon_row["candidateMean"],
            morning_row["candidatePeak"], afternoon_row["candidatePeak"],
        )
        actual_pair = classify_pm_window_pair(
            morning_row["actualMean"], afternoon_row["actualMean"],
            morning_row["actualPeak"], afternoon_row["actualPeak"],
        )
        called = bool(candidate_pair["called"])
        pair_records.append({
            "day": scored_day,
            "called": called,
            "candidateWinner": candidate_pair["winner"],
            "actualMaterial": bool(actual_pair["called"]),
            "actualWinner": actual_pair["winner"],
            "correct": bool(
                called and actual_pair["called"]
                and candidate_pair["winner"] == actual_pair["winner"]
            ),
        })
    pair_calls = [record for record in pair_records if record["called"]]
    correct_pair_calls = sum(record["correct"] for record in pair_calls)
    pair_wilson_lower = wilson_lower_bound(correct_pair_calls, len(pair_calls))
    pair_supported = bool(
        len(pair_records) >= AIR_QUALITY_MIN_PAIR_DAYS
        and len(pair_calls) >= AIR_QUALITY_MIN_PAIR_CALLS
        and pair_wilson_lower > AIR_QUALITY_MIN_PAIR_WILSON_LOWER
    )
    pair_backtest = {
        "eligiblePairCount": len(pair_records),
        "callCount": len(pair_calls),
        "abstentionCount": len(pair_records) - len(pair_calls),
        "correctCallCount": correct_pair_calls,
        "callAccuracyPct": (
            None if not pair_calls else
            round(100 * correct_pair_calls / len(pair_calls))
        ),
        "wilson95Lower": round(pair_wilson_lower, 3),
        "supported": pair_supported,
        "minimumPairDays": AIR_QUALITY_MIN_PAIR_DAYS,
        "minimumCalls": AIR_QUALITY_MIN_PAIR_CALLS,
        "minimumWilson95LowerExclusive": AIR_QUALITY_MIN_PAIR_WILSON_LOWER,
        "policy": (
            "Material mean separation plus agreeing >=5 ug/m3 peak separation; "
            "actual unresolved days count as incorrect calls"
        ),
    }
    expected_tasks = {task[0] for task in tasks}
    supported = bool(
        expected_tasks.issubset(by_task)
        and all(by_task[task].get("supported") for task in expected_tasks)
        and pair_supported
    )
    base.update({
        "supported": supported,
        "usedForDecision": supported,
        "status": "validated" if supported else "collecting",
        "scoredWindowCount": scored_window_count,
        "distinctDays": distinct_days,
        "remainingScoredWindows": max(
            0, AIR_QUALITY_MIN_SCORED_WINDOWS - scored_window_count
        ),
        "remainingDistinctDays": max(0, AIR_QUALITY_MIN_ISSUED_DAYS - distinct_days),
        "backtest": overall,
        "byTask": by_task,
        "pairBacktest": pair_backtest,
        "dependenceControl": "One fixed operational window per task and local day",
        "coverageNote": (
            "Residual bounds are empirical day-blocked heuristics, not calibrated probabilities."
        ),
        "note": (
            "Sensor-anchored CAMS changes pass separate exact-window mean and peak checks."
            if supported else
            "CAMS remains experimental until issued two-hour means and peaks beat persistence for each forecast task."
        ),
    })
    return base


def air_quality_issued_evidence(rows, as_of_epoch=None):
    """Single-flight cache for prospective evidence, refreshed per sensor hour."""
    as_of_epoch = int(as_of_epoch or rows[-1]["epoch"])
    with db() as conn:
        latest_run = conn.execute(
            "SELECT MAX(fetched_epoch) FROM air_quality_forecast_runs "
            "WHERE model_version IN (?,?) AND fetched_epoch<=?",
            (*AIR_QUALITY_COMPATIBLE_ARCHIVES, as_of_epoch)
        ).fetchone()[0]
    latest_sensor = min(as_of_epoch, int(rows[-1]["epoch"]))
    cache_key = (
        AIR_QUALITY_COMPATIBLE_ARCHIVES,
        int(latest_run or 0),
        int(latest_sensor or 0) // POLL_SECONDS,
        ride_forecast_sensor_revision(rows),
    )
    with air_quality_evidence_lock:
        if air_quality_evidence_cache.get("cacheKey") == cache_key:
            return air_quality_evidence_cache["value"]
        value = _compute_air_quality_issued_evidence(rows, as_of_epoch)
        air_quality_evidence_cache["cacheKey"] = cache_key
        air_quality_evidence_cache["value"] = value
        return value


def retrospective_air_quality_diagnostic(rows, payload):
    """Describe current CAMS alignment; never use revised past values as validation."""
    sensor = sensor_frame(rows).resample(
        "1h", label="right", closed="right"
    ).median(numeric_only=True)
    model = model_frame(payload)
    if sensor.empty or model.empty or "pm2_5" not in model:
        return {"available": False, "validationMode": "diagnostic_only"}
    joined = sensor[["pm02"]].join(
        model[["pm2_5"]].rename(columns={"pm2_5": "modelPm25"}), how="inner"
    ).dropna()
    if joined.empty:
        return {"available": False, "validationMode": "diagnostic_only"}
    error = joined["pm02"] - joined["modelPm25"]
    result = {
        "available": True,
        "validationMode": "retrieved_past_diagnostic_only",
        "pairedHours": int(len(joined)),
        "medianLevelBiasUgM3": round(float(error.median()), 1),
        "levelMaeUgM3": round(float(error.abs().mean()), 1),
        "levelSpearman": (
            None if len(joined) < 5
            else round(float(joined.corr(method="spearman").iloc[0, 1]), 2)
        ),
        "note": "Retrieved past CAMS values can diagnose bias but cannot validate an issued forecast.",
    }
    horizons = {}
    for horizon in (3, 6, 9, 12):
        change = pd.DataFrame({
            "actual": joined["pm02"].shift(-horizon) - joined["pm02"],
            "model": joined["modelPm25"].shift(-horizon) - joined["modelPm25"],
        }).dropna()
        if change.empty:
            continue
        persistence_mae = float(change["actual"].abs().mean())
        candidate_mae = float(
            (change["actual"] - AIR_QUALITY_DELTA_SHRINK * change["model"])
            .abs().mean()
        )
        horizons[str(horizon)] = {
            "pairs": int(len(change)),
            "persistenceMae": round(persistence_mae, 1),
            "candidateMae": round(candidate_mae, 1),
        }
    result["byHorizonHours"] = horizons
    return result


def summarize_weather_range(payload, start_epoch, end_epoch, evidence):
    # Instantaneous fields use [start,end); hourly precipitation and its
    # probability describe the preceding hour.  Include every preceding-hour
    # interval that overlaps a shifted session and prorate accumulation at the
    # two edges; probability remains the maximum of all overlapping hours.
    instant_points = [
        point for point in (payload or {}).get("hourly", [])
        if start_epoch <= int(point["epoch"]) < end_epoch
    ]
    accumulation_intervals = []
    for point in (payload or {}).get("hourly", []):
        point_end = int(point["epoch"])
        point_start = point_end - 3600
        overlap_seconds = max(
            0, min(end_epoch, point_end) - max(start_epoch, point_start)
        )
        if overlap_seconds > 0:
            accumulation_intervals.append((point, overlap_seconds / 3600.0))
    accumulation_points = [point for point, _ in accumulation_intervals]
    if not instant_points and not accumulation_points:
        return {"available": False}

    def values(key, points=None):
        source = instant_points if points is None else points
        return [float(point[key]) for point in source if point.get(key) is not None]

    wind180 = values("wind_speed_180m")
    wind10 = values("wind_speed_10m")
    # Open-Meteo gusts are preceding-hour maxima, so include every hourly
    # interval that overlaps the session rather than treating them as points.
    gust10 = values("wind_gusts_10m", accumulation_points)
    apparent = values("apparent_temperature")
    temperature = values("temperature_2m")
    humidity = values("relative_humidity_2m")
    cloud_cover = values("cloud_cover")
    pressure = values("pressure_msl")
    rain_probability = values("precipitation_probability", accumulation_points)
    precipitation = [
        float(point["precipitation"]) * overlap
        for point, overlap in accumulation_intervals
        if point.get("precipitation") is not None
    ]
    showers = [
        float(point["showers"]) * overlap
        for point, overlap in accumulation_intervals
        if point.get("showers") is not None
    ]
    direction180 = circular_mean([
        point.get("wind_direction_180m") for point in instant_points
    ])
    direction10 = circular_mean([
        point.get("wind_direction_10m") for point in instant_points
    ])
    direction925 = circular_mean([
        point.get("wind_direction_925hPa") for point in instant_points
    ])
    alignments = [
        source_alignment(point.get("wind_direction_925hPa"))
        for point in instant_points
    ]
    alignments = [value for value in alignments if value is not None]
    mean_wind180 = sum(wind180) / len(wind180) if wind180 else None

    if mean_wind180 is None:
        ventilation = "Airflow forecast unavailable"
        ventilation_state = "unknown"
    elif mean_wind180 > 14:
        ventilation = (
            "Stronger issued-wind dispersion signal"
            if evidence.get("supported") else
            "Strong modeled flow aloft · clearing not assured"
        )
        ventilation_state = "strong"
    elif mean_wind180 >= 10:
        ventilation = (
            "Moderate issued-wind dispersion signal"
            if evidence.get("supported") else
            "Moderate modeled flow aloft"
        )
        ventilation_state = "moderate"
    else:
        ventilation = "Weak modeled flow aloft"
        ventilation_state = "weak"

    max_rain = max(rain_probability) if rain_probability else None
    rain_sum = sum(precipitation) if precipitation else None
    shower_sum = sum(showers) if showers else None
    mean_cloud = sum(cloud_cover) / len(cloud_cover) if cloud_cover else None
    if mean_cloud is None:
        sky_label = "Sky forecast unavailable"
    elif mean_cloud <= 20:
        sky_label = "Mostly clear"
    elif mean_cloud <= 50:
        sky_label = "Partly cloudy"
    elif mean_cloud <= 80:
        sky_label = "Mostly cloudy"
    else:
        sky_label = "Overcast"
    if max_rain is None:
        rain_label = "Rain forecast unavailable"
    elif max_rain >= 60 or (rain_sum or 0) >= 1.0:
        rain_label = "Higher modeled rain chance"
    elif max_rain >= 30:
        rain_label = "Some modeled rain chance"
    else:
        rain_label = "Lower modeled rain chance"

    return {
        "available": True,
        "startEpoch": int(start_epoch),
        "endEpoch": int(end_epoch),
        "sourceFetchedEpoch": (payload or {}).get("fetchedEpoch"),
        "pointCount": len(instant_points),
        "accumulationPointCount": len(accumulation_points),
        "accumulationEquivalentHours": round(
            sum(overlap for _, overlap in accumulation_intervals), 2
        ),
        "apparentTemperatureMax": round(max(apparent), 1) if apparent else None,
        "temperatureMax": round(max(temperature), 1) if temperature else None,
        "relativeHumidityMean": (
            round(sum(humidity) / len(humidity)) if humidity else None
        ),
        "precipitationProbabilityMax": round(max_rain) if max_rain is not None else None,
        "precipitationMm": round(rain_sum, 1) if rain_sum is not None else None,
        "windSpeed10mMean": round(sum(wind10) / len(wind10), 1) if wind10 else None,
        "windGust10mMax": round(max(gust10), 1) if gust10 else None,
        "windDirection10m": round(direction10) if direction10 is not None else None,
        "windDirection10mCompass": compass_direction(direction10),
        "cloudCoverMean": round(mean_cloud) if mean_cloud is not None else None,
        "cloudCoverMax": round(max(cloud_cover)) if cloud_cover else None,
        "skyLabel": sky_label,
        "pressureMslMin": round(min(pressure), 1) if pressure else None,
        "showersMm": round(shower_sum, 1) if shower_sum is not None else None,
        "windSpeed180mMean": round(mean_wind180, 1) if mean_wind180 is not None else None,
        "windDirection180m": round(direction180) if direction180 is not None else None,
        "windDirection180mCompass": compass_direction(direction180),
        "windDirection925hPa": round(direction925) if direction925 is not None else None,
        "sourceAlignment925": round(sum(alignments) / len(alignments), 2) if alignments else None,
        "ventilationState": ventilation_state,
        "ventilationLabel": ventilation,
        "ventilationUsed": bool(evidence.get("supported")),
        "rainLabel": rain_label,
        "pmForecastAdjustment": 0.0,
    }


def next_feasible_ride_period(latest_epoch, config):
    """Return the next comparable 120-minute session reachable in a ride window."""
    now_local = datetime.fromtimestamp(latest_epoch, KL_TZ)
    for day_offset in range(4):
        day = (now_local + timedelta(days=day_offset)).date()
        window_start = datetime(
            day.year, day.month, day.day, config["start"], tzinfo=KL_TZ
        )
        window_end = datetime(
            day.year, day.month, day.day, config["end"], tzinfo=KL_TZ
        )
        earliest = max(
            window_start, now_local + timedelta(minutes=ARRIVAL_MINUTES)
        )
        session_epoch = math.ceil(earliest.timestamp() / 900) * 900
        session_start = datetime.fromtimestamp(session_epoch, KL_TZ)
        session_end = session_start + timedelta(minutes=TRAIL_MINUTES)
        if session_end > window_end:
            continue
        decision = session_start - timedelta(minutes=ARRIVAL_MINUTES)
        return (
            day, window_start, window_end, session_start, session_end, decision
        )
    return None


def next_ride_window(latest_epoch, config, payload, evidence):
    period = next_feasible_ride_period(latest_epoch, config)
    if period:
        day, _, _, session_start, session_end, _ = period
        result = summarize_weather_range(
            payload, int(session_start.timestamp()), int(session_end.timestamp()), evidence
        )
        if result.get("available"):
            result.update({
                "date": day.isoformat(),
                "label": config["label"],
                "window": f'{config["start"]:02d}:00–{config["end"]:02d}:00',
                "modeledSession": (
                    f'{session_start:%H:%M}–{session_end:%H:%M}'
                ),
            })
            return result
    return {"available": False}


def weather_outlook(rows, latest_epoch, include_runtime_status=True):
    payload = latest_weather_payload(latest_epoch)
    if not payload:
        status = {}
        if include_runtime_status:
            with weather_lock:
                status = dict(weather_status)
        subang = latest_subang_observation(latest_epoch)
        if subang:
            subang = dict(subang)
            subang.pop("present_weather", None)
        return {
            "available": False,
            "message": status.get("error") or "Weather forecast is collecting.",
            "subang": subang,
        }

    evidence = weather_evidence(
        rows, payload, min(int(latest_epoch), int(rows[-1]["epoch"]))
    )
    trail = summarize_weather_range(
        payload,
        latest_epoch + ARRIVAL_MINUTES * 60,
        latest_epoch + (ARRIVAL_MINUTES + TRAIL_MINUTES) * 60,
        evidence,
    )
    window_weather = {
        config["key"]: next_ride_window(latest_epoch, config, payload, evidence)
        for config in RIDE_WINDOWS
    }
    subang = latest_subang_observation(latest_epoch)
    if subang:
        subang = dict(subang)
        # Keep station weather text in storage for precipitation classification,
        # but do not publish free-text hazard labels in the dashboard API.
        subang.pop("present_weather", None)
        subang["ageMinutes"] = max(0, round((latest_epoch - subang["report_epoch"]) / 60))
        subang["windCompass"] = compass_direction(subang.get("wind_direction"))

    transport = dict(regional_transport_evidence(latest_epoch))
    # Compatibility field for existing clients: this is the issued alignment
    # over the +90 to +210 minute trail horizon, never a PM adjustment.
    transport["alignment925"] = (
        trail.get("sourceAlignment925") if trail.get("available") else None
    )
    return {
        "available": True,
        "source": payload.get("source"),
        "fetchedEpoch": payload.get("fetchedEpoch"),
        "ageMinutes": max(0, round((latest_epoch - payload.get("fetchedEpoch", 0)) / 60)),
        "resolution": payload.get("resolution", "hourly"),
        "trail": trail,
        "windows": window_weather,
        "evidence": evidence,
        "subang": subang,
        "transport": transport,
        "attributionUrl": "https://open-meteo.com/",
    }


def issued_weather_point(origin_epoch):
    """Return the nearest hourly weather point issued no later than the event."""
    with db() as conn:
        row = conn.execute(
            "SELECT fetched_epoch,payload FROM weather_forecast_runs "
            "WHERE fetched_epoch<=? ORDER BY fetched_epoch DESC LIMIT 1",
            (int(origin_epoch),),
        ).fetchone()
    if not row:
        return {}
    try:
        payload = json.loads(row["payload"])
        points = [
            point for point in payload.get("hourly", [])
            if point.get("epoch") is not None
            and abs(int(point["epoch"]) - int(origin_epoch)) <= 3600
        ]
        if not points:
            return {}
        point = min(points, key=lambda item: abs(int(item["epoch"]) - int(origin_epoch)))
        return {
            "fetchedEpoch": int(row["fetched_epoch"]),
            "validEpoch": int(point["epoch"]),
            "rainProbability": num(point.get("precipitation_probability")),
            "precipitationMm": num(point.get("precipitation")),
            "windSpeed10m": num(point.get("wind_speed_10m")),
            "windDirection10m": num(point.get("wind_direction_10m")),
            "windSpeed180m": num(point.get("wind_speed_180m")),
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def issued_weather_horizon(origin_epoch, hours=3):
    """Return a future weather horizon from a run issued no later than origin."""
    with db() as conn:
        row = conn.execute(
            "SELECT fetched_epoch,payload FROM weather_forecast_runs "
            "WHERE fetched_epoch<=? ORDER BY fetched_epoch DESC LIMIT 1",
            (int(origin_epoch),),
        ).fetchone()
    if not row:
        return {}
    try:
        payload = json.loads(row["payload"])
        points = sorted(
            (
                point for point in payload.get("hourly", [])
                if point.get("epoch") is not None
            ),
            key=lambda point: int(point["epoch"]),
        )
        baseline_candidates = [
            point for point in points
            if abs(int(point["epoch"]) - int(origin_epoch)) <= 3600
        ]
        future = [
            point for point in points
            if int(point["epoch"]) > int(origin_epoch)
        ][:max(1, int(hours))]
        if not baseline_candidates or len(future) < hours:
            return {}
        baseline = min(
            baseline_candidates,
            key=lambda point: abs(int(point["epoch"]) - int(origin_epoch)),
        )

        def values(field):
            return [
                value for value in (num(point.get(field)) for point in future)
                if value is not None
            ]

        def mean(field):
            field_values = values(field)
            return (
                sum(field_values) / len(field_values) if field_values else None
            )

        def maximum(field):
            field_values = values(field)
            return max(field_values) if field_values else None

        last = future[-1]
        baseline_temperature = num(baseline.get("temperature_2m"))
        future_temperature = num(last.get("temperature_2m"))
        baseline_humidity = num(baseline.get("relative_humidity_2m"))
        future_humidity = num(last.get("relative_humidity_2m"))
        return {
            "fetchedEpoch": int(row["fetched_epoch"]),
            "firstValidEpoch": int(future[0]["epoch"]),
            "lastValidEpoch": int(last["epoch"]),
            "pointCount": len(future),
            "rainProbabilityMax": maximum("precipitation_probability"),
            "precipitationMax": maximum("precipitation"),
            "boundaryLayerHeightMean": mean("boundary_layer_height"),
            "windSpeed180mMean": mean("wind_speed_180m"),
            "windSpeed10mMean": mean("wind_speed_10m"),
            "temperatureChange": (
                None if baseline_temperature is None or future_temperature is None
                else future_temperature - baseline_temperature
            ),
            "humidityChange": (
                None if baseline_humidity is None or future_humidity is None
                else future_humidity - baseline_humidity
            ),
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def subang_event_context(confirmation_epoch, latest_epoch):
    """Describe station changes available within two hours after confirmation."""
    end_epoch = min(int(latest_epoch), int(confirmation_epoch) + 2 * 3600)
    with db() as conn:
        rows = conn.execute(
            "SELECT report_epoch,timestamp,atmp,rhum,wind_speed_kmh,wind_direction,"
            "dewpoint,visibility_km,present_weather,precipitation_mm "
            "FROM subang_observations WHERE report_epoch BETWEEN ? AND ? "
            "AND COALESCE(first_fetched_epoch,fetched_epoch)<=? "
            "ORDER BY report_epoch",
            (int(confirmation_epoch) - 2 * 3600, end_epoch, int(latest_epoch)),
        ).fetchall()
    observations = [dict(row) for row in rows]
    if not observations:
        return {}
    before = [row for row in observations if row["report_epoch"] <= confirmation_epoch]
    baseline = before[-1] if before else observations[0]
    current = observations[-1]

    def change(key):
        first_value, last_value = num(baseline.get(key)), num(current.get(key))
        if first_value is None or last_value is None:
            return None
        return round(last_value - first_value, 1)

    current_wind = num(current.get("wind_speed_kmh"))
    baseline_wind = num(baseline.get("wind_speed_kmh"))
    return {
        "baselineEpoch": int(baseline["report_epoch"]),
        "currentEpoch": int(current["report_epoch"]),
        "windSpeedKmh": current_wind,
        "windCompass": compass_direction(current.get("wind_direction")),
        "windSpeedChange": (
            None if current_wind is None or baseline_wind is None
            else round(current_wind - baseline_wind, 1)
        ),
        "humidityChange": change("rhum"),
        "dewpointChange": change("dewpoint"),
        "temperatureChange": change("atmp"),
        "visibilityKm": num(current.get("visibility_km")),
        "presentWeather": current.get("present_weather"),
        "precipitationMm": num(current.get("precipitation_mm")),
    }


def clearance_timeline(rows, confirmation_epoch, onset_epoch=None):
    """Return a raw-reading event timeline with no future bucket labels."""
    latest_epoch = int(rows[-1]["epoch"]) if rows else int(confirmation_epoch)
    confirmation_epoch = min(latest_epoch, int(confirmation_epoch))
    records = []
    for row in rows:
        keys = set(row.keys())
        epoch = num(row["epoch"] if "epoch" in keys else None)
        pm25 = num(row["pm02"] if "pm02" in keys else None)
        if epoch is not None and pm25 is not None and epoch <= latest_epoch:
            records.append({"epoch": int(epoch), "pm02": pm25})
    frame = pd.DataFrame.from_records(records).sort_values("epoch")
    if frame.empty:
        return {"confirmationEpoch": confirmation_epoch}

    if onset_epoch is None or int(onset_epoch) > confirmation_epoch:
        before = frame.loc[frame["epoch"].between(
            confirmation_epoch - 45 * 60, confirmation_epoch
        )]
        if before.empty:
            onset_epoch = confirmation_epoch
        else:
            maximum = float(before["pm02"].max())
            onset_epoch = int(before.loc[before["pm02"].eq(maximum), "epoch"].iloc[-1])
    else:
        onset_epoch = max(int(frame["epoch"].iloc[0]), int(onset_epoch))

    event_values = frame.loc[frame["epoch"].ge(onset_epoch)].copy()
    low_row = event_values.loc[event_values["pm02"].idxmin()]
    low_epoch = int(low_row["epoch"])
    low_value = float(low_row["pm02"])
    rebound_threshold = max(low_value + 10.0, low_value * 1.20)
    after_low = event_values.loc[event_values["epoch"].ge(low_epoch)]
    rebound_epoch = None
    for position in range(1, len(after_low)):
        pair = after_low.iloc[position - 1:position + 1]["pm02"]
        if len(pair) == 2 and bool(pair.ge(rebound_threshold).all()):
            rebound_epoch = int(after_low.iloc[position - 1]["epoch"])
            break
    return {
        "onsetEpoch": int(onset_epoch),
        "confirmationEpoch": confirmation_epoch,
        "lowEpoch": low_epoch,
        "lowPm25": round(low_value, 1),
        "reboundEpoch": rebound_epoch,
        "reboundThreshold": round(rebound_threshold, 1),
    }


def dry_clearing_feature_frame(rows):
    """Build the one causal feature frame shared by live and replay detectors."""
    columns = [
        "epoch", "pm02", "pm10", "pm003Count", "atmp", "rhum",
        "heatindex", "rco2", "tvoc",
    ]
    if len(rows) < 4:
        return None, None, None

    records = []
    for row in rows:
        keys = set(row.keys())
        records.append({key: row[key] if key in keys else None for key in columns})
    frame = pd.DataFrame.from_records(records)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["epoch"]).sort_values("epoch")
    if frame.empty:
        return None, None, None

    latest_epoch = int(frame["epoch"].iloc[-1])
    raw_epochs = frame.pop("epoch")
    raw_index = pd.DatetimeIndex(
        pd.to_datetime(raw_epochs, unit="s", utc=True)
    ).tz_convert("Asia/Kuala_Lumpur")
    frame.index = raw_index
    bucket_last_epoch = pd.Series(
        raw_epochs.to_numpy(), index=raw_index, dtype="float64"
    ).resample(
        f"{DRY_FORMING_BUCKET_MINUTES}min", label="right", closed="right"
    ).max()
    frame = frame.resample(
        f"{DRY_FORMING_BUCKET_MINUTES}min", label="right", closed="right"
    ).median(numeric_only=True)
    # Never interpolate change-point evidence. A later reading must not fill a
    # missing historical bucket during causal replay.
    smooth = frame.rolling(3, min_periods=3).median()

    lookback = DRY_FORMING_LOOKBACK_BUCKETS
    smooth["pm25Change15"] = smooth["pm02"].diff(lookback)
    smooth["pm25Fraction15"] = smooth["pm02"].pct_change(
        lookback, fill_method=None
    )
    smooth["pm10Fraction15"] = smooth["pm10"].pct_change(
        lookback, fill_method=None
    )
    smooth["countFraction15"] = smooth["pm003Count"].pct_change(
        lookback, fill_method=None
    )
    smooth["humidityChange15"] = smooth["rhum"].diff(lookback)
    smooth["temperatureChange15"] = smooth["atmp"].diff(lookback)
    smooth["heatIndexChange15"] = smooth["heatindex"].diff(lookback)
    smooth["tvocFraction15"] = smooth["tvoc"].pct_change(
        lookback, fill_method=None
    )
    smooth["co2Fraction15"] = smooth["rco2"].pct_change(
        lookback, fill_method=None
    )
    smooth["persistentFall"] = (
        smooth["pm02"].diff().lt(0).rolling(3, min_periods=3).sum().ge(2)
    )

    # A nine-minute median at both ends of a 15-minute comparison needs eight
    # contiguous underlying buckets. Particle count is preferred; TVOC+CO2 is
    # a fallback for the older history collected before count was stored.
    rapid_bucket_count = lookback + 3
    base_complete = frame[["pm02", "pm10", "rhum"]].notna().all(axis=1).rolling(
        rapid_bucket_count, min_periods=rapid_bucket_count
    ).sum().eq(rapid_bucket_count)
    count_complete = frame["pm003Count"].notna().rolling(
        rapid_bucket_count, min_periods=rapid_bucket_count
    ).sum().eq(rapid_bucket_count)
    gas_complete = frame[["tvoc", "rco2"]].notna().all(axis=1).rolling(
        rapid_bucket_count, min_periods=rapid_bucket_count
    ).sum().eq(rapid_bucket_count)
    rapid_base = (
        base_complete
        & smooth["pm25Change15"].le(-DRY_FORMING_PM25_DROP)
        & smooth["pm25Fraction15"].le(-DRY_FORMING_PM25_FRACTION)
        & smooth["pm10Fraction15"].le(-DRY_FORMING_PM10_FRACTION)
        & smooth["persistentFall"]
    )
    smooth["rapidParticleSupport"] = (
        count_complete
        & smooth["countFraction15"].le(-DRY_FORMING_COUNT_FRACTION)
        & smooth["humidityChange15"].le(-DRY_FORMING_RH_DROP)
    )
    smooth["rapidGasSupport"] = (
        gas_complete
        & smooth["tvocFraction15"].le(-DRY_FORMING_TVOC_FRACTION)
        & smooth["co2Fraction15"].le(-DRY_FORMING_CO2_FRACTION)
    )
    smooth["rapidForming"] = rapid_base & (
        smooth["rapidParticleSupport"]
        | smooth["rapidGasSupport"]
    )

    # This named common-driver relationship is intentionally separate from
    # `rapidForming`: it may change the event label/recheck cadence, but it
    # cannot enter the rapid-event point-forecast overlay. Stored analogues show
    # frequent rebound after moist cooling, so temperature is not a PM lead.
    moist_lookback = 30 // DRY_FORMING_BUCKET_MINUTES
    moist_complete_count = moist_lookback + 3
    smooth["pm25Fraction30"] = smooth["pm02"].pct_change(
        moist_lookback, fill_method=None
    )
    smooth["pm10Fraction30"] = smooth["pm10"].pct_change(
        moist_lookback, fill_method=None
    )
    smooth["countFraction30"] = smooth["pm003Count"].pct_change(
        moist_lookback, fill_method=None
    )
    smooth["temperatureChange30"] = smooth["atmp"].diff(moist_lookback)
    smooth["heatIndexChange30"] = smooth["heatindex"].diff(moist_lookback)
    smooth["humidityChange30"] = smooth["rhum"].diff(moist_lookback)
    moist_base_complete = frame[
        ["pm02", "pm10", "atmp", "rhum", "heatindex"]
    ].notna().all(axis=1).rolling(
        moist_complete_count, min_periods=moist_complete_count
    ).sum().eq(moist_complete_count)
    moist_count_observed = frame["pm003Count"].notna().rolling(
        moist_complete_count, min_periods=moist_complete_count
    ).sum()
    moist_count_support = (
        (
            moist_count_observed.eq(moist_complete_count)
            & smooth["countFraction30"].le(
                -MOIST_COOLING_COUNT_FRACTION_30
            )
        )
        | moist_count_observed.eq(0)
    )
    local_hour = pd.Series(frame.index.hour, index=frame.index)
    smooth["rapidMoistCoolingSupport"] = (
        moist_base_complete
        & moist_count_support
        & local_hour.ge(6)
        & local_hour.lt(19)
        & smooth["temperatureChange30"].le(-MOIST_COOLING_TEMP_DROP_30)
        & smooth["humidityChange30"].ge(MOIST_COOLING_RH_RISE_30)
        & smooth["pm25Fraction30"].le(-MOIST_COOLING_PM25_FRACTION_30)
        & smooth["pm10Fraction30"].le(-MOIST_COOLING_PM10_FRACTION_30)
    )

    # A faster raw-cadence change watch can recognize a coherent break several
    # minutes sooner. It changes the nowcast/recheck interval only; the point
    # forecast override waits for the stronger smoothed rapid gate above.
    shock = DRY_SHOCK_LOOKBACK_BUCKETS
    shock_count = shock + 1
    shock_base_complete = frame[["pm02", "pm10", "rhum"]].notna().all(axis=1).rolling(
        shock_count, min_periods=shock_count
    ).sum().eq(shock_count)
    shock_count_complete = frame["pm003Count"].notna().rolling(
        shock_count, min_periods=shock_count
    ).sum().eq(shock_count)
    shock_gas_complete = frame[["tvoc", "rco2"]].notna().all(axis=1).rolling(
        shock_count, min_periods=shock_count
    ).sum().eq(shock_count)
    smooth["pm25Change9"] = frame["pm02"].diff(shock)
    smooth["pm25Fraction9"] = frame["pm02"].pct_change(shock, fill_method=None)
    smooth["pm10Fraction9"] = frame["pm10"].pct_change(shock, fill_method=None)
    smooth["countFraction9"] = frame["pm003Count"].pct_change(
        shock, fill_method=None
    )
    smooth["humidityChange9"] = frame["rhum"].diff(shock)
    smooth["tvocFraction9"] = frame["tvoc"].pct_change(shock, fill_method=None)
    smooth["co2Fraction9"] = frame["rco2"].pct_change(shock, fill_method=None)
    smooth["fallingSteps9"] = frame["pm02"].diff().lt(0).rolling(
        shock, min_periods=shock
    ).sum()
    smooth["pm10FallingSteps9"] = frame["pm10"].diff().lt(0).rolling(
        shock, min_periods=shock
    ).sum()
    shock_base = (
        shock_base_complete
        & smooth["pm25Change9"].le(-DRY_SHOCK_PM25_DROP)
        & smooth["pm25Fraction9"].le(-DRY_SHOCK_PM25_FRACTION)
        & smooth["pm10Fraction9"].le(-DRY_SHOCK_PM10_FRACTION)
        & smooth["fallingSteps9"].eq(shock)
        & smooth["pm10FallingSteps9"].ge(shock - 1)
    )
    shock_particle = (
        shock_count_complete
        & smooth["countFraction9"].le(-DRY_SHOCK_COUNT_FRACTION)
        & smooth["humidityChange9"].le(-DRY_SHOCK_RH_DROP)
    )
    shock_gas = (
        shock_gas_complete
        & smooth["tvocFraction9"].le(-DRY_FORMING_TVOC_FRACTION)
        & smooth["co2Fraction9"].le(-DRY_FORMING_CO2_FRACTION)
    )
    smooth["earlyJointShock"] = shock_base & (shock_particle | shock_gas)

    # The original rapid path missed coherent one-hour clearings.  This branch
    # requires a complete 60-minute joint decline, directional persistence and
    # no material rebound from the recent low.  It recognizes a regime already
    # in progress; it does not extrapolate the decline into the ride window.
    lookback45 = 45 // DRY_FORMING_BUCKET_MINUTES
    lookback60 = DRY_SUSTAINED_LOOKBACK_BUCKETS
    smooth["pm25Change45"] = smooth["pm02"].diff(lookback45)
    smooth["pm25Fraction45"] = smooth["pm02"].pct_change(
        lookback45, fill_method=None
    )
    smooth["pm25Change60"] = smooth["pm02"].diff(lookback60)
    smooth["pm25Fraction60"] = smooth["pm02"].pct_change(
        lookback60, fill_method=None
    )
    smooth["pm10Fraction45"] = smooth["pm10"].pct_change(
        lookback45, fill_method=None
    )
    smooth["countFraction45"] = smooth["pm003Count"].pct_change(
        lookback45, fill_method=None
    )
    smooth["humidityChange45"] = smooth["rhum"].diff(lookback45)
    falling_steps = smooth["pm02"].diff().lt(0)
    smooth["fallingSteps45"] = falling_steps.rolling(
        lookback45, min_periods=lookback45
    ).sum()
    smooth["fallingSteps15"] = falling_steps.rolling(
        lookback, min_periods=lookback
    ).sum()
    smooth["recentLow12"] = smooth["pm02"].rolling(
        1 + 12 // DRY_FORMING_BUCKET_MINUTES,
        min_periods=1 + 12 // DRY_FORMING_BUCKET_MINUTES,
    ).min()
    # The trailing median extends two buckets before the 60-minute feature
    # window, hence 23 complete raw buckets are required.
    sustained_complete_count = lookback60 + 3
    sustained_complete = frame[["pm02", "pm10", "pm003Count", "rhum"]].notna().all(
        axis=1
    ).rolling(
        sustained_complete_count, min_periods=sustained_complete_count
    ).sum().eq(sustained_complete_count)
    smooth["sustainedForming"] = (
        sustained_complete
        & smooth["pm25Change45"].le(-DRY_SUSTAINED_PM25_DROP)
        & smooth["pm25Fraction45"].le(-DRY_SUSTAINED_PM25_FRACTION)
        & smooth["pm25Change60"].le(-DRY_SUSTAINED_PM25_DROP)
        & smooth["pm25Fraction60"].le(-DRY_SUSTAINED_PM25_FRACTION)
        & smooth["pm10Fraction45"].le(-DRY_SUSTAINED_PM10_FRACTION)
        & smooth["countFraction45"].le(-DRY_SUSTAINED_COUNT_FRACTION)
        & smooth["humidityChange45"].le(-DRY_SUSTAINED_RH_DROP)
        & smooth["fallingSteps45"].ge(DRY_SUSTAINED_FALLS_45)
        & smooth["fallingSteps15"].ge(DRY_SUSTAINED_FALLS_15)
        & (smooth["pm02"] - smooth["recentLow12"]).le(
            DRY_SUSTAINED_MAX_FROM_RECENT_LOW
        )
    )
    smooth["forming"] = (
        smooth["earlyJointShock"]
        | smooth["rapidForming"]
        | smooth["sustainedForming"]
        | smooth["rapidMoistCoolingSupport"]
    )
    return smooth, bucket_last_epoch, latest_epoch


def fast_dry_clearing_signal(rows):
    """Detect early, rapid or sustained joint clearing using causal data only."""
    smooth, bucket_last_epoch, latest_epoch = dry_clearing_feature_frame(rows)
    if smooth is None:
        return None

    # Cluster the full retained candidate series.  Truncating it to a rolling
    # two-hour slice would eventually move an ongoing event's onset forward and
    # break its API identity even while evidence remained continuous.
    candidates = smooth.loc[smooth["forming"]]
    if candidates.empty:
        return None

    # Qualifying evidence can pause briefly during one coherent decline.  Keep
    # its onset latched for an hour so a later qualifying point does not
    # manufacture a second event; the visible state still expires after the
    # shorter hold interval when there is no fresh evidence.
    cluster_ids = candidates.index.to_series().diff().gt(
        pd.Timedelta(minutes=DRY_FORMING_REARM_MINUTES)
    ).cumsum()
    cluster = candidates.loc[cluster_ids.eq(cluster_ids.iloc[-1])]
    first_time = cluster.index[0]
    last_time = cluster.index[-1]
    first_raw_epoch = num(bucket_last_epoch.get(first_time))
    last_raw_epoch = num(bucket_last_epoch.get(last_time))
    detected_epoch = min(
        latest_epoch,
        int(first_raw_epoch) if first_raw_epoch is not None
        else int(first_time.timestamp()),
    )
    last_evidence_epoch = min(
        latest_epoch,
        int(last_raw_epoch) if last_raw_epoch is not None
        else int(last_time.timestamp()),
    )
    if latest_epoch - last_evidence_epoch > DRY_FORMING_HOLD_MINUTES * 60:
        return None

    # Stop the preliminary signal if particles have already materially rebounded.
    after_detection = smooth.loc[first_time:, "pm02"].dropna()
    if not after_detection.empty:
        minimum = float(after_detection.min())
        current = float(after_detection.iloc[-1])
        rebound_amount = max(5.0, minimum * 0.08)
        if current >= minimum + rebound_amount:
            return None

    has_shock = bool(cluster["earlyJointShock"].any())
    has_rapid = bool(cluster["rapidForming"].any())
    has_sustained = bool(cluster["sustainedForming"].any())
    has_moist_cooling = bool(cluster["rapidMoistCoolingSupport"].any())
    if has_moist_cooling:
        moist_points = cluster.loc[cluster["rapidMoistCoolingSupport"]]
        point = moist_points.loc[moist_points["pm25Fraction30"].idxmin()]
    elif has_sustained:
        sustained_points = cluster.loc[cluster["sustainedForming"]]
        point = sustained_points.loc[sustained_points["pm25Change60"].idxmin()]
    elif has_rapid:
        rapid_points = cluster.loc[cluster["rapidForming"]]
        point = rapid_points.loc[rapid_points["pm25Change15"].idxmin()]
    else:
        shock_points = cluster.loc[cluster["earlyJointShock"]]
        point = shock_points.loc[shock_points["pm25Change9"].idxmin()]
    rapid_time = (
        cluster.loc[cluster["rapidForming"]].index[0] if has_rapid else None
    )
    rapid_raw_epoch = (
        num(bucket_last_epoch.get(rapid_time)) if rapid_time is not None else None
    )
    detection_mode = (
        "moist_cooling_joint" if has_moist_cooling
        else "rapid_and_sustained" if has_rapid and has_sustained
        else "sustained_joint" if has_sustained
        else "rapid_joint" if has_rapid
        else "joint_shock"
    )

    def decline_metric(column, percent=False):
        value = num(point.get(column))
        if value is None or not math.isfinite(value):
            return None
        magnitude = max(0.0, -(100.0 if percent else 1.0) * value)
        return round(magnitude) if percent else round(magnitude, 1)

    def signed_metric(column, digits=0):
        value = num(point.get(column))
        if value is None or not math.isfinite(value):
            return None
        return round(value, digits)

    return {
        "active": True,
        "state": "forming",
        "kind": (
            "moist_cooling_particle_clearing_forming"
            if has_moist_cooling else "dry_clearing_forming"
        ),
        "label": (
            "Moist-cooling particle clearing forming"
            if has_moist_cooling else "Dry clearing forming"
        ),
        "details": "",
        "detectedEpoch": detected_epoch,
        "lastEvidenceEpoch": last_evidence_epoch,
        "ageMinutes": max(0, round((latest_epoch - detected_epoch) / 60)),
        "rainSupport": False,
        "dryAirMassSupport": False,
        "metrics": {
            "detectionMode": detection_mode,
            "earlyJointShock": has_shock,
            "rapidDetectedEpoch": (
                None if rapid_raw_epoch is None
                else min(latest_epoch, int(rapid_raw_epoch))
            ),
            "rapidParticleSupport": bool(point.get("rapidParticleSupport", False)),
            "rapidGasSupport": bool(point.get("rapidGasSupport", False)),
            "rapidMoistCoolingSupport": has_moist_cooling,
            "pm25Drop9": decline_metric("pm25Change9"),
            "pm25DropPct9": decline_metric("pm25Fraction9", percent=True),
            "pm10DropPct9": decline_metric("pm10Fraction9", percent=True),
            "particleCountDropPct9": decline_metric(
                "countFraction9", percent=True
            ),
            "humidityChange9": signed_metric("humidityChange9"),
            "pm25Drop15": decline_metric("pm25Change15"),
            "pm25DropPct15": decline_metric("pm25Fraction15", percent=True),
            "pm10DropPct15": decline_metric("pm10Fraction15", percent=True),
            "particleCountDropPct15": decline_metric("countFraction15", percent=True),
            "humidityChange15": signed_metric("humidityChange15"),
            "temperatureDrop30": decline_metric("temperatureChange30"),
            "heatIndexDrop30": decline_metric("heatIndexChange30"),
            "humidityRise30": (
                max(0, signed_metric("humidityChange30") or 0)
                if has_moist_cooling else None
            ),
            "tvocDropPct15": decline_metric("tvocFraction15", percent=True),
            "co2DropPct15": decline_metric("co2Fraction15", percent=True),
            "pm25Drop45": decline_metric("pm25Change45"),
            "pm25DropPct45": decline_metric("pm25Fraction45", percent=True),
            "pm25Drop60": decline_metric("pm25Change60"),
            "pm25DropPct60": decline_metric("pm25Fraction60", percent=True),
            "pm10DropPct45": decline_metric("pm10Fraction45", percent=True),
            "particleCountDropPct45": decline_metric(
                "countFraction45", percent=True
            ),
            "humidityChange45": signed_metric("humidityChange45"),
            "fallingSteps45": signed_metric("fallingSteps45"),
            "fallingSteps15": signed_metric("fallingSteps15"),
            "persistentFalls": True,
        },
        "timeline": {
            "onsetEpoch": detected_epoch,
            "confirmationEpoch": None,
            "lowEpoch": None,
            "lowPm25": None,
            "reboundEpoch": None,
        },
    }


def rapid_clearance_event_outlook(rows, clearance_signal=None):
    """Forecast a just-detected strong clearing from completed prior events."""
    clearance_signal = clearance_signal or {}
    if clearance_signal.get("state") not in ("forming", "active", "recent"):
        return {
            "available": False,
            "status": "no_current_clearance_event",
            "minimumCompletedEvents": RAPID_EVENT_MIN_COMPLETED,
            "minimumDistinctDays": RAPID_EVENT_MIN_DISTINCT_DAYS,
        }
    live_onset_epoch = num(
        (clearance_signal.get("timeline") or {}).get("onsetEpoch")
        or clearance_signal.get("detectedEpoch")
    )
    smooth, bucket_last_epoch, latest_epoch = dry_clearing_feature_frame(rows)
    if smooth is None:
        return {"available": False, "status": "collecting"}
    candidates = smooth.loc[smooth["rapidForming"]]
    if candidates.empty:
        return {"available": False, "status": "no_strong_event"}

    cluster_ids = candidates.index.to_series().diff().gt(
        pd.Timedelta(minutes=DRY_FORMING_REARM_MINUTES)
    ).cumsum()
    representatives = candidates.groupby(cluster_ids, sort=True).head(1)
    current_label = representatives.index[-1]
    current_raw_epoch = num(bucket_last_epoch.get(current_label))
    if current_raw_epoch is None:
        return {"available": False, "status": "missing_event_epoch"}
    current_raw_epoch = min(latest_epoch, int(current_raw_epoch))
    if (
        live_onset_epoch is None
        or current_raw_epoch < int(live_onset_epoch)
        or current_raw_epoch - int(live_onset_epoch) > DRY_FORMING_REARM_MINUTES * 60
    ):
        return {
            "available": False,
            "status": "not_triggered_for_current_event",
            "minimumCompletedEvents": RAPID_EVENT_MIN_COMPLETED,
            "minimumDistinctDays": RAPID_EVENT_MIN_DISTINCT_DAYS,
        }
    age_minutes = max(0.0, (latest_epoch - current_raw_epoch) / 60.0)

    raw_records = []
    for row in rows:
        keys = set(row.keys())
        epoch = num(row["epoch"] if "epoch" in keys else None)
        pm25 = num(row["pm02"] if "pm02" in keys else None)
        if epoch is not None and pm25 is not None:
            raw_records.append({
                "epoch": int(epoch),
                "pm02": pm25,
                "pm10": num(row["pm10"] if "pm10" in keys else None),
                "pm003Count": num(
                    row["pm003Count"] if "pm003Count" in keys else None
                ),
            })
    raw = pd.DataFrame.from_records(
        raw_records, columns=["epoch", "pm02", "pm10", "pm003Count"]
    ).sort_values("epoch")
    if raw.empty:
        return {"available": False, "status": "missing_raw_anchor"}

    completed = []
    for prior_label in representatives.index[:-1]:
        prior_raw_epoch = num(bucket_last_epoch.get(prior_label))
        if prior_raw_epoch is None:
            continue
        prior_raw_epoch = int(prior_raw_epoch)
        # Strict causal embargo: the full ride outcome had to be known before
        # the current event's first strong evidence arrived.
        if prior_raw_epoch + (ARRIVAL_MINUTES + TRAIL_MINUTES) * 60 > current_raw_epoch:
            continue
        origin_rows = raw.loc[raw["epoch"].le(prior_raw_epoch)]
        if origin_rows.empty:
            continue
        origin = float(origin_rows.iloc[-1]["pm02"])
        arrival_epoch = prior_raw_epoch + ARRIVAL_MINUTES * 60
        eligible_arrival_labels = bucket_last_epoch.loc[
            bucket_last_epoch.le(arrival_epoch)
            & bucket_last_epoch.notna()
        ].index
        arrival_label = (
            eligible_arrival_labels[-1] if len(eligible_arrival_labels) else None
        )
        arrival_value = (
            None if arrival_label is None else num(smooth["pm02"].get(arrival_label))
        )
        trail_values = raw.loc[
            raw["epoch"].between(
                prior_raw_epoch + ARRIVAL_MINUTES * 60,
                prior_raw_epoch + (ARRIVAL_MINUTES + TRAIL_MINUTES) * 60,
            ),
            "pm02",
        ].dropna()
        if arrival_value is None or len(trail_values) < 32:
            continue
        point = representatives.loc[prior_label]
        if origin <= 0:
            continue
        completed.append({
            "epoch": prior_raw_epoch,
            "origin": origin,
            "arrivalDelta": float(arrival_value) - origin,
            "trailMeanDelta": float(trail_values.mean()) - origin,
            "trailPeakDelta": float(trail_values.max()) - origin,
            "arrivalRatio": float(arrival_value) / origin,
            "trailMeanRatio": float(trail_values.mean()) / origin,
            "trailPeakRatio": float(trail_values.max()) / origin,
            "particleSupport": bool(point.get("rapidParticleSupport", False)),
            "gasSupport": bool(point.get("rapidGasSupport", False)),
        })

    independent_completed = []
    completed_days = set()
    for event in completed:
        event_day = local_dt(event["epoch"]).date()
        if event_day in completed_days:
            continue
        completed_days.add(event_day)
        independent_completed.append(event)
    support_count = len(independent_completed)
    distinct_day_count = len(completed_days)
    base = {
        "available": False,
        "status": (
            "ready" if support_count >= RAPID_EVENT_MIN_COMPLETED
            else "insufficient_completed_events"
        ),
        "eventEpoch": current_raw_epoch,
        "eventAgeMinutes": round(age_minutes, 1),
        "completedEventCount": support_count,
        "completedClusterCount": len(completed),
        "completedDistinctDays": distinct_day_count,
        "minimumCompletedEvents": RAPID_EVENT_MIN_COMPLETED,
        "minimumDistinctDays": RAPID_EVENT_MIN_DISTINCT_DAYS,
        "priorEventEpochs": [event["epoch"] for event in independent_completed],
        "experimental": True,
        "validated": False,
    }
    if (
        support_count < RAPID_EVENT_MIN_COMPLETED
        or distinct_day_count < RAPID_EVENT_MIN_DISTINCT_DAYS
    ):
        return base
    if age_minutes > RAPID_EVENT_OVERLAY_MINUTES:
        base["status"] = "expired"
        return base

    trigger_rows = raw.loc[raw["epoch"].le(current_raw_epoch)]
    post_trigger = raw.loc[raw["epoch"].gt(current_raw_epoch)]
    if trigger_rows.empty:
        base["status"] = "missing_event_anchor"
        return base
    trigger = trigger_rows.iloc[-1]

    def finite_value(value):
        parsed = num(value)
        return parsed if parsed is not None and math.isfinite(parsed) else None

    trigger_pm25 = finite_value(trigger.get("pm02"))
    trigger_pm10 = finite_value(trigger.get("pm10"))
    trigger_count = finite_value(trigger.get("pm003Count"))
    prior_trigger_levels = [
        float(event["origin"]) for event in independent_completed
        if num(event.get("origin")) is not None
    ]
    prior_trigger_low = min(prior_trigger_levels) if prior_trigger_levels else None
    prior_trigger_high = max(prior_trigger_levels) if prior_trigger_levels else None
    trigger_in_support = bool(
        trigger_pm25 is not None
        and prior_trigger_low is not None
        and prior_trigger_high is not None
        and prior_trigger_low <= trigger_pm25 <= prior_trigger_high
    )
    base["triggerLevelSupport"] = {
        "required": True,
        "currentTriggerPm25": (
            None if trigger_pm25 is None else round(trigger_pm25, 1)
        ),
        "priorMinimumPm25": (
            None if prior_trigger_low is None else round(prior_trigger_low, 1)
        ),
        "priorMaximumPm25": (
            None if prior_trigger_high is None else round(prior_trigger_high, 1)
        ),
        "insidePriorRange": trigger_in_support,
    }
    if not trigger_in_support:
        base["status"] = "trigger_level_outside_historical_support"
        return base
    base["postTriggerConfirmation"] = {
        "required": True,
        "minimumAgeMinutes": RAPID_EVENT_CONFIRM_MINUTES,
        "minimumSampleCount": RAPID_EVENT_CONFIRM_SAMPLES,
        "triggerPm25": (
            None if trigger_pm25 is None else round(trigger_pm25, 1)
        ),
        "triggerPm10": (
            None if trigger_pm10 is None else round(trigger_pm10, 1)
        ),
        "triggerParticleCount": (
            None if trigger_count is None else round(trigger_count)
        ),
        "postTriggerSampleCount": int(len(post_trigger)),
    }
    if (age_minutes < RAPID_EVENT_CONFIRM_MINUTES
            or len(post_trigger) < RAPID_EVENT_CONFIRM_SAMPLES):
        base["status"] = "awaiting_post_trigger_confirmation"
        base["postTriggerConfirmation"]["state"] = "awaiting"
        return base

    pm25_tolerance = max(1.0, 0.03 * float(trigger_pm25 or 0.0))
    pm10_tolerance = max(1.0, 0.03 * float(trigger_pm10 or 0.0))
    count_tolerance = max(150.0, 0.05 * float(trigger_count or 0.0))
    trigger_reversal = bool(
        (trigger_pm25 is not None
         and post_trigger["pm02"].gt(trigger_pm25 + pm25_tolerance).any())
        or (trigger_pm10 is not None
            and post_trigger["pm10"].gt(trigger_pm10 + pm10_tolerance).any())
        or (trigger_count is not None
            and post_trigger["pm003Count"].gt(
                trigger_count + count_tolerance
            ).any())
    )
    event_pm25 = raw.loc[raw["epoch"].ge(current_raw_epoch), "pm02"].dropna()
    two_reading_rebound = False
    single_step_rebound = False
    if len(event_pm25) >= 3:
        event_low = float(event_pm25.min())
        rebound_level = event_low + max(3.0, event_low * 0.08)
        two_reading_rebound = bool(event_pm25.iloc[-2:].ge(rebound_level).all())
        last_step_threshold = max(2.0, 0.05 * event_low)
        single_step_rebound = bool(
            event_pm25.iloc[-1] >= rebound_level
            and event_pm25.iloc[-1] - event_pm25.iloc[-2]
            >= last_step_threshold
        )
        base["postTriggerConfirmation"].update({
            "eventLowPm25": round(event_low, 1),
            "reversalLevelPm25": round(rebound_level, 1),
            "singleStepReboundThreshold": round(last_step_threshold, 1),
        })
    timeline_rebound = bool(
        (clearance_signal.get("timeline") or {}).get("reboundEpoch")
    )
    if (trigger_reversal or single_step_rebound
            or two_reading_rebound or timeline_rebound):
        base["status"] = "post_trigger_reversal"
        base["postTriggerConfirmation"].update({
            "state": "reversed",
            "triggerLevelExceeded": trigger_reversal,
            "singleStepRebound": single_step_rebound,
            "twoReadingRebound": two_reading_rebound,
            "timelineRebound": timeline_rebound,
        })
        return base

    base["postTriggerConfirmation"]["state"] = "confirmed"
    if age_minutes < RAPID_EVENT_FULL_WEIGHT_END_MINUTES:
        weight = 1.0
    elif age_minutes < RAPID_EVENT_PARTIAL_WEIGHT_END_MINUTES:
        weight = RAPID_EVENT_PARTIAL_WEIGHT
    else:
        weight = 0.0
    if weight <= 0:
        base["status"] = "event_overlay_complete"
        return base

    current_anchor = float(raw.iloc[-1]["pm02"])
    arrival_ratios = np.asarray(
        [event["arrivalRatio"] for event in independent_completed], dtype="float64"
    )
    mean_ratios = np.asarray(
        [event["trailMeanRatio"] for event in independent_completed], dtype="float64"
    )
    peak_ratios = np.asarray(
        [event["trailPeakRatio"] for event in independent_completed], dtype="float64"
    )

    def projected(ratio):
        # A prior outcome/origin ratio describes the completed event relative
        # to its trigger level. Apply it to this event's trigger, not to the
        # already-lower post-confirmation reading; multiplying the latter by
        # the ratio would count the observed first leg of the clearing twice.
        # As the evidence ages, blend that trigger-normalized target back
        # toward the latest reading (the persistence reference).
        event_target = max(0.0, float(trigger_pm25) * float(ratio))
        return max(0.0, current_anchor + weight * (event_target - current_anchor))

    arrival_scenarios = [projected(ratio) for ratio in arrival_ratios]
    mean_scenarios = [projected(ratio) for ratio in mean_ratios]
    peak_scenarios = [projected(ratio) for ratio in peak_ratios]
    arrival_ratio = float(np.median(arrival_ratios))
    mean_ratio = float(np.median(mean_ratios))
    peak_ratio = float(np.median(peak_ratios))
    arrival_point = projected(arrival_ratio)
    mean_point = projected(mean_ratio)
    peak_point = max(mean_point, projected(peak_ratio))
    base.update({
        "available": True,
        "status": "active",
        "method": "Scale-normalized rapid-clearance event analogue with historical error bounds",
        "currentRawAnchor": round(current_anchor, 1),
        "triggerRawAnchor": round(float(trigger_pm25), 1),
        "deltaWeight": weight,
        "normalization": "prior_outcome_ratio_applied_to_current_event_trigger",
        "arrivalRatio": round(arrival_ratio, 3),
        "trailMeanRatio": round(mean_ratio, 3),
        "trailPeakRatio": round(peak_ratio, 3),
        "arrivalDelta": round(arrival_point - current_anchor, 1),
        "trailMeanDelta": round(mean_point - current_anchor, 1),
        "trailPeakDelta": round(peak_point - current_anchor, 1),
        "arrivalPoint": round(arrival_point, 1),
        "trailMeanPoint": round(mean_point, 1),
        "trailPeakPoint": round(peak_point, 1),
        "arrivalScenarioLow": round(min(arrival_scenarios), 1),
        "arrivalScenarioHigh": round(max(arrival_scenarios), 1),
        "trailMeanScenarioLow": round(min(mean_scenarios), 1),
        "trailMeanScenarioHigh": round(max(mean_scenarios), 1),
        "trailPeakScenarioHigh": round(max(peak_scenarios), 1),
    })
    return json_safe(base)


def passing_shower_signal(rows):
    """Classify rapid joint-particle clearing with post-confirmation weather evidence."""
    forming_signal = fast_dry_clearing_signal(rows)
    raw_epoch_values = pd.Series(
        [int(row["epoch"]) for row in rows], dtype="float64"
    ) if rows else pd.Series(dtype="float64")
    raw_epoch_index = pd.DatetimeIndex(
        pd.to_datetime(raw_epoch_values, unit="s", utc=True)
    ).tz_convert("Asia/Kuala_Lumpur") if not raw_epoch_values.empty else pd.DatetimeIndex([])
    bucket_last_epoch = pd.Series(
        raw_epoch_values.to_numpy(), index=raw_epoch_index, dtype="float64"
    ).resample(
        f"{AIR_BUCKET_MINUTES}min", label="right", closed="right"
    ).max() if not raw_epoch_values.empty else pd.Series(dtype="float64")
    frame = sensor_frame(rows)
    complete_particles = frame.dropna(subset=["pm02", "pm10"])
    if len(complete_particles) < 3:
        if forming_signal:
            return forming_signal
        return {
            "active": False, "state": "none", "kind": "collecting",
            "label": "Collecting particle-change evidence…",
            "details": "About 30–45 minutes of PM2.5 and PM10 readings are needed.",
            "rainSupport": False,
        }

    # Keep empty 15-minute buckets in place.  Dropping them before positional
    # differences would turn a long collection gap into an apparent 30-minute
    # particle washout.
    paired = frame.copy()
    valid_particles = paired["pm02"].notna() & paired["pm10"].notna()

    paired["pm25Change30"] = paired["pm02"].diff(2)
    paired["pm25Fraction30"] = paired["pm02"].pct_change(2, fill_method=None)
    paired["pm10Change30"] = paired["pm10"].diff(2)
    paired["pm10Fraction30"] = paired["pm10"].pct_change(2, fill_method=None)
    paired["heatDrop30"] = paired["heatindex"].shift(2) - paired["heatindex"]
    paired["tempDrop30"] = paired["atmp"].shift(2) - paired["atmp"]
    paired["humidityRise30"] = paired["rhum"] - paired["rhum"].shift(2)
    paired["coarseParticles"] = (paired["pm10"] - paired["pm02"]).clip(lower=0)
    paired["fineShare"] = (100 * paired["pm02"] / paired["pm10"]).clip(lower=0, upper=100)
    paired["coarseSmooth"] = segmented_ewm(
        paired["coarseParticles"], 3
    ).where(valid_particles)
    paired["fineShareSmooth"] = segmented_ewm(
        paired["fineShare"], 3
    ).where(valid_particles)
    paired["coarseChange30"] = paired["coarseSmooth"].diff(2)
    paired["fineShareChange30"] = paired["fineShareSmooth"].diff(2)
    paired["mixWashoutSupport"] = (
        paired["fineShareChange30"].le(-MIX_FINE_SHARE_CHANGE)
        & paired["coarseChange30"].ge(MIX_COARSE_CHANGE)
    )

    paired["jointWashout"] = (
        paired["pm25Change30"].le(-WASHOUT_PM25_DROP)
        & paired["pm25Fraction30"].le(-WASHOUT_PM25_FRACTION)
        & paired["pm10Fraction30"].le(-WASHOUT_PM10_FRACTION)
    )
    daytime = pd.Series(
        (paired.index.hour >= 6) & (paired.index.hour < 19), index=paired.index
    )
    paired["coolingSupport"] = (
        daytime
        & paired["heatDrop30"].ge(3.0)
        & paired["tempDrop30"].ge(0.8)
        & paired["humidityRise30"].ge(3.0)
        & paired["pm25Change30"].le(-5.0)
        & paired["pm10Change30"].le(-5.0)
    )
    paired["rainSupport"] = paired["jointWashout"] & paired["coolingSupport"]

    latest_epoch = int(rows[-1]["epoch"])
    latest_local = pd.Timestamp(latest_epoch, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    events = paired.loc[
        (paired.index >= latest_local - pd.Timedelta(hours=3))
        & paired["jointWashout"]
    ]

    def metrics(point):
        pm25_drop = (
            None if pd.isna(point["pm25Change30"])
            else max(0.0, -float(point["pm25Change30"]))
        )
        pm25_pct = (
            None if pd.isna(point["pm25Fraction30"])
            else max(0.0, -100 * float(point["pm25Fraction30"]))
        )
        pm10_pct = (
            None if pd.isna(point["pm10Fraction30"])
            else max(0.0, -100 * float(point["pm10Fraction30"]))
        )
        return {
            "pm25Drop30": None if pm25_drop is None else round(pm25_drop, 1),
            "pm25DropPct30": None if pm25_pct is None else round(pm25_pct),
            "pm10DropPct30": None if pm10_pct is None else round(pm10_pct),
            "heatDrop30": None if pd.isna(point["heatDrop30"]) else round(float(point["heatDrop30"]), 1),
            "tempDrop30": None if pd.isna(point["tempDrop30"]) else round(float(point["tempDrop30"]), 1),
            "humidityRise30": None if pd.isna(point["humidityRise30"]) else round(float(point["humidityRise30"])),
            "fineShare": None if pd.isna(point["fineShareSmooth"]) else round(float(point["fineShareSmooth"])),
            "fineShareChange30": None if pd.isna(point["fineShareChange30"]) else round(float(point["fineShareChange30"]), 1),
            "coarseParticles": None if pd.isna(point["coarseSmooth"]) else round(float(point["coarseSmooth"]), 1),
            "coarseChange30": None if pd.isna(point["coarseChange30"]) else round(float(point["coarseChange30"]), 1),
            "particleMixSupport": bool(point["mixWashoutSupport"]),
        }

    if not events.empty:
        gaps = events.index.to_series().diff().gt(pd.Timedelta(minutes=60))
        cluster_ids = gaps.cumsum()
        latest_cluster = events.loc[cluster_ids.eq(cluster_ids.iloc[-1])]
        confirmed_last_raw = num(bucket_last_epoch.get(latest_cluster.index[-1]))
        confirmed_last_epoch = min(
            latest_epoch,
            int(confirmed_last_raw) if confirmed_last_raw is not None
            else int(latest_cluster.index[-1].timestamp()),
        )
        if (forming_signal and
                forming_signal["detectedEpoch"] > confirmed_last_epoch + 60 * 60):
            return forming_signal
        confirmation_time = latest_cluster.index[0]
        point = latest_cluster.loc[latest_cluster["pm25Change30"].idxmin()]
        confirmation_raw = num(bucket_last_epoch.get(confirmation_time))
        # A right-labelled live bucket keeps accepting samples. Recover the
        # first raw sample that actually made this bucket qualify so a later
        # refresh cannot move the confirmation time forward.
        bucket_start_epoch = int(
            (confirmation_time - pd.Timedelta(minutes=AIR_BUCKET_MINUTES)).timestamp()
        )
        bucket_end_epoch = int(confirmation_time.timestamp())
        candidate_epochs = [
            int(row["epoch"]) for row in rows
            if bucket_start_epoch < int(row["epoch"]) <= bucket_end_epoch
        ]
        for candidate_epoch in candidate_epochs:
            candidate_frame = sensor_frame([
                row for row in rows if int(row["epoch"]) <= candidate_epoch
            ])
            if len(candidate_frame) < 3 or confirmation_time not in candidate_frame.index:
                continue
            candidate_pm25 = candidate_frame["pm02"]
            candidate_pm10 = candidate_frame["pm10"]
            candidate_drop = num(candidate_pm25.diff(2).get(confirmation_time))
            candidate_pm25_fraction = num(
                candidate_pm25.pct_change(2, fill_method=None).get(confirmation_time)
            )
            candidate_pm10_fraction = num(
                candidate_pm10.pct_change(2, fill_method=None).get(confirmation_time)
            )
            if (
                candidate_drop is not None
                and candidate_pm25_fraction is not None
                and candidate_pm10_fraction is not None
                and candidate_drop <= -WASHOUT_PM25_DROP
                and candidate_pm25_fraction <= -WASHOUT_PM25_FRACTION
                and candidate_pm10_fraction <= -WASHOUT_PM10_FRACTION
            ):
                confirmation_raw = candidate_epoch
                break
        detected_epoch = min(
            latest_epoch,
            int(confirmation_raw) if confirmation_raw is not None
            else int(confirmation_time.timestamp()),
        )
        age = max(0, round((latest_epoch - detected_epoch) / 60))
        rain_support = bool(point["rainSupport"])
        event_metrics = metrics(point)
        forming_metrics = (forming_signal or {}).get("metrics") or {}
        for key in (
            "detectionMode", "earlyJointShock", "rapidDetectedEpoch",
            "rapidParticleSupport", "rapidGasSupport",
            "rapidMoistCoolingSupport", "temperatureDrop30",
            "heatIndexDrop30", "humidityRise30",
        ):
            if key in forming_metrics:
                event_metrics[key] = forming_metrics.get(key)
        timeline = clearance_timeline(
            rows,
            detected_epoch,
            (forming_signal or {}).get("detectedEpoch"),
        )
        event_start_epoch = timeline.get("onsetEpoch", detected_epoch)
        station = subang_event_context(event_start_epoch, latest_epoch)
        issued = issued_weather_point(detected_epoch)

        station_weather = str(station.get("presentWeather") or "").lower()
        station_dry = not any(
            word in station_weather for word in ("rain", "shower", "drizzle")
        )
        sensor_drying = (
            event_metrics.get("humidityRise30") is not None
            and event_metrics["humidityRise30"] <= -2
        )
        station_ventilation = (
            station.get("windSpeedKmh") is not None
            and station["windSpeedKmh"] >= DRY_CLEARING_WIND_SPEED
            and station.get("windSpeedChange") is not None
            and station["windSpeedChange"] >= DRY_CLEARING_WIND_JUMP
        )
        station_drying = (
            station.get("humidityChange") is not None
            and station["humidityChange"] <= -5
        ) or (
            station.get("dewpointChange") is not None
            and station["dewpointChange"] <= -1.5
        )
        forecast_dry = (
            issued.get("rainProbability") is not None
            and issued["rainProbability"] <= DRY_CLEARING_MAX_RAIN_PROBABILITY
            and issued.get("precipitationMm") is not None
            and issued["precipitationMm"] <= DRY_CLEARING_MAX_PRECIPITATION
        )
        dry_air_mass_clearing = bool(
            not rain_support and sensor_drying and station_dry
            and station_ventilation and station_drying and forecast_dry
        )
        prefix = "Detected now" if age <= 2 else f"Detected ~{age} min ago"
        details = (
            f'{prefix} · PM2.5 ↓{event_metrics["pm25Drop30"]:.1f} '
            f'({event_metrics["pm25DropPct30"]:.0f}%) in 30 min · '
            f'PM10 ↓{event_metrics["pm10DropPct30"]:.0f}%'
        )
        if rain_support:
            details += (
                f' · heat index ↓{event_metrics["heatDrop30"]:.1f} °C'
                f' · humidity ↑{event_metrics["humidityRise30"]:.0f}%'
            )
        elif dry_air_mass_clearing:
            details += (
                f' · Subang wind {station["windSpeedKmh"]:.1f} km/h '
                f'({station["windSpeedChange"]:+.1f}) · dry forecast'
            )
        if rain_support:
            kind = "rain_supported_washout"
            label = "Possible rain-linked particle washout"
        elif dry_air_mass_clearing:
            kind = "dry_air_mass_clearing"
            label = "Dry-air ventilation clearing confirmed"
        else:
            kind = "joint_clearance"
            label = "Rapid particle clearance or air-mass change"
        return {
            "active": age <= 30,
            "state": "active" if age <= 30 else "recent",
            "kind": kind,
            "label": label,
            "details": details,
            "detectedEpoch": detected_epoch,
            "lastEvidenceEpoch": max(
                confirmed_last_epoch,
                int((forming_signal or {}).get("lastEvidenceEpoch") or 0),
            ),
            "ageMinutes": age,
            "rainSupport": rain_support,
            "dryAirMassSupport": dry_air_mass_clearing,
            "metrics": event_metrics,
            "timeline": timeline,
            "subangEvidence": station,
            "issuedWeatherEvidence": issued,
        }

    if forming_signal:
        return forming_signal

    latest = paired.iloc[-1]
    current_metrics = metrics(latest)
    if pd.isna(latest["pm25Change30"]) or pd.isna(latest["pm10Fraction30"]):
        details = "A complete 30-minute comparison is still collecting."
    else:
        pm25_arrow = "↓" if latest["pm25Change30"] < 0 else "↑" if latest["pm25Change30"] > 0 else "→"
        pm10_pct = 100 * float(latest["pm10Fraction30"])
        pm10_arrow = "↓" if pm10_pct < 0 else "↑" if pm10_pct > 0 else "→"
        details = (
            f'Last 30 min · PM2.5 {pm25_arrow}{abs(float(latest["pm25Change30"])):.1f} µg/m³'
            f' · PM10 {pm10_arrow}{abs(pm10_pct):.0f}%'
        )
    return {
        "active": False, "state": "none", "kind": "none",
        "label": "No rapid particle washout detected", "details": details,
        "rainSupport": False, "metrics": current_metrics,
    }


def empirical_quantile(values, q, upper=False):
    """Finite-sample empirical quantile, conservative at the requested edge."""
    clean = pd.Series(values, dtype="float64").dropna()
    if clean.empty:
        return None
    if upper:
        ordered = clean.sort_values().tolist()
        rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * q)))
        return float(ordered[rank - 1])
    return float(clean.quantile(q, interpolation="lower"))


def finite_upper_rank_coverage(sample_count, q):
    """Exchangeable-sample coverage implied by the chosen upper order statistic."""
    if sample_count <= 0:
        return None
    rank = min(sample_count, max(1, math.ceil((sample_count + 1) * q)))
    return rank / (sample_count + 1)


def independent_origin_count(index, separation_minutes):
    """Count chronologically non-overlapping forecast origins."""
    separation = pd.Timedelta(minutes=separation_minutes)
    latest_selected = None
    count = 0
    for timestamp in sorted(pd.DatetimeIndex(index)):
        if latest_selected is None or timestamp - latest_selected >= separation:
            latest_selected = timestamp
            count += 1
    return count


def time_separated_candidates(frame, max_count, separation_minutes):
    """Greedily retain the closest candidates without adjacent-origin inflation."""
    if frame.empty:
        return frame.copy()
    separation = int(separation_minutes * 60 * 1_000_000_000)
    selected = []
    selected_ns = []
    for timestamp in frame.sort_values("distance").index:
        if all(abs(timestamp.value - prior) >= separation for prior in selected_ns):
            selected.append(timestamp)
            selected_ns.append(timestamp.value)
            if len(selected) >= max_count:
                break
    return frame.loc[selected].sort_index() if selected else frame.iloc[0:0].copy()


def independent_level_matches(frame, current, max_count, separation_minutes):
    candidates = frame.copy()
    candidates["distance"] = (candidates["origin"] - current).abs()
    return time_separated_candidates(candidates, max_count, separation_minutes)


def event_representatives(frame, flag_column, separation_minutes=60):
    """Return the first origin from each distinct flagged event cluster."""
    if frame.empty or flag_column not in frame:
        return frame.iloc[0:0].copy()
    flagged = frame.loc[frame[flag_column].fillna(False).astype(bool)].copy()
    if flagged.empty:
        return flagged
    new_event = flagged.index.to_series().diff().gt(
        pd.Timedelta(minutes=separation_minutes)
    )
    event_ids = new_event.cumsum()
    return flagged.groupby(event_ids, group_keys=False).head(1)


def shadow_dry_dispersion_watch(rows):
    """Test an issued-weather morning-clearing hypothesis without using it live."""
    base = {
        "enabled": True,
        "usedForDecision": False,
        "state": "collecting",
        "label": "Dry daytime dispersion watch",
        "validationState": "hypothesis_only",
        "auditedIndependentEventCount": 2,
        "minimumIndependentEventsBeforeReview": 5,
        "auditAsOf": "2026-08-29",
    }
    if len(rows) < 3:
        return base
    frame = sensor_frame(rows)
    if len(frame) < 3:
        return base
    latest_epoch = int(rows[-1]["epoch"])
    latest_local = local_dt(latest_epoch)
    recent = frame.iloc[-3:]
    required_columns = ["pm02", "pm10", "atmp", "rhum"]
    if recent[required_columns].isna().to_numpy().any():
        return base

    pm = frame["pm02"]
    pm10 = frame["pm10"]
    count = frame["pm003Count"]
    pm_moves = pm.diff().iloc[-2:]
    pm_change30 = num(pm.diff(2).iloc[-1])
    pm10_fraction30 = num(pm10.pct_change(2, fill_method=None).iloc[-1])
    count_fraction30 = num(count.pct_change(2, fill_method=None).iloc[-1])
    temperature_change30 = num(frame["atmp"].diff(2).iloc[-1])
    humidity_change30 = num(frame["rhum"].diff(2).iloc[-1])
    current = num(pm.iloc[-1])
    issued = issued_weather_horizon(latest_epoch, hours=3)
    if (current is None or pm_change30 is None or pm10_fraction30 is None
            or not issued):
        return base

    count_support = (
        True if count_fraction30 is None
        else count_fraction30 <= -DRY_DISPERSION_COUNT_FRACTION30
    )
    rain_probability = issued.get("rainProbabilityMax")
    temperature_change = issued.get("temperatureChange")
    humidity_change = issued.get("humidityChange")
    boundary_layer = issued.get("boundaryLayerHeightMean")
    wind_180m = issued.get("windSpeed180mMean")
    forecast_dry = bool(
        rain_probability is not None
        and rain_probability <= DRY_DISPERSION_MAX_RAIN_PROBABILITY
        and (issued.get("precipitationMax") or 0.0) <= DRY_CLEARING_MAX_PRECIPITATION
    )
    forecast_warming_drying = bool(
        temperature_change is not None and temperature_change > 0
        and humidity_change is not None and humidity_change < 0
    )
    ventilation_support = bool(
        (boundary_layer is not None
         and boundary_layer >= DRY_DISPERSION_MIN_BOUNDARY_LAYER_M)
        or (wind_180m is not None
            and wind_180m >= DRY_DISPERSION_MIN_WIND_180M_KMH)
    )
    active = bool(
        DRY_DISPERSION_START_HOUR <= latest_local.hour < DRY_DISPERSION_END_HOUR
        and current >= DRY_DISPERSION_MIN_PM25
        and bool(pm_moves.lt(0).all())
        and pm_change30 <= -DRY_DISPERSION_PM25_DROP30
        and pm10_fraction30 <= -DRY_DISPERSION_PM10_FRACTION30
        and count_support
        and forecast_dry
        and forecast_warming_drying
        and ventilation_support
    )
    base.update({
        "state": "watch" if active else "none",
        "active": active,
        "originEpoch": latest_epoch,
        "metrics": {
            "pm25": round(current, 1),
            "pm25Change30": round(pm_change30, 1),
            "pm10ChangePct30": round(100 * pm10_fraction30, 1),
            "particleCountChangePct30": (
                None if count_fraction30 is None
                else round(100 * count_fraction30, 1)
            ),
            "sensorTemperatureChange30": (
                None if temperature_change30 is None
                else round(temperature_change30, 1)
            ),
            "sensorHumidityChange30": (
                None if humidity_change30 is None
                else round(humidity_change30, 1)
            ),
            "issuedRainProbabilityMax": rain_probability,
            "issuedBoundaryLayerHeightMean": (
                None if boundary_layer is None else round(boundary_layer)
            ),
            "issuedWindSpeed180mMean": (
                None if wind_180m is None else round(wind_180m, 1)
            ),
            "issuedTemperatureChange": (
                None if temperature_change is None else round(temperature_change, 1)
            ),
            "issuedHumidityChange": (
                None if humidity_change is None else round(humidity_change, 1)
            ),
        },
        "issuedForecastFetchedEpoch": issued.get("fetchedEpoch"),
    })
    return base


def shadow_feature_frame(frame):
    """Origin-known features for an isolated robust analogue experiment."""
    pm = frame["pm02"]
    pm10 = frame["pm10"]
    features = pd.DataFrame(index=frame.index)
    features["pmLevel"] = pm
    for buckets, minutes in ((1, 15), (2, 30), (4, 60), (8, 120)):
        features[f"pmLag{minutes}"] = pm.shift(buckets)
        features[f"pmDelta{minutes}"] = pm.diff(buckets)
    ema_fast = segmented_ewm(pm, 3)
    ema_slow = segmented_ewm(pm, 8)
    features["emaGap"] = ema_fast - ema_slow
    median60 = pm.rolling(4, min_periods=3).median()
    features["rollingMad60"] = (
        (pm - median60).abs().rolling(4, min_periods=3).median()
    )
    features["rollingSd60"] = pm.rolling(4, min_periods=3).std()
    features["pm10Level"] = pm10
    features["coarseParticles"] = (pm10 - pm).clip(lower=0)
    features["fineShare"] = (100 * pm / pm10).clip(lower=0, upper=100)
    if "pm003Count" in frame:
        features["countPerPm25"] = frame["pm003Count"] / pm.replace(0, pd.NA)
        features["countDelta30"] = frame["pm003Count"].diff(2)
    features["temperature"] = frame["atmp"]
    features["temperatureDelta30"] = frame["atmp"].diff(2)
    features["humidity"] = frame["rhum"]
    features["humidityDelta30"] = frame["rhum"].diff(2)
    minute_of_day = frame.index.hour * 60 + frame.index.minute
    features["timeSin"] = [
        math.sin(2 * math.pi * minute / 1440) for minute in minute_of_day
    ]
    features["timeCos"] = [
        math.cos(2 * math.pi * minute / 1440) for minute in minute_of_day
    ]
    return features.replace([math.inf, -math.inf], pd.NA)


def robust_local_training(features, targets, query_time, train_end=None):
    """Prepare the original median/IQR estimator using contiguous numeric arrays."""
    if query_time not in features.index:
        return None
    response = targets.reindex(features.index).to_numpy(dtype=float, na_value=np.nan)
    if response.ndim == 1:
        response = response[:, None]
    matrix = features.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(response).all(axis=1)
    if train_end is not None:
        valid &= features.index < train_end
    if valid.sum() < SHADOW_ANALOG_MIN_ORIGINS:
        return None
    query = matrix[features.index.get_loc(query_time)]
    training = matrix[valid]
    usable = np.isfinite(query) & (np.isfinite(training).mean(axis=0) >= 0.60)
    if usable.sum() < 8:
        return None
    training = training[:, usable]
    query = query[usable]
    median = np.nanmedian(training, axis=0)
    q25, q75 = np.nanquantile(training, [0.25, 0.75], axis=0)
    scale = q75-q25
    fallback = np.nanstd(training, axis=0, ddof=1)
    scale = np.where(np.abs(scale) > 1e-9, scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale != 0), scale, 1.0)
    missing = ~np.isfinite(training)
    filled = np.where(missing, median, training)
    return filled, query, median, scale, missing, response[valid], features.index[valid], list(features.columns[usable])


def shadow_analogue_prediction(features, targets, query_time, train_end=None):
    """Predict deltas from robust, time-separated historical analogues."""
    target_columns = ["arrivalDelta", "trailMeanDelta", "trailPeakDelta"]
    prepared = robust_local_training(features, targets[target_columns], query_time, train_end)
    if prepared is None:
        return None
    filled, query, medians, scale, missing, responses, training_index, feature_columns = prepared
    normalized = np.clip((filled-query)/scale, -4.0, 4.0)
    distance = np.sqrt(np.mean(normalized**2, axis=1)) + missing.mean(axis=1)*0.5
    candidates = pd.DataFrame(responses, index=training_index, columns=target_columns)
    candidates["distance"] = distance
    matched = time_separated_candidates(
        candidates, SHADOW_ANALOG_MATCHES, SHADOW_ANALOG_SEPARATION_MINUTES
    )
    if len(matched) < SHADOW_ANALOG_MIN_MATCHES:
        return None
    weights = 1.0 / (0.25 + matched["distance"])

    def prediction(column):
        return weighted_median(zip(matched[column], weights))

    return {
        "arrivalDelta": prediction("arrivalDelta"),
        "trailMeanDelta": prediction("trailMeanDelta"),
        "trailPeakDelta": prediction("trailPeakDelta"),
        "trainingOriginCount": int(len(training_index)),
        "independentTrainingOriginCount": independent_origin_count(
            training_index, SHADOW_ANALOG_SEPARATION_MINUTES
        ),
        "matchedCount": int(len(matched)),
        "featureCount": int(len(feature_columns)),
        "featureColumns": feature_columns,
    }


def shadow_ridge_delta_prediction(features, target, query_time, train_end=None):
    """Fixed-alpha robust Ridge for a local forecast delta."""
    prepared = robust_local_training(features, target, query_time, train_end)
    if prepared is None:
        return None
    filled, query, medians, scale, missing, responses, training_index, feature_columns = prepared
    matrix = np.clip((filled-medians)/scale, -4.0, 4.0)
    query_matrix = np.clip((query-medians)/scale, -4.0, 4.0)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    query_design = np.concatenate([[1.0], query_matrix])
    penalty = np.eye(design.shape[1]) * SHADOW_RIDGE_ALPHA
    penalty[0, 0] = 0.0
    response = responses[:, 0]
    try:
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ response
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            design.T @ design + penalty, design.T @ response, rcond=None
        )[0]
    return {
        "delta": float(query_design @ coefficients),
        "trainingOriginCount": int(len(training_index)),
        "featureCount": int(len(feature_columns)),
    }


def local_pm_issued_evidence(frame, as_of_epoch):
    """Score the frozen candidate from recorded issues, never retrospective fits."""
    base = {"modelVersion": LOCAL_RIDGE_MODEL_VERSION,
            "validationMode": "recorded_issues_before_target",
            "eligible": False, "scoredCount": 0, "distinctDays": 0,
            "independentOriginCount": 0, "minimumDistinctDays": 7,
            "minimumIndependentOrigins": 20}
    try:
        with db() as conn:
            issues = conn.execute(
                "SELECT * FROM local_pm_forecast_issues WHERE model_version=? "
                "AND issued_epoch<=? AND origin_epoch>=? ORDER BY origin_epoch",
                (LOCAL_RIDGE_MODEL_VERSION, int(as_of_epoch),
                 int(as_of_epoch)-28*86400),
            ).fetchall()
    except sqlite3.Error:
        return {**base, "reason": "No prospective issue log yet",
                "gateOutcome": "unavailable", "failedPredicates": [],
                "unavailablePredicates": ["issue_log_available"], "notEvaluatedPredicates": [],
                "gateChecks": {"issue_log_available": {
                    "value": None, "operator": "==", "required": True, "threshold": True,
                    "margin": None, "passed": None, "outcome": "unavailable",
                    "reason": "prospective_issue_log_unavailable"}}}
    pm = fully_closed_sensor_frame(frame, as_of_epoch)["pm02"]
    records = []
    for issue in issues:
        origin = int(issue["origin_epoch"])
        issued = int(issue["issued_epoch"])
        try:
            payload = json.loads(issue["payload"])
            stored_clock = payload["forecastClock"]
            clock = ForecastClock(int(stored_clock["forecastIssuedEpoch"]), origin)
            if not targets_match(clock.metadata(), stored_clock):
                continue
            # Preserve the originally issued clock. Neither a delayed cache
            # write nor today's target definition can relabel a past issue.
            if not (0 <= clock.lag_seconds < AIR_BUCKET_MINUTES*60
                    and 0 <= issued-clock.issued_epoch <= 120):
                continue
            outcome = score_observed_window(pm, clock, as_of_epoch)
            if not outcome["available"]:
                continue
            weight = float(payload["trailMeanWeight"])
            anchor = float(issue["anchor_pm25"])
            prediction = max(0.0, anchor+weight*float(issue["trail_mean_delta"]))
        except (TypeError, ValueError, KeyError):
            continue
        if weight != AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT:
            continue
        start = pd.Timestamp(origin, unit="s", tz="UTC").tz_convert(pm.index.tz)
        records.append((origin, str(start.date()), outcome["mean"], anchor, prediction))
    if not records:
        return {**base, "reason": "Awaiting completed issued outcomes",
                "gateOutcome": "not_evaluated", "failedPredicates": [], "unavailablePredicates": [],
                "notEvaluatedPredicates": ["completed_issued_outcomes_available"],
                "gateChecks": {"completed_issued_outcomes_available": {
                    "value": 0, "operator": ">", "required": 0, "threshold": 0,
                    "margin": None, "passed": None, "outcome": "not_evaluated",
                    "reason": "no_completed_issued_outcomes_to_score"}}}
    scored = pd.DataFrame(records, columns=["origin", "day", "actual", "baseline", "prediction"])
    selected = []
    last_origin = None
    for pos, origin in enumerate(scored["origin"]):
        if last_origin is None or origin-last_origin >= 210*60:
            selected.append(pos)
            last_origin = origin

    def metric(group):
        error = group["prediction"].to_numpy()-group["actual"].to_numpy()
        baseline = group["baseline"].to_numpy()-group["actual"].to_numpy()
        mae, base_mae = np.abs(error).mean(), np.abs(baseline).mean()
        return {"mae": float(mae), "persistenceMae": float(base_mae),
                "gainUgM3": float(base_mae-mae),
                "gainPct": float(100*(1-mae/base_mae)) if base_mae > 1e-9 else 0.0,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "persistenceRmse": float(np.sqrt(np.mean(baseline**2))),
                "p90Error": float(np.quantile(np.abs(error), .9)),
                "persistenceP90Error": float(np.quantile(np.abs(baseline), .9)),
                "winFraction": float((np.abs(error)<np.abs(baseline)).mean())}

    overall, independent = metric(scored), metric(scored.iloc[selected])
    daily = {day: metric(group) for day, group in scored.groupby("day")}
    positive_days = sum(item["gainUgM3"] > 0 for item in daily.values())
    eligible = bool(len(daily) >= 7 and len(selected) >= 20
        and overall["gainPct"] >= 5 and overall["gainUgM3"] >= 1
        and independent["gainPct"] >= 5 and independent["winFraction"] >= .55
        and overall["rmse"] <= overall["persistenceRmse"]
        and overall["p90Error"] <= overall["persistenceP90Error"]
        and positive_days/len(daily) >= .70)
    # Keep unrounded predicate inputs: rounded display metrics can conceal a
    # just-failed threshold. These diagnostics do not select or refit a model.
    checks = {}
    for name, value, operator, required in (
            ("distinct_days", len(daily), ">=", 7),
            ("independent_origins", len(selected), ">=", 20),
            ("overall_mae_gain_pct", overall["gainPct"], ">=", 5),
            ("overall_mae_gain_ug_m3", overall["gainUgM3"], ">=", 1),
            ("independent_mae_gain_pct", independent["gainPct"], ">=", 5),
            ("independent_win_fraction", independent["winFraction"], ">=", .55),
            ("overall_rmse_not_worse", overall["rmse"], "<=", overall["persistenceRmse"]),
            ("overall_p90_not_worse", overall["p90Error"], "<=", overall["persistenceP90Error"]),
            ("positive_day_fraction", positive_days/len(daily), ">=", .70),
    ):
        value_finite = value is not None and math.isfinite(value)
        required_finite = required is not None and math.isfinite(required)
        passed = bool(value >= required if operator == ">=" else value <= required) if value_finite and required_finite else None
        margin = (value-required if operator == ">=" else required-value) if passed is not None else None
        checks[name] = {
            "value": value if value_finite else None, "operator": operator,
            "required": required if required_finite else None, "threshold": required if required_finite else None,
            "margin": margin if margin is not None and math.isfinite(margin) else None,
            "passed": passed, "outcome": "unavailable" if passed is None else "passed" if passed else "failed",
            "valueState": "finite" if value_finite else "missing" if value is None else "nonfinite",
        }
    return {**base, "eligible": eligible, "scoredCount": len(scored),
            "gateChecks": checks,
            "gateOutcome": "passed" if eligible else "unavailable" if any(check["outcome"] == "unavailable" for check in checks.values()) else "failed",
            "failedPredicates": [name for name, check in checks.items() if check["outcome"] == "failed"],
            "unavailablePredicates": [name for name, check in checks.items() if check["outcome"] == "unavailable"],
            "notEvaluatedPredicates": [],
            "distinctDays": len(daily), "independentOriginCount": len(selected),
            "metrics": {k: round(v, 3) for k, v in overall.items()},
            "independentMetrics": {k: round(v, 3) for k, v in independent.items()},
            "positiveDayCount": positive_days,
            "reason": ("Issued outcomes pass the fixed promotion checks" if eligible else
                       "Awaiting sufficient independent days and consistent error improvement"),
            "asOfEpoch": int(as_of_epoch)}


def cached_local_replay_prediction(features, targets, origin, train_end):
    """Reuse an immutable historical issue, invalidating on any causal input edit."""
    known_features = features.loc[features.index < train_end]
    known_targets = targets.reindex(known_features.index)
    digest = hashlib.blake2b(digest_size=16)
    for block in (known_features, known_targets, features.loc[[origin]]):
        values = block.to_numpy(dtype=float, na_value=np.nan).copy()
        values[np.isnan(values)] = np.nan  # Canonical missing-value representation.
        digest.update(block.index.asi8.tobytes())
        digest.update(values.tobytes())
    key = (str(DB_PATH), tuple(features.columns), SHADOW_RIDGE_ALPHA,
           SHADOW_ANALOG_MATCHES, SHADOW_ANALOG_SEPARATION_MINUTES,
           int(origin.timestamp()), int(train_end.timestamp()), digest.digest())
    with local_replay_lock:
        if key in local_replay_cache:
            local_replay_cache.move_to_end(key)
            return local_replay_cache[key]
    value = (
        shadow_analogue_prediction(features, targets, origin, train_end),
        shadow_ridge_delta_prediction(features, targets["trailMeanDelta"], origin, train_end),
    )
    with local_replay_lock:
        local_replay_cache[key] = value
        while len(local_replay_cache) > 512:
            local_replay_cache.popitem(last=False)
    return value


def shadow_analogue_outlook(frame, issue_epoch=None):
    """Return closed-bin local-model predictions and exact deployment replay.

    The current partial right-labelled bucket is useful to the nowcast but is
    not a completed forecast origin or outcome.  The model therefore learns
    and queries on the latest fully closed 15-minute bucket; its delta is later
    translated from the same completed 15-minute forecast anchor.
    """
    if issue_epoch is not None:
        frame = fully_closed_sensor_frame(frame, issue_epoch)
    if frame.empty:
        return {
            "enabled": True, "usedForDecision": False, "status": "collecting"
        }
    pm = frame["pm02"]
    if pd.isna(pm.iloc[-1]):
        return {"enabled": True, "usedForDecision": False, "status": "incomplete_closed_anchor"}
    features = shadow_feature_frame(frame)
    current_time = features.index[-1]
    clock = ForecastClock(
        int(issue_epoch or current_time.timestamp()), int(current_time.timestamp())
    )
    if (clock.lag_seconds >= AIR_BUCKET_MINUTES*60
            or clock.outcome_complete_lead_seconds >= SHADOW_TRAINING_EMBARGO_MINUTES*60):
        return {"enabled": True, "usedForDecision": False, "status": "stale_closed_anchor"}
    outcomes = target_series(pm, clock)
    targets = pd.DataFrame({
        "arrivalDelta": outcomes["arrival"] - pm,
        "trailMeanDelta": outcomes["mean"] - pm,
        "trailPeakDelta": outcomes["peak"] - pm,
    })
    current = float(pm.loc[current_time])
    current_train_end = current_time - pd.Timedelta(
        minutes=SHADOW_TRAINING_EMBARGO_MINUTES
    )
    current_prediction = shadow_analogue_prediction(
        features, targets, current_time, train_end=current_train_end
    )
    current_mean_ridge = shadow_ridge_delta_prediction(
        features, targets["trailMeanDelta"], current_time,
        train_end=current_train_end,
    )
    current_peak_ridge = shadow_ridge_delta_prediction(
        features, targets["trailPeakDelta"], current_time,
        train_end=current_train_end,
    )
    base = {
        "enabled": True,
        "usedForDecision": False,
        "method": "Replay-screened local analogue and robust Ridge deltas",
        "trainingEmbargoMinutes": SHADOW_TRAINING_EMBARGO_MINUTES,
        "status": "replay_screened_experimental" if current_prediction else "collecting",
        "forecastClock": clock.metadata(),
    }
    if not current_prediction or not current_mean_ridge:
        base["minimumTrainingOrigins"] = SHADOW_ANALOG_MIN_ORIGINS
        return base

    arrival_point = max(0.0, current + current_prediction["arrivalDelta"])
    trail_mean_point = max(0.0, current + current_mean_ridge["delta"])
    # The bounded matched-analogue peak replaces Ridge in production.  Ridge is
    # retained below as diagnostic telemetry because it recently extrapolated a
    # +35.9 µg/m³ delta from a stable 51.7 µg/m³ origin.
    trail_peak_analogue_point = max(
        trail_mean_point, current + current_prediction["trailPeakDelta"]
    )
    trail_peak_ridge_point = (
        None if not current_peak_ridge else max(
            trail_mean_point, current + current_peak_ridge["delta"]
        )
    )
    base.update({
        "featureAnchor": current,
        "featureAnchorEpoch": int(current_time.timestamp()),
        "featureSetVersion": "closed_15min_particle_met_decision_clock_v4",
        "arrivalDelta": float(current_prediction["arrivalDelta"]),
        "trailMeanDelta": float(current_mean_ridge["delta"]),
        "trailPeakDelta": float(current_prediction["trailPeakDelta"]),
        "arrivalPoint": round(arrival_point, 1),
        "trailMeanPoint": round(trail_mean_point, 1),
        "trailPeakPoint": round(trail_peak_analogue_point, 1),
        "trailPeakAnaloguePoint": round(trail_peak_analogue_point, 1),
        "trailPeakRidgePoint": (
            None if trail_peak_ridge_point is None else round(trail_peak_ridge_point, 1)
        ),
        "trainingOriginCount": current_prediction["trainingOriginCount"],
        "trailMeanModel": "robust_ridge_alpha_100_shrink_0.25_v1",
        "trailMeanRidgeTrainingOriginCount": current_mean_ridge["trainingOriginCount"],
        "independentTrainingOriginCount": current_prediction["independentTrainingOriginCount"],
        "matchedCount": current_prediction["matchedCount"],
        "featureCount": current_prediction["featureCount"],
    })

    completed = targets.dropna().index
    # Fix replay origins to the wall clock instead of inheriting whichever
    # 15-minute phase happened to begin the database. This makes successive
    # reviews comparable as older history ages out of the training window.
    hourly_completed = completed[completed.minute == 0]
    replay_origins = list(hourly_completed)[-SHADOW_BACKTEST_MAX_ORIGINS:]
    actual = {column: [] for column in targets.columns}
    predicted = {column: [] for column in targets.columns}
    raw_predicted = {column: [] for column in targets.columns}
    replay_times = []
    peak_components = []
    for origin in replay_origins:
        train_end = origin - pd.Timedelta(
            minutes=SHADOW_TRAINING_EMBARGO_MINUTES
        )
        forecast, mean_forecast = cached_local_replay_prediction(
            features, targets, origin, train_end
        )
        if not forecast or not mean_forecast:
            continue
        actual_arrival = float(targets.loc[origin, "arrivalDelta"])
        actual_mean = float(targets.loc[origin, "trailMeanDelta"])
        actual_peak = float(targets.loc[origin, "trailPeakDelta"])
        predicted_arrival = (
            AGGRESSIVE_ARRIVAL_ANALOGUE_WEIGHT * float(forecast["arrivalDelta"])
        )
        predicted_mean = (
            AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT * float(mean_forecast["delta"])
        )
        weighted_peak = (
            AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT
            * float(forecast["trailPeakDelta"])
        )
        actual["arrivalDelta"].append(actual_arrival)
        actual["trailMeanDelta"].append(actual_mean)
        actual["trailPeakDelta"].append(actual_peak)
        predicted["arrivalDelta"].append(predicted_arrival)
        predicted["trailMeanDelta"].append(predicted_mean)
        # This is replaced below after mean eligibility is known. Production
        # floors the peak at the deployed mean, which can independently be
        # either persistence or the screened analogue mean.
        predicted["trailPeakDelta"].append(weighted_peak)
        peak_components.append(weighted_peak)
        raw_predicted["arrivalDelta"].append(float(forecast["arrivalDelta"]))
        raw_predicted["trailMeanDelta"].append(float(mean_forecast["delta"]))
        raw_predicted["trailPeakDelta"].append(max(
            float(forecast["trailMeanDelta"]), float(forecast["trailPeakDelta"])
        ))
        replay_times.append(origin)

    test_count = len(actual["arrivalDelta"])
    if test_count:
        def error_values(column, source="deployed", positions=None):
            guesses = (
                predicted[column] if source == "deployed" else
                raw_predicted[column] if source == "raw" else
                [0.0] * test_count
            )
            selected = range(test_count) if positions is None else positions
            return [
                abs(actual[column][position] - guesses[position])
                for position in selected
            ]

        def mae(column, source="deployed", positions=None):
            errors = error_values(column, source, positions)
            return sum(errors) / len(errors) if errors else math.nan

        independent_positions = []
        last_independent_time = None
        independent_separation = pd.Timedelta(
            minutes=AGGRESSIVE_ANALOGUE_INDEPENDENT_SEPARATION_MINUTES
        )
        for position, stamp in enumerate(replay_times):
            if (last_independent_time is None
                    or stamp - last_independent_time >= independent_separation):
                independent_positions.append(position)
                last_independent_time = stamp
        independent_count = len(independent_positions)

        persistence_mae = {
            column: mae(column, "persistence") for column in targets.columns
        }
        deployed_mae = {
            column: mae(column, "deployed") for column in targets.columns
        }
        independent_persistence_mae = {
            column: mae(column, "persistence", independent_positions)
            for column in targets.columns
        }
        independent_deployed_mae = {
            column: mae(column, "deployed", independent_positions)
            for column in targets.columns
        }
        distinct_days = len({stamp.date() for stamp in replay_times})

        def skill(column):
            gain = persistence_mae[column] - deployed_mae[column]
            gain_fraction = (
                0.0 if persistence_mae[column] <= 1e-9
                else gain / persistence_mae[column]
            )
            independent_gain = (
                independent_persistence_mae[column]
                - independent_deployed_mae[column]
            )
            independent_gain_fraction = (
                0.0 if independent_persistence_mae[column] <= 1e-9
                else independent_gain / independent_persistence_mae[column]
            )
            deployed_errors = error_values(column, "deployed")
            persistence_errors = error_values(column, "persistence")
            win_fraction = sum(
                model_error < baseline_error
                for model_error, baseline_error in zip(
                    deployed_errors, persistence_errors
                )
            ) / len(deployed_errors)
            independent_deployed_errors = error_values(
                column, "deployed", independent_positions
            )
            independent_persistence_errors = error_values(
                column, "persistence", independent_positions
            )
            independent_win_fraction = (
                0.0 if not independent_deployed_errors else
                sum(
                    model_error < baseline_error
                    for model_error, baseline_error in zip(
                        independent_deployed_errors,
                        independent_persistence_errors,
                    )
                ) / len(independent_deployed_errors)
            )
            checks = {}
            for name, value, required in (
                    ("replay_origins", test_count, AGGRESSIVE_ANALOGUE_MIN_REPLAY_ORIGINS),
                    ("distinct_days", distinct_days, AGGRESSIVE_ANALOGUE_MIN_REPLAY_DAYS),
                    ("independent_origins", independent_count, AGGRESSIVE_ANALOGUE_MIN_INDEPENDENT_ORIGINS),
                    ("overall_mae_gain_fraction", gain_fraction, AGGRESSIVE_ANALOGUE_MIN_SKILL_GAIN),
                    ("independent_mae_gain_fraction", independent_gain_fraction, AGGRESSIVE_ANALOGUE_MIN_SKILL_GAIN),
                    ("overall_win_fraction", win_fraction, AGGRESSIVE_ANALOGUE_MIN_WIN_FRACTION),
                    ("independent_win_fraction", independent_win_fraction, AGGRESSIVE_ANALOGUE_MIN_WIN_FRACTION),
            ):
                value_finite = value is not None and math.isfinite(value)
                required_finite = required is not None and math.isfinite(required)
                passed = bool(value >= required) if value_finite and required_finite else None
                margin = value-required if passed is not None else None
                checks[name] = {
                    "value": value if value_finite else None, "operator": ">=",
                    "required": required if required_finite else None, "threshold": required if required_finite else None,
                    "margin": margin if margin is not None and math.isfinite(margin) else None,
                    "passed": passed, "outcome": "unavailable" if passed is None else "passed" if passed else "failed",
                    "valueState": "finite" if value_finite else "missing" if value is None else "nonfinite",
                }
            return {
                "target": column,
                "gateChecks": checks,
                "gateOutcome": "unavailable" if any(check["outcome"] == "unavailable" for check in checks.values()) else "passed" if all(check["passed"] for check in checks.values()) else "failed",
                "failedPredicates": [name for name, check in checks.items() if check["outcome"] == "failed"],
                "unavailablePredicates": [name for name, check in checks.items() if check["outcome"] == "unavailable"],
                "notEvaluatedPredicates": [],
                "eligible": bool(
                    test_count >= AGGRESSIVE_ANALOGUE_MIN_REPLAY_ORIGINS
                    and distinct_days >= AGGRESSIVE_ANALOGUE_MIN_REPLAY_DAYS
                    and independent_count >= AGGRESSIVE_ANALOGUE_MIN_INDEPENDENT_ORIGINS
                    and gain_fraction >= AGGRESSIVE_ANALOGUE_MIN_SKILL_GAIN
                    and independent_gain_fraction >= AGGRESSIVE_ANALOGUE_MIN_SKILL_GAIN
                    and win_fraction >= AGGRESSIVE_ANALOGUE_MIN_WIN_FRACTION
                    and independent_win_fraction
                    >= AGGRESSIVE_ANALOGUE_MIN_WIN_FRACTION
                ),
                "maeGainUgM3": round(gain, 2),
                "maeGainPct": round(100 * gain_fraction, 1),
                "winFraction": round(win_fraction, 3),
                "independentMae": round(independent_deployed_mae[column], 2),
                "independentPersistenceMae": round(
                    independent_persistence_mae[column], 2
                ),
                "independentMaeGainUgM3": round(independent_gain, 2),
                "independentMaeGainPct": round(
                    100 * independent_gain_fraction, 1
                ),
                "independentWinFraction": round(
                    independent_win_fraction, 3
                ),
            }

        # Reconstruct the two-stage production estimator exactly. Arrival and
        # mean eligibility are independent; only an eligible mean can become
        # the lower floor of the peak estimate. Recompute peak errors after
        # that gate so replay and live deployment cannot diverge.
        arrival_skill = skill("arrivalDelta")
        mean_skill = skill("trailMeanDelta")
        mean_skill["developmentReplayEligible"] = bool(mean_skill["eligible"])
        prospective = local_pm_issued_evidence(frame, int(current_time.timestamp()))
        base["prospectiveEvidence"] = prospective
        mean_skill["eligible"] = bool(mean_skill["developmentReplayEligible"] and prospective["eligible"])
        mean_skill["developmentFailedPredicates"] = list(mean_skill["failedPredicates"])
        mean_skill["developmentGateOutcome"] = mean_skill["gateOutcome"]
        mean_skill["prospectiveEligible"] = bool(prospective["eligible"])
        mean_skill["prospectiveGateChecks"] = prospective.get("gateChecks", {})
        mean_skill["prospectiveFailedPredicates"] = list(prospective.get("failedPredicates", []))
        mean_skill["prospectiveUnavailablePredicates"] = list(prospective.get("unavailablePredicates", []))
        mean_skill["prospectiveNotEvaluatedPredicates"] = list(prospective.get("notEvaluatedPredicates", []))
        prospective_outcome = prospective.get("gateOutcome", "passed" if prospective["eligible"] else "failed")
        mean_skill["gateChecks"]["prospective_evidence_eligible"] = {
            "value": bool(prospective["eligible"]) if prospective_outcome in ("passed", "failed") else None,
            "operator": "==", "required": True, "threshold": True, "margin": None,
            "passed": bool(prospective["eligible"]) if prospective_outcome in ("passed", "failed") else None,
            "outcome": prospective_outcome,
        }
        if prospective_outcome == "failed":
            mean_skill["failedPredicates"].append("prospective_evidence_eligible")
        elif prospective_outcome == "unavailable":
            mean_skill["unavailablePredicates"].append("prospective_evidence_eligible")
        elif prospective_outcome == "not_evaluated":
            mean_skill["notEvaluatedPredicates"].append("prospective_evidence_eligible")
        mean_skill["gateOutcome"] = (
            "unavailable" if mean_skill["unavailablePredicates"] else
            "not_evaluated" if mean_skill["notEvaluatedPredicates"] else
            "passed" if mean_skill["eligible"] else "failed"
        )
        mean_skill["promotionState"] = (
            "issued_evidence_supported" if mean_skill["eligible"] else "frozen_prospective_shadow"
        )
        mean_skill["promotionReason"] = prospective["reason"]
        deployed_mean_floor = (
            predicted["trailMeanDelta"] if mean_skill["eligible"]
            else [0.0] * test_count
        )
        predicted["trailPeakDelta"] = [
            max(mean_delta, peak_delta)
            for mean_delta, peak_delta in zip(
                deployed_mean_floor, peak_components
            )
        ]
        deployed_mae["trailPeakDelta"] = mae("trailPeakDelta", "deployed")
        independent_deployed_mae["trailPeakDelta"] = mae(
            "trailPeakDelta", "deployed", independent_positions
        )
        peak_skill = skill("trailPeakDelta")
        peak_residuals = [
            observed - estimate
            for observed, estimate in zip(
                actual["trailPeakDelta"], predicted["trailPeakDelta"]
            )
        ]

        base["backtest"] = {
            "mode": (
                "deployed_weights_fixed_clock_hourly_and_horizon_separated_walk_forward_"
                "330_minute_embargo_decision_clock"
            ),
            "targetVersion": clock.metadata()["targetVersion"],
            "featureToIssueLagSeconds": clock.lag_seconds,
            "maximumOutcomeLeadMinutes": clock.outcome_complete_lead_seconds / 60,
            "testOriginCount": test_count,
            "distinctDays": distinct_days,
            "minimumTestOrigins": AGGRESSIVE_ANALOGUE_MIN_REPLAY_ORIGINS,
            "minimumDistinctDays": AGGRESSIVE_ANALOGUE_MIN_REPLAY_DAYS,
            "independentOriginCount": independent_count,
            "minimumIndependentOrigins": (
                AGGRESSIVE_ANALOGUE_MIN_INDEPENDENT_ORIGINS
            ),
            "independentSeparationMinutes": (
                AGGRESSIVE_ANALOGUE_INDEPENDENT_SEPARATION_MINUTES
            ),
            "minimumMaeGainPct": round(
                100 * AGGRESSIVE_ANALOGUE_MIN_SKILL_GAIN
            ),
            "minimumWinFraction": AGGRESSIVE_ANALOGUE_MIN_WIN_FRACTION,
            "arrivalWeight": AGGRESSIVE_ARRIVAL_ANALOGUE_WEIGHT,
            "trailMeanWeight": AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT,
            "trailPeakWeight": AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT,
            "trailMeanSource": "robust_ridge_alpha_100",
            "trailPeakSource": "bounded_matched_analogue",
            "arrivalMae": round(deployed_mae["arrivalDelta"], 2),
            "arrivalPersistenceMae": round(persistence_mae["arrivalDelta"], 2),
            "trailMeanMae": round(deployed_mae["trailMeanDelta"], 2),
            "trailMeanPersistenceMae": round(persistence_mae["trailMeanDelta"], 2),
            "trailPeakMae": round(deployed_mae["trailPeakDelta"], 2),
            "trailPeakPersistenceMae": round(persistence_mae["trailPeakDelta"], 2),
            "rawAnalogueArrivalMae": round(mae("arrivalDelta", "raw"), 2),
            "rawRidgeTrailMeanMae": round(mae("trailMeanDelta", "raw"), 2),
            "rawAnalogueTrailPeakMae": round(mae("trailPeakDelta", "raw"), 2),
            "deploymentSkill": {
                "arrival": arrival_skill,
                "trailMean": mean_skill,
                "trailPeak": peak_skill,
            },
            "trailPeakDeploymentCombination": (
                "eligible_analogue_mean_floor" if mean_skill["eligible"]
                else "persistence_mean_floor"
            ),
        }
        upper_residual = empirical_quantile(peak_residuals, 0.90, upper=True)
        # Never let a rejected peak component influence a mean-only live
        # deployment. In that state the existing persistence peak bound remains
        # authoritative.
        if upper_residual is not None and peak_skill["eligible"]:
            deployed_mean_delta = (
                AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT
                * current_mean_ridge["delta"]
                if mean_skill["eligible"] else 0.0
            )
            deployed_peak_delta = max(
                deployed_mean_delta,
                AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT
                * current_prediction["trailPeakDelta"],
            )
            upper_delta = max(
                deployed_peak_delta, deployed_peak_delta + upper_residual
            )
            base["trailPeakUpperDelta"] = float(upper_delta)
            base["trailPeakUpper90"] = round(
                current + upper_delta, 1
            )
    return base


def cached_shadow_analogue_outlook(rows, issue_epoch=None):
    """Isolate and cache experimental work so it cannot break production analysis."""
    frame = sensor_frame(rows).copy()
    complete = frame.dropna(subset=["pm02", "pm10"])
    if complete.empty:
        return {
            "enabled": True, "usedForDecision": False, "status": "collecting"
        }
    frame = frame.loc[:complete.index[-1]]
    latest_raw_epoch = int(rows[-1]["epoch"])
    issue_epoch = int(issue_epoch or latest_raw_epoch)
    if not 0 <= issue_epoch-latest_raw_epoch <= SENSOR_DEGRADED_SECONDS:
        return {"enabled": True, "usedForDecision": False, "status": "sensor_stale"}
    closed = fully_closed_sensor_frame(frame, issue_epoch)
    if closed.empty:
        return {
            "enabled": True, "usedForDecision": False, "status": "collecting"
        }
    feature_revision = hashlib.blake2b(
        pd.util.hash_pandas_object(
            closed[
                [
                    column for column in (
                        "pm02", "pm10", "pm003Count", "atmp", "rhum",
                        "heatindex",
                    ) if column in closed
                ]
            ],
            index=True,
        ).values.tobytes(),
        digest_size=16,
    ).hexdigest()
    cache_key = (
        int(closed.index[0].timestamp()), int(closed.index[-1].timestamp()),
        int(closed["pm02"].notna().sum()), feature_revision,
        issue_epoch - int(closed.index[-1].timestamp()), LOCAL_RIDGE_MODEL_VERSION,
    )
    with shadow_lock:
        if shadow_cache["cacheKey"] == cache_key:
            return shadow_cache["value"]
        try:
            value = shadow_analogue_outlook(frame, issue_epoch)
        except Exception as error:
            value = {
                "enabled": True,
                "usedForDecision": False,
                "status": "unavailable",
                "errorType": type(error).__name__,
            }
        shadow_cache["cacheKey"] = cache_key
        shadow_cache["value"] = value
        return value


def record_local_pm_forecast_issue(forecast, issued_epoch=None):
    """Persist one frozen local-model issue per completed 15-minute origin."""
    forecast = forecast or {}
    origin_epoch = forecast.get("featureAnchorEpoch")
    anchor = num(forecast.get("featureAnchor"))
    mean_delta = num(forecast.get("trailMeanDelta"))
    forecast_clock = forecast.get("forecastClock")
    if origin_epoch is None or anchor is None or mean_delta is None or not forecast_clock:
        return False
    issued_epoch = int(issued_epoch or time.time())
    try:
        clock = ForecastClock(int(forecast_clock["forecastIssuedEpoch"]), int(origin_epoch))
        if (not targets_match(clock.metadata(), forecast_clock)
                or not 0 <= clock.lag_seconds < AIR_BUCKET_MINUTES*60
                or not 0 <= issued_epoch-clock.issued_epoch <= 120):
            return False
    except (TypeError, ValueError, KeyError):
        return False
    payload = {
        "modelVersion": LOCAL_RIDGE_MODEL_VERSION,
        "featureSetVersion": forecast.get("featureSetVersion"),
        "originEpoch": int(origin_epoch),
        "forecastClock": forecast_clock,
        "anchorPm25": anchor,
        "arrivalDelta": forecast.get("arrivalDelta"),
        "trailMeanDelta": mean_delta,
        "trailPeakDelta": forecast.get("trailPeakDelta"),
        "trailMeanWeight": AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT,
        "trainingEmbargoMinutes": forecast.get("trainingEmbargoMinutes"),
        "backtestAtIssue": forecast.get("backtest"),
        "role": "frozen_prospective_shadow_candidate",
    }
    try:
        with db() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO local_pm_forecast_issues "
                "(origin_epoch,model_version,issued_epoch,anchor_pm25,"
                "arrival_delta,trail_mean_delta,trail_peak_delta,payload) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    int(origin_epoch), LOCAL_RIDGE_MODEL_VERSION, issued_epoch,
                    anchor, num(forecast.get("arrivalDelta")), mean_delta,
                    num(forecast.get("trailPeakDelta")),
                    json.dumps(json_safe(payload), separators=(",", ":")),
                ),
            )
            cutoff = issued_epoch - RETENTION_DAYS * 86400
            conn.execute(
                "DELETE FROM local_pm_forecast_issues WHERE issued_epoch < ?",
                (cutoff,),
            )
            return bool(cursor.rowcount)
    except sqlite3.Error:
        return False


def aggressive_air_window_forecast(air_window, analogue, rapid_event=None):
    """Overlay responsive experimental points without recentering persistence bands."""
    if not AGGRESSIVE_FORECAST_ENABLED or not air_window.get("available"):
        return air_window
    model_anchor = num((analogue or {}).get("featureAnchor"))
    if model_anchor is None:
        model_anchor = num(air_window.get("current15"))
    forecast_anchor = num(air_window.get("current15"))
    if model_anchor is None or forecast_anchor is None:
        return air_window
    updated = dict(air_window)
    clock_matches = targets_match(
        air_window.get("forecastClock"), (analogue or {}).get("forecastClock")
    )
    replay = ((analogue or {}).get("backtest") or {}) if clock_matches else {}
    if not clock_matches:
        updated["legacyOverlayWithheld"] = "Different or unavailable forecast target clock"
    deployment_skill = replay.get("deploymentSkill") or {}

    arrival_shadow = num((analogue or {}).get("arrivalPoint"))
    arrival_delta = num((analogue or {}).get("arrivalDelta"))
    arrival = dict(updated.get("arrival") or {})
    arrival_skill = bool((deployment_skill.get("arrival") or {}).get("eligible"))
    if arrival_shadow is not None and arrival_skill:
        model_delta = (
            arrival_delta if arrival_delta is not None
            else arrival_shadow - model_anchor
        )
        point = max(
            0.0, forecast_anchor + AGGRESSIVE_ARRIVAL_ANALOGUE_WEIGHT * model_delta
        )
        arrival.update({
            "available": True,
            "baselinePoint": round(forecast_anchor, 1),
            "modelFeatureAnchor": round(model_anchor, 1),
            "persistenceAnchorRole": "latest_closed_15_minute_bucket_median",
            "point": round(point, 1),
            "pointRole": "aggressive_local_analogue",
            "forecastState": "aggressive_experimental_point_and_range",
            "pointApproximate": True,
            "usedForDecision": True,
            "validated": False,
            "method": "Replay-screened damped local analogue with historical error bounds",
            "confidence": "Low · aggressive experimental model",
            "rangeRole": "persistence_anchor_and_projection_containing_display_envelope",
            "aggressiveWeight": AGGRESSIVE_ARRIVAL_ANALOGUE_WEIGHT,
            "modelMae": replay.get("arrivalMae"),
            "persistenceMae": replay.get("arrivalPersistenceMae"),
            "deploymentSkill": deployment_skill.get("arrival"),
        })
        if arrival.get("rangeLow") is not None:
            arrival["rangeLow"] = round(min(float(arrival["rangeLow"]), point), 1)
            arrival["rangeHigh"] = round(max(float(arrival["rangeHigh"]), point), 1)
            arrival["decisionUpper"] = round(max(
                float(arrival.get("upper90") or 0), point
            ), 1)
    updated["arrival"] = arrival

    trail_shadow = num((analogue or {}).get("trailMeanPoint"))
    peak_shadow = num((analogue or {}).get("trailPeakAnaloguePoint"))
    trail_delta = num((analogue or {}).get("trailMeanDelta"))
    peak_delta = num((analogue or {}).get("trailPeakDelta"))
    trail = dict(updated.get("trail") or {})
    mean_skill = bool((deployment_skill.get("trailMean") or {}).get("eligible"))
    peak_skill = bool((deployment_skill.get("trailPeak") or {}).get("eligible"))
    mean_candidate_available = bool(
        trail_shadow is not None and math.isfinite(trail_shadow)
        and (trail_delta is None or math.isfinite(trail_delta))
    )
    peak_candidate_available = bool(
        peak_shadow is not None and math.isfinite(peak_shadow)
        and (peak_delta is None or math.isfinite(peak_delta))
    )
    # An eligible mean is independent of peak availability. Peak-only routing
    # keeps its existing prerequisites and all target/skill gates remain intact.
    mean_applied = bool(mean_candidate_available and mean_skill)
    peak_applied = bool(mean_candidate_available and peak_candidate_available and peak_skill)
    if mean_applied or peak_applied:
        mean_model_delta = (
            trail_delta if trail_delta is not None
            else trail_shadow - model_anchor
        )
        peak_model_delta = (
            peak_delta if peak_delta is not None
            else peak_shadow - model_anchor if peak_shadow is not None else 0.0
        )
        base_mean = num(trail.get("point"))
        if base_mean is None:
            base_mean = forecast_anchor
        mean_point = (
            max(
                0.0,
                forecast_anchor
                + AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT * mean_model_delta,
            )
            if mean_applied else base_mean
        )
        peak_point = max(
            mean_point,
            forecast_anchor
            + (
                AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT * peak_model_delta
                if peak_applied else 0.0
            ),
        )
        trail.update({
            "available": True,
            "baselinePoint": round(forecast_anchor, 1),
            "modelFeatureAnchor": round(model_anchor, 1),
            "persistenceAnchorRole": "latest_closed_15_minute_bucket_median",
            "point": round(mean_point, 1),
            "pointRole": (
                "aggressive_local_ridge" if mean_applied
                else "persistence_anchor"
            ),
            "forecastState": (
                "aggressive_experimental_point_and_range" if mean_applied
                else "aggressive_experimental_peak_only"
            ),
            "pointApproximate": mean_applied,
            "peakApproximate": peak_applied,
            "usedForDecision": True,
            "validated": False,
            "method": (
                "Experimental robust Ridge mean and bounded analogue peak"
                if mean_applied and peak_candidate_available else
                "Experimental robust Ridge mean; existing peak evidence unchanged"
                if mean_applied else
                "Persistence mean with experimental bounded analogue peak"
            ),
            "headline": (
                "Experimental PM2.5 mean and high-side peak"
                if mean_applied and peak_candidate_available else
                "Experimental PM2.5 mean; no new peak estimate"
                if mean_applied else
                "Persistence mean with experimental high-side peak"
            ),
            "confidence": (
                "Low · experimental local model" if mean_applied
                else "Low · experimental peak model"
            ),
            "rangeRole": "persistence_anchor_and_projection_containing_display_envelope",
            "aggressiveMeanWeight": (
                AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT if mean_applied else 0.0
            ),
            "aggressivePeakWeight": (
                AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT if peak_applied else 0.0
            ),
            "meanSkillEligible": mean_skill,
            "peakSkillEligible": peak_skill,
            "deploymentSkill": {
                "trailMean": deployment_skill.get("trailMean"),
                "trailPeak": deployment_skill.get("trailPeak"),
            },
            "modelMae": replay.get("trailMeanMae"),
            "persistenceMae": replay.get("trailMeanPersistenceMae"),
            "peakModelMae": replay.get("trailPeakMae"),
            "peakPersistenceMae": replay.get("trailPeakPersistenceMae"),
        })
        if peak_candidate_available:
            trail["projectedPeak"] = round(peak_point, 1)
        if trail.get("rangeLow") is not None:
            trail["rangeLow"] = round(min(float(trail["rangeLow"]), mean_point), 1)
            trail["rangeHigh"] = round(max(float(trail["rangeHigh"]), mean_point), 1)
            trail["decisionUpperMean"] = round(max(
                float(trail.get("upperMean") or 0), mean_point
            ), 1)
        shadow_upper_delta = (
            num((analogue or {}).get("trailPeakUpperDelta"))
            if peak_applied else None
        )
        shadow_upper = (
            forecast_anchor + shadow_upper_delta
            if shadow_upper_delta is not None else
            num((analogue or {}).get("trailPeakUpper90")) if peak_applied else None
        )
        if (shadow_upper_delta is None and shadow_upper is not None
                and model_anchor is not None):
            shadow_upper = forecast_anchor + (shadow_upper - model_anchor)
        if peak_candidate_available:
            trail["decisionPeakUpper"] = round(max(
                float(trail.get("peakUpper") or 0), peak_point,
                float(shadow_upper or 0),
            ), 1)
    updated["trail"] = trail

    # A strong, multi-sensor regime break is different from an ordinary level
    # match. For the first two samples only, route the point forecast through
    # completed prior clearing trajectories. The broader persistence envelope
    # is retained and expanded; it is never narrowed around this small sample.
    event_clock_matches = targets_match(
        air_window.get("forecastClock"), (rapid_event or {}).get("forecastClock")
    )
    if (rapid_event or {}).get("available") and not event_clock_matches:
        updated["rapidClearanceEvent"] = {
            **rapid_event,
            "usedForDecision": False,
            "pointOverlayWithheld": "Event-relative targets do not match the decision-clock horizon",
        }
    if (rapid_event or {}).get("available") and event_clock_matches:
        event_count = int(rapid_event.get("completedEventCount") or 0)
        event_anchor = num(rapid_event.get("currentRawAnchor"))
        event_arrival = num(rapid_event.get("arrivalPoint"))
        event_mean = num(rapid_event.get("trailMeanPoint"))
        event_peak = num(rapid_event.get("trailPeakPoint"))
        event_support = {
            "completedEventCount": event_count,
            "completedClusterCount": rapid_event.get("completedClusterCount"),
            "completedDistinctDays": rapid_event.get("completedDistinctDays"),
            "minimumCompletedEvents": rapid_event.get("minimumCompletedEvents"),
            "minimumDistinctDays": rapid_event.get("minimumDistinctDays"),
            "eventEpoch": rapid_event.get("eventEpoch"),
            "eventAgeMinutes": rapid_event.get("eventAgeMinutes"),
            "deltaWeight": rapid_event.get("deltaWeight"),
            "priorEventEpochs": rapid_event.get("priorEventEpochs"),
            "arrivalScenarioLow": rapid_event.get("arrivalScenarioLow"),
            "arrivalScenarioHigh": rapid_event.get("arrivalScenarioHigh"),
            "trailMeanScenarioLow": rapid_event.get("trailMeanScenarioLow"),
            "trailMeanScenarioHigh": rapid_event.get("trailMeanScenarioHigh"),
            "trailPeakScenarioHigh": rapid_event.get("trailPeakScenarioHigh"),
        }
        if event_anchor is not None and event_arrival is not None:
            arrival = dict(updated.get("arrival") or {})
            scenario_low = num(rapid_event.get("arrivalScenarioLow"))
            scenario_high = num(rapid_event.get("arrivalScenarioHigh"))
            arrival.update({
                "available": True,
                "baselinePoint": round(event_anchor, 1),
                "modelFeatureAnchor": round(model_anchor, 1),
                "persistenceAnchorRole": "latest_raw_sensor_reading",
                "persistenceAnchorEpoch": air_window.get("analysisBucketEndEpoch"),
                "point": round(event_arrival, 1),
                "pointRole": "aggressive_rapid_clearance_event",
                "forecastState": "aggressive_event_conditioned_point_and_range",
                "pointApproximate": True,
                "usedForDecision": True,
                "validated": False,
                "method": rapid_event.get("method"),
                "headline": "Strong clearing event pattern active",
                "confidence": "Low · event-conditioned experimental model",
                "rangeRole": "combined_persistence_and_event_scenario_display_envelope",
                "eventScenarioRangeLow": scenario_low,
                "eventScenarioRangeHigh": scenario_high,
                "rapidEventSupport": event_support,
            })
            existing_low = num(arrival.get("rangeLow"))
            existing_high = num(arrival.get("rangeHigh"))
            lows = [value for value in (existing_low, scenario_low, event_arrival)
                    if value is not None]
            highs = [value for value in (existing_high, scenario_high, event_arrival)
                     if value is not None]
            if lows and highs:
                arrival["rangeLow"] = round(max(0.0, min(lows)), 1)
                arrival["rangeHigh"] = round(max(highs), 1)
                arrival["decisionUpper"] = round(max(
                    value for value in (
                        num(arrival.get("decisionUpper")),
                        num(arrival.get("upper90")), scenario_high, event_arrival,
                    ) if value is not None
                ), 1)
            updated["arrival"] = arrival

        if event_anchor is not None and event_mean is not None and event_peak is not None:
            trail = dict(updated.get("trail") or {})
            scenario_low = num(rapid_event.get("trailMeanScenarioLow"))
            scenario_high = num(rapid_event.get("trailMeanScenarioHigh"))
            scenario_peak = num(rapid_event.get("trailPeakScenarioHigh"))
            trail.update({
                "available": True,
                "baselinePoint": round(event_anchor, 1),
                "modelFeatureAnchor": round(model_anchor, 1),
                "persistenceAnchorRole": "latest_raw_sensor_reading",
                "persistenceAnchorEpoch": air_window.get("analysisBucketEndEpoch"),
                "point": round(event_mean, 1),
                "projectedPeak": round(max(event_mean, event_peak), 1),
                "pointRole": "aggressive_rapid_clearance_event",
                "forecastState": "aggressive_event_conditioned_point_and_range",
                "pointApproximate": True,
                "usedForDecision": True,
                "validated": False,
                "method": rapid_event.get("method"),
                "headline": "Strong clearing event pattern active",
                "confidence": "Low · event-conditioned experimental model",
                "rangeRole": "combined_persistence_and_event_scenario_display_envelope",
                "eventScenarioMeanLow": scenario_low,
                "eventScenarioMeanHigh": scenario_high,
                "eventScenarioPeakHigh": scenario_peak,
                "rapidEventSupport": event_support,
            })
            lows = [value for value in (num(trail.get("rangeLow")), scenario_low, event_mean)
                    if value is not None]
            highs = [value for value in (num(trail.get("rangeHigh")), scenario_high, event_mean)
                     if value is not None]
            if lows and highs:
                trail["rangeLow"] = round(max(0.0, min(lows)), 1)
                trail["rangeHigh"] = round(max(highs), 1)
                trail["decisionUpperMean"] = round(max(
                    value for value in (
                        num(trail.get("decisionUpperMean")),
                        num(trail.get("upperMean")), scenario_high, event_mean,
                    ) if value is not None
                ), 1)
                trail["decisionPeakUpper"] = round(max(
                    value for value in (
                        num(trail.get("decisionPeakUpper")),
                        num(trail.get("peakUpper")), scenario_peak, event_peak,
                    ) if value is not None
                ), 1)
            updated["trail"] = trail
        updated["rapidClearanceEvent"] = rapid_event
    # Existing candidates only: no extra fits and no backfilled intermediates.
    # The caller may subsequently refresh persistence, so name this stage.
    reported_skill = ((analogue or {}).get("backtest") or {}).get("deploymentSkill") or {}
    selection = {}
    for target, candidate, delta, weight, route, selected, applied in (
        ("arrival", arrival_shadow, arrival_delta, AGGRESSIVE_ARRIVAL_ANALOGUE_WEIGHT,
         "arrival", "arrival", bool(arrival_shadow is not None and arrival_skill)),
        ("trailMean", trail_shadow, trail_delta, AGGRESSIVE_TRAIL_MEAN_RIDGE_WEIGHT,
         "trailMean", "trail", mean_applied),
        ("trailPeak", peak_shadow, peak_delta, AGGRESSIVE_TRAIL_PEAK_ANALOGUE_WEIGHT,
         "trailPeak", "trail", peak_applied),
    ):
        gate = reported_skill.get(route) or {}
        raw_delta = delta if delta is not None else candidate-model_anchor if candidate is not None else None
        candidate_available = bool(candidate is not None and math.isfinite(candidate)
                                   and raw_delta is not None and math.isfinite(raw_delta))
        candidate_state = ("available" if candidate_available else "missing" if candidate is None else "nonfinite")
        gate_outcome = gate.get("gateOutcome") or (
            "not_evaluated" if not gate or "eligible" not in gate else
            "unavailable" if gate["eligible"] is None else "passed" if gate["eligible"] else "failed"
        )
        blocked = []
        if not clock_matches:
            blocked.append("target_clock_mismatch")
        if not candidate_available:
            blocked.append("candidate_unavailable")
        if target == "trailPeak" and not mean_candidate_available:
            blocked.append("trail_mean_candidate_unavailable")
        if gate_outcome == "unavailable":
            blocked.append("deployment_evidence_unavailable")
        elif gate_outcome == "not_evaluated":
            blocked.append("deployment_gate_not_evaluated")
        elif gate_outcome == "failed":
            blocked.append("deployment_gate_not_eligible")
        point_key = "projectedPeak" if target == "trailPeak" else "point"
        selected_value = (updated.get(selected) or {}).get(point_key)
        selected_role = (updated.get(selected) or {}).get("pointRole")
        if target == "trailPeak":
            selected_role = (
                None if selected_value is None else
                "aggressive_rapid_clearance_event_peak" if selected_role == "aggressive_rapid_clearance_event" else
                "aggressive_local_analogue_peak" if peak_applied else
                "mean_or_persistence_floor_not_peak_model" if peak_candidate_available and mean_applied else
                "retained_existing_peak_estimate"
            )
        selection[target] = {
            "rawCandidatePoint": candidate, "rawCandidateDelta": raw_delta,
            "dampedCandidatePoint": (max(0.0, forecast_anchor + weight*raw_delta)
                                     if candidate_available else None),
            "gateEligible": bool(gate.get("eligible")), "gateOutcome": gate_outcome,
            "candidateAvailable": candidate_available, "candidateState": candidate_state,
            "localComponentApplied": applied, "blockedReasons": blocked,
            "selectionOutcome": ("applied" if applied else "not_evaluated" if not clock_matches else
                                 "unavailable" if not candidate_available or gate_outcome == "unavailable" else
                                 "failed" if gate_outcome == "failed" else "not_evaluated"),
            "gateChecks": gate.get("gateChecks", {}),
            "prospectiveGateChecks": gate.get("prospectiveGateChecks", {}),
            "failedPredicates": list(gate.get("failedPredicates", [])),
            "unavailablePredicates": list(gate.get("unavailablePredicates", [])),
            "notEvaluatedPredicates": list(gate.get("notEvaluatedPredicates", [])),
            "prospectiveFailedPredicates": list(gate.get("prospectiveFailedPredicates", [])),
            "prospectiveUnavailablePredicates": list(gate.get("prospectiveUnavailablePredicates", [])),
            "prospectiveNotEvaluatedPredicates": list(gate.get("prospectiveNotEvaluatedPredicates", [])),
            "selectedField": point_key,
            "pointBeforeOverlay": (air_window.get(selected) or {}).get(point_key),
            "pointAfterOverlay": selected_value,
            "pointRoleAfterOverlay": selected_role,
        }
    updated["nearTermSelectionDiagnostics"] = {
        "version": "near_gate_diagnostics_v1",
        "stage": "after_aggressive_overlay_before_persistence_refresh",
        "targetClockMatches": clock_matches, "forecastAnchor": forecast_anchor,
        "forecastClock": air_window.get("forecastClock"), "candidateForecastClock": (analogue or {}).get("forecastClock"),
        "modelFeatureAnchor": model_anchor, "targets": selection,
    }
    updated["forecastPolicy"] = "aggressive_experimental"
    return json_safe(updated)


def refresh_near_term_persistence(rows, washout_signal, air_window, issue_epoch):
    """Refresh only fallback references, never silently rebase trained models.

    Quantiles are recalculated from historical forecasts using the same recent
    raw reference and exact target clock, not shifted from the old error band.
    """
    if not air_window.get("available"):
        return air_window
    eligible = [name for name in ("arrival", "trail")
                if (air_window.get(name) or {}).get("pointRole") == "persistence_anchor"]
    if not eligible:
        return air_window
    refreshed = air_window_analysis(rows, washout_signal, issue_epoch, fresh_reference=True)
    updated = dict(air_window)
    for name in eligible:
        value = refreshed.get(name) or {}
        if (value.get("available") and (value.get("freshnessAdjustment") or {}).get("applied")
                and targets_match(air_window.get("forecastClock"), refreshed.get("forecastClock"))):
            value = dict(value)
            value["method"] = "Recent-sensor persistence with matched historical errors"
            value["modelVersion"] = "fresh_sensor_persistence_v1"
            old = air_window.get(name) or {}
            # The old peak screen used a closed-median mean floor. It must not
            # be relabelled as screened against the newly refreshed reference.
            if name == "trail" and old.get("peakApproximate"):
                value["closedReferencePeakDiagnostic"] = {
                    "point": old.get("projectedPeak"), "usedForDecision": False,
                    "reason": "Original peak screen used the older closed-median reference",
                }
            updated[name] = value
    return updated


def separate_observation_from_prediction(air_window):
    """An observed clearing/rising regime is not a forecast of its continuation.

    Apply only after all point-model overlays. Never change numbers or archived
    clocks here; dashboard and API must describe the estimator actually used.
    """
    updated = dict(air_window)
    state = updated.get("state")
    observed_label = (
        "Recent PM2.5 medians at or below 35 µg/m³"
        if state == "confirmed" else updated.get("label")
    )
    updated["observedMovement"] = {
        "role": "observed_recent_movement_not_future_direction",
        "state": state,
        "label": observed_label,
        "observedThroughEpoch": updated.get("analysisBucketEndEpoch"),
        "eventLabel": observed_label if state in (
            "forming", "confirmed", "rebound", "rapid_improvement", "fast_rise"
        ) and updated.get("available") else None,
    }
    arrival = dict(updated.get("arrival") or {})
    if arrival.get("available"):
        if arrival.get("pointRole") == "persistence_anchor":
            arrival["headline"] = (
                "Recent-sensor baseline · direction unresolved"
                if arrival.get("freshnessAdjustment", {}).get("applied")
                else "Persistence baseline · direction unresolved"
            )
        elif (arrival.get("rainModel") or {}).get("applied"):
            arrival["headline"] = "Rain-aware experimental 90-minute estimate"
        else:
            arrival["headline"] = "Experimental 90-minute estimate"
    updated["arrival"] = arrival
    return updated


def air_window_analysis(rows, washout_signal, issue_epoch=None, fresh_reference=False):
    """Nowcast arrival and the full +90-to-+210-minute trail exposure interval."""
    frame = sensor_frame(rows).copy()
    complete = frame.dropna(subset=["pm02", "pm10"])
    if len(complete) < 5:
        return {
            "available": False, "state": "collecting", "label": "Collecting particle history…",
            "details": "About one hour of readings is needed.", "recheckMinutes": 15,
            "arrival": {"available": False, "headline": "Not enough local history"},
            "trail": {"available": False, "headline": "Not enough local history"},
        }
    # Preserve missing buckets so positional shifts retain real elapsed time.
    frame = frame.loc[:complete.index[-1]]
    issue_epoch = int(issue_epoch or rows[-1]["epoch"])
    closed_frame = fully_closed_sensor_frame(frame, issue_epoch)
    closed_pm = closed_frame["pm02"].dropna()
    expected_anchor_epoch = issue_epoch // (AIR_BUCKET_MINUTES*60) * (AIR_BUCKET_MINUTES*60)
    fresh_reading = 0 <= issue_epoch-int(rows[-1]["epoch"]) <= SENSOR_DEGRADED_SECONDS
    if (closed_pm.empty or not fresh_reading
            or int(closed_pm.index[-1].timestamp()) != expected_anchor_epoch):
        return {
            "available": False, "state": "collecting" if fresh_reading else "stale",
            "label": "Collecting a complete current forecast bucket…" if fresh_reading else "Sensor reading is stale",
            "details": "The latest closed 15-minute bucket must have sufficient observations; gaps are not filled.",
            "recheckMinutes": 15,
            "arrival": {"available": False, "headline": "Forecast anchor collecting"},
            "trail": {"available": False, "headline": "Forecast anchor collecting"},
        }

    pm = frame["pm02"]
    pm10 = frame["pm10"]
    current = float(pm.iloc[-1])
    persistence_current = float(closed_pm.iloc[-1])
    persistence_epoch = int(closed_pm.index[-1].timestamp())
    clock = ForecastClock(issue_epoch, persistence_epoch)
    current_fast = num(rows[-1]["pm02"])
    if current_fast is None:
        current_fast = current
    change30_series = pm.diff(2)
    change60_series = pm.diff(4)
    pm10_fraction30 = pm10.pct_change(2, fill_method=None)
    pm10_fraction60 = pm10.pct_change(4, fill_method=None)
    ema_fast = segmented_ewm(pm, 3)
    ema_slow = segmented_ewm(pm, 8)
    ema_gap = ema_fast - ema_slow
    valid_particles = pm.notna() & pm10.notna()
    coarse = (pm10 - pm).clip(lower=0)
    fine_share = (100 * pm / pm10).clip(lower=0, upper=100)
    coarse_smooth = segmented_ewm(coarse, 3).where(valid_particles)
    fine_share_smooth = segmented_ewm(fine_share, 3).where(valid_particles)
    coarse_change30 = coarse_smooth.diff(2)
    fine_share_change30 = fine_share_smooth.diff(2)
    mix_rebound_flags = (
        fine_share_change30.ge(MIX_FINE_SHARE_CHANGE)
        & coarse_change30.le(-MIX_COARSE_CHANGE)
        & fine_share_smooth.ge(85.0)
    )
    standard_fast_rise_history = (
        change30_series.ge(FAST_RISE_PM25_CHANGE)
        & pm10_fraction30.ge(FAST_RISE_PM10_FRACTION)
        & ema_fast.gt(ema_slow)
    )
    mix_fast_rise_history = (
        change30_series.ge(MIX_FAST_RISE_PM25_CHANGE)
        & pm10_fraction30.ge(MIX_FAST_RISE_PM10_FRACTION)
        & ema_fast.gt(ema_slow)
        & mix_rebound_flags
    )
    fast_rise_history_flags = standard_fast_rise_history | mix_fast_rise_history
    fast_rise_flags = standard_fast_rise_history | mix_fast_rise_history
    washout_history_flags = (
        change30_series.le(-WASHOUT_PM25_DROP)
        & pm.pct_change(2, fill_method=None).le(-WASHOUT_PM25_FRACTION)
        & pm10_fraction30.le(-WASHOUT_PM10_FRACTION)
    )
    change30 = float(change30_series.iloc[-1])
    change60 = float(change60_series.iloc[-1])
    momentum_available = bool(
        math.isfinite(change30)
        and pd.notna(pm10_fraction30.iloc[-1])
    )
    mix_rebound_support = bool(mix_rebound_flags.iloc[-1])
    mix_washout_support = bool(
        fine_share_change30.iloc[-1] <= -MIX_FINE_SHARE_CHANGE
        and coarse_change30.iloc[-1] >= MIX_COARSE_CHANGE
    )
    fast_rise = bool(momentum_available and (
        fast_rise_flags.iloc[-1]
        or (fast_rise_flags.iloc[-3:].any() and change30 >= 5.0 and ema_gap.iloc[-1] > 0)
    ))

    event_recent = washout_signal.get("state") in ("active", "recent", "forming")
    forming_event = washout_signal.get("state") == "forming"
    minimum_since_event = current
    if event_recent and washout_signal.get("detectedEpoch"):
        event_time = pd.Timestamp(
            washout_signal["detectedEpoch"], unit="s", tz="UTC"
        ).tz_convert("Asia/Kuala_Lumpur")
        since_event = pm.loc[pm.index >= event_time - pd.Timedelta(minutes=AIR_BUCKET_MINUTES)]
        if not since_event.empty:
            minimum_since_event = float(since_event.min())

    rebound = event_recent and (
        change30 >= 5.0
        or (current - minimum_since_event >= 10.0 and current >= minimum_since_event * 1.20)
    )
    holding_lower = bool((pm.iloc[-2:] <= LOWER_PARTICLE_MARKER).all())
    rapid_improvement = event_recent or (change30 <= -20.0 and ema_gap.iloc[-1] <= -8.0)
    complete_60 = bool(
        len(frame) >= 5
        and frame[["pm02", "pm10"]].iloc[-5:].notna().to_numpy().all()
    )
    sustained_improvement = bool(
        complete_60
        and math.isfinite(change60)
        and change60 <= -8.0
        and pd.notna(pm10_fraction60.iloc[-1])
        and pm10_fraction60.iloc[-1] <= -0.06
        and pd.notna(ema_gap.iloc[-1])
        and ema_gap.iloc[-1] <= -2.0
    )

    if not momentum_available:
        state = "collecting"
        label = "Recent particle movement unavailable"
        recheck = 15
    elif fast_rise:
        state = "fast_rise"
        label = ("Fast PM2.5 rebound detected" if rebound
                 else "Fast PM2.5 rise detected")
        recheck = 15
    elif rebound:
        state = "rebound"
        label = "Particle rebound may be starting"
        recheck = 15
    elif holding_lower:
        state = "confirmed"
        label = "Sustained PM2.5 reduction detected"
        recheck = 30
    elif forming_event:
        state = "forming"
        label = washout_signal.get("label") or "Dry clearing forming"
        recheck = (
            9 if (washout_signal.get("metrics") or {}).get("detectionMode") == "joint_shock"
            else 15
        )
    elif rapid_improvement and current <= 55.0:
        state = "forming"
        label = "PM2.5 reduction forming"
        recheck = 15
    elif rapid_improvement:
        state = "rapid_improvement"
        label = "Rapid PM2.5 reduction detected"
        recheck = 15
    elif change30 >= 8.0:
        state = "rising"
        label = "PM2.5 rising"
        recheck = 15
    elif change30 <= -8.0 or sustained_improvement:
        state = "improving"
        label = "PM2.5 falling"
        recheck = 15
    elif current > 55.0:
        state = "stable_high"
        label = "PM2.5 holding near its recent level"
        recheck = 30
    else:
        state = "steady"
        label = "No strong particle shift"
        recheck = 30

    particle_mix = {
        "available": True,
        "relevant": False,
        "state": "no_interpretive_signal",
        "label": "",
        "details": "",
        "fineShare": round(float(fine_share_smooth.iloc[-1])),
        "fineShareChange30": round(float(fine_share_change30.iloc[-1]), 1),
        "coarseParticles": round(float(coarse_smooth.iloc[-1]), 1),
        "coarseChange30": round(float(coarse_change30.iloc[-1]), 1),
    }
    if (momentum_available and mix_rebound_support
            and state in ("fast_rise", "rebound", "rising")):
        particle_mix.update({
            "relevant": True,
            "state": "fine_haze_returning",
            "label": "Fine haze returning",
            "details": f'fine share ↑{fine_share_change30.iloc[-1]:.1f} points in 30 min',
        })
    elif (momentum_available and mix_washout_support
          and state in ("rapid_improvement", "forming", "improving")):
        particle_mix.update({
            "relevant": True,
            "state": "fine_haze_washing_out",
            "label": "Fine-particle share falling",
            "details": f'fine share ↓{abs(fine_share_change30.iloc[-1]):.1f} points in 30 min',
        })
    elif (momentum_available
          and coarse_smooth.iloc[-1] >= MIX_COARSE_ELEVATED
          and fine_share_smooth.iloc[-1] <= 85.0):
        particle_mix.update({
            "relevant": True,
            "state": "coarse_mix_elevated",
            "label": "Coarse-particle mix elevated",
            "details": f'{coarse_smooth.iloc[-1]:.1f} µg/m³ coarse fraction',
        })

    # Arrival: persistence was the strongest general point forecast in the
    # held-out-day test.  Outcomes use only fully closed buckets; the partial
    # live bucket remains available above for the responsive nowcast.
    outcome_pm = closed_frame["pm02"]
    origin_reference = outcome_pm.copy()
    closed_reference = persistence_current
    reference_role = "latest_closed_15_minute_bucket_median"
    reference_epoch = persistence_epoch
    freshness = {"applied": False, "version": forecast_freshness.VERSION}
    if fresh_reference:
        points, stamps, counts = forecast_freshness.references(
            rows, outcome_pm.index.asi8 // 10**9, clock.lag_seconds, issue_epoch
        )
        # Keep the strict closed-feature mask: this never fills a collection gap.
        origin_reference = pd.Series(points, index=outcome_pm.index).where(
            outcome_pm.notna()
        ).fillna(outcome_pm)
        if len(points) and math.isfinite(points[-1]):
            persistence_current = float(points[-1])
            reference_epoch = int(stamps[-1])
            reference_role = "trailing_5_minute_raw_sensor_median"
            freshness.update({
                "applied": True, "amountUgM3": round(persistence_current-closed_reference, 2),
                "closedReferencePm25UgM3": round(closed_reference, 1),
                "referenceEpoch": reference_epoch, "referenceCount": int(counts[-1]),
                "windowSeconds": 300, "featureAnchorEpoch": persistence_epoch,
                "policy": "reactive_reference_refresh_not_advance_event_prediction",
                "prospectivelyValidated": False,
            })
    outcomes = target_series(outcome_pm, clock)
    future = outcomes["arrival"]
    arrival_origins = pd.DataFrame({
        "origin": origin_reference, "future": future
    }).dropna()
    arrival_origins["error"] = arrival_origins["future"] - arrival_origins["origin"]
    arrival_matched = independent_level_matches(
        arrival_origins, persistence_current, ARRIVAL_NEIGHBORS,
        INTERVAL_MATCH_SEPARATION_MINUTES,
    )
    arrival_errors = arrival_matched["error"]
    arrival_available = len(arrival_errors) >= ARRIVAL_NEIGHBORS
    arrival_headlines = {
        "fast_rise": "PM2.5 rising now; +90 forecast remains persistence",
        "rebound": "PM2.5 rebounding now; +90 forecast remains persistence",
        "confirmed": "Lower PM2.5 observed; +90 forecast remains persistence",
        "forming": "PM2.5 reduction observed; +90 min remains uncertain",
        "rapid_improvement": "PM2.5 falling now; +90 forecast remains persistence",
        "rising": "PM2.5 rising now; +90 forecast remains persistence",
        "stable_high": "Current PM2.5 may persist",
        "improving": "PM2.5 falling now; +90 forecast remains persistence",
        "steady": "No strong 90-minute direction",
        "collecting": "Current PM2.5 used as the persistence reference",
    }
    arrival = {
        "available": arrival_available,
        "headline": arrival_headlines.get(state, "No strong 90-minute direction"),
        "minutes": ARRIVAL_MINUTES,
        "point": round(persistence_current, 1),
        "baselinePoint": round(persistence_current, 1),
        "pointRole": "persistence_anchor",
        "persistenceAnchorRole": reference_role,
        "persistenceAnchorEpoch": reference_epoch,
        "freshnessAdjustment": freshness,
        "forecastClock": clock.metadata(),
        "forecastIssuedEpoch": issue_epoch,
        "forecastState": "range_only",
        "method": "Persistence baseline with similar-level empirical errors",
        "sampleCount": int(len(arrival_origins)),
        "originCount": int(len(arrival_origins)),
        "independentOriginCount": independent_origin_count(
            arrival_origins.index, ARRIVAL_MINUTES
        ),
        "matchedCount": int(len(arrival_matched)),
        "matchedIndependentOriginCount": independent_origin_count(
            arrival_matched.index, ARRIVAL_MINUTES
        ),
        "matchedDistinctDays": len({
            timestamp.date() for timestamp in arrival_matched.index
        }),
        "matchedOriginsIndependent": False,
        "matchedOriginsTimeSeparatedMinutes": INTERVAL_MATCH_SEPARATION_MINUTES,
        "distinctDays": len({timestamp.date() for timestamp in arrival_origins.index}),
        "minimumRequired": ARRIVAL_NEIGHBORS,
        "confidence": "Low · limited local history",
        "calibrationState": "collecting",
        "expectedEpoch": clock.target_start_epoch,
    }
    if arrival_available:
        arrival_low = empirical_quantile(arrival_errors, 0.10, upper=False)
        arrival_high = empirical_quantile(arrival_errors, 0.90, upper=True)
        raw_arrival_low = max(0.0, persistence_current + arrival_low)
        raw_arrival_high = max(0.0, persistence_current + arrival_high)
        arrival.update({
            "rawRangeLow": round(raw_arrival_low, 1),
            "rawRangeHigh": round(raw_arrival_high, 1),
            "rawUpper90": round(raw_arrival_high, 1),
            "rangeLow": round(min(persistence_current, raw_arrival_low), 1),
            "rangeHigh": round(max(persistence_current, raw_arrival_high), 1),
            # Compatibility/display envelope bound. This may be wider than the
            # empirical q90 above and must never be labelled as that quantile.
            "upper90": round(max(persistence_current, raw_arrival_high), 1),
            "upperTargetCoverage": 0.90,
            "finiteSampleRankCoverage": round(
                finite_upper_rank_coverage(len(arrival_errors), 0.90), 3
            ),
            "finiteSampleUpper": True,
            "matchedCount": int(len(arrival_matched)),
        })

    # Trail target: mean and maximum across the two-hour ride after 90-minute travel.
    trail_origins = pd.DataFrame({
        "origin": origin_reference,
        "mean": outcomes["mean"],
        "peak": outcomes["peak"],
        "fastRise": fast_rise_history_flags.reindex(outcome_pm.index),
        "washout": washout_history_flags.reindex(outcome_pm.index),
    }).dropna()
    fast_rise_history = event_representatives(trail_origins, "fastRise")
    washout_history = event_representatives(trail_origins, "washout")
    fast_rise_days = len({timestamp.date() for timestamp in fast_rise_history.index})
    washout_days = len({timestamp.date() for timestamp in washout_history.index})
    clearance_mode = (washout_signal.get("metrics") or {}).get("detectionMode")
    rapid_clearance_regime = bool(
        state == "rapid_improvement"
        or (
            state == "forming"
            and (
                clearance_mode in ("rapid_joint", "rapid_and_sustained")
                or washout_signal.get("state") in ("active", "recent")
            )
        )
    )
    if (fast_rise and len(fast_rise_history) >= FAST_RISE_HISTORY_MIN
            and fast_rise_days >= EVENT_HISTORY_MIN_DAYS):
        trail_matched = fast_rise_history
        trail_method = "Persistence with prior fast-rise event errors"
        trail_confidence = "Low · few fast-rise episodes"
        required_trail_samples = FAST_RISE_HISTORY_MIN
        matched_event_count = int(len(fast_rise_history))
    elif (rapid_clearance_regime
          and len(washout_history) >= WASHOUT_HISTORY_MIN
          and washout_days >= EVENT_HISTORY_MIN_DAYS):
        trail_matched = washout_history
        trail_method = "Persistence with prior joint-washout event errors"
        trail_confidence = "Low · few washout episodes"
        required_trail_samples = WASHOUT_HISTORY_MIN
        matched_event_count = int(len(washout_history))
    else:
        trail_matched = independent_level_matches(
            trail_origins, persistence_current, ARRIVAL_NEIGHBORS,
            INTERVAL_MATCH_SEPARATION_MINUTES,
        )
        trail_method = "Persistence with similar-level trail errors"
        trail_confidence = "Low · limited local history"
        required_trail_samples = ARRIVAL_NEIGHBORS
        matched_event_count = None

    trail_mean_errors = trail_matched["mean"] - trail_matched["origin"]
    trail_peak_errors = trail_matched["peak"] - trail_matched["origin"]
    trail_available = len(trail_matched) >= required_trail_samples
    trail_point = persistence_current
    trail_headline = (
        "PM2.5 rising now; trail mean remains persistence" if fast_rise
        else "PM2.5 rising now; trail mean remains persistence"
        if state in ("rising", "rebound")
        else "Lower PM2.5 observed; trail mean remains persistence"
        if state == "confirmed"
        else "Observed clearing; +90 to +210 min remains uncertain" if state == "forming"
        else "Rapid PM2.5 change; +90 to +210 min remains uncertain" if state == "rapid_improvement"
        else "PM2.5 falling now; trail mean remains persistence"
        if state == "improving"
        else "Current PM2.5 used as the persistence reference"
    )
    trail = {
        "available": trail_available,
        "headline": trail_headline,
        "point": round(trail_point, 1),
        "baselinePoint": round(persistence_current, 1),
        "pointRole": "persistence_anchor",
        "persistenceAnchorRole": reference_role,
        "persistenceAnchorEpoch": reference_epoch,
        "freshnessAdjustment": freshness,
        "forecastState": "range_only",
        "startMinutes": ARRIVAL_MINUTES,
        "endMinutes": ARRIVAL_MINUTES + TRAIL_MINUTES,
        "forecastClock": clock.metadata(),
        "forecastIssuedEpoch": issue_epoch,
        "startEpoch": clock.target_start_epoch,
        "endEpoch": clock.target_end_epoch,
        "method": trail_method,
        "confidence": trail_confidence,
        "sampleCount": int(len(trail_matched)),
        "originCount": int(len(trail_origins)),
        "independentOriginCount": independent_origin_count(
            trail_origins.index, ARRIVAL_MINUTES + TRAIL_MINUTES
        ),
        "matchedCount": int(len(trail_matched)),
        "matchedIndependentOriginCount": independent_origin_count(
            trail_matched.index, ARRIVAL_MINUTES + TRAIL_MINUTES
        ),
        "matchedDistinctDays": len({
            timestamp.date() for timestamp in trail_matched.index
        }),
        "matchedEventCount": matched_event_count,
        "fastRiseEventCount": int(len(fast_rise_history)),
        "washoutEventCount": int(len(washout_history)),
        "fastRiseEventDays": fast_rise_days,
        "washoutEventDays": washout_days,
        "matchedOriginsIndependent": False,
        "matchedOriginsTimeSeparatedMinutes": INTERVAL_MATCH_SEPARATION_MINUTES,
        "distinctDays": len({timestamp.date() for timestamp in trail_origins.index}),
        "minimumRequired": int(required_trail_samples),
        "adjustment": 0.0,
        "calibrationState": "collecting",
    }
    if trail_available:
        mean_low_error = empirical_quantile(trail_mean_errors, 0.10, upper=False)
        mean_high_error = empirical_quantile(trail_mean_errors, 0.90, upper=True)
        peak_high_error = empirical_quantile(trail_peak_errors, 0.90, upper=True)
        empirical_low = persistence_current + mean_low_error
        empirical_high = persistence_current + mean_high_error
        empirical_peak_high = max(0.0, persistence_current + peak_high_error)
        trail.update({
            "rawRangeLow": round(max(0.0, empirical_low), 1),
            "rawRangeHigh": round(max(0.0, empirical_high), 1),
            "rawUpperMean90": round(max(0.0, empirical_high), 1),
            "rawPeakUpper90": round(empirical_peak_high, 1),
            "rangeLow": round(max(0.0, min(trail_point, empirical_low)), 1),
            "rangeHigh": round(max(trail_point, empirical_high), 1),
            "upperMean": round(max(trail_point, empirical_high), 1),
            "peakUpper": round(max(persistence_current, empirical_peak_high), 1),
            "upperTargetCoverage": 0.90,
            "finiteSampleRankCoverage": round(
                finite_upper_rank_coverage(len(trail_matched), 0.90), 3
            ),
            "finiteSampleUpper": True,
        })

    return json_safe({
        "available": True,
        "state": state,
        "label": label,
        "details": (
            f'30-minute PM2.5 change {change30:+.1f} µg/m³'
            if momentum_available
            else "A complete 30-minute comparison is collecting after a data gap."
        ),
        "current15": round(closed_reference, 1),
        "currentPartial15": round(current, 1),
        "currentFast": round(current_fast, 1),
        "forecastClock": clock.metadata(),
        "closedBucketEndEpoch": persistence_epoch,
        # The right-labelled live bucket may end after the latest raw sample;
        # never expose that label as if it were a future observation time.
        "analysisBucketEndEpoch": min(
            int(rows[-1]["epoch"]), int(pm.index[-1].timestamp())
        ),
        "analysisBucketMinutes": AIR_BUCKET_MINUTES,
        "change30": round(change30, 1),
        "change60": round(change60, 1),
        "sustainedImprovement": sustained_improvement,
        "fastRise": fast_rise,
        "fastSlowGap": round(float(ema_gap.iloc[-1]), 1),
        "pm10ChangePct30": round(100 * float(pm10_fraction30.iloc[-1]), 1),
        "particleMix": particle_mix,
        "minimumSinceEvent": round(minimum_since_event, 1) if event_recent else None,
        "recheckMinutes": recheck,
        "arrival": arrival,
        "trail": trail,
    })


def empirical_ride_window_baseline(
        frame, start_epoch, end_epoch, current, origin_epoch):
    """Persistence forecast with horizon-specific local error bounds."""
    base = {
        "available": False,
        "point": round(current, 1),
        "method": "Sensor persistence with horizon-specific empirical errors",
        "confidence": "Insufficient",
    }
    if (frame.empty or start_epoch is None or end_epoch is None
            or origin_epoch is None):
        return base
    # The right-labelled frame can end up to 15 minutes after its newest raw
    # sample.  Horizon arithmetic must start at that raw observation, not at the
    # future bucket label.
    origin_time = pd.Timestamp(
        int(origin_epoch), unit="s", tz="UTC"
    ).tz_convert("Asia/Kuala_Lumpur")
    start_time = pd.Timestamp(int(start_epoch), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    end_time = pd.Timestamp(int(end_epoch), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    bucket = pd.Timedelta(minutes=AIR_BUCKET_MINUTES)
    start_offset = max(1, math.floor((start_time - origin_time) / bucket) + 1)
    end_offset = math.floor((end_time - origin_time) / bucket)
    if end_offset < start_offset:
        return base

    pm = frame["pm02"]
    future = pd.concat(
        [pm.shift(-offset) for offset in range(start_offset, end_offset + 1)],
        axis=1,
    )
    origins = pd.DataFrame({
        "origin": pm,
        "mean": future.mean(axis=1, skipna=False),
        "peak": future.max(axis=1, skipna=False),
    }).dropna()
    origins["meanError"] = origins["mean"] - origins["origin"]
    origins["peakError"] = origins["peak"] - origins["origin"]
    matched = independent_level_matches(
        origins, current, RIDE_WINDOW_MATCHES,
        RIDE_WINDOW_MATCH_SEPARATION_MINUTES,
    )
    base.update({
        "originEpoch": int(origin_epoch),
        "originCount": int(len(origins)),
        "independentOriginCount": independent_origin_count(
            origins.index, RIDE_WINDOW_MATCH_SEPARATION_MINUTES
        ),
        "matchedCount": int(len(matched)),
        "matchedDistinctDays": len({stamp.date() for stamp in matched.index}),
        "minimumRequired": RIDE_WINDOW_MIN_MATCHES,
        "leadHours": round(
            max(0.0, (start_time - origin_time).total_seconds() / 3600.0), 1
        ),
        "durationHours": round(
            max(0.0, (end_time - start_time).total_seconds() / 3600.0), 1
        ),
    })
    if len(matched) < RIDE_WINDOW_MIN_MATCHES:
        return base

    low_error = empirical_quantile(matched["meanError"], 0.10, upper=False)
    high_error = empirical_quantile(matched["meanError"], 0.90, upper=True)
    peak_error = empirical_quantile(matched["peakError"], 0.90, upper=True)
    if low_error is None or high_error is None or peak_error is None:
        return base
    # This middle-half band is descriptive matched-history context, not a
    # calibrated prediction interval.  It is useful for explaining uncertainty
    # without promoting the much wider tail envelope as the forecast itself.
    typical_low_error = float(matched["meanError"].quantile(0.25))
    typical_high_error = float(matched["meanError"].quantile(0.75))
    base.update({
        "available": True,
        "confidence": "Low · limited local history",
        "rawRangeLow": round(max(0.0, current + low_error), 1),
        "rawRangeHigh": round(max(0.0, current + high_error), 1),
        "typicalRangeLow": round(max(0.0, current + typical_low_error), 1),
        "typicalRangeHigh": round(max(0.0, current + typical_high_error), 1),
        "rangeLow": round(max(0.0, min(current, current + low_error)), 1),
        "rangeHigh": round(max(current, current + high_error), 1),
        "rawPeakUpper90": round(max(0.0, current + peak_error), 1),
        "peakUpper": round(max(current, current + peak_error), 1),
        "rangeRole": "persistence_anchor_containing_display_envelope",
        "upperTargetCoverage": 0.90,
        "finiteSampleRankCoverage": round(
            finite_upper_rank_coverage(len(matched), 0.90), 3
        ),
    })
    return base


def cams_validation_task(window_key, lead_hours, session_start, observation_time):
    """Map a live forecast to a prospectively scored operational task."""
    start_clock = (session_start.hour, session_start.minute)
    observation_minutes = observation_time.hour * 60 + observation_time.minute
    near_morning_check = abs(observation_minutes - (7 * 60 + 30)) <= 15
    near_afternoon_check = abs(observation_minutes - (12 * 60 + 30)) <= 15
    if (window_key == "morning" and start_clock == (9, 0)
            and near_morning_check and 1.0 <= lead_hours <= 2.25):
        return "morning"
    if (window_key == "afternoon" and start_clock == (14, 0)
            and near_morning_check and 6.0 <= lead_hours <= 7.25):
        return "afternoon_provisional"
    if (window_key == "afternoon" and start_clock == (14, 0)
            and near_afternoon_check and 1.0 <= lead_hours <= 2.25):
        return "afternoon_final"
    return None


def cams_window_candidate(
        payload, latest_epoch, current, start_epoch, end_epoch, window_key,
        sensor_epoch=None,
):
    """Anchor an exactly time-aligned CAMS change to the local sensor level."""
    base = {"available": False, "usedForDecision": False}
    model = model_frame(payload)
    if model.empty or "pm2_5" not in model:
        return base
    now = pd.Timestamp(int(latest_epoch), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    start = pd.Timestamp(int(start_epoch), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    end = pd.Timestamp(int(end_epoch), unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    lead_hours = max(0.0, (start - now).total_seconds() / 3600.0)
    validation_task = cams_validation_task(window_key, lead_hours, start, now)
    sensor_epoch = int(sensor_epoch or latest_epoch)
    sensor_time = pd.Timestamp(sensor_epoch, unit="s", tz="UTC").tz_convert(
        "Asia/Kuala_Lumpur"
    )
    model_current, target_points = aligned_model_projection(
        model, sensor_time, start, end
    )
    expected_target_points = int(
        (end - start).total_seconds() // (AIR_BUCKET_MINUTES * 60)
    )
    if model_current is None or len(target_points) != expected_target_points:
        return base
    raw_mean = float(target_points.mean())
    raw_peak = float(target_points.max())
    candidate_mean = max(
        0.0, current + AIR_QUALITY_DELTA_SHRINK * (raw_mean - model_current)
    )
    candidate_peak = max(
        candidate_mean,
        current + AIR_QUALITY_DELTA_SHRINK * (raw_peak - model_current),
    )
    fetched_epoch = int((payload or {}).get("fetchedEpoch") or 0)
    age_seconds = max(0, int(latest_epoch) - fetched_epoch) if fetched_epoch else None
    if age_seconds is not None and age_seconds > AIR_QUALITY_DEGRADED_SECONDS:
        return {
            "available": False,
            "usedForDecision": False,
            "stale": True,
            "source": (payload or {}).get("source"),
            "fetchedEpoch": fetched_epoch,
            "ageMinutes": round(age_seconds / 60),
        }
    return {
        "available": True,
        "usedForDecision": False,
        "source": (payload or {}).get("source"),
        "modelVersion": AIR_QUALITY_FORECAST_METHOD_VERSION,
        "sourceArchiveVersion": AIR_QUALITY_MODEL_VERSION,
        "fetchedEpoch": fetched_epoch or None,
        "ageMinutes": (
            None if not fetched_epoch
            else round(age_seconds / 60)
        ),
        "issuedAgeEligible": bool(
            age_seconds is not None
            and age_seconds <= AIR_QUALITY_MAX_DECISION_AGE_SECONDS
        ),
        "sourcePointCount": int(len(target_points)),
        "sourcePointRole": "15_minute_interval_centres_including_session_endpoint",
        "leadHours": round(lead_hours, 1),
        "validationTask": validation_task,
        "leadValidated": validation_task is not None,
        "rawModelCurrent": round(model_current, 1),
        "modelCurrentAtEpoch": sensor_epoch,
        "modelCurrentAgeMinutes": 0,
        "sensorAnchorAgeMinutes": round(
            max(0.0, (int(latest_epoch) - sensor_epoch) / 60.0), 1
        ),
        "interpolation": "linear_time",
        "rawModelMean": round(raw_mean, 1),
        "rawModelPeak": round(raw_peak, 1),
        "sensorAnchor": round(current, 1),
        "deltaShrink": AIR_QUALITY_DELTA_SHRINK,
        "mean": round(candidate_mean, 1),
        "peak": round(candidate_peak, 1),
    }


def ride_forecast_sensor_revision(rows):
    """Return a stable digest of the raw particle series used by the forecast.

    A row can be replaced at an existing epoch by AirGradient's
    ``INSERT OR REPLACE`` write.  First/last epochs and row count therefore do
    not identify the input data.  Hashing the already-loaded particle columns
    prevents a corrected row from receiving a projection cached for its former
    values.
    """
    digest = hashlib.blake2b(digest_size=16)
    for row in rows:
        keys = set(row.keys()) if hasattr(row, "keys") else set()
        values = (
            row["epoch"],
            row["pm02"] if "pm02" in keys else None,
            row["pm10"] if "pm10" in keys else None,
        )
        digest.update(repr(values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def ride_forecast_cache_key(
        rows, weather_windows, as_of_epoch, regime_break,
        include_runtime_status, payload):
    """Identify every causal input that can change the cached projection."""
    payload_epoch = int((payload or {}).get("fetchedEpoch") or 0)
    payload_revision = hashlib.blake2b(
        json.dumps(
            payload or {}, sort_keys=True, separators=(",", ":"),
            default=str,
        ).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return (
        int(rows[0]["epoch"]), int(rows[-1]["epoch"]), len(rows),
        ride_forecast_sensor_revision(rows),
        payload_epoch, payload_revision,
        int(as_of_epoch) // POLL_SECONDS, AGGRESSIVE_FORECAST_ENABLED,
        bool(regime_break), bool(include_runtime_status),
        tuple(
            (
                key,
                (weather_windows.get(key) or {}).get("startEpoch"),
                (weather_windows.get(key) or {}).get("endEpoch"),
            )
            for key in ("morning", "afternoon")
        ),
    )


def cached_ride_window_particle_forecast(
        rows, weather_windows, as_of_epoch=None, regime_break=False,
        include_runtime_status=True):
    """Return one cached or single-flight calculation for both ride windows."""
    as_of_epoch = int(as_of_epoch or rows[-1]["epoch"])
    payload = latest_air_quality_payload(as_of_epoch)
    cache_key = ride_forecast_cache_key(
        rows, weather_windows, as_of_epoch, regime_break,
        include_runtime_status, payload,
    )

    while True:
        with ride_forecast_lock:
            if ride_forecast_cache.get("cacheKey") == cache_key:
                return ride_forecast_cache["value"]
            pending = ride_forecast_flights.get(cache_key)
            if pending is None:
                pending = {"event": threading.Event()}
                ride_forecast_flights[cache_key] = pending
                break
        # Wait outside the lock.  If the owner failed, loop and let one waiter
        # become the new owner rather than caching an incomplete calculation.
        pending["event"].wait()
        if pending.get("succeeded"):
            return pending["value"]

    try:
        value = _compute_ride_window_particle_forecast(
            rows, weather_windows, as_of_epoch, regime_break,
            include_runtime_status, payload,
        )
    except BaseException:
        with ride_forecast_lock:
            ride_forecast_flights.pop(cache_key, None)
            pending["event"].set()
        raise

    with ride_forecast_lock:
        cached_issued = num(
            (ride_forecast_cache.get("value") or {}).get("forecastIssuedEpoch")
        )
        # A slower historical/replay request must not evict a newer live value.
        if cached_issued is None or as_of_epoch >= cached_issued:
            ride_forecast_cache["cacheKey"] = cache_key
            ride_forecast_cache["value"] = value
        pending["value"] = value
        pending["succeeded"] = True
        ride_forecast_flights.pop(cache_key, None)
        pending["event"].set()
    return value


def _compute_ride_window_particle_forecast(
        rows, weather_windows, as_of_epoch, regime_break,
        include_runtime_status, payload):
    """Compute both ride-window particle forecasts without cache locking."""
    payload_epoch = int((payload or {}).get("fetchedEpoch") or 0)
    frame = sensor_frame(rows)
    complete = frame.dropna(subset=["pm02"])
    if complete.empty:
        return {
            "available": False, "status": "collecting",
            "modelVersion": AIR_QUALITY_FORECAST_METHOD_VERSION,
            "forecastIssuedEpoch": int(as_of_epoch),
            "forecastIssuedRole": "cached_forecast_generation_reference_time",
            "windows": {},
        }
    frame = frame.loc[:complete.index[-1]]
    current = num(rows[-1]["pm02"])
    if current is None:
        current = float(frame["pm02"].iloc[-1])
    latest_epoch = as_of_epoch
    issued = air_quality_issued_evidence(rows, as_of_epoch)
    diagnostic = retrospective_air_quality_diagnostic(rows, payload)
    forecasts = {}
    for key in ("morning", "afternoon"):
        window = weather_windows.get(key) or {}
        start_epoch = window.get("startEpoch")
        end_epoch = window.get("endEpoch")
        baseline = empirical_ride_window_baseline(
            frame, start_epoch, end_epoch, current, int(rows[-1]["epoch"])
        )
        candidate = (
            cams_window_candidate(
                payload, latest_epoch, current, start_epoch, end_epoch, key,
                int(rows[-1]["epoch"]),
            )
            if payload and start_epoch is not None and end_epoch is not None
            else {"available": False, "usedForDecision": False}
        )
        task_evidence = (
            (issued.get("byTask") or {}).get(candidate.get("validationTask")) or {}
        )
        model_current = num(candidate.get("rawModelCurrent"))
        candidate_lead = num(candidate.get("leadHours"))
        # A live particle regime break invalidates a near-term CAMS change, but
        # should not erase a time-aligned next-day planning scenario.  Long-lead
        # estimates remain context-only until their exact task is validated.
        regime_suppressed = bool(
            regime_break
            and candidate.get("available")
            and candidate_lead is not None
            and candidate_lead <= 3.0
        )
        validated_candidate = bool(
            candidate.get("available")
            and candidate.get("issuedAgeEligible")
            and candidate.get("leadValidated")
            and task_evidence.get("supported")
            and not regime_suppressed
        )
        # An exact operational lead does not substitute for prospective skill.
        # Until its task passes the full issued-forecast gate, retain the CAMS
        # point as provisional context only and never use it for comparison.
        aggressive_candidate = False
        context_candidate = bool(
            AGGRESSIVE_FORECAST_ENABLED
            and candidate.get("available")
            and candidate.get("issuedAgeEligible")
            and not validated_candidate
            and not regime_suppressed
        )
        use_candidate = validated_candidate
        display_candidate = use_candidate or context_candidate
        candidate = dict(candidate)
        candidate["usedForDecision"] = use_candidate
        candidate["usedForDisplay"] = display_candidate
        candidate["suppressedByRegimeBreak"] = regime_suppressed
        candidate["regimeSuppressionHorizonHours"] = 3.0
        candidate["sensorModelLevelDifference"] = (
            None if model_current is None else round(abs(model_current - current), 1)
        )
        candidate["sensorModelLevelDifferenceRole"] = (
            "diagnostic_only; stable level bias is removed by the local sensor anchor"
        )
        candidate["regimeSuppressionReason"] = (
            "local_particle_regime_break" if regime_suppressed else None
        )
        candidate["validationState"] = (
            "prospectively_validated" if validated_candidate
            else "aggressive_experimental" if aggressive_candidate
                else "provisional_context_only" if context_candidate
                else "regime_break_suppressed" if regime_suppressed
                else "not_used"
        )
        point = float(candidate["mean"]) if display_candidate else current
        result = {
            **baseline,
            "available": bool(baseline.get("available") or display_candidate),
            "point": round(point, 1),
            "projectedPeak": (
                round(float(candidate["peak"]), 1) if display_candidate else None
            ),
            "baselinePoint": round(current, 1),
            "pointRole": (
                "validated_forecast" if validated_candidate
                else "aggressive_cams_forecast" if aggressive_candidate
                else "provisional_cams_context" if context_candidate
                else "persistence_anchor"
            ),
            "forecastState": (
                "validated_point_residual_span_collecting" if validated_candidate
                else "aggressive_experimental_point_and_range" if aggressive_candidate
                else "provisional_context_not_used_for_comparison" if context_candidate
                else "range_only"
            ),
            "pointApproximate": bool(aggressive_candidate or context_candidate),
            "validated": bool(validated_candidate),
            "usedForDecision": use_candidate,
            "candidate": candidate,
            "modelStatus": (
                "applied" if validated_candidate
                else "aggressive" if aggressive_candidate
                else "provisional_context" if context_candidate
                else "regime_break_suppressed" if regime_suppressed
                else "validated_not_applicable" if issued.get("supported")
                else "learning"
            ),
            "method": (
                "Prospectively validated sensor-anchored CAMS delta"
                if validated_candidate else
                "Aggressive experimental sensor-anchored CAMS delta"
                if aggressive_candidate else
                "Provisional aligned CAMS context; excluded from the window comparison"
                if context_candidate else
                "Sensor persistence with horizon-specific empirical errors"
            ),
        }
        if validated_candidate:
            calibration = task_evidence.get("residualCalibration") or {}
            low_residual = num(calibration.get("meanLow"))
            high_residual = num(calibration.get("meanHigh"))
            peak_residual = num(calibration.get("peakUpper"))
            if None not in (low_residual, high_residual, peak_residual):
                candidate_low = max(0.0, float(candidate["mean"]) + low_residual)
                candidate_high = max(0.0, float(candidate["mean"]) + high_residual)
                candidate_peak_upper = max(
                    float(candidate["peak"]), float(candidate["peak"]) + peak_residual
                )
                rounded_low = round(candidate_low, 1)
                rounded_high = round(candidate_high, 1)
                if rounded_high > rounded_low:
                    result.update({
                        "comparisonRangeLow": rounded_low,
                        "comparisonRangeHigh": rounded_high,
                        "comparisonPeakUpper": round(candidate_peak_upper, 1),
                        "comparisonIntervalState": "available_uncalibrated_empirical_span",
                        "forecastState": "validated_point_with_uncalibrated_empirical_span",
                        "comparisonIntervalMethod": (
                            "Task-specific empirical CAMS residual span; comparison only"
                        ),
                    })
                    result["uncertaintyMethod"] = (
                        "Task-specific empirical CAMS residual span; "
                        "matched-history evidence remains separate"
                    )
                else:
                    # A zero-width finite replay span is absence of resolved
                    # dispersion, not evidence of a certain forecast.
                    result.update({
                        "comparisonIntervalState": "unavailable_zero_width_residual_span",
                        "forecastState": "validated_point_residual_span_unresolved",
                        "comparisonIntervalMethod": (
                            "Not published because the finite CAMS residual span "
                            "rounds to zero width"
                        ),
                    })
                    result["uncertaintyMethod"] = (
                        "CAMS point available; task-specific residual spread unresolved"
                    )
        elif aggressive_candidate or context_candidate:
            # No task-specific residual calibration exists yet.  Keep the
            # experimental point separate from matched-history error evidence;
            # combining them would create a union with no coherent coverage.
            if baseline.get("available"):
                result.update({
                    "uncertaintyMethod": (
                        "Experimental point with separate uncalibrated "
                        "matched-history error evidence"
                    ),
                    "rangeRole": "persistence_anchor_containing_display_envelope",
                })
            else:
                result.update({
                    "uncertaintyMethod": (
                        "Experimental point; matched-history error evidence collecting"
                    ),
                    "rangeRole": "collecting",
                })
        forecasts[key] = result

    used_window_count = sum(
        1 for item in forecasts.values() if item.get("usedForDecision")
    )
    context_window_count = sum(
        1 for item in forecasts.values()
        if str(item.get("forecastState") or "").startswith("provisional_")
    )
    model_used = used_window_count > 0
    model_used_for_comparison = bool(
        forecasts and used_window_count == len(forecasts)
        and ((issued.get("pairBacktest") or {}).get("supported"))
    )
    regime_suppressed_count = sum(
        1 for item in forecasts.values()
        if (item.get("candidate") or {}).get("suppressedByRegimeBreak")
    )
    air_quality_error = None
    if include_runtime_status:
        with weather_lock:
            air_quality_error = weather_status.get("air_quality_error")
    source_age_minutes = (
        None if not payload_epoch
        else max(0, round((as_of_epoch - payload_epoch) / 60))
    )
    value = json_safe({
        "available": any(item.get("available") for item in forecasts.values()),
        # This is the reference time of the calculation that produced this
        # cached value.  Window labels copy it instead of moving on every HTTP
        # response while the underlying forecast remains cached.
        "forecastIssuedEpoch": int(as_of_epoch),
        "forecastIssuedRole": "cached_forecast_generation_reference_time",
        "status": (
            "validated" if issued.get("supported")
            else "aggressive_experimental" if model_used
            else "provisional_context" if context_window_count
            else "learning"
        ),
        "aggressiveMode": AGGRESSIVE_FORECAST_ENABLED,
        "applicationStatus": (
            "regime_break_sensor_persistence"
            if regime_suppressed_count else
            "aggressive_applied_to_comparison"
            if AGGRESSIVE_FORECAST_ENABLED and model_used_for_comparison
            else "applied_to_comparison" if model_used_for_comparison
            else "applied_individually_not_pair_validated" if model_used
            else "provisional_context_only" if context_window_count
            else "persistence_for_current_lead"
        ),
        "usedForDecision": model_used,
        "usedForComparison": model_used_for_comparison,
        "usedWindowCount": used_window_count,
        "provisionalContextWindowCount": context_window_count,
        "regimeBreakSuppressed": bool(regime_suppressed_count),
        "modelVersion": AIR_QUALITY_FORECAST_METHOD_VERSION,
        "sourceStatus": {
            "available": bool(payload),
            "fetchedEpoch": payload_epoch or None,
            "ageMinutes": source_age_minutes,
            "stale": bool(
                source_age_minutes is not None
                and source_age_minutes * 60 > AIR_QUALITY_DEGRADED_SECONDS
            ),
            "error": air_quality_error,
        },
        "productionMethod": (
            "Aggressive experimental sensor-anchored CAMS delta"
            if AGGRESSIVE_FORECAST_ENABLED and model_used_for_comparison
            else "Sensor-anchored CAMS delta" if model_used_for_comparison else
            "Per-window replay screen: CAMS or sensor persistence"
            if model_used else
            "Time-aligned CAMS context outside tested forecast leads"
            if context_window_count else "Sensor persistence"
        ),
        "issuedEvidence": issued,
        "retrospectiveDiagnostic": diagnostic,
        "windows": forecasts,
    })
    return value


def apply_near_term_local_window_forecast(ride_forecast, windows, air_window):
    """Use the tested +90-to-210 local horizon for an aligned ride session.

    Morning/Afternoon weather and regional CAMS scenarios can exist at much
    longer leads.  When the selected two-hour session is actually the same
    interval as the local trail horizon (exact same boundaries), this
    bridge prevents a second, differently anchored forecast from replacing it.
    """
    result = dict(ride_forecast or {})
    forecasts = {
        key: dict(value or {})
        for key, value in ((result.get("windows") or {}).items())
    }
    trail = dict((air_window or {}).get("trail") or {})
    anchor_epoch = num((air_window or {}).get("closedBucketEndEpoch"))
    if not trail.get("available") or anchor_epoch is None:
        result["windows"] = forecasts
        return result

    expected_start = trail.get("startEpoch")
    expected_end = trail.get("endEpoch")
    if expected_start is None or expected_end is None:
        result["windows"] = forecasts
        return result
    bridged_count = 0
    model_count = 0
    for key, target in (windows or {}).items():
        start_epoch = target.get("startEpoch")
        end_epoch = target.get("endEpoch")
        if start_epoch is None or end_epoch is None:
            continue
        start_difference = int(start_epoch) - expected_start
        end_difference = int(end_epoch) - expected_end
        if start_difference != 0 or end_difference != 0:
            continue

        existing = forecasts.get(key, {})
        modeled_mean = trail.get("point")
        if modeled_mean is None:
            continue
        point_role = trail.get("pointRole") or "persistence_anchor"
        model_applied = point_role != "persistence_anchor"
        peak_applied = bool(trail.get("peakApproximate"))
        bridged = {
            **existing,
            "available": True,
            "source": "local_near_term_trail_forecast",
            "regionalScenario": {
                "available": bool(existing.get("candidate", {}).get("available")),
                "point": existing.get("point"),
                "projectedPeak": existing.get("projectedPeak"),
                "state": existing.get("forecastState"),
            },
            "point": modeled_mean,
            "projectedPeak": trail.get("projectedPeak"),
            "baselinePoint": (
                trail.get("baselinePoint")
                if trail.get("baselinePoint") is not None
                else (air_window or {}).get("current15")
            ),
            "pointRole": point_role,
            "persistenceAnchorRole": trail.get("persistenceAnchorRole"),
            "persistenceAnchorEpoch": trail.get("persistenceAnchorEpoch"),
            "freshnessAdjustment": trail.get("freshnessAdjustment") or {},
            "forecastState": (
                "local_near_term_experimental"
                if model_applied else
                "local_near_term_persistence_with_experimental_peak"
                if peak_applied else
                "local_near_term_persistence"
            ),
            "pointApproximate": bool(trail.get("pointApproximate")),
            "peakApproximate": peak_applied,
            "validated": False,
            "usedForDecision": bool(model_applied or peak_applied),
            "modelStatus": (
                "local_near_term_experimental"
                if model_applied or peak_applied else
                "local_near_term_persistence"
            ),
            "method": "Local +90-to-210-minute " + str(
                trail.get("method") or "persistence forecast"
            ),
            "confidence": trail.get("confidence") or "Low",
            "leadHours": round(
                max(0.0, (int(start_epoch) - int(anchor_epoch)) / 3600.0), 1
            ),
            "localHorizonAlignmentMinutes": round(start_difference / 60),
        }
        for field in (
            "rawRangeLow", "rawRangeHigh", "rangeLow", "rangeHigh",
            "upperMean", "peakUpper", "decisionUpperMean",
            "decisionPeakUpper", "matchedCount", "matchedDistinctDays",
            "meanSkillEligible", "peakSkillEligible", "deploymentSkill",
            "modelMae", "persistenceMae", "peakModelMae",
            "peakPersistenceMae",
        ):
            if trail.get(field) is not None:
                bridged[field] = trail[field]
        forecasts[key] = bridged
        bridged_count += 1
        model_count += int(model_applied or peak_applied)

    result["windows"] = forecasts
    result["nearTermLocalWindowCount"] = bridged_count
    if bridged_count:
        result["applicationStatus"] = (
            "local_near_term_experimental"
            if model_count else "local_near_term_persistence"
        )
        result["usedForDecision"] = bool(
            result.get("usedForDecision") or model_count
        )
        result["usedWindowCount"] = sum(
            bool(item.get("usedForDecision")) for item in forecasts.values()
        )
        # A local near-term card and a long-lead regional scenario are not the
        # same forecast task and cannot form a validated pair comparison.
        result["usedForComparison"] = False
    return json_safe(result)


def regional_haze_window_outlook(window, as_of_epoch, payload, weather_payload):
    """A physical regional model trend, separate from calibrated TTDI PM.

    CAMS already includes emissions, atmospheric transport and removal. Avoid
    double-counting a cyclone/season or adding an unvalidated wind PM penalty.
    The wind paths explain route uncertainty, not PM mass or source identity.
    """
    start_epoch, end_epoch = window.get("startEpoch"), window.get("endEpoch")
    base = {
        "available": False, "method": "cams_regional_trend_air_paths_v1",
        "role": "regional_model_outlook_not_local_concentration",
        "usedForLocalPmPoint": False, "validatedAtTTDI": False,
        "sourceAttributionEstablished": False,
        "label": "Regional PM forecast unavailable", "direction": "unavailable",
        "forecastIssuedAt": iso_from_epoch(as_of_epoch),
        "startAt": iso_from_epoch(start_epoch), "endAt": iso_from_epoch(end_epoch),
        "confidence": "Experimental regional outlook",
    }
    if start_epoch is None or end_epoch is None:
        return base
    try:
        base["airPath"] = haze_transport.window_paths(
            DB_PATH, start_epoch, end_epoch, as_of_epoch
        )
    except Exception as error:
        # An optional regional source must not interrupt sensor forecasts.
        base["airPath"] = {"available": False, "label": "Air-path model unavailable",
                           "error": type(error).__name__, "usedForLocalPmPoint": False}
    if not payload:
        return base
    age = as_of_epoch - int(payload.get("fetchedEpoch") or 0)
    base["sourceFetchedAt"] = iso_from_epoch(payload.get("fetchedEpoch"))
    base["sourceAgeMinutes"] = round(age / 60, 1)
    base["source"] = "Open-Meteo / CAMS Global"
    base["sourceUrl"] = "https://open-meteo.com/en/docs/air-quality-api"
    base["emissionsBasis"] = "CAMS includes GFAS fire-emission estimates; no independent hotspot feed"
    if age < 0 or age > AIR_QUALITY_DEGRADED_SECONDS:
        base["label"] = "Regional PM source out of date"
        return base
    origin = pd.Timestamp(as_of_epoch, unit="s", tz="UTC").tz_convert("Asia/Kuala_Lumpur")
    start = pd.Timestamp(start_epoch, unit="s", tz="UTC").tz_convert(origin.tz)
    end = pd.Timestamp(end_epoch, unit="s", tz="UTC").tz_convert(origin.tz)
    model = model_frame(payload)
    now, values = aligned_model_projection(model, origin, start, end)
    if now is None or values.empty or values.isna().any():
        return base
    mean = float(values.mean())
    change = mean - now
    # Resolution rule for a model trend, NOT a health threshold or skill gate.
    threshold = max(3.0, 0.10 * now)
    direction = "rising" if change >= threshold else "easing" if change <= -threshold else "similar"
    base.update({
        "available": True, "direction": direction,
        "label": {"rising": "Regional PM increasing", "easing": "Regional PM easing",
                  "similar": "Little regional PM change"}[direction],
        "regionalCurrentPm25UgM3": round(now, 1),
        "regionalMeanPm25UgM3": round(mean, 1),
        "regionalPeakPm25UgM3": round(float(values.max()), 1),
        "regionalChangeUgM3": round(change, 1),
        "regionalChangePct": round(100 * change / now) if now >= 5 else None,
        "trendResolutionUgM3": round(threshold, 1),
        "regionalCoordinates": {"latitude": payload.get("latitude"), "longitude": payload.get("longitude")},
        "modelNativeResolutionHours": 3, "modelGridApproxKm": 45,
        "localConcentrationResolved": False,
        "limitation": "Regional PM includes multiple sources. This trend is not a validated TTDI concentration forecast or proof of Sumatra fire attribution.",
    })
    # Surface coupling is context: boundary-layer growth can dilute a surface
    # plume OR entrain elevated smoke. Do not apply a one-way cooling/mixing rule.
    wf = model_frame(weather_payload)
    if not wf.empty and "boundary_layer_height" in wf:
        layer = model_window_values(wf, start, end, "boundary_layer_height")
        height = num((base.get("airPath") or {}).get("approxHeightM"))
        if not layer.empty and not layer.isna().any() and height is not None:
            reaches = int((layer >= height).sum())
            base["surfaceCoupling"] = {
                "role": "modeled_vertical_context_not_pm_adjustment",
                "boundaryLayerMinM": round(float(layer.min())),
                "boundaryLayerMaxM": round(float(layer.max())),
                "windLevelApproxM": height,
                "label": "Daytime mixing may connect to the elevated layer" if reaches else "Elevated wind may be decoupled from the surface",
            }
    return base


def regional_haze_comparison(windows, previous):
    """Compare one regional plume trajectory, never rank rides or health."""
    result = copy.deepcopy(previous)
    morning = (windows.get("morning") or {}).get("hazeOutlook") or {}
    afternoon = (windows.get("afternoon") or {}).get("hazeOutlook") or {}
    result["localPmComparison"] = {key: previous.get(key) for key in ("headline", "summary", "pm")}
    result["policyId"] = "neutral_regional_haze_comparison_v1"
    result["regionalHaze"] = {"available": False, "usedForLocalPmComparison": False,
                              "role": "regional_model_direction_only", "lowerRegionalWindow": None}
    result["model"]["displayMethod"] = "Regional plume model + time-varying air paths"
    result["ready"] = bool(morning.get("available") and afternoon.get("available"))
    if not result["ready"]:
        result.update(headline="Regional haze outlook collecting",
                      summary="Local PM estimates and weather remain separate; regional model data is unavailable or out of date.",
                      confidence="Insufficient")
        return result
    if (morning.get("startAt", "")[:10] != afternoon.get("startAt", "")[:10]):
        # Use local calendar dates; startAt is UTC and these sessions are daytime KL.
        result.update(ready=False, headline="The windows refer to different dates",
                      summary="Regional trends are shown for each date; no same-day comparison.", confidence="Low")
        return result
    m, a = morning["regionalMeanPm25UgM3"], afternoon["regionalMeanPm25UgM3"]
    gap = a - m
    threshold = max(3.0, 0.10 * max(m, a))
    lower = "morning" if gap >= threshold else "afternoon" if gap <= -threshold else None
    result["regionalHaze"].update(available=True, lowerRegionalWindow=lower,
                                   differenceAfternoonMinusMorningUgM3=round(gap,1),
                                   separationThresholdUgM3=round(threshold,1))
    headline = ("Regional haze model: afternoon higher" if lower == "morning" else
                "Regional haze model: afternoon lower" if lower == "afternoon" else
                "Regional haze model: similar in both windows")
    result.update(headline=headline, confidence="Experimental regional outlook",
                  summary="Model direction, not a confirmed difference in TTDI concentrations. Air paths and local weather are shown below.")
    return result


def local_window_pm_comparison(windows, prediction):
    """Describe numerical PM estimates without ranking rides or requiring weather."""
    p = {key: (windows.get(key) or {}).get("particleForecast") or {}
         for key in ("morning", "afternoon")}
    ready = all(p[key].get("available") and num(p[key].get("point")) is not None
                for key in p)
    result = {
        "ready": ready, "role": "descriptive_air_and_weather_outlook",
        "policyId": "local_window_pm_comparison_v1",
        "prescriptiveRecommendation": False, "confidence": "Low · experimental",
        "model": {"modelVersion": prediction.get("modelVersion"),
                  "modelsByWindow": prediction.get("modelsByWindow") or {},
                  "status": "experimental_local_window_model",
                  "displayLabel": "Experimental PM forecast",
                  "modelPointApplied": ready, "directComparisonEligible": False,
                  "aggressiveMode": True},
        "weather": {key: compact_weather_window(
            (windows.get(key) or {}).get("weatherForecast")) for key in p},
        "pm": {"morningMeanPm25UgM3": p["morning"].get("point"),
               "afternoonMeanPm25UgM3": p["afternoon"].get("point"),
               "differenceMorningMinusAfternoonUgM3": None,
               "directComparisonEligible": False, "usedForComparison": False,
               "lowerModeledPmWindow": None, "lowerExperimentalScenarioWindow": None},
        "headline": "PM2.5 forecast unavailable",
        "summary": "A fresh local estimate is required for both sessions.",
    }
    if not ready:
        return result
    dates = [local_dt(windows[key]["startEpoch"]).date() for key in p]
    if dates[0] != dates[1]:
        result.update(ready=False, headline="Forecasts refer to different dates",
                      summary="Each card shows its dated PM2.5 estimate; no same-day comparison.")
        return result
    m, a = float(p["morning"]["point"]), float(p["afternoon"]["point"])
    gap = m-a
    error = [num((p[key].get("modelEvidence") or {}).get("mae")) for key in p]
    uncertainty = max([value for value in error if value is not None], default=0.0)
    lower = "afternoon" if gap >= 1 else "morning" if gap <= -1 else None
    result["pm"].update({
        "differenceMorningMinusAfternoonUgM3": round(gap, 1),
        "lowerExperimentalScenarioWindow": lower,
        "differenceExceedsIndividualMeanErrors": bool(uncertainty and abs(gap) > uncertainty),
    })
    result["headline"] = (
        f'{lower.capitalize()} PM2.5 estimate ≈{abs(gap):.0f} µg/m³ lower'
        if lower else "Morning and afternoon PM2.5 estimates are similar"
    )
    result["summary"] = (
        "Afternoon uses an experimental direction model; the session ordering is not validated."
        if any((value.get("directionalModel") or {}).get("applied") for value in p.values()) else
        "The difference is smaller than recent forecast errors; the ordering is uncertain."
        if uncertainty and abs(gap) <= uncertainty else
        "Experimental concentration estimates, not a validated morning/afternoon ranking."
    )
    return result


def forecast_comparison(windows, weather, ride_forecast):
    """Return neutral PM and weather facts without selecting a ride window."""
    morning = windows.get("morning") or {}
    afternoon = windows.get("afternoon") or {}
    morning_pm = morning.get("particleForecast") or {}
    afternoon_pm = afternoon.get("particleForecast") or {}
    morning_weather = morning.get("weatherForecast") or {}
    afternoon_weather = afternoon.get("weatherForecast") or {}
    evidence = ride_forecast.get("issuedEvidence") or {}
    base = {
        "ready": False,
        "role": "descriptive_air_and_weather_outlook",
        "method": "PM forecast comparison with separate weather context",
        "policyId": "neutral_air_weather_outlook_v2",
        "prescriptiveRecommendation": False,
        "model": {
            "status": ride_forecast.get("status", "collecting"),
            "applicationStatus": ride_forecast.get("applicationStatus"),
            "modelPointApplied": bool(ride_forecast.get("usedForDecision")),
            "directComparisonEligible": bool(
                ride_forecast.get("usedForComparison")
            ),
            "modelVersion": ride_forecast.get("modelVersion"),
            "aggressiveMode": bool(ride_forecast.get("aggressiveMode")),
            "scoredWindowCount": evidence.get("scoredWindowCount", 0),
            "minimumScoredWindows": evidence.get(
                "minimumScoredWindows", AIR_QUALITY_MIN_SCORED_WINDOWS
            ),
            "distinctDays": evidence.get("distinctDays", 0),
            "minimumDistinctDays": evidence.get(
                "minimumDistinctDays", AIR_QUALITY_MIN_ISSUED_DAYS
            ),
            "note": evidence.get("note"),
        },
    }

    weather_ready = bool(
        morning_weather.get("available") and afternoon_weather.get("available")
    )
    same_day = bool(
        weather_ready
        and morning_weather.get("date")
        and morning_weather.get("date") == afternoon_weather.get("date")
    )
    base["ready"] = weather_ready and same_day
    base["weather"] = {
        "morning": {
            "apparentTemperatureMaxC": morning_weather.get(
                "apparentTemperatureMax"
            ),
            "temperatureMaxC": morning_weather.get("temperatureMax"),
            "relativeHumidityMeanPct": morning_weather.get(
                "relativeHumidityMean"
            ),
            "sky": morning_weather.get("skyLabel"),
            "cloudCoverMeanPct": morning_weather.get("cloudCoverMean"),
            "windSpeedMeanKmh": morning_weather.get("windSpeed10mMean"),
            "windGustMaxKmh": morning_weather.get("windGust10mMax"),
            "rainProbabilityMaxPct": morning_weather.get(
                "precipitationProbabilityMax"
            ),
            "precipitationMm": morning_weather.get("precipitationMm"),
        },
        "afternoon": {
            "apparentTemperatureMaxC": afternoon_weather.get(
                "apparentTemperatureMax"
            ),
            "temperatureMaxC": afternoon_weather.get("temperatureMax"),
            "relativeHumidityMeanPct": afternoon_weather.get(
                "relativeHumidityMean"
            ),
            "sky": afternoon_weather.get("skyLabel"),
            "cloudCoverMeanPct": afternoon_weather.get("cloudCoverMean"),
            "windSpeedMeanKmh": afternoon_weather.get("windSpeed10mMean"),
            "windGustMaxKmh": afternoon_weather.get("windGust10mMax"),
            "rainProbabilityMaxPct": afternoon_weather.get(
                "precipitationProbabilityMax"
            ),
            "precipitationMm": afternoon_weather.get("precipitationMm"),
        },
    }

    def modeled_point(particle):
        if not particle.get("available"):
            return None
        if particle.get("pointRole") == "persistence_anchor":
            return None
        return num(particle.get("point"))

    morning_point = modeled_point(morning_pm)
    afternoon_point = modeled_point(afternoon_pm)
    direct_eligible = bool(
        ride_forecast.get("usedForComparison")
        and
        morning_pm.get("usedForDecision")
        and afternoon_pm.get("usedForDecision")
        and morning_point is not None
        and afternoon_point is not None
    )
    lower_window = None
    scenario_lower = None
    materially_separated = False
    difference = None
    pair = None
    if morning_point is not None and afternoon_point is not None:
        difference = morning_point - afternoon_point
        pair = classify_pm_window_pair(
            morning_point, afternoon_point,
            morning_pm.get("projectedPeak"),
            afternoon_pm.get("projectedPeak"),
        )
        if pair.get("called"):
            scenario_lower = pair.get("winner")
        if direct_eligible and pair.get("called"):
            lower_window = pair.get("winner")
            materially_separated = True

    base["pm"] = {
        "morningMeanPm25UgM3": morning_point,
        "afternoonMeanPm25UgM3": afternoon_point,
        "differenceMorningMinusAfternoonUgM3": (
            round(difference, 1) if difference is not None else None
        ),
        "directComparisonEligible": direct_eligible,
        "materiallySeparated": materially_separated,
        "lowerModeledPmWindow": lower_window,
        "lowerExperimentalScenarioWindow": scenario_lower,
        "usedForComparison": bool(lower_window),
    }

    if not weather_ready:
        base.update({
            "headline": "Morning and afternoon weather is collecting",
            "summary": "Particle estimates remain available in their individual cards.",
            "confidence": "Insufficient",
        })
    elif not same_day:
        base.update({
            "headline": "The two cards refer to different dates",
            "summary": "No direct same-day PM or weather comparison is shown.",
            "confidence": "Insufficient",
        })
    elif morning_point is None or afternoon_point is None:
        base.update({
            "headline": "A PM2.5 estimate is missing for one or both windows",
            "summary": (
                "No supported modeled PM2.5 point is available for both windows; "
                "weather values are shown separately."
            ),
            "confidence": "Low",
        })
    elif lower_window:
        label = lower_window.capitalize()
        base.update({
            "headline": f"Lower modeled PM2.5: {label}",
            "summary": (
                f"Morning ≈{morning_point:.0f} and Afternoon "
                f"≈{afternoon_point:.0f} µg/m³. This describes the particle "
                "forecast only; weather is reported separately."
            ),
            "confidence": "Low" if not all(
                item.get("validated") for item in (morning_pm, afternoon_pm)
            ) else "Medium",
        })
    elif scenario_lower:
        label = scenario_lower.capitalize()
        base.update({
            "headline": f"Experimental PM scenario: {label} lower",
            "summary": (
                f"Morning ≈{morning_point:.0f} and Afternoon "
                f"≈{afternoon_point:.0f} µg/m³; modeled difference "
                f"{abs(difference):.0f} µg/m³. This comparison is not yet "
                "validated; weather remains a separate outlook."
            ),
            "confidence": "Low",
        })
    else:
        mean_gap = abs(float(pair.get("meanGap") or difference or 0.0))
        mean_threshold = float(pair.get("meanThreshold") or 5.0)
        if mean_gap < mean_threshold:
            headline = "Morning and afternoon PM2.5 estimates are similar"
            explanation = (
                f"The {mean_gap:.1f} µg/m³ gap is below the "
                f"{mean_threshold:.1f} µg/m³ separation threshold, so neither "
                "window is identified as lower."
            )
        elif not pair.get("peakAgrees"):
            headline = "PM2.5 mean and peak estimates do not agree"
            explanation = (
                "The mean difference does not identify the same lower-PM "
                "window as the peak estimates."
            )
        else:
            headline = "No reliable PM2.5 separation"
            explanation = "The current estimates do not pass the separation rule."
        estimate_prefix = "Provisional estimates" if not direct_eligible else "Estimates"
        base.update({
            "headline": headline,
            "summary": (
                f"{estimate_prefix}: Morning ≈{morning_point:.1f} and Afternoon "
                f"≈{afternoon_point:.1f} µg/m³. {explanation} Weather is shown "
                "separately."
            ),
            "confidence": "Low",
        })
    return base


def medianv(values):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    middle = len(vals) // 2
    if len(vals) % 2:
        return vals[middle]
    return (vals[middle - 1] + vals[middle]) / 2


def quantile(values, q):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    position = (len(vals) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return vals[lower]
    fraction = position - lower
    return vals[lower] * (1 - fraction) + vals[upper] * fraction


def weighted_median(samples):
    usable = sorted((float(value), float(weight)) for value, weight in samples
                    if value is not None and weight > 0)
    if not usable:
        return None
    halfway = sum(weight for _, weight in usable) / 2
    running = 0.0
    for value, weight in usable:
        running += weight
        if running >= halfway:
            return value
    return usable[-1][0]


def ride_day_summaries(rows, config):
    grouped = {}
    for row in rows:
        local = local_dt(row["epoch"])
        if not (config["start"] <= local.hour < config["end"]):
            continue
        bucket_epoch = (int(row["epoch"]) // 900) * 900
        bucket = grouped.setdefault(local.date(), {}).setdefault(
            bucket_epoch, {"pm02": [], "heatindex": []}
        )
        if row["pm02"] is not None:
            bucket["pm02"].append(float(row["pm02"]))
        heat = row["heatindex"] if row["heatindex"] is not None else row["atmp"]
        if heat is not None:
            bucket["heatindex"].append(float(heat))

    result = {}
    expected = (config["end"] - config["start"]) * 4
    for day, raw_buckets in grouped.items():
        points = []
        for epoch, values in sorted(raw_buckets.items()):
            pm = medianv(values["pm02"])
            heat = medianv(values["heatindex"])
            points.append({"epoch": epoch, "pm02": pm, "heatindex": heat})

        pm_points = [(p["epoch"], p["pm02"]) for p in points if p["pm02"] is not None]
        heat_values = [p["heatindex"] for p in points if p["heatindex"] is not None]
        pm_count = len(pm_points)
        heat_count = len(heat_values)
        coverage = min(pm_count, heat_count) / expected if expected else 0
        if pm_count < MIN_WINDOW_BUCKETS or heat_count < MIN_WINDOW_BUCKETS:
            continue

        one_hour_means = []
        pm_by_epoch = dict(pm_points)
        for start_epoch, _ in pm_points:
            expected_epochs = [start_epoch + step * 900 for step in range(4)]
            if all(epoch in pm_by_epoch for epoch in expected_epochs):
                one_hour_means.append(
                    sum(pm_by_epoch[epoch] for epoch in expected_epochs) / 4
                )
        if not one_hour_means:
            continue

        result[day] = {
            "date": day,
            "pmMean": sum(value for _, value in pm_points) / pm_count,
            "pmWorstHour": max(one_hour_means),
            "heatPeak": max(heat_values),
            "coverage": min(1.0, coverage),
            "bucketCount": min(pm_count, heat_count),
        }
    return result


def baseline_metric(day_rows, key, newest_day):
    values = [row[key] for row in day_rows]
    weighted = []
    for row in day_rows:
        age_days = max(0, (newest_day - row["date"]).days)
        weight = 0.5 ** (age_days / HISTORY_HALF_LIFE_DAYS)
        weighted.append((row[key], weight))
    return {
        "median": round(weighted_median(weighted), 1),
        "q1": round(quantile(values, 0.25), 1),
        "q3": round(quantile(values, 0.75), 1),
    }


def paired_history(rows):
    per_window = {config["key"]: ride_day_summaries(rows, config)
                  for config in RIDE_WINDOWS}
    paired_dates = sorted(set(per_window["morning"]) & set(per_window["afternoon"]))
    all_dates = sorted(set().union(*(set(values) for values in per_window.values())))
    newest_day = all_dates[-1] if all_dates else local_dt(rows[-1]["epoch"]).date()
    baselines = {}
    for config in RIDE_WINDOWS:
        key = config["key"]
        day_rows = [per_window[key][day] for day in sorted(per_window[key])]
        baseline = {
            "available": bool(day_rows),
            "validDays": len(per_window[key]),
            "pairedDays": len(paired_dates),
        }
        if day_rows:
            baseline.update({
                "pmMean": baseline_metric(day_rows, "pmMean", newest_day),
                "pmWorstHour": baseline_metric(day_rows, "pmWorstHour", newest_day),
                "heatPeak": baseline_metric(day_rows, "heatPeak", newest_day),
                "coveragePct": round(100 * sum(row["coverage"] for row in day_rows) / len(day_rows)),
            })
        baselines[key] = baseline

    paired_rows = [(per_window["morning"][day], per_window["afternoon"][day])
                   for day in paired_dates]
    coverage_pct = None
    if paired_rows:
        coverage_pct = round(50 * sum(m["coverage"] + a["coverage"] for m, a in paired_rows) / len(paired_rows))
    return baselines, paired_rows, coverage_pct


def ride_target(latest_epoch, config):
    now = local_dt(latest_epoch)
    today = now.date()
    forecasted_label = f'Forecasted at {now:%H:%M}'
    period = next_feasible_ride_period(latest_epoch, config)
    if period is None:
        return {
            "targetLabel": "Unavailable",
            "rideLabel": f'{config["start"]:02d}:00–{config["end"]:02d}:00',
            "startEpoch": None,
            "endEpoch": None,
            "leadHours": None,
            "active": False,
            "currentApplicable": False,
            "decisionEpoch": None,
            "decisionLabel": "Logistics reference unavailable",
            "forecastIssuedEpoch": int(latest_epoch),
            "forecastedLabel": forecasted_label,
            # Deprecated 1.x display alias retained without delay wording.
            "recheckLabel": forecasted_label,
        }
    target_day, _, _, session_start, session_end, decision = period
    lead_hours = max(0.0, (session_start - now).total_seconds() / 3600)
    current_applicable = lead_hours <= 2.25
    day_label = "Today" if target_day == today else "Tomorrow"
    return {
        "targetLabel": day_label,
        "rideLabel": f'{config["start"]:02d}:00–{config["end"]:02d}:00',
        "modeledSessionLabel": f'{session_start:%H:%M}–{session_end:%H:%M}',
        "startEpoch": int(session_start.timestamp()),
        "endEpoch": int(session_end.timestamp()),
        "leadHours": round(lead_hours, 1),
        "active": False,
        "currentApplicable": current_applicable,
        "decisionEpoch": int(decision.timestamp()),
        "decisionLabel": f'Logistics reference {decision:%H:%M}',
        "forecastIssuedEpoch": int(latest_epoch),
        "forecastedLabel": forecasted_label,
        # Deprecated 1.x display alias retained without delay wording.
        "recheckLabel": forecasted_label,
    }


def apply_cached_forecast_issue(windows, ride_forecast, fallback_epoch):
    """Stamp window labels with the generation time of the cached forecast."""
    issued_epoch = int(
        (ride_forecast or {}).get("forecastIssuedEpoch") or fallback_epoch
    )
    issued_label = f'Forecasted at {local_dt(issued_epoch):%H:%M}'
    for window in windows.values():
        window["forecastIssuedEpoch"] = issued_epoch
        window["forecastedLabel"] = issued_label
        # Deprecated 1.x display alias retained without delay wording.
        window["recheckLabel"] = issued_label
    return issued_epoch


def analysis(days, as_of_epoch=None):
    """Instant live snapshots; explicit historical replays remain synchronous."""
    if as_of_epoch is not None:
        return _compute_analysis(days, as_of_epoch=as_of_epoch)
    days = max(1.0, min(float(days), 30.0))
    return analysis_delivery.response(days)


def archive_published_analysis(result):
    """Preserve actual publication time; never relabel a forecast as newer."""
    if not result.get("available"):
        result["computation"]["publishedForecastArchived"] = False
        return
    try:
        with db() as conn:
            record_dashboard_issue(conn, result, DASHBOARD_BUILD, int(time.time()))
        result["computation"]["publishedForecastArchived"] = True
    except (sqlite3.Error, ValueError) as error:
        result["computation"]["publishedForecastArchived"] = False
        print(f"[Forecast archive] {type(error).__name__}: {error}")


def enrich_analysis_trials(result, days, context):
    """Attach diagnostics after primary publication, at its original cutoff.

    The single producer executes this phase too: legacy model caches are not
    newly shared across primary/diagnostic threads. A trial keeps its own issue
    metadata if its existing model cache returns an older issued comparator.
    """
    if not result.get("available"):
        return result
    issued = int(result["forecastIssuedEpoch"])
    if context.get("issueEpoch") != issued:
        raise ValueError("Diagnostic sensor snapshot does not match primary issue")
    # Retain the original private sensor snapshot: a late backfill/correction
    # must not enter diagnostics claiming the earlier primary issue time.
    rows = context["rows"]
    windows = {key: {field: window.get(field) for field in ("startEpoch", "endEpoch")}
               for key, window in result.get("windows", {}).items()}
    rain = cached_rain_learning("near", rows, {}, issued)
    rain.update({"appliedToPrimaryForecast": False, "status": "evaluation_only",
                 "promotionReason": "fixed_paired_replay_did_not_improve_accuracy"})
    rain_sessions = cached_rain_learning("sessions", rows, windows, issued)
    correction = cams_local_correction.predict_windows(DB_PATH, rows, windows, issued)
    publication = int(time.time())
    rain_pm_forecast.record_issue(DB_PATH, rain, publication)
    window_pm_predictor.record_issue(DB_PATH, rain_sessions, publication)
    cams_local_correction.record_issue(DB_PATH, correction, publication)
    enriched = copy.deepcopy(result)
    enriched["rainLearning"] = rain
    for key, window in enriched.get("windows", {}).items():
        particle = window.get("particleForecast") or {}
        trial = correction.get(key) or {}
        rain_session = rain_sessions.get(key) or {}
        particle["rainLearning"].update({
            "trialAvailable": bool(rain_session.get("available")),
            "trialModelVersion": rain_session.get("modelVersion"),
            "trialMeanPm25UgM3": rain_session.get("mean"),
            "trialWeatherFetchedEpoch": rain_session.get("weatherFetchedEpoch"),
            "trialForecastIssuedEpoch": rain_session.get("forecastedAtEpoch"),
        })
        particle["regionalCorrectionTrial"].update({
            "available": bool(trial.get("available")),
            "meanPm25UgM3": trial.get("mean"),
            "regionalCorrection": trial.get("regionalCorrection"),
            "modelEvidence": trial.get("modelEvidence") or {},
            "forecastIssuedAt": iso_from_epoch(trial.get("forecastedAtEpoch")),
            "sensorAnchorAt": iso_from_epoch(trial.get("originEpoch")),
            "regionalDataRetrievedAt": iso_from_epoch(trial.get("camsFetchedEpoch")),
            "reason": trial.get("reason"),
        })
    return enriched


def analysis_worker():
    """One producer, independent of browser count; publish before diagnostics."""
    while True:
        analysis_delivery.schedule(RIDE_API_ANALYSIS_DAYS)
        days = analysis_delivery.take_job()
        if days is None:
            analysis_delivery.wake.wait(5)
            analysis_delivery.wake.clear()
            continue
        try:
            started = time.perf_counter()
            trial_context = {}
            result = _compute_analysis(days, include_trials=False, trial_context=trial_context)
            result["computation"] = {
                "completedAt": iso_now(),
                "seconds": round(time.perf_counter()-started, 3),
                "inputObservationEpoch": (result.get("current") or {}).get("epoch"),
                "refreshIntervalSeconds": 60,
                "maximumClockCacheSeconds": 600,
            }
            archive_published_analysis(result)
            analysis_delivery.publish_primary(days, result)
            try:
                diagnostics_started = time.perf_counter()
                enriched = enrich_analysis_trials(result, days, trial_context)
                enriched["computation"]["diagnosticsCompletedAt"] = iso_now()
                enriched["computation"]["diagnosticsSeconds"] = round(time.perf_counter()-diagnostics_started, 3)
                archive_published_analysis(enriched)
                analysis_delivery.publish_diagnostics(days, enriched)
            except Exception as error:
                analysis_delivery.fail(days, error, diagnostics=True)
                print(f"[Analysis diagnostics] {type(error).__name__}: {error}")
        except Exception as error:
            analysis_delivery.fail(days, error)
            print(f"[Analysis] {type(error).__name__}: {error}")
        finally:
            analysis_delivery.finish(days)


_rain_trial_cache = OrderedDict()


def cached_rain_learning(kind, rows, windows, issue_epoch):
    """Diagnostic fits once per closed bucket/source vintage, never relabelled.

    Primary forecasts do not use this cache. Reused trials keep their exact
    original issue/targets; replay cannot read a later cached issue. Full closed
    sensor data and archive payload hashes catch in-place input corrections.
    """
    origin = int(issue_epoch) // 900 * 900
    closed = [{k: row[k] for k in ("epoch", "pm02", "atmp", "rhum")}
              for row in rows if row["epoch"] <= origin]
    sensor_revision = hashlib.blake2b(
        json.dumps(closed, sort_keys=True, default=str).encode(), digest_size=12
    ).hexdigest()
    source_revision = tuple(window_pm_predictor._raw_runs(DB_PATH, table, issue_epoch)[1]
        for table in ("weather_forecast_runs", "air_quality_forecast_runs"))
    targets = tuple((key, value.get("startEpoch"), value.get("endEpoch"))
                    for key, value in sorted((windows or {}).items()))
    key = (str(DB_PATH), kind, origin, sensor_revision, source_revision, targets)
    cached = _rain_trial_cache.get(key)
    if cached and cached[0] <= issue_epoch and issue_epoch - cached[0] < 900:
        _rain_trial_cache.move_to_end(key)
        return copy.deepcopy(cached[1])
    value = (rain_pm_forecast.predict_nearterm(DB_PATH, rows, issue_epoch)
             if kind == "near" else window_pm_predictor.predict_windows(
                 DB_PATH, rows, windows, issue_epoch, rain_learning=True))
    _rain_trial_cache[key] = (issue_epoch, copy.deepcopy(value))
    while len(_rain_trial_cache) > 4:
        _rain_trial_cache.popitem(last=False)
    return value


def _compute_analysis(days, as_of_epoch=None, include_trials=True, trial_context=None):
    """Build live analysis or a causally bounded historical replay."""
    replay_epoch = None if as_of_epoch is None else int(as_of_epoch)
    # One frozen issue clock for every target, fit, archive and public response.
    # Freeze the actual computation clock, not the minute's earlier boundary:
    # a source fetched seconds into this minute is already available now.
    reference_epoch = int(time.time()) if replay_epoch is None else replay_epoch
    cutoff = int(reference_epoch - max(1.0, min(float(days), 30.0)) * 86400)
    with db() as conn:
        rows = conn.execute(
            "SELECT epoch,pm02,pm10,pm003Count,atmp,rhum,heatindex,rco2,tvoc,"
            "noxIndex,tvocIndex FROM readings WHERE epoch>=? AND epoch<=? "
            "ORDER BY epoch",
            (cutoff, reference_epoch)
        ).fetchall()
        first_stored_epoch = conn.execute(
            "SELECT MIN(epoch) FROM readings WHERE epoch<=?", (reference_epoch,)
        ).fetchone()[0]

    if not rows:
        return {"available": False, "message": "No local history yet. Leave the server running.",
                "outlook": {"available": False}, "current": {}, "windows": {},
                "airWindow": {"available": False, "state": "collecting",
                              "label": "Collecting particle history…",
                              "arrival": {"available": False, "headline": "Not enough local history"}},
                "shadowForecast": {"enabled": True, "usedForDecision": False,
                                   "status": "collecting"},
                "rideForecast": {"available": False, "status": "collecting"},
                "comparison": {
                    "ready": False,
                    "role": "descriptive_air_and_weather_outlook",
                    "headline": "Outlook unavailable",
                    "summary": "No local particle readings yet.",
                    "prescriptiveRecommendation": False,
                },
                "showerSignal": passing_shower_signal([]),
                "weather": {"available": False, "message": "Weather forecast is collecting."}}

    if trial_context is not None:
        trial_context.update(issueEpoch=reference_epoch, rows=rows)
    shower_signal = passing_shower_signal(rows)
    air_window = air_window_analysis(rows, shower_signal, reference_epoch)
    shadow_forecast = dict(cached_shadow_analogue_outlook(rows, reference_epoch))
    if replay_epoch is None:
        record_local_pm_forecast_issue(shadow_forecast, int(time.time()))
        shadow_forecast["prospectiveIssueLogActive"] = True
        shadow_forecast["prospectiveModelVersion"] = LOCAL_RIDGE_MODEL_VERSION
    rapid_event = rapid_clearance_event_outlook(rows, shower_signal)
    shadow_forecast["rapidClearanceEvent"] = rapid_event
    # Keep the very small-sample weather-assisted early watch quarantined from
    # the dashboard state, external API and production persistence forecast.
    shadow_forecast["dryDispersionWatch"] = shadow_dry_dispersion_watch(rows)
    air_window = aggressive_air_window_forecast(
        air_window, shadow_forecast, rapid_event
    )
    air_window = refresh_near_term_persistence(rows, shower_signal, air_window, reference_epoch)
    # A separately versioned learned rain association is collected, not promoted:
    # the fixed September 9 paired replay failed to improve overall wet-case error.
    # Its points and residuals must never relabel or replace the primary estimator.
    try:
        rain_trial = (cached_rain_learning("near", rows, {}, reference_epoch)
                      if include_trials else {"available": False, "reason": "background_diagnostics_pending"})
    except Exception as error:
        print(f"[Rain PM trial] {type(error).__name__}: {error}")
        rain_trial = {"available": False, "reason": "calculation_unavailable"}
    rain_trial.update({"appliedToPrimaryForecast": False, "status": "evaluation_only",
                       "promotionReason": "fixed_paired_replay_did_not_improve_accuracy"})
    if replay_epoch is None and include_trials:
        try:
            rain_pm_forecast.record_issue(DB_PATH, rain_trial, int(time.time()))
        except Exception as error:
            print(f"[Rain PM archive] {type(error).__name__}: {error}")
    air_window = separate_observation_from_prediction(air_window)
    # This is the final published stage, after the observed-reference refresh.
    # Earlier diagnostic stages retain their own points; no second model fit.
    if air_window.get("nearTermSelectionDiagnostics"):
        diagnostic = air_window["nearTermSelectionDiagnostics"]
        diagnostic["dashboardBuild"] = DASHBOARD_BUILD
        diagnostic["numericalPolicyIdentifier"] = "near_routing_v3_independent_mean_peak"
        diagnostic["forecastIssuedEpoch"] = reference_epoch
        diagnostic["evidenceType"] = "observation_time_reconstruction" if replay_epoch is not None else "computed_for_publication"
        diagnostic["publishedStage"] = {
            key: {name: (air_window.get(key) or {}).get(name) for name in
                  ("point", "pointRole", "projectedPeak", "peakApproximate", "forecastClock",
                   "persistenceAnchorRole", "persistenceAnchorEpoch", "freshnessAdjustment")}
            for key in ("arrival", "trail")
        }
    if AGGRESSIVE_FORECAST_ENABLED and (
            (air_window.get("arrival") or {}).get("usedForDecision")
            or (air_window.get("trail") or {}).get("usedForDecision")):
        shadow_forecast["usedForDecision"] = True
        shadow_forecast["deploymentState"] = "aggressive_experimental"
    latest_epoch = int(rows[-1]["epoch"])
    forecast_epoch = reference_epoch
    weather = weather_outlook(
        rows, forecast_epoch, include_runtime_status=replay_epoch is None
    )
    baselines, paired_rows, coverage_pct = paired_history(rows)
    windows = {}
    for config in RIDE_WINDOWS:
        target = ride_target(forecast_epoch, config)
        baseline = baselines[config["key"]]
        windows[config["key"]] = {
            **target,
            "baseline": baseline,
            "weatherForecast": weather.get("windows", {}).get(config["key"], {}),
        }
    regime_break = bool(
        shower_signal.get("state") in ("forming", "active")
        or (
            shower_signal.get("state") == "recent"
            and num(shower_signal.get("ageMinutes")) is not None
            and float(shower_signal["ageMinutes"]) <= 30
        )
    )
    particle_windows = {
        key: {
            "startEpoch": window.get("startEpoch"),
            "endEpoch": window.get("endEpoch"),
        }
        for key, window in windows.items()
    }
    ride_forecast = cached_ride_window_particle_forecast(
        rows, particle_windows, forecast_epoch, regime_break,
        include_runtime_status=replay_epoch is None,
    )
    ride_forecast = apply_near_term_local_window_forecast(
        ride_forecast, windows, air_window
    )
    # Only the explicitly approved morning-issued afternoon scope is replaced.
    # The independently archived CAMS transform remains diagnostic only.
    try:
        session_estimates = window_pm_predictor.predict_windows(
            DB_PATH, rows, particle_windows, forecast_epoch
        )
    except Exception as error:
        print(f"[Window PM] {type(error).__name__}: {error}")
        session_estimates = {}
    try:
        session_estimates = afternoon_direction_forecast.apply_experimental_afternoon(
            DB_PATH, rows, particle_windows, forecast_epoch, session_estimates
        )
    except Exception as error:
        print(f"[Afternoon direction] Keeping existing model: {type(error).__name__}: {error}")
    try:
        rain_session_trials = (cached_rain_learning("sessions", rows, particle_windows, forecast_epoch)
                               if include_trials else {})
    except Exception as error:
        print(f"[Window rain trial] {type(error).__name__}: {error}")
        rain_session_trials = {}
    if replay_epoch is None:
        try:
            window_pm_predictor.record_issue(DB_PATH, session_estimates, int(time.time()))
            if include_trials:
                window_pm_predictor.record_issue(DB_PATH, rain_session_trials, int(time.time()))
        except Exception as error:
            print(f"[Window PM archive] {type(error).__name__}: {error}")
    # A new regional-to-local candidate is evaluated independently. Its replay
    # did not beat the deployed model, so it cannot replace a primary estimate.
    try:
        correction_trials = (cams_local_correction.predict_windows(
            DB_PATH, rows, particle_windows, forecast_epoch
        ) if include_trials else {})
    except Exception as error:
        print(f"[CAMS correction trial] {type(error).__name__}: {error}")
        correction_trials = {}
    if replay_epoch is None and include_trials:
        try:
            cams_local_correction.record_issue(DB_PATH, correction_trials, int(time.time()))
        except Exception as error:
            print(f"[CAMS correction trial archive] {type(error).__name__}: {error}")
    scoped_direction_applied = any((value.get("directionalModel") or {}).get("applied")
                                   for value in session_estimates.values())
    window_prediction = {
        "modelVersion": "session_scoped_direction_v1" if scoped_direction_applied else window_pm_predictor.MODEL_VERSION,
        "modelsByWindow": {key: value.get("modelVersion") for key, value in session_estimates.items()},
        "note": "Exact-session experimental PM estimates. The adaptive local ensemble is retained except for the explicitly experimental 07:00–08:00 issue of the same-day 14:00–16:00 session, which uses a rise/steady/fall model. Only completed earlier outcomes train either model. Retrospective exploration is not prospective validation.",
        "predictions": session_estimates,
        "prospectivelyValidated": False,
        "regionalCorrectionTrial": {
            "modelVersion": cams_local_correction.MODEL_VERSION,
            "status": "live_experimental_comparator",
            "appliedToPrimaryForecast": False,
            "prospectivelyValidated": False,
            "note": "CAMS session concentration plus a learned evolution of the TTDI-minus-CAMS correction. Historical replay did not beat the current model overall. This fixed candidate is recorded separately for future evaluation; it does not replace the headline forecast or select a session.",
        },
    }
    for key, window in windows.items():
        estimate = session_estimates.get(key) or {}
        trial = correction_trials.get(key) or {}
        rain_session = rain_session_trials.get(key) or {}
        particle = {
            "available": bool(estimate.get("available")),
            "source": estimate.get("source", "local_session_adaptive_model"),
            "modelVersion": estimate.get("modelVersion", window_pm_predictor.MODEL_VERSION),
            "point": estimate.get("mean"),
            "pointRole": "experimental_window_mean",
            "pointApproximate": True,
            "forecastState": estimate.get("forecastState", "experimental_local_window_mean"),
            "directionalModel": estimate.get("directionalModel") or {},
            "modelSelection": estimate.get("modelSelection") or {},
            "baselinePoint": estimate.get("sensorAnchor"),
            "persistenceAnchorRole": estimate.get("persistenceAnchorRole") or (
                "trailing_5_minute_raw_sensor_median"
                if (estimate.get("freshnessAdjustment") or {}).get("applied")
                else "latest_closed_15_minute_bucket_median"
            ),
            "persistenceAnchorEpoch": estimate.get("sensorReferenceEpoch") or estimate.get("originEpoch"),
            "freshnessAdjustment": estimate.get("freshnessAdjustment") or {},
            "rainContext": estimate.get("rainContext") or estimate.get("rainFeatures") or {},
            "weatherSourceClock": estimate.get("weatherSourceClock") or {},
            "rainLearning": {**(estimate.get("rainLearning") or {}),
                "trialAvailable": bool(rain_session.get("available")),
                "appliedToPrimaryForecast": False,
                "trialModelVersion": rain_session.get("modelVersion"),
                "trialMeanPm25UgM3": rain_session.get("mean"),
                "trialWeatherFetchedEpoch": rain_session.get("weatherFetchedEpoch"),
                "trialForecastIssuedEpoch": rain_session.get("forecastedAtEpoch"),
                "status": "evaluation_only",
                "reason": "fixed_paired_replay_did_not_improve_accuracy"},
            "closedReferenceMeanPm25UgM3": estimate.get("closedMainPrediction"),
            "rawRangeLow": estimate.get("rawRangeLow"),
            "rawRangeHigh": estimate.get("rawRangeHigh"),
            "uncertaintyMethod": estimate.get("uncertaintyMethod", "exact_lead_prequential_mean_residuals"),
            "modelEvidence": estimate.get("modelEvidence") or {},
            "modelNote": estimate.get("sourceCaveat"),
            "regionalCorrectionTrial": {
                "available": bool(trial.get("available")),
                "modelVersion": cams_local_correction.MODEL_VERSION,
                "appliedToPrimaryForecast": False,
                "prospectivelyValidated": False,
                "eligibleForDirectComparison": False,
                "meanPm25UgM3": trial.get("mean"),
                "regionalCorrection": trial.get("regionalCorrection"),
                "modelEvidence": trial.get("modelEvidence") or {},
                "forecastIssuedAt": iso_from_epoch(trial.get("forecastedAtEpoch")),
                "sensorAnchorAt": iso_from_epoch(trial.get("originEpoch")),
                "regionalDataRetrievedAt": iso_from_epoch(trial.get("camsFetchedEpoch")),
                "reason": trial.get("reason"),
            },
            "componentWeights": estimate.get("weights") or {},
            "componentEstimates": estimate.get("components") or {},
            "componentEstimatesRole": estimate.get("componentsRole"),
            "sourceSnapshots": {
                "camsFetchedAt": iso_from_epoch(estimate.get("camsFetchedEpoch")),
                "weatherFetchedAt": iso_from_epoch(estimate.get("weatherFetchedEpoch")),
            },
            "meanSkillEligible": False, "peakSkillEligible": False,
            "validated": False, "usedForComparison": False,
            "usedForDecision": bool(estimate.get("available")),
            "leadHours": estimate.get("leadHours"), "durationHours": 2,
            "confidence": ("Low · experimental direction model" if (estimate.get("directionalModel") or {}).get("applied")
                           else "Low · experimental PM model" if estimate.get("available") else "Insufficient"),
            "method": estimate.get("method") or ("Adaptive local PM ensemble · aligned weather · exact 2-hour session"
                       if (estimate.get("rainContext") or estimate.get("rainFeatures") or {}).get("available")
                       else "Adaptive local PM ensemble · recent-sensor refresh · exact 2-hour session"
                       if (estimate.get("freshnessAdjustment") or {}).get("applied")
                       else "Adaptive local PM ensemble · closed-reference fallback · exact 2-hour session"),
            "message": ("PM model unavailable: " + str(estimate.get("reason", "calculation unavailable")).replace("_", " ")),
        }
        confidence = particle["confidence"]
        if (window["currentApplicable"]
                and shower_signal.get("state") in ("active", "recent", "forming")):
            confidence = "Low · changing particle regime"
        window["particleForecast"] = particle
        window["weatherForecast"]["usedForParticleForecast"] = bool(estimate.get("weatherAvailable"))
        window["weatherForecast"]["pmInputSourceAligned"] = (
            estimate.get("weatherFetchedEpoch") == window["weatherForecast"].get("sourceFetchedEpoch")
            if estimate.get("weatherAvailable") else None
        )
        window["confidence"] = confidence
        issued = int(estimate.get("forecastedAtEpoch") or forecast_epoch)
        window["forecastIssuedEpoch"] = issued
        window["forecastedLabel"] = f'Forecasted at {local_dt(issued):%H:%M}'
        window["recheckLabel"] = window["forecastedLabel"]
    comparison = local_window_pm_comparison(windows, window_prediction)
    regional_payload = latest_air_quality_payload(forecast_epoch)
    regional_weather = latest_weather_payload(forecast_epoch)
    for window in windows.values():
        window["hazeOutlook"] = regional_haze_window_outlook(
            window, forecast_epoch, regional_payload, regional_weather
        )
    # Regional transport remains optional API context, never the comparison headline.
    if forecast_epoch - latest_epoch > SENSOR_UNAVAILABLE_SECONDS:
        comparison.update({
            "ready": False,
            "confidence": "Insufficient",
            "headline": "Sensor data unavailable",
            "summary": "Fresh particle data is required for the window outlook.",
        })

    current = {key: rows[-1][key] for key in
               ("epoch", "pm02", "pm10", "heatindex", "atmp", "rhum")}
    return json_safe({
        "available": True,
        "forecastIssuedEpoch": reference_epoch,
        "dataCoverage": collection_summary(rows, reference_epoch),
        "historyHours": round((latest_epoch - int(first_stored_epoch)) / 3600, 2),
        "current": current,
        "outlook": trend_outlook(rows, shower_signal),
        "airWindow": air_window,
        "shadowForecast": shadow_forecast,
        "rainLearning": rain_trial,
        "forecastPolicy": {
            "id": "neutral_particle_forecast_v10_reviewed_reference_and_routing",
            "nearTermNumericalPolicy": "near_routing_v3_independent_mean_peak",
            "sessionWeightPolicy": window_pm_predictor.WEIGHT_POLICY_VERSION,
            "recentSensorReferenceRefresh": True,
            "aggressive": AGGRESSIVE_FORECAST_ENABLED,
            "weatherUsedForParticleForecast": any(v.get("weatherAvailable") for v in session_estimates.values()),
            "nearTermWeatherUsedForParticleForecast": False,
            "rainLearningAppliedToPrimaryForecast": False,
            "rainLearningStatus": "evaluation_only_no_demonstrated_skill_gain",
            "windowForecastMethod": window_prediction["modelVersion"],
            "windowForecastModels": {key: value.get("modelVersion") for key, value in session_estimates.items()},
            "prescriptiveRecommendation": False,
        },
        "rideForecast": ride_forecast,
        "windowPrediction": window_prediction,
        "showerSignal": shower_signal,
        "weather": weather,
        "windows": windows,
        "comparison": comparison,
        "sampleCount": len(rows),
    })


def analysis_as_of(as_of_epoch, days=RIDE_API_ANALYSIS_DAYS):
    """Public pure replay seam: no sensor, model, weather, or station future data."""
    return analysis(days, as_of_epoch=as_of_epoch)


def iso_from_epoch(epoch):
    if epoch is None:
        return None
    return datetime.fromtimestamp(
        int(epoch), timezone.utc
    ).isoformat().replace("+00:00", "Z")


def compact_weather_window(window):
    result = {"available": bool((window or {}).get("available"))}
    if not result["available"]:
        return result
    result.update({
        "startAt": iso_from_epoch(window.get("startEpoch")),
        "endAt": iso_from_epoch(window.get("endEpoch")),
        "modeledSession": window.get("modeledSession"),
        "forecastFetchedAt": iso_from_epoch(window.get("sourceFetchedEpoch")),
        "usedForParticleForecast": bool(window.get("usedForParticleForecast")),
        "pmInputSourceAligned": window.get("pmInputSourceAligned"),
        "sourcePointCount": window.get("pointCount"),
        "precipitationSourcePointCount": window.get("accumulationPointCount"),
        "precipitationEquivalentHours": window.get("accumulationEquivalentHours"),
        "apparentTemperatureMaxC": window.get("apparentTemperatureMax"),
        "temperatureMaxC": window.get("temperatureMax"),
        "relativeHumidityMeanPct": window.get("relativeHumidityMean"),
        "sky": window.get("skyLabel"),
        "cloudCoverMeanPct": window.get("cloudCoverMean"),
        "cloudCoverMaxPct": window.get("cloudCoverMax"),
        "precipitationProbabilityMaxPct": window.get("precipitationProbabilityMax"),
        "precipitationMm": window.get("precipitationMm"),
        "rainSignal": window.get("rainLabel"),
        "airflow": {
            "context": window.get("ventilationLabel"),
            "modeledWind10mMeanKmh": window.get("windSpeed10mMean"),
            "modeledWindGust10mMaxKmh": window.get("windGust10mMax"),
            "modeledWindDirection10mDegrees": window.get("windDirection10m"),
            "modeledWindDirection10m": window.get(
                "windDirection10mCompass"
            ),
            "modeledWind180mMeanKmh": window.get("windSpeed180mMean"),
            "modeledDirection180m": window.get("windDirection180mCompass"),
            "usedForParticleForecast": bool(window.get("usedForParticleForecast")),
        },
    })
    return result


def compact_regional_transport(transport):
    """Expose bounded shadow evidence without publishing diagnostic telemetry."""
    transport = transport or {}
    evidence = transport.get("evidence") or {}
    validation = evidence.get("validation") or {}
    current = transport.get("current") or {}
    by_lag = evidence.get("byLag") or {}
    return {
        "available": bool(transport),
        "state": transport.get("state"),
        "label": transport.get("label"),
        "reason": transport.get("reason"),
        "usedForParticleForecast": False,
        "sourceBearingDegrees": transport.get("sourceBearing"),
        "sourceAssumption": transport.get("sourceAssumption"),
        "preRegisteredLagHours": transport.get("preRegisteredLagHours") or [],
        "current": {
            "available": bool(current),
            "originAt": iso_from_epoch(current.get("originEpoch")),
            "modeledWindSpeed925hPaKmh": current.get("windSpeed925hPa"),
            "modeledWindDirection925hPaDegrees": current.get(
                "windDirection925hPa"
            ),
            "modeledWindDirection925hPa": current.get(
                "windDirection925hPaCompass"
            ),
            "sourceAlignment925hPa": current.get("sourceAlignment925"),
            "sourceComponent925hPaKmh": current.get("sourceComponent925"),
        },
        "evidence": {
            "issuedForecastDays": evidence.get("issuedForecastDays", 0),
            "issuedOriginHours": evidence.get("issuedOriginHours", 0),
            "lowAlignment925hPaOrigins": evidence.get(
                "counterfactualOrigins", 0
            ),
            "lowAlignmentDefinition": evidence.get(
                "counterfactualDefinition"
            ),
            "completedPairsByLagHours": {
                str(lag): (by_lag.get(str(lag)) or {}).get("completedPairs", 0)
                for lag in TRANSPORT_SHADOW_LAGS_HOURS
            },
            "walkForwardScoredOriginsByLagHours": {
                str(lag): (
                    ((by_lag.get(str(lag)) or {}).get("walkForward") or {})
                    .get("scoredOrigins", 0)
                )
                for lag in TRANSPORT_SHADOW_LAGS_HOURS
            },
            "validation": {
                "eligible": bool(validation.get("eligible")),
                "stableLagCount": validation.get("stableLagCount", 0),
                "requiredStableLags": validation.get(
                    "requiredStableLags", TRANSPORT_SHADOW_MIN_STABLE_LAGS
                ),
                "reasons": validation.get("reasons") or [],
            },
        },
    }


def compact_window_baseline(baseline):
    baseline = baseline or {}

    def distribution(key):
        values = baseline.get(key) or {}
        return {
            "recencyWeightedMedian": values.get("median"),
            "unweightedObservedQ1": values.get("q1"),
            "unweightedObservedQ3": values.get("q3"),
        }

    return {
        "available": bool(baseline.get("available")),
        "validDays": baseline.get("validDays", 0),
        "pairedDays": baseline.get("pairedDays", 0),
        "coveragePct": baseline.get("coveragePct"),
        "pm25MeanUgM3": distribution("pmMean"),
        "pm25WorstHourUgM3": distribution("pmWorstHour"),
        "heatIndexPeakC": distribution("heatPeak"),
    }


def compact_window_particle_forecast(forecast):
    forecast = forecast or {}
    candidate = forecast.get("candidate") or {}
    forecast_state = str(forecast.get("forecastState") or "")
    point_role = forecast.get("pointRole")
    is_session_model = point_role == "experimental_window_mean"
    has_model_point = bool(forecast.get("available") and num(forecast.get("point")) is not None and point_role in {
        "validated_forecast", "aggressive_cams_forecast",
        "provisional_cams_context", "aggressive_local_ridge",
        "aggressive_rapid_clearance_event", "experimental_window_mean",
    })
    has_peak_model_point = bool(
        forecast.get("peakApproximate") or has_model_point
    )
    validation_state = (
        "validated" if forecast.get("validated") else
        "mixed_persistence_mean_experimental_peak"
        if (forecast.get("peakApproximate") and not has_model_point) else
        "experimental_not_validated"
        if (forecast_state.startswith("aggressive_")
            or forecast_state.startswith("experimental_")
            or forecast_state.startswith("local_near_term_experimental")) else
        "provisional_context_only"
        if forecast_state.startswith("provisional_") else
        "persistence_only"
    )
    comparison_low = num(forecast.get("comparisonRangeLow"))
    comparison_high = num(forecast.get("comparisonRangeHigh"))
    comparison_interval_available = bool(
        comparison_low is not None and comparison_high is not None
        and comparison_high > comparison_low
    )
    return {
        "available": bool(forecast.get("available")),
        "modelPointApplied": bool(has_model_point),
        "forecastState": forecast.get("forecastState"),
        "pointRole": point_role,
        "method": forecast.get("method"),
        "modelVersion": forecast.get("modelVersion"),
        "directionalModel": forecast.get("directionalModel") or {},
        "modelSelection": forecast.get("modelSelection") or {},
        "confidence": confidence_code(forecast.get("confidence")),
        "confidenceDetail": forecast.get("confidence"),
        "baselineMeanPm25UgM3": forecast.get("baselinePoint"),
        "persistenceAnchorRole": forecast.get(
            "persistenceAnchorRole", "latest_raw_sensor_reading"
        ),
        "persistenceAnchorAt": iso_from_epoch(
            forecast.get("persistenceAnchorEpoch")
        ),
        "projectedMeanPm25UgM3": (
            forecast.get("point") if has_model_point else None
        ),
        "projectedPeakPm25UgM3": (
            forecast.get("projectedPeak") if has_peak_model_point else None
        ),
        "approximate": bool(forecast.get("pointApproximate")),
        "validationState": validation_state,
        "validatedModelPointApplied": bool(forecast.get("validated")),
        "planningEstimate": {
            "available": bool(has_model_point or has_peak_model_point),
            "meanPm25UgM3": forecast.get("point") if has_model_point else None,
            "peakPm25UgM3": (
                forecast.get("projectedPeak") if has_peak_model_point else None
            ),
            "role": point_role,
            "validationState": validation_state,
            "eligibleForDirectComparison": bool(
                has_model_point and forecast.get(
                    "usedForComparison", forecast.get("usedForDecision")
                )
            ),
        },
        "modelEvidence": forecast.get("modelEvidence") or {},
        "modelNote": forecast.get("modelNote"),
        "freshnessAdjustment": forecast.get("freshnessAdjustment") or {},
        "rainContext": forecast.get("rainContext") or {},
        "weatherSourceClock": forecast.get("weatherSourceClock") or {},
        "rainLearning": forecast.get("rainLearning") or {},
        "componentWeights": forecast.get("componentWeights") or {},
        "componentEstimatesPm25UgM3": forecast.get("componentEstimates") or {},
        "componentEstimatesRole": forecast.get("componentEstimatesRole"),
        "closedReferenceMeanPm25UgM3": forecast.get("closedReferenceMeanPm25UgM3"),
        "sourceSnapshots": forecast.get("sourceSnapshots") or {},
        "regionalCorrectionTrial": forecast.get("regionalCorrectionTrial"),
        "uncertainty": {
            "state": (
                "collecting" if forecast.get("rawRangeLow") is None else
                "uncalibrated_replay_errors" if is_session_model else
                "uncalibrated_matched_history"
                if forecast.get("rawRangeLow") is not None else "collecting"
            ),
            "displayRole": "secondary_explanation",
            "typicalMatchedHistoryBandPm25UgM3": {
                "low": forecast.get("typicalRangeLow"),
                "high": forecast.get("typicalRangeHigh"),
                "quantiles": [0.25, 0.75],
                "calibrated": False,
                "role": "descriptive_middle_half_of_matched_outcomes",
            },
            "conservativeMatchedHistorySpanPm25UgM3": {
                "low": forecast.get("rawRangeLow"),
                "high": forecast.get("rawRangeHigh"),
                "quantiles": [0.10, 0.90],
                "nominalCoverage": None,
                "calibrated": False,
                "role": "uncalibrated_historical_error_span",
            },
            "highSidePeakMarkerPm25UgM3": forecast.get("peakUpper"),
            "highSidePeakQuantile": None,
            "highSidePeakMarkerRole": (
                "persistence_anchor_containing_marker_not_a_quantile"
            ),
            "empiricalPeakQ90Pm25UgM3": {
                "value": forecast.get("rawPeakUpper90"),
                "quantile": 0.90,
                "nominalCoverage": None,
                "calibrated": False,
                "role": "uncalibrated_empirical_matched_history_quantile",
            },
            "note": (
                "No model-specific prediction interval is available; the previous model's residual range is not reused."
                if (forecast.get("directionalModel") or {}).get("applied") else
                "Exact-lead historical replay residuals for the session mean; overlapping outcomes, not a calibrated prediction interval."
                if is_session_model else
                "Descriptive matched-history evidence; not the planning estimate "
                "and not a calibrated prediction interval."
            ),
        },
        "meanRangePm25UgM3": {
            "low": forecast.get("rawRangeLow"),
            "high": forecast.get("rawRangeHigh"),
            "quantiles": [0.10, 0.90],
            "nominalCoverage": None,
            "calibrated": False,
            "role": "uncalibrated_historical_error_span",
            "displayRole": "secondary_explanation",
            "deprecated": True,
        },
        "upperPeakPm25UgM3": forecast.get("peakUpper"),
        "uncertaintyMethod": forecast.get("uncertaintyMethod"),
        "comparisonIntervalPm25UgM3": {
            "available": comparison_interval_available,
            "state": (
                forecast.get("comparisonIntervalState")
                or ("available_uncalibrated_empirical_span"
                    if comparison_interval_available else "not_available")
            ),
            "low": comparison_low if comparison_interval_available else None,
            "high": comparison_high if comparison_interval_available else None,
            "upperPeak": (
                forecast.get("comparisonPeakUpper")
                if comparison_interval_available else None
            ),
            "method": forecast.get("comparisonIntervalMethod"),
            "nominalCoverage": None,
            "calibrated": False,
            "role": "task_specific_empirical_residual_span",
            "usedOnlyWhenTaskValidated": True,
        },
        "leadHours": forecast.get("leadHours"),
        "durationHours": forecast.get("durationHours"),
        "support": {
            "originCount": forecast.get("originCount"),
            "independentOriginCount": forecast.get("independentOriginCount"),
            "matchedCount": forecast.get("matchedCount"),
            "matchedDistinctDays": forecast.get("matchedDistinctDays"),
        },
        "experimentalCamsCandidate": {
            "available": bool(candidate.get("available")),
            "modelPointApplied": bool(candidate.get("usedForDecision")),
            "usedForDisplay": bool(candidate.get("usedForDisplay")),
            "source": candidate.get("source"),
            "fetchedAt": iso_from_epoch(candidate.get("fetchedEpoch")),
            "ageMinutes": candidate.get("ageMinutes"),
            "modelCurrentAgeMinutes": candidate.get("modelCurrentAgeMinutes"),
            "issuedAgeEligible": bool(candidate.get("issuedAgeEligible")),
            "validationState": candidate.get("validationState"),
            "suppressedByRegimeBreak": bool(
                candidate.get("suppressedByRegimeBreak")
            ),
            "sensorModelLevelDifferenceUgM3": candidate.get(
                "sensorModelLevelDifference"
            ),
            "sensorModelLevelDifferenceRole": candidate.get(
                "sensorModelLevelDifferenceRole"
            ),
            "regimeSuppressionReason": candidate.get("regimeSuppressionReason"),
            "regimeSuppressionHorizonHours": candidate.get(
                "regimeSuppressionHorizonHours"
            ),
            "validationTask": candidate.get("validationTask"),
            "leadValidated": bool(candidate.get("leadValidated")),
            "sensorAnchoredMeanPm25UgM3": candidate.get("mean"),
            "sensorAnchoredPeakPm25UgM3": candidate.get("peak"),
            "rawModelMeanPm25UgM3": candidate.get("rawModelMean"),
            "rawModelPeakPm25UgM3": candidate.get("rawModelPeak"),
            "deltaShrink": candidate.get("deltaShrink"),
        },
    }


def confidence_code(value):
    text = str(value or "").strip().lower()
    for level in ("high", "medium", "low", "insufficient"):
        if text.startswith(level):
            return level
    return "insufficient"


def exposure_method_id(value):
    methods = {
        "Rain-aware local PM model": "rain_nearterm_ridge_v1",
        "Recent-sensor persistence with matched historical errors":
            "fresh_sensor_persistence_v1",
        "Persistence baseline with similar-level empirical errors":
            "persistence_similar_level_errors_v1",
        "Gated damped momentum with prior fast-rise errors":
            "gated_momentum_fast_rise_errors_v1",
        "Persistence with prior fast-rise event errors":
            "persistence_fast_rise_event_errors_v2",
        "Persistence with prior joint-washout errors":
            "persistence_joint_clearance_errors_v1",
        "Persistence with prior joint-washout event errors":
            "persistence_joint_clearance_event_errors_v2",
        "Persistence with similar-level trail errors":
            "persistence_similar_level_trail_errors_v1",
        "Damped aggressive local analogue with persistence envelope":
            "aggressive_local_analogue_arrival_v1",
        "Aggressive local analogue mean with Ridge peak":
            "aggressive_local_analogue_trail_v1",
        "Aggressive rapid-clearance event analogue with persistence envelope":
            "aggressive_rapid_clearance_event_v1",
        "Replay-screened damped local analogue with historical error bounds":
            "replay_screened_local_analogue_arrival_v3",
        "Experimental local analogue mean and bounded analogue peak":
            "replay_screened_local_analogue_mean_peak_v3",
        "Persistence mean with experimental bounded analogue peak":
            "replay_screened_local_analogue_peak_only_v3",
        "Experimental robust Ridge mean and bounded analogue peak":
            "issued_screened_ridge_mean_analogue_peak_v1",
        "Scale-normalized rapid-clearance event analogue with historical error bounds":
            "scale_normalized_rapid_clearance_event_v2",
    }
    return methods.get(value, "unavailable")


def ride_api_links():
    """Canonical relative links keep localhost and LAN clients portable."""
    return {
        "self": "/api/v1/mtb/environment-evidence",
        "openapi": "/api/openapi.json",
        "jsonSchema": "/api/v1/mtb/environment-evidence/schema",
        "documentation": "/docs/coach-api.md",
        "sourceCurrent": "/api/current",
        "sourceAnalysis": "/api/analysis?days=28",
    }


def coach_api_document_bytes():
    """Read the living Coach guide on every request so edits need no restart."""
    try:
        return COACH_API_DOC_PATH.read_bytes()
    except OSError:
        return (
            "# MTB environmental evidence API\n\n"
            "The documentation file is temporarily unavailable. Discover the "
            "machine contract at `/api/openapi.json`.\n"
        ).encode("utf-8")


def ride_evidence_json_schema():
    """Stable JSON Schema for the public environmental evidence boundary."""
    nullable_number = {"type": ["number", "null"]}
    nullable_string = {"type": ["string", "null"]}
    empirical_range_schema = {
        "type": ["object", "null"],
        "description": (
            "Raw finite-sample empirical endpoints. This is descriptive and "
            "is not a calibrated prediction interval."
        ),
        "properties": {
            "low": nullable_number,
            "high": nullable_number,
            "quantiles": {
                "type": "array",
                "prefixItems": [{"const": 0.10}, {"const": 0.90}],
                "minItems": 2,
                "maxItems": 2,
            },
            "nominalCoverage": {"type": "null"},
            "calibrated": {"const": False},
        },
        "additionalProperties": True,
    }
    display_envelope_schema = {
        "type": ["object", "null"],
        "description": (
            "Display envelope widened to contain named anchors or scenarios; "
            "it has no quantile or coverage interpretation."
        ),
        "properties": {
            "low": nullable_number,
            "high": nullable_number,
            "containsPersistenceAnchor": {"type": "boolean"},
            "containsBaseline": {"type": "boolean"},
            "containsExperimentalProjection": {"type": "boolean"},
            "coverageClaimed": {"const": False},
            "role": nullable_string,
        },
        "additionalProperties": True,
    }
    empirical_upper_schema = {
        "type": ["object", "null"],
        "description": (
            "Raw finite-sample empirical upper quantile; not an anchor-expanded "
            "display bound and not calibrated."
        ),
        "properties": {
            "value": nullable_number,
            "quantile": {"const": 0.90},
            "finiteSampleRank": {"type": "boolean"},
            "finiteSampleRankCoverage": nullable_number,
            "calibrated": {"const": False},
            "role": nullable_string,
        },
        "additionalProperties": True,
    }
    issue_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string"},
            "severity": {
                "type": "string",
                "enum": ["warning", "error"],
            },
            "message": {"type": "string"},
        },
        "additionalProperties": True,
    }
    exposure_schema = {
        "type": "object",
        "required": ["available"],
        "properties": {
            "available": {"type": "boolean"},
            "offsetMinutes": {"type": ["integer", "null"], "description": "Offset from forecastIssuedAt, not response delivery."},
            "startOffsetMinutes": {"type": ["integer", "null"], "description": "Offset from forecastIssuedAt, not response delivery."},
            "endOffsetMinutes": {"type": ["integer", "null"], "description": "Offset from forecastIssuedAt, not response delivery."},
            "expectedAt": {
                "type": ["string", "null"], "format": "date-time"
            },
            "startAt": {"type": ["string", "null"], "format": "date-time"},
            "endAt": {"type": ["string", "null"], "format": "date-time"},
            "forecastIssuedAt": {"type": ["string", "null"], "format": "date-time"},
            "targetVersion": nullable_string,
            "targetDefinition": nullable_string,
            "meanTargetDefinition": nullable_string,
            "peakTargetDefinition": nullable_string,
            "basedOnObservedAt": {
                "type": ["string", "null"], "format": "date-time"
            },
            "persistenceAnchorPm25UgM3": nullable_number,
            "persistenceAnchorRole": nullable_string,
            "persistenceAnchorAt": {
                "type": ["string", "null"], "format": "date-time"
            },
            "modelFeatureAnchorPm25UgM3": nullable_number,
            "freshnessAdjustment": {
                "type": "object",
                "description": "Experimental observed-reference refresh; does not anticipate sudden events. The feature anchor and recent reference timestamps differ.",
                "properties": {
                    "applied": {"type": "boolean"},
                    "amountUgM3": nullable_number,
                    "featureAnchorEpoch": nullable_number,
                    "referenceEpoch": nullable_number,
                    "referenceCount": {"type": ["integer", "null"]},
                    "prospectivelyValidated": {"const": False},
                },
                "additionalProperties": True,
            },
            "projectedPm25UgM3": nullable_number,
            "projectedMeanPm25UgM3": nullable_number,
            "projectedPeakPm25UgM3": nullable_number,
            "baselinePm25UgM3": {
                **nullable_number,
                "deprecated": True,
                "description": "Deprecated 1.x alias for the persistence anchor.",
            },
            "baselineMeanPm25UgM3": {
                **nullable_number,
                "deprecated": True,
                "description": "Deprecated 1.x alias for the persistence anchor.",
            },
            "pointRole": nullable_string,
            "forecastState": nullable_string,
            "approximate": {"type": "boolean"},
            "validationState": nullable_string,
            "meanValidationState": nullable_string,
            "peakValidationState": nullable_string,
            "likelyRangePm25UgM3": empirical_range_schema,
            "likelyMeanRangePm25UgM3": empirical_range_schema,
            "forecastRangePm25UgM3": display_envelope_schema,
            "forecastMeanRangePm25UgM3": display_envelope_schema,
            "forecastUpperPm25UgM3": nullable_number,
            "forecastPeakUpperPm25UgM3": nullable_number,
            "upperPm25UgM3": empirical_upper_schema,
            "upperMeanPm25UgM3": empirical_upper_schema,
            "upperPeakPm25UgM3": empirical_upper_schema,
            "decisionEnvelopePm25UgM3": {
                **display_envelope_schema,
                "deprecated": True,
            },
            "decisionMeanEnvelopePm25UgM3": {
                **display_envelope_schema,
                "deprecated": True,
            },
            "decisionUpperPm25UgM3": {
                **nullable_number,
                "deprecated": True,
            },
            "decisionPeakRiskMarkerPm25UgM3": {
                **nullable_number,
                "deprecated": True,
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low", "insufficient"],
            },
            "confidenceDetail": nullable_string,
            "methodId": {"type": "string"},
            "rapidClearanceEventSupport": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "support": {"type": "object", "additionalProperties": True},
        },
        "additionalProperties": True,
    }
    ride_particle_schema = {
        "type": ["object", "null"],
        "properties": {
            "modelVersion": nullable_string,
            "modelSelection": {
                "type": "object", "additionalProperties": True,
                "description": "Actual selected estimator, scope/fallback reason and original issue. Scheduled handoff is not measured PM movement; cached points retain their estimator and target clocks.",
            },
            "directionalModel": {
                "type": "object",
                "description": "Scoped experimental same-day 14:00–16:00 direction model for 07:00–08:00 Malaysia issues. Empty outside that estimator. Development improvement is not prospective validation; false clearing remains possible.",
                "properties": {
                    "applied": {"type": "boolean"},
                    "direction": {"enum": ["rising", "falling", "steady"]},
                    "changeFromFreshUgM3": nullable_number,
                    "experimental": {"const": True},
                    "prospectivelyValidated": {"const": False},
                    "scope": {"type": "object", "additionalProperties": True},
                    "retrospectiveReview": {"type": "object", "additionalProperties": True},
                },
                "additionalProperties": True,
            },
            "regionalCorrectionTrial": {
                "type": ["object", "null"],
                "description": (
                    "Separate experimental CAMS plus learned local correction. "
                    "Never substitutes for projectedMeanPm25UgM3 or comparison. "
                    "Retrieval time is not native CAMS model initialization time."
                ),
                "properties": {
                    "available": {"type": "boolean"},
                    "modelVersion": nullable_string,
                    "meanPm25UgM3": nullable_number,
                    "appliedToPrimaryForecast": {"const": False},
                    "prospectivelyValidated": {"const": False},
                    "eligibleForDirectComparison": {"const": False},
                    "forecastIssuedAt": {"type": ["string", "null"], "format": "date-time"},
                    "sensorAnchorAt": {"type": ["string", "null"], "format": "date-time"},
                    "regionalDataRetrievedAt": {"type": ["string", "null"], "format": "date-time"},
                    "reason": nullable_string,
                    "modelEvidence": {"type": "object", "additionalProperties": True},
                    "regionalCorrection": {
                        "type": ["object", "null"],
                        "description": "Trial mean = max(0, regional mean + signed learned correction), allowing rounding. Not an additional fire contribution.",
                        "properties": {
                            "regionalMeanPm25UgM3": nullable_number,
                            "learnedCorrectionPm25UgM3": nullable_number,
                            "unboundedMeanPm25UgM3": nullable_number,
                            "nonnegativeFloorApplied": {"type": "boolean"},
                            "currentRegionalPm25UgM3": nullable_number,
                            "currentLocalCorrectionPm25UgM3": nullable_number,
                            "method": nullable_string,
                        },
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            },
            "uncertainty": {
                "type": ["object", "null"],
                "properties": {
                    "highSidePeakQuantile": {
                        "type": "null",
                        "deprecated": True,
                        "description": (
                            "Always null: the adjacent marker may be widened "
                            "and has no quantile interpretation."
                        ),
                    },
                    "highSidePeakMarkerRole": nullable_string,
                    "empiricalPeakQ90Pm25UgM3": empirical_upper_schema,
                },
                "additionalProperties": True,
            },
            "meanRangePm25UgM3": {
                **empirical_range_schema,
                "deprecated": True,
            },
            "comparisonIntervalPm25UgM3": {
                "type": ["object", "null"],
                "properties": {
                    "available": {"type": "boolean"},
                    "state": nullable_string,
                    "low": nullable_number,
                    "high": nullable_number,
                    "upperPeak": nullable_number,
                    "nominalCoverage": {"type": "null"},
                    "calibrated": {"const": False},
                    "role": {"const": "task_specific_empirical_residual_span"},
                    "usedOnlyWhenTaskValidated": {"const": True},
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }
    ride_window_schema = {
        "type": "object",
        "required": ["forecastIssuedAt", "forecastedLabel"],
        "properties": {
            "targetDay": {**nullable_string, "description": "Display day relative to the original forecast issue; startAt/endAt retain the absolute session date."},
            "startAt": {"type": ["string", "null"], "format": "date-time"},
            "endAt": {"type": ["string", "null"], "format": "date-time"},
            "rideWindow": nullable_string,
            "modeledSession": nullable_string,
            "leadHours": nullable_number,
            "active": {"type": "boolean"},
            "currentConditionsApplicable": {"type": "boolean"},
            "decisionAt": {
                "type": ["string", "null"], "format": "date-time",
                "deprecated": True,
            },
            "decisionLabel": {
                **nullable_string,
                "deprecated": True,
            },
            "logisticsReferenceAt": {
                "type": ["string", "null"], "format": "date-time"
            },
            "logisticsReferenceLabel": nullable_string,
            "forecastIssuedAt": {
                "type": ["string", "null"], "format": "date-time"
            },
            "forecastedLabel": nullable_string,
            "recheck": {
                **nullable_string,
                "deprecated": True,
                "description": (
                    "Deprecated 1.x display alias of forecastedLabel; it no "
                    "longer communicates a future action time."
                ),
            },
            "confidence": nullable_string,
            "particleForecast": ride_particle_schema,
            "hazeOutlook": {
                "type": "object",
                "description": "Regional CAMS direction and simplified air-path context; never a TTDI concentration or smoke-source attribution.",
                "required": ["available", "role", "usedForLocalPmPoint", "validatedAtTTDI"],
                "properties": {
                    "available": {"type": "boolean"},
                    "role": {"const": "regional_model_outlook_not_local_concentration"},
                    "usedForLocalPmPoint": {"const": False},
                    "validatedAtTTDI": {"const": False},
                    "sourceAttributionEstablished": {"const": False},
                    "regionalMeanPm25UgM3": nullable_number,
                    "regionalCurrentPm25UgM3": nullable_number,
                    "regionalChangePct": nullable_number,
                    "direction": nullable_string,
                    "airPath": {"type": "object", "additionalProperties": True},
                },
                "additionalProperties": True,
            },
            "weatherForecast": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "observedHistory": {
                "type": ["object", "null"],
                "additionalProperties": True,
            },
            "historicalBaseline": {
                "type": ["object", "null"],
                "deprecated": True,
                "description": "Deprecated 1.x alias of observedHistory.",
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "/api/v1/mtb/environment-evidence/schema",
        "title": "Bukit Kiara MTB environmental evidence",
        "description": (
            "Live air-quality and weather evidence for an external consumer. "
            "It does not select a ride window or contain a training prescription."
        ),
        "type": "object",
        "required": [
            "schemaVersion", "kind", "generatedAt", "evidenceId",
            "contract", "boundary", "status", "links",
        ],
        "properties": {
            "schemaVersion": {"const": RIDE_API_SCHEMA_VERSION},
            "kind": {"const": "mtb_environment_evidence"},
            "generatedAt": {"type": "string", "format": "date-time"},
            "evidenceId": {"type": ["string", "null"]},
            "contract": {
                "type": "object",
                "required": [
                    "schemaVersion", "revision", "openapi", "jsonSchema",
                    "documentation", "revisionRole", "evidencePollSeconds",
                    "refreshPolicy",
                ],
                "properties": {
                    "schemaVersion": {"const": RIDE_API_SCHEMA_VERSION},
                    "revision": {
                        "type": "string",
                        "pattern": "^sha256:[0-9a-f]{16}$",
                        "description": (
                            "Opaque equality token for the schema, service "
                            "description, and guide; "
                            "never an evidence-freshness marker."
                        ),
                    },
                    "openapi": {"const": "/api/openapi.json"},
                    "jsonSchema": {
                        "const": "/api/v1/mtb/environment-evidence/schema"
                    },
                    "documentation": {"const": "/docs/coach-api.md"},
                    "revisionRole": {
                        "const": "contract_and_documentation_only"
                    },
                    "evidencePollSeconds": {
                        "type": "integer",
                        "minimum": 60,
                        "description": (
                            "Recommended passive GET cadence; polling does not "
                            "trigger source collection."
                        ),
                    },
                    "refreshPolicy": {
                        "type": "object",
                        "required": [
                            "mode", "useIfNoneMatch", "reloadWhen", "resources"
                        ],
                        "properties": {
                            "mode": {"const": "conditional_get"},
                            "useIfNoneMatch": {"const": True},
                            "reloadWhen": {
                                "type": "array",
                                "items": {
                                    "enum": ["schemaVersion", "revision"]
                                },
                            },
                            "resources": {
                                "type": "array",
                                "description": (
                                    "Only these advertised contract resources "
                                    "support conditional GET."
                                ),
                                "items": {
                                    "enum": [
                                        "/api/openapi.json",
                                        "/api/v1/mtb/environment-evidence/schema",
                                        "/docs/coach-api.md"
                                    ]
                                },
                            },
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
            "location": {"type": "object", "additionalProperties": True},
            "logistics": {"type": "object", "additionalProperties": True},
            "boundary": {
                "type": "object",
                "properties": {
                    "role": {"const": "environmental_evidence"},
                    "trainingPrescriptionIncluded": {"const": False},
                    "historicalSeriesIncluded": {"const": False},
                },
                "additionalProperties": True,
            },
            "status": {
                "type": "object",
                "required": ["state", "usable", "fresh", "issues"],
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["ok", "degraded", "unavailable"],
                    },
                    "usable": {"type": "boolean"},
                    "fresh": {
                        "type": "boolean",
                        "description": "Sensor is within its freshness limit, the original forecast issue is at most 120 seconds old, and no analysis update error is active. Response generation time does not establish freshness.",
                    },
                    "forecastAgeSeconds": {
                        "type": ["integer", "null"], "minimum": 0,
                        "description": "Seconds since the original forecast issue, measured at response delivery; null while no forecast exists.",
                    },
                    "analysisUpdating": {"type": "boolean"},
                    "message": {"type": "string"},
                    "issues": {"type": "array", "items": issue_schema},
                },
                "additionalProperties": True,
            },
            "observation": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "required": ["id", "timestamp", "ageSeconds", "particles"],
                        "properties": {
                            "id": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "epoch": {"type": "integer"},
                            "ageSeconds": {"type": "integer", "minimum": 0},
                            "particles": {
                                "type": "object",
                                "properties": {
                                    "pm25UgM3": nullable_number,
                                    "pm10UgM3": nullable_number,
                                },
                                "additionalProperties": True,
                            },
                            "heat": {
                                "type": "object",
                                "properties": {
                                    "heatIndexC": nullable_number,
                                    "temperatureC": nullable_number,
                                    "relativeHumidityPct": nullable_number,
                                },
                                "additionalProperties": True,
                            },
                        },
                        "additionalProperties": True,
                    },
                ]
            },
            "particleNowcast": {
                "type": "object",
                "required": ["available"],
                "properties": {
                    "available": {"type": "boolean"},
                    "role": {"const": "observed_recent_movement_not_future_direction"},
                    "state": nullable_string,
                    "label": nullable_string,
                    "change30MinutesUgM3": nullable_number,
                    "change60MinutesUgM3": nullable_number,
                    "fastRise": {"type": "boolean"},
                    "sustainedImprovement": {"type": "boolean"},
                    "recheckMinutes": {"type": ["integer", "null"]},
                },
                "additionalProperties": True,
            },
            "exposureOutlook": {
                "type": "object",
                "properties": {
                    "arrival": exposure_schema,
                    "onTrail": exposure_schema,
                },
                "additionalProperties": True,
            },
            "dataCoverage": {"type": "object", "additionalProperties": True},
            "weather": {
                "type": "object",
                "required": ["available"],
                "properties": {
                    "available": {"type": "boolean"},
                    "forecastSource": nullable_string,
                    "forecastFetchedAt": {
                        "type": ["string", "null"], "format": "date-time"
                    },
                    "forecastAgeMinutes": nullable_number,
                    "trailPeriod": {
                        "type": ["object", "null"],
                        "additionalProperties": True,
                    },
                    "regionalTransport": {
                        "type": "object",
                        "required": [
                            "available", "usedForParticleForecast", "current",
                            "evidence",
                        ],
                        "properties": {
                            "available": {"type": "boolean"},
                            "state": nullable_string,
                            "label": nullable_string,
                            "reason": nullable_string,
                            "usedForParticleForecast": {"const": False},
                            "sourceBearingDegrees": nullable_number,
                            "preRegisteredLagHours": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "current": {
                                "type": "object",
                                "required": ["available"],
                                "properties": {
                                    "available": {"type": "boolean"},
                                    "originAt": {
                                        "type": ["string", "null"],
                                        "format": "date-time",
                                    },
                                    "modeledWindSpeed925hPaKmh": nullable_number,
                                    "modeledWindDirection925hPaDegrees": nullable_number,
                                    "modeledWindDirection925hPa": nullable_string,
                                    "sourceAlignment925hPa": nullable_number,
                                    "sourceComponent925hPaKmh": nullable_number,
                                },
                                "additionalProperties": True,
                            },
                            "evidence": {
                                "type": "object",
                                "properties": {
                                    "issuedForecastDays": {
                                        "type": "integer", "minimum": 0
                                    },
                                    "issuedOriginHours": {
                                        "type": "integer", "minimum": 0
                                    },
                                    "lowAlignment925hPaOrigins": {
                                        "type": "integer", "minimum": 0
                                    },
                                    "validation": {
                                        "type": "object",
                                        "properties": {
                                            "eligible": {"type": "boolean"},
                                            "stableLagCount": {
                                                "type": "integer", "minimum": 0
                                            },
                                            "requiredStableLags": {
                                                "type": "integer", "minimum": 0
                                            },
                                            "reasons": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                        "additionalProperties": True,
                                    },
                                },
                                "additionalProperties": True,
                            },
                        },
                        "description": (
                            "Causal, issued-forecast regional-wind evidence in "
                            "shadow mode. It never alters PM2.5 forecasts."
                        ),
                        "additionalProperties": True,
                    },
                    "subangReference": {
                        "type": "object",
                        "required": ["role", "usedForParticleForecast"],
                        "properties": {
                            "role": {"const": "corroborating_context"},
                            "usedForParticleForecast": {"const": False},
                        },
                        "description": (
                            "Observed nearby weather context and provenance; "
                            "not an input to PM2.5 point or range forecasts."
                        ),
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            },
            "clearanceEvent": {
                "type": "object",
                "required": ["detected"],
                "properties": {
                    "detected": {"type": "boolean"},
                    "state": nullable_string,
                    "kind": nullable_string,
                    "label": nullable_string,
                    "detectionMode": nullable_string,
                    "eventConditionedForecast": {
                        "type": ["object", "null"],
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            },
            "rideWindows": {
                "type": "object",
                "properties": {
                    "comparison": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                    "morning": ride_window_schema,
                    "afternoon": ride_window_schema,
                },
                "additionalProperties": True,
            },
            "evidenceQuality": {
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "limitations": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": True},
                    },
                },
                "additionalProperties": True,
            },
            "provenance": {
                "type": "object",
                "properties": {
                    "analysisComputedAt": {"type": ["string", "null"], "format": "date-time"},
                    "analysisForecastIssuedAt": {
                        "type": ["string", "null"], "format": "date-time",
                        "description": "Frozen calculation issue time. Forecast targets and analytical state retain this reference when a completed snapshot is delivered again.",
                    },
                    "analysisDelivery": {
                        "type": "object",
                        "description": "Live snapshot delivery status, separate from sensor/source clocks and model accuracy. Historical replay omits this field.",
                        "properties": {
                            "state": {"enum": ["ready", "updating", "initializing", "error", "expired"]},
                            "updating": {"type": "boolean"},
                            "forecastAgeSeconds": {"type": ["number", "null"], "minimum": 0},
                            "servedAtEpoch": {"type": "integer"},
                            "maximumAgeSeconds": {"const": 600},
                            "retryAfterSeconds": {"type": ["number", "null"], "minimum": 0},
                            "error": nullable_string,
                            "diagnostics": {"enum": ["pending", "complete", "error"]},
                        },
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": True,
            },
            "links": {
                "type": "object",
                "required": [
                    "self", "openapi", "jsonSchema", "documentation"
                ],
                "properties": {
                    "self": {"const": "/api/v1/mtb/environment-evidence"},
                    "openapi": {"const": "/api/openapi.json"},
                    "jsonSchema": {
                        "const": "/api/v1/mtb/environment-evidence/schema"
                    },
                    "documentation": {"const": "/docs/coach-api.md"},
                    "sourceCurrent": {"type": "string"},
                    "sourceAnalysis": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "allOf": [{
            "if": {
                "properties": {
                    "status": {
                        "properties": {
                            "state": {"enum": ["ok", "degraded"]}
                        },
                        "required": ["state"],
                    }
                },
                "required": ["status"],
            },
            "then": {
                "required": [
                    "location", "logistics", "observation", "particleNowcast",
                    "exposureOutlook", "weather", "clearanceEvent",
                    "rideWindows", "evidenceQuality", "provenance",
                ],
                "properties": {
                    "observation": {"type": "object"},
                    "exposureOutlook": {
                        "required": ["arrival", "onTrail"]
                    },
                    "rideWindows": {
                        "required": ["comparison", "morning", "afternoon"]
                    },
                },
            },
        }],
        "additionalProperties": True,
    }


def canonical_json_bytes(value):
    return json.dumps(
        json_safe(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def coach_api_contract_revision(document_bytes=None):
    document_bytes = (
        coach_api_document_bytes() if document_bytes is None else document_bytes
    )
    material = (
        canonical_json_bytes(ride_evidence_json_schema())
        + b"\0"
        + document_bytes
        + b"\0"
        + canonical_json_bytes(_ride_api_openapi(None))
    )
    return "sha256:" + hashlib.sha256(material).hexdigest()[:16]


def coach_api_contract_metadata(revision=None):
    links = ride_api_links()
    return {
        "schemaVersion": RIDE_API_SCHEMA_VERSION,
        "revision": revision or coach_api_contract_revision(),
        "openapi": links["openapi"],
        "jsonSchema": links["jsonSchema"],
        "documentation": links["documentation"],
        "revisionRole": "contract_and_documentation_only",
        "evidencePollSeconds": POLL_SECONDS,
        "refreshPolicy": {
            "mode": "conditional_get",
            "useIfNoneMatch": True,
            "reloadWhen": ["schemaVersion", "revision"],
            "resources": [
                links["openapi"], links["jsonSchema"], links["documentation"]
            ],
        },
    }


def _ride_api_openapi(revision):
    evidence_schema = ride_evidence_json_schema()
    contract_headers = {
        "ETag": {"$ref": "#/components/headers/ETag"},
        "Cache-Control": {"$ref": "#/components/headers/CacheControl"},
        "X-API-Schema-Version": {
            "$ref": "#/components/headers/ApiSchemaVersion"
        },
        "X-Contract-Revision": {
            "$ref": "#/components/headers/ContractRevision"
        },
    }
    evidence_headers = {
        "Link": {"$ref": "#/components/headers/DiscoveryLink"},
        "Cache-Control": {
            "$ref": "#/components/headers/EvidenceCacheControl"
        },
        "X-API-Schema-Version": {
            "$ref": "#/components/headers/ApiSchemaVersion"
        },
        "X-Contract-Revision": {
            "$ref": "#/components/headers/ContractRevision"
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Bukit Kiara MTB Environmental Evidence API",
            "version": RIDE_API_SCHEMA_VERSION,
            "description": (
                "Read-only local air-quality and weather evidence. The service "
                "does not select a ride window or issue a training prescription."
            ),
            "x-contract-revision": revision,
        },
        "servers": [{
            "url": "/",
            "description": "Resolve against the same dashboard host.",
        }],
        "paths": {
            "/api/v1/mtb/environment-evidence": {
                "get": {
                    "operationId": "getMtbEnvironmentEvidence",
                    "summary": "Get current ride-environment evidence",
                    "description": (
                        "Returns a fresh sensor observation, particle nowcast, "
                        "90-minute-arrival and on-trail outlooks, neutral weather "
                        "context, and a descriptive Morning/Afternoon comparison."
                    ),
                    "responses": {
                        "200": {
                            "description": "Current or degraded usable evidence.",
                            "headers": evidence_headers,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MtbEnvironmentEvidence"
                                    }
                                }
                            },
                        },
                        "503": {
                            "description": "Current evidence is unavailable.",
                            "headers": evidence_headers,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MtbEnvironmentEvidence"
                                    }
                                }
                            },
                        },
                        "500": {
                            "description": "Evidence generation failed.",
                            "headers": evidence_headers,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/MtbEnvironmentEvidence"
                                    }
                                }
                            },
                        },
                    },
                }
            },
            "/api/v1/mtb/environment-evidence/schema": {
                "get": {
                    "operationId": "getMtbEnvironmentEvidenceSchema",
                    "summary": "Get the JSON Schema contract",
                    "parameters": [{
                        "$ref": "#/components/parameters/IfNoneMatch"
                    }],
                    "responses": {
                        "200": {
                            "description": "JSON Schema 2020-12 document.",
                            "headers": contract_headers,
                            "content": {
                                "application/schema+json": {
                                    "schema": {"type": "object"}
                                }
                            },
                        },
                        "304": {
                            "description": "The cached schema is current.",
                            "headers": contract_headers,
                        },
                    },
                }
            },
            "/api/openapi.json": {
                "get": {
                    "operationId": "getOpenApiDocument",
                    "summary": "Get the OpenAPI service description",
                    "parameters": [{
                        "$ref": "#/components/parameters/IfNoneMatch"
                    }],
                    "responses": {
                        "200": {
                            "description": "OpenAPI 3.1 service description.",
                            "headers": contract_headers,
                            "content": {
                                "application/vnd.oai.openapi+json": {
                                    "schema": {"type": "object"}
                                }
                            },
                        },
                        "304": {
                            "description": "The cached description is current.",
                            "headers": contract_headers,
                        },
                    },
                }
            },
            "/docs/coach-api.md": {
                "get": {
                    "operationId": "getCoachApiGuide",
                    "summary": "Get the living environmental integration guide",
                    "parameters": [{
                        "$ref": "#/components/parameters/IfNoneMatch"
                    }],
                    "responses": {
                        "200": {
                            "description": "Markdown integration guide.",
                            "headers": contract_headers,
                            "content": {
                                "text/markdown": {
                                    "schema": {"type": "string"}
                                }
                            },
                        },
                        "304": {
                            "description": "The cached guide is current.",
                            "headers": contract_headers,
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {"MtbEnvironmentEvidence": evidence_schema},
            "parameters": {
                "IfNoneMatch": {
                    "name": "If-None-Match",
                    "in": "header",
                    "required": False,
                    "description": "ETag previously returned for this resource.",
                    "schema": {"type": "string"},
                }
            },
            "headers": {
                "ETag": {
                    "description": "Strong validator for the exact response bytes.",
                    "schema": {"type": "string"},
                },
                "CacheControl": {
                    "description": "Contract resources may be stored but revalidated.",
                    "schema": {"type": "string", "example": "no-cache"},
                },
                "EvidenceCacheControl": {
                    "description": "Dynamic evidence must not be stored.",
                    "schema": {
                        "type": "string",
                        "example": "no-store, no-cache, must-revalidate",
                    },
                },
                "ApiSchemaVersion": {
                    "description": "Evidence schema version.",
                    "schema": {"type": "string"},
                },
                "ContractRevision": {
                    "description": "Opaque contract-resource equality token.",
                    "schema": {"type": "string"},
                },
                "DiscoveryLink": {
                    "description": (
                        "Links to OpenAPI, JSON Schema, and the guide."
                    ),
                    "schema": {"type": "string"},
                },
            },
        },
    }


def ride_api_openapi(revision=None):
    return _ride_api_openapi(revision or coach_api_contract_revision())


def representation_etag(payload):
    return '"sha256-' + hashlib.sha256(payload).hexdigest()[:32] + '"'


def if_none_match_matches(header_value, etag):
    if not header_value:
        return False
    candidates = {part.strip() for part in header_value.split(",")}
    return "*" in candidates or etag in candidates or f"W/{etag}" in candidates


def ride_conditions_api(days=RIDE_API_ANALYSIS_DAYS, now_epoch=None):
    """Return compact, non-prescriptive air-quality and weather evidence."""
    generated_epoch = int(time.time() if now_epoch is None else now_epoch)
    days = max(1.0, min(float(days), 30.0))
    result = analysis(
        days, as_of_epoch=generated_epoch if now_epoch is not None else None
    )
    forecast_epoch = result.get("forecastIssuedEpoch")
    delivery = copy.deepcopy(result.get("delivery") or {}) if now_epoch is None else {}
    forecast_age_seconds = (
        max(0, generated_epoch - int(forecast_epoch))
        if forecast_epoch is not None else
        max(0, int(delivery["forecastAgeSeconds"]))
        if delivery.get("forecastAgeSeconds") is not None else None
    )
    if delivery:
        # Response clocks describe delivery, never a new forecast calculation.
        delivery["forecastAgeSeconds"] = forecast_age_seconds
        delivery["servedAtEpoch"] = generated_epoch
    forecast_expired = now_epoch is None and (
        delivery.get("state") == "expired"
        or (forecast_age_seconds is not None and forecast_age_seconds > 600)
    )
    if delivery and forecast_expired:
        delivery["state"] = "expired"
    analysis_provenance = {
        "analysisDaysRequested": days,
        "analysisComputedAt": (result.get("computation") or {}).get("completedAt"),
        "analysisForecastIssuedAt": iso_from_epoch(forecast_epoch),
    }
    if delivery:
        analysis_provenance["analysisDelivery"] = delivery
    current_epoch = (result.get("current") or {}).get("epoch")
    with db() as conn:
        reading = rowdict(conn.execute(
            "SELECT * FROM readings WHERE epoch=?", (current_epoch,)
        ).fetchone()) if current_epoch is not None else None
    # Runtime collector errors have no historical timestamp.  They are valid
    # live service-health evidence, but including today's process state in an
    # `now_epoch` replay would make the same historical request change later.
    collector = {}
    air_quality_error = None
    if now_epoch is None:
        with status_lock:
            collector = dict(collector_status)
        with weather_lock:
            air_quality_error = weather_status.get("air_quality_error")
    air_quality_payload = (
        latest_air_quality_payload(int(forecast_epoch))
        if forecast_epoch is not None else None
    )

    base = {
        "schemaVersion": RIDE_API_SCHEMA_VERSION,
        "kind": "mtb_environment_evidence",
        "generatedAt": iso_from_epoch(generated_epoch),
        "evidenceId": (
            f'airgradient-{LOCATION_ID}-{int(current_epoch)}-v2'
            if current_epoch is not None else None
        ),
        "contract": coach_api_contract_metadata(),
        "links": ride_api_links(),
        "location": {
            "name": "Bukit Kiara / Taman Tun Dr Ismail",
            "latitude": WEATHER_LATITUDE,
            "longitude": WEATHER_LONGITUDE,
            "timezone": "Asia/Kuala_Lumpur",
        },
        "logistics": {
            "decisionToTrailMinutes": ARRIVAL_MINUTES,
            "typicalTrailDurationMinutes": {"minimum": 90, "maximum": TRAIL_MINUTES},
            "modeledTrailDurationMinutes": TRAIL_MINUTES,
            "modeledExposureWindowMinutes": {
                "start": ARRIVAL_MINUTES,
                "end": ARRIVAL_MINUTES + TRAIL_MINUTES,
            },
        },
        "boundary": {
            "role": "environmental_evidence",
            "trainingPrescriptionIncluded": False,
            "historicalSeriesIncluded": False,
        },
    }

    if (not reading or not result.get("available")
            or reading.get("pm02") is None or forecast_expired):
        unavailable_code = (
            "analysis_expired" if forecast_expired else
            "analysis_initializing" if delivery.get("state") == "initializing" else
            "analysis_unavailable" if not result.get("available") else
            "current_reading_unavailable"
        )
        base.update({
            "status": {
                "state": "unavailable", "usable": False, "fresh": False,
                "message": "Current environmental evidence is unavailable.",
                "issues": [{"code": unavailable_code, "severity": "error"}],
                "forecastAgeSeconds": forecast_age_seconds,
                "analysisUpdating": bool(delivery.get("updating")),
            },
            "observation": None,
            "particleNowcast": {"available": False},
            "exposureOutlook": {
                "arrival": {"available": False}, "onTrail": {"available": False},
            },
            "weather": {"available": False},
            "clearanceEvent": {"detected": False},
            "rideWindows": {},
            "evidenceQuality": {
                "state": "unavailable",
                "limitations": [{"code": "analysis_unavailable"}],
            },
            "provenance": {
                "sensor": {"provider": "AirGradient", "locationId": LOCATION_ID},
                **analysis_provenance,
            },
        })
        return base

    reading_epoch = int(reading["epoch"])
    age_seconds = max(0, generated_epoch - reading_epoch)
    operational_issues = []
    if now_epoch is None and forecast_age_seconds is not None and forecast_age_seconds > 120:
        operational_issues.append({
            "code": "analysis_stale", "severity": "warning",
            "message": f'The last completed forecast was issued {forecast_age_seconds} seconds ago; its targets retain that issue time.',
        })
    if delivery.get("error"):
        operational_issues.append({
            "code": "analysis_update_error", "severity": "warning",
            "message": "The latest analysis update failed; the last completed forecast is being served.",
        })
    if age_seconds > SENSOR_UNAVAILABLE_SECONDS:
        operational_issues.append({
            "code": "sensor_unavailable", "severity": "error",
            "message": f'Latest reading is {age_seconds // 60} minutes old.',
        })
    elif age_seconds > SENSOR_DEGRADED_SECONDS:
        operational_issues.append({
            "code": "sensor_stale", "severity": "warning",
            "message": f'Latest reading is {age_seconds // 60} minutes old.',
        })
    if collector.get("error"):
        operational_issues.append({
            "code": "sensor_collector_error", "severity": "warning",
            "message": str(collector["error"]),
        })

    air_window = result.get("airWindow") or {}
    arrival = air_window.get("arrival") or {}
    trail = air_window.get("trail") or {}
    weather = result.get("weather") or {}
    weather_age_minutes = (
        max(0, round((generated_epoch - int(weather["fetchedEpoch"])) / 60))
        if weather.get("fetchedEpoch") is not None else weather.get("ageMinutes")
    )
    weather_evidence = weather.get("evidence") or {}
    if not weather.get("available"):
        operational_issues.append({
            "code": "weather_unavailable", "severity": "warning",
            "message": weather.get("message") or "Weather evidence is unavailable.",
        })
    elif (weather_age_minutes is not None
          and weather_age_minutes * 60 > WEATHER_DEGRADED_SECONDS):
        operational_issues.append({
            "code": "weather_forecast_stale", "severity": "warning",
            "message": f'Weather forecast is {weather_age_minutes} minutes old.',
        })

    if any(issue["severity"] == "error" for issue in operational_issues):
        service_state = "unavailable"
    elif operational_issues:
        service_state = "degraded"
    else:
        service_state = "ok"
    service_message = {
        "ok": "Environmental evidence is current.",
        "degraded": "Environmental evidence is partially degraded; inspect status issues.",
        "unavailable": "Environmental evidence is too stale or incomplete for a current assessment.",
    }[service_state]

    limitations = []
    if not arrival.get("available"):
        limitations.append({
            "code": "arrival_outlook_collecting",
            "message": "The 90-minute arrival outlook is not yet available.",
        })
    elif str(arrival.get("confidence", "")).lower().startswith("low"):
        limitations.append({
            "code": "arrival_outlook_low_confidence",
            "message": str(arrival.get("confidence")),
        })
    if not trail.get("available"):
        limitations.append({
            "code": "trail_outlook_collecting",
            "message": "The on-trail exposure outlook is not yet available.",
        })
    elif str(trail.get("confidence", "")).lower().startswith("low"):
        limitations.append({
            "code": "trail_outlook_low_confidence",
            "message": str(trail.get("confidence")),
        })
    if not weather_evidence.get("supported"):
        limitations.append({
            "code": "weather_particle_link_unvalidated",
            "message": "Specific weather–PM causal links remain unvalidated. The experimental session ensemble may use issued weather features; no physical cause is asserted.",
        })
    comparison = result.get("comparison") or {}
    if not comparison.get("ready"):
        comparison_reason = (
            comparison.get("summary") or "The two window outlooks are still collecting."
        )
        limitations.append({
            "code": (
                "morning_afternoon_different_dates"
                if "different dates" in str(comparison_reason).lower()
                else "morning_afternoon_forecast_collecting"
            ),
            "message": comparison_reason,
        })
    ride_forecast = result.get("rideForecast") or {}
    issued_particle = ride_forecast.get("issuedEvidence") or {}
    if result.get("windowPrediction"):
        limitations.append({
            "code": "local_session_pm_model_experimental",
            "message": "Session forecasts are experimental. The 07:00–08:00 issue for same-day 14:00–16:00 may use the direction model; other sessions retain the adaptive local model. Inspect each window's modelVersion and directionalModel. Retrospective exploration is not prospective validation.",
        })
        if (result["windowPrediction"].get("regionalCorrectionTrial")):
            limitations.append({
                "code": "cams_local_correction_trial_not_applied",
                "message": "The separate CAMS plus learned local correction trial did not beat the current model overall in development replay. It is recorded for future evaluation, not applied to the primary forecasts or session comparison.",
            })
    elif not issued_particle.get("supported"):
        limitations.append({
            "code": "ride_window_particle_model_experimental",
            "message": (
                f'{issued_particle.get("scoredWindowCount", 0)} issued CAMS ride windows '
                "scored; aggressive CAMS centres are experimental and not validated."
            ),
        })
    elif not ride_forecast.get("usedForComparison"):
        limitations.append({
            "code": "ride_window_particle_model_not_applicable",
            "message": (
                "The PM model is validated but is not applied outside its tested "
                "decision lead and exact modeled session."
            ),
        })
    if air_quality_payload is None or air_quality_error:
        limitations.append({
            "code": "cams_particle_forecast_unavailable",
            "message": str(air_quality_error or "The CAMS candidate is unavailable."),
        })
    elif (generated_epoch - int(air_quality_payload.get("fetchedEpoch") or 0)
          > AIR_QUALITY_DEGRADED_SECONDS):
        limitations.append({
            "code": "cams_particle_forecast_stale",
            "message": "The CAMS candidate is stale; persistence remains available.",
        })

    particle_mix = air_window.get("particleMix") or {}
    mix_signal = None
    if particle_mix.get("relevant"):
        mix_signal = {
            "state": particle_mix.get("state"),
            "label": particle_mix.get("label"),
            "fineSharePct": particle_mix.get("fineShare"),
            "coarseParticlesUgM3": particle_mix.get("coarseParticles"),
        }

    shower = result.get("showerSignal") or {}
    timeline = shower.get("timeline") or {}
    shower_metrics = shower.get("metrics") or {}
    rapid_event = (result.get("shadowForecast") or {}).get(
        "rapidClearanceEvent"
    ) or {}
    event_detected = shower.get("state") in ("active", "recent", "forming")
    clearance_event = {
        "detected": event_detected,
        "state": shower.get("state"),
        "kind": shower.get("kind"),
        "label": shower.get("label"),
        "ageMinutes": shower.get("ageMinutes"),
        "lastEvidenceAt": iso_from_epoch(shower.get("lastEvidenceEpoch")),
        "detectionMode": shower_metrics.get("detectionMode"),
        "rainSupport": bool(shower.get("rainSupport")),
        "dryAirMassSupport": bool(shower.get("dryAirMassSupport")),
        "eventConditionedForecast": {
            "available": bool(rapid_event.get("available")),
            "status": rapid_event.get("status"),
            "completedPriorEvents": rapid_event.get("completedEventCount"),
            "completedPriorDistinctDays": rapid_event.get("completedDistinctDays"),
            "minimumPriorEvents": rapid_event.get("minimumCompletedEvents"),
            "minimumPriorDistinctDays": rapid_event.get("minimumDistinctDays"),
            "eventAt": iso_from_epoch(rapid_event.get("eventEpoch")),
            "eventAgeMinutes": rapid_event.get("eventAgeMinutes"),
            "deltaWeight": rapid_event.get("deltaWeight"),
            "normalization": rapid_event.get("normalization"),
            "arrivalRatio": rapid_event.get("arrivalRatio"),
            "trailMeanRatio": rapid_event.get("trailMeanRatio"),
            "trailPeakRatio": rapid_event.get("trailPeakRatio"),
            "experimental": True,
        },
    }
    if event_detected:
        clearance_event["timeline"] = {
            "onsetAt": iso_from_epoch(timeline.get("onsetEpoch")),
            "confirmedAt": iso_from_epoch(timeline.get("confirmationEpoch")),
            "lowestAt": iso_from_epoch(timeline.get("lowEpoch")),
            "lowestPm25UgM3": timeline.get("lowPm25"),
            "reboundAt": iso_from_epoch(timeline.get("reboundEpoch")),
        }

    window_payloads = {}
    for key in ("morning", "afternoon"):
        window = (result.get("windows") or {}).get(key) or {}
        observed_history = compact_window_baseline(window.get("baseline"))
        window_payloads[key] = {
            "targetDay": window.get("targetLabel"),
            "startAt": iso_from_epoch(window.get("startEpoch")),
            "endAt": iso_from_epoch(window.get("endEpoch")),
            "rideWindow": window.get("rideLabel"),
            "modeledSession": window.get("modeledSessionLabel"),
            "leadHours": window.get("leadHours"),
            "active": bool(window.get("active")),
            "currentConditionsApplicable": bool(window.get("currentApplicable")),
            "logisticsReferenceAt": iso_from_epoch(window.get("decisionEpoch")),
            "logisticsReferenceLabel": window.get("decisionLabel"),
            # Deprecated 1.x aliases; use logisticsReferenceAt/Label.
            "decisionAt": iso_from_epoch(window.get("decisionEpoch")),
            "decisionLabel": window.get("decisionLabel"),
            "forecastIssuedAt": iso_from_epoch(window.get("forecastIssuedEpoch")),
            "forecastedLabel": window.get("forecastedLabel"),
            # Deprecated 1.x display alias retained without delay wording.
            "recheck": window.get("forecastedLabel"),
            "confidence": window.get("confidence"),
            "particleForecast": compact_window_particle_forecast(
                window.get("particleForecast")
            ),
            "observedHistory": observed_history,
            # Deprecated 1.x alias retained for the existing coach parser.
            "historicalBaseline": observed_history,
            "weatherForecast": compact_weather_window(window.get("weatherForecast")),
            "hazeOutlook": {
                "available": False,
                "role": "regional_model_outlook_not_local_concentration",
                "usedForLocalPmPoint": False,
                "validatedAtTTDI": False,
                "sourceAttributionEstablished": False,
                **(window.get("hazeOutlook") or {}),
                "airPath": {key: value for key, value in
                            ((window.get("hazeOutlook") or {}).get("airPath") or {}).items()
                            if key != "paths"},
            },
        }

    subang = weather.get("subang") or {}
    subang_age_minutes = (
        max(0, round((generated_epoch - int(subang["report_epoch"])) / 60))
        if subang.get("report_epoch") is not None else subang.get("ageMinutes")
    )
    trail_weather = compact_weather_window(weather.get("trail"))
    regional_transport = compact_regional_transport(weather.get("transport"))
    base.update({
        "status": {
            "state": service_state,
            "usable": service_state != "unavailable",
            "fresh": (age_seconds <= SENSOR_DEGRADED_SECONDS
                      and (forecast_age_seconds is None or forecast_age_seconds <= 120)
                      and not delivery.get("error")),
            "forecastAgeSeconds": forecast_age_seconds,
            "analysisUpdating": bool(delivery.get("updating")),
            "message": service_message,
            "issues": operational_issues,
        },
        "observation": {
            "id": f'airgradient-{LOCATION_ID}-{reading_epoch}',
            "timestamp": reading.get("timestamp") or iso_from_epoch(reading_epoch),
            "epoch": reading_epoch,
            "ageSeconds": age_seconds,
            "particles": {
                "pm25UgM3": reading.get("pm02"),
                "pm10UgM3": reading.get("pm10"),
            },
            "heat": {
                "heatIndexC": reading.get("heatindex"),
                "temperatureC": reading.get("atmp"),
                "relativeHumidityPct": reading.get("rhum"),
            },
        },
        "dataCoverage": result.get("dataCoverage") or {},
        "particleNowcast": {
            "available": bool(air_window.get("available")),
            "role": "observed_recent_movement_not_future_direction",
            "state": air_window.get("state"),
            "label": (air_window.get("observedMovement") or {}).get("label", air_window.get("label")),
            "change30MinutesUgM3": air_window.get("change30"),
            "change60MinutesUgM3": air_window.get("change60"),
            "fastRise": bool(air_window.get("fastRise")),
            "sustainedImprovement": bool(
                air_window.get("sustainedImprovement")
            ),
            "minimumSinceEventUgM3": air_window.get("minimumSinceEvent"),
            "recheckMinutes": air_window.get("recheckMinutes"),
            "analysisBucketEndAt": iso_from_epoch(
                air_window.get("analysisBucketEndEpoch")
            ),
            "analysisBucketMinutes": air_window.get("analysisBucketMinutes"),
            "particleMixSignal": mix_signal,
        },
        "exposureOutlook": {
            "arrival": {
                "available": bool(arrival.get("available")),
                "freshnessAdjustment": arrival.get("freshnessAdjustment") or {},
                "rainModel": arrival.get("rainModel") or {},
                "modelEvidence": arrival.get("modelEvidence") or {},
                "offsetMinutes": arrival.get("minutes", ARRIVAL_MINUTES),
                "expectedAt": iso_from_epoch(arrival.get("expectedEpoch")),
                "forecastIssuedAt": iso_from_epoch(arrival.get("forecastIssuedEpoch")),
                "targetVersion": (arrival.get("forecastClock") or {}).get("targetVersion"),
                "targetDefinition": (arrival.get("forecastClock") or {}).get("arrivalTargetDefinition"),
                "basedOnObservedAt": reading.get("timestamp") or iso_from_epoch(reading_epoch),
                "headline": arrival.get("headline"),
                "persistenceAnchorPm25UgM3": (
                    arrival.get("baselinePoint")
                    if arrival.get("baselinePoint") is not None else arrival.get("point")
                ),
                "persistenceAnchorRole": arrival.get(
                    "persistenceAnchorRole", "latest_closed_15_minute_bucket_median"
                ),
                "persistenceAnchorAt": iso_from_epoch(
                    arrival.get("persistenceAnchorEpoch")
                ),
                "modelFeatureAnchorPm25UgM3": arrival.get("modelFeatureAnchor"),
                "projectedPm25UgM3": arrival.get("point"),
                # Deprecated 1.x alias now retains its literal baseline meaning.
                "baselinePm25UgM3": (
                    arrival.get("baselinePoint")
                    if arrival.get("baselinePoint") is not None else arrival.get("point")
                ),
                "pointRole": arrival.get("pointRole"),
                "forecastState": arrival.get("forecastState"),
                "approximate": bool(arrival.get("pointApproximate")),
                "validationState": (
                    "validated" if arrival.get("validated") else
                    "experimental_not_validated"
                    if arrival.get("pointApproximate")
                    else "persistence_only"
                ),
                "likelyRangePm25UgM3": {
                    "low": arrival.get("rawRangeLow"),
                    "high": arrival.get("rawRangeHigh"),
                    "quantiles": [0.10, 0.90], "calibrated": False,
                },
                "forecastRangePm25UgM3": {
                    "low": arrival.get("rangeLow"),
                    "high": arrival.get("rangeHigh"),
                    "containsPersistenceAnchor": True,
                    "containsExperimentalProjection": bool(arrival.get("pointApproximate")),
                    "role": arrival.get("rangeRole") or "anchor_containing_display_envelope",
                    "coverageClaimed": False,
                },
                # Deprecated 1.x alias; use forecastRangePm25UgM3.
                "decisionEnvelopePm25UgM3": {
                    "low": arrival.get("rangeLow"),
                    "high": arrival.get("rangeHigh"),
                    "containsBaseline": True,
                    "containsExperimentalProjection": bool(arrival.get("pointApproximate")),
                    "role": arrival.get("rangeRole") or "persistence_error_envelope",
                    "coverageClaimed": False,
                },
                "upperPm25UgM3": {
                    "value": (
                        arrival.get("rawUpper90")
                        if arrival.get("rawUpper90") is not None
                        else arrival.get("rawRangeHigh")
                    ),
                    "quantile": 0.90,
                    "finiteSampleRank": bool(arrival.get("finiteSampleUpper")),
                    "finiteSampleRankCoverage": arrival.get("finiteSampleRankCoverage"),
                    "calibrated": False,
                    "role": arrival.get("upperQuantileRole") or "uncalibrated_empirical_persistence_error_quantile",
                },
                "forecastUpperPm25UgM3": (
                    arrival.get("decisionUpper")
                    if arrival.get("decisionUpper") is not None
                    else arrival.get("upper90")
                ),
                # Deprecated 1.x alias; use forecastUpperPm25UgM3.
                "decisionUpperPm25UgM3": (
                    arrival.get("decisionUpper")
                    if arrival.get("decisionUpper") is not None
                    else arrival.get("upper90")
                ),
                "confidence": confidence_code(arrival.get("confidence")),
                "confidenceDetail": arrival.get("confidence"),
                "methodId": exposure_method_id(arrival.get("method")),
                "rapidClearanceEventSupport": arrival.get("rapidEventSupport"),
                "support": {
                    "originCount": arrival.get("originCount"),
                    "independentOriginCount": arrival.get("independentOriginCount"),
                    "matchedCount": arrival.get("matchedCount"),
                    "matchedIndependentOriginCount": arrival.get(
                        "matchedIndependentOriginCount"
                    ),
                    "matchedDistinctDays": arrival.get("matchedDistinctDays"),
                    "distinctDays": arrival.get("distinctDays"),
                    "minimumRequired": arrival.get("minimumRequired"),
                },
            },
            "onTrail": {
                "available": bool(trail.get("available")),
                "freshnessAdjustment": trail.get("freshnessAdjustment") or {},
                "rainModel": trail.get("rainModel") or {},
                "modelEvidence": trail.get("modelEvidence") or {},
                "startOffsetMinutes": trail.get("startMinutes", ARRIVAL_MINUTES),
                "endOffsetMinutes": trail.get("endMinutes", ARRIVAL_MINUTES + TRAIL_MINUTES),
                "startAt": iso_from_epoch(trail.get("startEpoch")),
                "endAt": iso_from_epoch(trail.get("endEpoch")),
                "forecastIssuedAt": iso_from_epoch(trail.get("forecastIssuedEpoch")),
                "targetVersion": (trail.get("forecastClock") or {}).get("targetVersion"),
                "meanTargetDefinition": (trail.get("forecastClock") or {}).get("windowMeanTargetDefinition"),
                "peakTargetDefinition": (trail.get("forecastClock") or {}).get("windowPeakTargetDefinition"),
                "basedOnObservedAt": reading.get("timestamp") or iso_from_epoch(reading_epoch),
                "headline": trail.get("headline"),
                "persistenceAnchorPm25UgM3": (
                    trail.get("baselinePoint")
                    if trail.get("baselinePoint") is not None else trail.get("point")
                ),
                "persistenceAnchorRole": trail.get(
                    "persistenceAnchorRole", "latest_closed_15_minute_bucket_median"
                ),
                "persistenceAnchorAt": iso_from_epoch(
                    trail.get("persistenceAnchorEpoch")
                ),
                "modelFeatureAnchorPm25UgM3": trail.get("modelFeatureAnchor"),
                "projectedMeanPm25UgM3": trail.get("point"),
                "projectedPeakPm25UgM3": trail.get("projectedPeak"),
                # Deprecated 1.x alias now retains its literal baseline meaning.
                "baselineMeanPm25UgM3": (
                    trail.get("baselinePoint")
                    if trail.get("baselinePoint") is not None else trail.get("point")
                ),
                "pointRole": trail.get("pointRole"),
                "forecastState": trail.get("forecastState"),
                "approximate": bool(trail.get("pointApproximate")),
                "validationState": (
                    "validated" if trail.get("validated") else
                    "mixed_persistence_mean_experimental_peak"
                    if (trail.get("peakApproximate") and not trail.get("pointApproximate")) else
                    "experimental_not_validated"
                    if trail.get("pointApproximate")
                    else "persistence_only"
                ),
                "meanValidationState": (
                    "validated" if trail.get("validated") else
                    "experimental_not_validated" if trail.get("pointApproximate")
                    else "persistence_only"
                ),
                "peakValidationState": (
                    "validated" if trail.get("validated") else
                    "experimental_not_validated" if trail.get("peakApproximate")
                    else "persistence_only"
                ),
                "likelyMeanRangePm25UgM3": {
                    "low": trail.get("rawRangeLow"),
                    "high": trail.get("rawRangeHigh"),
                    "quantiles": [0.10, 0.90], "calibrated": False,
                },
                "forecastMeanRangePm25UgM3": {
                    "low": trail.get("rangeLow"),
                    "high": trail.get("rangeHigh"),
                    "containsPersistenceAnchor": True,
                    "containsExperimentalProjection": bool(trail.get("pointApproximate")),
                    "role": trail.get("rangeRole") or "anchor_containing_display_envelope",
                    "coverageClaimed": False,
                },
                # Deprecated 1.x alias; use forecastMeanRangePm25UgM3.
                "decisionMeanEnvelopePm25UgM3": {
                    "low": trail.get("rangeLow"),
                    "high": trail.get("rangeHigh"),
                    "containsBaseline": True,
                    "containsExperimentalProjection": bool(trail.get("pointApproximate")),
                    "role": trail.get("rangeRole") or "persistence_error_envelope",
                    "coverageClaimed": False,
                },
                "upperMeanPm25UgM3": {
                    "value": (
                        trail.get("rawUpperMean90")
                        if trail.get("rawUpperMean90") is not None
                        else trail.get("rawRangeHigh")
                    ),
                    "quantile": 0.90,
                    "finiteSampleRank": bool(trail.get("finiteSampleUpper")),
                    "finiteSampleRankCoverage": trail.get("finiteSampleRankCoverage"),
                    "calibrated": False,
                    "role": trail.get("upperQuantileRole") or "uncalibrated_empirical_persistence_error_quantile",
                },
                "upperPeakPm25UgM3": {
                    "value": trail.get("rawPeakUpper90"),
                    "quantile": 0.90,
                    "finiteSampleRank": bool(trail.get("finiteSampleUpper")),
                    "finiteSampleRankCoverage": trail.get("finiteSampleRankCoverage"),
                    "calibrated": False,
                    "role": "uncalibrated_empirical_persistence_error_quantile",
                },
                "forecastPeakUpperPm25UgM3": (
                    trail.get("decisionPeakUpper")
                    if trail.get("decisionPeakUpper") is not None
                    else trail.get("peakUpper")
                ),
                # Deprecated 1.x alias; use forecastPeakUpperPm25UgM3.
                "decisionPeakRiskMarkerPm25UgM3": (
                    trail.get("decisionPeakUpper")
                    if trail.get("decisionPeakUpper") is not None
                    else trail.get("peakUpper")
                ),
                "confidence": confidence_code(trail.get("confidence")),
                "confidenceDetail": trail.get("confidence"),
                "methodId": exposure_method_id(trail.get("method")),
                "rapidClearanceEventSupport": trail.get("rapidEventSupport"),
                "support": {
                    "originCount": trail.get("originCount"),
                    "independentOriginCount": trail.get("independentOriginCount"),
                    "matchedCount": trail.get("matchedCount"),
                    "matchedIndependentOriginCount": trail.get(
                        "matchedIndependentOriginCount"
                    ),
                    "matchedDistinctDays": trail.get("matchedDistinctDays"),
                    "matchedEventCount": trail.get("matchedEventCount"),
                    "distinctDays": trail.get("distinctDays"),
                    "minimumRequired": trail.get("minimumRequired"),
                },
            },
        },
        "weather": {
            "available": bool(weather.get("available")),
            "forecastSource": weather.get("source"),
            "forecastFetchedAt": iso_from_epoch(weather.get("fetchedEpoch")),
            "forecastAgeMinutes": weather_age_minutes,
            "trailPeriod": trail_weather,
            "role": "forecast_inputs_and_secondary_weather_context",
            "usedForParticleForecast": bool((result.get("forecastPolicy") or {}).get("weatherUsedForParticleForecast")),
            "particleRelationshipValidated": False,
            "rainLearning": {
                "status": "evaluation_only",
                "appliedToPrimaryForecast": False,
                "modelVersion": (result.get("rainLearning") or {}).get("modelVersion"),
                "weatherFetchedAt": iso_from_epoch((result.get("rainLearning") or {}).get("weatherFetchedEpoch")),
                "reason": "Fixed paired tests did not demonstrate better PM forecast accuracy.",
                "context": (result.get("rainLearning") or {}).get("rainContext") or {},
            },
            "regionalTransport": regional_transport,
            "subangReference": {
                "role": "corroborating_context",
                "usedForParticleForecast": False,
                "station": "SUBANG",
                "wigosId": SUBANG_WIGOS_ID,
                "timestamp": subang.get("timestamp"),
                "ageMinutes": subang_age_minutes,
                "temperatureC": subang.get("atmp"),
                "relativeHumidityPct": subang.get("rhum"),
                "windSpeedKmh": subang.get("wind_speed_kmh"),
                "windDirection": subang.get("windCompass"),
                "visibilityKm": subang.get("visibility_km"),
            },
        },
        "clearanceEvent": clearance_event,
        "rideWindows": {
            "comparison": {
                "status": "ready" if comparison.get("ready") else "collecting",
                "role": comparison.get("role"),
                "prescriptiveRecommendation": False,
                "policyId": comparison.get("policyId"),
                "aggressiveExperimental": bool(
                    (comparison.get("model") or {}).get("aggressiveMode")
                ),
                "headline": comparison.get("headline"),
                "summary": comparison.get("summary"),
                "pm": comparison.get("pm"),
                "weather": comparison.get("weather"),
                "confidence": confidence_code(comparison.get("confidence")),
                "particleModel": comparison.get("model"),
            },
            **window_payloads,
        },
        "evidenceQuality": {
            "state": "limited" if limitations else "adequate",
            "limitations": limitations,
            "localMeanIssuedValidation": (result.get("shadowForecast") or {}).get("prospectiveEvidence"),
        },
        "provenance": {
            "forecastPolicy": result.get("forecastPolicy") or {},
            "nearTermSelectionDiagnostics": air_window.get("nearTermSelectionDiagnostics") or {},
            "sensor": {
                "provider": "AirGradient",
                "locationId": LOCATION_ID,
                "measurement": "raw particle concentration",
                "observedAt": reading.get("timestamp") or iso_from_epoch(reading_epoch),
                "ageSeconds": age_seconds,
                "pollSeconds": POLL_SECONDS,
                "staleAfterSeconds": SENSOR_DEGRADED_SECONDS,
                "expiredAfterSeconds": SENSOR_UNAVAILABLE_SECONDS,
            },
            "weatherForecast": {
                "provider": "Open-Meteo Best Match",
                "fetchedAt": iso_from_epoch(weather.get("fetchedEpoch")),
                "ageMinutes": weather_age_minutes,
                "validation": {
                    "supported": bool(weather_evidence.get("supported")),
                    "originCount": weather_evidence.get("originCount"),
                    "minimumOrigins": weather_evidence.get("minimumOrigins"),
                },
            },
            "particleForecast": {
                "provider": "Local PM ensemble / AirGradient + Open-Meteo / CAMS",
                "modelVersion": (result.get("windowPrediction") or {}).get("modelVersion"),
                "modelsByWindow": (result.get("windowPrediction") or {}).get("modelsByWindow") or {},
                "modelPointApplied": any((v.get("particleForecast") or {}).get("available") for v in (result.get("windows") or {}).values()),
                "directComparisonEligible": False,
                "fetchedAt": iso_from_epoch(
                    (air_quality_payload or {}).get("fetchedEpoch")
                ),
                "error": air_quality_error,
                "attributionUrl": "https://open-meteo.com/en/docs/air-quality-api",
                "legacyCamsValidation": {
                    "mode": issued_particle.get("validationMode"),
                    "supported": bool(issued_particle.get("supported")),
                    "scoredWindowCount": issued_particle.get(
                        "scoredWindowCount"
                    ),
                    "minimumScoredWindows": issued_particle.get(
                        "minimumScoredWindows"
                    ),
                    "distinctDays": issued_particle.get("distinctDays"),
                    "minimumDistinctDays": issued_particle.get(
                        "minimumDistinctDays"
                    ),
                },
                "validation": {
                    "mode": "exact_lead_prequential_replay",
                    "supported": False, "prospectivelyValidated": False,
                    "byWindow": {key: (value.get("particleForecast") or {}).get("modelEvidence") or {}
                                 for key, value in (result.get("windows") or {}).items()},
                },
                "regionalCorrectionTrial": (result.get("windowPrediction") or {}).get("regionalCorrectionTrial"),
            },
            "weatherReference": {
                "provider": "MET Malaysia", "station": "SUBANG",
                "wigosId": SUBANG_WIGOS_ID,
            },
            **analysis_provenance,
            "historyHoursAvailable": result.get("historyHours"),
            "sampleCount": result.get("sampleCount"),
        },
        "links": ride_api_links(),
    })
    return json_safe(base)


def csv_export(days):
    cutoff = int(time.time() - max(1.0, min(float(days), RETENTION_DAYS)) * 86400)
    with db() as conn:
        rows = conn.execute("SELECT * FROM readings WHERE epoch>=? ORDER BY epoch", (cutoff,)).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    cols = ["timestamp"] + FIELDS
    writer.writerow(cols)
    for r in rows:
        writer.writerow([r[c] for c in cols])
    return output.getvalue().encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)

    def send_data(self, status, content_type, payload, filename=None,
                  cache_control=None, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Cache-Control",
            cache_control or "no-store, no-cache, must-revalidate",
        )
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        for name, value in (headers or {}).items():
            self.send_header(name, str(value))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def send_json(self, obj, status=200, headers=None):
        self.send_data(status, "application/json; charset=utf-8",
                       json.dumps(
                           json_safe(obj), ensure_ascii=False, allow_nan=False
                       ).encode("utf-8"), headers=headers)

    def send_versioned_data(self, content_type, payload, contract_revision=None):
        """Serve a reusable contract resource with validator-aware caching."""
        etag = representation_etag(payload)
        contract_revision = contract_revision or coach_api_contract_revision()
        headers = {
            "ETag": etag,
            "X-API-Schema-Version": RIDE_API_SCHEMA_VERSION,
            "X-Contract-Revision": contract_revision,
        }
        if_none_match = ",".join(
            self.headers.get_all("If-None-Match", failobj=[]) or []
        )
        if if_none_match_matches(if_none_match, etag):
            self.send_response(304)
            self.send_header("Cache-Control", "no-cache")
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            return
        self.send_data(
            200, content_type, payload, cache_control="no-cache", headers=headers
        )

    def send_versioned_json(self, obj, content_type, contract_revision=None):
        payload = json.dumps(
            json_safe(obj), ensure_ascii=False, allow_nan=False,
            sort_keys=True, indent=2,
        ).encode("utf-8")
        self.send_versioned_data(
            content_type, payload, contract_revision=contract_revision
        )

    @staticmethod
    def coach_discovery_headers(contract):
        return {
            "Link": (
                '</api/openapi.json>; rel="service-desc"; '
                'type="application/vnd.oai.openapi+json", '
                '</api/v1/mtb/environment-evidence/schema>; '
                'rel="describedby"; type="application/schema+json", '
                '</docs/coach-api.md>; rel="help"; type="text/markdown"'
            ),
            "X-API-Schema-Version": RIDE_API_SCHEMA_VERSION,
            "X-Contract-Revision": contract["revision"],
        }

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            self.send_data(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
            return

        if route == "/favicon.ico":
            self.send_data(204, "image/x-icon", b"")
            return

        if route == "/api/openapi.json":
            revision = coach_api_contract_revision()
            self.send_versioned_json(
                ride_api_openapi(revision),
                "application/vnd.oai.openapi+json;version=3.1; charset=utf-8",
                contract_revision=revision,
            )
            return

        if route == "/api/v1/mtb/environment-evidence/schema":
            revision = coach_api_contract_revision()
            self.send_versioned_json(
                ride_evidence_json_schema(),
                "application/schema+json; charset=utf-8",
                contract_revision=revision,
            )
            return

        if route == "/docs/coach-api.md":
            document = coach_api_document_bytes()
            revision = coach_api_contract_revision(document)
            self.send_versioned_data(
                "text/markdown; charset=utf-8", document,
                contract_revision=revision,
            )
            return

        if route == "/api/v1/mtb/environment-evidence":
            try:
                payload = ride_conditions_api()
                contract = payload["contract"]
                discovery_headers = self.coach_discovery_headers(contract)
                status = 503 if payload["status"]["state"] == "unavailable" else 200
                self.send_json(payload, status, headers=discovery_headers)
            except Exception as error:
                print("[HTTP] Environment evidence error:", error)
                contract = coach_api_contract_metadata()
                discovery_headers = self.coach_discovery_headers(contract)
                self.send_json({
                    "schemaVersion": RIDE_API_SCHEMA_VERSION,
                    "kind": "mtb_environment_evidence",
                    "generatedAt": iso_now(),
                    "evidenceId": None,
                    "contract": contract,
                    "boundary": {
                        "role": "environmental_evidence",
                        "trainingPrescriptionIncluded": False,
                        "historicalSeriesIncluded": False,
                    },
                    "status": {
                        "state": "unavailable", "usable": False, "fresh": False,
                        "message": "Environmental evidence could not be generated.",
                        "issues": [{"code": "evidence_generation_failed", "severity": "error"}],
                    },
                    "links": ride_api_links(),
                }, 500, headers=discovery_headers)
            return

        if route == "/api/current":
            r = latest()
            if r:
                r["ageSeconds"] = max(0, int(time.time()) - int(r["epoch"]))
            with db() as conn:
                first_epoch = conn.execute("SELECT MIN(epoch) FROM readings").fetchone()[0]
            stored_hours = ((r["epoch"]-first_epoch)/3600
                            if r and first_epoch is not None else 0)
            with status_lock:
                s = dict(collector_status)
            self.send_json({
                "reading": r,
                "collector": s,
                "pollSeconds": POLL_SECONDS,
                "dashboardBuild": DASHBOARD_BUILD,
                "historyHours": round(stored_hours, 2),
            })
            return

        if route == "/api/history":
            try:
                hours = float(qs.get("hours", ["24"])[0])
            except ValueError:
                hours = 24
            self.send_json(history(hours))
            return

        if route == "/api/analysis":
            try:
                days = float(qs.get("days", [str(RIDE_API_ANALYSIS_DAYS)])[0])
            except ValueError:
                days = RIDE_API_ANALYSIS_DAYS
            self.send_json(analysis(days))
            return

        if route == "/api/weather":
            cutoff = int(time.time()) - 30 * 86400
            with db() as conn:
                rows = conn.execute(
                    "SELECT epoch,pm02,pm10,atmp,rhum,heatindex FROM readings "
                    "WHERE epoch>=? ORDER BY epoch", (cutoff,)
                ).fetchall()
            latest_epoch = int(rows[-1]["epoch"]) if rows else int(time.time())
            with weather_lock:
                status = {key: value for key, value in weather_status.items()
                          if key not in ("forecast", "air_quality", "subang")}
            self.send_json({
                "weather": weather_outlook(rows, latest_epoch),
                "collector": status,
                "pollSeconds": WEATHER_POLL_SECONDS,
            })
            return

        if route == "/api/refresh":
            data = fetch_reading()
            if data is None:
                with status_lock:
                    err = collector_status["error"]
                self.send_json({"ok": False, "error": err}, 502)
            else:
                self.send_json({"ok": True})
            return

        if route == "/api/export.csv":
            try:
                days = float(qs.get("days", ["30"])[0])
            except ValueError:
                days = 30
            payload = csv_export(days)
            self.send_data(
                200, "text/csv; charset=utf-8", payload,
                filename=f"Bukit_Kiara_AirGradient_{int(days)}days.csv"
            )
            return

        self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    init_db()

    # Bind first so a duplicate instance exits before starting any collectors.
    server = ThreadingHTTPServer((HOST, PORT), Handler)

    t = threading.Thread(target=collector, daemon=True)
    t.start()
    weather_thread = threading.Thread(target=weather_collector, daemon=True)
    weather_thread.start()
    threading.Thread(target=haze_transport.collector, args=(DB_PATH,), daemon=True).start()
    threading.Thread(target=analysis_worker, daemon=True).start()

    url = f"http://localhost:{PORT}/"

    print("=" * 66)
    print("BUKIT KIARA RIDE CONDITIONS")
    print("=" * 66)
    print(f"Dashboard : {url}")
    print(f"API test  : {url}api/current")
    print(f"History DB: {DB_PATH}")
    print("Sampling  : every 3 minutes")
    print("Weather   : Open-Meteo every 15 min; CAMS particles hourly; Subang reference")
    print("Keep this window open. Press Ctrl+C to stop.")
    print("=" * 66)

    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.server_close()
