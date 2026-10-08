"""Small, read-only projection of published forecasts for a LAN RLCD device.

No network access, fitting, telemetry writes, or new forecasting decisions.
All epochs are UTC Unix seconds; display them in Asia/Kuala_Lumpur.
"""
import math
import json
from datetime import datetime, timedelta, timezone
import rlcd_history
import weather_contracts
import momentum_display

SCHEMA_VERSION = 1
API_PATH = "/api/rlcd/v1"
MAX_PAYLOAD_BYTES = 8192
DEVICE_HISTORY_MAX_POINTS = 64


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def epoch(value):
    value = number(value)
    return int(value) if value is not None and value > 0 else None


def age(now, stamp):
    return now - stamp if stamp is not None and stamp <= now else None


def probability(value):
    value = number(value)
    return value if value is not None and 0 <= value <= 1 else None


def enforce_payload_budget(payload):
    """Thin only device graph points; keep summaries, gaps and forecasts intact.

    Repeated selection is over already issued graph points, so the method
    explicitly identifies this secondary display-only reduction. Actual PM
    extrema/endpoints remain observations. HTTP callers can fail closed if
    fixed prediction fields alone exceed the device's byte limit.
    """
    history = payload.get("history") or {}
    original_points = list(history.get("points") or ())
    for maximum in (DEVICE_HISTORY_MAX_POINTS, 56, 48, 40, 32, 24, 16, 8, 4):
        if len(original_points) > maximum:
            points, method = rlcd_history._select(original_points, history["start_epoch"], maximum)
            history["points"] = points
            history["method"] = "device_secondary_selection_" + method
        size = len(json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        if size <= MAX_PAYLOAD_BYTES:
            return payload
    raise ValueError("RLCD prediction fields exceed the 8192-byte device payload budget")


def weather_point(point):
    stamp = epoch(point.get("epoch"))
    return {
        "valid_epoch": stamp,
        "temperature_c": number(point.get("temperature_2m")),
        "humidity_pct": number(point.get("relative_humidity_2m")),
        "feels_like_c": number(point.get("apparent_temperature")),
        "wind_kmh": number(point.get("wind_speed_10m")),
        "wind_from_deg": number(point.get("wind_direction_10m")),
        "gust_kmh": number(point.get("wind_gusts_10m")),
        "cloud_pct": number(point.get("cloud_cover")),
        "rain_start_epoch": stamp - 3600 if stamp else None,
        "rain_end_epoch": stamp,
        "rain_chance_pct": number(point.get("precipitation_probability")),
        "rain_mm": number(point.get("precipitation")),
    }


def window_weather(source, now, issued, start, end):
    source = source or {}
    fetched = epoch(source.get("sourceFetchedEpoch"))
    source_age = age(now, fetched)
    coverage_ok = weather_contracts.coverage_verified(source, start, end)
    summary_ok = all(number(source.get(field)) is not None for field in (
        "temperatureMax", "apparentTemperatureMax", "relativeHumidityMean",
        "windSpeed10mMean", "windGust10mMax", "precipitationProbabilityMax", "precipitationMm"))
    available = bool(source.get("available") and fetched is not None and issued is not None
                     and fetched <= issued and source_age is not None and source_age <= 7200
                     and start is not None and end is not None
                     and epoch(source.get("startEpoch")) == start
                     and epoch(source.get("endEpoch")) == end and coverage_ok and summary_ok)
    return {
        "available": available,
        "fresh": bool(available and source_age <= 1800),
        "fetched_epoch": fetched,
        "age_s": source_age,
        "coverage": weather_contracts.compact_coverage(source),
        "coverage_verified": coverage_ok,
        "unavailable_reason": (None if available else "weather_coverage_unverified_legacy_summary"
                               if "weatherCoverage" not in source else source.get("coverageReason")
                               or "weather_clock_or_coverage_unavailable"),
        "temperature_max_c": number(source.get("temperatureMax")) if available else None,
        "feels_like_max_c": number(source.get("apparentTemperatureMax")) if available else None,
        "humidity_mean_pct": number(source.get("relativeHumidityMean")) if available else None,
        "wind_mean_kmh": number(source.get("windSpeed10mMean")) if available else None,
        "wind_from_deg": number(source.get("windDirection10m")) if available else None,
        "gust_max_kmh": number(source.get("windGust10mMax")) if available else None,
        "rain_chance_max_pct": number(source.get("precipitationProbabilityMax")) if available else None,
        "rain_mm": number(source.get("precipitationMm")) if available else None,
    }


def projection(source, allowed):
    """Copy the published numeric head; validation never substitutes a point."""
    source = source or {}
    value = number(source.get("point"))
    available = bool(allowed and source.get("available") and value is not None)
    raw_role = source.get("pointRole") in ("raw_model_output", "experimental_model_output")
    experimental = bool(raw_role or source.get("pointApproximate")
                        or source.get("pointRole") == "experimental_window_mean")
    validated = source.get("validated") is True
    performance = source.get("performanceStatus")
    if not isinstance(performance, str):
        performance = "validated" if validated else "experimental_unvalidated" if experimental else "unvalidated"
    low = number(source.get("rawRangeLow")) if available else None
    high = number(source.get("rawRangeHigh")) if available else None
    return {
        "available": available,
        "pm25_ugm3": value if available else None,
        "reference_pm25_ugm3": number(source.get("baselinePoint")) if allowed else None,
        "role": source.get("pointRole"),
        "model": source.get("modelVersion") or source.get("forecastMethod"),
        "experimental": experimental,
        "validated": validated,
        "qualified": source.get("qualified") is True,
        "used_for_decision": source.get("usedForDecision") is True,
        "performance_status": performance,
        "range_low_ugm3": low,
        "range_high_ugm3": high,
        "range_kind": "empirical_q10_q90" if low is not None and high is not None else "unavailable",
        "calibrated": False,
    }


def tennis_morning(weather, now):
    """Fixed 07:00–09:00 weather only; keep today's window until its end."""
    local = datetime.fromtimestamp(now, timezone(timedelta(hours=8)))
    start = local.replace(hour=7, minute=0, second=0, microsecond=0)
    if local >= start + timedelta(hours=2):
        start += timedelta(days=1)
    start, end = int(start.timestamp()), int((start + timedelta(hours=2)).timestamp())
    fetched = epoch(weather.get("fetchedEpoch"))
    weather_age = age(now, fetched)
    source_ok = weather_age is not None and weather_age <= 7200
    points = {epoch(p.get("epoch")): p for p in weather.get("hourly", []) if epoch(p.get("epoch"))}
    def complete_values(field, stamps, lower=None, upper=None):
        values = [number((points.get(stamp) or {}).get(field)) for stamp in stamps]
        if not source_ok or any(v is None or (lower is not None and v < lower)
                                or (upper is not None and v > upper) for v in values):
            return None
        return values
    rain = complete_values("precipitation_probability", (start + 3600, end), 0, 100)
    feels = complete_values("apparent_temperature", (start, start + 3600))
    wind = complete_values("wind_speed_10m", (start, start + 3600), 0)
    available = rain is not None and feels is not None and wind is not None
    return {
        "available": available,
        "fresh": bool(available and weather_age <= 1800),
        "start_epoch": start, "end_epoch": end,
        "active": start <= now < end,
        "summary_epoch": now,
        "fetched_epoch": fetched, "age_s": weather_age,
        "source": weather.get("source"),
        "kind": "hourly_model_weather_only",
        "rain_chance_max_pct": max(rain) if rain is not None else None,
        "feels_like_max_c": round(max(feels), 1) if feels is not None else None,
        "wind_mean_kmh": round(sum(wind) / len(wind), 1) if wind is not None else None,
    }


def build_payload(reading, analysis, weather, now, *, dashboard_build=None, history_rows=()):
    """Project already collected inputs, rejecting future/expired clocks."""
    now = int(now)
    reading, analysis, weather = reading or {}, analysis or {}, weather or {}
    observed = epoch(reading.get("epoch"))
    sensor_age = age(now, observed)
    sensor_ok = sensor_age is not None and sensor_age <= 900
    current = {
        "available": bool(sensor_ok and number(reading.get("pm02")) is not None),
        "fresh": bool(sensor_ok and sensor_age <= 420),
        "observed_epoch": observed,
        "age_s": sensor_age,
        "source": "AirGradient TTDI 86311",
        "pm25_ugm3": number(reading.get("pm02")) if sensor_ok else None,
        "temperature_c": number(reading.get("atmp")) if sensor_ok else None,
        "humidity_pct": number(reading.get("rhum")) if sensor_ok else None,
        "heat_index_c": number(reading.get("heatindex")) if sensor_ok else None,
    }
    fetched = epoch(weather.get("fetchedEpoch"))
    weather_age = age(now, fetched)
    weather_ok = weather_age is not None and weather_age <= 7200
    points = sorted((p for p in weather.get("hourly", []) if epoch(p.get("epoch"))),
                    key=lambda p: p["epoch"]) if weather_ok else []
    current_points = [p for p in points if 0 <= now - p["epoch"] < 3600]
    future_points = [p for p in points if now < p["epoch"] <= now + 6 * 3600][:6]
    weather_out = {
        "available": bool(weather_ok and (current_points or future_points)),
        "fresh": bool(weather_ok and weather_age <= 1800),
        "fetched_epoch": fetched,
        "age_s": weather_age,
        "source": weather.get("source"),
        "kind": "hourly_model_not_local_observation",
        "rain_kind": "preceding_hour_amount_and_probability_not_exact_onset",
        "gust_kind": "preceding_hour_maximum",
        "current_hour": weather_point(current_points[-1]) if current_points else None,
        "next_hours": [weather_point(p) for p in future_points],
        "uv_index": None,
        "uv_available": False,
    }
    issued = epoch(analysis.get("forecastIssuedEpoch"))
    forecast_age = age(now, issued)
    delivery = analysis.get("delivery") or {}
    forecast_ok = bool(sensor_ok and analysis.get("available") and forecast_age is not None
                       and forecast_age <= 600 and delivery.get("state") != "expired")
    air = analysis.get("airWindow") or {}
    clock = air.get("forecastClock") or {}
    clock_ok = bool(forecast_ok and epoch(clock.get("forecastIssuedEpoch")) == issued
                    and epoch(clock.get("arrivalTargetEpoch")) == issued + 5400
                    and epoch(clock.get("windowStartEpoch")) == issued + 5400
                    and epoch(clock.get("windowEndEpoch")) == issued + 12600)
    arrival_source = air.get("arrival") or {}
    near_ok = bool(clock_ok and epoch(arrival_source.get("forecastIssuedEpoch")) == issued
                   and epoch(arrival_source.get("expectedEpoch")) == issued + 5400)
    near = projection(arrival_source, near_ok)
    near["target_epoch"] = epoch(clock.get("arrivalTargetEpoch")) if clock_ok else None
    event = air.get("firstCrossingEventForecast") or {}
    event_probs = [probability(event.get(key)) for key in
                   ("probabilityRise", "probabilityDrop", "probabilityNoCrossing")]
    event_ok = bool(clock_ok and event.get("available")
                    and epoch(event.get("forecastIssuedEpoch")) == issued
                    and epoch(event.get("validUntilEpoch")) == near["target_epoch"]
                    and number(event.get("changeThresholdUgM3")) == 20
                    and all(p is not None for p in event_probs)
                    and abs(sum(event_probs) - 1) <= 1e-6)
    qualification = event.get("qualification")
    qualification = qualification if isinstance(qualification, dict) else {}
    event_qualified = bool(event_ok and qualification.get("operationalUseEligible") is True)
    diagnostic_direction = event.get("diagnosticDirection", event.get("direction", "unresolved"))
    if diagnostic_direction not in ("rise", "drop", "unresolved"):
        diagnostic_direction = "unresolved"
    operational_direction = event.get("direction", "unresolved") if event_qualified else "unresolved"
    if operational_direction not in ("rise", "drop", "unresolved"):
        operational_direction = "unresolved"
    near["first20"] = {
        "available": event_ok,
        "display_text": momentum_display.display_text(event) if event_ok else None,
        "rise_probability": event_probs[0] if event_ok else None,
        "drop_probability": event_probs[1] if event_ok else None,
        "none_probability": event_probs[2] if event_ok else None,
        "direction": operational_direction if event_ok else None,
        "diagnostic_direction": diagnostic_direction if event_ok else None,
        "qualification": {
            "state": "qualified" if event_qualified else "unqualified_or_legacy",
            "operational_use_eligible": event_qualified,
            "actual_cadence_validated": qualification.get("actualCadenceValidated") is True,
            "calibration_validated": qualification.get("calibrationValidated") is True,
        },
        "reference_ugm3": number(event.get("referencePm")) if event_ok else None,
        "model": event.get("modelVersion"),
        "experimental": True,
        "calibrated": False,
    }
    arrival_change = air.get("arrivalChangeForecast") or {}
    arrival_tails = [probability(arrival_change.get(key)) for key in
                     ("probabilityFall20", "probabilityFall40", "probabilityRise20", "probabilityRise40")]
    arrival_center = probability(arrival_change.get("probabilityWithin20"))
    change_ok = bool(clock_ok and arrival_change.get("available")
                     and epoch(arrival_change.get("forecastIssuedEpoch")) == issued
                     and epoch(arrival_change.get("arrivalEpoch")) == issued + 5400
                     and arrival_change.get("arrivalLeadMinutes") == 90
                     and arrival_change.get("target") == "exact_issue_plus90_arrival_median_proxy_delta_from_fresh5_reference"
                     and number(arrival_change.get("referencePm")) is not None
                     and all(value is not None for value in arrival_tails)
                     and arrival_center is not None
                     and arrival_tails[1] <= arrival_tails[0] + 1e-12
                     and arrival_tails[3] <= arrival_tails[2] + 1e-12
                     and abs(arrival_tails[0] + arrival_tails[2] + arrival_center - 1) <= 1e-8)
    near["arrival_change"] = {
        "available": change_ok,
        "fall20": arrival_tails[0] if change_ok else None,
        "fall40": arrival_tails[1] if change_ok else None,
        "rise20": arrival_tails[2] if change_ok else None,
        "rise40": arrival_tails[3] if change_ok else None,
        "reference_ugm3": number(arrival_change.get("referencePm")) if change_ok else None,
        "arrival_epoch": epoch(arrival_change.get("arrivalEpoch")) if change_ok else None,
        "model": arrival_change.get("modelVersion"),
    }
    trail_source = air.get("trail") or {}
    ride_ok = bool(clock_ok and epoch(trail_source.get("forecastIssuedEpoch")) == issued
                   and epoch(trail_source.get("startEpoch")) == issued + 5400
                   and epoch(trail_source.get("endEpoch")) == issued + 12600)
    ride = projection(trail_source, ride_ok)
    extrema = trail_source.get("rideExtrema") or {}
    extrema_clock = extrema.get("forecastClock") or {}
    extrema_ok = bool(clock_ok and extrema.get("available")
                      and epoch(extrema_clock.get("forecastIssuedEpoch")) == issued
                      and epoch(extrema_clock.get("windowStartEpoch")) == issued + 5400
                      and epoch(extrema_clock.get("windowEndEpoch")) == issued + 12600)
    ride["minimum_maximum"] = {
        "available": extrema_ok,
        "minimum_ugm3": number(extrema.get("low")) if extrema_ok else None,
        "maximum_ugm3": number(extrema.get("high")) if extrema_ok else None,
        "resolution_minutes": 15,
        "kind": "predicted_window_minimum_maximum",
        "model": extrema.get("modelVersion"),
    }
    ride.update(start_epoch=epoch(clock.get("windowStartEpoch")) if clock_ok else None,
                end_epoch=epoch(clock.get("windowEndEpoch")) if clock_ok else None,
                weather=window_weather((analysis.get("weather") or {}).get("trail") if clock_ok else None,
                                       now, issued, epoch(clock.get("windowStartEpoch")),
                                       epoch(clock.get("windowEndEpoch"))))
    chance = air.get("cyclingWindowForecast") or {}
    chance_ok = bool(clock_ok and chance.get("available")
                     and epoch(chance.get("forecastIssuedEpoch")) == issued
                     and epoch(chance.get("startEpoch")) == ride["start_epoch"]
                     and epoch(chance.get("endEpoch")) == ride["end_epoch"]
                     and number(chance.get("cutoffUgM3")) == 70
                     and probability(chance.get("chanceMeanAtOrBelowCutoff")) is not None)
    ride["mean_le70"] = {
        "available": chance_ok,
        "probability": probability(chance.get("chanceMeanAtOrBelowCutoff")) if chance_ok else None,
        "model": chance.get("modelVersion"),
        "experimental": True,
        "calibrated": False,
    }
    sessions = {}
    for name in ("morning", "afternoon"):
        session = (analysis.get("windows") or {}).get(name) or {}
        start, end = epoch(session.get("startEpoch")), epoch(session.get("endEpoch"))
        session_ok = bool(forecast_ok and epoch(session.get("forecastIssuedEpoch")) == issued
                          and start and end and start > issued and end - start == 7200)
        sessions[name] = {
            "issued_epoch": epoch(session.get("forecastIssuedEpoch")),
            "start_epoch": start,
            "end_epoch": end,
            "logistics_epoch": epoch(session.get("decisionEpoch")),
            "pm": projection(session.get("particleForecast"), session_ok),
            "weather": window_weather(session.get("weatherForecast") if session_ok else None,
                                      now, issued, start, end),
        }
    forecast_available = bool(forecast_ok and (near["available"] or any(
        s["pm"]["available"] for s in sessions.values())))
    complete_outlook = bool(near["available"] and ride["available"] and ride["weather"]["available"]
                            and all(s["pm"]["available"] and s["weather"]["available"]
                                    for s in sessions.values()))
    fresh = bool(current["available"] and current["fresh"] and weather_out["available"]
                 and weather_out["fresh"] and forecast_available
                 and complete_outlook and forecast_age <= 120 and not delivery.get("error"))
    usable = bool(current["available"] or weather_out["available"] or near["available"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_epoch": now,
        "timezone": "Asia/Kuala_Lumpur",
        "refresh_after_s": 60,
        "dashboard_build": dashboard_build,
        "status": "ok" if fresh else "degraded" if usable else "unavailable",
        "current": current,
        "weather": weather_out,
        "tennis_morning": tennis_morning(weather, now),
        "history": rlcd_history.build_history(history_rows, now, max_points=DEVICE_HISTORY_MAX_POINTS),
        "forecast": {"available": forecast_available, "fresh": bool(forecast_available and forecast_age <= 120
                                                               and not delivery.get("error")),
                     "issued_epoch": issued, "age_s": forecast_age,
                     "near90": near, "ride90_210": ride, "sessions": sessions},
        "limitations": ["single_local_pm_sensor", "experimental_models_uncalibrated",
                        "rain_and_wind_do_not_guarantee_clearing", "70_is_user_planning_cutoff_not_safety_limit",
                        "windows_are_outlooks_not_sport_recommendations"],
    }
    return enforce_payload_budget(result)
