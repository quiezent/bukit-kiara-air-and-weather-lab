#pragma once

#include <ArduinoJson.h>
#include <cmath>
#include "First20Event.h"

struct RidePmRange {
  bool available = false;
  bool minimumMaximum = false;
  float low = NAN;
  float high = NAN;
};

inline bool forecastPmNumberValid(JsonVariantConst value) {
  if (!value.is<float>()) return false;
  const float number = value.as<float>();
  return std::isfinite(number) && number >= 0 && number <= 9999.9f;
}

inline bool forecastModelPointAvailable(JsonVariantConst pm) {
  return pm["available"].is<bool>() && pm["available"].as<bool>()
      && forecastPmNumberValid(pm["pm25_ugm3"])
      && (pm["role"] == "raw_model_output" || pm["role"] == "experimental_model_output"
          || pm["role"] == "experimental_window_mean");
}

inline RidePmRange ridePmRange(JsonVariantConst ride) {
  RidePmRange result;
  const JsonVariantConst extrema = ride["minimum_maximum"];
  // A supplied modern object, including explicit null/unavailable, owns these
  // semantics. Its failure must not resurrect old quantiles as extrema.
  if (!extrema.isUnbound()) {
    if (!extrema.is<JsonObjectConst>() || !extrema["available"].is<bool>()
        || !extrema["available"].as<bool>()
        || extrema["kind"] != "predicted_window_minimum_maximum"
        || !extrema["resolution_minutes"].is<uint32_t>()
        || extrema["resolution_minutes"].as<uint32_t>() != 15
        || !forecastPmNumberValid(extrema["minimum_ugm3"])
        || !forecastPmNumberValid(extrema["maximum_ugm3"])) return result;
    result.low = extrema["minimum_ugm3"].as<float>();
    result.high = extrema["maximum_ugm3"].as<float>();
    result.minimumMaximum = true;
  } else {
    if (!ride["available"].is<bool>() || !ride["available"].as<bool>()
        || ride["range_kind"] != "empirical_q10_q90"
        || !forecastPmNumberValid(ride["range_low_ugm3"])
        || !forecastPmNumberValid(ride["range_high_ugm3"])) return result;
    result.low = ride["range_low_ugm3"].as<float>();
    result.high = ride["range_high_ugm3"].as<float>();
  }
  if (result.high < result.low) return RidePmRange{};
  result.available = true;
  return result;
}

inline bool windowWeatherCoverageAvailable(JsonVariantConst weather) {
  if (weather["coverage_verified"].isUnbound() && weather["coverage"].isUnbound()) return true;
  return weather["coverage_verified"].is<bool>() && weather["coverage_verified"].as<bool>()
      && weather["coverage"]["complete"].is<bool>() && weather["coverage"]["complete"].as<bool>();
}

inline First20Call first20ModelCall(JsonVariantConst event) {
  if (!event["available"].is<bool>() || !event["available"].as<bool>()
      || !event["rise_probability"].is<double>() || !event["drop_probability"].is<double>()
      || !event["none_probability"].is<double>() || !event["reference_ugm3"].is<double>()) {
    return First20Call::Invalid;
  }
  // The model summary describes the raw winner. Operational qualification and
  // direction remain separate metadata; no qualification is inferred here.
  const JsonVariantConst direction = event["diagnostic_direction"].isUnbound()
      ? event["direction"] : event["diagnostic_direction"];
  const First20Call declared = direction == "rise" ? First20Call::Rise
      : direction == "drop" ? First20Call::Drop
      : direction == "unresolved" ? First20Call::Unresolved : First20Call::Invalid;
  return classifyFirst20(event["rise_probability"].as<double>(), event["drop_probability"].as<double>(),
      event["none_probability"].as<double>(), event["reference_ugm3"].as<double>(), declared);
}

inline bool first20ModelNoChangeMostLikely(JsonVariantConst event) {
  return first20ModelCall(event) == First20Call::Unresolved
      && event["none_probability"].as<double>() > event["rise_probability"].as<double>()
      && event["none_probability"].as<double>() > event["drop_probability"].as<double>();
}
