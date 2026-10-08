#pragma once

#include "SpeechClips.h"
#include <cmath>
#include <cstddef>
#include <cstdint>

// Capture this on the loop task. Playback never accesses the live JSON document.
struct WeatherReadoutSnapshot {
  bool currentAvailable = false;
  bool currentOld = false;
  float pm25 = NAN;
  float temperature = NAN;
  float humidity = NAN;
  float heatIndex = NAN;
  bool forecastAvailable = false;
  bool forecastOld = false;
  float wind = NAN;
  float rainChance = NAN;
};

inline bool readoutValueValid(float value, float minimum, float maximum) {
  return std::isfinite(value) && value >= minimum && value <= maximum;
}

inline bool readoutHasCurrent(const WeatherReadoutSnapshot &data) {
  return data.currentAvailable && (readoutValueValid(data.pm25, 0, 9999.9f)
      || readoutValueValid(data.temperature, -100, 100)
      || readoutValueValid(data.humidity, 0, 100)
      || readoutValueValid(data.heatIndex, -100, 100));
}

inline bool readoutHasForecast(const WeatherReadoutSnapshot &data) {
  return data.forecastAvailable && (readoutValueValid(data.wind, 0, 1000)
      || readoutValueValid(data.rainChance, 0, 100));
}

struct SpeechPlaylist {
  static constexpr size_t kCapacity = 384;
  SpeechClip clips[kCapacity]{};
  size_t count = 0;

  bool add(SpeechClip clip) {
    if (count == kCapacity) return false;
    clips[count++] = clip;
    return true;
  }
};

inline bool speechWholeNumber(SpeechPlaylist &out, unsigned value) {
  if (value >= 1000) {
    if (!speechWholeNumber(out, value / 1000) || !out.add(SpeechClip::Thousand)) return false;
    value %= 1000;
    if (!value) return true;
  }
  if (value >= 100) {
    if (!out.add(static_cast<SpeechClip>(value / 100)) || !out.add(SpeechClip::Hundred)) return false;
    value %= 100;
    if (!value) return true;
  }
  if (value < 20) return out.add(static_cast<SpeechClip>(value));
  // N0..N19 are followed by N20, N30, ... N90 in the clip table.
  if (!out.add(static_cast<SpeechClip>(20 + value / 10 - 2))) return false;
  return value % 10 == 0 || out.add(static_cast<SpeechClip>(value % 10));
}

inline bool speechNumberPrecision(SpeechPlaylist &out, float value, unsigned decimalPlaces) {
  if (decimalPlaces > 2 || !std::isfinite(value)) return false;
  const unsigned factor = decimalPlaces == 2 ? 100 : decimalPlaces == 1 ? 10 : 1;
  const float maximum = decimalPlaces == 2 ? 9999.99f : 9999.9f;
  if (std::fabs(value) > maximum) return false;
  const long scaled = std::lround(std::fabs(value) * factor);
  if (scaled > long(10000 * factor - 1)) return false;
  // Build transactionally, so an invalid or oversized measurement adds nothing.
  SpeechPlaylist candidate = out;
  if (value < 0 && scaled && !candidate.add(SpeechClip::Minus)) return false;
  if (!speechWholeNumber(candidate, unsigned(scaled / factor))) return false;
  if (decimalPlaces) {
    if (!candidate.add(SpeechClip::Point)) return false;
    for (unsigned place = factor / 10; place; place /= 10)
      if (!candidate.add(static_cast<SpeechClip>((scaled / place) % 10))) return false;
  }
  out = candidate;
  return true;
}

inline bool speechNumber(SpeechPlaylist &out, float value, bool decimal) {
  return speechNumberPrecision(out, value, decimal ? 1 : 0);
}

inline bool speechMeasurement(SpeechPlaylist &out, SpeechClip label, float value,
                              SpeechClip unit, bool decimal, float minimum, float maximum) {
  if (!std::isfinite(value) || value < minimum || value > maximum) return true;
  SpeechPlaylist candidate = out;
  if (!candidate.add(label) || !speechNumber(candidate, value, decimal) || !candidate.add(unit)) return false;
  out = candidate;
  return true;
}

inline bool buildWeatherReadout(const WeatherReadoutSnapshot &data, SpeechPlaylist &out) {
  out = SpeechPlaylist{};
  if (readoutHasCurrent(data)) {
    if (data.currentOld && !out.add(SpeechClip::CurrentDataOld)) return false;
    if (!speechMeasurement(out, SpeechClip::CurrentPM25, data.pm25, SpeechClip::Micrograms, true, 0, 9999.9f)
        || !speechMeasurement(out, SpeechClip::OutdoorTemperature, data.temperature, SpeechClip::DegreesCelsius, true, -100, 100)
        || !speechMeasurement(out, SpeechClip::Humidity, data.humidity, SpeechClip::Percent, false, 0, 100)
        || !speechMeasurement(out, SpeechClip::HeatIndex, data.heatIndex, SpeechClip::DegreesCelsius, true, -100, 100)) return false;
  } else if (!out.add(SpeechClip::Unavailable)) return false;
  if (readoutHasForecast(data)) {
    if (data.forecastOld && !out.add(SpeechClip::ForecastOld)) return false;
    if (!speechMeasurement(out, SpeechClip::ForecastWind, data.wind, SpeechClip::KilometresPerHour, false, 0, 1000)
        || !speechMeasurement(out, SpeechClip::RainNextHour, data.rainChance, SpeechClip::Percent, false, 0, 100)) return false;
  } else if (!out.add(SpeechClip::ForecastUnavailable)) return false;
  if (!out.count) return out.add(SpeechClip::Unavailable);
  return true;
}
