// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <limits>

// User-selected starting correction for desk use outside the direct airflow.
// This is an adjustable temperature offset, not a sensor factory calibration.
constexpr float kIndoorDefaultTemperatureOffsetC = -4.0f;

struct IndoorCompensationResult {
  bool valid = false;
  double temperatureC = std::numeric_limits<double>::quiet_NaN();
  double humidityPct = std::numeric_limits<double>::quiet_NaN();
  bool humidityClamped = false;
};

inline bool indoorOffsetValid(double offset) {
  return std::isfinite(offset) && offset >= -10.0 && offset <= 10.0;
}

inline IndoorCompensationResult indoorCompensate(double rawTemperatureC, double rawHumidityPct,
                                                double offset) {
  if (!indoorOffsetValid(offset) || !std::isfinite(rawTemperatureC)
      || rawTemperatureC < -40.0 || rawTemperatureC > 125.0
      || !std::isfinite(rawHumidityPct) || rawHumidityPct < 0.0 || rawHumidityPct > 100.0) return {};

  const double correctedTemperatureC = rawTemperatureC + offset;
  if (!std::isfinite(correctedTemperatureC)
      || correctedTemperatureC < -40.0 || correctedTemperatureC > 125.0) return {};

  IndoorCompensationResult result;
  result.valid = true;
  result.temperatureC = correctedTemperatureC;
  if (offset == 0.0) {
    // Preserve the exact raw values, including the zero/100 percent endpoints.
    result.humidityPct = rawHumidityPct;
    return result;
  }

  // Sensirion Humidity Sensors Design Guide, equation 1: preserve water-vapor
  // partial pressure while correcting the temperature used for relative RH.
  // Evaluate the exponent in double precision before the caller stores floats.
  constexpr double a = 17.62;
  constexpr double b = 243.21;
  const double exponent = a * b * (rawTemperatureC - correctedTemperatureC)
      / ((b + rawTemperatureC) * (b + correctedTemperatureC));
  const double correctedHumidityPct = rawHumidityPct * std::exp(exponent);
  if (!std::isfinite(correctedHumidityPct)) return {};
  result.humidityClamped = correctedHumidityPct < 0.0 || correctedHumidityPct > 100.0;
  result.humidityPct = correctedHumidityPct < 0.0 ? 0.0
      : correctedHumidityPct > 100.0 ? 100.0 : correctedHumidityPct;
  return result;
}
