# SPDX-License-Identifier: Apache-2.0
"""Compile the actual pure indoor compensation helper without ESP32 hardware.

Run from the package: python WeatherDashboard/tests/indoor_compensation_test.py --cxx g++ --build-dir build/host/indoor_compensation
Use repeatable --cxx-arg=ARG for compiler drivers such as zig c++.
"""

import argparse
from pathlib import Path
import subprocess
import sys


HARNESS = r"""
#include "IndoorCompensation.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>

size_t checks = 0;

bool close(double left, double right, double tolerance = 1e-10) {
  return std::fabs(left - right) <= tolerance;
}

double saturationProxy(double temperature) {
  return std::exp(17.62 * temperature / (243.21 + temperature));
}

double vaporPressureProxy(double temperature, double humidity) {
  return humidity / 100.0 * saturationProxy(temperature);
}

double dewPoint(double temperature, double humidity) {
  const double gamma = std::log(humidity / 100.0) + 17.62 * temperature / (243.21 + temperature);
  return 243.21 * gamma / (17.62 - gamma);
}

void invalid(double temperature, double humidity, double offset) {
  const auto result = indoorCompensate(temperature, humidity, offset);
  assert(!result.valid && std::isnan(result.temperatureC) && std::isnan(result.humidityPct));
  assert(!result.humidityClamped);
  ++checks;
}

void offsetAndInputLimits() {
  assert(close(kIndoorDefaultTemperatureOffsetC, -4.8, 1e-6)); ++checks;
  for (double offset : {-10.0, -4.8, -0.0, 0.0, 4.8, 10.0}) {
    assert(indoorOffsetValid(offset)); ++checks;
  }
  for (double offset : {-10.000000001, 10.000000001, -11.0, 11.0,
                         double(NAN), double(INFINITY), double(-INFINITY)}) {
    assert(!indoorOffsetValid(offset)); ++checks;
    invalid(25, 50, offset);
  }
  for (double temperature : {-40.000000001, 125.000000001, -45.0, 130.0,
                              double(NAN), double(INFINITY), double(-INFINITY)}) {
    invalid(temperature, 50, 0);
  }
  for (double humidity : {-0.000000001, 100.000000001, -1.0, 101.0,
                           double(NAN), double(INFINITY), double(-INFINITY)}) {
    invalid(25, humidity, 0);
  }
  // Raw inputs can be rated while their corrected temperature is out of range.
  invalid(-40, 50, -.00001);
  invalid(125, 50, .00001);
  invalid(-35, 50, -10);
  invalid(120, 50, 10);
  for (const auto &result : {indoorCompensate(-40, 50, 0),
                            indoorCompensate(125, 50, 0),
                            indoorCompensate(-30, 10, -10),
                            indoorCompensate(115, 10, 10)}) {
    assert(result.valid && result.temperatureC >= -40 && result.temperatureC <= 125);
    assert(result.humidityPct >= 0 && result.humidityPct <= 100); ++checks;
  }
}

void identityAndPhotoReadings() {
  for (double temperature : {-40.0, 0.0, 26.0, 30.8, 125.0}) {
    for (double humidity : {0.0, 0.0001, 50.0, 58.4, 100.0}) {
      const auto result = indoorCompensate(temperature, humidity, 0);
      assert(result.valid && result.temperatureC == temperature && result.humidityPct == humidity);
      assert(!result.humidityClamped); ++checks;
    }
  }
  const auto desk = indoorCompensate(30.8, 58.4, -4.8);
  assert(desk.valid && close(desk.temperatureC, 26.0));
  // The earlier handheld comparison was 26C/75%; its temperature difference
  // supplies a starting offset selected for desk use. RH is compensated from
  // the raw reading, not independently forced to the comparator's percentage.
  assert(desk.humidityPct > 77 && desk.humidityPct < 78);
  assert(!desk.humidityClamped && !close(desk.humidityPct, 75)); ++checks;
  const auto defaults = indoorCompensate(30.8f, 58.4f, kIndoorDefaultTemperatureOffsetC);
  assert(defaults.valid && close(defaults.temperatureC, 26.0, 2e-6));
  assert(close(defaults.humidityPct, desk.humidityPct, 1e-5)); ++checks;
  // The later 30.2C vs28C airflow comparison has a different local offset.
  // Applying the desk offset still yields25.4C; this helper never invents an
  // airflow-specific adjustment or overwrites a caller-selected offset.
  const auto airflowWithDeskOffset = indoorCompensate(30.2, 50, -4.8);
  assert(airflowWithDeskOffset.valid && close(airflowWithDeskOffset.temperatureC, 25.4));
  const auto airflowComparison = indoorCompensate(30.2, 50, -2.2);
  assert(airflowComparison.valid && close(airflowComparison.temperatureC, 28.0)); ++checks;
}

void moistureAndDewPointInvariants() {
  for (double temperature : {-30.0, -5.0, 0.0, 15.0, 26.0, 30.8, 50.0, 100.0, 115.0}) {
    for (double humidity : {0.1, 1.0, 10.0, 40.0, 75.0, 95.0}) {
      for (double offset : {-10.0, -4.8, -1.0, 0.0, 1.0, 4.8, 10.0}) {
        const auto result = indoorCompensate(temperature, humidity, offset);
        assert(result.valid);
        if (result.humidityClamped) {
          assert(offset < 0 && result.humidityPct == 100); ++checks;
          continue; // Supersaturation is clamped, so the original dew point no longer applies.
        }
        const double before = vaporPressureProxy(temperature, humidity);
        const double after = vaporPressureProxy(result.temperatureC, result.humidityPct);
        assert(std::fabs(before - after) <= 1e-12 * (1 + std::fabs(before)));
        assert(close(dewPoint(temperature, humidity),
                     dewPoint(result.temperatureC, result.humidityPct), 1e-10));
        if (offset < 0) assert(result.humidityPct > humidity);
        if (offset > 0) assert(result.humidityPct < humidity);
        ++checks;
      }
    }
  }
  // Reversing an unclamped correction returns the original T/RH pair.
  const auto lower = indoorCompensate(30.8, 58.4, -4.8);
  const auto reverse = indoorCompensate(lower.temperatureC, lower.humidityPct, 4.8);
  assert(reverse.valid && close(reverse.temperatureC, 30.8)
      && close(reverse.humidityPct, 58.4)); ++checks;
}

void zeroAndClampedHumidity() {
  for (double temperature : {-30.0, 0.0, 30.0, 115.0}) {
    for (double offset : {-10.0, -4.8, 0.0, 4.8, 10.0}) {
      const auto dry = indoorCompensate(temperature, 0, offset);
      assert(dry.valid && dry.humidityPct == 0 && !dry.humidityClamped); ++checks;
      const auto saturated = indoorCompensate(temperature, 100, offset);
      assert(saturated.valid && saturated.humidityPct <= 100 && saturated.humidityPct >= 0);
      assert(saturated.humidityClamped == (offset < 0));
      if (offset <= 0) assert(saturated.humidityPct == 100);
      else assert(saturated.humidityPct < 100); ++checks;
    }
  }
  const auto condensation = indoorCompensate(30, 95, -10);
  assert(condensation.valid && condensation.temperatureC == 20
      && condensation.humidityPct == 100 && condensation.humidityClamped); ++checks;
  const auto boundary = indoorCompensate(-40, 100, 10);
  assert(boundary.valid && boundary.temperatureC == -30 && boundary.humidityPct < 100
      && !boundary.humidityClamped); ++checks;
}

int main() {
  offsetAndInputLimits(); identityAndPhotoReadings();
  moistureAndDewPointInvariants(); zeroAndClampedHumidity();
  std::printf("PASS: %zu indoor-compensation assertions; desk/airflow offsets, rated/input bounds, nonfinite rejection, zero identity, dew-point/vapor preservation, reverse correction and RH clamping\n", checks);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    firmware = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "indoor_compensation_test.cpp"
    source.write_text(HARNESS, encoding="utf-8")
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = build / ("indoor_compensation_test" + suffix)
    subprocess.run([args.cxx, *args.cxx_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(firmware), str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
