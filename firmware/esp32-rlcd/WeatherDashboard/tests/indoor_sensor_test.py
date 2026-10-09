# SPDX-License-Identifier: Apache-2.0
"""Exercise production IndoorSensor.cpp with a clock and SHTC3 I2C simulator.

The real compensation, driver, CRC handling, cadence and calibration setter are
compiled unchanged. Recorded SHTC3 packets and independent numeric expectations
drive the public API; no Arduino SDK, network or hardware is needed.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ARDUINO = r"""
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
using std::min;
using std::max;
uint32_t millis();
void delay(uint32_t milliseconds);
void delayMicroseconds(uint32_t microseconds);
"""


WIRE = r"""
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>
struct SensorFrame {
  std::array<uint8_t, 6> bytes;
  uint8_t received = 6;
};
struct CommandRecord {
  uint16_t code;
  uint64_t microseconds;
  uint8_t result;
};
class FakeWire {
 public:
  bool begun = false;
  int sda = -1, scl = -1;
  uint32_t frequency = 0, timeout = 0;
  uint16_t failCommand = 0, lastCommand = 0;
  unsigned failCount = 0, measurementReads = 0, idReads = 0;
  bool corruptId = false;
  uint32_t lastWakeAttemptMs = 0;
  uint64_t lastWakeUs = 0, lastMeasurementUs = 0, lastReadUs = 0;
  std::deque<SensorFrame> frames;
  std::vector<CommandRecord> commands;
  void begin(int data, int clock, uint32_t hz);
  void setTimeOut(uint32_t milliseconds);
  void beginTransmission(uint8_t address);
  size_t write(uint8_t byte);
  uint8_t endTransmission();
  uint8_t requestFrom(uint8_t address, uint8_t count);
  int read();
 private:
  std::vector<uint8_t> transmit, receive;
  size_t readIndex = 0;
};
extern FakeWire Wire;
"""


HARNESS = r"""
#include "IndoorSensor.h"
#include "Wire.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static uint64_t clockUs = 1000000;
static size_t checks = 0;
FakeWire Wire;
uint32_t millis() { return static_cast<uint32_t>(clockUs / 1000); }
void delay(uint32_t ms) { clockUs += uint64_t(ms) * 1000; }
void delayMicroseconds(uint32_t us) { clockUs += us; }
static void check(bool condition, const char *message) {
  ++checks;
  if (!condition) {
    std::fprintf(stderr, "FAIL at check %zu: %s (clock=%u ms)\n", checks, message, millis());
    std::abort();
  }
}
static bool close(float actual, double expected, double tolerance = .003) {
  return std::isfinite(actual) && std::fabs(double(actual) - expected) <= tolerance;
}
void FakeWire::begin(int data, int clock, uint32_t hz) {
  begun = true; sda = data; scl = clock; frequency = hz;
}
void FakeWire::setTimeOut(uint32_t milliseconds) { timeout = milliseconds; }
void FakeWire::beginTransmission(uint8_t address) {
  check(address == 0x70, "SHTC3 address for commands"); transmit.clear();
}
size_t FakeWire::write(uint8_t byte) { transmit.push_back(byte); return 1; }
uint8_t FakeWire::endTransmission() {
  check(transmit.size() == 2, "two-byte SHTC3 command");
  lastCommand = uint16_t(uint16_t(transmit[0]) << 8) | transmit[1];
  uint8_t result = 0;
  if (lastCommand == failCommand && failCount) { --failCount; result = 2; }
  commands.push_back({lastCommand, clockUs, result});
  if (lastCommand == 0x3517) { lastWakeAttemptMs = millis(); lastWakeUs = clockUs; }
  if (lastCommand == 0x7866 && !result) {
    check(clockUs - lastWakeUs >= 240, "wake delay meets SHTC3 minimum");
    lastMeasurementUs = clockUs;
  }
  return result;
}
uint8_t FakeWire::requestFrom(uint8_t address, uint8_t count) {
  check(address == 0x70, "SHTC3 address for reads");
  receive.clear(); readIndex = 0;
  if (count == 3) {
    check(lastCommand == 0xEFC8, "read follows ID command");
    // Sensirion's published CRC example: BE EF -> 92.
    receive = {0xBE, 0xEF, uint8_t(corruptId ? 0x93 : 0x92)}; ++idReads;
    return 3;
  }
  check(count == 6 && lastCommand == 0x7866, "normal temperature-first measurement request");
  check(clockUs - lastMeasurementUs >= 20000, "conversion delay before data read");
  check(!frames.empty(), "a scripted measurement exists");
  const SensorFrame frame = frames.front(); frames.pop_front();
  receive.assign(frame.bytes.begin(), frame.bytes.begin() + frame.received);
  ++measurementReads; lastReadUs = clockUs;
  return frame.received;
}
int FakeWire::read() {
  check(readIndex < receive.size(), "read stays within received I2C bytes");
  return receive[readIndex++];
}

// Encoded SHTC3 words and CRC bytes are fixed protocol fixtures, not values
// calculated by the production correction helper. Numeric expectations below
// were evaluated separately from the temperature/RH equations.
static const SensorFrame frame32{{0x70,0xA4,0x82,0x8C,0xCD,0x1D}}; // ~32C, 55%.
static const SensorFrame frame34{{0x73,0x91,0x9F,0x80,0x00,0xA2}}; // ~34C, 50%.
static const SensorFrame frame30{{0x6D,0xB7,0xBC,0x87,0xAE,0xEF}}; // ~30C, 53%.
static const SensorFrame frame122{{0xF4,0x4C,0x5A,0x33,0x33,0x88}}; // ~122C, 20%.
static const SensorFrame frame124{{0xF7,0x39,0x7A,0x33,0x33,0x88}}; // ~124C, 20%.
static const SensorFrame frameHumid{{0x5F,0x16,0x1A,0xF3,0x33,0x22}}; // ~20C, 95%.
static void reset() { Wire = FakeWire{}; clockUs = 1000000; }
static void nextAttempt() {
  const uint32_t elapsed = uint32_t(millis() - Wire.lastWakeAttemptMs);
  if (elapsed < 10000) delay(10000 - elapsed);
}
static void take(IndoorSensor &sensor, SensorFrame frame) {
  Wire.frames.push_back(frame); nextAttempt(); sensor.update();
}
static bool sameFloat(float left, float right) {
  return left == right || (std::isnan(left) && std::isnan(right));
}
static void sameReading(const IndoorReading &left, const IndoorReading &right) {
  check(left.valid == right.valid && left.sensorId == right.sensorId, "reading identity and validity unchanged");
  check(sameFloat(left.temperatureC, right.temperatureC) && sameFloat(left.humidityPct, right.humidityPct),
        "corrected current values unchanged");
  check(sameFloat(left.rawTemperatureC, right.rawTemperatureC) && sameFloat(left.rawHumidityPct, right.rawHumidityPct),
        "factory-converted raw values unchanged");
  check(left.humidityClamped == right.humidityClamped, "humidity clamp flag unchanged");
  check(sameFloat(left.minimumC, right.minimumC) && sameFloat(left.maximumC, right.maximumC)
        && sameFloat(left.minimumHumidity, right.minimumHumidity) && sameFloat(left.maximumHumidity, right.maximumHumidity),
        "all range endpoints unchanged");
  check(left.lastGoodMs == right.lastGoodMs && left.samples == right.samples && left.errors == right.errors,
        "observation timestamp, sample count and error count unchanged");
}
static void rangesEmpty(const IndoorReading &reading) {
  check(std::isnan(reading.minimumC) && std::isnan(reading.maximumC)
        && std::isnan(reading.minimumHumidity) && std::isnan(reading.maximumHumidity), "invalid ranges remain empty");
}
static void rangesAtCurrent(const IndoorReading &reading) {
  check(reading.minimumC == reading.temperatureC && reading.maximumC == reading.temperatureC
        && reading.minimumHumidity == reading.humidityPct && reading.maximumHumidity == reading.humidityPct,
        "new range generation contains only its current corrected measurement");
}

static void centralValuesCadenceAndTransport() {
  reset(); Wire.frames.push_back(frame32); IndoorSensor sensor; sensor.begin();
  const IndoorReading first = sensor.reading();
  check(Wire.begun && Wire.sda == 13 && Wire.scl == 14 && Wire.frequency == 100000 && Wire.timeout == 50,
        "driver owns the configured shared I2C bus");
  check(first.valid && first.samples == 1 && first.errors == 0 && first.sensorId == 0xBEEF,
        "valid ID and first observation pass actual CRC processing");
  check(close(sensor.temperatureOffset(), -4.8), "default board-heat correction");
  check(close(first.rawTemperatureC, 32.0004272461) && close(first.rawHumidityPct, 55.0003051758),
        "manufacturer raw conversion retained");
  check(close(first.temperatureC, 27.2004272461) && close(first.humidityPct, 72.5111353726),
        "central corrected temperature and compensated humidity");
  check(!first.humidityClamped, "ordinary compensated humidity is not clamped");
  rangesAtCurrent(first);
  check(Wire.commands.size() == 7 && Wire.commands[1].code == 0x805D && Wire.commands[2].code == 0xEFC8,
        "reset and ID sequence precede first measurement");
  check(Wire.commands[1].microseconds - Wire.commands[0].microseconds >= 240
        && Wire.commands[2].microseconds - Wire.commands[1].microseconds >= 2000,
        "wake/reset timing respected during initialization");
  check(Wire.lastCommand == 0xB098, "measurement always returns sensor to sleep");
  const uint32_t attempted = Wire.lastWakeAttemptMs;
  const unsigned reads = Wire.measurementReads;
  clockUs = uint64_t(attempted + 9999) * 1000; sensor.update();
  check(Wire.measurementReads == reads, "no new measurement before ten-second cadence");
  sameReading(first, sensor.reading());
  Wire.frames.push_back(frame34); clockUs = uint64_t(attempted + 10000) * 1000; sensor.update();
  const auto &second = sensor.reading();
  check(second.samples == 2 && Wire.measurementReads == reads + 1, "measurement resumes at exact ten-second boundary");
  check(close(second.minimumC, 27.2004272461) && close(second.maximumC, 29.2004730225)
        && close(second.minimumHumidity, 65.6551142013) && close(second.maximumHumidity, 72.5111353726),
        "ranges track corrected observations rather than raw values");
}

static void recalibrationAndRejectedOffsets() {
  reset(); Wire.frames.push_back(frame32); IndoorSensor sensor; sensor.begin(); take(sensor, frame34);
  const IndoorReading before = sensor.reading();
  const uint64_t unchangedClock = clockUs; const size_t commands = Wire.commands.size();
  check(sensor.setTemperatureOffset(-2), "fresh offset change accepted");
  const auto &changed = sensor.reading();
  check(changed.valid && close(changed.temperatureC, 32.0004730225) && close(changed.humidityPct, 55.9447813467),
        "setter recalculates the same retained raw observation");
  check(changed.rawTemperatureC == before.rawTemperatureC && changed.rawHumidityPct == before.rawHumidityPct
        && changed.samples == before.samples && changed.errors == before.errors && changed.lastGoodMs == before.lastGoodMs,
        "recalibration acquires no measurement and refreshes no observation metadata");
  check(clockUs == unchangedClock && Wire.commands.size() == commands, "setter does not delay or access I2C");
  rangesAtCurrent(changed);
  take(sensor, frame30);
  check(close(sensor.reading().minimumC, 28.0003814697) && close(sensor.reading().maximumC, 32.0004730225),
        "later samples extend only the new calibration range");
  const IndoorReading retained = sensor.reading();
  check(sensor.setTemperatureOffset(-2), "same offset succeeds"); sameReading(retained, sensor.reading());
  for (float invalid : {-10.01f, 10.01f, NAN, INFINITY, -INFINITY}) {
    check(!sensor.setTemperatureOffset(invalid), "out-of-range or nonfinite offset rejected");
    check(sensor.temperatureOffset() == -2, "invalid offset does not change calibration");
    sameReading(retained, sensor.reading());
  }
  check(sensor.setTemperatureOffset(10) && sensor.reading().valid, "inclusive +10-degree boundary accepted");
  check(sensor.setTemperatureOffset(-10) && sensor.reading().valid, "inclusive -10-degree boundary accepted");
}

static void staleAndNeverMeasuredSamples() {
  reset(); IndoorSensor neverMeasured;
  check(neverMeasured.setTemperatureOffset(0), "offset can be configured before first observation");
  check(!neverMeasured.reading().valid && !neverMeasured.reading().samples, "setter cannot fabricate first observation");
  Wire.failCommand = 0x3517; Wire.failCount = 1; neverMeasured.begin();
  check(!neverMeasured.reading().valid && neverMeasured.reading().errors == 1, "initial I2C error remains invalid");
  check(neverMeasured.setTemperatureOffset(-3), "calibration still accepts configuration after an I2C error");
  check(!neverMeasured.reading().valid && !neverMeasured.reading().samples, "error plus setter cannot fabricate data");

  reset(); Wire.frames.push_back(frame32); IndoorSensor sensor; sensor.begin();
  const IndoorReading previous = sensor.reading();
  clockUs = uint64_t(previous.lastGoodMs + 30001) * 1000;
  check(sensor.setTemperatureOffset(-3), "stale sample accepts configuration without acquiring data");
  check(!sensor.reading().valid && std::isnan(sensor.reading().temperatureC) && std::isnan(sensor.reading().humidityPct),
        "expired retained observation cannot be resurrected by setter");
  rangesEmpty(sensor.reading());
  check(sensor.reading().rawTemperatureC == previous.rawTemperatureC && sensor.reading().rawHumidityPct == previous.rawHumidityPct
        && sensor.reading().samples == previous.samples && sensor.reading().lastGoodMs == previous.lastGoodMs,
        "stale recalibration retains raw provenance and original clock");
  Wire.failCommand = 0x3517; Wire.failCount = 1; sensor.update();
  check(!sensor.reading().valid && sensor.reading().errors == 1, "later I2C error cannot resurrect explicitly invalid data");
}

static void ratedCorrectionFailureAndRecovery() {
  reset(); Wire.frames.push_back(frame122); IndoorSensor sensor;
  check(sensor.setTemperatureOffset(0), "identity correction configured before begin"); sensor.begin();
  const IndoorReading initial = sensor.reading();
  check(initial.valid && close(initial.temperatureC, 121.999816895)
        && initial.temperatureC == initial.rawTemperatureC && initial.humidityPct == initial.rawHumidityPct,
        "zero correction preserves exact factory values");
  check(sensor.setTemperatureOffset(10), "valid configuration may put a retained measurement outside its rated range");
  check(!sensor.reading().valid && std::isnan(sensor.reading().temperatureC), "out-of-range corrected measurement is unavailable");
  rangesEmpty(sensor.reading());
  check(sensor.reading().samples == initial.samples && sensor.reading().lastGoodMs == initial.lastGoodMs,
        "failed corrected value does not refresh its observation");
  Wire.failCommand = 0x3517; Wire.failCount = 1; nextAttempt(); sensor.update();
  check(!sensor.reading().valid && sensor.reading().errors == 1, "fresh retained raw data is not resurrected by I2C failure");
  take(sensor, frame124);
  check(!sensor.reading().valid && sensor.reading().errors == 2 && sensor.reading().samples == 1,
        "received but invalid corrected sample is not counted as a new observation");
  rangesEmpty(sensor.reading());
  check(sensor.reading().rawTemperatureC == initial.rawTemperatureC && sensor.reading().lastGoodMs == initial.lastGoodMs,
        "rejected measurement preserves prior raw provenance and clock");
  take(sensor, frame30);
  check(sensor.reading().valid && sensor.reading().samples == 2 && close(sensor.reading().temperatureC, 40.0003814697)
        && close(sensor.reading().humidityPct, 30.4612043934), "next valid sample recovers using configured offset");
  rangesAtCurrent(sensor.reading());
  check(sensor.reading().lastGoodMs != initial.lastGoodMs, "recovery gets its own measurement timestamp");
}

static void crcAndTransportRetention() {
  reset(); Wire.frames.push_back(frame32); IndoorSensor sensor; sensor.begin();
  const IndoorReading original = sensor.reading();
  SensorFrame badTemperature = frame34; badTemperature.bytes[2] ^= 1;
  SensorFrame badHumidity = frame34; badHumidity.bytes[5] ^= 1;
  take(sensor, badTemperature);
  check(sensor.reading().valid && sensor.reading().errors == 1 && sensor.reading().samples == 1,
        "bad temperature CRC retains a recent good reading");
  take(sensor, badHumidity);
  check(sensor.reading().valid && sensor.reading().errors == 2, "bad humidity CRC follows same retention policy");
  // Land the failed update exactly 30 seconds after its last good observation;
  // wake settling (500us) plus conversion (20ms) are actual production delays.
  clockUs = uint64_t(original.lastGoodMs + 30000) * 1000 - 20500;
  Wire.frames.push_back(badTemperature); sensor.update();
  check(millis() - original.lastGoodMs == 30000 && sensor.reading().valid,
        "ordinary CRC failure retains data at inclusive 30-second boundary");
  take(sensor, badHumidity);
  check(!sensor.reading().valid && sensor.reading().errors == 4, "ordinary failure expires data beyond 30 seconds");
  check(sensor.reading().lastGoodMs == original.lastGoodMs && sensor.reading().samples == original.samples
        && sensor.reading().rawTemperatureC == original.rawTemperatureC, "failed frames never refresh or replace observation");
  check(Wire.lastCommand == 0xB098, "CRC failure still sends sensor back to sleep");
  take(sensor, frame34);
  check(sensor.reading().valid && sensor.reading().samples == 2, "valid CRC packet recovers expired reading");
  SensorFrame shortRead = frame30; shortRead.received = 5;
  take(sensor, shortRead);
  check(sensor.reading().valid && sensor.reading().samples == 2 && sensor.reading().errors == 5,
        "short I2C transfer retains a recent good observation");

  reset(); Wire.corruptId = true; Wire.frames.push_back(frame32); IndoorSensor badId; badId.begin();
  check(badId.reading().sensorId == 0 && badId.reading().valid, "bad ID CRC cannot publish a sensor ID or block valid measurements");
  reset(); Wire.frames.push_back(frameHumid); IndoorSensor humid; humid.begin();
  check(humid.reading().valid && humid.reading().humidityPct == 100 && humid.reading().humidityClamped,
        "compensation clamp is centrally retained in sensor reading");
}

static void millisRollover() {
  reset(); clockUs = uint64_t(UINT32_MAX - 9000) * 1000;
  Wire.frames.push_back(frame32); IndoorSensor sensor; sensor.begin();
  const uint32_t before = sensor.reading().lastGoodMs;
  take(sensor, frame30);
  check(sensor.reading().valid && sensor.reading().samples == 2 && sensor.reading().lastGoodMs < before,
        "measurement cadence and freshness survive millis rollover");
  check(sensor.setTemperatureOffset(-2) && sensor.reading().valid,
        "fresh retained sample recalibrates after millis rollover");
}

int main() {
  centralValuesCadenceAndTransport(); recalibrationAndRejectedOffsets();
  staleAndNeverMeasuredSamples(); ratedCorrectionFailureAndRecovery();
  crcAndTransportRetention(); millisRollover();
  std::printf("PASS: %zu production sensor checks; fixed I2C packets, raw/corrected values, CRC/cadence, range reset, rejected offsets, stale/error non-resurrection, recovery and millis rollover\n", checks);
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
    (build / "Arduino.h").write_text(ARDUINO, encoding="utf-8")
    (build / "Wire.h").write_text(WIRE, encoding="utf-8")
    source = build / "indoor_sensor_test.cpp"
    source.write_text(HARNESS, encoding="utf-8")
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = build / ("indoor_sensor_test" + suffix)
    subprocess.run([args.cxx, *args.cxx_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(build), "-I", str(firmware),
                    str(source), str(firmware / "IndoorSensor.cpp"), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
