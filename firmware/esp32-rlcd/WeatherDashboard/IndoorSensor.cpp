#include "IndoorSensor.h"
#include <Wire.h>

// Sensirion SHTC3 at 0x70. SDA=13/SCL=14 from Waveshare's schematic.
// Retain the manufacturer's raw conversion, then centrally apply the user's
// temperature correction and humidity compensation for the board's heat.
static constexpr uint8_t ADDRESS = 0x70;

void IndoorSensor::resetRanges() {
  value.minimumC = value.maximumC = NAN;
  value.minimumHumidity = value.maximumHumidity = NAN;
  rangesInitialized = false;
}

void IndoorSensor::recordRanges() {
  if (!rangesInitialized) {
    value.minimumC = value.maximumC = value.temperatureC;
    value.minimumHumidity = value.maximumHumidity = value.humidityPct;
    rangesInitialized = true;
  } else {
    value.minimumC = min(value.minimumC, value.temperatureC);
    value.maximumC = max(value.maximumC, value.temperatureC);
    value.minimumHumidity = min(value.minimumHumidity, value.humidityPct);
    value.maximumHumidity = max(value.maximumHumidity, value.humidityPct);
  }
}

bool IndoorSensor::setTemperatureOffset(float offset) {
  if (!indoorOffsetValid(offset)) return false;
  if (offset == temperatureOffsetC) return true;
  temperatureOffsetC = offset;
  resetRanges();
  // Configuration changes never acquire a measurement or refresh its clock.
  // An expired/invalid retained sample must not become valid again here.
  if (value.valid && value.samples && uint32_t(millis() - value.lastGoodMs) <= 30000) {
    const IndoorCompensationResult corrected = indoorCompensate(
        value.rawTemperatureC, value.rawHumidityPct, temperatureOffsetC);
    if (corrected.valid) {
      value.temperatureC = corrected.temperatureC;
      value.humidityPct = corrected.humidityPct;
      value.humidityClamped = corrected.humidityClamped;
      recordRanges();
      return true;
    }
  }
  value.valid = false;
  value.temperatureC = value.humidityPct = NAN;
  value.humidityClamped = false;
  return true;
}

bool IndoorSensor::command(uint16_t code) {
  Wire.beginTransmission(ADDRESS);
  Wire.write(uint8_t(code >> 8));
  Wire.write(uint8_t(code));
  return Wire.endTransmission() == 0;
}

bool IndoorSensor::receive(uint8_t *bytes, uint8_t count) {
  if (Wire.requestFrom(ADDRESS, count) != count) return false;
  for (uint8_t i = 0; i < count; i++) bytes[i] = Wire.read();
  return true;
}

uint8_t IndoorSensor::crc(const uint8_t *bytes) {
  uint8_t result = 0xff;
  for (uint8_t i = 0; i < 2; i++) {
    result ^= bytes[i];
    for (uint8_t bit = 0; bit < 8; bit++)
      result = result & 0x80 ? uint8_t((result << 1) ^ 0x31) : uint8_t(result << 1);
  }
  return result;
}

void IndoorSensor::begin() {
  Wire.begin(13, 14, 100000);
  Wire.setTimeOut(50);
  if (!command(0x3517)) { value.errors++; return; }
  // A one-tick task delay can resume in less than the required 240 us.
  delayMicroseconds(500);
  command(0x805d);
  delay(2);
  uint8_t bytes[3];
  if (command(0xefc8) && receive(bytes, 3) && crc(bytes) == bytes[2])
    value.sensorId = uint16_t(bytes[0] << 8) | bytes[1];
  command(0xb098);
  update();
}

void IndoorSensor::update() {
  if (value.samples && uint32_t(millis() - lastAttemptMs) < 10000) return;
  if (!value.samples && lastAttemptMs && uint32_t(millis() - lastAttemptMs) < 10000) return;
  lastAttemptMs = millis();
  uint8_t bytes[6];
  bool ok = command(0x3517);
  delayMicroseconds(500);
  if (ok) ok = command(0x7866); // Temperature first, normal mode, no clock stretching.
  delay(20);
  if (ok) ok = receive(bytes, 6);
  command(0xb098); // Sleep even on errors to reduce sensor self-heating.
  if (!ok || crc(bytes) != bytes[2] || crc(bytes + 3) != bytes[5]) {
    value.errors++;
    value.valid = value.valid && value.samples && uint32_t(millis() - value.lastGoodMs) <= 30000;
    return;
  }
  uint16_t temperatureRaw = uint16_t(bytes[0] << 8) | bytes[1];
  uint16_t humidityRaw = uint16_t(bytes[3] << 8) | bytes[4];
  const float rawTemperatureC = -45.0f + 175.0f * temperatureRaw / 65536.0f;
  const float rawHumidityPct = 100.0f * humidityRaw / 65536.0f;
  const IndoorCompensationResult corrected = indoorCompensate(
      rawTemperatureC, rawHumidityPct, temperatureOffsetC);
  if (!corrected.valid) {
    value.errors++;
    value.valid = value.valid && value.samples && uint32_t(millis() - value.lastGoodMs) <= 30000;
    return;
  }
  value.rawTemperatureC = rawTemperatureC;
  value.rawHumidityPct = rawHumidityPct;
  value.temperatureC = corrected.temperatureC;
  value.humidityPct = corrected.humidityPct;
  value.humidityClamped = corrected.humidityClamped;
  recordRanges();
  value.samples++;
  value.lastGoodMs = millis();
  value.valid = true;
}
