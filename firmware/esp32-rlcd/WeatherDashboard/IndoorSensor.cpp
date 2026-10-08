#include "IndoorSensor.h"
#include <Wire.h>

// Sensirion SHTC3 at 0x70. SDA=13/SCL=14 from Waveshare's schematic.
// Use the manufacturer's calibrated conversion without the vendor example's
// fixed -4 C adjustment, which has not been calibrated for this individual unit.
static constexpr uint8_t ADDRESS = 0x70;

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
    value.valid = value.samples && uint32_t(millis() - value.lastGoodMs) <= 30000;
    return;
  }
  uint16_t temperatureRaw = uint16_t(bytes[0] << 8) | bytes[1];
  uint16_t humidityRaw = uint16_t(bytes[3] << 8) | bytes[4];
  value.temperatureC = -45.0f + 175.0f * temperatureRaw / 65536.0f;
  value.humidityPct = 100.0f * humidityRaw / 65536.0f;
  if (!value.samples) {
    value.minimumC = value.maximumC = value.temperatureC;
    value.minimumHumidity = value.maximumHumidity = value.humidityPct;
  } else {
    value.minimumC = min(value.minimumC, value.temperatureC);
    value.maximumC = max(value.maximumC, value.temperatureC);
    value.minimumHumidity = min(value.minimumHumidity, value.humidityPct);
    value.maximumHumidity = max(value.maximumHumidity, value.humidityPct);
  }
  value.samples++;
  value.lastGoodMs = millis();
  value.valid = true;
}
