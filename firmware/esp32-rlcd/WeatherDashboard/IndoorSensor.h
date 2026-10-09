#pragma once
#include <Arduino.h>
#include "IndoorCompensation.h"

struct IndoorReading {
  bool valid = false;
  uint16_t sensorId = 0;
  float temperatureC = NAN;
  float humidityPct = NAN;
  float rawTemperatureC = NAN;
  float rawHumidityPct = NAN;
  bool humidityClamped = false;
  float minimumC = NAN;
  float maximumC = NAN;
  float minimumHumidity = NAN;
  float maximumHumidity = NAN;
  uint32_t lastGoodMs = 0;
  uint32_t samples = 0;
  uint32_t errors = 0;
};

class IndoorSensor {
 public:
  void begin();
  void update();
  bool setTemperatureOffset(float offset);
  float temperatureOffset() const { return temperatureOffsetC; }
  const IndoorReading &reading() const { return value; }
 private:
  IndoorReading value;
  float temperatureOffsetC = kIndoorDefaultTemperatureOffsetC;
  bool rangesInitialized = false;
  uint32_t lastAttemptMs = 0;
  void resetRanges();
  void recordRanges();
  bool command(uint16_t code);
  bool receive(uint8_t *bytes, uint8_t count);
  static uint8_t crc(const uint8_t *bytes);
};
