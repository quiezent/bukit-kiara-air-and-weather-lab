#pragma once
#include <Arduino.h>

struct IndoorReading {
  bool valid = false;
  uint16_t sensorId = 0;
  float temperatureC = NAN;
  float humidityPct = NAN;
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
  const IndoorReading &reading() const { return value; }
 private:
  IndoorReading value;
  uint32_t lastAttemptMs = 0;
  bool command(uint16_t code);
  bool receive(uint8_t *bytes, uint8_t count);
  static uint8_t crc(const uint8_t *bytes);
};
