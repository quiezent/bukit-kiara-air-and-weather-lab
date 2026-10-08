#pragma once
#include <Arduino.h>

struct WeatherTransfer {
  String body;
  String endpoint;
  String serverHost;
  String error;
  String discovery;
  uint32_t attempts = 0;
  uint32_t discoveries = 0;
  uint32_t receivedMs = 0;
  int httpCode = 0;
  bool pending = false;
};

class WeatherClient {
 public:
  void begin();
  void refresh(bool rediscover = false);
  WeatherTransfer take();
 private:
  static void task(void *context);
};
