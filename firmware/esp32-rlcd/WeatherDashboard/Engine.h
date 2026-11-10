#pragma once
// The device's own "server": collects the TTDI sensor, the neighbour stations and
// Open-Meteo over HTTPS, keeps history on flash, and builds the same payload the
// pages used to read from the laptop's /api/rlcd/v1 (schema 1).
#include <Arduino.h>
#include <ArduinoJson.h>

struct EngineStatus {
  bool storageReady = false;
  uint32_t readings = 0, polls = 0;
  uint32_t ttdiOk = 0, ttdiFail = 0, neighbourOk = 0, neighbourFail = 0, weatherOk = 0, weatherFail = 0;
  int lastHttpCode = 0;
  uint32_t lastReadingEpoch = 0, lastWeatherEpoch = 0, lastIssueEpoch = 0;
  uint32_t taskStackFree = 0;
  size_t storageUsed = 0, storageTotal = 0;
  uint32_t modelCutoff = 0, modelFitted = 0, fitStage = 0, lastFitMs = 0;
  uint32_t arrivalCutoff = 0, sessionCutoff = 0, arrivalFitMs = 0, sessionFitMs = 0, observeMs = 0;
  uint32_t storageFormats = 0;
  int lastTlsError = 0;
  uint32_t lastHttpsOkMs = 0;
};

class Engine {
 public:
  // Mounts storage and restores history. Call once from setup().
  bool begin();
  // Starts the collection task. It waits for Wi-Fi and a valid clock by itself.
  void start();
  // Moves the newest payload into doc; false when nothing new is ready.
  bool takePayload(JsonDocument &doc);
  // Asks for an immediate collection pass (KEY or POST /refresh).
  void refresh();
  EngineStatus status() const;
  // True when no feed has answered for limitMs and no nightly fit is running.
  bool collectionStalled(uint32_t limitMs) const;
  // The inputs and outputs of the newest forecast, for checking against the laptop.
  String debugJson() const;
  // The web page's charts: 6, 24 or 168 hours, thinned to about 480 rows.
  String historyJson(int hours) const;
};
