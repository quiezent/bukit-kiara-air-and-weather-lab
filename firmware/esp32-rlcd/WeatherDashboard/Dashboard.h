#pragma once
#include <Arduino.h>
#include <ArduinoJson.h>
#include <U8g2lib.h>
#include "IndoorSensor.h"
#include "HistoryGraphs.h"
#include "ForecastSemantics.h"

struct DashboardContext {
  const JsonDocument *weather = nullptr;
  const IndoorReading *indoor = nullptr;
  uint32_t weatherReceivedMs = 0;
  uint32_t apiLastSuccessMs = 0;
  bool apiReachable = false;
  bool wifiConnected = false;
  int wifiRssi = 0;
  float batteryV = 0;
  uint8_t page = 0;
  const char *navigationHint = "Read Info";
  const char *voiceHint = nullptr;
};

uint32_t dashboardNow(const DashboardContext &context);
bool dashboardCurrentAvailable(const DashboardContext &context);
bool dashboardForecastAvailable(const DashboardContext &context);
bool dashboardRideWindowAvailable(const DashboardContext &context);
RidePmRange dashboardRidePmRange(const DashboardContext &context);
HistoryPmSummary dashboardHistoryPmSummary(const DashboardContext &context);
String dashboardCurrentStatus(const DashboardContext &context);
String dashboardArrivalStatus(const DashboardContext &context);
String dashboardFirst20Status(const DashboardContext &context);
void drawDashboard(U8G2 &gfx, const DashboardContext &context);
