#pragma once
#include <Arduino.h>
#include <ArduinoJson.h>
#include <U8g2lib.h>
#include "IndoorSensor.h"
#include "HistoryGraphs.h"
#include "ForecastSemantics.h"

constexpr uint8_t kDashboardPageCount = 3;

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
  bool readoutActive = false;
  uint32_t nowMs = 0;
  uint32_t uptimeS = 0;
  String address;  // The device's own address while Wi-Fi is up, for opening the web page.
};

// Full-frame push time, measured around sendBuffer() on every draw.
struct DashboardDisplayTiming {
  uint32_t sendBufferUsLast = 0;
  uint32_t sendBufferUsMax = 0;
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
DashboardDisplayTiming dashboardDisplayTiming();
