#include "WeatherClient.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <atomic>

namespace {
SemaphoreHandle_t transferMutex = nullptr;
WeatherTransfer latest;
std::atomic<bool> requested{true};
std::atomic<bool> discoveryRequested{false};
static constexpr uint32_t MAX_BODY_BYTES = 16384;
static constexpr uint32_t REFRESH_MS = 60000;
}

void WeatherClient::begin() {
  transferMutex = xSemaphoreCreateMutex();
  if (!transferMutex) return;
  xTaskCreate(task, "weather-api", 8192, nullptr, 1, nullptr);
}

void WeatherClient::refresh(bool rediscover) {
  if (rediscover) discoveryRequested.store(true);
  requested.store(true);
}

WeatherTransfer WeatherClient::take() {
  WeatherTransfer result;
  if (!transferMutex) { result.error = "Network task unavailable"; return result; }
  xSemaphoreTake(transferMutex, portMAX_DELAY);
  result = latest;
  latest.pending = false;
  latest.body = "";
  xSemaphoreGive(transferMutex);
  return result;
}

void WeatherClient::task(void *) {
  bool mdnsReady = false;
  bool wasConnected = false;
  IPAddress address;
  uint16_t port = 0;
  String host, path = "/api/rlcd/v1", discovery;
  uint32_t lastAttempt = 0, lastDiscovery = 0, attempts = 0, discoveries = 0;
  uint32_t failures = 0;
  for (;;) {
    bool connected = WiFi.status() == WL_CONNECTED;
    if (!connected) {
      wasConnected = false;
      vTaskDelay(pdMS_TO_TICKS(250));
      continue;
    }
    if (!wasConnected) {
      wasConnected = true;
      address = IPAddress();
      requested.store(true);
    }
    if (!mdnsReady) {
      mdnsReady = MDNS.begin("waveshare-weather");
      if (mdnsReady) {
        MDNS.addService("http", "tcp", 80);
        MDNS.addServiceTxt("http", "tcp", "device", "waveshare-rlcd");
      }
    }
    bool nowRequested = requested.exchange(false);
    if (!nowRequested && lastAttempt && uint32_t(millis() - lastAttempt) < REFRESH_MS) {
      vTaskDelay(pdMS_TO_TICKS(250));
      continue;
    }
    lastAttempt = millis();
    attempts++;
    bool rediscover = discoveryRequested.exchange(false);
    if (!uint32_t(address) || failures >= 2 || rediscover || uint32_t(millis() - lastDiscovery) > 300000) {
      lastDiscovery = millis();
      discoveries++;
      address = IPAddress();
      if (mdnsReady) {
        int count = MDNS.queryService("ttdi-weather", "tcp");
        for (int i = 0; i < count; i++) {
          IPAddress candidate = MDNS.address(i);
          String apiPath = MDNS.txt(i, "api_path");
          if (uint32_t(candidate) && MDNS.port(i) && (apiPath == "" || apiPath.startsWith("/api/")) && MDNS.txt(i, "schema_version") == "1") {
            address = candidate;
            port = MDNS.port(i);
            host = MDNS.hostname(i);
            path = apiPath.length() ? apiPath : "/api/rlcd/v1";
            discovery = "dns-sd";
            break;
          }
        }
        if (!uint32_t(address) && host.length()) {
          // Re-resolve the previously advertised hostname if a PTR browse fails.
          // No DHCP address or fabricated server hostname is embedded here.
          String query = host;
          if (query.endsWith(".")) query.remove(query.length() - 1);
          if (query.endsWith(".local")) query.remove(query.length() - 6);
          address = MDNS.queryHost(query, 2000);
          discovery = "mdns-host";
        }
      }
    }
    WeatherTransfer result;
    result.attempts = attempts;
    result.discoveries = discoveries;
    result.serverHost = host;
    result.discovery = discovery;
    result.receivedMs = millis();
    result.pending = true;
    if (!uint32_t(address)) {
      result.error = "Finding weather server";
      failures++;
    } else {
      result.endpoint = "http://" + address.toString() + ":" + String(port) + path;
      NetworkClient client;
      HTTPClient http;
      http.setConnectTimeout(3000);
      http.setTimeout(3000);
      http.useHTTP10(true);
      http.setReuse(false);
      http.setUserAgent("Waveshare-RLCD/1");
      if (http.begin(client, result.endpoint)) {
        http.addHeader("Accept", "application/json");
        result.httpCode = http.GET();
        int length = http.getSize();
        if (result.httpCode == 200 && length > 0 && length <= int(MAX_BODY_BYTES)) {
          result.body = http.getString();
          if (result.body.length() == size_t(length)) failures = 0;
          else { result.body = ""; result.error = "Incomplete weather response"; failures++; }
        } else {
          result.error = result.httpCode == 200 ? "Weather response size invalid" : "Weather server unavailable";
          failures++;
        }
        http.end();
      } else { result.error = "Weather connection failed"; failures++; }
    }
    result.receivedMs = millis();
    xSemaphoreTake(transferMutex, portMAX_DELAY);
    latest = std::move(result);
    xSemaphoreGive(transferMutex);
    // Retry discovery/failed requests in 15 s, preserving a responsive UI.
    if (failures) lastAttempt = millis() - REFRESH_MS + 15000;
  }
}
