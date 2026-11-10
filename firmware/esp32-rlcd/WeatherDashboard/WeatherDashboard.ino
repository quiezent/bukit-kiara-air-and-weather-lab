#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <Preferences.h>
#include <time.h>
#include <ArduinoJson.h>
#include <LittleFS.h>
#include "ST7305_U8g2.h"
#include "Dashboard.h"
#include "IndoorSensor.h"
#include "ButtonInput.h"
#include "Engine.h"
#include "PageReadout.h"
#include "ReadoutPlayer.h"
#include "WebPage.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cmath>

// The build script sets these for every C++ file, the WebServer library included.
static_assert(WEBSERVER_MAX_LINE_WAIT == 1000 && WEBSERVER_MAX_HEADER_WAIT == 2000,
              "build with -DWEBSERVER_MAX_LINE_WAIT=1000 -DWEBSERVER_MAX_HEADER_WAIT=2000 for every C++ file");

static constexpr const char *kFirmwareVersion = "weather-dashboard-40-public.1";

static ST7305_U8g2 lcd(11, 12, 5, 40, 41);
static U8G2 *gfx = nullptr;
static Preferences settings;
static Preferences indoorSettings;
static bool indoorSettingsReady = false;
static WebServer server(80);
static String ssid, password, serialLine;
static bool pendingSave = false;
static bool wasConnected = false;
static bool serverStarted = false;
static bool timedOut = false;
static uint32_t connectionStarted = 0;
static uint32_t lastDraw = 0;
static JsonDocument weatherDocument;
static IndoorSensor indoorSensor;
static Engine engine;
static ReadoutPlayer readoutPlayer;
static SpeechPlaylist readoutPlaylist;
static String lastReadoutText;
static uint32_t readoutRequests = 0;
static uint32_t weatherReceivedMs = 0;
static uint32_t apiLastSuccessMs = 0;
static uint8_t currentPage = 0;
static float batteryVoltage = 0;
static uint32_t frameCount = 0;
static uint32_t lastBatteryRead = 0;
static bool forceDraw = true;
static ButtonTaps buttonTaps;
static esp_timer_handle_t buttonTimer = nullptr;

// BAT_ADC is GPIO4 (ADC1 channel 3), with a 200k/100k divider.
static constexpr uint8_t BATTERY_ADC_PIN = 4;
static constexpr uint8_t BATTERY_SAMPLE_COUNT = 32;

static uint32_t readBatteryAdcMillivolts() {
  analogReadMilliVolts(BATTERY_ADC_PIN);  // Discard the first conversion.
  uint32_t sum = 0;
  for (uint8_t i = 0; i < BATTERY_SAMPLE_COUNT; i++) {
    sum += analogReadMilliVolts(BATTERY_ADC_PIN);
    delayMicroseconds(100);
  }
  return (sum + BATTERY_SAMPLE_COUNT / 2) / BATTERY_SAMPLE_COUNT;
}

static bool indoorCorrectionSaved() {
  return indoorSettingsReady && indoorSettings.isKey("temp_offset")
      && indoorSettings.getFloat("temp_offset", NAN) == indoorSensor.temperatureOffset();
}

static void handleIndoorCorrection() {
  if (!server.hasArg("temperature_offset_c")) {
    server.send(400, "application/json", "{\"error\":\"temperature_offset_c_required\"}");
    return;
  }
  const String input = server.arg("temperature_offset_c");
  char *end = nullptr;
  const float offset = input.length() && input.length() <= 24 ? strtof(input.c_str(), &end) : NAN;
  if (!end || end == input.c_str() || end != input.c_str() + input.length() || !indoorOffsetValid(offset)) {
    server.send(400, "application/json", "{\"error\":\"offset_must_be_finite_between_minus_10_and_10\"}");
    return;
  }
  if (!indoorSettingsReady
      || ((!indoorCorrectionSaved() || offset != indoorSensor.temperatureOffset())
          && indoorSettings.putFloat("temp_offset", offset) != sizeof(float))) {
    server.send(500, "application/json", "{\"error\":\"correction_save_failed\"}");
    return;
  }
  indoorSensor.setTemperatureOffset(offset);
  forceDraw = true;
  server.send(200, "application/json", "{\"saved\":true,\"temperature_offset_c\":" + String(indoorSensor.temperatureOffset(), 2) + "}");
}

static DashboardContext dashboardContext() {
  DashboardContext context;
  context.weather = &weatherDocument;
  context.indoor = &indoorSensor.reading();
  context.weatherReceivedMs = weatherReceivedMs;
  context.apiLastSuccessMs = apiLastSuccessMs;
  context.apiReachable = apiLastSuccessMs != 0;
  context.wifiConnected = WiFi.status() == WL_CONNECTED;
  context.wifiRssi = context.wifiConnected ? WiFi.RSSI() : 0;
  context.batteryV = batteryVoltage;
  context.page = currentPage;
  context.readoutActive = readoutPlayer.busy();
  context.nowMs = millis();
  context.uptimeS = uint32_t(esp_timer_get_time() / 1000000ULL);
  if (context.wifiConnected) context.address = WiFi.localIP().toString();
  return context;
}

static void drawScreen() {
  drawDashboard(*gfx, dashboardContext());
  frameCount++;
  forceDraw = false;
  lastDraw = millis();
}

// CPU load per core from the idle tasks' run time. FreeRTOS counts it in 32-bit
// microseconds (it wraps every 71 minutes), so loop() samples it every minute.
struct CpuLoad {
  uint32_t lastClock = 0, lastIdle[2] = {0, 0}, lastSampleMs = 0;
  uint64_t span = 0, idle[2] = {0, 0};
  float minute[2] = {NAN, NAN}, peak[2] = {0, 0};
};
static CpuLoad cpuLoad;

static void sampleCpu() {
  const uint32_t clock = uint32_t(esp_timer_get_time());
  uint32_t idle[2];
  for (int core = 0; core < 2; ++core) {
    TaskStatus_t info;
    vTaskGetInfo(xTaskGetIdleTaskHandleForCore(core), &info, pdFALSE, eRunning);
    idle[core] = info.ulRunTimeCounter;
  }
  if (cpuLoad.lastClock) {
    const uint32_t span = clock - cpuLoad.lastClock;
    if (span) {
      cpuLoad.span += span;
      for (int core = 0; core < 2; ++core) {
        const uint32_t free = std::min(uint32_t(idle[core] - cpuLoad.lastIdle[core]), span);
        cpuLoad.idle[core] += free;
        cpuLoad.minute[core] = 100.0f * (1.0f - float(free) / float(span));
        cpuLoad.peak[core] = std::max(cpuLoad.peak[core], cpuLoad.minute[core]);
      }
    }
  }
  cpuLoad.lastClock = clock;
  cpuLoad.lastIdle[0] = idle[0];
  cpuLoad.lastIdle[1] = idle[1];
}

static float tenths(float value) { return std::round(value * 10.0f) / 10.0f; }

static String statusJson() {
  const EngineStatus s = engine.status();
  JsonDocument doc;
  doc["firmware"] = kFirmwareVersion;
  doc["uptime_s"] = uint32_t(esp_timer_get_time() / 1000000ULL);  // millis() wraps after 49.7 days.
  doc["page"] = currentPage;
  doc["frames"] = frameCount;
  doc["clock"] = uint32_t(time(nullptr));
  doc["wifi"]["connected"] = WiFi.status() == WL_CONNECTED;
  doc["wifi"]["rssi"] = WiFi.RSSI();
  doc["wifi"]["ip"] = WiFi.localIP().toString();
  doc["wifi"]["power_save"] = int(WiFi.getSleep());  // 0 none, 1 minimum modem sleep, 2 maximum.
  JsonObject cpu = doc["cpu"].to<JsonObject>();
  cpu["mhz"] = getCpuFrequencyMhz();
  for (int core = 0; core < 2; ++core) {
    if (std::isfinite(cpuLoad.minute[core])) cpu["busy_last_minute_pct"].add(tenths(cpuLoad.minute[core]));
    else cpu["busy_last_minute_pct"].add(nullptr);
    cpu["busy_peak_minute_pct"].add(tenths(cpuLoad.peak[core]));
    if (cpuLoad.span) cpu["busy_since_boot_pct"].add(tenths(100.0f * (1.0f - float(double(cpuLoad.idle[core]) / double(cpuLoad.span)))));
    else cpu["busy_since_boot_pct"].add(nullptr);
  }
  cpu["measured_s"] = uint32_t(cpuLoad.span / 1000000ULL);
  doc["heap"]["internal_free"] = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  doc["heap"]["internal_min"] = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
  doc["heap"]["psram_free"] = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  doc["battery_v"] = batteryVoltage;
  JsonObject e = doc["engine"].to<JsonObject>();
  e["storage_ready"] = s.storageReady;
  e["storage_used"] = s.storageUsed;
  e["storage_total"] = s.storageTotal;
  e["readings"] = s.readings;
  e["polls"] = s.polls;
  e["ttdi_ok"] = s.ttdiOk;
  e["ttdi_fail"] = s.ttdiFail;
  e["neighbour_ok"] = s.neighbourOk;
  e["neighbour_fail"] = s.neighbourFail;
  e["weather_ok"] = s.weatherOk;
  e["weather_fail"] = s.weatherFail;
  e["last_http"] = s.lastHttpCode;
  e["last_reading"] = s.lastReadingEpoch;
  e["last_weather"] = s.lastWeatherEpoch;
  e["last_issue"] = s.lastIssueEpoch;
  e["task_stack_free"] = s.taskStackFree;
  e["model_cutoff"] = s.modelCutoff;
  e["model_fitted"] = s.modelFitted;
  e["fit_stage"] = s.fitStage;
  e["last_fit_ms"] = s.lastFitMs;
  e["arrival_cutoff"] = s.arrivalCutoff;
  e["arrival_fit_ms"] = s.arrivalFitMs;
  e["session_cutoff"] = s.sessionCutoff;
  e["session_fit_ms"] = s.sessionFitMs;
  e["observe_ms"] = s.observeMs;
  e["storage_formats"] = s.storageFormats;
  e["last_tls_error"] = s.lastTlsError;
  JsonObject r = doc["readout"].to<JsonObject>();
  r["busy"] = readoutPlayer.busy();
  r["requests"] = readoutRequests;
  r["completed"] = readoutPlayer.completed();
  r["stopped"] = readoutPlayer.stopped();
  r["planned_seconds"] = readoutPlayer.plannedSamples() / 16000.0f;
  r["stack_free_min"] = readoutPlayer.stackFreeMin();
  r["error"] = readoutPlayer.lastError();
  String out;
  serializeJson(doc, out);
  return out;
}

static void serveScreen() {
  constexpr size_t kImageBytes = 15000;
  static uint8_t *image = nullptr;
  if (!image) image = static_cast<uint8_t *>(heap_caps_malloc(kImageBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!image) { server.send(503, "application/json", "{\"error\":\"screen_buffer_unavailable\"}"); return; }
  memset(image, 0, kImageBytes);
  const uint8_t *buffer = gfx->getBufferPtr();
  // U8g2 R1 maps user (x,y) to base (299-y,x), vertical-top-LSB, 304-byte tile-row stride.
  for (int y = 0; y < 300; y++)
    for (int x = 0; x < 400; x++)
      if (buffer[(x / 8) * 304 + 299 - y] & (1 << (x % 8)))
        image[y * 50 + x / 8] |= 0x80 >> (x % 8);
  const char header[] = "P4\n400 300\n";
  server.setContentLength(sizeof(header) - 1 + kImageBytes);
  server.send(200, "image/x-portable-bitmap", "");
  server.client().write(reinterpret_cast<const uint8_t *>(header), sizeof(header) - 1);
  server.client().write(image, kImageBytes);
}

// KEY: read the displayed page aloud; a second press stops it.
static bool requestReadout() {
  if (readoutPlayer.busy() || weatherDocument.isNull()) return false;
  ++readoutRequests;
  if (!buildCurrentPageReadout(dashboardContext(), readoutPlaylist, lastReadoutText)) return false;
  const bool started = readoutPlayer.start(readoutPlaylist);
  forceDraw = true;
  Serial.println("{\"event\":\"readout\",\"started\":" + String(started ? "true" : "false")
      + ",\"page\":" + String(currentPage) + ",\"error\":\"" + String(readoutPlayer.lastError()) + "\"}");
  return started;
}

static void handleKey() {
  if (!readoutPlayer.busy()) { requestReadout(); return; }
  if (readoutPlayer.stop()) forceDraw = true;
}

static void selectPage(uint8_t page) {
  if (page >= kDashboardPageCount) return;
  currentPage = page;
  forceDraw = true;
}

static void sampleButtons(void *) {
  buttonTaps.sample(digitalRead(18) == HIGH, digitalRead(0) == HIGH, millis());
}

static void readButtons() {
  if (!buttonTimer) sampleButtons(nullptr);
  ButtonTap tap;
  while (buttonTaps.take(tap)) {
    if (tap == ButtonTap::Boot) selectPage((currentPage + 1) % kDashboardPageCount);
    else handleKey();
  }
}

// History seeding from an existing archive. USB "SEED" opens a ten-minute window.
static uint32_t seedUntilMs = 0;
static File seedFile;
static size_t seedBytes = 0;
static bool seedOpened = false;
static String seedPath;
static const char *const kSeedTemp = "/w/seed.tmp";

static constexpr uint32_t kCollectionStallMs = 30UL * 60UL * 1000UL;

static bool seedPathValid(const String &path) {
  if (!(path.startsWith("/r/") || path.startsWith("/p/") || path.startsWith("/x/") || path.startsWith("/g/") || path.startsWith("/s/"))) return false;
  return path.endsWith(".bin") && path.indexOf("..") < 0 && path.length() < 40;
}

// Cleared once expired, so the 32-bit millis wrap cannot reopen the window weeks later.
static bool seedAllowed() {
  if (seedUntilMs && int32_t(seedUntilMs - millis()) <= 0) seedUntilMs = 0;
  return seedUntilMs != 0;
}

static void handleSeedBody() {
  HTTPRaw &raw = server.raw();
  if (raw.status == RAW_START) {
    // WebServer parses URL arguments only after the raw body, so the path is a header.
    const String path = server.header("X-Seed-Path");
    seedBytes = 0;
    seedOpened = false;
    if (seedAllowed() && seedPathValid(path)) {
      seedPath = path;
      seedFile = LittleFS.open(kSeedTemp, FILE_WRITE, true);
      seedOpened = bool(seedFile);
    }
  } else if (raw.status == RAW_WRITE) {
    if (seedFile) seedBytes += seedFile.write(raw.buf, raw.currentSize);
  } else if (seedFile) {
    seedFile.close();
    // Only a whole upload replaces the day file, so a cut connection leaves no torn rows.
    if (raw.status == RAW_END && seedBytes == size_t(server.clientContentLength())) {
      LittleFS.remove(seedPath);
      seedOpened = LittleFS.rename(kSeedTemp, seedPath);
    } else {
      LittleFS.remove(kSeedTemp);
      seedOpened = false;
    }
  }
}

static void beginWifi(const String &newSsid, const String &newPassword, bool saveAfterConnect) {
  WiFi.disconnect(false, false);
  ssid = newSsid;
  password = newPassword;
  pendingSave = saveAfterConnect;
  timedOut = false;
  wasConnected = false;
  connectionStarted = millis();
  WiFi.begin(ssid.c_str(), password.c_str());
  Serial.println("{\"event\":\"wifi_connecting\"}");
}

static int hexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static bool decodeHex(const String &hex, String &out) {
  out = "";
  if (hex == "-") return true;
  if (hex.length() % 2) return false;
  for (size_t i = 0; i < hex.length(); i += 2) {
    int high = hexDigit(hex[i]), low = hexDigit(hex[i + 1]);
    if (high < 0 || low < 0 || (high == 0 && low == 0)) return false;
    out += char((high << 4) | low);
  }
  return true;
}

static void handleCommand(const String &line) {
  if (line == "STATUS") { Serial.println(statusJson()); return; }
  if (line == "SEED") { seedUntilMs = millis() + 600000; Serial.println("{\"event\":\"seed_open\"}"); return; }
  if (line == "READINFO") { handleKey(); return; }
  if (line == "REFRESH") { engine.refresh(); Serial.println("{\"event\":\"refresh\"}"); return; }
  if (line.length() == 6 && line.startsWith("VIEW ") && line[5] >= '0' && line[5] < char('0' + kDashboardPageCount)) {
    selectPage(line[5] - '0');
    Serial.println("{\"event\":\"view\"}");
    return;
  }
  if (line.startsWith("WIFI ")) {
    const int split = line.indexOf(' ', 5);
    String newSsid, newPassword;
    if (split < 0 || !decodeHex(line.substring(5, split), newSsid) || !decodeHex(line.substring(split + 1), newPassword) || !newSsid.length()) {
      Serial.println("{\"error\":\"wifi_arguments\"}");
      return;
    }
    beginWifi(newSsid, newPassword, true);
    return;
  }
  Serial.println("{\"error\":\"unknown_command\"}");
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.begin(115200);
  Serial.setTxTimeoutMs(0);
  delay(500);
  pinMode(BATTERY_ADC_PIN, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(BATTERY_ADC_PIN, ADC_11db);
  batteryVoltage = readBatteryAdcMillivolts() * 3.0f / 1000.0f;
  pinMode(18, INPUT_PULLUP);
  pinMode(0, INPUT_PULLUP);
  setenv("TZ", "MYT-8", 1);
  tzset();
  indoorSettingsReady = indoorSettings.begin("weather-indoor", false);
  if (indoorSettingsReady) {
    const float savedOffset = indoorSettings.getFloat("temp_offset", kIndoorDefaultTemperatureOffsetC);
    if (indoorOffsetValid(savedOffset)) indoorSensor.setTemperatureOffset(savedOffset);
  }
  indoorSensor.begin();
  lcd.begin(0, U8G2_R1);
  gfx = lcd.getU8g2();
  settings.begin("weather-wifi", false);
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setHostname("waveshare-weather");
  WiFi.setAutoReconnect(true);
  server.on("/status", HTTP_GET, []() { server.send(200, "application/json", statusJson()); });
  server.on("/indoor-correction", HTTP_POST, handleIndoorCorrection);
  server.on("/", HTTP_GET, []() {
    server.sendHeader("Content-Encoding", "gzip");
    server.sendHeader("Cache-Control", "no-cache");
    server.send_P(200, "text/html; charset=utf-8", reinterpret_cast<const char *>(kWebPageGz), kWebPageGzSize);
  });
  server.on("/history", HTTP_GET, []() {
    const int hours = server.arg("hours").toInt();
    if (hours != 6 && hours != 24 && hours != 168) { server.send(400, "application/json", "{\"error\":\"hours_must_be_6_24_or_168\"}"); return; }
    server.send(200, "application/json", engine.historyJson(hours));
  });
  server.on("/weather", HTTP_GET, []() {
    if (weatherDocument.isNull()) { server.send(503, "application/json", "{\"error\":\"weather_unavailable\"}"); return; }
    weatherDocument["served_epoch"] = uint32_t(time(nullptr));
    String payload;
    serializeJson(weatherDocument, payload);
    server.send(200, "application/json", payload);
  });
  server.on("/screen.pbm", HTTP_GET, serveScreen);
  server.on("/read-info", HTTP_POST, []() {
    if (readoutPlayer.busy()) { server.send(409, "application/json", "{\"error\":\"readout_busy\"}"); return; }
    const bool started = requestReadout();
    server.send(started ? 202 : 503, "application/json", String("{\"started\":") + (started ? "true" : "false") + "}");
  });
  server.on("/debug/forecast", HTTP_GET, []() { server.send(200, "application/json", engine.debugJson()); });
  server.on("/view", HTTP_POST, []() {
    String page = server.arg("page");
    if (page.length() != 1 || page[0] < '0' || page[0] >= char('0' + kDashboardPageCount)) { server.send(400, "text/plain", "page must be 0, 1 or 2"); return; }
    selectPage(page[0] - '0');
    server.send(200, "application/json", "{\"page\":" + page + "}");
  });
  static const char *seedHeaders[] = {"X-Seed-Path"};
  server.collectHeaders(seedHeaders, 1);
  server.on("/seed", HTTP_POST, []() {
    if (!seedAllowed()) { server.send(403, "application/json", "{\"error\":\"seed_closed\"}"); return; }
    const bool complete = seedOpened && seedBytes == size_t(server.clientContentLength());
    server.send(complete ? 200 : 500, "application/json", "{\"opened\":" + String(seedOpened ? "true" : "false") + ",\"bytes\":" + String(seedBytes) + "}");
  }, handleSeedBody);
  server.on("/seed-done", HTTP_POST, []() {
    if (!seedAllowed()) { server.send(403, "application/json", "{\"error\":\"seed_closed\"}"); return; }
    server.send(200, "application/json", "{\"restarting\":true}");
    delay(200);
    ESP.restart();
  });
  server.on("/refresh", HTTP_POST, []() { engine.refresh(); server.send(202, "application/json", "{\"refresh\":\"requested\"}"); });
  drawScreen();
  const bool ready = engine.begin();
  Serial.println("{\"event\":\"ready\",\"firmware\":\"" + String(kFirmwareVersion) + "\",\"engine\":" + (ready ? "true" : "false") + "}");
  engine.start();
  String storedSsid = settings.getString("ssid", "");
  if (storedSsid.length()) beginWifi(storedSsid, settings.getString("password", ""), false);
  esp_timer_create_args_t buttonTimerArgs = {};
  buttonTimerArgs.callback = sampleButtons;
  buttonTimerArgs.name = "buttons";
  if (esp_timer_create(&buttonTimerArgs, &buttonTimer) != ESP_OK) buttonTimer = nullptr;
  else if (esp_timer_start_periodic(buttonTimer, 10000) != ESP_OK) { esp_timer_delete(buttonTimer); buttonTimer = nullptr; }
  drawScreen();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleCommand(serialLine); serialLine = ""; }
    else if (c != '\r') { if (serialLine.length() < 256) serialLine += c; else serialLine = ""; }
  }
  const bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !wasConnected) {
    if (pendingSave) {
      const size_t savedSsidBytes = settings.putString("ssid", ssid);
      const size_t savedPasswordBytes = settings.putString("password", password);
      if (savedSsidBytes == ssid.length() && savedPasswordBytes == password.length()) pendingSave = false;
      else Serial.println("{\"error\":\"wifi_save_failed\"}");
    }
    timedOut = false;
    configTime(8 * 3600, 0, "pool.ntp.org", "time.cloudflare.com");
    if (!serverStarted) {
      server.begin();
      serverStarted = true;
      if (MDNS.begin("ttdi-weather")) MDNS.addService("http", "tcp", 80);
    }
    Serial.println(statusJson());
    engine.refresh();
    forceDraw = true;
  }
  if (!connected && wasConnected) connectionStarted = millis();
  if (!connected && ssid.length() && !timedOut && uint32_t(millis() - connectionStarted) > 45000) {
    timedOut = true;
    Serial.println("{\"error\":\"wifi_timeout\"}");
  }
  // The core stops retrying after some failures, such as a router that rejects the
  // device while it is still booting after a power cut.
  if (!connected && ssid.length() && uint32_t(millis() - connectionStarted) >= 90000) {
    WiFi.disconnect(false, false);
    WiFi.begin(ssid.c_str(), password.c_str());
    connectionStarted = millis();
    timedOut = false;
    Serial.println("{\"event\":\"wifi_retry\"}");
  }
  // Associated but nothing answering, or the engine task stuck: every card is blank
  // and collection gaps are permanent, so restart rather than wait for someone to notice.
  if (connected && uint32_t(millis() - connectionStarted) > kCollectionStallMs && engine.collectionStalled(kCollectionStallMs)) {
    Serial.println("{\"event\":\"collection_stalled_restart\"}");
    delay(100);
    ESP.restart();
  }
  seedAllowed();
  wasConnected = connected;
  indoorSensor.update();
  if (engine.takePayload(weatherDocument)) {
    weatherReceivedMs = millis();
    apiLastSuccessMs = millis();
    forceDraw = true;
  }
  readButtons();
  const bool wasReading = readoutPlayer.busy();
  readoutPlayer.update();
  if (wasReading && !readoutPlayer.busy()) forceDraw = true;
  if (uint32_t(millis() - lastBatteryRead) >= 10000) {
    batteryVoltage = readBatteryAdcMillivolts() * 3.0f / 1000.0f;
    lastBatteryRead = millis();
  }
  if (!cpuLoad.lastClock || uint32_t(millis() - cpuLoad.lastSampleMs) >= 60000) {
    sampleCpu();
    cpuLoad.lastSampleMs = millis();
  }
  if (serverStarted) server.handleClient();
  if (forceDraw || uint32_t(millis() - lastDraw) >= 5000) drawScreen();
  delay(5);
}
