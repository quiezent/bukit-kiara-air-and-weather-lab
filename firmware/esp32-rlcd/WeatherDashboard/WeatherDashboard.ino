#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <time.h>
#include <ArduinoJson.h>
#include "ST7305_U8g2.h"
#include "Dashboard.h"
#include "IndoorSensor.h"
#include "WeatherClient.h"
#include "BatteryEstimate.h"
#include "VoiceControl.h"
#include "ReadoutPlayer.h"
#include "ButtonInput.h"
#include "PageReadout.h"
#include "ArrivalChange.h"

// Waveshare's ST7305 driver, using its documented SPI pin assignments.
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
static WeatherClient weatherClient;
static VoiceControl voiceControl;
static ReadoutPlayer readoutPlayer;
static uint32_t readoutRequests = 0;
static String lastReadoutText;
static uint32_t lastReadoutObservedEpoch = 0;
static uint32_t lastReadoutForecastEpoch = 0;
static uint8_t lastReadoutPage = 0;
static uint32_t lastReadoutGeneratedEpoch = 0;
static WeatherTransfer weatherTransfer;
static uint32_t weatherReceivedMs = 0;
static uint32_t apiLastSuccessMs = 0;
static bool apiReachable = false;
static String weatherError = "Finding weather server";
static uint8_t currentPage = 0;
static float batteryVoltage = 0;
static uint32_t frameCount = 0;
static uint32_t lastWeatherPoll = 0;
static uint32_t lastBatteryRead = 0;
static bool forceDraw = true;
static int lastVoiceCommand = 0;
static int lastVoicePhrase = 0;
static float lastVoiceConfidence = 0;
static uint32_t lastVoiceMs = 0;
static uint32_t staleVoiceEvents = 0;
static bool voiceAcknowledgement = false;
static char voiceHint[28] = "";
static VoiceDiagnostic lastVoiceDiagnostic;
static bool hasVoiceDiagnostic = false;

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
  if (input.length() == 0 || input.length() > 24) {
    server.send(400, "application/json", "{\"error\":\"offset_must_be_finite_between_minus_10_and_10\"}");
    return;
  }
  char *end = nullptr;
  const float offset = strtof(input.c_str(), &end);
  if (end == input.c_str() || !end || end != input.c_str() + input.length()
      || !indoorOffsetValid(offset)) {
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
  server.send(200, "application/json", "{\"saved\":true,\"temperature_offset_c\":"
      + String(indoorSensor.temperatureOffset(), 2) + "}");
}

// All input paths select pages on the Arduino loop task.
static void selectPage(uint8_t page) {
  if (page >= kDashboardPageCount) return;
  currentPage = page;
  forceDraw = true;
}

static DashboardContext dashboardContext() {
  DashboardContext context;
  context.weather = &weatherDocument;
  context.indoor = &indoorSensor.reading();
  context.weatherReceivedMs = weatherReceivedMs;
  context.apiLastSuccessMs = apiLastSuccessMs;
  context.apiReachable = apiReachable && WiFi.status() == WL_CONNECTED;
  context.wifiConnected = WiFi.status() == WL_CONNECTED;
  context.wifiRssi = context.wifiConnected ? WiFi.RSSI() : 0;
  context.batteryV = batteryVoltage;
  context.page = currentPage;
  if (voiceControl.status().ready && voiceAcknowledgement) context.voiceHint = voiceHint;
  else if (readoutPlayer.busy()) context.voiceHint = "Voice: reading";
  return context;
}

// BAT_ADC is GPIO4 (ADC1 channel 3), with a 200k/100k divider.
// ETA6098 STAT drives only the CHG LED; no charging-status GPIO exists.
static constexpr uint8_t BATTERY_ADC_PIN = 4;
static constexpr uint8_t BATTERY_SAMPLE_COUNT = 32;

static uint32_t readBatteryAdcMillivolts() {
  analogReadMilliVolts(BATTERY_ADC_PIN); // Discard the first conversion.
  uint32_t sum = 0;
  for (uint8_t i = 0; i < BATTERY_SAMPLE_COUNT; i++) {
    sum += analogReadMilliVolts(BATTERY_ADC_PIN);
    delayMicroseconds(100);
  }
  return (sum + BATTERY_SAMPLE_COUNT / 2) / BATTERY_SAMPLE_COUNT;
}

static String jsonString(const String &value) {
  String out = "\"";
  for (size_t i = 0; i < value.length(); i++) {
    const uint8_t c = value[i];
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c < 0x20) { char escaped[7]; snprintf(escaped, sizeof(escaped), "\\u%04x", c); out += escaped; }
    else out += char(c);
  }
  return out + "\"";
}

static String voiceDiagnosticJson() {
  if (!hasVoiceDiagnostic) return "null";
  const VoiceDiagnostic &trace = lastVoiceDiagnostic;
  String out = "{\"sequence\":" + String(trace.sequence);
  out += ",\"session_id\":" + String(trace.sessionId);
  out += ",\"reason\":" + jsonString(voiceDiagnosticReasonName(trace.reason));
  out += ",\"ended_ms\":" + String(trace.endedMs);
  out += ",\"duration_ms\":" + String(trace.sessionDurationMs);
  out += ",\"model_frames\":" + String(trace.modelFrames);
  out += ",\"speech_samples\":" + String(trace.speechSamples);
  out += ",\"vad_cache_samples\":" + String(trace.vadCacheSamples);
  out += ",\"model_state\":" + String(trace.modelState);
  out += ",\"detect_state\":" + String(trace.detectState);
  out += ",\"afe_trigger_channel\":" + String(trace.afeTriggerChannel);
  out += ",\"afe_raw_channels\":" + String(trace.afeRawChannels);
  out += ",\"afe_channel_changes\":" + String(trace.afeChannelChanges);
  out += ",\"vad_energy_min_dbfs\":" + (std::isfinite(trace.minimumSpeechDbfs) ? String(trace.minimumSpeechDbfs, 2) : String("null"));
  out += ",\"vad_energy_max_dbfs\":" + (std::isfinite(trace.maximumSpeechDbfs) ? String(trace.maximumSpeechDbfs, 2) : String("null"));
  out += ",\"graph_phonemes\":" + jsonString(trace.string);
  out += ",\"raw_phonemes\":" + jsonString(trace.rawString);
  out += ",\"graph_phonemes_truncated\":" + String(trace.stringTruncated ? "true" : "false");
  out += ",\"raw_phonemes_truncated\":" + String(trace.rawStringTruncated ? "true" : "false");
  out += ",\"api_result_count\":" + String(trace.apiResultCount);
  out += ",\"candidates_valid\":" + String(trace.candidatesValid ? "true" : "false");
  out += ",\"candidates\":[";
  for (uint8_t i = 0; i < trace.candidateCount; ++i) {
    if (i) out += ',';
    out += "{\"command_id\":" + String(trace.commandIds[i]);
    out += ",\"phrase_id\":" + String(trace.phraseIds[i]);
    out += ",\"probability\":" + (std::isfinite(trace.probabilities[i]) ? String(trace.probabilities[i], 4) : String("null")) + "}";
  }
  return out + "]}";
}

static String statusJson() {
  const bool connected = WiFi.status() == WL_CONNECTED;
  const uint32_t batteryAdcMv = readBatteryAdcMillivolts();
  batteryVoltage = batteryAdcMv * 3.0f / 1000.0f;
  String out = "{\"firmware\":\"weather-dashboard-27-public.1\",\"connected\":";
  out += connected ? "true" : "false";
  out += ",\"ssid\":" + jsonString(ssid);
  out += ",\"ip\":" + jsonString(connected ? WiFi.localIP().toString() : "");
  out += ",\"rssi\":" + String(connected ? WiFi.RSSI() : 0);
  out += ",\"wifi_status\":" + String(int(WiFi.status()));
  out += ",\"clock_synced\":";
  out += time(nullptr) > 1700000000 ? "true" : "false";
  out += ",\"credentials_saved\":";
  out += (!pendingSave && settings.getString("ssid", "") == ssid && ssid.length()) ? "true" : "false";
  out += ",\"flash_bytes\":" + String(ESP.getFlashChipSize());
  out += ",\"psram_bytes\":" + String(ESP.getPsramSize());
  out += ",\"heap_free_bytes\":" + String(ESP.getFreeHeap());
  out += ",\"heap_min_free_bytes\":" + String(ESP.getMinFreeHeap());
  out += ",\"display_width\":400,\"display_height\":300";
  out += ",\"battery_voltage_v\":" + String(batteryAdcMv * 3.0f / 1000.0f, 3);
  int batteryPercent = batteryPercentEstimate(batteryVoltage);
  out += ",\"battery_percent_estimate\":" + (batteryPercent >= 0 ? String(batteryPercent) : String("null"));
  out += ",\"battery_percent_is_estimate\":true,\"battery_percent_method\":" + jsonString(kBatteryPercentMethod);
  out += ",\"battery_adc_millivolts\":" + String(batteryAdcMv);
  out += ",\"battery_adc_pin\":4,\"battery_sample_count\":32";
  out += ",\"battery_charging\":null,\"battery_charging_status_supported\":false";
  const IndoorReading &indoor = indoorSensor.reading();
  out += ",\"indoor\":{\"available\":" + String(indoor.valid ? "true" : "false");
  out += ",\"temperature_c\":" + (indoor.valid ? String(indoor.temperatureC, 2) : String("null"));
  out += ",\"humidity_pct\":" + (indoor.valid ? String(indoor.humidityPct, 2) : String("null"));
  out += ",\"raw_temperature_c\":" + (indoor.valid ? String(indoor.rawTemperatureC, 2) : String("null"));
  out += ",\"raw_humidity_pct\":" + (indoor.valid ? String(indoor.rawHumidityPct, 2) : String("null"));
  out += ",\"temperature_offset_c\":" + String(indoorSensor.temperatureOffset(), 2);
  out += ",\"humidity_temperature_compensated\":" + String(indoorSensor.temperatureOffset() != 0 ? "true" : "false");
  out += ",\"humidity_clamped\":" + String(indoor.humidityClamped ? "true" : "false");
  out += ",\"correction_method\":\"board-heat-magnus-v1\",\"correction_is_estimate\":true";
  out += ",\"correction_saved\":" + String(indoorCorrectionSaved() ? "true" : "false");
  out += ",\"minimum_temperature_c\":" + (indoor.valid ? String(indoor.minimumC, 2) : String("null"));
  out += ",\"maximum_temperature_c\":" + (indoor.valid ? String(indoor.maximumC, 2) : String("null"));
  out += ",\"minimum_humidity_pct\":" + (indoor.valid ? String(indoor.minimumHumidity, 2) : String("null"));
  out += ",\"maximum_humidity_pct\":" + (indoor.valid ? String(indoor.maximumHumidity, 2) : String("null"));
  out += ",\"age_s\":" + (indoor.samples ? String((millis() - indoor.lastGoodMs) / 1000) : String("null"));
  out += ",\"sensor_id\":" + String(indoor.sensorId);
  out += ",\"samples\":" + String(indoor.samples) + ",\"errors\":" + String(indoor.errors) + "}";
  DashboardContext context = dashboardContext();
  out += ",\"weather_api\":{\"connected\":" + String(context.apiReachable ? "true" : "false");
  out += ",\"endpoint\":" + jsonString(weatherTransfer.endpoint);
  out += ",\"hostname\":" + jsonString(weatherTransfer.serverHost);
  out += ",\"discovery\":" + jsonString(weatherTransfer.discovery);
  out += ",\"discovery_count\":" + String(weatherTransfer.discoveries);
  out += ",\"attempts\":" + String(weatherTransfer.attempts);
  out += ",\"http_code\":" + String(weatherTransfer.httpCode);
  out += ",\"error\":" + (weatherError.length() ? jsonString(weatherError) : String("null"));
  out += ",\"cache_age_s\":" + (apiLastSuccessMs ? String((millis() - apiLastSuccessMs) / 1000) : String("null"));
  out += ",\"outdoor_usable\":" + String(dashboardCurrentAvailable(context) ? "true" : "false");
  out += ",\"forecast_usable\":" + String(dashboardForecastAvailable(context) ? "true" : "false") + "}";
  out += ",\"first20_status_text\":" + jsonString(dashboardFirst20Status(context));
  uint32_t first20Issued = weatherDocument["forecast"]["issued_epoch"].as<uint32_t>();
  out += ",\"first20_issued_epoch\":" + (first20Issued ? String(first20Issued) : String("null"));
  out += ",\"current_status_text\":" + jsonString(dashboardCurrentStatus(context));
  out += ",\"current_status_observed_epoch\":" + String(weatherDocument["current"]["observed_epoch"].as<uint32_t>());
  const ArrivalChangeResult arrival = dashboardForecastAvailable(context)
      ? decodeArrivalChange(weatherDocument["forecast"]["near90"], first20Issued, dashboardNow(context))
      : ArrivalChangeResult{};
  const char *arrivalOutcome = arrival.outcome == ArrivalChangeOutcome::Within20 ? "within20"
      : arrival.outcome == ArrivalChangeOutcome::Fall20 ? "fall20"
      : arrival.outcome == ArrivalChangeOutcome::Rise20 ? "rise20" : nullptr;
  out += ",\"arrival_status_text\":" + jsonString(dashboardArrivalStatus(context));
  out += ",\"arrival_available\":" + String(arrival.available ? "true" : "false");
  out += ",\"arrival_outcome\":" + (arrivalOutcome ? jsonString(arrivalOutcome) : String("null"));
  out += ",\"arrival_outcome_probability\":" + (arrival.available && isfinite(arrival.probability)
      ? String(arrival.probability, 9) : String("null"));
  out += ",\"arrival_issued_epoch\":" + (arrival.available ? String(arrival.issuedEpoch) : String("null"));
  out += ",\"arrival_target_epoch\":" + (arrival.available ? String(arrival.arrivalEpoch) : String("null"));
  out += ",\"arrival_reference_epoch\":" + (arrival.available ? String(arrival.freshReferenceEpoch) : String("null"));
  const RidePmRange rideRange = dashboardRidePmRange(context);
  const JsonVariantConst ride = weatherDocument["forecast"]["ride90_210"];
  const bool rideMeanAvailable = dashboardRideWindowAvailable(context) && forecastModelPointAvailable(ride);
  out += ",\"ride_pm25_display\":{\"kind\":" + jsonString(rideRange.available
      ? rideRange.minimumMaximum ? "predicted_window_minimum_maximum" : "empirical_q10_q90"
      : rideMeanAvailable ? "window_mean" : "unavailable");
  out += ",\"low_ugm3\":" + (rideRange.available ? String(rideRange.low, 3) : String("null"));
  out += ",\"high_ugm3\":" + (rideRange.available ? String(rideRange.high, 3) : String("null"));
  out += ",\"mean_ugm3\":" + (rideMeanAvailable ? String(ride["pm25_ugm3"].as<float>(), 3) : String("null"));
  out += ",\"resolution_minutes\":" + String(rideRange.available && rideRange.minimumMaximum ? "15" : "null") + "}";
  out += ",\"history_available\":" + String(weatherDocument["history"]["available"].as<bool>() ? "true" : "false");
  out += ",\"history_points\":" + String(weatherDocument["history"]["points"].size());
  HistoryPmSummary pmSummary = dashboardHistoryPmSummary(context);
  out += ",\"history_pm25_summary\":{\"available\":" + String(pmSummary.available ? "true" : "false");
  out += ",\"average_ugm3\":" + (pmSummary.available ? String(pmSummary.average, 3) : String("null"));
  out += ",\"lowest_ugm3\":" + (pmSummary.available ? String(pmSummary.lowest, 3) : String("null"));
  out += ",\"highest_ugm3\":" + (pmSummary.available ? String(pmSummary.highest, 3) : String("null"));
  out += ",\"sample_count\":" + String(pmSummary.sampleCount);
  out += ",\"start_epoch\":" + String(weatherDocument["history"]["start_epoch"].as<uint32_t>());
  out += ",\"end_epoch\":" + String(weatherDocument["history"]["end_epoch"].as<uint32_t>()) + "}";
  out += ",\"page\":" + String(currentPage) + ",\"frames\":" + String(frameCount);
  const VoiceStatus voice = voiceControl.status();
  out += ",\"voice\":{\"enabled\":" + String(voice.enabled ? "true" : "false");
  out += ",\"ready\":" + String(voice.ready ? "true" : "false");
  out += ",\"playback_suspended\":" + String(voice.playbackSuspended ? "true" : "false");
  out += ",\"wake_word_required\":false,\"wakenet_enabled\":false";
  out += ",\"model\":" + jsonString(voice.model);
  out += ",\"front_end\":" + jsonString(voice.frontEnd);
  out += ",\"input_format\":" + jsonString(voice.inputFormat);
  out += ",\"microphone_channels\":" + String(voice.microphoneChannels);
  out += ",\"microphone_slots\":" + jsonString(voice.microphoneSlots);
  out += ",\"microphone_1_peak\":" + String(voice.mic1Peak);
  out += ",\"microphone_1_rms\":" + String(voice.mic1Rms);
  out += ",\"microphone_2_peak\":" + String(voice.mic2Peak);
  out += ",\"microphone_2_rms\":" + String(voice.mic2Rms);
  out += ",\"vad_model\":" + jsonString(voice.vadModel);
  out += ",\"noise_suppression\":" + jsonString(voice.noiseSuppression);
  out += ",\"aec_enabled\":" + String(voice.echoCancellation ? "true" : "false");
  out += ",\"bss_enabled\":" + String(voice.blindSourceSeparation ? "true" : "false");
  out += ",\"recognition_audio_source\":\"original_first_microphone\"";
  out += ",\"bss_output_selected\":false";
  out += ",\"fixed_first_channel\":" + String(voice.fixedFirstChannel ? "true" : "false");
  out += ",\"fixed_output_channel\":" + String(voice.fixedOutputChannel ? "true" : "false");
  out += ",\"sample_rate_hz\":" + String(VoiceConfig::kSampleRate);
  out += ",\"minimum_confidence\":" + String(VoiceConfig::kMinimumConfidence, 2);
  out += ",\"error\":" + (voice.error[0] ? jsonString(voice.error) : String("null"));
  out += ",\"audio_frames\":" + String(voice.audioFrames);
  out += ",\"feed_backpressure_frames\":" + String(voice.feedBackpressureFrames);
  out += ",\"recognition_frames\":" + String(voice.recognitionFrames);
  out += ",\"microphone_peak\":" + String(voice.microphonePeak);
  out += ",\"microphone_rms\":" + String(voice.microphoneRms);
  out += ",\"clipped_samples\":" + String(voice.clippedSamples);
  out += ",\"accepted\":" + String(voice.accepted);
  out += ",\"rejected\":" + String(voice.rejected);
  out += ",\"dropped_events\":" + String(voice.droppedEvents);
  out += ",\"stale_events\":" + String(staleVoiceEvents);
  out += ",\"recognition_timeouts\":" + String(voice.timeouts);
  out += ",\"speech_sessions\":" + String(voice.sessionsStarted);
  out += ",\"sessions_without_command\":" + String(voice.sessionsWithoutCommand);
  out += ",\"model_frames\":" + String(voice.modelFrames);
  out += ",\"loaded_phrases\":" + String(voice.loadedPhrases);
  out += ",\"diagnostics_dropped\":" + String(voice.diagnosticsDropped);
  out += ",\"last_session\":" + voiceDiagnosticJson();
  out += ",\"last_command\":" + (lastVoiceCommand ? jsonString(voiceCommandName(lastVoiceCommand)) : String("null"));
  out += ",\"last_phrase_id\":" + (lastVoiceCommand ? String(lastVoicePhrase) : String("null"));
  out += ",\"last_confidence\":" + (lastVoiceCommand ? String(lastVoiceConfidence, 3) : String("null"));
  out += ",\"last_command_age_ms\":" + (lastVoiceCommand ? String(uint32_t(millis() - lastVoiceMs)) : String("null")) + "}";
  out += ",\"readout\":{\"busy\":" + String(readoutPlayer.busy() ? "true" : "false");
  out += ",\"requests\":" + String(readoutRequests);
  out += ",\"completed\":" + String(readoutPlayer.completed());
  out += ",\"samples_written\":" + String(readoutPlayer.samplesWritten());
  out += ",\"planned_seconds\":" + String(readoutPlayer.plannedSamples() / 16000.0f, 2);
  out += ",\"page\":" + String(lastReadoutPage);
  out += ",\"source_generated_epoch\":" + String(lastReadoutGeneratedEpoch);
  out += ",\"error\":" + (readoutPlayer.lastError()[0] ? jsonString(readoutPlayer.lastError()) : String("null"));
  out += ",\"last_text\":" + jsonString(lastReadoutText);
  out += ",\"observed_epoch\":" + String(lastReadoutObservedEpoch);
  out += ",\"forecast_fetched_epoch\":" + String(lastReadoutForecastEpoch) + "}";
  out += ",\"buttons\":{\"key_gpio\":18,\"key_action\":\"read_info\",\"boot_gpio\":0,\"boot_action\":\"next_page\"}";
  out += ",\"uptime_seconds\":" + String(millis() / 1000) + "}";
  return out;
}

static void drawSetupScreen() {
  drawDashboard(*gfx, dashboardContext());
  frameCount++;
  forceDraw = false;
  lastDraw = millis();
}

static void processWeatherTransfer() {
  if (uint32_t(millis() - lastWeatherPoll) < 100) return;
  lastWeatherPoll = millis();
  WeatherTransfer result = weatherClient.take();
  if (!result.pending) return;
  weatherTransfer = result;
  if (result.body.length()) {
    JsonDocument candidate;
    DeserializationError error = deserializeJson(candidate, result.body, DeserializationOption::NestingLimit(12));
    if (!error && candidate["schema_version"] == 1 && candidate["generated_epoch"].as<uint32_t>() > 1700000000
        && candidate["current"].is<JsonObject>() && candidate["weather"].is<JsonObject>() && candidate["forecast"].is<JsonObject>()) {
      weatherDocument = std::move(candidate);
      weatherReceivedMs = result.receivedMs;
      apiLastSuccessMs = millis();
      apiReachable = true;
      weatherError = "";
      Serial.println("{\"event\":\"weather_updated\"}");
    } else { apiReachable = false; weatherError = "Invalid weather response"; }
  } else { apiReachable = false; weatherError = result.error; }
  weatherTransfer.body = "";
  forceDraw = true;
}

static void serveScreen() {
  static uint8_t image[15000];
  memset(image, 0, sizeof(image));
  const uint8_t *buffer = gfx->getBufferPtr();
  // U8g2 R1 maps user (x,y) to base (299-y,x), vertical-top-LSB,
  // with a 304-byte tile-row stride. Export the actual drawn 400x300 frame.
  for (int y = 0; y < 300; y++)
    for (int x = 0; x < 400; x++)
      if (buffer[(x / 8) * 304 + 299 - y] & (1 << (x % 8)))
        image[y * 50 + x / 8] |= 0x80 >> (x % 8);
  const char header[] = "P4\n400 300\n";
  server.setContentLength(sizeof(header) - 1 + sizeof(image));
  server.send(200, "image/x-portable-bitmap", "");
  server.client().write(reinterpret_cast<const uint8_t *>(header), sizeof(header) - 1);
  server.client().write(image, sizeof(image));
}

static bool requestReadout() {
  if (readoutPlayer.busy()) return false;
  DashboardContext context = dashboardContext();
  SpeechPlaylist playlist;
  ++readoutRequests;
  if (!buildCurrentPageReadout(context, playlist, lastReadoutText)) return false;
  lastReadoutPage = context.page;
  lastReadoutGeneratedEpoch = weatherDocument["generated_epoch"].as<uint32_t>();
  lastReadoutObservedEpoch = weatherDocument["current"]["observed_epoch"].as<uint32_t>();
  lastReadoutForecastEpoch = weatherDocument["weather"]["fetched_epoch"].as<uint32_t>();
  voiceAcknowledgement = false;
  bool started = readoutPlayer.start(playlist, voiceControl);
  forceDraw = true;
  Serial.println("{\"event\":\"weather_readout\",\"started\":" + String(started ? "true" : "false")
      + ",\"page\":" + String(lastReadoutPage)
      + ",\"text\":" + jsonString(lastReadoutText)
      + ",\"error\":" + (readoutPlayer.lastError()[0] ? jsonString(readoutPlayer.lastError()) : String("null")) + "}");
  return started;
}

static void readButtons() {
  static ReleasedButton key, boot;
  uint32_t now = millis();
  if (boot.update(digitalRead(0) == HIGH, now)) selectPage(nextDashboardPage(currentPage));
  if (key.update(digitalRead(18) == HIGH, now)) requestReadout();
}

static void processVoiceCommands() {
  VoiceEvent event;
  while (voiceControl.take(event)) {
    if (!voiceEventIsFresh(millis(), event.recognizedMs, VoiceConfig::kMaximumEventAgeMs)) {
      ++staleVoiceEvents;
      continue;
    }
    if (event.commandId == int(VoiceCommand::ReadInfo)) {
      if (!requestReadout()) continue;
    } else {
      uint8_t page;
      if (!pageForVoiceCommand(event.commandId, currentPage, page)) continue;
      selectPage(page);
    }
    lastVoiceCommand = event.commandId;
    lastVoicePhrase = event.phraseId;
    lastVoiceConfidence = event.confidence;
    lastVoiceMs = event.recognizedMs;
    voiceAcknowledgement = true;
    snprintf(voiceHint, sizeof(voiceHint), "Voice: %s", voiceCommandName(event.commandId));
    Serial.println("{\"event\":\"voice_command\",\"command\":" + jsonString(voiceCommandName(event.commandId))
        + ",\"command_id\":" + String(event.commandId) + ",\"phrase_id\":" + String(event.phraseId)
        + ",\"confidence\":" + String(event.confidence, 3) + ",\"page\":" + String(currentPage) + "}");
  }
  if (voiceAcknowledgement && uint32_t(millis() - lastVoiceMs) >= VoiceConfig::kAcknowledgementMs) {
    voiceAcknowledgement = false;
    forceDraw = true;
  }
}

// Only the main loop serializes snapshots. Audio workers copy bounded data
// into a zero-wait queue and never write diagnostic text to USB.
static void processVoiceDiagnostics() {
  VoiceDiagnostic trace;
  while (voiceControl.takeDiagnostic(trace)) {
    lastVoiceDiagnostic = trace;
    hasVoiceDiagnostic = true;
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
  if (line == "READINFO") { requestReadout(); return; }
  if (line == "REFRESH" || line == "DISCOVER") {
    weatherClient.refresh(line == "DISCOVER");
    Serial.println("{\"event\":\"weather_refresh_requested\"}");
    return;
  }
  if (line.length() == 6 && line.startsWith("VIEW ") && line[5] >= '0' && line[5] <= '2') {
    selectPage(line[5] - '0');
    Serial.println(statusJson());
    return;
  }
  if (line == "SCAN") {
    int count = WiFi.scanNetworks();
    Serial.print("{\"networks\":[");
    for (int i = 0; i < count; i++) {
      if (i) Serial.print(',');
      Serial.print("{\"ssid\":" + jsonString(WiFi.SSID(i)) + ",\"rssi\":" + String(WiFi.RSSI(i)) + ",\"channel\":" + String(WiFi.channel(i)) + ",\"auth\":" + String(int(WiFi.encryptionType(i))) + "}");
    }
    Serial.println("]}");
    WiFi.scanDelete();
    return;
  }
  if (line.startsWith("WIFI ")) {
    int separator = line.indexOf(' ', 5);
    String nextSsid, nextPassword;
    if (separator > 5 && decodeHex(line.substring(5, separator), nextSsid) && decodeHex(line.substring(separator + 1), nextPassword)
        && nextSsid.length() >= 1 && nextSsid.length() <= 32 && nextPassword.length() <= 64) {
      beginWifi(nextSsid, nextPassword, true);
      return;
    }
  }
  Serial.println("{\"error\":\"invalid_command\"}");
}

void setup() {
  // USB is an optional console. An attached host that is not reading must
  // never stall the audio workers or the loop that hands audio to the speaker.
  // Hold complete status replies when a terminal is reading; otherwise drop
  // console output immediately instead of waiting for the host.
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
  server.on("/", HTTP_GET, []() { server.send(200, "text/plain", "TTDI Weather dashboard\nKEY: read the displayed page aloud.\nBOOT: next page.\nVoice: next, next page, back, read info, read page, read, overview, page one, forecast, page two, graph, page three. No wake word.\nPOST /read-info to test the same readout as KEY.\nSee /status for microphone, voice and readout status.\n"); });
  server.on("/read-info", HTTP_POST, []() {
    if (readoutPlayer.busy()) { server.send(409, "application/json", "{\"error\":\"readout_busy\"}"); return; }
    bool started = requestReadout();
    server.send(started ? 202 : 503, "application/json", "{\"started\":" + String(started ? "true" : "false")
        + ",\"page\":" + String(lastReadoutPage)
        + ",\"text\":" + jsonString(lastReadoutText)
        + ",\"error\":" + (readoutPlayer.lastError()[0] ? jsonString(readoutPlayer.lastError()) : String("null")) + "}");
  });
  server.on("/weather", HTTP_GET, []() {
    if (weatherDocument.isNull()) { server.send(503, "application/json", "{\"error\":\"weather_unavailable\"}"); return; }
    String payload;
    serializeJson(weatherDocument, payload);
    server.send(200, "application/json", payload);
  });
  server.on("/screen.pbm", HTTP_GET, serveScreen);
  server.on("/view", HTTP_POST, []() {
    String page = server.arg("page");
    if (page.length() != 1 || page[0] < '0' || page[0] > '2') { server.send(400, "text/plain", "page must be 0, 1 or 2"); return; }
    selectPage(page[0] - '0');
    drawSetupScreen();
    server.send(200, "application/json", "{\"page\":" + page + "}");
  });
  server.on("/refresh", HTTP_POST, []() {
    weatherClient.refresh(server.arg("rediscover") == "1");
    server.send(202, "application/json", "{\"refresh\":\"requested\"}");
  });
  drawSetupScreen();
  Serial.println("{\"event\":\"voice_initializing\"}");
  const bool voiceReady = voiceControl.begin();
  Serial.println("{\"event\":\"voice_startup\",\"ready\":" + String(voiceReady ? "true" : "false")
      + ",\"error\":" + (voiceReady ? String("null") : jsonString(voiceControl.status().error)) + "}");
  weatherClient.begin();
  Serial.println("{\"event\":\"ready\",\"firmware\":\"weather-dashboard-27-public.1\"}");
  String storedSsid = settings.getString("ssid", "");
  if (storedSsid.length()) beginWifi(storedSsid, settings.getString("password", ""), false);
  drawSetupScreen();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleCommand(serialLine); serialLine = ""; }
    else if (c != '\r') {
      if (serialLine.length() < 256) serialLine += c;
      else serialLine = "";
    }
  }
  bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !wasConnected) {
    if (pendingSave) {
      size_t savedSsidBytes = settings.putString("ssid", ssid);
      size_t savedPasswordBytes = settings.putString("password", password);
      if (savedSsidBytes == ssid.length() && savedPasswordBytes == password.length()) pendingSave = false;
      else Serial.println("{\"error\":\"wifi_save_failed\"}");
    }
    timedOut = false;
    configTime(8 * 3600, 0, "pool.ntp.org", "time.cloudflare.com");
    if (!serverStarted) { server.begin(); serverStarted = true; }
    Serial.println(statusJson());
    weatherClient.refresh(true);
  }
  if (!connected && ssid.length() && !timedOut && uint32_t(millis() - connectionStarted) > 45000) {
    timedOut = true;
    Serial.println("{\"error\":\"wifi_timeout\"}");
  }
  wasConnected = connected;
  indoorSensor.update();
  processWeatherTransfer();
  bool wasReading = readoutPlayer.busy();
  readoutPlayer.update(voiceControl);
  if (wasReading && !readoutPlayer.busy()) {
    forceDraw = true;
    Serial.println("{\"event\":\"weather_readout_finished\",\"error\":"
        + (readoutPlayer.lastError()[0] ? jsonString(readoutPlayer.lastError()) : String("null")) + "}");
  }
  readButtons();
  processVoiceCommands();
  processVoiceDiagnostics();
  if (uint32_t(millis() - lastBatteryRead) >= 10000) {
    batteryVoltage = readBatteryAdcMillivolts() * 3.0f / 1000.0f;
    lastBatteryRead = millis();
  }
  if (serverStarted) server.handleClient();
  if (forceDraw || uint32_t(millis() - lastDraw) >= 5000) drawSetupScreen();
  delay(5);
}
