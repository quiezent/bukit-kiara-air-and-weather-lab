#include "Engine.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <NetworkClientSecure.h>
#include <LittleFS.h>
#include <Preferences.h>
#include <atomic>
#include <time.h>
#include <functional>
#include <vector>
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "PmFeatures.h"
#include "TlsRoots.h"
#define HGB_REAL double
#define HGB_ALLOC(bytes) ps_malloc(bytes)
#define HGB_FREE(pointer) free(pointer)
#include "HgbTrainer.h"
#define ARRIVAL_ALLOC(bytes) ps_malloc(bytes)
#define ARRIVAL_FREE(pointer) free(pointer)
#include "ArrivalLogistic.h"
#include "SessionFeatures.h"
#include "ObservedLabel.h"

namespace {
constexpr uint32_t kTtdiSeconds = 180, kNeighbourSeconds = 300, kWeatherSeconds = 900;
constexpr size_t kMaxReadings = 14500;  // 28 days at 3 min, plus a margin.
constexpr size_t kMaxPolls = 600;       // About 8 hours of six stations.
constexpr int kMaxHours = 48;
constexpr uint32_t kKeepReadingDays = 35, kKeepPollDays = 3, kKeepGasDays = 3, kKeepSessionDays = 16;
constexpr size_t kMaxGas = 1000;        // About 50 hours at 3 minutes.
constexpr size_t kMaxSamples = 520;     // The status line reads the last 24 hours.
const char *const kFsLabel = "storage";

const char *const kTtdiUrl = "https://api.airgradient.com/public/api/v1/world/locations/86311/measures/current";
const char *const kWeatherUrl =
    "https://api.open-meteo.com/v1/forecast?latitude=3.1411106257487&longitude=101.62749852676"
    "&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation_probability,precipitation,"
    "cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m,wind_speed_925hPa,wind_direction_925hPa"
    "&timeformat=unixtime&timezone=UTC&past_hours=6&forecast_hours=36";

struct Hour {
  uint32_t epoch;
  double temperature, humidity, feelsLike, rainChance, rainMm, cloud, wind, windFrom, gust, wind925, windFrom925;
};

struct WeatherRun {
  uint32_t fetched = 0;
  Hour hours[kMaxHours];
  int count = 0;
  const Hour *at(uint32_t epoch) const {
    for (int i = 0; i < count; ++i) if (hours[i].epoch == epoch) return &hours[i];
    return nullptr;
  }
};

pm::Reading *readings = nullptr;
size_t readingCount = 0;
pm::Poll *polls = nullptr;
size_t pollCount = 0;
double lastPollEnd[pm::kStations];
constexpr int kWeatherRuns = 6;
WeatherRun *runs = nullptr;
int runNext = 0;
WeatherRun *weather = nullptr;  // The newest run.
pm::Window *window = nullptr;

SemaphoreHandle_t mutex = nullptr;
JsonDocument *pending = nullptr;
bool pendingReady = false;
std::atomic<bool> refreshRequested{true};
EngineStatus stats;

// ---------------------------------------------------------------- small helpers
bool validClock(uint32_t now) { return now > 1700000000; }

int64_t daysFromCivil(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + int64_t(doe) - 719468;
}

bool parseIso(const char *text, double &out) {
  if (!text) return false;
  int y, mo, d, h, mi, s;
  if (sscanf(text, "%d-%d-%dT%d:%d:%d", &y, &mo, &d, &h, &mi, &s) != 6) return false;
  out = double(daysFromCivil(y, unsigned(mo), unsigned(d)) * 86400 + h * 3600 + mi * 60 + s);
  return true;
}

// ArduinoJson 7 keeps a short decimal as float32, so "16.8" reads back as 16.799999237.
// This restores the two-decimal value the server parses; a number that really needs more
// digits arrives as a double and is left alone.
double exactDecimal(double value) {
  const double rounded = std::nearbyint(value * 100.0) / 100.0;
  return float(rounded) == float(value) ? rounded : value;
}

double jsonNumber(JsonVariantConst value) {
  if (value.is<bool>() || !(value.is<double>() || value.is<long>())) return pm::kNaN;
  const double number = value.as<double>();
  return std::isfinite(number) ? exactDecimal(number) : pm::kNaN;
}

double jsonFloat(JsonVariantConst value) { return jsonNumber(value); }

String dayStamp(uint32_t epoch) {
  const time_t local = time_t(epoch) + 8 * 3600;
  struct tm parts;
  gmtime_r(&local, &parts);
  char text[12];
  strftime(text, sizeof(text), "%Y%m%d", &parts);
  return text;
}

bool httpsGet(const String &url, String &body, int &code) {
  NetworkClientSecure client;
  client.setCACert(kTlsRoots);
  client.setHandshakeTimeout(20);  // Seconds. The default of 120 lets one stalled host hold every feed.
  HTTPClient http;
  http.setConnectTimeout(15000);
  http.setTimeout(15000);
  http.useHTTP10(true);
  http.setReuse(false);
  http.setUserAgent("TTDI-RLCD/40");
  code = -1;
  if (!http.begin(client, url)) return false;
  http.addHeader("Accept", "application/json");
  code = http.GET();
  stats.lastHttpCode = code;
  if (code < 0) {
    char detail[8];
    stats.lastTlsError = client.lastError(detail, sizeof(detail));
  }
  bool ok = false;
  if (code == 200) {
    body = http.getString();
    ok = body.length() > 0;
    if (ok) stats.lastHttpsOkMs = millis();
  }
  http.end();
  client.stop();
  return ok;
}

// ---------------------------------------------------------------- storage
void appendRecord(const char *folder, uint32_t epoch, const void *data, size_t bytes) {
  if (!stats.storageReady) return;
  const String path = String("/") + folder + "/" + dayStamp(epoch) + ".bin";
  File file = LittleFS.open(path, FILE_APPEND, true);
  if (file) { file.write(static_cast<const uint8_t *>(data), bytes); file.close(); }
}

// Names are yyyymmdd.bin: lexical order is date order.
std::vector<String> dayFiles(const char *folder) {
  std::vector<String> names;
  File dir = LittleFS.open(String("/") + folder);
  if (!dir || !dir.isDirectory()) return names;
  for (File entry = dir.openNextFile(); entry; entry = dir.openNextFile()) names.push_back(entry.name());
  std::sort(names.begin(), names.end());
  return names;
}

void pruneFolder(const char *folder, uint32_t keepDays, uint32_t now) {
  const String oldest = dayStamp(now - keepDays * 86400) + ".bin";
  for (const String &name : dayFiles(folder))
    if (name < oldest) LittleFS.remove(String("/") + folder + "/" + name);
}

// Loads the newest rows that fit: the oldest are the ones to drop when history exceeds capacity.
template <typename Row>
size_t loadFolder(const char *folder, Row *rows, size_t capacity, const String &firstName) {
  std::vector<String> names;
  size_t total = 0;
  for (const String &name : dayFiles(folder)) {
    if (name < firstName) continue;
    File file = LittleFS.open(String("/") + folder + "/" + name, FILE_READ);
    if (!file) continue;
    total += file.size() / sizeof(Row);
    file.close();
    names.push_back(name);
  }
  size_t skip = total > capacity ? total - capacity : 0;
  size_t count = 0;
  for (const String &name : names) {
    File file = LittleFS.open(String("/") + folder + "/" + name, FILE_READ);
    if (!file) continue;
    const size_t inFile = file.size() / sizeof(Row);
    if (skip >= inFile) { skip -= inFile; file.close(); continue; }
    if (skip) { file.seek(skip * sizeof(Row)); skip = 0; }
    while (count < capacity && file.read(reinterpret_cast<uint8_t *>(&rows[count]), sizeof(Row)) == sizeof(Row)) ++count;
    file.close();
  }
  return count;
}

void saveWeatherBody(const String &body) {
  if (!stats.storageReady) return;
  File file = LittleFS.open("/w/latest.tmp", FILE_WRITE, true);
  if (!file) return;
  file.print(body);
  file.close();
  LittleFS.remove("/w/latest.json");
  LittleFS.rename("/w/latest.tmp", "/w/latest.json");
}

// ---------------------------------------------------------------- collection
bool parseWeather(const String &body, uint32_t fetched) {
  JsonDocument doc;
  if (deserializeJson(doc, body)) return false;
  JsonObjectConst hourly = doc["hourly"];
  JsonArrayConst times = hourly["time"];
  if (times.isNull() || times.size() == 0) return false;
  WeatherRun &run = runs[runNext];
  run.count = 0;
  int index = 0;
  for (JsonVariantConst stamp : times) {
    if (run.count == kMaxHours) break;
    Hour &h = run.hours[run.count++];
    h.epoch = stamp.as<uint32_t>();
    h.temperature = jsonFloat(hourly["temperature_2m"][index]);
    h.humidity = jsonFloat(hourly["relative_humidity_2m"][index]);
    h.feelsLike = jsonFloat(hourly["apparent_temperature"][index]);
    h.rainChance = jsonFloat(hourly["precipitation_probability"][index]);
    h.rainMm = jsonFloat(hourly["precipitation"][index]);
    h.cloud = jsonFloat(hourly["cloud_cover"][index]);
    h.wind = jsonFloat(hourly["wind_speed_10m"][index]);
    h.windFrom = jsonFloat(hourly["wind_direction_10m"][index]);
    h.gust = jsonFloat(hourly["wind_gusts_10m"][index]);
    h.wind925 = jsonFloat(hourly["wind_speed_925hPa"][index]);
    h.windFrom925 = jsonFloat(hourly["wind_direction_925hPa"][index]);
    ++index;
  }
  run.fetched = fetched;
  stats.lastWeatherEpoch = fetched;
  if (!run.count) return false;
  weather = &run;
  runNext = (runNext + 1) % kWeatherRuns;
  return true;
}

void addReading(const pm::Reading &row) {
  xSemaphoreTake(mutex, portMAX_DELAY);
  if (readingCount == kMaxReadings) {
    memmove(readings, readings + 480, (kMaxReadings - 480) * sizeof(pm::Reading));
    readingCount -= 480;
  }
  readings[readingCount++] = row;
  xSemaphoreGive(mutex);
  stats.readings = readingCount;
  stats.lastReadingEpoch = uint32_t(row.epoch);
  appendRecord("r", uint32_t(row.epoch), &row, sizeof(row));
}

struct Gas { double epoch, count, tvoc, co2; };
Gas *gases = nullptr;
size_t gasCount = 0;

void addGas(const Gas &row) {
  if (gasCount == kMaxGas) {
    memmove(gases, gases + 60, (kMaxGas - 60) * sizeof(Gas));
    gasCount -= 60;
  }
  gases[gasCount++] = row;
  appendRecord("g", uint32_t(row.epoch), &row, sizeof(row));
}

bool fetchTtdi(uint32_t) {
  String body;
  int code;
  if (!httpsGet(kTtdiUrl, body, code)) { ++stats.ttdiFail; return false; }
  const uint32_t received = uint32_t(time(nullptr));
  JsonDocument doc;
  double observed;
  if (deserializeJson(doc, body) || !parseIso(doc["timestamp"], observed)) { ++stats.ttdiFail; return false; }
  ++stats.ttdiOk;
  if (readingCount && observed <= readings[readingCount - 1].epoch) return false;  // Not a new observation.
  if (observed > received + 60) return false;  // The server allows the same 60 s.
  double pmValue = jsonNumber(doc["pm02"]);
  if (std::isfinite(pmValue) && pmValue < 0) pmValue = pm::kNaN;
  addReading({observed, pmValue, jsonNumber(doc["atmp"]), jsonNumber(doc["rhum"]), double(received),
              jsonNumber(doc["heatindex"]), jsonNumber(doc["pm10"])});
  addGas({observed, jsonNumber(doc["pm003Count"]), jsonNumber(doc["tvoc"]), jsonNumber(doc["rco2"])});
  return true;
}

uint32_t nextTtdi = 0;

// As on the laptop: a failed poll or a stale reading retries after 30 s, not the full period.
bool pollTtdi() {
  const uint32_t started = uint32_t(time(nullptr));
  const uint32_t failures = stats.ttdiFail;
  const bool changed = fetchTtdi(started);
  const uint32_t finished = uint32_t(time(nullptr));
  const bool fresh = stats.ttdiFail == failures && stats.lastReadingEpoch && finished <= stats.lastReadingEpoch + 420;
  nextTtdi = std::max(finished + 5, started + (fresh ? kTtdiSeconds : 30));
  return changed;
}

void publish(uint32_t now);

void addPoll(const pm::Poll &row) {
  if (pollCount == kMaxPolls) {
    memmove(polls, polls + 60, (kMaxPolls - 60) * sizeof(pm::Poll));
    pollCount -= 60;
  }
  // Stations report slightly out of order; keep the array sorted by end clock.
  size_t at = pollCount;
  while (at > 0 && polls[at - 1].end > row.end) --at;
  memmove(polls + at + 1, polls + at, (pollCount - at) * sizeof(pm::Poll));
  polls[at] = row;
  ++pollCount;
  stats.polls = pollCount;
  appendRecord("p", uint32_t(row.end), &row, sizeof(row));
}

void fetchNeighbours() {
  for (int station = 0; station < pm::kStations; ++station) {
    // One stalled host must not hold up the sensor: a missed reading is never recovered.
    if (uint32_t(time(nullptr)) >= nextTtdi && pollTtdi()) publish(uint32_t(time(nullptr)));
    const String url = String("https://map-data.airgradient.com/map/api/v1/locations/")
        + String(pm::kStationTable[station].id) + "/measures/current";
    String body;
    int code;
    if (!httpsGet(url, body, code)) { ++stats.neighbourFail; continue; }
    const double received = double(time(nullptr));
    JsonDocument doc;
    double measured;
    if (deserializeJson(doc, body) || !parseIso(doc["measuredAt"], measured)) { ++stats.neighbourFail; continue; }
    ++stats.neighbourOk;
    const double value = jsonNumber(doc["pm25"]);
    if (!std::isfinite(value) || value < 0 || value > 1000 || measured > received) continue;
    if (measured == lastPollEnd[station]) continue;  // First receipt of each value only.
    lastPollEnd[station] = measured;
    addPoll({int32_t(station), 0, measured, value, received});
    vTaskDelay(1);
  }
}

bool fetchWeather() {
  String body;
  int code;
  if (!httpsGet(kWeatherUrl, body, code)) { ++stats.weatherFail; return false; }
  if (!parseWeather(body, uint32_t(time(nullptr)))) { ++stats.weatherFail; return false; }
  ++stats.weatherOk;
  saveWeatherBody(body);
  return true;
}

// ---------------------------------------------------------------- payload
double roundTo(double value, int digits) {
  const double scale = digits == 1 ? 10.0 : 1.0;
  return std::nearbyint(value * scale) / scale;  // Halves to even, as Python's round().
}

void setNumber(JsonObject object, const char *key, double value) {
  if (std::isfinite(value)) object[key] = value;
  else object[key] = nullptr;
}

// One hourly weather field over an exact window, as weather_contracts.py: instant
// fields use the block they start, interval fields the block they end.
struct FieldStat { bool complete = true; double high = -INFINITY, sum = 0, weight = 0; };

template <typename Getter>
FieldStat windowField(uint32_t start, uint32_t end, bool interval, Getter get) {
  FieldStat out;
  const uint32_t first = start / 3600 * 3600;
  const uint32_t last = (end + 3599) / 3600 * 3600;
  for (uint32_t block = first; block < last; block += 3600) {
    const double overlap = double(std::min(end, block + 3600) - std::max(start, block));
    if (overlap <= 0) continue;
    const Hour *hour = weather->at(interval ? block + 3600 : block);
    const double value = hour ? double(get(*hour)) : pm::kNaN;
    if (!std::isfinite(value)) { out.complete = false; continue; }
    const double weight = overlap / 3600;
    out.high = std::max(out.high, value);
    out.sum += value * weight;
    out.weight += weight;
  }
  return out;
}

bool weatherUsable(uint32_t now) { return weather->count && now >= weather->fetched && now - weather->fetched <= 7200; }

void addWindowWeather(JsonObject out, uint32_t now, uint32_t issued, uint32_t start, uint32_t end) {
  const bool usable = weatherUsable(now) && start && end > start && weather->fetched <= issued;
  const FieldStat temperature = windowField(start, end, false, [](const Hour &h) { return h.temperature; });
  const FieldStat feels = windowField(start, end, false, [](const Hour &h) { return h.feelsLike; });
  const FieldStat humidity = windowField(start, end, false, [](const Hour &h) { return h.humidity; });
  const FieldStat wind = windowField(start, end, false, [](const Hour &h) { return h.wind; });
  const FieldStat gust = windowField(start, end, true, [](const Hour &h) { return h.gust; });
  const FieldStat rainMm = windowField(start, end, true, [](const Hour &h) { return h.rainMm; });
  const FieldStat rainChance = windowField(start, end, true, [](const Hour &h) { return h.rainChance; });
  const bool complete = usable && temperature.complete && feels.complete && humidity.complete && wind.complete
      && gust.complete && rainMm.complete && rainChance.complete;
  const uint32_t age = weather->count && now >= weather->fetched ? now - weather->fetched : UINT32_MAX;
  out["available"] = complete;
  out["fresh"] = complete && age <= 1800;
  if (weather->fetched) out["fetched_epoch"] = weather->fetched; else out["fetched_epoch"] = nullptr;
  out["coverage_verified"] = complete;
  out["coverage"]["complete"] = complete;
  if (complete) {
    setNumber(out, "temperature_max_c", roundTo(temperature.high, 1));
    setNumber(out, "feels_like_max_c", roundTo(feels.high, 1));
    setNumber(out, "humidity_mean_pct", roundTo(humidity.sum / humidity.weight, 0));
    setNumber(out, "wind_mean_kmh", roundTo(wind.sum / wind.weight, 1));
    setNumber(out, "gust_max_kmh", roundTo(gust.high, 1));
    setNumber(out, "rain_chance_max_pct", roundTo(rainChance.high, 0));
    setNumber(out, "rain_mm", roundTo(rainMm.sum, 1));
  }
}

void addTennis(JsonObject out, uint32_t now) {
  const uint32_t localDay = (now + 8 * 3600) / 86400;
  uint32_t start = localDay * 86400 + 7 * 3600 - 8 * 3600;
  if (now >= start + 7200) start += 86400;
  const uint32_t end = start + 7200;
  const bool source = weatherUsable(now);
  const Hour *a = weather->at(start), *b = weather->at(start + 3600), *c = weather->at(end);
  const bool points = source && a && b && c
      && std::isfinite(b->rainChance) && std::isfinite(c->rainChance)
      && std::isfinite(a->feelsLike) && std::isfinite(b->feelsLike)
      && std::isfinite(a->wind) && std::isfinite(b->wind);
  out["available"] = points;
  out["fresh"] = points && now - weather->fetched <= 1800;
  out["start_epoch"] = start;
  out["end_epoch"] = end;
  if (weather->fetched) out["fetched_epoch"] = weather->fetched; else out["fetched_epoch"] = nullptr;
  if (points) {
    out["rain_chance_max_pct"] = std::max(b->rainChance, c->rainChance);
    out["feels_like_max_c"] = roundTo(std::max(a->feelsLike, b->feelsLike), 1);
    out["wind_mean_kmh"] = roundTo((double(a->wind) + double(b->wind)) / 2, 1);
  }
}

void addHistory(JsonObject out, uint32_t now) {
  const uint32_t start = now - 6 * 3600;
  size_t first = readingCount;
  while (first > 0 && readings[first - 1].epoch >= start) --first;
  const size_t available = readingCount - first;
  if (available > 128) first = readingCount - 128;  // At 3-minute cadence the window holds about 120 rows.
  out["available"] = available > 0;
  const uint32_t observed = available ? uint32_t(readings[readingCount - 1].epoch) : 0;
  out["fresh"] = available && now >= observed && now - observed <= 420;
  if (available) out["observed_epoch"] = observed; else out["observed_epoch"] = nullptr;
  out["start_epoch"] = start;
  out["end_epoch"] = now;
  JsonArray columns = out["columns"].to<JsonArray>();
  columns.add("epoch"); columns.add("pm25_ugm3"); columns.add("temperature_c"); columns.add("heat_index_c");
  out["gap_after_s"] = 600;
  out["gap_kind"] = "source_collection_before_downsampling";
  JsonArray points = out["points"].to<JsonArray>();
  JsonArray gaps = out["gaps"].to<JsonArray>();
  double sum = 0, low = INFINITY, high = -INFINITY;
  uint32_t samples = 0;
  for (size_t i = first; i < readingCount; ++i) {
    const pm::Reading &row = readings[i];
    JsonArray point = points.add<JsonArray>();
    point.add(uint32_t(row.epoch));
    const auto put = [&](double value) { if (std::isfinite(value)) point.add(roundTo(value, 1)); else point.add(nullptr); };
    put(row.pm); put(row.temperature); put(row.heatIndex);
    if (std::isfinite(row.pm)) { sum += row.pm; low = std::min(low, row.pm); high = std::max(high, row.pm); ++samples; }
    if (i > first && row.epoch - readings[i - 1].epoch > 600) {
      JsonArray gap = gaps.add<JsonArray>();
      gap.add(uint32_t(readings[i - 1].epoch));
      gap.add(uint32_t(row.epoch));
    }
  }
  JsonObject summary = out["pm25_summary"].to<JsonObject>();
  summary["available"] = samples > 0;
  summary["sample_count"] = samples;
  if (samples) { summary["average_ugm3"] = sum / samples; summary["lowest_ugm3"] = low; summary["highest_ugm3"] = high; }
}

obs::Sample *samples = nullptr;
obs::Scratch observeScratch{nullptr, 0, nullptr, 0};

// The line under the outdoor number, from the last 24 hours of readings.
const char *statusLine(uint32_t now) {
  if (!samples || !observeScratch.three || !observeScratch.fifteen) return obs::kDefaultText;
  size_t first = pm::lowerBoundEpoch(readings, readingCount, double(now) - obs::kDeviceLookback - 1);
  if (readingCount - first > kMaxSamples) first = readingCount - kMaxSamples;
  size_t used = 0, gas = 0;
  for (size_t i = first; i < readingCount; ++i) {
    const pm::Reading &r = readings[i];
    while (gas < gasCount && gases[gas].epoch < r.epoch) ++gas;
    const bool hasGas = gas < gasCount && gases[gas].epoch == r.epoch;
    samples[used++] = {r.epoch, r.pm, r.pm10, hasGas ? gases[gas].count : obs::kNaN, r.temperature, r.humidity,
                       r.heatIndex, hasGas ? gases[gas].co2 : obs::kNaN, hasGas ? gases[gas].tvoc : obs::kNaN};
  }
  const uint32_t started = millis();
  const obs::Result result = obs::observe(samples, used, double(now), obs::kDeviceLookback, observeScratch);
  stats.observeMs = millis() - started;
  return result.displayText;
}

void addCurrent(JsonObject out, uint32_t now) {
  const pm::Reading *row = readingCount ? &readings[readingCount - 1] : nullptr;
  const uint32_t age = row && now >= row->epoch ? now - uint32_t(row->epoch) : UINT32_MAX;
  const bool ok = age <= 900 && row && std::isfinite(row->pm);
  out["available"] = ok;
  out["fresh"] = ok && age <= 420;
  if (row) out["observed_epoch"] = uint32_t(row->epoch); else out["observed_epoch"] = nullptr;
  if (ok) {
    setNumber(out, "pm25_ugm3", row->pm);
    setNumber(out, "temperature_c", row->temperature);
    setNumber(out, "humidity_pct", row->humidity);
    setNumber(out, "heat_index_c", row->heatIndex);
  }
  out["display_text"] = statusLine(now);
}

void addWeatherNow(JsonObject out, uint32_t now) {
  const bool usable = weatherUsable(now);
  out["fresh"] = usable && now - weather->fetched <= 1800;
  if (weather->fetched) out["fetched_epoch"] = weather->fetched; else out["fetched_epoch"] = nullptr;
  const Hour *current = nullptr;
  JsonArray next = out["next_hours"].to<JsonArray>();
  if (usable)
    for (int i = 0; i < weather->count; ++i) {
      const Hour &h = weather->hours[i];
      if (now >= h.epoch && now - h.epoch < 3600) current = &h;
      if (h.epoch > now && h.epoch <= now + 6 * 3600 && next.size() < 6) {
        JsonObject item = next.add<JsonObject>();
        item["valid_epoch"] = h.epoch;
        item["rain_start_epoch"] = h.epoch - 3600;
        item["rain_end_epoch"] = h.epoch;
        setNumber(item, "rain_chance_pct", h.rainChance);
      }
    }
  if (current) setNumber(out["current_hour"].to<JsonObject>(), "wind_kmh", current->wind);
  out["available"] = usable && (current || next.size());
}

// The next 2 h session in its window that still leaves the 90-minute trip.
uint32_t nextSession(uint32_t now, int startHour) {
  uint32_t day = (now + 8 * 3600) / 86400;
  for (int offset = 0; offset < 4; ++offset) {
    const uint32_t start = (day + offset) * 86400 + startHour * 3600 - 8 * 3600;
    if (start >= now + 5400) return start;
  }
  return 0;
}

// ---------------------------------------------------------------- wind and input rows
const WeatherRun *selectRun(uint32_t issue) {
  const WeatherRun *best = nullptr;
  for (int i = 0; i < kWeatherRuns; ++i) {
    const WeatherRun &run = runs[i];
    if (!run.count || run.fetched > issue || issue - run.fetched > 7200) continue;
    if (!best || run.fetched > best->fetched) best = &run;
  }
  return best;
}

double towardAt(const WeatherRun &run, bool level925, bool east, double when) {
  const auto component = [&](const Hour &h) {
    const double speed = level925 ? h.wind925 : h.wind;
    const double angle = std::fmod(double(level925 ? h.windFrom925 : h.windFrom), 360.0) * pm::kPi / 180;
    return east ? -speed * std::sin(angle) : -speed * std::cos(angle);
  };
  if (run.count < 2) return pm::kNaN;
  int hi = 0;
  while (hi < run.count && double(run.hours[hi].epoch) < when) ++hi;
  if (hi < run.count && double(run.hours[hi].epoch) == when) return component(run.hours[hi]);
  if (hi == 0 || hi >= run.count || run.hours[hi].epoch - run.hours[hi - 1].epoch > 3600) return pm::kNaN;
  const double low = component(run.hours[hi - 1]), high = component(run.hours[hi]);
  const double fraction = (when - run.hours[hi - 1].epoch) / double(run.hours[hi].epoch - run.hours[hi - 1].epoch);
  return low + (high - low) * fraction;
}

bool fillWind(uint32_t issue, pm::Wind &wind) {
  const WeatherRun *run = selectRun(issue);
  if (!run) return false;
  for (int k = 0; k < 4; ++k) {
    const double when = double(issue) + k * 1800.0;
    wind.u10[k] = towardAt(*run, false, true, when);
    wind.v10[k] = towardAt(*run, false, false, when);
    wind.u925[k] = towardAt(*run, true, true, when);
    wind.v925[k] = towardAt(*run, true, false, when);
  }
  return true;
}

struct OriginRow {
  double issue, fresh;
  double x[pm::kInputCount];
};

void storeOriginRow(uint32_t issue) {
  OriginRow row;
  row.issue = issue;
  pm::Wind wind;
  const bool hasWind = fillWind(issue, wind);
  row.fresh = pm::inputRow(readings, readingCount, polls, pollCount, issue, hasWind ? &wind : nullptr, row.x, *window) ? 1 : 0;
  appendRecord("x", issue, &row, sizeof(row));
}

// Weather inputs of the session model, as issued at each :00 and :30 origin. They cannot be
// rebuilt later; the sensor inputs can, so those are made again from readings at the fit.
constexpr int kSessionWeatherColumns = ses::kTargetWind925MeanKmh - ses::kPriorRainMm + 1;
struct SessionRow {
  double origin;
  double weather[ses::kLeadCount][kSessionWeatherColumns + 1];  // The extra column is the accepted flag.
};

ses::Run *sessionRun = nullptr;

void loadSessionRun(const WeatherRun &in, ses::Run &out) {
  out.fetched = in.fetched;
  out.invalid = false;
  out.count = std::min(in.count, ses::kMaxHours);
  for (int i = 0; i < out.count; ++i) {
    const Hour &h = in.hours[i];
    out.hours[i] = {double(h.epoch), h.rainMm, h.rainChance, h.temperature, h.humidity, h.wind, h.wind925};
    ses::sanitize(out.hours[i]);
  }
}

void storeSessionRow(uint32_t origin) {
  const WeatherRun *run = selectRun(origin);
  if (!run || !sessionRun) return;
  loadSessionRun(*run, *sessionRun);
  static SessionRow row;
  row.origin = origin;
  double x[ses::kInputs];
  for (int lead = 0; lead < ses::kLeadCount; ++lead) {
    const double start = double(origin) + ses::kLeads[lead] * 60.0;
    const bool accepted = ses::weatherInputs(sessionRun, origin, start, start + 7200, x);
    memcpy(row.weather[lead], x + ses::kPriorRainMm, kSessionWeatherColumns * sizeof(double));
    row.weather[lead][kSessionWeatherColumns] = accepted ? 1 : 0;
  }
  appendRecord("s", origin, &row, sizeof(row));
}

// ---------------------------------------------------------------- nightly models
constexpr uint32_t kModelMagic = 0x54544433;  // "TTD3"
constexpr uint32_t kArrivalMagic = 0x54544134, kSessionMagic = 0x54545331;  // "TTA4", "TTS1"
constexpr int kHeads = 4;  // arrival, ride mean, ride min, ride max
constexpr uint32_t kMinTrainingOrigins = 240;
constexpr size_t kMaxTrainingRows = 3200;

struct PointModel {
  uint32_t cutoff = 0, fitted = 0;
  hgb::Forest heads[kHeads];
  double *z = nullptr;
  uint32_t zCount = 0;
  void release() { for (auto &head : heads) head.release(); free(z); z = nullptr; zCount = 0; }
};

PointModel *model = nullptr;
SemaphoreHandle_t modelMutex = nullptr;
std::atomic<uint32_t> fitStage{0};
std::atomic<uint32_t> lastFitMs{0};
hgb::Trainer *trainer = nullptr;

uint32_t midnightOf(uint32_t epoch) { return (epoch + 8 * 3600) / 86400 * 86400 - 8 * 3600; }

struct ArrivalFit { uint32_t cutoff = 0, fitted = 0; arrival::Model model; };
struct SessionFit { uint32_t cutoff = 0, fitted = 0; hgb::Forest forest; ses::Imputer imputer; };
ArrivalFit *arrivalFit = nullptr;
SessionFit *sessionFit = nullptr;
std::atomic<uint32_t> arrivalFitMs{0}, sessionFitMs{0};

void fitYield() { vTaskDelay(1); }

bool saveModel(const PointModel &m) {
  File file = LittleFS.open("/m/model.tmp", FILE_WRITE, true);
  if (!file) return false;
  const auto put = [&](const void *data, size_t bytes) { return file.write(static_cast<const uint8_t *>(data), bytes) == bytes; };
  bool ok = put(&kModelMagic, 4) && put(&m.cutoff, 4) && put(&m.fitted, 4) && put(&m.zCount, 4);
  for (int h = 0; h < kHeads && ok; ++h) {
    const hgb::Forest &f = m.heads[h];
    ok = put(&f.baseline, 8) && put(&f.treeCount, 2) && put(&f.nodeCount, 4)
        && put(f.rootOffsets, f.treeCount * sizeof(uint32_t)) && put(f.nodes, f.nodeCount * sizeof(hgb::Node));
  }
  ok = ok && put(m.z, m.zCount * sizeof(double));
  file.close();
  if (!ok) { LittleFS.remove("/m/model.tmp"); return false; }
  LittleFS.remove("/m/model.bin");
  return LittleFS.rename("/m/model.tmp", "/m/model.bin");
}

PointModel *loadModel() {
  File file = LittleFS.open("/m/model.bin", FILE_READ);
  if (!file) return nullptr;
  PointModel *m = new PointModel();
  const auto get = [&](void *data, size_t bytes) { return file.read(static_cast<uint8_t *>(data), bytes) == bytes; };
  uint32_t magic = 0;
  bool ok = get(&magic, 4) && magic == kModelMagic && get(&m->cutoff, 4) && get(&m->fitted, 4) && get(&m->zCount, 4);
  for (int h = 0; h < kHeads && ok; ++h) {
    hgb::Forest &f = m->heads[h];
    ok = get(&f.baseline, 8) && get(&f.treeCount, 2) && get(&f.nodeCount, 4);
    if (!ok) break;
    f.rootOffsets = static_cast<uint32_t *>(ps_malloc(f.treeCount * sizeof(uint32_t)));
    f.nodes = static_cast<hgb::Node *>(ps_malloc(f.nodeCount * sizeof(hgb::Node)));
    ok = f.rootOffsets && f.nodes && get(f.rootOffsets, f.treeCount * sizeof(uint32_t)) && get(f.nodes, f.nodeCount * sizeof(hgb::Node));
  }
  if (ok) {
    m->z = static_cast<double *>(ps_malloc((m->zCount ? m->zCount : 1) * sizeof(double)));
    ok = m->z && get(m->z, m->zCount * sizeof(double));
  }
  file.close();
  if (!ok) { m->release(); delete m; return nullptr; }
  return m;
}

// One night's fit: four heads plus the out-of-fold residuals behind P(mean <= 70).
bool fitModels(uint32_t cutoff) {
  OriginRow *rows = static_cast<OriginRow *>(ps_malloc(kMaxTrainingRows * sizeof(OriginRow)));
  double *X = static_cast<double *>(ps_malloc(kMaxTrainingRows * pm::kInputCount * sizeof(double)));
  double *y = static_cast<double *>(ps_malloc(kMaxTrainingRows * sizeof(double)));
  double *medians = static_cast<double *>(ps_malloc((28 * 96 + 64) * sizeof(double)));
  const size_t labelCapacity = kMaxReadings;
  double *labelEpochs = static_cast<double *>(ps_malloc(labelCapacity * sizeof(double)));
  double *labelValues = static_cast<double *>(ps_malloc(labelCapacity * sizeof(double)));
  PointModel *next = new PointModel();
  bool ok = false;
  do {
    if (!rows || !X || !y || !medians || !labelEpochs || !labelValues) break;
    // Rows: 15-minute origins strictly inside the training window.
    fitStage = 1;
    size_t used = 0;
    const String firstDay = dayStamp(cutoff - 29 * 86400) + ".bin";
    for (const String &name : dayFiles("x")) {
      if (name < firstDay) continue;
      File file = LittleFS.open(String("/x/") + name, FILE_READ);
      if (!file) continue;
      OriginRow row;
      while (file.read(reinterpret_cast<uint8_t *>(&row), sizeof(row)) == sizeof(row)) {
        const uint32_t issue = uint32_t(row.issue);
        if (issue % 900 || issue <= cutoff - 28 * 86400 || issue >= cutoff - 330 * 60) continue;
        if (issue + 5400 >= cutoff || issue + 12600 >= cutoff) continue;
        if (row.fresh == 0 || !(row.x[0] >= 0)) continue;
        if (used && issue <= uint32_t(rows[used - 1].issue)) continue;
        if (used < kMaxTrainingRows) rows[used++] = row;
      }
      file.close();
      vTaskDelay(1);
    }
    if (used < kMinTrainingOrigins) break;
    // Labels: the same 15-minute medians as production.
    fitStage = 2;
    const uint32_t base = uint32_t(rows[0].issue);
    size_t labelCount = 0;
    xSemaphoreTake(mutex, portMAX_DELAY);
    for (size_t i = 0; i < readingCount && labelCount < labelCapacity; ++i) {
      const pm::Reading &r = readings[i];
      if (r.epoch > base - 900 && r.epoch <= cutoff && std::isfinite(r.pm) && r.pm >= 0) {
        labelEpochs[labelCount] = r.epoch;
        labelValues[labelCount++] = r.pm;
      }
    }
    xSemaphoreGive(mutex);
    const size_t bucketCount = std::min<size_t>((cutoff - base) / 900 + 1, 28 * 96 + 64);
    double scratch[64];
    size_t cursor = 0;
    for (size_t k = 0; k < bucketCount; ++k)
      medians[k] = pm::bucketMedian(labelEpochs, labelValues, labelCount, double(base) + k * 900.0, cursor, scratch);
    pm::Targets *targets = static_cast<pm::Targets *>(ps_malloc(used * sizeof(pm::Targets)));
    if (!targets) break;
    for (size_t i = 0; i < used; ++i)
      targets[i] = pm::gridTargets(medians, bucketCount, (uint32_t(rows[i].issue) - base) / 900);
    // Point heads learn the change from the fresh reference.
    const hgb::Params params;
    const auto copyRows = [&](auto pick, size_t &count) {
      count = 0;
      for (size_t i = 0; i < used; ++i) {
        const double target = pick(targets[i]);
        if (!std::isfinite(target)) continue;
        memcpy(X + count * pm::kInputCount, rows[i].x, pm::kInputCount * sizeof(double));
        y[count] = target - rows[i].x[0];
        ++count;
      }
    };
    const auto fitHead = [&](int head, auto pick) {
      size_t count;
      copyRows(pick, count);
      fitStage = 3 + head;
      return count >= kMinTrainingOrigins && trainer->fit(X, y, uint32_t(count), pm::kInputCount, params, next->heads[head], fitYield);
    };
    bool heads = fitHead(0, [](const pm::Targets &t) { return t.arrival; })
        && fitHead(1, [](const pm::Targets &t) { return t.rideMean; })
        && fitHead(2, [](const pm::Targets &t) { return t.rideMin; })
        && fitHead(3, [](const pm::Targets &t) { return t.rideMax; });
    if (!heads) { free(targets); break; }
    // Out-of-fold ride-mean predictions over three contiguous day blocks.
    fitStage = 7;
    size_t usable = 0;
    uint32_t *rowDay = static_cast<uint32_t *>(ps_malloc(used * sizeof(uint32_t)));
    uint32_t *days = static_cast<uint32_t *>(ps_malloc(used * sizeof(uint32_t)));
    double *oof = static_cast<double *>(ps_malloc(used * sizeof(double)));
    uint16_t *keep = static_cast<uint16_t *>(ps_malloc(used * sizeof(uint16_t)));
    next->z = static_cast<double *>(ps_malloc(used * sizeof(double)));
    if (!rowDay || !days || !oof || !keep || !next->z) { free(targets); free(rowDay); free(days); free(oof); free(keep); break; }
    for (size_t i = 0; i < used; ++i)
      if (std::isfinite(targets[i].rideMean)) { keep[usable] = uint16_t(i); rowDay[usable] = uint32_t(rows[i].issue) / 86400; ++usable; }
    size_t dayCount = 0;
    for (size_t i = 0; i < usable; ++i)
      if (!dayCount || days[dayCount - 1] != rowDay[i]) days[dayCount++] = rowDay[i];
    bool folds = dayCount >= 3;
    size_t startDay = 0;
    for (int block = 0; block < 3 && folds; ++block) {
      const size_t length = dayCount / 3 + (size_t(block) < dayCount % 3 ? 1 : 0);
      const uint32_t firstHeld = days[startDay], lastHeld = days[startDay + length - 1];
      startDay += length;
      size_t count = 0;
      for (size_t i = 0; i < usable; ++i) {
        if (rowDay[i] >= firstHeld && rowDay[i] <= lastHeld) continue;
        const OriginRow &row = rows[keep[i]];
        memcpy(X + count * pm::kInputCount, row.x, pm::kInputCount * sizeof(double));
        y[count] = targets[keep[i]].rideMean - row.x[0];
        ++count;
      }
      hgb::Forest forest;
      fitStage = 8 + block;
      if (!trainer->fit(X, y, uint32_t(count), pm::kInputCount, params, forest, fitYield)) { folds = false; break; }
      for (size_t i = 0; i < usable; ++i)
        if (rowDay[i] >= firstHeld && rowDay[i] <= lastHeld) {
          const OriginRow &row = rows[keep[i]];
          oof[i] = row.x[0] + hgb::predict(forest, row.x);
        }
      forest.release();
    }
    if (folds) {
      for (size_t i = 0; i < usable; ++i) {
        const double anchor = std::max(oof[i], 0.0);
        next->z[i] = std::log((targets[keep[i]].rideMean + 5.0) / (anchor + 5.0));
      }
      std::sort(next->z, next->z + usable);
      next->zCount = uint32_t(usable);
    }
    free(targets); free(rowDay); free(days); free(oof); free(keep);
    if (!folds) break;
    next->cutoff = cutoff;
    next->fitted = uint32_t(time(nullptr));
    fitStage = 11;
    saveModel(*next);
    xSemaphoreTake(modelMutex, portMAX_DELAY);
    PointModel *old = model;
    model = next;
    xSemaphoreGive(modelMutex);
    if (old) { old->release(); delete old; }
    next = nullptr;
    ok = true;
  } while (false);
  free(rows); free(X); free(y); free(medians); free(labelEpochs); free(labelValues);
  if (next) { next->release(); delete next; }
  fitStage = 0;
  return ok;
}

// A private copy of the readings up to a clock, so a long fit never holds the collection lock.
pm::Reading *snapshotReadings(uint32_t through, size_t &count) {
  count = 0;
  pm::Reading *copy = static_cast<pm::Reading *>(ps_malloc(kMaxReadings * sizeof(pm::Reading)));
  if (!copy) return nullptr;
  xSemaphoreTake(mutex, portMAX_DELAY);
  const size_t n = pm::lowerBoundEpoch(readings, readingCount, through);
  memcpy(copy, readings, n * sizeof(pm::Reading));
  xSemaphoreGive(mutex);
  count = n;
  return copy;
}

void saveFile(const char *temporary, const char *path, const std::function<bool(File &)> &body) {
  File file = LittleFS.open(temporary, FILE_WRITE, true);
  if (!file) return;
  const bool ok = body(file);
  file.close();
  if (!ok) { LittleFS.remove(temporary); return; }
  LittleFS.remove(path);
  LittleFS.rename(temporary, path);
}

// The +90 arrival-change classifier: 28 sensor inputs at 5-minute clocks, all rebuilt
// from the readings, so it needs no stored rows of its own.
bool fitArrival(uint32_t cutoff) {
  constexpr size_t kRows = 28 * 288 + 1, kBuckets = 28 * 96 + 8;
  fitStage = 20;
  size_t count = 0;
  pm::Reading *rows = snapshotReadings(cutoff, count);
  double *X = static_cast<double *>(ps_malloc(kRows * arrival::kInputs * sizeof(double)));
  uint8_t *labels = static_cast<uint8_t *>(ps_malloc(kRows));
  double *issues = static_cast<double *>(ps_malloc(kRows * sizeof(double)));
  double *medians = static_cast<double *>(ps_malloc(kBuckets * sizeof(double)));
  double *labelEpochs = static_cast<double *>(ps_malloc(kMaxReadings * sizeof(double)));
  double *labelValues = static_cast<double *>(ps_malloc(kMaxReadings * sizeof(double)));
  pm::Window *w = static_cast<pm::Window *>(ps_malloc(sizeof(pm::Window)));
  ArrivalFit *next = static_cast<ArrivalFit *>(ps_malloc(sizeof(ArrivalFit)));
  bool ok = false;
  size_t used = 0;
  arrival::FitInfo info;
  do {
    if (!rows || !X || !labels || !issues || !medians || !labelEpochs || !labelValues || !w || !next) break;
    size_t labelCount = 0;
    for (size_t i = 0; i < count && labelCount < kMaxReadings; ++i) {
      const pm::Reading &r = rows[i];
      if (r.epoch < cutoff && r.visible <= cutoff && std::isfinite(r.pm) && r.pm >= 0) {
        labelEpochs[labelCount] = r.epoch;
        labelValues[labelCount++] = r.pm;
      }
    }
    const int64_t firstIssue = (int64_t(cutoff) - 28 * 86400 + 299) / 300 * 300;
    const double bucket0 = double(firstIssue / 900 * 900);
    const size_t buckets = std::min<size_t>(size_t((double(cutoff) - bucket0) / 900) + 1, kBuckets);
    double scratch[64];
    size_t cursor = 0;
    for (size_t k = 0; k < buckets; ++k)
      medians[k] = pm::bucketMedian(labelEpochs, labelValues, labelCount, bucket0 + k * 900.0, cursor, scratch);
    fitStage = 21;
    double all[pm::kSensorInputs + pm::kControls + pm::kTimeInputs];
    uint32_t pause = 0;
    for (int64_t issue = firstIssue; issue < int64_t(cutoff) - arrival::kEmbargoSeconds && used < kRows; issue += 300) {
      pm::selectWindow(rows, count, double(issue), *w);
      const bool fresh = pm::sensorInputs(*w, double(issue), all);
      arrival::Label label;
      if (fresh && std::isfinite(all[0]) && all[0] >= 0
          && arrival::arrivalLabel(medians, buckets, bucket0, issue, all[0], cutoff, label)) {
        memcpy(X + used * arrival::kInputs, all, arrival::kInputs * sizeof(double));
        labels[used] = uint8_t(label.cls);
        issues[used] = double(issue);
        ++used;
      }
      if (!(++pause % 32)) vTaskDelay(1);
    }
    if (!arrival::enoughHistory(issues, used)) break;
    free(rows); rows = nullptr;
    free(labelEpochs); labelEpochs = nullptr;
    free(labelValues); labelValues = nullptr;
    fitStage = 22;
    new (next) ArrivalFit();
    if (!arrival::fit(X, labels, uint32_t(used), next->model, &info, fitYield)) break;
    next->cutoff = cutoff;
    next->fitted = uint32_t(time(nullptr));
    fitStage = 23;
    saveFile("/m/arrival.tmp", "/m/arrival.bin", [&](File &file) {
      const uint32_t size = sizeof(ArrivalFit);
      return file.write(reinterpret_cast<const uint8_t *>(&kArrivalMagic), 4) == 4
          && file.write(reinterpret_cast<const uint8_t *>(&size), 4) == 4
          && file.write(reinterpret_cast<const uint8_t *>(next), size) == size;
    });
    xSemaphoreTake(modelMutex, portMAX_DELAY);
    ArrivalFit *old = arrivalFit;
    arrivalFit = next;
    xSemaphoreGive(modelMutex);
    free(old);
    next = nullptr;
    ok = true;
  } while (false);
  Serial.printf("{\"event\":\"arrival_fit\",\"ok\":%s,\"rows\":%lu,\"iterations\":%u,\"evaluations\":%u,\"converged\":%s}\n",
                ok ? "true" : "false", (unsigned long)used, info.iterations, info.evaluations, info.converged ? "true" : "false");
  free(rows); free(X); free(labels); free(issues); free(medians); free(labelEpochs); free(labelValues); free(w); free(next);
  fitStage = 0;
  return ok;
}

// One pooled two-hour model serves the Morning and Afternoon cards. Its inputs are the
// server's Morning inputs without the three CAMS columns, which the device does not fetch.
bool fitSession(uint32_t cutoff) {
  constexpr size_t kOrigins = ses::kLookbackDays * 48 + 2, kRows = kOrigins * ses::kLeadCount;
  constexpr size_t kBuckets = ses::kLookbackDays * 96 + 2;
  fitStage = 30;
  size_t count = 0;
  pm::Reading *rows = snapshotReadings(cutoff, count);
  double *X = static_cast<double *>(ps_malloc(kRows * ses::kInputs * sizeof(double)));
  double *y = static_cast<double *>(ps_malloc(kRows * sizeof(double)));
  uint32_t *days = static_cast<uint32_t *>(ps_malloc(kRows * sizeof(uint32_t)));
  double *medians = static_cast<double *>(ps_malloc(kBuckets * sizeof(double)));
  SessionRow *row = static_cast<SessionRow *>(ps_malloc(sizeof(SessionRow)));
  double *Z = nullptr;
  SessionFit *next = new SessionFit();
  bool ok = false;
  size_t used = 0;
  do {
    if (!rows || !X || !y || !days || !medians || !row) break;
    const uint32_t first = cutoff - ses::kLookbackDays * 86400;
    const size_t buckets = std::min<size_t>((cutoff - first) / 900 + 1, kBuckets);
    for (size_t k = 0; k < buckets; ++k) {
      medians[k] = ses::closedBucket(rows, count, double(first) + k * 900.0, &pm::Reading::pm);
      if (!(k % 64)) vTaskDelay(1);
    }
    fitStage = 31;
    const String firstDay = dayStamp(first) + ".bin";
    uint32_t lastOrigin = 0;
    for (const String &name : dayFiles("s")) {
      if (name < firstDay) continue;
      File file = LittleFS.open(String("/s/") + name, FILE_READ);
      if (!file) continue;
      while (file.read(reinterpret_cast<uint8_t *>(row), sizeof(SessionRow)) == sizeof(SessionRow)) {
        const uint32_t origin = uint32_t(row->origin);
        if (origin % 1800 || origin < first || origin >= cutoff || origin <= lastOrigin) continue;
        lastOrigin = origin;
        const size_t visible = pm::lowerBoundEpoch(rows, count, origin);
        const size_t originIndex = (origin - first) / 900;
        for (int lead = 0; lead < ses::kLeadCount && used < kRows; ++lead) {
          const double complete = ses::labelComplete(origin, ses::kLeads[lead]);
          if (!(complete < cutoff) || row->weather[lead][kSessionWeatherColumns] == 0) continue;
          const double label = ses::sessionLabel(medians, buckets, originIndex, ses::kLeads[lead]);
          if (!std::isfinite(label)) continue;
          double *x = X + used * ses::kInputs;
          const double start = double(origin) + ses::kLeads[lead] * 60.0;
          ses::sensorInputs(rows, visible, origin, start, start + 7200, x);
          if (!std::isfinite(x[ses::kFresh]) || !std::isfinite(x[ses::kClosedLevel])) continue;
          x[ses::kCamsCurrent] = x[ses::kCamsTarget] = x[ses::kCamsDelta] = ses::kNaN;
          memcpy(x + ses::kPriorRainMm, row->weather[lead], kSessionWeatherColumns * sizeof(double));
          y[used] = label - x[ses::kFresh];
          days[used] = (uint32_t(complete) + 8 * 3600) / 86400;
          ++used;
        }
        vTaskDelay(1);
      }
      file.close();
    }
    if (used < ses::kMinTrainingRows) break;
    int dayCount = 0;
    for (size_t i = 0; i < used; ++i) {
      bool seen = false;
      for (size_t j = 0; j < i && !seen; ++j) seen = days[j] == days[i];
      if (!seen) ++dayCount;
    }
    if (dayCount < ses::kMinTargetDays) break;
    free(rows); rows = nullptr;
    fitStage = 32;
    next->imputer.fit(X, used);
    const size_t outputs = next->imputer.outputs;
    Z = static_cast<double *>(ps_malloc(used * outputs * sizeof(double)));
    if (!Z) break;
    for (size_t i = 0; i < used; ++i) next->imputer.transform(X + i * ses::kInputs, Z + i * outputs);
    free(X); X = nullptr;
    hgb::Params params;
    params.maxIter = ses::kIterations;
    params.learningRate = ses::kLearningRate;
    params.maxLeaves = ses::kMaxLeaves;
    params.minSamplesLeaf = ses::kMinSamplesLeaf;
    params.l2 = ses::kL2;
    params.maxDepth = ses::kMaxDepth;
    fitStage = 33;
    if (!trainer->fit(Z, y, uint32_t(used), uint16_t(outputs), params, next->forest, fitYield)) break;
    next->cutoff = cutoff;
    next->fitted = uint32_t(time(nullptr));
    fitStage = 34;
    saveFile("/m/session.tmp", "/m/session.bin", [&](File &file) {
      const hgb::Forest &f = next->forest;
      const auto put = [&](const void *data, size_t bytes) { return file.write(static_cast<const uint8_t *>(data), bytes) == bytes; };
      return put(&kSessionMagic, 4) && put(&next->cutoff, 4) && put(&next->fitted, 4) && put(&next->imputer, sizeof(ses::Imputer))
          && put(&f.baseline, 8) && put(&f.treeCount, 2) && put(&f.nodeCount, 4)
          && put(f.rootOffsets, f.treeCount * sizeof(uint32_t)) && put(f.nodes, f.nodeCount * sizeof(hgb::Node));
    });
    xSemaphoreTake(modelMutex, portMAX_DELAY);
    SessionFit *old = sessionFit;
    sessionFit = next;
    xSemaphoreGive(modelMutex);
    if (old) { old->forest.release(); delete old; }
    next = nullptr;
    ok = true;
  } while (false);
  Serial.printf("{\"event\":\"session_fit\",\"ok\":%s,\"rows\":%lu}\n", ok ? "true" : "false", (unsigned long)used);
  free(rows); free(X); free(y); free(days); free(medians); free(row); free(Z);
  if (next) { next->forest.release(); delete next; }
  fitStage = 0;
  return ok;
}

ArrivalFit *loadArrival() {
  File file = LittleFS.open("/m/arrival.bin", FILE_READ);
  if (!file) return nullptr;
  uint32_t magic = 0, size = 0;
  ArrivalFit *m = static_cast<ArrivalFit *>(ps_malloc(sizeof(ArrivalFit)));
  const bool ok = m && file.read(reinterpret_cast<uint8_t *>(&magic), 4) == 4 && magic == kArrivalMagic
      && file.read(reinterpret_cast<uint8_t *>(&size), 4) == 4 && size == sizeof(ArrivalFit)
      && file.read(reinterpret_cast<uint8_t *>(m), size) == size;
  file.close();
  if (!ok) { free(m); return nullptr; }
  return m;
}

SessionFit *loadSession() {
  File file = LittleFS.open("/m/session.bin", FILE_READ);
  if (!file) return nullptr;
  SessionFit *m = new SessionFit();
  hgb::Forest &f = m->forest;
  const auto get = [&](void *data, size_t bytes) { return file.read(static_cast<uint8_t *>(data), bytes) == bytes; };
  uint32_t magic = 0;
  bool ok = get(&magic, 4) && magic == kSessionMagic && get(&m->cutoff, 4) && get(&m->fitted, 4) && get(&m->imputer, sizeof(ses::Imputer))
      && get(&f.baseline, 8) && get(&f.treeCount, 2) && get(&f.nodeCount, 4);
  if (ok) {
    f.rootOffsets = static_cast<uint32_t *>(ps_malloc(f.treeCount * sizeof(uint32_t)));
    f.nodes = static_cast<hgb::Node *>(ps_malloc(f.nodeCount * sizeof(hgb::Node)));
    ok = f.rootOffsets && f.nodes && get(f.rootOffsets, f.treeCount * sizeof(uint32_t)) && get(f.nodes, f.nodeCount * sizeof(hgb::Node));
  }
  file.close();
  if (!ok) { f.release(); delete m; return nullptr; }
  return m;
}

RTC_NOINIT_ATTR uint32_t restartedForCutoff;  // Survives a restart; power-on garbage cannot match a cutoff.

void fitTask(void *) {
  uint32_t nextAttempt = 0;
  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(20000));
    const uint32_t now = uint32_t(time(nullptr));
    if (!validClock(now)) continue;
    if (nextAttempt > now + 3600) nextAttempt = 0;  // The clock was stepped back.
    if (now < nextAttempt) continue;
    const uint32_t cutoff = midnightOf(now);
    if (now < cutoff + 300) continue;  // Just after midnight, as on the laptop.
    bool ok = true;
    if (!model || model->cutoff != cutoff) {
      const uint32_t started = millis();
      const bool done = fitModels(cutoff);
      lastFitMs = millis() - started;
      Serial.printf("{\"event\":\"fit\",\"ok\":%s,\"cutoff\":%lu,\"ms\":%lu}\n", done ? "true" : "false", (unsigned long)cutoff, (unsigned long)lastFitMs.load());
      ok = done;
    }
    if (!arrivalFit || arrivalFit->cutoff != cutoff) {
      const uint32_t started = millis();
      ok = fitArrival(cutoff) && ok;
      arrivalFitMs = millis() - started;
    }
    if (!sessionFit || sessionFit->cutoff != cutoff) {
      const uint32_t started = millis();
      ok = fitSession(cutoff) && ok;
      sessionFitMs = millis() - started;
    }
    if (!ok) {
      // Retries free what they take, so a fragmented PSRAM stays fragmented until a restart.
      if (restartedForCutoff != cutoff) {
        restartedForCutoff = cutoff;
        Serial.println("{\"event\":\"fit_failed_restart\"}");
        delay(100);
        ESP.restart();
      }
      nextAttempt = std::min(now + 3600, cutoff + 86400 + 300);  // Never past the next midnight's fit.
    }
  }
}

// ---------------------------------------------------------------- publishing the forecast
void addPoint(JsonObject out, double value) {
  const bool ok = std::isfinite(value);
  out["available"] = ok;
  out["role"] = "raw_model_output";
  out["calibrated"] = false;
  if (ok) out["pm25_ugm3"] = value; else out["pm25_ugm3"] = nullptr;
}

struct LastForecast {
  uint32_t issue = 0;
  bool fresh = false, hasWind = false;
  double x[pm::kInputCount];
  double point[kHeads];
  double chance = pm::kNaN;
  double arrivalP[arrival::kClasses];
  double session[2];
} *lastForecast = nullptr;

// The exact +90 change classes, in the field names ArrivalChange.h reads.
void addArrivalChange(JsonObject out, uint32_t issued, uint32_t referenceEpoch, double reference, const double *probability) {
  const arrival::Tails t = arrival::tails(probability);
  out["available"] = true;
  out["fall20"] = t.fall20;
  out["fall40"] = t.fall40;
  out["rise20"] = t.rise20;
  out["rise40"] = t.rise40;
  out["within20"] = t.within20;
  out["issued_epoch"] = issued;
  out["fresh_reference_epoch"] = referenceEpoch;
  out["reference_ugm3"] = reference;
  out["arrival_epoch"] = issued + 5400;
  out["model"] = "device_arrival_logistic_recall_f2_v2";
  const char *outcome = nullptr, *text = "Arrival outcome uncertain";
  double top = pm::kNaN;
  if (t.fall20 > t.rise20 && t.fall20 > t.within20) { outcome = "fall20"; text = "Fall on arrival"; top = t.fall20; }
  else if (t.rise20 > t.fall20 && t.rise20 > t.within20) { outcome = "rise20"; text = "Rise on arrival"; top = t.rise20; }
  else if (t.within20 > t.fall20 && t.within20 > t.rise20) { outcome = "within20"; text = "No change on arrival"; top = t.within20; }
  out["display_text"] = text;
  if (outcome) { out["outcome"] = outcome; out["outcome_probability"] = top; }
  else { out["outcome"] = nullptr; out["outcome_probability"] = nullptr; }
}

// The session model's two-hour mean for one window, or NaN.
double sessionPoint(uint32_t issued, uint32_t start, uint32_t end) {
  const WeatherRun *run = selectRun(issued);
  if (!run || !sessionRun || !start) return pm::kNaN;
  loadSessionRun(*run, *sessionRun);
  double x[ses::kInputs], z[ses::kMaxOutputs];
  if (!ses::inputRow(readings, readingCount, sessionRun, issued, start, end, x)) return pm::kNaN;
  double point = pm::kNaN;
  xSemaphoreTake(modelMutex, portMAX_DELAY);
  if (sessionFit && sessionFit->cutoff == midnightOf(issued)) {
    sessionFit->imputer.transform(x, z);
    point = std::max(0.0, x[ses::kFresh] + hgb::predict(sessionFit->forest, z));
  }
  xSemaphoreGive(modelMutex);
  return point;
}

double addSession(JsonObject out, int startHour, uint32_t now, uint32_t issued) {
  const uint32_t start = nextSession(now, startHour);
  out["issued_epoch"] = issued;
  out["start_epoch"] = start;
  out["end_epoch"] = start + 7200;
  const double point = sessionPoint(issued, start, start + 7200);
  JsonObject pmOut = out["pm"].to<JsonObject>();
  addPoint(pmOut, point);
  pmOut["model"] = "device_session_pooled_hgb_v1";
  addWindowWeather(out["weather"].to<JsonObject>(), now, issued, start, start + 7200);
  return point;
}

// Literal model outputs; anything missing stays unavailable.
void addForecast(JsonObject out, uint32_t now) {
  const pm::Reading *row = readingCount ? &readings[readingCount - 1] : nullptr;
  const bool sensorOk = row && now >= row->epoch && now - uint32_t(row->epoch) <= 900;
  const uint32_t issued = now;
  double point[kHeads] = {pm::kNaN, pm::kNaN, pm::kNaN, pm::kNaN};
  double chance = pm::kNaN;
  pm::Wind wind;
  static double x[pm::kInputCount];
  const bool hasWind = fillWind(issued, wind);
  const bool fresh = pm::inputRow(readings, readingCount, polls, pollCount, issued, hasWind ? &wind : nullptr, x, *window);
  const int spCount = int(x[pm::kSensorInputs + pm::kControls + pm::kTimeInputs + pm::kStationInputs]);
  xSemaphoreTake(modelMutex, portMAX_DELAY);
  if (model && model->cutoff == midnightOf(issued) && fresh && hasWind && spCount > 0) {
    for (int h = 0; h < kHeads; ++h) point[h] = x[0] + hgb::predict(model->heads[h], x);
    if (model->zCount && std::isfinite(point[1])) {
      const double threshold = std::log((70.0 + 5.0) / (std::max(point[1], 0.0) + 5.0));
      chance = double(std::upper_bound(model->z, model->z + model->zCount, threshold) - model->z) / double(model->zCount);
    }
  }
  xSemaphoreGive(modelMutex);
  // The newest finite reading behind the fresh reference.
  uint32_t referenceEpoch = 0;
  for (int i = window->count - 1; i >= 0; --i)
    if (std::isfinite(window->pm[i])) {
      if (window->epoch[i] > double(issued) - 300) referenceEpoch = uint32_t(window->epoch[i]);
      break;
    }
  double arrivalP[arrival::kClasses];
  for (double &p : arrivalP) p = pm::kNaN;
  bool arrivalOk = false;
  xSemaphoreTake(modelMutex, portMAX_DELAY);
  if (arrivalFit && arrivalFit->cutoff == midnightOf(issued) && fresh && referenceEpoch && std::isfinite(x[0]) && x[0] >= 0)
    arrivalOk = arrival::predict(arrivalFit->model, x, arrivalP);
  xSemaphoreGive(modelMutex);
  if (lastForecast) {
    xSemaphoreTake(mutex, portMAX_DELAY);
    lastForecast->issue = issued;
    lastForecast->fresh = fresh;
    lastForecast->hasWind = hasWind;
    memcpy(lastForecast->x, x, sizeof(x));
    memcpy(lastForecast->point, point, sizeof(point));
    lastForecast->chance = chance;
    memcpy(lastForecast->arrivalP, arrivalP, sizeof(arrivalP));
    xSemaphoreGive(mutex);
  }
  const bool anyPoint = std::isfinite(point[0]);
  out["issued_epoch"] = issued;
  JsonObject near = out["near90"].to<JsonObject>();
  addPoint(near, point[0]);
  near["target_epoch"] = issued + 5400;
  near["first20"]["available"] = false;
  JsonObject change = near["arrival_change"].to<JsonObject>();
  if (arrivalOk) addArrivalChange(change, issued, referenceEpoch, x[0], arrivalP);
  else change["available"] = false;
  JsonObject ride = out["ride90_210"].to<JsonObject>();
  addPoint(ride, point[1]);
  ride["start_epoch"] = issued + 5400;
  ride["end_epoch"] = issued + 12600;
  JsonObject extrema = ride["minimum_maximum"].to<JsonObject>();
  const bool extremaOk = anyPoint && std::isfinite(point[2]) && std::isfinite(point[3]);
  extrema["available"] = extremaOk;
  extrema["kind"] = "predicted_window_minimum_maximum";
  extrema["resolution_minutes"] = 15;
  if (extremaOk) { extrema["minimum_ugm3"] = point[2]; extrema["maximum_ugm3"] = point[3]; }
  JsonObject le70 = ride["mean_le70"].to<JsonObject>();
  le70["available"] = std::isfinite(chance);
  if (std::isfinite(chance)) le70["probability"] = chance;
  addWindowWeather(ride["weather"].to<JsonObject>(), now, issued, issued + 5400, issued + 12600);
  JsonObject sessions = out["sessions"].to<JsonObject>();
  const double morning = addSession(sessions["morning"].to<JsonObject>(), 9, now, issued);
  const double afternoon = addSession(sessions["afternoon"].to<JsonObject>(), 14, now, issued);
  if (lastForecast) {
    xSemaphoreTake(mutex, portMAX_DELAY);
    lastForecast->session[0] = morning;
    lastForecast->session[1] = afternoon;
    xSemaphoreGive(mutex);
  }
  const bool published = sensorOk && (anyPoint || arrivalOk || std::isfinite(morning) || std::isfinite(afternoon));
  out["available"] = published;
  out["fresh"] = published;
}

void publish(uint32_t now) {
  JsonDocument *doc = new JsonDocument();
  JsonObject root = doc->to<JsonObject>();
  root["schema_version"] = 1;
  root["generated_epoch"] = now;
  root["timezone"] = "Asia/Kuala_Lumpur";
  root["source"] = "device";
  addCurrent(root["current"].to<JsonObject>(), now);
  addWeatherNow(root["weather"].to<JsonObject>(), now);
  addTennis(root["tennis_morning"].to<JsonObject>(), now);
  addHistory(root["history"].to<JsonObject>(), now);
  addForecast(root["forecast"].to<JsonObject>(), now);
  xSemaphoreTake(mutex, portMAX_DELAY);
  delete pending;
  pending = doc;
  pendingReady = true;
  stats.lastIssueEpoch = now;
  xSemaphoreGive(mutex);
}

void loadWeatherFromFlash() {
  File file = LittleFS.open("/w/latest.json", FILE_READ);
  if (!file) return;
  const uint32_t fetched = uint32_t(file.getLastWrite());
  String body = file.readString();
  file.close();
  parseWeather(body, fetched);
}

void task(void *) {
  uint32_t nextNeighbour = 0, nextWeather = 0, nextPublish = 0, nextPrune = 0, lastOrigin = 0;
  for (;;) {
    stats.taskStackFree = uxTaskGetStackHighWaterMark(nullptr);
    const uint32_t now = uint32_t(time(nullptr));
    if (WiFi.status() != WL_CONNECTED || !validClock(now)) { vTaskDelay(pdMS_TO_TICKS(500)); continue; }
    if (refreshRequested.exchange(false)) nextTtdi = nextNeighbour = nextWeather = nextPublish = 0;
    // A deadline further away than its own interval came from a clock that has since been corrected.
    const auto clampDeadline = [now](uint32_t &next, uint32_t interval) { if (next > now + interval + 60) next = 0; };
    clampDeadline(nextTtdi, kTtdiSeconds);
    clampDeadline(nextNeighbour, kNeighbourSeconds);
    clampDeadline(nextWeather, kWeatherSeconds);
    clampDeadline(nextPublish, 60);
    clampDeadline(nextPrune, 3600);
    bool changed = false;
    if (now >= nextTtdi) changed |= pollTtdi();
    if (now >= nextNeighbour) { fetchNeighbours(); nextNeighbour = uint32_t(time(nullptr)) + kNeighbourSeconds; }
    if (now >= nextWeather) {
      const bool ok = fetchWeather();
      changed |= ok;
      nextWeather = uint32_t(time(nullptr)) + (ok ? kWeatherSeconds : 120);
    }
    const uint32_t clock = uint32_t(time(nullptr));
    const uint32_t origin = clock - clock % 900;
    if (origin != lastOrigin && clock >= origin + 20) {
      storeOriginRow(origin);
      if (origin % 1800 == 0) storeSessionRow(origin);
      lastOrigin = origin;
    }
    if (changed || clock >= nextPublish) { publish(clock); nextPublish = clock + 60; }
    if (clock >= nextPrune) {
      // By the newest record, not the clock, so one wrong time reply cannot delete history.
      const uint32_t newest = stats.lastReadingEpoch ? std::min(clock, stats.lastReadingEpoch) : 0;
      if (newest) {
        pruneFolder("r", kKeepReadingDays, newest);
        pruneFolder("p", kKeepPollDays, newest);
        pruneFolder("x", kKeepReadingDays, newest);
        pruneFolder("g", kKeepGasDays, newest);
        pruneFolder("s", kKeepSessionDays, newest);
      }
      stats.storageUsed = LittleFS.usedBytes();
      stats.storageTotal = LittleFS.totalBytes();
      nextPrune = clock + 3600;
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
}  // namespace

bool Engine::begin() {
  mutex = xSemaphoreCreateMutex();
  readings = static_cast<pm::Reading *>(ps_malloc(kMaxReadings * sizeof(pm::Reading)));
  polls = static_cast<pm::Poll *>(ps_malloc(kMaxPolls * sizeof(pm::Poll)));
  runs = new (ps_malloc(kWeatherRuns * sizeof(WeatherRun))) WeatherRun[kWeatherRuns];
  weather = runs;
  modelMutex = xSemaphoreCreateMutex();
  trainer = new (ps_malloc(sizeof(hgb::Trainer))) hgb::Trainer();
  window = static_cast<pm::Window *>(ps_malloc(sizeof(pm::Window)));
  gases = static_cast<Gas *>(ps_malloc(kMaxGas * sizeof(Gas)));
  samples = static_cast<obs::Sample *>(ps_malloc(kMaxSamples * sizeof(obs::Sample)));
  observeScratch = {static_cast<obs::Bucket3 *>(ps_malloc(obs::kDeviceThreeBuckets * sizeof(obs::Bucket3))), obs::kDeviceThreeBuckets,
                    static_cast<obs::Bucket15 *>(ps_malloc(obs::kDeviceFifteenBuckets * sizeof(obs::Bucket15))), obs::kDeviceFifteenBuckets};
  sessionRun = new (ps_malloc(sizeof(ses::Run))) ses::Run();
  if (!mutex || !modelMutex || !trainer || !readings || !polls || !runs || !window || !gases || !sessionRun) return false;
  for (double &end : lastPollEnd) end = -1;
  // A mount failure formats only after retries, and the count is reported in /status.
  // A first boot on a new storage area legitimately lands here once.
  bool mounted = false;
  for (int attempt = 0; attempt < 3 && !mounted; ++attempt) {
    mounted = LittleFS.begin(false, "/fs", 10, kFsLabel);
    if (!mounted) delay(250);
  }
  Preferences engineSettings;
  const bool settingsOpen = engineSettings.begin("engine", false);
  stats.storageFormats = settingsOpen ? engineSettings.getUInt("formats", 0) : 0;
  if (!mounted) {
    mounted = LittleFS.begin(true, "/fs", 10, kFsLabel);
    if (mounted && settingsOpen) engineSettings.putUInt("formats", ++stats.storageFormats);
  }
  if (settingsOpen) engineSettings.end();
  stats.storageReady = mounted;
  if (!stats.storageReady) return false;
  for (const char *folder : {"/r", "/p", "/x", "/g", "/s", "/w", "/m"}) LittleFS.mkdir(folder);
  stats.storageUsed = LittleFS.usedBytes();
  stats.storageTotal = LittleFS.totalBytes();
  const uint32_t now = uint32_t(time(nullptr));
  (void)now;
  readingCount = loadFolder("r", readings, kMaxReadings, String("0"));
  // A day file can overlap the previous boot only by append order; keep rows sorted and unique.
  std::stable_sort(readings, readings + readingCount, [](const pm::Reading &a, const pm::Reading &b) { return a.epoch < b.epoch; });
  size_t unique = 0;
  for (size_t i = 0; i < readingCount; ++i)
    if (!unique || readings[i].epoch != readings[unique - 1].epoch) readings[unique++] = readings[i];
  readingCount = unique;
  stats.readings = readingCount;
  if (readingCount) stats.lastReadingEpoch = uint32_t(readings[readingCount - 1].epoch);
  std::vector<String> pollDays = dayFiles("p");
  const String firstPollDay = pollDays.size() >= 2 ? pollDays[pollDays.size() - 2] : String("0");
  pollCount = loadFolder("p", polls, kMaxPolls, firstPollDay);
  std::stable_sort(polls, polls + pollCount, [](const pm::Poll &a, const pm::Poll &b) { return a.end < b.end; });
  for (size_t i = 0; i < pollCount; ++i) {
    const int32_t station = polls[i].station;
    if (station >= 0 && station < pm::kStations) lastPollEnd[station] = std::max(lastPollEnd[station], polls[i].end);
  }
  stats.polls = pollCount;
  std::vector<String> gasDays = dayFiles("g");
  const String firstGasDay = gasDays.size() >= 3 ? gasDays[gasDays.size() - 3] : String("0");
  gasCount = loadFolder("g", gases, kMaxGas, firstGasDay);
  std::stable_sort(gases, gases + gasCount, [](const Gas &a, const Gas &b) { return a.epoch < b.epoch; });
  loadWeatherFromFlash();
  model = loadModel();
  arrivalFit = loadArrival();
  sessionFit = loadSession();
  lastForecast = new (ps_malloc(sizeof(LastForecast))) LastForecast();
  return true;
}

void Engine::start() {
  xTaskCreatePinnedToCore(task, "engine", 24576, nullptr, 1, nullptr, 0);
  xTaskCreatePinnedToCore(fitTask, "fit", 16384, nullptr, 1, nullptr, 1);
}

bool Engine::takePayload(JsonDocument &doc) {
  xSemaphoreTake(mutex, portMAX_DELAY);
  const bool ready = pendingReady;
  if (ready) {
    doc = std::move(*pending);
    pendingReady = false;
  }
  xSemaphoreGive(mutex);
  return ready;
}

void Engine::refresh() { refreshRequested.store(true); }

String Engine::debugJson() const {
  JsonDocument doc;
  if (lastForecast) {
    xSemaphoreTake(mutex, portMAX_DELAY);
    doc["issue"] = lastForecast->issue;
    doc["fresh"] = lastForecast->fresh;
    doc["has_wind"] = lastForecast->hasWind;
    JsonArray x = doc["x"].to<JsonArray>();
    for (double value : lastForecast->x) { if (std::isfinite(value)) x.add(value); else x.add(nullptr); }
    JsonArray point = doc["point"].to<JsonArray>();
    for (double value : lastForecast->point) { if (std::isfinite(value)) point.add(value); else point.add(nullptr); }
    if (std::isfinite(lastForecast->chance)) doc["chance"] = lastForecast->chance;
    JsonArray arrivalOut = doc["arrival"].to<JsonArray>();
    for (double value : lastForecast->arrivalP) { if (std::isfinite(value)) arrivalOut.add(value); else arrivalOut.add(nullptr); }
    JsonArray sessionOut = doc["session"].to<JsonArray>();
    for (double value : lastForecast->session) { if (std::isfinite(value)) sessionOut.add(value); else sessionOut.add(nullptr); }
    xSemaphoreGive(mutex);
  }
  String out;
  serializeJson(doc, out);
  return out;
}

String Engine::historyJson(int hours) const {
  constexpr size_t kBins = 240;
  constexpr size_t kMaxGaps = 64;
  static int32_t *lowest = static_cast<int32_t *>(ps_malloc(kBins * sizeof(int32_t)));
  static int32_t *highest = static_cast<int32_t *>(ps_malloc(kBins * sizeof(int32_t)));
  if (!lowest || !highest) return "{\"error\":\"history_memory\"}";
  const uint32_t now = uint32_t(time(nullptr));
  const uint32_t span = uint32_t(hours) * 3600;
  const double from = double(now) - span;
  String out;
  out.reserve(24000);
  char text[96];
  const auto number = [&](double value) {
    if (std::isfinite(value)) { snprintf(text, sizeof(text), "%.1f", value); out += text; }
    else out += "null";
  };
  xSemaphoreTake(mutex, portMAX_DELAY);
  const size_t first = pm::lowerBoundEpoch(readings, readingCount, from);
  const size_t rows = readingCount - first;
  const double storedDays = readingCount ? (readings[readingCount - 1].epoch - readings[0].epoch) / 86400.0 : 0;
  snprintf(text, sizeof(text), "{\"hours\":%d,\"stored_days\":%.1f,\"source_rows\":%u,", hours, storedDays, unsigned(rows));
  out += text;
  out += "\"columns\":[\"epoch\",\"pm25_ugm3\",\"pm10_ugm3\",\"temperature_c\",\"heat_index_c\"],\"points\":[";
  const auto addRow = [&](size_t index, bool &comma) {
    const pm::Reading &r = readings[index];
    if (comma) out += ",";
    comma = true;
    snprintf(text, sizeof(text), "[%lu,", (unsigned long)r.epoch);
    out += text;
    number(r.pm); out += ","; number(r.pm10); out += ","; number(r.temperature); out += ","; number(r.heatIndex); out += "]";
  };
  bool comma = false;
  double sum = 0, low = INFINITY, high = -INFINITY;
  uint32_t samplesSeen = 0;
  for (size_t i = first; i < readingCount; ++i) {
    const double value = readings[i].pm;
    if (std::isfinite(value)) { sum += value; low = std::min(low, value); high = std::max(high, value); ++samplesSeen; }
  }
  if (rows <= 2 * kBins) {
    for (size_t i = first; i < readingCount; ++i) addRow(i, comma);
  } else {
    // Each time bin keeps its lowest and highest PM2.5, so a thinned spike stays visible.
    for (size_t b = 0; b < kBins; ++b) lowest[b] = highest[b] = -1;
    for (size_t i = first; i < readingCount; ++i) {
      const size_t b = std::min<size_t>(size_t((readings[i].epoch - from) * kBins / span), kBins - 1);
      if (lowest[b] < 0) lowest[b] = highest[b] = int32_t(i);
      const double value = readings[i].pm;
      if (!std::isfinite(value)) continue;
      if (!std::isfinite(readings[lowest[b]].pm) || value < readings[lowest[b]].pm) lowest[b] = int32_t(i);
      if (!std::isfinite(readings[highest[b]].pm) || value > readings[highest[b]].pm) highest[b] = int32_t(i);
    }
    for (size_t b = 0; b < kBins; ++b) {
      if (lowest[b] < 0) continue;
      const size_t a = std::min(lowest[b], highest[b]), c = std::max(lowest[b], highest[b]);
      addRow(a, comma);
      if (c != a) addRow(c, comma);
    }
  }
  out += "],\"gaps\":[";
  comma = false;
  size_t gapCount = 0;
  for (size_t i = first + 1; i < readingCount && gapCount < kMaxGaps; ++i) {
    if (readings[i].epoch - readings[i - 1].epoch <= 600) continue;
    snprintf(text, sizeof(text), "%s[%lu,%lu]", comma ? "," : "", (unsigned long)readings[i - 1].epoch, (unsigned long)readings[i].epoch);
    out += text;
    comma = true;
    ++gapCount;
  }
  xSemaphoreGive(mutex);
  out += "],\"summary\":{\"available\":";
  out += samplesSeen ? "true" : "false";
  snprintf(text, sizeof(text), ",\"sample_count\":%lu", (unsigned long)samplesSeen);
  out += text;
  if (samplesSeen) {
    out += ",\"average_ugm3\":"; number(sum / samplesSeen);
    out += ",\"lowest_ugm3\":"; number(low);
    out += ",\"highest_ugm3\":"; number(high);
  }
  out += "}}";
  return out;
}

bool Engine::collectionStalled(uint32_t limitMs) const {
  return fitStage.load() == 0 && uint32_t(millis() - stats.lastHttpsOkMs) > limitMs;
}

EngineStatus Engine::status() const {
  EngineStatus out = stats;
  xSemaphoreTake(modelMutex, portMAX_DELAY);
  out.modelCutoff = model ? model->cutoff : 0;
  out.modelFitted = model ? model->fitted : 0;
  out.arrivalCutoff = arrivalFit ? arrivalFit->cutoff : 0;
  out.sessionCutoff = sessionFit ? sessionFit->cutoff : 0;
  xSemaphoreGive(modelMutex);
  out.arrivalFitMs = arrivalFitMs.load();
  out.sessionFitMs = sessionFitMs.load();
  out.fitStage = fitStage.load();
  out.lastFitMs = lastFitMs.load();
  return out;
}
