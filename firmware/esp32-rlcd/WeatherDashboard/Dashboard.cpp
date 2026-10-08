#include "Dashboard.h"
#include "BatteryEstimate.h"
#include "HistoryGraphs.h"
#include "First20Event.h"
#include "ArrivalChange.h"
#include "DashboardTypography.h"
#include <time.h>

namespace {
using DashboardTypography::text;
using DashboardTypography::width;
String value(JsonVariantConst number, uint8_t decimals = 0) {
  if (number.isNull() || !number.is<float>()) return "--";
  float result = number.as<float>();
  return isfinite(result) ? String(result, static_cast<unsigned int>(decimals)) : "--";
}

String measured(float number, uint8_t decimals = 1) {
  return isfinite(number) ? String(number, static_cast<unsigned int>(decimals)) : "--";
}

String clockText(uint32_t epoch, const char *format = "%H:%M") {
  if (epoch < 1700000000) return "--:--";
  time_t stamp = epoch;
  struct tm local;
  localtime_r(&stamp, &local);
  char text[40];
  strftime(text, sizeof(text), format, &local);
  return text;
}

String windowText(JsonVariantConst start, JsonVariantConst end, bool dated = true) {
  if (start.isNull() || end.isNull()) return "Window unavailable";
  return clockText(start.as<uint32_t>(), dated ? "%a %H:%M" : "%H:%M") + "-" + clockText(end.as<uint32_t>());
}

uint32_t ageSeconds(uint32_t now, JsonVariantConst stamp) {
  uint32_t epoch = stamp.as<uint32_t>();
  return epoch && now >= epoch ? now - epoch : UINT32_MAX;
}

String ageText(uint32_t age) {
  if (age == UINT32_MAX) return "No reading";
  if (age < 60) return String(age) + "s ago";
  if (age < 3600) return String(age / 60) + "m ago";
  return String(age / 3600) + "h ago";
}

void pmReading(U8G2 &gfx, int x, int baseline, String number, const uint8_t *font, int width) {
  gfx.setFont(font);
  if (gfx.getStrWidth(number.c_str()) > width - 44) gfx.setFont(u8g2_font_helvB14_tf);
  if (gfx.getStrWidth(number.c_str()) > width - 44) number = "--";
  int numberWidth = gfx.getStrWidth(number.c_str());
  text(gfx, x, baseline, number, width - 44);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, x + numberWidth + 8, baseline - 3, "µg/m³", 36);
}

JsonVariantConst section(const DashboardContext &context, const char *key) {
  return (*context.weather)[key];
}

bool weatherAvailable(const DashboardContext &context) {
  JsonVariantConst weather = section(context, "weather");
  return weather["available"].as<bool>() && ageSeconds(dashboardNow(context), weather["fetched_epoch"]) <= 7200;
}

bool weatherFresh(const DashboardContext &context) {
  JsonVariantConst weather = section(context, "weather");
  return weather["fresh"].as<bool>() && ageSeconds(dashboardNow(context), weather["fetched_epoch"]) <= 1800;
}

bool windowWeatherAvailable(const DashboardContext &context, JsonVariantConst weather) {
  return dashboardForecastAvailable(context) && weather["available"].as<bool>()
      && ageSeconds(dashboardNow(context), weather["fetched_epoch"]) <= 7200
      && weather["fetched_epoch"].as<uint32_t>() <= section(context, "forecast")["issued_epoch"].as<uint32_t>()
      && windowWeatherCoverageAvailable(weather);
}

bool windowWeatherFresh(const DashboardContext &context, JsonVariantConst weather) {
  return weather["fresh"].as<bool>()
      && ageSeconds(dashboardNow(context), weather["fetched_epoch"]) <= 1800;
}

bool sessionAvailable(const DashboardContext &context, JsonVariantConst session) {
  return dashboardForecastAvailable(context) && forecastModelPointAvailable(session["pm"])
      && session["end_epoch"].as<uint32_t>() > dashboardNow(context);
}

void heading(U8G2 &gfx, const DashboardContext &context, const char *title) {
  const bool demo = context.weather && (*context.weather)["demo"].is<bool>()
      && (*context.weather)["demo"].as<bool>();
  if (demo) title = "DEMO WEATHER";
  gfx.setDrawColor(1);
  gfx.drawBox(0, 0, 400, 27);
  gfx.setDrawColor(0);
  gfx.setFont(u8g2_font_helvB12_tf);
  text(gfx, 10, 19, title, 195);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 213, 18, clockText(dashboardNow(context), "%a %d %b %H:%M"), 150);
  text(gfx, 375, 18, String(context.page + 1) + "/3", 25);
  gfx.setDrawColor(1);
}

int buttonIconWidth(U8G2 &gfx, const char *label) {
  gfx.setFont(u8g2_font_5x7_tf);
  int iconWidth = gfx.getStrWidth(label) + 6;
  gfx.setFont(u8g2_font_helvB08_tf);
  return iconWidth;
}

void buttonIcon(U8G2 &gfx, int x, const char *label) {
  int iconWidth = buttonIconWidth(gfx, label);
  gfx.drawRFrame(x, 283, iconWidth, 14, 2);
  gfx.setFont(u8g2_font_5x7_tf);
  gfx.drawStr(x + 3, 293, label);
  gfx.setFont(u8g2_font_helvB08_tf);
}

void footer(U8G2 &gfx, const DashboardContext &context) {
  gfx.drawHLine(8, 279, 384);
  gfx.setFont(u8g2_font_helvB08_tf);
  String signal = context.wifiConnected ? "WiFi: " + String(context.wifiRssi) + "dBm" : "WiFi: --";
  int signalWidth = width(gfx, signal.c_str());
  int signalX = (400 - signalWidth) / 2;
  buttonIcon(gfx, 8, "KEY");
  int labelX = 8 + buttonIconWidth(gfx, "KEY") + 6;
  text(gfx, labelX, 295, "Read Info", signalX - labelX - 8);
  text(gfx, signalX, 295, signal);
  if (context.voiceHint && context.voiceHint[0]) {
    String acknowledgement = context.voiceHint;
    int availableWidth = 392 - signalX - signalWidth - 8;
    // Fit before positioning so even a long acknowledgement stays right
    // aligned and leaves a clear gap beside the centered Wi-Fi signal.
    while (acknowledgement.length() && width(gfx, acknowledgement.c_str()) > availableWidth) {
      unsigned int last = acknowledgement.length() - 1;
      while (last && (uint8_t(acknowledgement[last]) & 0xc0) == 0x80) --last;
      acknowledgement.remove(last);
    }
    text(gfx, 392 - width(gfx, acknowledgement.c_str()), 295, acknowledgement, availableWidth);
    return;
  }
  int bootWidth = buttonIconWidth(gfx, "BOOT");
  int nextWidth = width(gfx, "Next Page");
  int bootX = 392 - nextWidth - 6 - bootWidth;
  buttonIcon(gfx, bootX, "BOOT");
  text(gfx, bootX + bootWidth + 6, 295, "Next Page", nextWidth);
}

void currentStatus(U8G2 &gfx, const String &status) {
  constexpr int maxWidth = 187;
  if (width(gfx, status.c_str()) <= maxWidth) {
    text(gfx, 12, 111, status, maxWidth);
    return;
  }
  // A break at an ASCII space preserves complete UTF-8 words and the full
  // server label. Both baselines stay below the PM number and above the rule.
  int split = status.lastIndexOf(' ');
  while (split > 0) {
    const String first = status.substring(0, split);
    const String second = status.substring(split + 1);
    if (width(gfx, first.c_str()) <= maxWidth
        && width(gfx, second.c_str()) <= maxWidth) {
      text(gfx, 12, 105, first, maxWidth);
      text(gfx, 12, 118, second, maxWidth);
      return;
    }
    split = status.lastIndexOf(' ', split - 1);
  }
  // Do not turn an unexpectedly oversized label into a different, clipped
  // status. Known web labels fit the two-line area at the existing bold font.
  text(gfx, 12, 111, "Status unavailable", maxWidth);
}

void overview(U8G2 &gfx, const DashboardContext &context) {
  heading(gfx, context, "TTDI WEATHER");
  JsonVariantConst current = section(context, "current");
  JsonVariantConst weather = section(context, "weather");
  JsonVariantConst forecast = section(context, "forecast");
  JsonVariantConst near = forecast["near90"];
  const IndoorReading &indoor = *context.indoor;
  bool currentOk = dashboardCurrentAvailable(context);
  bool forecastOk = dashboardForecastAvailable(context);
  bool hourlyOk = weatherAvailable(context);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 12, 47, "OUTDOOR PM2.5");
  text(gfx, 221, 47, "INDOOR");
  gfx.drawVLine(207, 35, 82);
  pmReading(gfx, 12, 88, currentOk ? value(current["pm25_ugm3"], 1) : "--", u8g2_font_helvB24_tf, 187);
  gfx.setFont(u8g2_font_helvB24_tf);
  text(gfx, 220, 88, indoor.valid ? measured(indoor.temperatureC) + " C" : "-- C", 173);
  gfx.setFont(u8g2_font_helvB08_tf);
  currentStatus(gfx, dashboardCurrentStatus(context));
  gfx.setFont(u8g2_font_helvB12_tf);
  text(gfx, 221, 112, indoor.valid ? "RH " + measured(indoor.humidityPct) + "%" : "Sensor unavailable", 174);
  gfx.drawHLine(8, 123, 384);
  gfx.setFont(u8g2_font_helvB08_tf);
  String outdoor = currentOk ? "OUT " + value(current["temperature_c"], 1) + " C  " + value(current["humidity_pct"]) + "% RH" : "Outdoor observations unavailable";
  if (hourlyOk) outdoor += " | Wind fcst " + value(weather["current_hour"]["wind_kmh"]) + "km/h";
  text(gfx, 12, 141, outdoor);
  text(gfx, 12, 162, hourlyOk ? weatherFresh(context) ? "RAIN FCST" : "RAIN OLD" : "RAIN --", 80);
  for (int i = 0; i < 3; i++) {
    JsonVariantConst hour = weather["next_hours"][i];
    String label = hourlyOk && !hour.isNull() ? clockText(hour["rain_start_epoch"], "%H") + "-" + clockText(hour["rain_end_epoch"], "%H") + "h " + value(hour["rain_chance_pct"]) + "%" : "--";
    text(gfx, 100 + i * 100, 162, label, 95);
  }
  gfx.drawHLine(8, 175, 384);
  gfx.drawVLine(207, 184, 78);
  gfx.setFont(u8g2_font_helvB10_tf);
  text(gfx, 12, 195, "MTB  +90 MIN", 187);
  bool nearWindowOk = forecastOk && near["target_epoch"].is<uint32_t>()
      && near["target_epoch"].as<uint32_t>() > dashboardNow(context);
  bool nearOk = nearWindowOk && forecastModelPointAvailable(near);
  gfx.setFont(u8g2_font_helvB08_tf);
  bool pmFresh = forecast["fresh"].as<bool>() && ageSeconds(dashboardNow(context), forecast["issued_epoch"]) <= 120;
  text(gfx, 12, 214, nearWindowOk ? clockText(near["target_epoch"], "%a %H:%M") + (pmFresh ? "" : " | OLD") : "Forecast unavailable", 187);
  pmReading(gfx, 12, 242, nearOk ? value(near["pm25_ugm3"], 1) : "--", u8g2_font_helvB18_tf, 187);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 12, 258, dashboardArrivalStatus(context), 187);

  // This is a separate 07:00-09:00 target. Never relabel the MTB morning
  // outlook's 09:00-11:00 weather or particle estimates as tennis data.
  JsonVariantConst tennis = section(context, "tennis_morning");
  gfx.setFont(u8g2_font_helvB10_tf);
  text(gfx, 221, 195, "TENNIS  MORNING", 175);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 221, 214, windowText(tennis["start_epoch"], tennis["end_epoch"]), 175);
  JsonVariantConst tw = tennis;
  bool tennisWeather = tw["available"].as<bool>()
      && ageSeconds(dashboardNow(context), tw["fetched_epoch"]) <= 7200
      && tennis["end_epoch"].as<uint32_t>() > dashboardNow(context);
  gfx.setFont(u8g2_font_helvB14_tf);
  text(gfx, 221, 237, tennisWeather ? String(windowWeatherFresh(context, tw) ? "Rain " : "OLD rain ") + value(tw["rain_chance_max_pct"]) + "%" : "Rain --", 175);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 221, 258, tennisWeather ? "Feels " + value(tw["feels_like_max_c"], 1) + "C | Wind " + value(tw["wind_mean_kmh"]) : "Weather unavailable", 175);
  footer(gfx, context);
}

void sportSession(U8G2 &gfx, const DashboardContext &context, const char *name, int x, const char *label) {
  JsonVariantConst session = section(context, "forecast")["sessions"][name];
  JsonVariantConst weather = session["weather"];
  bool pmOk = sessionAvailable(context, session);
  bool weatherOk = windowWeatherAvailable(context, weather) && session["end_epoch"].as<uint32_t>() > dashboardNow(context);
  gfx.setFont(u8g2_font_helvB12_tf);
  text(gfx, x, 168, label, 179);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, x, 187, windowText(session["start_epoch"], session["end_epoch"]), 179);
  pmReading(gfx, x, 214, pmOk ? value(session["pm"]["pm25_ugm3"], 1) : "--", u8g2_font_helvB18_tf, 179);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, x, 232, weatherOk ? String(windowWeatherFresh(context, weather) ? "Rain " : "OLD rain ") + value(weather["rain_chance_max_pct"]) + "% | " + value(weather["rain_mm"], 1) + "mm" : "Rain unavailable", 179);
  text(gfx, x, 248, weatherOk ? "Feels " + value(weather["feels_like_max_c"], 1) + "C | RH " + value(weather["humidity_mean_pct"]) + "%" : "Weather unavailable", 179);
  text(gfx, x, 264, weatherOk ? "Wind " + value(weather["wind_mean_kmh"]) + " | Gust " + value(weather["gust_max_kmh"]) + "km/h" : "Wind unavailable", 179);
}

void sports(U8G2 &gfx, const DashboardContext &context) {
  heading(gfx, context, "MTB Ride Forecast");
  JsonVariantConst forecast = section(context, "forecast");
  JsonVariantConst ride = forecast["ride90_210"];
  bool windowOk = dashboardRideWindowAvailable(context);
  bool rideOk = windowOk && forecastModelPointAvailable(ride);
  const RidePmRange range = dashboardRidePmRange(context);
  uint32_t age = ageSeconds(dashboardNow(context), forecast["issued_epoch"]);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 12, 44, "Issued " + clockText(forecast["issued_epoch"]) + " | " + ageText(age) + (age > 120 || !forecast["fresh"].as<bool>() ? " | OLD" : ""));
  gfx.setFont(u8g2_font_helvB12_tf);
  text(gfx, 12, 66, "RIDE +90 TO +210 MIN");
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 12, 84, windowOk ? windowText(ride["start_epoch"], ride["end_epoch"])
      + (!range.available && rideOk ? " | MEAN" : "") : "Forecast unavailable", 230);
  String rideRange = range.available ? measured(range.low) + "-" + measured(range.high)
      : rideOk ? value(ride["pm25_ugm3"], 1) : "--";
  pmReading(gfx, 12, 118, rideRange, u8g2_font_helvB24_tf, 220);
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 252, 85, "P(MEAN <=70)", 137);
  gfx.setFont(u8g2_font_helvB24_tf);
  float chance = ride["mean_le70"]["probability"].is<float>()
      ? ride["mean_le70"]["probability"].as<float>() : NAN;
  bool chanceOk = windowOk && ride["mean_le70"]["available"].as<bool>()
      && isfinite(chance) && chance >= 0 && chance <= 1;
  text(gfx, 252, 118, chanceOk ? String(ride["mean_le70"]["probability"].as<float>() * 100, 1) + "%" : "--%", 137);
  JsonVariantConst rw = ride["weather"];
  gfx.setFont(u8g2_font_helvB08_tf);
  bool weatherOk = windowOk && windowWeatherAvailable(context, rw);
  text(gfx, 12, 138, weatherOk ? String(windowWeatherFresh(context, rw) ? "Rain " : "OLD rain ") + value(rw["rain_chance_max_pct"]) + "% | Feels " + value(rw["feels_like_max_c"], 1) + "C | Gust " + value(rw["gust_max_kmh"]) + "km/h" : "Ride weather unavailable");
  gfx.drawHLine(8, 147, 384);
  gfx.drawVLine(207, 154, 107);
  sportSession(gfx, context, "morning", 12, "MORNING");
  sportSession(gfx, context, "afternoon", 221, "AFTERNOON");
  footer(gfx, context);
}

float observationNumber(JsonVariantConst number) {
  return !number.isNull() && number.is<float>() ? number.as<float>() : NAN;
}

void historyPage(U8G2 &gfx, const DashboardContext &context) {
  heading(gfx, context, "OUTDOOR HISTORY");
  JsonVariantConst history = section(context, "history");
  JsonArrayConst columns = history["columns"];
  bool expectedColumns = columns.size() == 4 && columns[0] == "epoch"
      && columns[1] == "pm25_ugm3" && columns[2] == "temperature_c" && columns[3] == "heat_index_c";
  HistoryPoint points[HISTORY_GRAPH_POINT_LIMIT];
  HistoryGap gaps[HISTORY_GRAPH_GAP_LIMIT];
  HistorySeries series;
  series.points = points;
  series.gaps = gaps;
  series.sourceGapsKnown = history["gap_kind"] == "source_collection_before_downsampling"
      && history["gaps"].is<JsonArrayConst>();
  series.startEpoch = history["start_epoch"].as<uint32_t>();
  series.endEpoch = history["end_epoch"].as<uint32_t>();
  series.nowEpoch = dashboardNow(context);
  series.gapThresholdSeconds = history["gap_after_s"].as<uint32_t>();
  if (!series.gapThresholdSeconds) series.gapThresholdSeconds = 600;
  series.available = expectedColumns && history["available"].as<bool>();
  series.fresh = history["fresh"].as<bool>();
  series.pmSummary = dashboardHistoryPmSummary(context);
  if (series.available) {
    for (JsonArrayConst point : history["points"].as<JsonArrayConst>()) {
      if (series.count == HISTORY_GRAPH_POINT_LIMIT) break;
      if (point.size() != 4) continue;
      points[series.count++] = {point[0].as<uint32_t>(), observationNumber(point[1]),
          observationNumber(point[2]), observationNumber(point[3])};
    }
    for (JsonArrayConst gap : history["gaps"].as<JsonArrayConst>()) {
      if (series.gapCount == HISTORY_GRAPH_GAP_LIMIT) break;
      if (gap.size() != 2) continue;
      gaps[series.gapCount++] = {gap[0].as<uint32_t>(), gap[1].as<uint32_t>()};
    }
  }
  drawHistoryGraphs(gfx, series);
  gfx.drawHLine(8, 220, 384);
  const IndoorReading &indoor = *context.indoor;
  int percent = batteryPercentEstimate(context.batteryV);
  gfx.setFont(u8g2_font_helvB10_tf);
  text(gfx, 12, 237, "Battery " + String(context.batteryV, 2) + " V  "
      + (percent >= 0 ? "~" + String(percent) + "%" : String("--%")) + " (voltage estimate)");
  gfx.setFont(u8g2_font_helvB08_tf);
  text(gfx, 12, 252, "Indoor T " + (indoor.valid ? measured(indoor.temperatureC) : String("--"))
      + "C | Range " + measured(indoor.minimumC) + "-" + measured(indoor.maximumC) + "C");
  text(gfx, 12, 271, "RH " + (indoor.valid ? measured(indoor.humidityPct, 0) : String("--"))
      + "% | " + measured(indoor.minimumHumidity, 0) + "-" + measured(indoor.maximumHumidity, 0) + "% since restart");
  footer(gfx, context);
}
}

uint32_t dashboardNow(const DashboardContext &context) {
  uint32_t clock = time(nullptr);
  uint32_t generated = (*context.weather)["generated_epoch"].as<uint32_t>();
  uint32_t estimate = generated ? generated + uint32_t(millis() - context.weatherReceivedMs) / 1000 : 0;
  return clock > 1700000000 ? max(clock, estimate) : estimate;
}

bool dashboardCurrentAvailable(const DashboardContext &context) {
  JsonVariantConst current = (*context.weather)["current"];
  return current["available"].as<bool>() && ageSeconds(dashboardNow(context), current["observed_epoch"]) <= 900;
}

bool dashboardForecastAvailable(const DashboardContext &context) {
  JsonVariantConst forecast = (*context.weather)["forecast"];
  return forecast["available"].is<bool>() && forecast["available"].as<bool>()
      && forecast["issued_epoch"].is<uint32_t>()
      && forecast["issued_epoch"].as<uint32_t>() >= 1700000000
      && ageSeconds(dashboardNow(context), forecast["issued_epoch"]) <= 600;
}

bool dashboardRideWindowAvailable(const DashboardContext &context) {
  const JsonVariantConst ride = (*context.weather)["forecast"]["ride90_210"];
  const uint32_t start = ride["start_epoch"].as<uint32_t>();
  const uint32_t end = ride["end_epoch"].as<uint32_t>();
  return dashboardForecastAvailable(context) && ride["start_epoch"].is<uint32_t>()
      && ride["end_epoch"].is<uint32_t>() && start >= 1700000000
      && end > start && end - start == 7200 && end > dashboardNow(context);
}

RidePmRange dashboardRidePmRange(const DashboardContext &context) {
  return dashboardRideWindowAvailable(context)
      ? ridePmRange((*context.weather)["forecast"]["ride90_210"]) : RidePmRange{};
}

HistoryPmSummary dashboardHistoryPmSummary(const DashboardContext &context) {
  HistoryPmSummary summary;
  JsonVariantConst history = (*context.weather)["history"];
  JsonVariantConst source = history["pm25_summary"];
  if (!history["available"].as<bool>() || !source["available"].as<bool>()
      || !source["average_ugm3"].is<float>() || !source["lowest_ugm3"].is<float>()
      || !source["highest_ugm3"].is<float>() || !source["sample_count"].is<uint32_t>()) return summary;
  summary.average = source["average_ugm3"].as<float>();
  summary.lowest = source["lowest_ugm3"].as<float>();
  summary.highest = source["highest_ugm3"].as<float>();
  summary.sampleCount = source["sample_count"].as<uint32_t>();
  summary.available = summary.sampleCount > 0 && isfinite(summary.average)
      && isfinite(summary.lowest) && isfinite(summary.highest) && summary.lowest >= 0
      && summary.lowest <= summary.average && summary.average <= summary.highest;
  if (!summary.available) return HistoryPmSummary();
  return summary;
}

String dashboardCurrentStatus(const DashboardContext &context) {
  const JsonVariantConst current = (*context.weather)["current"];
  if (!dashboardCurrentAvailable(context) || !current["observed_epoch"].is<uint32_t>()
      || !forecastPmNumberValid(current["pm25_ugm3"])) return "Unavailable";
  const uint32_t age = ageSeconds(dashboardNow(context), current["observed_epoch"]);
  const bool fresh = current["fresh"].is<bool>() && current["fresh"].as<bool>() && age <= 420;
  if (!current["display_text"].is<const char *>()) return fresh ? "Status unavailable" : "OLD";
  const char *headline = current["display_text"].as<const char *>();
  if (!headline || !headline[0]) return fresh ? "Status unavailable" : "OLD";
  return (fresh ? String("") : String("OLD: ")) + headline;
}

String dashboardArrivalStatus(const DashboardContext &context) {
  if (!dashboardForecastAvailable(context)) return "Arrival unavailable";
  const JsonVariantConst forecast = (*context.weather)["forecast"];
  const ArrivalChangeResult arrival = decodeArrivalChange(forecast["near90"],
      forecast["issued_epoch"].as<uint32_t>(), dashboardNow(context));
  if (!arrival.available) return "Arrival unavailable";
  if (arrival.outcome == ArrivalChangeOutcome::Uncertain) return String(arrival.displayText);
  // The original probability owns the result; round only its visible percent.
  // The date row already carries OLD, leaving room for the complete headline.
  return String(arrival.displayText) + " " + String(arrival.probability * 100, 1) + "%";
}

String dashboardFirst20Status(const DashboardContext &context) {
  JsonVariantConst forecast = (*context.weather)["forecast"];
  JsonVariantConst near = forecast["near90"];
  JsonVariantConst event = near["first20"];
  if (!dashboardForecastAvailable(context)
      || near["target_epoch"].as<uint32_t>() <= dashboardNow(context)
      || !near["target_epoch"].is<uint32_t>()) return "Event unavailable";
  if (!event["available"].is<bool>() || !event["available"].as<bool>()
      || !event["display_text"].is<const char *>()) return "Event unavailable";
  const char *headline = event["display_text"].as<const char *>();
  if (!headline || !headline[0]) return "Event unavailable";
  bool fresh = forecast["fresh"].as<bool>()
      && ageSeconds(dashboardNow(context), forecast["issued_epoch"]) <= 120;
  return (fresh ? String("") : String("OLD: ")) + headline;
}

void drawDashboard(U8G2 &gfx, const DashboardContext &context) {
  gfx.clearBuffer();
  gfx.setDrawColor(1);
  if (context.page == 1) sports(gfx, context);
  else if (context.page == 2) historyPage(gfx, context);
  else overview(gfx, context);
  gfx.sendBuffer();
}
