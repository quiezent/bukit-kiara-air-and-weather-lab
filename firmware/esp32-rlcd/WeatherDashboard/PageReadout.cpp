#include "PageReadout.h"
#include "BatteryEstimate.h"
#include "ForecastSemantics.h"
#include "ArrivalChange.h"
#include "ObservedStatus.h"

namespace {
constexpr uint32_t kValidEpoch = 1700000000;

uint32_t age(uint32_t now, JsonVariantConst stamp) {
  const uint32_t epoch = stamp.as<uint32_t>();
  return epoch && now >= epoch ? now - epoch : UINT32_MAX;
}

float number(JsonVariantConst value) {
  return !value.isNull() && value.is<float>() ? value.as<float>() : NAN;
}

bool weatherAvailable(uint32_t now, JsonVariantConst weather) {
  return weather["available"].as<bool>() && age(now, weather["fetched_epoch"]) <= 7200;
}

bool weatherOld(uint32_t now, JsonVariantConst weather) {
  return !weather["fresh"].as<bool>() || age(now, weather["fetched_epoch"]) > 1800;
}

bool forecastFresh(uint32_t now, JsonVariantConst forecast) {
  return forecast["fresh"].as<bool>() && age(now, forecast["issued_epoch"]) <= 120;
}

unsigned rainPrecision(float rain) {
  return std::isfinite(rain) && rain == std::round(rain) ? 0 : 1;
}

uint32_t epoch(JsonVariantConst value) {
  return value.is<uint32_t>() && value.as<uint32_t>() >= kValidEpoch
      ? value.as<uint32_t>() : 0;
}

bool validWindow(JsonVariantConst window, uint32_t now) {
  const uint32_t start = epoch(window["start_epoch"]);
  const uint32_t end = epoch(window["end_epoch"]);
  return start && end > start && end - start == 7200 && end > now;
}

JsonVariantConst rainHour(JsonVariantConst weather, uint32_t target) {
  const auto contains = [target](JsonVariantConst hour) {
    const uint32_t start = epoch(hour["rain_start_epoch"]);
    const uint32_t end = epoch(hour["rain_end_epoch"]);
    return start && end > start && end - start == 3600 && start <= target && target < end;
  };
  JsonVariantConst current = weather["current_hour"];
  if (contains(current)) return current;
  for (JsonVariantConst hour : weather["next_hours"].as<JsonArrayConst>()) {
    if (contains(hour)) return hour;
  }
  return JsonVariantConst();
}

// A page is captured into immutable clip IDs and text before the playback task
// starts. On overflow the caller receives neither a partial playlist nor text.
class Readout {
 public:
  Readout(SpeechPlaylist &playlist, String &transcript) : out(playlist), text(transcript) {}

  bool phrase(SpeechClip clip, const char *words) {
    if (!out.add(clip)) return false;
    text += words;
    text += ' ';
    return true;
  }

  void sentence() {
    text.trim();
    text += ". ";
  }

  bool announcement(SpeechClip clip, const char *words) {
    if (!phrase(clip, words)) return false;
    sentence();
    return true;
  }

  bool unavailable() { return phrase(SpeechClip::UnavailableValue, "unavailable"); }

  bool value(float amount, unsigned decimals) {
    // Quantize through the same formatter as the screen before composing audio.
    // This also keeps the spoken value and transcript identical at rounding ties.
    String displayed(amount, decimals);
    displayed.trim();
    if (!speechNumberPrecision(out, displayed.toFloat(), decimals)) return false;
    text += displayed;
    text += ' ';
    return true;
  }

  bool measurement(SpeechClip label, const char *words, float amount,
                   SpeechClip unit, const char *unitWords, unsigned decimals,
                   float minimum, float maximum, bool available = true) {
    if (!phrase(label, words)) return false;
    if (!available || !readoutValueValid(amount, minimum, maximum)) {
      if (!unavailable()) return false;
    } else if (!value(amount, decimals) || !phrase(unit, unitWords)) return false;
    sentence();
    return true;
  }

  bool labelledValue(SpeechClip label, const char *words, float amount,
                     unsigned decimals, float minimum, float maximum,
                     bool available = true) {
    return phrase(label, words) && (available && readoutValueValid(amount, minimum, maximum)
        ? value(amount, decimals) : unavailable());
  }

  bool labelledSentence(SpeechClip label, const char *words, float amount,
                        unsigned decimals, float minimum, float maximum,
                        bool available = true) {
    if (!labelledValue(label, words, amount, decimals, minimum, maximum, available)) return false;
    sentence();
    return true;
  }

  void separator() {
    text.trim();
    text += ", ";
  }

  bool range(SpeechClip label, const char *words, float low, float high, unsigned decimals,
             float minimum, float maximum, bool available = true) {
    if (!phrase(label, words)) return false;
    if (!available || !readoutValueValid(low, minimum, maximum)
        || !readoutValueValid(high, minimum, maximum) || high < low) {
      if (!unavailable()) return false;
    } else if (!value(low, decimals) || !phrase(SpeechClip::Until, "to")
        || !value(high, decimals)) return false;
    sentence();
    return true;
  }

 private:
  SpeechPlaylist &out;
  String &text;
};

bool overview(Readout &out, const DashboardContext &context, uint32_t now) {
  const JsonVariantConst current = (*context.weather)["current"];
  const JsonVariantConst weather = (*context.weather)["weather"];
  const JsonVariantConst forecast = (*context.weather)["forecast"];
  const JsonVariantConst near = forecast["near90"];
  const bool currentOk = dashboardCurrentAvailable(context);
  const bool currentOld = !current["fresh"].is<bool>() || !current["fresh"].as<bool>()
      || age(now, current["observed_epoch"]) > 420;
  const bool weatherOk = weatherAvailable(now, weather);
  if (currentOk && currentOld && !out.announcement(SpeechClip::CurrentDataOld, "Outdoor data is old")) return false;
  if (!out.labelledSentence(SpeechClip::CurrentPM25, "Current outdoor PM2.5", number(current["pm25_ugm3"]),
      1, 0, 9999.9f, currentOk)) return false;
  // Match the screen's source-clock and concentration eligibility. The finite
  // phrase captures both words and clip IDs now, before the playback task starts.
  if (currentOk && current["observed_epoch"].is<uint32_t>()
      && forecastPmNumberValid(current["pm25_ugm3"])) {
    const ObservedStatusPhrase status = observedStatusPhrase(current["display_text"]);
    // The server prefix still identifies exact labels, but only the status
    // body is spoken; its matching recording is already self-contained.
    if (!out.announcement(status.clip, status.words)) return false;
  }
  if (!out.measurement(SpeechClip::OutdoorTemperature, "Outdoor temperature", number(current["temperature_c"]),
      SpeechClip::DegreesCelsius, "degrees Celsius", 1, -100, 100, currentOk)) return false;
  if (weatherOk && weatherOld(now, weather)
      && !out.announcement(SpeechClip::ForecastOld, "Weather forecast is old")) return false;
  const JsonVariantConst hour = rainHour(weather, now);
  if (!out.measurement(SpeechClip::RainChance, "Rain chance", number(hour["rain_chance_pct"]),
      SpeechClip::Percent, "percent", 0, 0, 100, weatherOk && !hour.isNull())) return false;
  const bool forecastOk = dashboardForecastAvailable(context);
  const bool nearWindowOk = forecastOk && epoch(near["target_epoch"]) > now;
  const bool nearModelOk = nearWindowOk && forecastModelPointAvailable(near);
  if (forecastOk && !forecastFresh(now, forecast)
      && !out.announcement(SpeechClip::Old, "Data is old")) return false;
  if (!out.labelledSentence(SpeechClip::Near90Forecast, "Ninety minute forecast PM2.5", number(near["pm25_ugm3"]),
      1, 0, 9999.9f, nearModelOk)) return false;
  const ArrivalChangeResult arrival = forecastOk
      ? decodeArrivalChange(near, epoch(forecast["issued_epoch"]), now) : ArrivalChangeResult{};
  if (arrival.available && (arrival.outcome == ArrivalChangeOutcome::Within20
      || arrival.outcome == ArrivalChangeOutcome::Rise20 || arrival.outcome == ArrivalChangeOutcome::Fall20)) {
    const bool rise = arrival.outcome == ArrivalChangeOutcome::Rise20;
    const bool fall = arrival.outcome == ArrivalChangeOutcome::Fall20;
    const SpeechClip label = rise ? SpeechClip::RiseOnArrival
        : fall ? SpeechClip::FallOnArrival : SpeechClip::NoChangeOnArrival;
    const char *words = rise ? "Rise on arrival" : fall ? "Fall on arrival" : "No change on arrival";
    // The native endpoint winner is independent of first20 crossings and the
    // primary concentration point. Capture its validated score without using
    // the current sensor value to recalculate an issued model probability.
    const float probability = static_cast<float>(arrival.probability * 100.0);
    if (!out.phrase(label, words)) return false;
    out.separator();
    if (!out.value(probability, 1) || !out.phrase(SpeechClip::Percent, "percent")) return false;
    out.sentence();
  } else if (!(arrival.available && arrival.outcome == ArrivalChangeOutcome::Uncertain
      ? out.announcement(SpeechClip::ArrivalOutcomeUncertain, "Arrival outcome uncertain")
      : out.announcement(SpeechClip::UnavailableValue, "unavailable"))) return false;
  const JsonVariantConst nextHour = rainHour(weather, now <= UINT32_MAX - 3600 ? now + 3600 : 0);
  return out.measurement(SpeechClip::RainInAnHour, "Rain chance in an hour", number(nextHour["rain_chance_pct"]),
      SpeechClip::Percent, "percent", 0, 0, 100, weatherOk && !nextHour.isNull());
}

struct SessionLabel {
  SpeechClip clip = SpeechClip::UnavailableValue;
  const char *words = nullptr;
};

SessionLabel sessionLabel(JsonVariantConst session, uint32_t now, bool morning) {
  if (!validWindow(session, now)) return {};
  constexpr uint64_t malaysiaOffset = 8 * 3600;
  const uint64_t today = (uint64_t(now) + malaysiaOffset) / 86400;
  const uint64_t targetDay = (uint64_t(epoch(session["start_epoch"])) + malaysiaOffset) / 86400;
  if (targetDay == today) return morning ? SessionLabel{SpeechClip::TodayMorning, "This morning"}
      : SessionLabel{SpeechClip::TodayAfternoon, "This afternoon"};
  if (targetDay == today + 1) return morning ? SessionLabel{SpeechClip::TomorrowMorning, "Tomorrow morning"}
      : SessionLabel{SpeechClip::TomorrowAfternoon, "Tomorrow afternoon"};
  return {};
}

bool sessionPmAvailable(JsonVariantConst session, JsonVariantConst forecast, uint32_t now) {
  const JsonVariantConst pm = session["pm"];
  return validWindow(session, now) && epoch(session["issued_epoch"]) == epoch(forecast["issued_epoch"])
      && epoch(session["start_epoch"]) > epoch(session["issued_epoch"])
      && forecastModelPointAvailable(pm);
}

bool sessionRainWarning(Readout &out, JsonVariantConst session, JsonVariantConst forecast,
                        uint32_t now, SessionLabel label, bool &oldAnnounced) {
  const JsonVariantConst weather = session["weather"];
  const float rain = number(weather["rain_chance_max_pct"]);
  if (!label.words || !validWindow(session, now)
      || epoch(session["issued_epoch"]) != epoch(forecast["issued_epoch"])
      || epoch(session["start_epoch"]) <= epoch(session["issued_epoch"])
      || epoch(weather["fetched_epoch"]) > epoch(session["issued_epoch"])
      || !weatherAvailable(now, weather) || !windowWeatherCoverageAvailable(weather)
      || !readoutValueValid(rain, 0, 100) || rain <= 75) return true;
  if (weatherOld(now, weather) && !oldAnnounced) {
    if (!out.announcement(SpeechClip::Old, "Data is old")) return false;
    oldAnnounced = true;
  }
  if (!out.phrase(SpeechClip::Warning, "Warning") || !out.phrase(label.clip, label.words)
      || !out.phrase(SpeechClip::HighRainChance, "has a high rain chance of")
      || !out.value(rain, rainPrecision(rain)) || !out.phrase(SpeechClip::Percent, "percent")) return false;
  out.sentence();
  return true;
}

bool sports(Readout &out, const DashboardContext &context, uint32_t now) {
  const JsonVariantConst forecast = (*context.weather)["forecast"];
  const JsonVariantConst ride = forecast["ride90_210"];
  const JsonVariantConst weather = ride["weather"];
  const bool forecastOk = dashboardForecastAvailable(context);
  const bool rideWindowOk = forecastOk && validWindow(ride, now);
  const RidePmRange range = ridePmRange(ride);
  const bool rangeOk = rideWindowOk && range.available;
  const bool meanOk = rideWindowOk && forecastModelPointAvailable(ride);
  const bool chanceOk = rideWindowOk && ride["mean_le70"]["available"].as<bool>();
  const bool weatherOk = forecastOk && validWindow(ride, now) && weatherAvailable(now, weather)
      && epoch(weather["fetched_epoch"]) <= epoch(forecast["issued_epoch"])
      && windowWeatherCoverageAvailable(weather);
  bool oldAnnounced = forecastOk && !forecastFresh(now, forecast);
  if (oldAnnounced && !out.announcement(SpeechClip::Old, "Data is old")) return false;
  if (!(rangeOk ? out.range(SpeechClip::PMForecastRange, "PM2.5 forecast range", range.low, range.high,
          1, 0, 9999.9f)
      : meanOk ? out.labelledSentence(SpeechClip::PMForecast, "PM2.5 forecast", number(ride["pm25_ugm3"]),
          1, 0, 9999.9f)
      : out.range(SpeechClip::PMForecastRange, "PM2.5 forecast range", NAN, NAN, 1, 0, 9999.9f, false))
      || !out.measurement(SpeechClip::ChanceBelow70Is, "Chance the ride window average seventy or less is",
          number(ride["mean_le70"]["probability"]) * 100, SpeechClip::Percent, "percent", 1, 0, 100, chanceOk)) return false;
  if (weatherOk && weatherOld(now, weather) && !oldAnnounced) {
    if (!out.announcement(SpeechClip::Old, "Data is old")) return false;
    oldAnnounced = true;
  }
  const float rain = number(weather["rain_chance_max_pct"]);
  const bool warn = weatherOk && readoutValueValid(rain, 0, 100) && rain > 75;
  if (!out.measurement(warn ? SpeechClip::RideRainWarning : SpeechClip::RainChance,
      warn ? "Warning, ride window rain chance" : "Rain chance", rain,
      SpeechClip::Percent, "percent", rainPrecision(rain), 0, 100, weatherOk)) return false;
  const JsonVariantConst morning = forecast["sessions"]["morning"];
  const JsonVariantConst afternoon = forecast["sessions"]["afternoon"];
  const SessionLabel morningLabel = sessionLabel(morning, now, true);
  const SessionLabel afternoonLabel = sessionLabel(afternoon, now, false);
  const bool compareOk = forecastOk && forecastFresh(now, forecast) && morningLabel.words && afternoonLabel.words
      && sessionPmAvailable(morning, forecast, now) && sessionPmAvailable(afternoon, forecast, now);
  if (!compareOk) {
    if (!out.announcement(SpeechClip::ComparisonUnavailable, "Morning and afternoon comparison unavailable")) return false;
  } else {
    // Compare exactly the one-decimal values visible on the screen.
    const float morningMean = String(number(morning["pm"]["pm25_ugm3"]), 1).toFloat();
    const float afternoonMean = String(number(afternoon["pm"]["pm25_ugm3"]), 1).toFloat();
    if (morningMean == afternoonMean) {
      if (!out.announcement(SpeechClip::SimilarPM, "Morning and afternoon PM2.5 forecasts are similar")) return false;
    } else {
      const SessionLabel lower = morningMean < afternoonMean ? morningLabel : afternoonLabel;
      const SessionLabel higher = morningMean < afternoonMean ? afternoonLabel : morningLabel;
      if (!out.phrase(lower.clip, lower.words)
          || !out.phrase(SpeechClip::LowerForecastPMThan, "has lower forecast PM2.5 than")
          || !out.phrase(higher.clip, higher.words)) return false;
      out.sentence();
    }
  }
  return !forecastOk || (sessionRainWarning(out, morning, forecast, now, morningLabel, oldAnnounced)
      && sessionRainWarning(out, afternoon, forecast, now, afternoonLabel, oldAnnounced));
}

struct Latest {
  uint32_t epoch = 0;
  float value = NAN;
};

void includeLatest(Latest &latest, uint32_t epoch, float value) {
  if (isfinite(value) && epoch >= latest.epoch) latest = {epoch, value};
}

bool historyPage(Readout &out, const DashboardContext &context, uint32_t now) {
  const JsonVariantConst history = (*context.weather)["history"];
  const JsonArrayConst columns = history["columns"];
  const bool expectedColumns = columns.size() == 4 && columns[0] == "epoch"
      && columns[1] == "pm25_ugm3" && columns[2] == "temperature_c" && columns[3] == "heat_index_c";
  const bool available = expectedColumns && history["available"].as<bool>();
  const uint32_t start = history["start_epoch"].as<uint32_t>();
  const uint32_t end = history["end_epoch"].as<uint32_t>();
  Latest pm;
  if (available) {
    uint16_t accepted = 0;
    for (JsonArrayConst point : history["points"].as<JsonArrayConst>()) {
      if (accepted == HISTORY_GRAPH_POINT_LIMIT) break;
      if (point.size() != 4) continue;
      ++accepted;
      const uint32_t epoch = point[0].as<uint32_t>();
      if (epoch < kValidEpoch || epoch < start || epoch > end) continue;
      const float pmValue = number(point[1]);
      if (pmValue >= 0) includeLatest(pm, epoch, pmValue);
    }
  }
  const auto old = [&](const Latest &latest) {
    return latest.epoch && (!history["fresh"].as<bool>() || !available
        || (now >= latest.epoch && now - latest.epoch > 420));
  };
  HistoryPmSummary summary = dashboardHistoryPmSummary(context);
  const bool summaryOk = start >= kValidEpoch && end > start && end <= now && summary.available;
  if ((old(pm) || (summaryOk && (!history["fresh"].as<bool>()
      || age(now, history["observed_epoch"]) > 420)))
      && !out.announcement(SpeechClip::Old, "Data is old")) return false;
  if (!out.labelledValue(SpeechClip::AveragePM, "Average PM2.5", summary.average, 1, 0, 9999.9f, summaryOk)) return false;
  out.separator();
  if (!out.labelledValue(SpeechClip::Lowest, "lowest", summary.lowest, 1, 0, 9999.9f, summaryOk)) return false;
  out.separator();
  if (!out.labelledValue(SpeechClip::Highest, "highest", summary.highest, 1, 0, 9999.9f, summaryOk)) return false;
  out.sentence();
  const IndoorReading indoor = context.indoor ? *context.indoor : IndoorReading{};
  if (!out.labelledValue(SpeechClip::IndoorTemperature, "Indoor temperature", indoor.temperatureC,
      1, -100, 100, indoor.valid)) return false;
  if (indoor.valid && readoutValueValid(indoor.temperatureC, -100, 100)
      && !out.phrase(SpeechClip::DegreesCelsius, "degrees Celsius")) return false;
  out.separator();
  if (!out.labelledValue(SpeechClip::Lowest, "lowest", indoor.minimumC,
      1, -100, 100)) return false;
  if (readoutValueValid(indoor.minimumC, -100, 100)
      && !out.phrase(SpeechClip::DegreesCelsius, "degrees Celsius")) return false;
  out.sentence();
  if (!out.measurement(SpeechClip::IndoorHumidity, "Indoor humidity", indoor.humidityPct,
      SpeechClip::Percent, "percent", 1, 0, 100, indoor.valid)) return false;
  const int capacity = batteryPercentEstimate(context.batteryV);
  if (!out.labelledValue(SpeechClip::Battery, "Battery", context.batteryV, 2, 2, 4.5f, capacity >= 0)) return false;
  if (capacity >= 0 && !out.phrase(SpeechClip::Volts, "volts")) return false;
  out.separator();
  if (!(capacity >= 0 ? out.value(capacity, 0) : out.unavailable())
      || !out.phrase(SpeechClip::PercentCapacity, "percent capacity")) return false;
  out.sentence();
  return true;
}
}  // namespace

bool buildCurrentPageReadout(const DashboardContext &context, SpeechPlaylist &out,
                            String &spokenText) {
  out = SpeechPlaylist{};
  spokenText = "";
  Readout readout(out, spokenText);
  bool ok;
  if (!context.weather) {
    ok = readout.announcement(SpeechClip::UnavailableValue, "unavailable");
  } else {
    const uint32_t now = dashboardNow(context);
    const JsonVariantConst demo = (*context.weather)["demo"];
    ok = !demo.is<bool>() || !demo.as<bool>()
        || readout.announcement(SpeechClip::DemoData, "Demonstration data");
    ok = ok && (context.page == 1 ? sports(readout, context, now)
        : context.page == 2 ? historyPage(readout, context, now)
        : overview(readout, context, now));
  }
  if (!ok || !out.count) {
    out = SpeechPlaylist{};
    spokenText = "";
    return false;
  }
  spokenText.trim();
  return true;
}
