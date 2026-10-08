// Exercise the production playlist and button helpers without Arduino or PCM.
#include "../WeatherReadout.h"
#include "../ButtonInput.h"
#include <cassert>
#include <initializer_list>
#include <limits>
#include <cstdio>

using C = SpeechClip;

static void expect(const SpeechPlaylist &actual, std::initializer_list<C> wanted) {
  assert(actual.count == wanted.size());
  size_t position = 0;
  for (C clip : wanted) assert(actual.clips[position++] == clip);
}

static void expectSame(const SpeechPlaylist &actual, const SpeechPlaylist &wanted) {
  assert(actual.count == wanted.count);
  for (size_t i = 0; i < actual.count; ++i) assert(actual.clips[i] == wanted.clips[i]);
}

static void number(float value, bool decimal, std::initializer_list<C> wanted) {
  SpeechPlaylist actual;
  assert(speechNumber(actual, value, decimal));
  expect(actual, wanted);
}

static void numericChecks() {
  number(0, false, {C::N0});
  number(-0.0f, true, {C::N0, C::Point, C::N0});
  number(-0.04f, true, {C::N0, C::Point, C::N0});
  number(-0.05f, true, {C::Minus, C::N0, C::Point, C::N1});
  number(2.5f, false, {C::N3});
  number(-2.5f, false, {C::Minus, C::N3});
  number(25.25f, true, {C::N20, C::N5, C::Point, C::N3});
  number(19.96f, true, {C::N20, C::Point, C::N0});
  number(137.1f, true, {C::N1, C::Hundred, C::N30, C::N7, C::Point, C::N1});
  number(100, false, {C::N1, C::Hundred});
  number(101, false, {C::N1, C::Hundred, C::N1});
  number(999, false, {C::N9, C::Hundred, C::N90, C::N9});
  number(1000, false, {C::N1, C::Thousand});
  number(1001, false, {C::N1, C::Thousand, C::N1});
  number(1010, false, {C::N1, C::Thousand, C::N10});
  number(1100, false, {C::N1, C::Thousand, C::N1, C::Hundred});
  number(9999.9f, true, {C::N9, C::Thousand, C::N9, C::Hundred,
      C::N90, C::N9, C::Point, C::N9});
  number(-100, true, {C::Minus, C::N1, C::Hundred, C::Point, C::N0});

  // Protect the enum-to-number mapping used by speechWholeNumber.
  for (unsigned value = 0; value < 20; ++value) {
    SpeechPlaylist clips;
    assert(speechNumber(clips, float(value), false));
    assert(clips.count == 1 && clips.clips[0] == static_cast<C>(value));
  }
  for (unsigned value = 20; value <= 90; value += 10) {
    SpeechPlaylist clips;
    assert(speechNumber(clips, float(value), false));
    assert(clips.count == 1 && clips.clips[0] == static_cast<C>(20 + value / 10 - 2));
  }

  SpeechPlaylist baseline;
  assert(baseline.add(C::CurrentPM25));
  const float invalid[] = {std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(), 10000, -10000};
  for (float value : invalid) {
    SpeechPlaylist actual = baseline;
    assert(!speechNumber(actual, value, true));
    expectSame(actual, baseline);
  }
  SpeechPlaylist actual = baseline;
  assert(!speechNumber(actual, 9999.9f, false)); // Rounds beyond the spoken limit.
  expectSame(actual, baseline);

  // Overflow must not leave a label, partial number, or unit in the playlist.
  SpeechPlaylist nearFull;
  for (size_t i = 0; i < SpeechPlaylist::kCapacity - 1; ++i) assert(nearFull.add(C::N0));
  actual = nearFull;
  assert(!speechNumber(actual, 21, false));
  expectSame(actual, nearFull);
  assert(!speechMeasurement(actual, C::Humidity, 55, C::Percent, false, 0, 100));
  expectSame(actual, nearFull);
  assert(actual.add(C::N1));
  const SpeechPlaylist full = actual;
  assert(!actual.add(C::N2));
  expectSame(actual, full);

  for (float value : {NAN, -1.0f, 100.1f}) {
    actual = baseline;
    assert(speechMeasurement(actual, C::Humidity, value, C::Percent, false, 0, 100));
    expectSame(actual, baseline);
  }
}

static void weatherChecks() {
  WeatherReadoutSnapshot snapshot;
  SpeechPlaylist actual;
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::Unavailable, C::ForecastUnavailable});

  // API availability flags cannot turn missing/invalid values into observations.
  snapshot.currentAvailable = true;
  snapshot.forecastAvailable = true;
  assert(!readoutHasCurrent(snapshot) && !readoutHasForecast(snapshot));
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::Unavailable, C::ForecastUnavailable});
  snapshot.currentOld = true;
  snapshot.forecastOld = true;
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::Unavailable, C::ForecastUnavailable}); // No old-data intro without data.
  snapshot.pm25 = -1;
  snapshot.temperature = 101;
  snapshot.humidity = 101;
  snapshot.heatIndex = -101;
  snapshot.wind = 1001;
  snapshot.rainChance = -1;
  assert(!readoutHasCurrent(snapshot) && !readoutHasForecast(snapshot));
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::Unavailable, C::ForecastUnavailable});
  snapshot = WeatherReadoutSnapshot{};

  snapshot.currentAvailable = true;
  snapshot.pm25 = 162.2f;
  snapshot.temperature = 27.6f;
  snapshot.humidity = 64;
  snapshot.heatIndex = 30.5f;
  snapshot.forecastAvailable = true;
  snapshot.wind = 3;
  snapshot.rainChance = 2;
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::CurrentPM25, C::N1, C::Hundred, C::N60, C::N2, C::Point, C::N2, C::Micrograms,
      C::OutdoorTemperature, C::N20, C::N7, C::Point, C::N6, C::DegreesCelsius,
      C::Humidity, C::N60, C::N4, C::Percent,
      C::HeatIndex, C::N30, C::Point, C::N5, C::DegreesCelsius,
      C::ForecastWind, C::N3, C::KilometresPerHour,
      C::RainNextHour, C::N2, C::Percent});

  // Current readings and forecast wind/rain have independent availability/age.
  snapshot.currentOld = true;
  snapshot.forecastOld = true;
  assert(buildWeatherReadout(snapshot, actual));
  assert(actual.clips[0] == C::CurrentDataOld);
  size_t oldForecast = 0;
  while (oldForecast < actual.count && actual.clips[oldForecast] != C::ForecastOld) ++oldForecast;
  assert(oldForecast < actual.count && actual.clips[oldForecast + 1] == C::ForecastWind);
  snapshot.currentAvailable = false;
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::Unavailable, C::ForecastOld, C::ForecastWind, C::N3,
      C::KilometresPerHour, C::RainNextHour, C::N2, C::Percent});
  snapshot.forecastAvailable = false;
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::Unavailable, C::ForecastUnavailable});

  // Out-of-domain readings are skipped rather than spoken as valid observations.
  snapshot.currentAvailable = true;
  snapshot.currentOld = false;
  snapshot.pm25 = NAN;
  snapshot.temperature = 101;
  snapshot.humidity = 101;
  snapshot.heatIndex = -25.25f;
  assert(buildWeatherReadout(snapshot, actual));
  expect(actual, {C::HeatIndex, C::Minus, C::N20, C::N5, C::Point, C::N3,
      C::DegreesCelsius, C::ForecastUnavailable});

  // All supported extreme readings fit in the bounded playlist with valid IDs.
  snapshot.currentOld = true;
  snapshot.forecastAvailable = true;
  snapshot.forecastOld = true;
  snapshot.pm25 = 9999.9f;
  snapshot.temperature = -100;
  snapshot.humidity = 100;
  snapshot.heatIndex = -100;
  snapshot.wind = 1000;
  snapshot.rainChance = 100;
  assert(buildWeatherReadout(snapshot, actual));
  assert(actual.count <= SpeechPlaylist::kCapacity);
  for (size_t i = 0; i < actual.count; ++i) assert(actual.clips[i] < C::Count);
}

static void buttonChecks() {
  ReleasedButton key;
  ReleasedButton boot;
  assert(!key.update(true, 0));
  assert(!boot.update(true, 0));
  assert(!key.update(false, 100));
  assert(!key.update(false, 139));
  assert(!key.update(false, 140)); // Debounced press produces no action.
  assert(!boot.update(false, 150));
  assert(!boot.update(false, 190));
  assert(!key.update(true, 200));
  assert(!key.update(true, 239));
  assert(key.update(true, 240)); // KEY release is independent of held BOOT.
  assert(!key.update(true, 241));
  assert(!boot.update(true, 250));
  assert(boot.update(true, 290));
  assert(!boot.update(true, 300));

  // Bounce restarts the 40 ms stability interval on both edges.
  ReleasedButton bouncing;
  assert(!bouncing.update(false, 100));
  assert(!bouncing.update(true, 120));
  assert(!bouncing.update(false, 130));
  assert(!bouncing.update(false, 169));
  assert(!bouncing.update(false, 170));
  assert(!bouncing.update(true, 200));
  assert(!bouncing.update(false, 220));
  assert(!bouncing.update(true, 230));
  assert(!bouncing.update(true, 269));
  assert(bouncing.update(true, 270));
  assert(!bouncing.update(true, 500));

  // A press shorter than the stability interval never counts as a release.
  ReleasedButton shortPress;
  assert(!shortPress.update(false, 100));
  assert(!shortPress.update(true, 120));
  assert(!shortPress.update(true, 160));

  ReleasedButton wrapped;
  assert(!wrapped.update(false, UINT32_MAX - 50));
  assert(!wrapped.update(false, UINT32_MAX - 10));
  assert(!wrapped.update(true, UINT32_MAX - 5));
  assert(!wrapped.update(true, 33)); // 39 ms across millis rollover.
  assert(wrapped.update(true, 34)); // Exactly 40 ms.
  assert(!wrapped.update(true, 35));
}

int main() {
  numericChecks();
  weatherChecks();
  buttonChecks();
  puts("PASS: spoken values, rounding, bounded transactional playlists, current/forecast labels, unavailable/stale data, independent release buttons, bounce and millis rollover");
}
