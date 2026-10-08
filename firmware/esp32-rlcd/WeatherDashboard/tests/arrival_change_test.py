# SPDX-License-Identifier: Apache-2.0
"""Compile the production arrival decoder against real ArduinoJson on the host.

No firmware upload, serial port, HTTP server, model or audio hardware is used.
Example: python WeatherDashboard/tests/arrival_change_test.py --cxx g++ --build-dir build/host/arrival
"""

import argparse
from pathlib import Path
import subprocess
import sys


HARNESS = r"""
#include "ArrivalChange.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>

using Outcome = ArrivalChangeOutcome;
constexpr uint32_t now = 1791494400;
constexpr uint32_t issued = now - 60;
size_t checks = 0;

JsonDocument fixture(double fall = .04, double rise = .04, double within = .92,
                     const char *outcome = "within20", const char *text = "No change on arrival") {
  JsonDocument near;
  near["available"] = true;
  near["pm25_ugm3"] = 159.8;
  near["reference_pm25_ugm3"] = 162.3;
  near["target_epoch"] = issued + 5400;
  auto change = near["arrival_change"].to<JsonObject>();
  change["available"] = true;
  change["fall20"] = fall;
  change["fall40"] = .001;
  change["rise20"] = rise;
  change["rise40"] = .001;
  change["within20"] = within;
  change["reference_ugm3"] = 162.3;
  change["issued_epoch"] = issued;
  change["arrival_epoch"] = issued + 5400;
  change["fresh_reference_epoch"] = issued - 30;
  change["display_text"] = text;
  if (outcome) {
    change["outcome"] = outcome;
    change["outcome_probability"] = std::strcmp(outcome, "fall20") == 0 ? fall
        : std::strcmp(outcome, "rise20") == 0 ? rise : within;
  } else {
    change["outcome"] = nullptr;
    change["outcome_probability"] = nullptr;
  }
  return near;
}

ArrivalChangeResult decode(const JsonDocument &near, uint32_t parent = issued, uint32_t clock = now) {
  return decodeArrivalChange(near.as<JsonVariantConst>(), parent, clock);
}

void unavailable(const JsonDocument &near, uint32_t parent = issued, uint32_t clock = now) {
  const auto result = decode(near, parent, clock);
  assert(!result.available && result.outcome == Outcome::Unavailable);
  assert(result.displayText == nullptr && std::isnan(result.probability));
  assert(!result.issuedEpoch && !result.arrivalEpoch && !result.freshReferenceEpoch);
  assert(std::isnan(result.referenceUgM3) && std::isnan(result.within20));
  ++checks;
}

void accepted(const JsonDocument &near, Outcome outcome, double score) {
  const auto result = decode(near);
  assert(result.available && result.outcome == outcome);
  assert(result.displayText && result.displayText[0]);
  if (outcome == Outcome::Uncertain) assert(std::isnan(result.probability));
  else assert(result.probability == score);  // Native score, without rounding/normalization.
  assert(result.issuedEpoch == issued && result.arrivalEpoch == issued + 5400);
  ++checks;
}

void winnersAndTies() {
  auto center = fixture(); accepted(center, Outcome::Within20, .92);
  auto result = decode(center);
  assert(std::strcmp(result.displayText, "No change on arrival") == 0);
  assert(result.referenceUgM3 == 162.3 && result.freshReferenceEpoch == issued - 30);
  assert(result.fall20 == .04 && result.rise20 == .04 && result.within20 == .92);
  auto fall = fixture(.75, .1, .15, "fall20", "Fall on arrival");
  accepted(fall, Outcome::Fall20, .75);
  auto rise = fixture(.1, .75, .15, "rise20", "Rise on arrival");
  accepted(rise, Outcome::Rise20, .75);
  // A unique argmax does not need to exceed 50 percent.
  auto small = fixture(.4, .35, .25, "fall20", "Fall on arrival");
  accepted(small, Outcome::Fall20, .4);
  auto directionalTie = fixture(.4, .4, .2, nullptr, "Arrival outcome uncertain");
  accepted(directionalTie, Outcome::Uncertain, NAN);
  auto centerTie = fixture(.4, .2, .4, nullptr, "Arrival outcome uncertain");
  accepted(centerTie, Outcome::Uncertain, NAN);
  auto otherTie = fixture(.2, .4, .4, nullptr, "Arrival outcome uncertain");
  accepted(otherTie, Outcome::Uncertain, NAN);
  auto triple = fixture(1.0 / 3, 1.0 / 3, 1.0 / 3, nullptr, "Arrival outcome uncertain");
  accepted(triple, Outcome::Uncertain, NAN);
  // Tiny differences remain unique raw winners, matching exact JavaScript equality.
  auto tiny = fixture(.4 + 1e-13, .4, .2 - 1e-13, "fall20", "Fall on arrival");
  accepted(tiny, Outcome::Fall20, .4 + 1e-13);
  auto allCenter = fixture(0, 0, 1);
  allCenter["arrival_change"]["fall40"] = 0;
  allCenter["arrival_change"]["rise40"] = 0;
  accepted(allCenter, Outcome::Within20, 1);
}

void independentHeads() {
  auto near = fixture();
  near["available"] = false;
  near["pm25_ugm3"] = nullptr;
  near["role"] = "reference_only";
  near["first20"]["available"] = false;
  near["first20"]["none_probability"] = .01;
  near["first20"]["display_text"] = "Unrelated first crossing";
  accepted(near, Outcome::Within20, .92);
  near.remove("first20");
  accepted(near, Outcome::Within20, .92);
  near["reference_pm25_ugm3"] = nullptr;
  accepted(near, Outcome::Within20, .92); // A rejected numerical source clock must not hide a valid native arrival head.
  near.remove("arrival_change");
  near["first20"]["available"] = true;
  near["first20"]["rise_probability"] = .92;
  near["first20"]["display_text"] = "Fall on arrival";
  unavailable(near);  // Never borrow even plausibly worded first-crossing data.
  near = fixture(); near["arrival_change"].remove("within20");
  unavailable(near);  // No 1-fall20-rise20 reconstruction.
  near = fixture(); near["arrival_change"]["within20"] = nullptr;
  unavailable(near);
}

void requiredTypesAndValues() {
  JsonDocument empty; unavailable(empty);
  auto near = fixture(); near["arrival_change"] = nullptr; unavailable(near);
  near = fixture(); near["arrival_change"] = true; unavailable(near);
  for (unsigned type = 0; type < 5; ++type) {
    near = fixture(); auto c = near["arrival_change"];
    if (type == 0) c["available"] = false;
    if (type == 1) c["available"] = nullptr;
    if (type == 2) c["available"] = "true";
    if (type == 3) c["available"] = 1;
    if (type == 4) c.remove("available");
    unavailable(near);
  }
  const char *fields[] = {"fall20", "fall40", "rise20", "rise40", "within20"};
  for (const char *field : fields) {
    for (unsigned type = 0; type < 9; ++type) {
      near = fixture(); auto c = near["arrival_change"];
      if (type == 0) c[field] = nullptr;
      if (type == 1) c.remove(field);
      if (type == 2) c[field] = "0.04";
      if (type == 3) c[field] = true;
      if (type == 4) c[field] = NAN;
      if (type == 5) c[field] = INFINITY;
      if (type == 6) c[field] = -INFINITY;
      if (type == 7) c[field] = -.0001;
      if (type == 8) c[field] = 1.0001;
      unavailable(near);
    }
  }
  for (unsigned type = 0; type < 5; ++type) {
    near = fixture(); auto c = near["arrival_change"];
    if (type == 0) c["display_text"] = nullptr;
    if (type == 1) c.remove("display_text");
    if (type == 2) c["display_text"] = false;
    if (type == 3) c["display_text"] = 42;
    if (type == 4) c["display_text"] = "";
    unavailable(near);
  }
}

void tailAndOutcomeCoherence() {
  auto near = fixture(); near["arrival_change"]["fall40"] = .05; unavailable(near);
  near = fixture(); near["arrival_change"]["rise40"] = .05; unavailable(near);
  near = fixture(); near["arrival_change"]["within20"] = .94; unavailable(near);
  near = fixture(); near["arrival_change"]["within20"] = .90; unavailable(near);
  near = fixture(); near["arrival_change"]["fall40"] = .04 + 5e-13;
  accepted(near, Outcome::Within20, .92);  // Documented small nesting tolerance.
  near["arrival_change"]["fall40"] = .04 + 2e-12; unavailable(near);
  near = fixture(); near["arrival_change"]["within20"] = .92 + 5e-9;
  near["arrival_change"]["outcome_probability"] = .92 + 5e-9;
  accepted(near, Outcome::Within20, .92 + 5e-9);  // No renormalization.
  near["arrival_change"]["within20"] = .92 + 2e-8;
  near["arrival_change"]["outcome_probability"] = .92 + 2e-8;
  unavailable(near);
  for (unsigned type = 0; type < 7; ++type) {
    near = fixture(); auto c = near["arrival_change"];
    if (type == 0) c["outcome"] = "rise20";
    if (type == 1) c["outcome"] = "unresolved";
    if (type == 2) c["outcome"] = nullptr;
    if (type == 3) c.remove("outcome");
    if (type == 4) c["outcome"] = true;
    if (type == 5) c["outcome"] = 0;
    if (type == 6) c["outcome_probability"] = .5;
    unavailable(near);
  }
  for (unsigned type = 0; type < 7; ++type) {
    near = fixture(); auto c = near["arrival_change"];
    if (type == 0) c["outcome_probability"] = nullptr;
    if (type == 1) c.remove("outcome_probability");
    if (type == 2) c["outcome_probability"] = "0.92";
    if (type == 3) c["outcome_probability"] = true;
    if (type == 4) c["outcome_probability"] = NAN;
    if (type == 5) c["outcome_probability"] = INFINITY;
    if (type == 6) c["outcome_probability"] = 1.01;
    unavailable(near);
  }
  for (unsigned type = 0; type < 4; ++type) {
    near = fixture(.4, .4, .2, nullptr, "Arrival outcome uncertain");
    auto c = near["arrival_change"];
    if (type == 0) c["outcome"] = "within20";
    if (type == 1) c["outcome_probability"] = 0;
    if (type == 2) c.remove("outcome");
    if (type == 3) c.remove("outcome_probability");
    unavailable(near);
  }
}

void referencesAndClocks() {
  auto near = fixture(); near.remove("reference_pm25_ugm3");
  accepted(near, Outcome::Within20, .92);  // Reference match is enforced when nonnull and supplied.
  for (const char *field : {"reference_ugm3", "reference_pm25_ugm3"}) {
    for (unsigned type = 0; type < 7; ++type) {
      near = fixture(); JsonVariant value = std::strcmp(field, "reference_ugm3") == 0
          ? near["arrival_change"][field].as<JsonVariant>() : near[field].as<JsonVariant>();
      if (type == 0) value.set(nullptr);
      if (type == 1) value.set("162.3");
      if (type == 2) value.set(false);
      if (type == 3) value.set(NAN);
      if (type == 4) value.set(INFINITY);
      if (type == 5) value.set(-.01);
      if (type == 6) value.set(162.3 + 1e-6);
      if (type == 0 && std::strcmp(field, "reference_pm25_ugm3") == 0)
        accepted(near, Outcome::Within20, .92);
      else unavailable(near);
    }
  }
  near = fixture(); near["arrival_change"]["reference_ugm3"] = 162.3 + 5e-8;
  accepted(near, Outcome::Within20, .92);
  near = fixture(.75, .1, .15, "fall20", "Fall on arrival");
  near["reference_pm25_ugm3"] = 0; near["arrival_change"]["reference_ugm3"] = 0;
  accepted(near, Outcome::Fall20, .75); // Arrival validation does not reuse first20's drop-reference rule.
  for (const char *field : {"issued_epoch", "arrival_epoch", "fresh_reference_epoch", "target_epoch"}) {
    for (unsigned type = 0; type < 7; ++type) {
      near = fixture(); JsonVariant value = std::strcmp(field, "target_epoch") == 0
          ? near[field].as<JsonVariant>() : near["arrival_change"][field].as<JsonVariant>();
      if (type == 0) value.set(nullptr);
      if (type == 1) value.set("1791494340");
      if (type == 2) value.set(true);
      if (type == 3) value.set(1791494340.0);
      if (type == 4) value.set(-1);
      if (type == 5) value.set(UINT64_MAX);
      if (type == 6) {
        if (std::strcmp(field, "target_epoch") == 0) near.remove(field);
        else near["arrival_change"].remove(field);
      }
      unavailable(near);
    }
  }
  near = fixture(); near["arrival_change"]["issued_epoch"] = issued - 1; unavailable(near);
  near = fixture(); near["arrival_change"]["arrival_epoch"] = issued + 5399; unavailable(near);
  near = fixture(); near["target_epoch"] = issued + 5401; unavailable(near);
  near = fixture(); near["arrival_change"]["fresh_reference_epoch"] = issued - 240;
  accepted(near, Outcome::Within20, .92);
  near["arrival_change"]["fresh_reference_epoch"] = issued - 241; unavailable(near);
  near["arrival_change"]["fresh_reference_epoch"] = 0; unavailable(near);
  near["arrival_change"]["fresh_reference_epoch"] = issued + 1; unavailable(near);
  near = fixture(); unavailable(near, issued, issued + 5400); // Target expired, including exact boundary.
  unavailable(near, issued, issued + 5401);
  unavailable(near, issued + 1);
  unavailable(near, issued, issued - 1); // Future issue clock.
  unavailable(near, UINT32_MAX, UINT32_MAX); // Addition cannot overflow.
  unavailable(near, 0, now);
  // Parent age limit is a separate Dashboard guard, not an accidental head dependency.
  const auto oldParent = decode(near, issued, issued + 601);
  assert(oldParent.available); ++checks;
}

int main() {
  winnersAndTies(); independentHeads(); requiredTypesAndValues();
  tailAndOutcomeCoherence(); referencesAndClocks();
  std::printf("PASS: %zu arrival-decoder assertions; independent native arrival head, exact winners/ties, strict null/types, clocks, fresh references, nested tails and probability coherence\n", checks);
}
"""


def main():
    firmware = Path(__file__).resolve().parents[1]
    package = firmware.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--cxx-arg", action="append", default=[])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--arduino-json-src", type=Path,
                        default=package / ".state/arduino-user/libraries/ArduinoJson/src",
                        help="ArduinoJson 7.4.3 src directory")
    args = parser.parse_args()
    arduino_json = args.arduino_json_src.resolve()
    for required in (firmware / "ArrivalChange.h", arduino_json / "ArduinoJson.h"):
        if not required.is_file():
            parser.error(f"Required test input not found: {required}")
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "arrival_change_test.cpp"
    source.write_text(HARNESS, encoding="utf-8")
    suffix = ".exe" if sys.platform == "win32" else ""
    executable = build / ("arrival_change_test" + suffix)
    subprocess.run([args.cxx, *args.cxx_arg, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(firmware), "-I", str(arduino_json),
                    str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
