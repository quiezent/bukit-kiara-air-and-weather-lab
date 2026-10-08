// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ArduinoJson.h>
#include <cmath>
#include <cstdint>
#include <limits>

// This is the concentration change at the exact +90-minute arrival target.
// It is independent of the first crossing of a threshold before that target.
enum class ArrivalChangeOutcome { Unavailable, Within20, Fall20, Rise20, Uncertain };

struct ArrivalChangeResult {
  bool available = false;
  ArrivalChangeOutcome outcome = ArrivalChangeOutcome::Unavailable;
  double probability = std::numeric_limits<double>::quiet_NaN();
  // Borrowed from the JsonDocument; callers must copy it before replacing the
  // document. Text is the server's exact dynamic label, without a percentage.
  const char *displayText = nullptr;
  uint32_t issuedEpoch = 0;
  uint32_t arrivalEpoch = 0;
  uint32_t freshReferenceEpoch = 0;
  double referenceUgM3 = std::numeric_limits<double>::quiet_NaN();
  double fall20 = std::numeric_limits<double>::quiet_NaN();
  double fall40 = std::numeric_limits<double>::quiet_NaN();
  double rise20 = std::numeric_limits<double>::quiet_NaN();
  double rise40 = std::numeric_limits<double>::quiet_NaN();
  double within20 = std::numeric_limits<double>::quiet_NaN();
};

namespace ArrivalChangeDetail {
inline bool probabilityValid(JsonVariantConst value) {
  if (!value.is<double>()) return false;
  const double number = value.as<double>();
  return std::isfinite(number) && number >= 0.0 && number <= 1.0;
}

inline bool referenceValid(JsonVariantConst value) {
  return value.is<double>() && std::isfinite(value.as<double>()) && value.as<double>() >= 0.0;
}
}  // namespace ArrivalChangeDetail

// The caller separately gates the parent forecast's availability/600s expiry.
// A missing numeric +90-minute point does not invalidate an independently
// published arrival-change head. Native within20/outcome/probability are never
// reconstructed from a point estimate or from first20 probabilities.
inline ArrivalChangeResult decodeArrivalChange(JsonVariantConst near, uint32_t parentIssue,
                                               uint32_t now) {
  const JsonVariantConst change = near["arrival_change"];
  if (!change.is<JsonObjectConst>() || !change["available"].is<bool>()
      || !change["available"].as<bool>() || !change["issued_epoch"].is<uint32_t>()
      || !change["arrival_epoch"].is<uint32_t>()
      || !change["fresh_reference_epoch"].is<uint32_t>()
      || !near["target_epoch"].is<uint32_t>() || parentIssue < 1700000000
      || parentIssue > now || parentIssue > UINT32_MAX - 5400) return {};

  const uint32_t issued = change["issued_epoch"].as<uint32_t>();
  const uint32_t arrival = change["arrival_epoch"].as<uint32_t>();
  const uint32_t referenceEpoch = change["fresh_reference_epoch"].as<uint32_t>();
  if (issued != parentIssue || arrival != issued + 5400
      || arrival != near["target_epoch"].as<uint32_t>() || arrival <= now
      || !referenceEpoch || referenceEpoch > issued || issued - referenceEpoch > 240) return {};

  using ArrivalChangeDetail::probabilityValid;
  if (!probabilityValid(change["fall20"]) || !probabilityValid(change["fall40"])
      || !probabilityValid(change["rise20"]) || !probabilityValid(change["rise40"])
      || !probabilityValid(change["within20"])
      || !ArrivalChangeDetail::referenceValid(change["reference_ugm3"])) return {};
  const double fall = change["fall20"].as<double>();
  const double fall40 = change["fall40"].as<double>();
  const double rise = change["rise20"].as<double>();
  const double rise40 = change["rise40"].as<double>();
  const double within = change["within20"].as<double>();
  const double reference = change["reference_ugm3"].as<double>();
  if (fall40 > fall + 1e-12 || rise40 > rise + 1e-12
      || std::fabs(fall + rise + within - 1.0) > 1e-8) return {};
  const JsonVariantConst pointReference = near["reference_pm25_ugm3"];
  // A numerical head can be unavailable (or have rejected source clocks)
  // while the independent native arrival head still has a valid reference.
  // Compare the numerical head's reference only when it actually supplies one.
  if (!pointReference.isUnbound() && !pointReference.isNull()
      && (!ArrivalChangeDetail::referenceValid(pointReference)
          || std::fabs(reference - pointReference.as<double>()) > 1e-7)) return {};

  // Exact comparisons match the server/web argmax. Roundoff tolerance is for
  // probability coherence only; it must not turn a unique winner into a tie.
  const ArrivalChangeOutcome outcome = fall > rise && fall > within ? ArrivalChangeOutcome::Fall20
      : rise > fall && rise > within ? ArrivalChangeOutcome::Rise20
      : within > fall && within > rise ? ArrivalChangeOutcome::Within20
      : ArrivalChangeOutcome::Uncertain;
  const JsonVariantConst nativeOutcome = change["outcome"];
  const JsonVariantConst nativeProbability = change["outcome_probability"];
  if (nativeOutcome.isUnbound() || nativeProbability.isUnbound()) return {};
  double probability = std::numeric_limits<double>::quiet_NaN();
  if (outcome == ArrivalChangeOutcome::Uncertain) {
    if (!nativeOutcome.isNull() || !nativeProbability.isNull()) return {};
  } else {
    const char *expected = outcome == ArrivalChangeOutcome::Fall20 ? "fall20"
        : outcome == ArrivalChangeOutcome::Rise20 ? "rise20" : "within20";
    if (nativeOutcome != expected || !probabilityValid(nativeProbability)) return {};
    probability = outcome == ArrivalChangeOutcome::Fall20 ? fall
        : outcome == ArrivalChangeOutcome::Rise20 ? rise : within;
    if (std::fabs(nativeProbability.as<double>() - probability) > 1e-12) return {};
    // Preserve the native score for narration; the comparison above validates
    // it but does not substitute or normalize any API output.
    probability = nativeProbability.as<double>();
  }
  if (!change["display_text"].is<const char *>()) return {};
  const char *displayText = change["display_text"].as<const char *>();
  if (!displayText || !displayText[0]) return {};

  ArrivalChangeResult result;
  result.available = true;
  result.outcome = outcome;
  result.probability = probability;
  result.displayText = displayText;
  result.issuedEpoch = issued;
  result.arrivalEpoch = arrival;
  result.freshReferenceEpoch = referenceEpoch;
  result.referenceUgM3 = reference;
  result.fall20 = fall;
  result.fall40 = fall40;
  result.rise20 = rise;
  result.rise40 = rise40;
  result.within20 = within;
  return result;
}
