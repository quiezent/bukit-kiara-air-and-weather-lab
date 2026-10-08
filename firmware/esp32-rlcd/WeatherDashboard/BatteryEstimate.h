#pragma once

// A coarse display estimate from terminal voltage, not a measured state of
// charge. The cell model, current, temperature, and relaxed OCV are unknown.
// These endpoints are UI references, not charge/discharge control thresholds.
constexpr float kBatteryEstimateEmptyV = 3.0f;
constexpr float kBatteryEstimateFullV = 4.2f;
constexpr int kBatteryEstimateStepPct = 5;
constexpr char kBatteryPercentMethod[] = "voltage-linear-3.0-to-4.2V-5pct";
constexpr char kBatteryPercentDescription[] = "Approximate voltage estimate";

// Returns 0..100 in 5-point steps, or -1 for an implausible/non-finite sample.
// Comparisons reject NaN and infinity without depending on an Arduino API.
constexpr int batteryPercentEstimate(float voltage) {
  if (!(voltage >= 2.0f && voltage <= 4.5f)) return -1;
  if (voltage <= kBatteryEstimateEmptyV) return 0;
  if (voltage >= kBatteryEstimateFullV) return 100;
  const float percent = 100.0f * (voltage - kBatteryEstimateEmptyV) /
                        (kBatteryEstimateFullV - kBatteryEstimateEmptyV);
  return static_cast<int>((percent + kBatteryEstimateStepPct / 2.0f) /
                          kBatteryEstimateStepPct) * kBatteryEstimateStepPct;
}
