#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

enum class VoiceDiagnosticReason : uint8_t { Detected, Rejected, Timeout, SilenceTail };

inline const char *voiceDiagnosticReasonName(VoiceDiagnosticReason reason) {
  switch (reason) {
    case VoiceDiagnosticReason::Detected: return "detected";
    case VoiceDiagnosticReason::Rejected: return "rejected";
    case VoiceDiagnosticReason::Timeout: return "timeout";
    case VoiceDiagnosticReason::SilenceTail: return "silence_tail";
  }
  return "unknown";
}

// A copied decoder snapshot, not a word transcript or an executable command.
// Strings use MultiNet7's phoneme/token alphabet. Queue transport owns every
// byte, so clean()/the next model frame cannot change a published diagnostic.
struct VoiceDiagnostic {
  static constexpr size_t kCandidateCapacity = 5;
  uint32_t sequence = 0, sessionId = 0, endedMs = 0, sessionDurationMs = 0;
  uint32_t modelFrames = 0, speechSamples = 0, afeChannelChanges = 0, vadCacheSamples = 0;
  VoiceDiagnosticReason reason = VoiceDiagnosticReason::SilenceTail;
  int modelState = -1, detectState = -1, afeTriggerChannel = -1, afeRawChannels = 0, apiResultCount = 0;
  float minimumSpeechDbfs = NAN, maximumSpeechDbfs = NAN;
  uint8_t candidateCount = 0;
  bool candidatesValid = false;
  int commandIds[kCandidateCapacity]{};
  int phraseIds[kCandidateCapacity]{};
  float probabilities[kCandidateCapacity]{};
  char string[256]{};
  char rawString[256]{};
  bool stringTruncated = false, rawStringTruncated = false;
};

template <size_t OutputSize, size_t InputSize>
inline bool copyVoiceDiagnosticText(char (&output)[OutputSize], const char (&input)[InputSize]) {
  static_assert(OutputSize > 0 && InputSize > 0, "Diagnostic arrays need storage");
  size_t copied = 0;
  while (copied < OutputSize - 1 && copied < InputSize && input[copied]) {
    output[copied] = input[copied];
    ++copied;
  }
  output[copied] = '\0';
  return copied == InputSize || (copied < InputSize && input[copied] != '\0');
}

// Works directly with esp_mn_results_t, while remaining host-testable. SDK num
// can be zero until detection, and arrays can retain old bytes after reset.
// Only a real DETECTED return plus DETECTED API state (SDK enum value 1) exposes
// candidate scores. clean() alone preserves old state until the next detect().
template <typename Result>
inline void copyVoiceDiagnosticResults(VoiceDiagnostic &out, const Result *result) {
  out.modelState = result ? int(result->state) : -1;
  out.apiResultCount = result ? result->num : 0;
  out.candidatesValid = result && out.detectState == 1 && int(result->state) == 1 && result->num > 0;
  out.candidateCount = 0;
  for (size_t i = 0; i < VoiceDiagnostic::kCandidateCapacity; ++i) {
    out.commandIds[i] = 0;
    out.phraseIds[i] = -1;
    out.probabilities[i] = 0;
  }
  out.string[0] = out.rawString[0] = '\0';
  out.stringTruncated = out.rawStringTruncated = false;
  if (!result) return;
  out.stringTruncated = copyVoiceDiagnosticText(out.string, result->string);
  out.rawStringTruncated = copyVoiceDiagnosticText(out.rawString, result->raw_string);
  if (!out.candidatesValid) return;
  out.candidateCount = uint8_t(result->num < int(VoiceDiagnostic::kCandidateCapacity)
      ? result->num : VoiceDiagnostic::kCandidateCapacity);
  for (size_t i = 0; i < out.candidateCount; ++i) {
    out.commandIds[i] = result->command_id[i];
    out.phraseIds[i] = result->phrase_id[i];
    out.probabilities[i] = result->prob[i]; // Preserve SDK evidence; JSON must guard non-finite values.
  }
}
