// Compile against the exact installed SDK interface; no PCM or model library.
#include "../VoiceDiagnostic.h"
#include "esp_mn_iface.h"
#include <cassert>
#include <cstring>
#include <type_traits>
#include <initializer_list>
#include <cstdio>

static_assert(ESP_MN_STATE_DETECTED == 1, "Diagnostic candidate-state gate matches SDK");
static_assert(ESP_MN_RESULT_MAX_NUM == VoiceDiagnostic::kCandidateCapacity, "SDK candidate capacity");
static_assert(sizeof(esp_mn_results_t::string) == sizeof(VoiceDiagnostic::string), "Constrained text bound");
static_assert(sizeof(esp_mn_results_t::raw_string) == sizeof(VoiceDiagnostic::rawString), "Raw text bound");
static_assert(std::is_trivially_copyable<VoiceDiagnostic>::value, "Queue transports owned bytes");

int main() {
  esp_mn_results_t result{};
  result.state = ESP_MN_STATE_DETECTED;
  result.num = 5;
  std::strcpy(result.string, "RfD gNFb");
  std::strcpy(result.raw_string, "RfD gNFb");
  for (size_t i = 0; i < 5; ++i) {
    result.command_id[i] = int(i + 1);
    result.phrase_id[i] = int(10 + i);
    result.prob[i] = 0.9f - float(i) * 0.1f;
  }
  VoiceDiagnostic copied;
  copied.detectState = ESP_MN_STATE_DETECTED;
  copyVoiceDiagnosticResults(copied, &result);
  assert(copied.candidatesValid && copied.candidateCount == 5);
  for (size_t i = 0; i < 5; ++i) {
    assert(copied.commandIds[i] == result.command_id[i]);
    assert(copied.phraseIds[i] == result.phrase_id[i]);
    assert(copied.probabilities[i] == result.prob[i]);
  }
  assert(!std::strcmp(copied.string, "RfD gNFb"));
  assert(!std::strcmp(copied.rawString, "RfD gNFb"));

  // A getter returns model-owned storage. A later SDK clean cannot change the
  // copied queue item (and probabilities retain actual SDK values).
  std::memset(&result, 0, sizeof(result));
  assert(copied.commandIds[0] == 1 && copied.probabilities[0] == 0.9f);
  assert(!std::strcmp(copied.rawString, "RfD gNFb"));

  std::memset(result.string, 'x', sizeof(result.string));
  std::memset(result.raw_string, 'y', sizeof(result.raw_string));
  result.raw_string[255] = '\0';
  copyVoiceDiagnosticResults(copied, &result);
  assert(copied.stringTruncated && !copied.rawStringTruncated);
  assert(std::strlen(copied.string) == 255 && copied.string[255] == '\0');
  assert(std::strlen(copied.rawString) == 255 && copied.rawString[255] == '\0');
  char narrow[4];
  const char shortInput[] = "abcdef";
  assert(copyVoiceDiagnosticText(narrow, shortInput));
  assert(!std::strcmp(narrow, "abc"));
  char wider[10];
  const char nonTerminated[] = {'a', 'b'};
  assert(copyVoiceDiagnosticText(wider, nonTerminated));
  assert(!std::strcmp(wider, "ab"));

  // The array bytes may be stale between sessions. DETECTING and TIMEOUT never
  // expose candidate IDs/scores, even if a stale num is positive.
  result.num = 5;
  result.command_id[0] = 7;
  result.phrase_id[0] = 42;
  result.prob[0] = 0.99f;
  for (esp_mn_state_t state : {ESP_MN_STATE_DETECTING, ESP_MN_STATE_TIMEOUT}) {
    result.state = state;
    copyVoiceDiagnosticResults(copied, &result);
    assert(copied.apiResultCount == 5 && copied.modelState == int(state));
    assert(!copied.candidatesValid && copied.candidateCount == 0);
    for (size_t i = 0; i < 5; ++i) {
      assert(copied.commandIds[i] == 0 && copied.phraseIds[i] == -1 && copied.probabilities[i] == 0);
    }
  }
  result.state = ESP_MN_STATE_DETECTED;
  result.num = -1;
  copyVoiceDiagnosticResults(copied, &result);
  assert(copied.apiResultCount == -1 && !copied.candidatesValid && !copied.candidateCount);
  result.num = 0;
  copyVoiceDiagnosticResults(copied, &result);
  assert(!copied.candidatesValid && !copied.candidateCount);
  result.num = 100; // Defensive clamp avoids reading outside the SDK arrays.
  copyVoiceDiagnosticResults(copied, &result);
  assert(copied.apiResultCount == 100 && copied.candidateCount == 5 && copied.candidatesValid);
  assert(copied.commandIds[0] == 7 && copied.phraseIds[0] == 42 && copied.probabilities[0] == 0.99f);
  // A clean() can leave SDK state/num stale before another detect() is called.
  copied.detectState = -1;
  copyVoiceDiagnosticResults(copied, &result);
  assert(!copied.candidatesValid && !copied.candidateCount);
  copied.detectState = ESP_MN_STATE_DETECTING;
  copyVoiceDiagnosticResults(copied, &result);
  assert(!copied.candidatesValid && !copied.candidateCount);
  copyVoiceDiagnosticResults<esp_mn_results_t>(copied, nullptr);
  assert(copied.modelState == -1 && !copied.candidateCount && !copied.candidatesValid);
  assert(!copied.string[0] && !copied.rawString[0] && !copied.stringTruncated && !copied.rawStringTruncated);
  assert(!std::strcmp(voiceDiagnosticReasonName(VoiceDiagnosticReason::Timeout), "timeout"));
  assert(!std::strcmp(voiceDiagnosticReasonName(VoiceDiagnosticReason::SilenceTail), "silence_tail"));
  puts("PASS: exact SDK shape, owned snapshots, bounded/NUL text, candidate clamps and stale-score exclusion");
}
