#include "../VoiceSession.h"
#include <assert.h>
#include <stdio.h>
#include <vector>

int main() {
  // MultiNet's duration is spent by detect() calls, including silent audio.
  // Waiting longer than several entire windows must consume none of it.
  constexpr uint32_t frameSamples = 512;
  VoiceRecognitionSession session(9600); // 600 ms at 16 kHz.
  unsigned modelFrames = 0;
  for (unsigned frame = 0; frame < 2000; ++frame) { // 64 seconds idle.
    assert(session.observe(false, frameSamples)
        == VoiceRecognitionSession::FrameAction::Ignore);
  }
  assert(modelFrames == 0);
  assert(session.observe(true, frameSamples)
      == VoiceRecognitionSession::FrameAction::Start);
  ++modelFrames;
  for (unsigned frame = 0; frame < 20; ++frame) {
    assert(session.observe(true, frameSamples)
        == VoiceRecognitionSession::FrameAction::Continue);
    ++modelFrames;
  }
  assert(modelFrames == 21); // The preceding silence did not consume the window.

  // Final-word context is retained through the full silence tail. A brief
  // intra-utterance pause cannot end or restart recognition.
  for (unsigned frame = 0; frame < 18; ++frame) {
    session.observe(false, frameSamples);
    assert(!session.finished());
  }
  session.observe(true, frameSamples);
  assert(!session.finished());
  for (unsigned frame = 0; frame < 18; ++frame) {
    session.observe(false, frameSamples);
    assert(!session.finished());
  }
  session.observe(false, frameSamples);
  assert(session.finished());
  session.stop();
  assert(session.observe(false, frameSamples)
      == VoiceRecognitionSession::FrameAction::Ignore);
  assert(session.observe(true, frameSamples)
      == VoiceRecognitionSession::FrameAction::Start);
  session.stop();

  // Saturation remains safe even for a long silence batch.
  session.observe(true, frameSamples);
  session.observe(false, UINT32_MAX);
  session.observe(false, UINT32_MAX);
  assert(session.finished());
  session.stop();

  // A cache need not be a whole model frame. Its oldest samples (including the
  // first syllable) must precede current speech without loss or duplication.
  int16_t buffer[frameSamples];
  VoiceFrameAssembler assembler(buffer, frameSamples);
  std::vector<int16_t> source(2048), recognized;
  for (size_t i = 0; i < source.size(); ++i) source[i] = int16_t(i + 1);
  auto consume = [&](int16_t *audio) {
    recognized.insert(recognized.end(), audio, audio + frameSamples);
    return true;
  };
  assert(assembler.append(source.data(), 768, consume)); // AFE cache.
  assert(recognized.size() == frameSamples);
  assert(assembler.append(source.data() + 768, 512, consume)); // First speech frame.
  assert(assembler.append(source.data() + 1280, 768, consume)); // Subsequent audio.
  assert(recognized == source);

  // A detected/timeout result stops consumption of the rest of that utterance.
  // Resetting before the next session must discard its partial/stale samples.
  assembler.reset();
  assert(assembler.append(source.data(), 128, consume));
  unsigned calls = 0;
  assert(!assembler.append(source.data() + 128, 1024, [&](int16_t *) {
    ++calls;
    return false;
  }));
  assert(calls == 1);
  assembler.reset();
  recognized.clear();
  assert(assembler.append(source.data() + 1024, frameSamples, consume));
  assert(recognized.front() == 1025 && recognized.back() == 1536);
  assert(assembler.append(nullptr, 0, consume));
  puts("PASS: idle timeout budget, utterance boundaries, VAD pre-roll continuity, frame assembly");
}
