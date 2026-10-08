// Host-side checks for command routing and the timing behavior that prevents
// a single utterance from advancing several pages. No Arduino dependencies.
#include "../VoiceNavigation.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main() {
  uint8_t page = 0;
  const uint8_t expected[] = {1, 2, 0, 1, 2, 0};
  for (uint8_t target : expected) {
    assert(pageForVoiceCommand(int(VoiceCommand::Next), page, page));
    assert(page == target);
  }
  page = 0;
  const uint8_t backwards[] = {2, 1, 0, 2, 1, 0};
  for (uint8_t target : backwards) {
    assert(pageForVoiceCommand(int(VoiceCommand::Back), page, page));
    assert(page == target);
  }
  for (uint8_t from = 0; from < 3; ++from) {
    page = from;
    assert(pageForVoiceCommand(int(VoiceCommand::Back), page, page));
    assert(pageForVoiceCommand(int(VoiceCommand::Next), page, page));
    assert(page == from);
  }
  // Direct page commands work regardless of the currently displayed page.
  for (uint8_t from = 0; from < 3; ++from) {
    assert(pageForVoiceCommand(int(VoiceCommand::Overview), from, page) && page == 0);
    assert(pageForVoiceCommand(int(VoiceCommand::Sports), from, page) && page == 1);
    assert(pageForVoiceCommand(int(VoiceCommand::History), from, page) && page == 2);
  }
  page = 2;
  assert(!pageForVoiceCommand(0, 1, page) && page == 2);
  assert(!pageForVoiceCommand(99, 1, page) && page == 2);

  // Read Info is an accepted action, not page navigation. Recognition must
  // dispatch it without changing the currently displayed page.
  for (uint8_t from = 0; from < 3; ++from) {
    page = from;
    assert(knownVoiceCommand(int(VoiceCommand::ReadInfo)));
    assert(!pageForVoiceCommand(int(VoiceCommand::ReadInfo), from, page));
    assert(page == from);
  }
  assert(strcmp(voiceCommandName(int(VoiceCommand::ReadInfo)), "read info") == 0);
  for (int command = 1; command <= 6; ++command) assert(knownVoiceCommand(command));
  assert(!knownVoiceCommand(-1));
  assert(!knownVoiceCommand(0));
  assert(!knownVoiceCommand(7));
  assert(!knownVoiceCommand(99));

  // 16 kHz audio: at least 900 ms since detection and 300 ms continuous
  // silence are BOTH required, including when frames are processed in a burst.
  VoiceUtteranceGate gate(14400, 4800);
  assert(gate.ready());
  gate.block();
  gate.observe(true, 20000); // A long utterance is never rearmed by time alone.
  assert(!gate.ready());
  gate.observe(false, 4799);
  assert(!gate.ready());
  gate.observe(false, 1);
  assert(gate.ready());

  gate.block();
  gate.observe(false, 4800); // Silence alone is insufficient before the gap.
  assert(!gate.ready());
  gate.observe(false, 9600);
  assert(gate.ready());

  gate.block();
  gate.observe(true, 14400);
  gate.observe(false, 4000);
  gate.observe(true, 512); // A speech frame breaks the consecutive-silence run.
  gate.observe(false, 4799);
  assert(!gate.ready());
  gate.observe(false, 1);
  assert(gate.ready());

  // Saturating counters do not wrap after a long run of background speech.
  gate.block();
  gate.observe(true, UINT32_MAX);
  gate.observe(true, UINT32_MAX);
  assert(!gate.ready());
  gate.observe(false, UINT32_MAX);
  assert(gate.ready());

  // Discard delayed events after a blocking operation, including millis wrap.
  assert(voiceEventIsFresh(3000, 500, 2500));
  assert(!voiceEventIsFresh(3001, 500, 2500));
  assert(voiceEventIsFresh(100, UINT32_MAX - 99, 2500));
  assert(!voiceEventIsFresh(3000, UINT32_MAX - 99, 2500));
  puts("PASS: page routing, read-info dispatch, utterance rearming, timing saturation, stale-event checks");
}
