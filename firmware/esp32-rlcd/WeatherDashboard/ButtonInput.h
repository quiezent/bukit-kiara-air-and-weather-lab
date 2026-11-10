#pragma once
#include <atomic>
#include <cstdint>

class ReleasedButton {
 public:
  bool update(bool released, uint32_t now) {
    if (released != raw_) { raw_ = released; changedAt_ = now; }
    if (uint32_t(now - changedAt_) < 40 || released == stable_) return false;
    stable_ = released;
    return released;
  }
 private:
  bool raw_ = true;
  bool stable_ = true;
  uint32_t changedAt_ = 0;
};

enum class ButtonTap : uint8_t { Key, Boot };

// Debounced KEY and BOOT taps, kept in the order they happened. A 10 ms timer
// samples the pins off the loop task, so a tap made while loop() is blocked in
// an HTTP transfer is still handled on its next pass. One producer (the
// sampler) and one consumer (the loop); a full ring drops the newest tap.
class ButtonTaps {
 public:
  void sample(bool keyReleased, bool bootReleased, uint32_t now) {
    if (boot_.update(bootReleased, now)) push(ButtonTap::Boot);
    if (key_.update(keyReleased, now)) push(ButtonTap::Key);
  }
  bool take(ButtonTap &tap) {
    const uint32_t tail = tail_.load();
    if (tail == head_.load()) return false;
    tap = taps_[tail % kCapacity];
    tail_.store(tail + 1);
    return true;
  }
  uint32_t dropped() const { return dropped_.load(); }

 private:
  static constexpr uint32_t kCapacity = 8;
  void push(ButtonTap tap) {
    const uint32_t head = head_.load();
    if (head - tail_.load() >= kCapacity) { dropped_.fetch_add(1); return; }
    taps_[head % kCapacity] = tap;
    head_.store(head + 1);
  }
  ReleasedButton key_, boot_;
  ButtonTap taps_[kCapacity] = {};
  std::atomic<uint32_t> head_{0}, tail_{0}, dropped_{0};
};
