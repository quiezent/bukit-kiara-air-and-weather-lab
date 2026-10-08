#pragma once
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
