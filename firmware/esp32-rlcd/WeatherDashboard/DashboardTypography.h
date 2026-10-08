#pragma once
#include <Arduino.h>
#include <U8g2lib.h>

namespace DashboardTypography {
// Proportional bold letters retain their counters; a pixel of tracking keeps
// adjacent strokes apart on the reflective panel.
inline bool tracked(U8G2 &gfx) {
  return gfx.getU8g2()->font == u8g2_font_helvB08_tf;
}

inline int width(U8G2 &gfx, const char *line) {
  int result = gfx.getUTF8Width(line);
  if (!tracked(gfx)) return result;
  int glyphs = 0;
  for (const uint8_t *p = reinterpret_cast<const uint8_t *>(line); *p; ++p) {
    if ((*p & 0xc0) != 0x80) ++glyphs;
  }
  return result + (glyphs > 0 ? glyphs - 1 : 0);
}

inline void draw(U8G2 &gfx, int x, int baseline, const char *line) {
  if (!tracked(gfx)) {
    gfx.drawUTF8(x, baseline, line);
    return;
  }
  const char *p = line;
  while (*p) {
    // Keep multibyte µ, ³ and ° together when adding letter spacing.
    char glyph[5] = {};
    unsigned int bytes = 1;
    while (bytes < 4 && p[bytes] && (uint8_t(p[bytes]) & 0xc0) == 0x80) ++bytes;
    for (unsigned int i = 0; i < bytes; ++i) glyph[i] = p[i];
    x += gfx.drawUTF8(x, baseline, glyph) + 1;
    p += bytes;
  }
}

inline void text(U8G2 &gfx, int x, int baseline, String line, int maxWidth = 380) {
  while (line.length() && width(gfx, line.c_str()) > maxWidth) {
    unsigned int last = line.length() - 1;
    while (last && (uint8_t(line[last]) & 0xc0) == 0x80) --last;
    line.remove(last);
  }
  draw(gfx, x, baseline, line.c_str());
}
}
