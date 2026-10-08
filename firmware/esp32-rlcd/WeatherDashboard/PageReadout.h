#pragma once

#include "Dashboard.h"
#include "WeatherReadout.h"

// A concise summary of the selected page's main values, built on the loop task.
// Only the completed playlist and transcript are handed to playback; it never
// reads the live JSON or sensor state. Missing/old states retain their meaning.
bool buildCurrentPageReadout(const DashboardContext &context, SpeechPlaylist &out,
                            String &spokenText);
