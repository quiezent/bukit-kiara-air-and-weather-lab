#pragma once

#include "VoiceNavigation.h"
#include <cstddef>

// Checked-in MultiNet7 English registrations; edit this table to customize commands.
// Initial phonemes used Espressif's offline converter at commit
// a2bc8a64d9995155024dd871bce2be28dd1794d1, tool/multinet_g2p.py.
// https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/speech_command_recognition/README.html
// The user's 12 spoken phrases plus five pronunciation variants.
// Distinct variant labels share action IDs; primary read commands use REED.
namespace WeatherVoicePhrases {
struct Phrase { VoiceCommand command; const char *text; const char *phonemes; };
inline constexpr Phrase kPhrases[] = {
    {VoiceCommand::Next, "next", "NfKST" },
    {VoiceCommand::Next, "next page", "NfKST Pdq" },
    {VoiceCommand::Back, "back", "BaK" },
    {VoiceCommand::ReadInfo, "read info", "RmD gNFb" },
    {VoiceCommand::ReadInfo, "read page", "RmD Pdq" },
    {VoiceCommand::ReadInfo, "read", "RmD" },
    {VoiceCommand::Overview, "overview", "bVkVYo" },
    {VoiceCommand::Overview, "page one", "Pdq WcN" },
    {VoiceCommand::Sports, "forecast", "FeRKaST" },
    {VoiceCommand::Sports, "page two", "Pdq To" },
    {VoiceCommand::History, "graph", "GRaF" },
    {VoiceCommand::History, "page three", "Pdq vRm" },
    {VoiceCommand::History, "graph alternate", "GRnF" },
    {VoiceCommand::Overview, "overview alternate", "bVcVYo" },
    {VoiceCommand::Sports, "forecast alternate", "FeKnST" },
    {VoiceCommand::ReadInfo, "read page observed", "RfD Pdq" },
    {VoiceCommand::ReadInfo, "read alternate", "RfD" },
};
inline constexpr size_t kCount = sizeof(kPhrases) / sizeof(kPhrases[0]);
static_assert(kCount == 17, "Keep the 12 requested phrases and five reviewed variants");
}  // namespace WeatherVoicePhrases
