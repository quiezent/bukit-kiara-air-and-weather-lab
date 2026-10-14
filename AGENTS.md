# Working on Bukit Kiara Air and Weather Lab

Claude (Anthropic) has maintained both the forecasting server and the ESP32 device since 13 October 2026, taking over from the two Codex workstreams at the owner's request. Read [CONTRIBUTING.md](CONTRIBUTING.md), especially **Shared project maintenance**, and [docs/PUBLICATION.md](docs/PUBLICATION.md) before making contributions.

- Releases are generated from the private workspace by a publishing script that re-derives `app/`, the ported `tests/`, `firmware/esp32-rlcd/WeatherDashboard/` and `research/models/` from the private source. Hand edits there are overwritten; make them in the private source, or in the public-only files the latest manifest in `docs/research/` does not list.
- Center the riding decision: advance warning before preparation, +90-minute arrival conditions, then +90–210-minute ride context. Keep first-crossing and arrival-change targets distinct.
- Coordinate server/device contract changes across workstreams. Preserve native numerical meaning, units, original issue/reference/target clocks and unavailable-data behavior in the API, display and speech.
- Inspect the current branch, local changes and remote state before editing or publishing. Preserve the other workstream's contributions; integrate newer commits without force-pushing over shared history.
- Keep local deployment descriptions separate from the dated public source snapshot. A journal update does not publish a new model implementation or firmware release.
- Use first-person journal entries identifying the workstream and evidence actually checked. Distinguish original issued forecasts from reconstructions, software correctness from prediction skill, and requirements from implemented features.
- Use Markdown for human-readable reports and notes. Keep private databases, model artifacts, operational payloads, credentials, network identifiers and private conversation references out of public contributions.

Run checks appropriate to the affected behavior. For documentation, verify links and consistency with the evidence. Code and shared-contract changes require relevant tests and end-to-end checks; passing implementation tests alone does not establish forecasting accuracy.
