# Working on Bukit Kiara Air and Weather Lab

The forecasting-server and ESP32-device Codex workstreams share project maintenance at the owner's request. Read [CONTRIBUTING.md](CONTRIBUTING.md), especially **Shared project maintenance**, and [docs/PUBLICATION.md](docs/PUBLICATION.md) before making contributions.

- Center the riding decision: advance warning before preparation, +90-minute arrival conditions, then +90–210-minute ride context. Keep first-crossing and arrival-change targets distinct.
- Coordinate server/device contract changes across workstreams. Preserve native numerical meaning, units, original issue/reference/target clocks and unavailable-data behavior in the API, display and speech.
- Inspect the current branch, local changes and remote state before editing or publishing. Preserve the other workstream's contributions; integrate newer commits without force-pushing over shared history.
- Keep local deployment descriptions separate from the dated public source snapshot. A journal update does not publish a new model implementation or firmware release.
- Use first-person journal entries identifying the workstream and evidence actually checked. Distinguish original issued forecasts from reconstructions, software correctness from prediction skill, and requirements from implemented features.
- Use Markdown for human-readable reports and notes. Keep private databases, model artifacts, operational payloads, credentials, network identifiers and private conversation references out of public contributions.

Run checks appropriate to the affected behavior. For documentation, verify links and consistency with the evidence. Code and shared-contract changes require relevant tests and end-to-end checks; passing implementation tests alone does not establish forecasting accuracy.
