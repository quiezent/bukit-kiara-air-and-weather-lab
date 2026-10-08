# Help me make the forecast less wrong

I welcome specific counterexamples and testable alternatives. Please separate a bug report, an atmospheric hypothesis and a demonstrated forecasting improvement—they are different contributions.

For a forecast review, identify the issue time, exact target interval, model/build version, input availability, prediction and completed observation. Keep original issued forecasts separate from historical reconstructions. Do not post private databases or personal information; a synthetic reproducer or an appropriately shareable aggregate is usually a better public starting point.

For a new model, compare identical targets and information cutoffs with persistence and the selected incumbent. Include percentage and concentration error, direction, false clearing, missing-data behavior and results across different dates/events. Overlapping hourly forecasts are not independent events. State which outcomes you already inspected while choosing the model.

For code, run the portable tests, preserve the API's original forecast timestamps and do not silently expand experimental scope. Numerical changes need separate evaluation; passing software tests does not establish predictive skill.

For AI-assisted work, disclose the role of the agent, the evidence it actually checked and the remaining uncertainty. A fresh context can reduce inherited framing, but multiple agents can share blind spots. Debate is useful; measured outcomes decide.

No contribution license or contributor agreement has been selected for this initial public snapshot. Do not submit material you lack permission to share.

## Shared project maintenance

At the owner's request, the forecasting-server and ESP32-device Codex workstreams jointly maintain this project. The owner sets its purpose and retains repository control. Both workstreams contribute to the project direction, shared documentation and developer journal.

| Workstream | Primary responsibility |
|---|---|
| Forecasting server | Sensor collection and database provenance, predictive models, target definitions, evaluation, server APIs and forecasting evidence |
| ESP32 device | Firmware, display, speech recognition, readouts, audio, hardware integration and device evidence |
| Shared | Server/device contract, end-to-end meaning, README, publication scope and project direction |

The primary objective is advance warning of substantial PM2.5 falls or rises before the rider starts preparing. The +90-minute value supports the arrival judgment; the following two-hour forecast provides ride context. Learned ≥20/≥40 arrival-change probabilities are a modeling requirement, not an assertion that those heads are already deployed.

For changes crossing the interface, coordinate model/output semantics, API versions, units, issue/reference/target clocks, unavailable or stale data and the device's payload budget. Verify affected display and spoken behavior against the same issued server data. A presentation change must preserve the model output's meaning; it must not manufacture a probability or silently substitute a different PM2.5 number.

Write journal entries in the contributor's first-person developer voice and identify the workstream. State what that contributor actually verified and attribute findings from the other workstream. Retain earlier chapters and evidence with their original dates; later findings belong in dated additions or clearly identified corrections.

Before publishing, inspect the current remote and local changes, integrate newer commits and review edits to shared files. Use ordinary fast-forward updates or a reviewed pull request, preserving the other workstream's contributions and unrelated local work. Coordinate overlapping edits before publishing them.

Every publication must identify the dated source, firmware and evidence it includes. Keep newer local deployments distinct from the bundled public snapshot. Publish authorized, sanitized material and preserve the existing private-data boundary.
