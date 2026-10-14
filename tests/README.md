# Public regression tests

From the repository root, with the app's runtime dependencies installed:

```console
python -m unittest discover -s tests -v
```

On 14 October 2026 the suite ran **567 tests: 565 passed and 2 skipped**, in about a minute on Python 3.11 with Node.js 24. The two skipped tests compare the server's spatial model with the research model on the private 13 October snapshot. They run only when `TTDI_RESEARCH_DATA` (or `data/` at the repository root) holds that snapshot.

The tests use generated sensor values, target windows, training records and temporary SQLite databases and model files. They need no operational database, fitted deployment assets, credentials or network. Node.js is required for the checks that execute the dashboard's real embedded JavaScript. Some heavy models and provider calls are mocked to isolate publication seams.

Coverage includes:

- exact issue and target clocks, final-bucket closure and incomplete targets;
- causal, receipt-visible inputs and receipt-aware training;
- native Ridge, HGB, spatial-HGB and logistic outputs, and the nested ≥20/≥40 arrival tails;
- direct concentration publication and unavailable states;
- the neighbour feature builder and its research parity, and the atomic nightly refit;
- immutable issue records and SQLite WAL behavior;
- device (RLCD) projection and retained current observations.

55 modules are ported from the server's own suite, with imports pointed at `app/`. Three are public-only (`test_dashboard_contracts.py`, `test_forecast_clock.py`, `test_fresh_scoring.py`). Five server modules stay private because they need unpublished assets. The [source manifest](../docs/research/2026-10-14-source-release.json) lists them with the reason for each.

Passing tests shows the software does what it says. It is not forecast-accuracy evidence.
