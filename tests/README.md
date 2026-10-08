# Public regression tests

From the repository root, with the app's runtime dependencies installed:

```console
python -m unittest discover -s tests -v
```

The current 187 tests use generated sensor values, target windows, training
records and temporary SQLite databases/model files. They require no operational
database, saved research snapshot, fitted deployment assets, credentials or network.
Dashboard imports and its tested pure functions run with storage, network,
server, browser, and thread startup entry points blocked.

Coverage includes exact issue/target clocks, final-bucket closure, incomplete
targets, causal receipt-visible inputs, native Ridge/HGB/logistic outputs, nested
arrival ≥20/≥40 tails, direct concentration publication, unavailable states,
immutable issue records, Coach/RLCD projection and retained Current observations.
Node.js is required for the integration checks executing the real embedded web
JavaScript. Some unrelated heavy models and provider calls are mocked to isolate
publication seams. This is software verification, not forecast-accuracy evidence.

176 checks come from the current 15-module synthetic production suite; 11 retained
checks cover clock, scoring and import/issue contracts. September snapshot tests
requiring the former afternoon fallback/selection and numerical overlay contract
were replaced by current direct-publication coverage. Their original source and
the corresponding old application remain in Git history.

An additional isolated cold HTTP smoke of the packaged server verified the HTML,
Current, initializing analysis, JSON Schema and Markdown-guide routes against an
empty database, with provider networking and background workers disabled. That
check confirms packaging/initialization, not live collector operation or warning skill.
