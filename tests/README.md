# Public regression tests

From the repository root, with the app's runtime dependencies installed:

```console
python -m unittest discover -s tests -v
```

The 20 tests use generated sensor values, target windows, and training records.
They require no database, saved research snapshot, credentials, or network.
Dashboard imports and its tested pure functions run with storage, network,
server, browser, and thread startup entry points blocked.

Coverage includes exact issue-clock targets and final-bucket closure, incomplete
targets, causal fresh references and completed-only adaptive weights, independent
arrival/mean application when a peak candidate is absent, afternoon scope
boundaries and retained issue provenance, and classifier fallback when support
is insufficient. Some model preparation and prediction calls are mocked to
isolate the handoff contract. These checks do not establish forecast accuracy,
prospective validation, collector reliability, or HTTP endpoint behavior.
