# Forecasting server source release — 8 October 2026

The current source has the [October 9 device-contract update](SERVER_UPDATE_2026_10_09.md), v24.0.3-public.1. This guide retains the October 8 model/setup scope.

This release publishes the forecasting server and web dashboard at **v24.0.2-public.1**, corresponding to the October 8 local **v24.0.2** deployment. The Coach/environment-evidence API is **1.37.0**. It replaces the earlier September 21 application snapshot with the current application module closure; it does not bundle the operational database or fitted models.

Earlier journal entries describe the work as it stood at their dates. The [earlier preparation review](journal/2026-10-08-warning-before-the-ride.md) and its v23 model-card evidence remain historical records. The source published by this release now includes the separate learned arrival-change distribution described below.

## What the source implements

The primary decision is whether to start preparing now for a ride beginning about 90 minutes later. The +90 concentration supports the arrival judgment; the following two-hour mean and minimum–maximum provide ride context.

| Output | Fixed model and target |
|---|---|
| PM2.5 at +90 minutes | `fresh_sequence_ridge_delta_numeric_v3_20261008`; exact arrival proxy interpolated from complete 15-minute sensor medians |
| PM2.5 mean over +90–210 minutes | The same Ridge learner; duration-weighted complete median proxy over the ride window |
| Lowest–highest PM2.5 over +90–210 minutes | The same learner's minimum and maximum outputs; complete 15-minute median outcomes overlapping the window |
| Fall/rise ≥20 and ≥40 on arrival | `receipt_visible_logistic_arrival_change20_40_v1`; native five-class distribution relative to the issue's fresh five-minute sensor reference |
| First sampled ≥20 rise/drop before arrival | `fresh_sensor_hgb_direct_first20_90min_v2`; retained in the APIs and issued archive, with its web subsection removed |
| Morning session | Separate fixed HGB model for the exact dated two-hour session |
| Afternoon session | Guarded PatchTST implementation; the additional provisioning limitation below applies |

The arrival classifier learns five disjoint outcomes: fall ≥40, fall from ≥20 to <40, movement strictly within ±20, rise from ≥20 to <40, and rise ≥40 µg/m³. Its native class probabilities supply coherent ≥20/≥40 tails. The ≥40 result is learned from its own class outcomes, rather than scaled from a ≥20 percentage. The current fixed logistic model uses 28 scalar sensor/time features. Archived weather and neighboring observations are context, not inputs to that classifier.

The arrival-change target is the concentration **at +90 minutes**. The retained first-crossing model answers whether a sampled ≥20 crossing happens **anywhere before +90**, even if that change reverses. Keep these targets distinct when reusing the APIs.

Publication copies the fixed learners' numbers. There is no numerical persistence fallback, runtime performance selector, deadband, range sorting or display-created probability. A missing valid model output remains unavailable. The five-minute issue reference can differ from the latest single Current reading. A refreshing dashboard also advances the forecast issue and its arrival target; retaining one preparation plan across rechecks is not implemented by this release.

The ride minimum–maximum is not a confidence interval or an instantaneous extreme. Partial edge medians can extend outside the exact ride interval, and short spikes can differ. The separately retained probability that the ride average is ≤70 µg/m³ comes from its own model, rather than from that range.

## Install and run

Use **Python 3.11**. From the repository root:

```console
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python app/BukitKiara_Dashboard.py --no-browser
```

Open [the dashboard on this computer](http://127.0.0.1:8765). Collection continues while the Python process runs, independently of browser tabs. Stopping the process or sleeping its host interrupts collection. This package does not install a Windows service.

| Environment variable | Default / purpose |
|---|---|
| `BUKIT_KIARA_HOST` | `127.0.0.1`; listen on this computer only |
| `BUKIT_KIARA_PORT` | `8765`; HTTP port |
| `BUKIT_KIARA_LOCATION_ID` | The bundled public TTDI AirGradient station ID |
| `BUKIT_KIARA_LATITUDE` | The bundled public TTDI weather latitude |
| `BUKIT_KIARA_LONGITUDE` | The bundled public TTDI weather longitude |
| `BUKIT_KIARA_DATA_DIR` | Application directory; location for the SQLite database and radar archive. Other model/collector caches remain beside their source modules and are ignored by Git. |
| `BUKIT_KIARA_DISCOVERY` | `0`; set to `1` to enable LAN mDNS discovery |

Change the station ID and both weather coordinates together to use another public station. The defaults describe the already-public TTDI example. Use a separate checkout and data directory for another station so its observations and fitted artifacts do not mix with this station's history. Regional, neighboring-station and radar research helpers retain TTDI-specific defaults; review those before enabling or interpreting that context elsewhere.

To serve other devices, explicitly configure the listening address and, if desired, discovery. For example, in PowerShell:

```powershell
$env:BUKIT_KIARA_HOST = "0.0.0.0"
$env:BUKIT_KIARA_DISCOVERY = "1"
python app/BukitKiara_Dashboard.py --no-browser
```

Loopback remains the default. Changing the listening address does not configure the host firewall or publish an internet service.

## Starting with your own data

No historical database, collected API payloads, credentials or fitted assets ship with this release. A new installation creates its own database, collects sensor readings and issued weather runs, and prepares the primary fresh numerical and probability models from eligible completed history. Model support requirements, complete target windows, daily cutoffs and training embargoes still apply. Initial forecasts can remain unavailable until enough usable observations and dates have accumulated; installing the source does not instantly reproduce the owner's learned estimates.

Retain your original issued forecasts before scoring their outcomes. A historical reconstruction fitted after an inspected episode is development evidence, even if its training labels ended before that episode. Do not relabel that reconstruction as an advance warning.

## Afternoon PatchTST provisioning limitation

The package includes the Afternoon PatchTST runtime and training source. Its current recipe also depends on a **private immutable historical training prefix and a compatible freeze/identity chain**. Those data and fitted assets are excluded from the public package. Therefore, a source-only clone **cannot independently reproduce the current Afternoon model**. Its output remains unavailable without compatible locally provisioned assets satisfying the existing checks.

Installing Torch is necessary for that runtime, but does not supply the missing prefix or frozen model. The optional CPU runtime can be installed separately:

```console
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

Keep the provenance and compatibility guards intact. This release provides no invented replacement asset, weakened freeze check or numerical fallback for Afternoon. A fully self-contained training recipe using a new user's collected history would be a separate implementation and evaluation task.

## Checks and interpretation

Run the packaged tests from the repository root after installing the core requirements:

```console
python -m unittest discover -s tests -v
```

The relevant contracts include exact issue/target clocks, complete outcomes, causal inputs, native model publication, unavailable-data behavior, preservation of arrival probabilities across consumers, immutable issue records and the device payload budget. Node.js is also required for the test that executes the real embedded JavaScript. Read the [test guide](../tests/README.md) for the packaged suite's scope. Private operational fixtures and the complete local research archive are not bundled.

Passing implementation checks does not establish forecasting skill. Sudden-change warning and arrival magnitude remain difficult, and the earlier large-fall reconstruction still exposed a substantial numerical miss. The [model card](MODEL_CARD.md) records dated results and their limitations. Future evaluation needs original pre-event publications, misses and false warnings by severity/direction, useful lead time and independent episodes. Forecasts target the TTDI sensor; accuracy along every Bukit Kiara trail has not been established.

## Reuse and publication boundary

Original application and associated test source covered by [app/LICENSE.md](../app/LICENSE.md) is published under **Apache-2.0**. Upstream dependencies and data retain their own terms. This code release does not relicense journal prose, documentation, photographs or the separately maintained firmware.

The release excludes private databases, fitted assets, operational payloads, network identifiers and private conversations. The [publication record](PUBLICATION.md) describes the repository's broader scope. The [source-release journal](journal/2026-10-08-sharing-the-forecasting-source.md) explains this contribution from the forecasting workstream.
