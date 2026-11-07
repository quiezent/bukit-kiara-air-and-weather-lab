# Server v25.2.0-public.1, source of 7 November 2026

The application is **v25.2.0-public.1**, matching local build `2026-10-14-eager-event-models-v25.2.0`. The device API stays **RLCD schema 1**. Installation is still described in the [October 8 release guide](SERVER_RELEASE_2026_10_08.md). I am Claude; I have maintained the server since 13 October.

**The build label understates what is in this release.** It was not bumped for three later changes: the Afternoon source-age fix (17 October), the fire-hotspot collector (21 October) and PatchTST runtime revision v3 (24 October). The [source manifest](research/2026-10-14-eager-event-models-v25.2.0-source-release.json) gives the SHA-256 of every source file this release generates in `app/` and `research/models/`.

## What changed since v25.1.0-public.1

**Recall-weighted event models (v25.2.0, 14 October).** The owner wanted more warnings of a coming fall, even at the cost of false alarms, because a correct fall warning is a chance to ride. Both event models now carry that preference in their training loss, and only there. Outputs are still the literal class probabilities and argmax, with no threshold. Every output carries `calibrated: false` and a `recallWeighting` block that records the weights.

| Output | v25.1.0 | v25.2.0 |
|---|---|---|
| First ≥20 crossing within 90 min | First-crossing HGB v2 (frozen assets) | [`first_crossing_logistic_model.py`](../app/first_crossing_logistic_model.py), `p1_logistic_first20_90min_recall_f12_r6_v1`: multinomial logistic, C = 0.001, 82 inputs from the spatial model's feature builder, class weights drop 12 / none 1 / rise 6, refit nightly on 28 days |
| ≥20/≥40 change by arrival | Arrival logistic v1 | [`arrival_logistic_model.py`](../app/arrival_logistic_model.py) v2, `receipt_visible_logistic_arrival_change20_40_recall_f2_v2`: the same recipe with both fall classes weighted 2 |

The weights were chosen on the 10–23 September tune folds, scoring each candidate on one objective: 5 × missed ≥20 falls + 2.5 × missed rises + false-call clusters + wrong-way clusters. The new first-crossing model trains from history the server collects, so it no longer needs private frozen assets. It needs the neighbour archive, issued wind and at least 300 completed training issues. There is a known side effect. With the 12× drop weight, the model calls "drop" almost always above about 150 µg/m³: 92% of the time this season. In the September–October development data, a fall followed at those levels only about a third of the time.

**Other changes:**
- **Afternoon card blanking fixed.** The source-age check was judged at generation time instead of the issue clock, which blanked the card on about 1.5% of issues. It is now judged at the issue clock, as for every other model.
- **Fire-hotspot collector** ([`firms_collector.py`](../app/firms_collector.py)). Every 30 minutes it stores NASA FIRMS near-real-time detections over Sumatra, the Peninsula and Borneo (94–120°E, 8°S–8°N; three VIIRS satellites and MODIS) in its own SQLite archive. Each detection keeps the clock of its first receipt. No forecast uses the data yet; it exists to separate imported smoke from dry weather.
  - It needs a free FIRMS MAP KEY in `app/private/firms.json` (`{"mapKey": "..."}`), which is git-ignored.
  - Without a key it logs one line and sleeps. The key is never logged or stored, and URLs and errors are stored with it masked.
- **PatchTST runtime revision v3.** A training seal on 20 October–3 November was left over from a retired research protocol. It was starving the Afternoon model's rolling 14-day fit, down to 10 target days, and one day the card read 0.0. Revision v3 lifts the seal through a declared override and leaves the hash-chained original untouched. A fresh clone still lacks the private PatchTST prefix and weights, so that card stays unavailable there.
- **Research harness.** It can now read the v25.2.0 issued records.

## How the published models did on their issued records
The [model card](MODEL_CARD.md#the-season-on-the-issued-records--7-november-2026) has the full table. In short, over 14 October to 7 November (25 issue days):
- **+90 point and ride window.** They beat persistence: +90 −0.79 µg/m³ [−1.32, −0.27], and ride mean −1.53 [−2.42, −0.67].
  - For the +90 and ride mean, almost all of the gain came from the haze of 14–23 October; afterwards they tied persistence.
  - The +90 gain falls short of the project's own pre-declared bar: 1.3 µg/m³ over 70 days.
- **First ≥20 crossing.** In the haze it caught 30 of 33 falls, typically an hour ahead, by calling a change on about half of the haze issues. On the tuning objective it is not distinguishable from a plain "call a fall whenever PM ≥ 80" rule.
- **Arrival change.** It rarely called, and mostly late.
- **Morning and Afternoon cards.** Both tie persistence at their canonical clocks. On every 15-minute issue outside the sealed-window days, the Afternoon card was worse than persistence (+3.5 [+0.6, +6.7]). That is suggestive rather than established.

## Verification
From the repository root, `python -m unittest discover -s tests` ran **607 tests**:
- 603 passed and 4 were skipped. The skipped ones are the research-parity tests, which need the private research snapshot and outputs.
- It took about 75 s on Python 3.11.9 with the pinned requirements.
- The suite ports 58 of the server's test modules, including new ones for the first-crossing logistic, the recall weighting and the fire collector.

These checks show the software does what it says. They do not show forecast skill.

## Packaging
The packaging is unchanged from the [October 14 release](SERVER_RELEASE_2026_10_14.md#packaging):
- loopback by default, with `BUKIT_KIARA_*` configuration;
- LAN discovery only with `BUKIT_KIARA_DISCOVERY=1`;
- the `-public.1` suffix;
- tests that import from `app/`.

`app/` holds 70 modules and 31 runtime research files. The fire collector keeps its key in `app/private/` and its SQLite archive in `app/research/external_firms/`. The key file and the archive, with its `-wal`/`-shm` files, are git-ignored.

**Owner's name removed.** In `arrival_logistic_model.py` and `first_crossing_logistic_model.py`, the owner's name in two docstrings and in the evidence summaries the API serves is replaced by "the owner". Every other module is a byte copy.
