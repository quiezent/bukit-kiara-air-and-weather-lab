# I checked the rain drop against the warning I actually issued

*9 October 2026 · Forecasting-server workstream*

The owner reported rain in TTDI and across Kuala Lumpur around 15:30. I checked that report against the sensor and the forecasts published beforehand. The sharp PM2.5 phase ran from 148.8 to 55.3 µg/m³ between 15:22:58 and 15:43:58. Outdoor temperature fell from 30.9 to 29.3 °C while relative humidity rose from 52% to 56%. I preserve the owner's rain confirmation as human evidence; the sensor does not measure rainfall.

There was an earlier indication. At 14:14:15, the published arrival distribution gave a 52.6% probability of a ≥20 fall and 20.8% of a ≥40 fall, approximately 72 minutes before the sharp phase. Its own completed +90 target was a fall of 83.6 µg/m³. That helps the preparation decision. But the numerical forecast was 166.6 when the completed arrival proxy was 90.6. Earlier false warnings and a mistaken fall indication during the subsequent rebound remain part of the record. I cannot turn one correct indication into a claim of dependable advance warning.

## I tested the cooling and humidity hypothesis

I froze two probability recipes before fitting: the current regularized multinomial logistic learner with richer causal sensor summaries, and a strongly regularized histogram gradient boosting classifier using the same summaries. The extra inputs contain temperature/humidity window means, differences, slopes, spreads and eight interactions. The learner uses them directly; they do not select or replace a concentration.

I compared the same 3,232 historical issues on 34 dates, through 7 October. These are exposed chronological development comparisons, not independent confirmation. The already inspected 8 and 9 October events remain separate diagnostics.

| Arrival probability model | Multiclass Brier ↓ | Log loss ↓ |
|---|---:|---:|
| Current scalar logistic | 0.155885 | 0.365475 |
| Richer logistic | 0.164004 | 0.392122 |
| Richer HGB | 0.155886 | 0.378399 |

The richer inputs did not improve the overall probability forecast. They also failed to provide stronger early ≥40 warning in the 9 October replay. I retained all five declared evaluation thresholds, false endpoint calls, movement matches and whether each warning was correct at its own arrival. I did not choose a favorable threshold after scoring.

I also tested a fixed HGB numerical learner with and without the richer inputs. Historical MAE was 8.489 and 8.521 µg/m³ respectively, against 8.112 for a receipt-aware reconstruction of the fixed Ridge recipe. That Ridge reconstruction is not the originally fitted v3 asset: its historical input availability is corrected, and its outputs can differ. None of the reconstructed numerical candidates warned of a ≥20 or ≥40 fall across all 98 original pre-onset issue clocks examined. Matching those clocks and input hashes does not make a later-fitted model an original issued forecast.

I am sharing the [pure feature transform and numerical prototype](../../app/research/rain_precursors_20261009/README.md). They remain offline research, with their failures disclosed.

## I repaired the training data path

The numerical model's refresh read the current sensor table without receipt or revision metadata. A correction visible later could therefore replace the value used for an earlier training origin. I reproduced this in temporary SQLite: an original 20 reading became 200 after a later correction. The old training path used 200 at the earlier issue; the corrected path retains 20 there, while legitimately using 200 as a completed outcome when it was available by the fit cutoff.

The published and locally deployed **v24.0.4** uses a separately identified [Ridge v4 module](../../app/fresh_numeric_model_v4.py) and asset. It preserves the 208 inputs, alpha=100, four arrival/ride targets, daily midnight cutoff and 330-minute training embargo. Inputs select confirmed revisions at each original origin. Labels use the latest revision confirmed by the training cutoff. Partial receipt ledgers fail closed, and untracked legacy receipt history remains unknown. The model's actual fit time is retained. The adapter continues to copy native outputs.

Eight new synthetic checks and the affected local suite passed. The public package passed **236 tests**. Independent review challenged the partial-ledger and metadata behavior before deployment. These tests establish the repair's software behavior; they do not establish better rain forecasting. The existing daily fit can learn the newly completed event after the next midnight cutoff without backdating the owner's rain report into an earlier input.

## What I still cannot claim

Small local temperature/humidity shifts appeared shortly before the sharp phase; strong cooling mostly accompanied it. Archived weather predicted rain earlier, and radar and other PM stations offer plausible upstream evidence. The current learners do not use these sources. Strict confirmed weather history begins on 8 October, while the radar and neighboring-observation archives have weaker availability evidence. I cannot represent those inputs as a long, verified pre-onset training history.

The arrival target also interpolates complete 15-minute medians. It can smooth a sudden collapse and differs from the sensor reading at the moment the rider arrives. A separately versioned fresh-arrival target deserves evaluation; I have not changed or relabelled today's targets.

I corrected a real learning-data defect and kept the unsuccessful candidates out of the live forecast. Preparation-scale warning of large drops and rebounds remains an open prediction problem. The [release note](../SERVER_RAIN_TRAINING_2026_10_09.md) records the actual source and verification scope. Private observations, input snapshots and fitted weights remain local.
