# Fixed rain-precursor development prototypes

These two reusable Python modules are **offline research**. They are not imported by the live dashboard. Original code falls under [the forecasting Apache-2.0 license](../../LICENSE.md).

`precursor_features.transform(frame, base_columns)` adds 32 continuous temperature/humidity summaries and interactions to a DataFrame of the app's causal `fresh_sensor_features` columns. Use the 28 base columns for the probability study (60 total), or all 208 columns for the numerical study (240 total). It requires the existing issue-relative 3-minute sequence inputs and leaves missing values missing. It performs no fitting, target construction, source retrieval, imputation or prediction selection.

`numeric_candidate.ArrivalRegressor` fits one fixed HistGradientBoostingRegressor to physical +90 arrival targets minus the same issue's fresh reference. Its `predict` returns physical concentrations through the estimator's own inverse target transform. Its fixed recipe is 100 iterations, learning rate 0.05, seven leaves/depth three, minimum 80 samples per leaf, L2=10, squared error, no early stopping, seed 20261009. It does not clip, select, replace or post-adjust predictions. Supply the column index of `freshPm25` to its constructor.

The probability recipes used the current app's multinomial logistic parameters (`arrival_logistic_model.PARAMETERS`), with training-only median imputation/missing indicators and StandardScaler; and a HistGradientBoostingClassifier with the same tree parameters above and `loss="log_loss"`. Native five-class mass and inclusive ≥20/≥40 tails were preserved. No class midpoint generated a concentration.

The study froze recipes before fitting. It used daily Kuala Lumpur midnight cutoffs, rolling 28 days, aligned five-minute training origins and separate fifteen-minute+137-second evaluation clocks. Probability/HGB-arrival training had a 120-minute origin embargo and fully completed arrival labels; the paired fixed Ridge recipe retained its 330-minute embargo and joint four-target eligibility. Preserve known receipt availability at every origin and fit preprocessing only on the training slice. Historical receipt-unknown data cannot establish original live availability.

| Exposed development comparison, 3,232 cases / 34 dates through 7 October | Result |
|---|---:|
| Scalar logistic Brier / log loss | 0.155885 / 0.365475 |
| Richer logistic Brier / log loss | 0.164004 / 0.392122 |
| Richer HGB classifier Brier / log loss | 0.155886 / 0.378399 |
| HGB numerical MAE, 240 inputs | 8.489 µg/m³ |
| Same HGB numerical MAE, 208 inputs | 8.521 µg/m³ |
| Receipt-aware fixed Ridge reconstruction MAE | 8.112 µg/m³ |

The candidates did not warrant deployment. The inspected 8 and 9 October drops remained separate diagnostics; they were not untouched confirmation. Evaluate each warning against its own completed +90 target, pre-onset lead, missed episodes, all declared thresholds and false endpoint warnings. Overlapping origins are not independent events. The private input archive and fitted study states are not bundled, so these published aggregates are not a reproducible public benchmark dataset. The [journal](../../../docs/journal/2026-10-09-learning-from-a-rain-drop.md) describes the study and repair separately.
