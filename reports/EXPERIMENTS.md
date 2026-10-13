# SteamWeave experiment report

[中文](EXPERIMENTS.zh-CN.md) · [Protocol](../docs/EXPERIMENT_PROTOCOL.md) · [Raw metrics](fold_metrics.csv) · [Paired predictions](predictions.csv.gz)

## Scope and result provenance

Completed **23 outer evaluations**, **151 model evaluations**, **300 pressure-test cases** and **6 neural fits**. Grouped CV and temporal supervised comparisons use seeds 17/41/73. Neural reconstruction uses only seed 17. Six ablations and pressure tests use the complete five grouped folds of seed 17. Standard deviations below describe fold/run variation, not confidence intervals.

The original paper reports MAE 3.20 adu, MSE 45.57 adu² and MRE 6.76% for its three-branch model ([Table 3](https://www.researchgate.net/publication/349102644_A_multi-branch_deep_neural_network_model_for_failure_prognostics_based_on_multimodal_data)). This literature row is not merged into the local ranking: original splits, data realization, optimization and hardware are not fully matched. The local DNN is an architecture reconstruction with disclosed training choices, not the authors' model checkpoint.

## Primary grouped comparison

| Model | MAE / adu | MSE / adu² | MRE / % | Folds / runs |
| --- | ---: | ---: | ---: | ---: |
| SteamWeave · full | 4.853 ± 0.405 | 49.469 ± 9.868 | 8.742 ± 0.737 | 15 |
| Direct feature fusion | 5.173 ± 0.423 | 52.163 ± 9.903 | 9.201 ± 0.680 | 15 |
| Fixed expert average | 7.569 ± 0.740 | 94.550 ± 18.685 | 15.271 ± 1.318 | 15 |
| RF · numeric (100 trees) | 7.441 ± 0.588 | 110.160 ± 19.143 | 13.260 ± 1.050 | 15 |
| SVR · numeric (C=100) | 6.688 ± 0.650 | 87.704 ± 20.011 | 11.691 ± 1.125 | 15 |
| DNN reconstruction · budgeted | 13.914 ± 1.617 | 333.024 ± 61.000 | 29.357 ± 4.038 | 5 |

![Comparison](figures/model_comparison.png)

The full system's fold-mean MAE is **4.853 adu**. Group-macro MAE is 4.793 adu. All methods share held-out input-trajectory groups and development boundaries. Neural numbers use five fits versus fifteen for the multiseed classical/system comparisons.

## Ablations and paired uncertainty

| Model | MAE / adu | MSE / adu² | MRE / % | Folds / runs |
| --- | ---: | ---: | ---: | ---: |
| SteamWeave · full | 4.919 ± 0.465 | 50.583 ± 11.977 | 8.843 ± 0.865 | 5 |
| No maintenance expert | 4.856 ± 0.427 | 49.070 ± 10.732 | 8.713 ± 0.814 | 5 |
| No freshness metadata | 5.011 ± 0.471 | 51.957 ± 11.471 | 9.056 ± 0.979 | 5 |
| No LLM semantics | 4.894 ± 0.433 | 50.568 ± 11.503 | 8.818 ± 0.770 | 5 |
| No modality dropout | 4.781 ± 0.445 | 48.372 ± 10.958 | 8.650 ± 0.925 | 5 |
| No current text expert | 4.742 ± 0.459 | 47.942 ± 10.722 | 8.573 ± 0.882 | 5 |
| No current curve expert | 6.869 ± 0.707 | 90.729 ± 18.160 | 12.086 ± 1.515 | 5 |

| Comparator removed | Δ MAE / adu | Bootstrap 95% | Holm p |
| --- | ---: | --- | ---: |
| No maintenance expert | -0.068 | [-0.125, -0.014] | 0.0853 |
| No freshness metadata | 0.090 | [0.034, 0.144] | 0.01733 |
| No LLM semantics | -0.019 | [-0.113, 0.074] | 0.8078 |
| No modality dropout | -0.137 | [-0.203, -0.074] | 0.001416 |
| No current text expert | -0.155 | [-0.227, -0.081] | 0.001416 |
| No current curve expert | 1.916 | [1.663, 2.161] | 1.118e-08 |

Positive delta means removing the component worsens MAE; negative delta means the simpler comparator is better. Confidence intervals resample thirty trajectory groups. Signed-rank tests are paired on group-level errors; Holm correction covers the six ablations. Inference assumes the groups are adequate sampling units and should not be generalized beyond this synthetic copy.

![Ablations](figures/ablation_paired.png)

Current report content and LLM semantic features do not have an assumed accuracy benefit. The no-text ablation retains original numeric and earlier timing/maintenance context; it is not the elimination of all text-derived history. No test-set ranking is used to silently replace the production model.

## Missing evidence and delayed reports

![Missing evidence](figures/missing_modality_robustness.png)

Loss masks are paired between the full model and the no-dropout ablation. Artificial loss rates cover 0–100%, with current curve, text or both removed. Trailing contiguous curve outages cover 30/50/70%. Natural missing curves, reused reports, new repairs and horizon bins are reported separately in [pressure_tests.json](pressure_tests.json).

![Delayed reports](figures/report_delay.png)

Report delays of 10/25/50 atu use earlier observed text only, adjusting the observable age context. They are synthetic perturbations. The local data audit found zero naturally observed repair-after-text-repeat examples, so such interaction effects are not claimed as empirically established from natural data.

## Future periods and diagnostic directory-ID splits

| Model | MAE / adu | MSE / adu² | MRE / % | Folds / runs |
| --- | ---: | ---: | ---: | ---: |
| SteamWeave · full | 11.140 ± 0.000 | 187.207 ± 0.000 | 13.246 ± 0.000 | 3 |
| Direct feature fusion | 11.755 ± 0.000 | 217.475 ± 0.000 | 14.253 ± 0.000 | 3 |
| Fixed expert average | 20.069 ± 0.000 | 635.077 ± 0.000 | 21.588 ± 0.000 | 3 |
| RF · numeric (100 trees) | 14.722 ± 0.041 | 297.460 ± 0.853 | 17.623 ± 0.061 | 3 |
| SVR · numeric (C=100) | 22.078 ± 0.000 | 893.094 ± 0.000 | 24.473 ± 0.000 | 3 |
| DNN reconstruction · budgeted | 21.019 ± 0.000 | 656.058 ± 0.000 | 23.294 ± 0.000 | 1 |

Temporal MAE is **11.140 adu**. Three seeded full-model fits have identical point results because the configured histogram learner is deterministic on this fixed temporal partition; they are not independent datasets.

| Model | MAE / adu | MSE / adu² | MRE / % | Folds / runs |
| --- | ---: | ---: | ---: | ---: |
| SteamWeave · full | 3.856 ± 0.264 | 31.503 ± 4.290 | 6.925 ± 0.459 | 5 |
| Direct feature fusion | 3.711 ± 0.454 | 27.827 ± 7.500 | 6.736 ± 0.707 | 5 |
| Fixed expert average | 5.860 ± 0.517 | 58.440 ± 10.435 | 11.517 ± 1.124 | 5 |
| RF · numeric (100 trees) | 5.869 ± 0.637 | 72.386 ± 14.243 | 10.671 ± 1.008 | 5 |
| SVR · numeric (C=100) | 6.576 ± 0.684 | 85.929 ± 16.626 | 11.502 ± 0.982 | 5 |

Directory-ID diagnostic MAE is **3.856 adu**. Identical input trajectories can cross training/test boundaries here, and random record validation differs from grouped development. The lower error is not an independence-controlled generalization result, nor a reproduction of the paper's original data realization.

![Protocols and coverage](figures/protocols_and_coverage.png)

Coverage is empirical and widths are recorded in the JSON/CSV outputs. Calibration is event-based but observations remain dependent. Training times exclude one-time API/feature preparation; cross-hardware speedup relative to the published CPU experiment is not inferred.

## Neural reconstruction budget

The image/TextCNN/numeric dimensions follow the consulted architecture description; missing optimizer, preprocessing and schedule choices are disclosed in each training history. Of 6 fits, **1 reached the 80-epoch cap**. Validation selects the best checkpoint within that budget. These results measure this implementation and budget, not the optimized performance ceiling of the paper's approach. Extra neural seeds and higher budgets remain follow-up work.

## Reproducibility artifacts

- [Frozen manifest](experiment_manifest.json), [data audit](data_audit.json), [paper reference](paper_reference.json).
- [Summary](summary.json), [fold-level metrics](fold_metrics.csv), [paired analysis](paired_analysis.json).
- [All unperturbed paired predictions](predictions.csv.gz), [split/inner-fold manifests](run_manifests/).
- [Pressure conditions and metrics](pressure_tests.json), [neural epoch histories](neural_training_histories.json).
- PNG and editable SVG figures in [figures/](figures/).

## Interpretation

Multimodal availability, report timing, maintenance context and split design should be studied independently. A component's role in the architecture is not proof of its predictive value. This experiment supports reproducible comparisons on a synthetic benchmark with duplicate-aware accounting, not a claim of industrial deployment accuracy or universal superiority over the literature.
