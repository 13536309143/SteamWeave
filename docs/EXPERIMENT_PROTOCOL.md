# Frozen experiment protocol

## Information boundary

The prediction target is `nextinslevel.txt`. Available inputs comprise the current four numerical quantities, current plotted transient if present, the available inspection report, and earlier observations of the same device. Report generation timestamps are not assumed known. Duplicate grouping is a dataset audit operation, not an inference feature.

No API call receives target labels. The bundled semantic cache is fixed across all reruns. Numeric/target normalization in the neural reconstruction is learned from the fitting partition only. The fixed lexical vocabulary and pretrained GloVe subset contain no target information.

## Protocols

| Protocol | Outer evaluation | Development boundary | Role |
| --- | --- | --- | --- |
| `grouped_cv` | Five folds of the 30 duplicate-aware input groups | Three development groups for calibration/early stopping; other groups for fitting | Primary comparison |
| `temporal` | Later observations after each device's 80% inspection-event cutoff | Training targets completed before 65% cutoff; middle events for calibration/validation | Future-period generalization |
| `paper_device_cv` | Five folds of fifty directory IDs | Random 5% of development records for validation | Diagnostic alignment with the described paper protocol; copy overlap explicitly logged |

Seeds: 17, 41 and 73 for grouped/temporal supervised comparisons; seed 17 for the device-ID diagnostic. Neural reconstruction uses seed 17 for five grouped folds and one temporal evaluation. It is not a three-seed neural study. Its capped training schedule and validation-selected best epoch are recorded for every fit.

## Models and controlled changes

The primary system trains four experts and uses four-fold grouped out-of-fold predictions for fusion. Classical baselines include direct-feature histogram boosting, fixed expert averaging, numeric RF with 100 trees, and numeric RBF SVR with C=100. The six ablations remove the maintenance expert, freshness metadata, LLM semantic features, fusion modality-dropout augmentation, current text expert, or current image expert. Original numerical maintenance fields remain in the maintenance-expert ablation; it is not a removal of all maintenance information.

The neural baseline reconstructs the published branch dimensions using raw grayscale images, a GloVe TextCNN and the four numerical quantities. Added training choices are Adam, learning rate 0.001, batch size 128, up to 80 epochs, patience 12, train-only z-scores, fine-tuned word embeddings, missing-image zero matrices and a linear head. These choices are disclosed; this is not the authors' original code or an exact published-result reproduction.

## Robustness and uncertainty

Artificial current-modality loss rates: 0, 10%, 30%, 50%, 70%, 100% for image, text and both. Continuous image outages cover the trailing 30%, 50% or 70% of each device's test records. These are paired between the full model and the no-dropout ablation. Delayed-report tests retrieve reports from earlier observations at delays of 10, 25 and 50 atu; unavailable earlier reports are empty. This is a synthetic stress test, not recovered ground-truth report age.

Observed slices include naturally missing curves, reused texts, newly detected repairs and forecast horizons. Quantile intervals are calibrated on held-out target-inspection event maxima. Coverage and width are empirical; dependence and distribution shift prevent interpreting nominal coverage as a deployment guarantee.

## Analysis

Report MAE, MSE, RMSE and MRE (%). Zero-target observations, if present, are explicitly excluded from MRE and counted. Aggregate fold means and seed means separately. Pair model predictions on identical test records. Bootstrap differences over the thirty trajectory groups; do not bootstrap individual rows as independent observations. Ablation signed-rank tests, when reported, use Holm correction across the six comparisons.

Training timing excludes one-time API semantic extraction and feature preparation. Hardware differs from the paper, so no cross-hardware speedup claim is made. Raw split manifests, nested-fold group lists, predictions and training histories support every reported result.
