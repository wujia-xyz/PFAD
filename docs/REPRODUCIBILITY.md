# Reproducing the study

## Primary experiment

All 14 specimens participate in outer leave-one-specimen-out evaluation. Within each outer training cohort, three inner folds select the ranking score and retained fraction. The held-out specimen does not contribute teacher labels to model fitting or policy selection. Specimen means, after equal anatomy averaging, are the independent units in the reported comparisons.

`scripts/reproduce.py train` runs the attentive student with soft targets, seed 17, 12 epochs, and the frozen optimization settings. It first produces the outer predictions and then applies the policy selected from the inner predictions. The allowed retained fractions are 0.85, 0.90, and 0.95.

The original predictions used for the primary results are available in the prediction release. Download these into a fresh repository to replay the paper geometry directly:

```bash
python scripts/fetch_release.py predictions
python scripts/reproduce.py evaluate
```

The fetch utility preserves existing files when their hashes match and refuses to replace a different file. Use a separate checkout when comparing an independent rerun with the released predictions.

## Saved models and numerical replay

The original primary run retained predictions but did not save its network weights. The released PFAD weights are the 14 refits subsequently used in acquisition sensitivity. Their clean selection agreement with the original predictions is at least 99.9396%. They are identified as refits in the release records. Use the original prediction asset for an exact replay of the primary point and surface results; use the model asset for new inference and acquisition sensitivity.

The Set Transformer release contains the selected outer checkpoint for each specimen and the policy chosen by inner validation. Its two SAB blocks and PMA pooling consume the same per-record source evidence and query context as PFAD. The adapted attention is checked against the attributed upstream implementation on unpadded input.

For a fresh comparator run, prepare the complete cohort and restore the original predictions needed for the PFAD replay comparison:

```bash
python scripts/fetch_release.py predictions
python scripts/reproduce.py comparator
```

## Acquisition sensitivity

The protocol in `configs/pfad/tcsvt/acquisition_robustness_v1.json` defines all tested conditions before aggregation. Both predictors and their clean-data policies remain fixed. The stages recompute acquisition features, freeze predictions, evaluate geometry, and aggregate the complete cohort:

```bash
python scripts/fetch_release.py models
python scripts/fetch_release.py predictions
python scripts/reproduce.py sensitivity
```

This workflow also requires all unperturbed features and the original PFAD meshes produced by `evaluate`. It can be computationally expensive. Individual stages remain available in `scripts/audits/pfad_tcsvt/` for resuming an interrupted run.

- Sweep budgets use rounding and a two-record minimum. Nominal 75% and 50% budgets retain 126 and 86 of 135 records in aggregate.
- Frame strides two and four evaluate every sampling phase, retaining original frame indices.
- Pose perturbations apply one independent rigid error to each record, with paired rotation angles in degrees and translation norms in millimeters. The tested magnitudes are 0.5, 1, and 2, with three repeats.

The reference remains the CT-visible surface from the original complete acquisition. The sensitivity endpoints are point-set geometry; perturbed meshes are not part of that analysis. Repeats are averaged within each specimen–anatomy condition before equal anatomy averaging. The primary comparison family has 84 paired tests with Holm adjustment.

## Context ablations and initialization

The context study retains the input dimensions and freezes factorized scoring at 90% retention. It compares full context with zeroed anatomy entries and with only the anatomy entries retained:

```bash
python scripts/audits/pfad_tmi_revision/run_context_ablation.py --device cuda
python scripts/audits/pfad_tmi_revision/run_context_ablation.py --full-only --seed 29 --device cuda
python scripts/audits/pfad_tmi_revision/run_context_ablation.py --full-only --seed 43 --device cuda
```

These commands need the complete feature and teacher archives. Each held-out label archive is opened only after its predictions have been saved. The optional comparison with an original fixed-retention prediction cache runs when that cache is present; it is not required to compute the ablation endpoints. The result release includes the original context and seed records.

Architectural and teacher-target ablations use `pfad selector-ablation-loocv`. Its `--student-variant`, `--teacher-target`, `--score`, and `--retention` arguments expose the studied controls. Keep the fixed-policy ablation separate from the nested policy used for the primary result.

## Registration and UltraBoneUDF

The registration scripts use the frozen source points and released CT geometry to measure recovery of an imposed rigid displacement. They cover the original point-to-point budget, a longer iteration budget, point-to-plane ICP, and a robust point-to-plane objective. Run `python scripts/reproduce.py registration` after restoring the original predictions and dataset. This endpoint is retrospective alignment recovery, not independently measured clinical target registration error.

`scripts/run_ultraboneudf_downstream.py` contains the adapter for the separately installed [UltraBoneUDF reference implementation](https://github.com/luohwu/UltraBoneUDF). The paper uses the first fibula record of each specimen, the official fitting configuration and 30,000 iterations, then retains mesh faces whose centroids are within 2 mm of frozen PFAD support. Installation and commands are detailed in [ULTRABONEUDF.md](ULTRABONEUDF.md).

## Inspect and regenerate results

```bash
python scripts/report_results.py
python scripts/plot_acquisition.py
python scripts/fetch_release.py results
```

The compact result files in Git contain the complete aggregate and specimen values. The result release supplies the supporting case records. The plot script reproduces the single-column acquisition sensitivity figure from the stored summary. It does not retrain a model or select conditions.

The reported clean means of PFAD and the Set Transformer are close. Their nonsignificant differences do not constitute an equivalence or noninferiority test. The repository retains the measured values and tested conditions rather than replacing unsuccessful comparisons with development results.
