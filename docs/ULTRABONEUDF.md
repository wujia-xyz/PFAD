# Optional UltraBoneUDF compatibility experiment

This experiment uses the official UltraBoneUDF implementation at commit `b345daee9b6f01b625c979748745cf026bc5a9e9`:

```bash
git clone https://github.com/luohwu/UltraBoneUDF.git third_party/UltraBoneUDF
git -C third_party/UltraBoneUDF checkout b345daee9b6f01b625c979748745cf026bc5a9e9
```

Follow that repository's installation instructions, including its CUDA extensions and DualMesh-UDF dependencies. The official network, loss, configuration, and 30,000 fitting iterations are retained. The PFAD adapter prepares frozen point inputs and evaluates the resulting meshes; it does not modify the official model.

For one specimen, export the frozen fibula predictions:

```bash
python scripts/run_ultraboneudf_downstream.py export \
  --prediction-archive results/first_arrival_refinement/rebuild_v2/selector_attentive_loocv/nested_policy/specimen01_fibula.npz \
  --output-root outputs/udf/specimen01_fibula
```

Select the first entry of `records` in the generated `manifest.json`. For specimen01, this is `record01`:

```bash
python scripts/run_ultraboneudf_downstream.py run-manifest \
  --manifest outputs/udf/specimen01_fibula/manifest.json \
  --run-root outputs/udf/specimen01_fibula/official_i30000_d9 \
  --method raw --record record01 --maxiter 30000 --mesh-depth 9
python scripts/run_ultraboneudf_downstream.py evaluate \
  --manifest outputs/udf/specimen01_fibula/manifest.json \
  --run-root outputs/udf/specimen01_fibula/official_i30000_d9 \
  --dataset-root data/UltraBones100k \
  --method raw --method support_gated_raw --record record01 \
  --support-radius-mm 2 \
  --output outputs/udf/specimen01_fibula/official_i30000_d9/evaluation_first_record_support_gate_2mm.json
```

Repeat the first-record selection for all 14 specimens before aggregation. The evaluated comparison is the original raw mesh versus its post-hoc PFAD support gate. Fitting a second implicit field to the selected point set is a separate experiment and is not the reported compatibility result.

The archive of paper results contains the completed per-specimen evaluations. A full rerun of the optional official model needs its own CUDA toolchain and can take substantially longer than the PFAD inference example.
