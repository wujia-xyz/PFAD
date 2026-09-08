# Data preparation

The study uses the 14 lower-limb specimens in UltraBones100k: foot, tibia, and fibula, with 42 specimen–anatomy cases and 135 acquisition records. Keep the dataset's specimen and record identifiers unchanged.

The upstream dataset and download instructions are available at:

- https://github.com/luohwu/UltraBones100k
- https://huggingface.co/datasets/luohwu/UltraBones100k

The dataset is distributed by its authors under CC BY 4.0. Cite the UltraBones100k paper when using it. The PFAD preparation utility reads the specimen ZIPs from Hugging Face and extracts the tracking records, predicted masks, and CT bone meshes needed by this study. It does not download raw ultrasound images or unrelated reconstruction outputs.

```text
data/UltraBones100k/
  specimen01/
    CT_bone_segmentations/
      foot.stl
      tibia.stl
      fibula.stl
    ultrasound_records/
      foot/
        record05/
          tracking.csv
          Labels_pred/
            <timestamp>_label_pred.png
```

Record identifiers differ across specimens and anatomies. The software discovers them from the filesystem; do not rename them to make a specimen appear to have the same records as another.

```bash
python scripts/prepare_ultrabones_subset.py --output-root data/UltraBones100k
```

The preparation utility supports repeated `--specimen` and `--anatomy` arguments. The full upstream specimen archives are roughly 42 GB in total; the selected extraction stores only the required subset. Network transfer depends on remote ZIP access and caching.

An existing copy can be used without moving it:

```bash
export PFAD_DATA_ROOT=/path/to/UltraBones100k
python scripts/reproduce.py prepare --specimen specimen01 --anatomy foot
```

`prepare` writes CT-free feature archives under `results/first_arrival_refinement/rebuild_v2/features/` and separate CT teacher archives under `teacher_labels/`. Each outer training run opens teacher archives only for its training specimens. Generating a teacher archive does not authorize its use as a held-out input.

The primary input is the upstream released prediction, not a segmentation model retrained in this repository. The results therefore characterize the PFAD reconstruction stage on that released resource.
