# Attribution and material licenses

## PFAD code

The original PFAD implementation, experiment adapters, and documentation are licensed under MIT. See `LICENSE`.

## Set Transformer

`src/pfad_tcsvt/models.py` adapts the Set Transformer attention implementation by Juho Lee and collaborators. The reference is https://github.com/juho-lee/set_transformer at commit `73432c640ac78140496d6738416c54d32c686d65`. Its MIT license is retained in `third_party/set_transformer/LICENSE`, and `modules.py` is included for the numerical fidelity check.

## UltraBones100k and dataset-derived artifacts

The dataset belongs to its original authors and is distributed under Creative Commons Attribution 4.0: https://github.com/luohwu/UltraBones100k and https://creativecommons.org/licenses/by/4.0/ .

The released prediction archives and the ultrasound/CT examples in `assets/pfad_overview.png` derive from this dataset. These materials are distributed with attribution under CC BY 4.0. The predictions contain PFAD processing of the upstream released response masks and tracked geometry; the overview illustrates the PFAD training and reconstruction paths. The PFAD MIT license does not replace the dataset's terms.

Dataset citation: L. Wu et al., “UltraBones100k: A reliable automated labeling method and large-scale dataset for ultrasound-based bone surface extraction,” *Computers in Biology and Medicine*, vol. 194, 2025, article 110435.

## UltraBoneUDF

The optional downstream adapter invokes an independently downloaded UltraBoneUDF installation. Its official code and license are at https://github.com/luohwu/UltraBoneUDF . That implementation is not redistributed as PFAD code.
