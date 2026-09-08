# Tested environments

The original study used PyTorch 2.11.0. Training and feature work used Python 3.13; geometric evaluation used Python 3.10 because of the available Open3D wheel. Python 3.10 can run the complete public workflow.

| Package | Training environment | Geometry environment |
|---|---:|---:|
| Python | 3.13 | 3.10.20 |
| PyTorch | 2.11.0+cu128 | 2.11.0 |
| NumPy | 2.4.3 | 2.1.3 |
| SciPy | 1.17.1 | 1.15.3 |
| pandas | 3.0.2 | 2.2.3 |
| scikit-image | 0.26.0 | 0.25.2 |
| trimesh | 4.12.2 | 5.1.0 |
| Pillow | 12.1.1 | 11.0.0 |
| Matplotlib | 3.10.8 | 3.10.0 |
| Open3D | — | 0.19.0 |

`pyproject.toml` specifies compatible dependency ranges. These versions describe the actual study environments, rather than implying that every possible dependency combination has been validated.

Install the appropriate PyTorch CPU or CUDA build before PFAD when controlling accelerator dependencies. The model checks run on CPU. Full nested training benefits from a CUDA device; geometric evaluation is largely CPU work.

For the optional UltraBoneUDF experiment, follow that project's own installation instructions and use its pinned reference version. Its implicit fitting and meshing dependencies are not required for the primary PFAD reconstruction.
