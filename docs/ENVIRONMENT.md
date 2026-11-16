# Recorded environment and software checks

The experiments were run with 64-bit CPython 3.13.5 on Windows. The retained
training records identify an NVIDIA RTX 4070 Ti SUPER and CUDA 12.6. The public
copy's software checks use CPU tensors and synthetic fixtures; they do not
repeat training, load a retinal dataset or evaluate a trained checkpoint.

## Package versions

These versions were read from the study environment's installed distribution
metadata. They are environment provenance, not a newly tested cross-platform
installation matrix.

| Distribution | Recorded version | Use |
|---|---|---|
| `torch` | `2.11.0+cu126` | Models, losses, tensor operations and saved checkpoints |
| `torchvision` | `0.26.0+cu126` | Geometric transforms in the retained baseline source chain |
| `numpy` | `2.2.6` | Arrays, saved records and resampling indices |
| `scipy` | `1.16.3` | Distance transforms, components and retained analysis helpers |
| `Pillow` | `12.0.0` | Fundus/annotation decoding and dataset-specific nearest projection |
| `opencv-python` | `4.12.0.88` | CLAHE, image resizing and HRF nearest projection |
| `scikit-image` | `0.26.0` | Hard skeletonization |
| `matplotlib` | `3.10.8` | Scientific figure tooling |
| `tzdata` | `2025.3` | IANA data for the standard-library `zoneinfo` time zones on Windows |

`requirements.txt` pins the base PyTorch and Torchvision versions while leaving
the hardware-specific wheel choice explicit. The recorded wheels include the
`+cu126` local build suffix. Choose a matching supported PyTorch/Torchvision
build for the target machine using the
[official PyTorch installation guidance](https://pytorch.org/get-started/locally/),
then install the remaining pinned requirements. A CPU build is sufficient for
the synthetic checks. The retained training execution used the CUDA build.

The requirements list direct dependencies rather than every transitive package.
For example, `packaging` and `PyYAML` were installed in the recorded environment
but are not directly imported by this repository's experiment source. The tests
use Python's built-in `unittest`; `pytest` was not part of the recorded environment
and is not required. No network access or model-weight download is needed by
the synthetic test suite.

## Dependency boundary

The modern model, objective, metric, loader and report modules are imported as
`flow_matching_refiner` modules. The audited OOF loader also reads retained
producer source from `paired_reproduction`, including `official_model.py`, the
three dataset drivers, the full-run composer and the component/OOF helper
chain. `diffusion_refiner/models.py` and `diffusion_refiner/utils.py` supply
legacy model definitions used by that chain.

The dataset-driver files `paired_reproduction/run.py`, `run_drive.py` and
`run_hrf_baseline.py` contain historical top-level execution. Do not import them
as a dependency check or execute them as a test. The portable adapter accesses
only the audited source contract. Real reproduction additionally needs the
configured images, annotations, retained baseline/fold checkpoints and OOF
maps; these are not fabricated by dependency installation or synthetic tests.

## Running the CPU checks

From the repository root, use the following explicit test list:

```console
python -m unittest -v flow_matching_refiner.test_model flow_matching_refiner.test_pixel_model flow_matching_refiner.test_esfm_model flow_matching_refiner.test_structure_metrics flow_matching_refiner.test_refinement_objectives_v3 flow_matching_refiner.test_oof_data_all flow_matching_refiner.test_native_patch_data flow_matching_refiner.test_native_patch_model
```

These checks cover bridge/integrator consistency, loss gradients and masking,
OOF membership safeguards, annotation bitmap placement, refusal to upscale a
non-native label, patch alignment, positive-window tile coverage, dataset-specific
nearest projection and exact zero-residual identity near the FP32 threshold.
They use tiny generated tensors or temporary fixture images. Their synthetic
forward/backward operations are software tests, not new scientific results.

Run the separate portable-path and guarded-entry-point checks with:

```console
python -m unittest discover -s tests -p "test_portable_paths.py" -v
```

These additional checks use temporary text/JSON fixtures and mocked dispatch.
They cover historical alias boundaries and original-source bytes, parent/symlink
escape rejection, output isolation, configuration ordering, missing-input
preflight and the explicit execution guard. Even the test that supplies
`--execute` dispatches only to a mock; it cannot run a scientific phase.

The final recorded run passed all 56 checks: 42 core checks and 14 portability
checks, with no failures or skips. It also compiled all 47 Python files, checked
safe dependency imports and compiled the three legacy source-composer outputs
without executing them. Eleven scientific core modules and 32 selected
scientific functions, including training and prediction functions, retained the
same abstract syntax trees as the preserved source. These facts describe the
recorded public-copy validation, rather than a fresh-install or dataset rerun.

The public-copy validation record is stored separately as a timestamped JSON
file in `docs/`. It identifies the checks actually completed. Compilation and
safe imports do not establish end-to-end dataset reproduction, fresh-install
compatibility, native-grid accuracy, external generalization or numerical
identity on another accelerator.
