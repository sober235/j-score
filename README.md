# diffusion_model_t1rho

Research code for score-based diffusion reconstruction experiments on multi-coil T1rho MRI data.

## Scope

This repository is a cleaned release snapshot of the training and sampling code. Large datasets, generated results, and machine-specific runtime outputs are intentionally excluded.

## Repository layout

- `main.py` - CLI entry point for training or sampling
- `run_lib.py` - training and sampling loops
- `configs/` - experiment configuration files
- `models/` - score model architectures
- `utils/` - dataset loading, FFT helpers, masks, and I/O utilities
- `op/` - CUDA/C++ ops used by the model

## Environment

This project mixes PyTorch model code with some TensorFlow utility/logging dependencies from the original score-based codebase.

Recommended:

1. Create a Python 3.10 environment.
2. Install a CUDA-matched PyTorch build first.
3. Install the remaining Python packages:

```bash
pip install -r requirements.txt
```

## Required data paths

Hardcoded personal paths were removed from the main training/sampling path. Set dataset and mask locations with environment variables before running.

### T1rho training / sampling

```bash
export T1RHO_TRAIN_KSPACE_FILE=/path/to/train/cor_raw_full.h5
export T1RHO_TRAIN_MAPS_FILE=/path/to/train/t1rho_maps.h5
export T1RHO_SAMPLE_KSPACE_FILE=/path/to/sample/raw_full.h5
export T1RHO_SAMPLE_MAPS_FILE=/path/to/sample/csm_full.h5
export T1RHO_MASK_PATH=/path/to/released/mask/file.mat
```

For public use:

- `T1RHO_MASK_PATH` should point to the actual released mask file on the user's machine.
- The sampling checkpoint is expected under the released checkpoint directory inside `results/` following the code's current lookup logic.
- Please make sure both the checkpoint path and the mask path are updated to the actual storage locations in your local environment before running the code.

### Optional legacy datasets

If you use the legacy fastMRI / cardiac / 5T code paths, also set the corresponding environment variables referenced in `utils/datasets.py`.

## Usage

Train:

```bash
bash train_fastMRI.sh ve
```

Sample:

```bash
bash test_fastMRI.sh ve
```

Outputs are written under the `results/` workdir by default.

## Notes before publishing or reuse

- Dataset schemas are assumed by the loaders and are not fully generalized.
- Several utility scripts remain research-oriented and may need additional cleanup if you plan to support more datasets publicly.
- The custom CUDA ops in `op/` require a compatible compiler/CUDA environment.
