# J-Score: Joint Distribution Learning with Score-based Diffusion for Accelerating T1rho Mapping

Official PyTorch implementation of the paper:

> **J-Score: Joint Distribution Learning with Score-based Diffusion for Accelerating T1rho Mapping**
> Congcong Liu, Yuanyuan Liu, Chentao Cao, Jing Cheng, Qingyong Zhu, Tian Zhou, Chen Luo, Yanjie Zhu, Haifeng Wang, Zhuo-Xu Cui, Dong Liang
> *IEEE Transactions on Medical Imaging*, 2025
> DOI: [10.1109/TMI.2025.3606660](https://doi.org/10.1109/TMI.2025.3606660)

## Overview

T1rho mapping requires acquiring multiple images at different spin-lock times (TL), making it inherently time-consuming. J-Score accelerates T1rho MRI acquisition by learning the **joint distribution** of multi-coil k-space data across all TL time points using a score-based stochastic differential equation (SDE) framework. The reverse diffusion process with predictor-corrector sampling reconstructs high-quality T1rho maps from heavily undersampled measurements.

Key features:
- Joint score-based diffusion over multi-TL, multi-coil k-space
- VE-SDE (NCSN++) and MSSDE (DDPM) variants
- Predictor-corrector (PC) sampling with Langevin MCMC corrector
- Supports low-frequency ACS conditioning during training

## Repository Layout

```
j-score/
├── configs/                  # Experiment configuration files
│   ├── default_fastMRI_configs.py
│   ├── ve/ncsnpp_continuous.py   # VE-SDE + NCSN++ (T1rho, primary)
│   └── vp/ddpm_continuous.py     # MSSDE + DDPM (fastMRI reference)
├── models/                   # Score model architectures (NCSN++, DDPM)
├── op/                       # Custom CUDA/C++ operations
├── utils/                    # FFT helpers, dataset loader, mask generation
├── data_prepare/             # Data conversion utilities
├── mask/                     # Undersampling mask files (.mat)
├── results/                  # Training outputs and checkpoints
├── main.py                   # CLI entry point
├── run_lib.py                # Training and sampling loops
├── sde_lib.py                # SDE definitions (VE, VP, SubVP, MSSDE)
├── sampling.py               # Predictor-corrector sampler
├── losses.py                 # Score matching loss and optimizer
├── evaluation.py             # Reconstruction quality metrics (NMSE, PSNR, SSIM)
├── train_fastMRI.sh          # Training launch script
├── test_fastMRI.sh           # Sampling launch script
└── freeinstall.sh            # Conda environment setup
```

## Environment Setup

### Option A: One-command setup (recommended)

```bash
bash freeinstall.sh
conda activate jscore
```

### Option B: Manual setup

```bash
conda create --name jscore python=3.10 -y
conda activate jscore
# Install PyTorch with CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** The custom CUDA ops in `op/` require a compatible CUDA compiler (`nvcc`). Ensure your CUDA toolkit version matches your PyTorch build.

## Data Preparation

### Data format

J-Score expects HDF5 (`.h5`) files. If your sensitivity maps are stored as MATLAB v7.3 `.mat` files, convert them first:

```bash
export MAT73_INPUT_FILE=/path/to/csm.mat
export H5_OUTPUT_FILE=/path/to/csm.h5
export MAT73_DATASET_KEY=csm
python data_prepare/save_h5py.py
```

### Required HDF5 keys

| File | Environment Variable | HDF5 Key | Shape |
|------|----------------------|----------|-------|
| Training k-space | `T1RHO_TRAIN_KSPACE_FILE` | `kspace` | `[nx, ny, nt, nc, nsubj]` |
| Training sensitivity maps | `T1RHO_TRAIN_MAPS_FILE` | `maps` | `[nsubj, nt, nc, nx, ny]` |
| Sampling k-space | `T1RHO_SAMPLE_KSPACE_FILE` | `raw` | `[nx, ny, nt, nc, nsubj]` |
| Sampling sensitivity maps | `T1RHO_SAMPLE_MAPS_FILE` | `csm` | `[nx, ny, nt, nc, nsubj]` |
| Sampling mask | `T1RHO_MASK_PATH` | `mask` (in `.mat`) | `[nx, ny]` |

### Undersampling masks

Pre-generated masks for T1rho experiments are stored in `mask/`. Set the environment variable to point to the appropriate mask file:

```bash
export T1RHO_MASK_PATH=/path/to/mask/low_frequency_acs10.mat
```

## Training

Set the required environment variables and launch training:

```bash
export T1RHO_TRAIN_KSPACE_FILE=/path/to/train_kspace.h5
export T1RHO_TRAIN_MAPS_FILE=/path/to/train_maps.h5

# Train with VE-SDE + NCSN++ (T1rho, recommended)
bash train_fastMRI.sh ve

# Resume from an existing checkpoint directory
bash train_fastMRI.sh ve 2022_11_04T23_23_58_ncsnpp_vesde_N_1000
```

Checkpoints are saved under:

```
results/<timestamped_run_id>/checkpoints/checkpoint_<step>.pth
```

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir results/
```

## Sampling / Reconstruction

```bash
export T1RHO_SAMPLE_KSPACE_FILE=/path/to/sample_raw.h5
export T1RHO_SAMPLE_MAPS_FILE=/path/to/sample_csm.h5
export T1RHO_MASK_PATH=/path/to/mask/low_frequency_acs10.mat

# Sample with VE-SDE + NCSN++
bash test_fastMRI.sh ve
```

The sampling script reads the checkpoint from the directory specified by `sampling.folder` and `sampling.ckpt` in the config file (`configs/ve/ncsnpp_continuous.py`). Before running, update these fields to match your checkpoint location:

```python
sampling.folder = '<your_run_id>'   # directory name under results/
sampling.ckpt   = <checkpoint_step> # checkpoint number to load
```

Reconstructed images are saved as `.mat` files under `results/<sampling.folder>/`.

## Evaluation

Compute NMSE, PSNR, and SSIM between reconstructions and reference images:

```bash
python evaluation.py \
    --recon_dir results/<sampling.folder> \
    --gt_file /path/to/ground_truth.mat \
    --recon_key recon \
    --gt_key label
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025jscore,
  title   = {J-Score: Joint Distribution Learning with Score-based Diffusion for Accelerating T1rho Mapping},
  author  = {Liu, Congcong and Liu, Yuanyuan and Cao, Chentao and Cheng, Jing and Zhu, Qingyong and Zhou, Tian and Luo, Chen and Zhu, Yanjie and Wang, Haifeng and Cui, Zhuo-Xu and Liang, Dong},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025},
  doi     = {10.1109/TMI.2025.3606660}
}
```

## Acknowledgements

This codebase builds on [Score-based SDE](https://github.com/yang-song/score_sde_pytorch) by Yang Song et al. and references the [HFS-SDE](https://github.com/Aboriginer/HFS-SDE) codebase by Chentao Cao et al.
