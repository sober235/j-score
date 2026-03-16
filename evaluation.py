"""Evaluation script for T1rho MRI reconstruction quality.

Computes NMSE, PSNR, and SSIM between reconstructed and reference images.

Usage:
    python evaluation.py \
        --recon_dir results/<sampling.folder> \
        --gt_file /path/to/ground_truth.mat \
        --recon_key recon \
        --gt_key label
"""

import os
import sys
import argparse

import numpy as np
import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_nmse(gt, pred):
    """Normalized Mean Squared Error."""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def compute_psnr(gt, pred):
    """PSNR on magnitude images normalized to [0, 1]."""
    gt_mag = np.abs(gt) / np.max(np.abs(gt))
    pred_mag = np.abs(pred) / np.max(np.abs(gt))
    return peak_signal_noise_ratio(gt_mag, pred_mag, data_range=1.0)


def compute_ssim(gt, pred):
    """SSIM on magnitude images normalized to [0, 1]."""
    gt_mag = np.abs(gt) / np.max(np.abs(gt))
    pred_mag = np.abs(pred) / np.max(np.abs(gt))
    return structural_similarity(gt_mag, pred_mag, data_range=1.0)


def evaluate_folder(recon_dir, gt_file, recon_key='recon', gt_key='label'):
    """Evaluate all .mat reconstructions in recon_dir against ground truth.

    Args:
        recon_dir:  Directory containing reconstruction .mat files.
        gt_file:    Path to the ground truth .mat file.
        recon_key:  HDF5/MAT key for the reconstruction array.
        gt_key:     HDF5/MAT key for the ground truth array.
    """
    gt_data = scio.loadmat(gt_file)[gt_key]

    recon_files = sorted(f for f in os.listdir(recon_dir) if f.endswith('.mat'))
    if not recon_files:
        print(f'No .mat files found in {recon_dir}')
        return

    nmse_list, psnr_list, ssim_list = [], [], []

    for fname in recon_files:
        recon_path = os.path.join(recon_dir, fname)
        pred = np.squeeze(scio.loadmat(recon_path)[recon_key])
        gt = np.squeeze(gt_data)

        nmse_val = compute_nmse(gt, pred)
        psnr_val = compute_psnr(gt, pred)
        ssim_val = compute_ssim(gt, pred)

        nmse_list.append(nmse_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f'[{fname}]  NMSE={nmse_val:.4e}  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}')

    print('\n--- Summary ---')
    print(f'NMSE : {np.mean(nmse_list):.4e} +/- {np.std(nmse_list):.4e}')
    print(f'PSNR : {np.mean(psnr_list):.2f} +/- {np.std(psnr_list):.2f} dB')
    print(f'SSIM : {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MRI reconstruction quality (NMSE, PSNR, SSIM).')
    parser.add_argument('--recon_dir', type=str, required=True,
                        help='Directory containing reconstruction .mat files.')
    parser.add_argument('--gt_file', type=str, required=True,
                        help='Path to ground truth .mat file.')
    parser.add_argument('--recon_key', type=str, default='recon',
                        help='Key for reconstruction array in .mat file (default: recon).')
    parser.add_argument('--gt_key', type=str, default='label',
                        help='Key for ground truth array in .mat file (default: label).')
    args = parser.parse_args()

    evaluate_folder(args.recon_dir, args.gt_file, args.recon_key, args.gt_key)


if __name__ == '__main__':
    sys.exit(main())
