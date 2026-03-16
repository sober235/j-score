import os
import sys

import numpy as np
import scipy.io as scio
import torch
from numpy.lib.stride_tricks import as_strided


DEFAULT_MASK_DIR = os.environ.get('MASK_OUTPUT_DIR', 'mask')


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)


def gaussian_random_mask(shape, acc, sample_n):
    """Create a Gaussian-random Cartesian mask."""
    n, nx, ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(nx, 0.5 / (nx / 10.0) ** 2)
    lam = nx / (2.0 * acc)
    n_lines = int(nx / acc)

    pdf_x += lam / nx
    if sample_n:
        pdf_x[nx // 2 - sample_n // 2:nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((n, nx))
    for i in range(n):
        idx = np.random.choice(nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, nx // 2 - sample_n // 2:nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (n, nx, ny), (size * nx, size, 0))
    return mask.reshape(shape)


def _save_mask(filename, mask):
    os.makedirs(DEFAULT_MASK_DIR, exist_ok=True)
    scio.savemat(os.path.join(DEFAULT_MASK_DIR, filename), {'mask': np.squeeze(mask)})


def get_blur_mask(image_size, rate):
    mask = torch.zeros([1, 1, image_size, image_size], dtype=torch.complex128)
    x_start = torch.div(image_size, 2 * rate, rounding_mode='floor')
    x_end = image_size - x_start
    mask[:, :, x_start:x_end, x_start:x_end] = 1.0
    return mask


def get_uniform_random_mask(image_size, acc, acs_lines=18):
    center_line_idx = np.arange((image_size - acs_lines) // 2, (image_size + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(image_size), center_line_idx)
    np.random.shuffle(outer_line_idx)

    lines_num = int(image_size / acc) - acs_lines
    random_line_idx = outer_line_idx[:lines_num]

    mask = np.zeros((image_size,))
    mask[center_line_idx] = 1.0
    mask[random_line_idx] = 1.0
    mask = np.repeat(mask[np.newaxis, :], image_size, axis=0)

    filename = f'random_uniform_acc{acc}_DGM_acs{acs_lines}.mat'
    _save_mask(filename, mask)
    return mask


def get_equispaced_mask(mask_type, acc, acs_lines=16, total_lines=320):
    center_line_idx = np.arange((total_lines - acs_lines) // 2, (total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = outer_line_idx[::acc]

    mask = np.zeros((total_lines,))
    mask[center_line_idx] = 1.0
    mask[random_line_idx] = 0.0 if mask_type == 'low_frequency' else 1.0
    mask = np.repeat(mask[np.newaxis, :], total_lines, axis=0)

    if mask_type == 'low_frequency':
        filename = f'low_frequency_acs{acs_lines}.mat'
    else:
        filename = f'uniform_acc{acc}_acs{acs_lines}.mat'
    _save_mask(filename, mask)
    return mask


def get_cartesian_mask(acc, acs_lines=24, image_size=320):
    shape = (1, image_size, image_size)
    mask = gaussian_random_mask(shape, acc, sample_n=acs_lines)
    mask = np.transpose(mask, (0, 2, 1))
    _save_mask(f'uniform_random_dgm_acc{acc}_acs{acs_lines}.mat', mask)
    print('generate cartesian mask, acc =', acc)
    return mask


def main():
    get_uniform_random_mask(image_size=384, acc=6, acs_lines=24)


if __name__ == '__main__':
    sys.exit(main())
