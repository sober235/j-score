import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import FFT2c, IFFT2c, crop


def _require_env_path(env_name, description):
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {env_name} for {description}.")
    return value



class T1rhoDataSet_h5(Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        if mode == 'train':
            self.kspace_file = _require_env_path('T1RHO_TRAIN_KSPACE_FILE', 'T1rho training k-space file')
            self.maps_file = _require_env_path('T1RHO_TRAIN_MAPS_FILE', 'T1rho training sensitivity map file')
        elif mode == 'sample':
            self.kspace_file = _require_env_path('T1RHO_SAMPLE_KSPACE_FILE', 'T1rho sampling k-space file')
            self.maps_file = _require_env_path('T1RHO_SAMPLE_MAPS_FILE', 'T1rho sampling sensitivity map file')
        else:
            raise NotImplementedError

        self.mode = mode
        if self.mode != 'sample':
            with h5py.File(self.kspace_file, 'r') as data:
                remove_idx = [29, 89, 119, 149, 178, 179, 209, 239, 299, 329, 359, 388, 389]
                self.kspace = np.array(data['kspace'])
                self.kspace = np.transpose(self.kspace, [2, 3, 4, 0, 1])
                keep_idx = sorted(set(range(self.kspace.shape[0])) - set(remove_idx))
                self.kspace = self.kspace[keep_idx, ...]
                self.keep_idx = keep_idx
        else:
            with h5py.File(self.kspace_file, 'r') as data:
                self.kspace = np.array(data['raw'])
                self.kspace = np.transpose(self.kspace, [2, 3, 4, 0, 1])

        if self.mode != 'sample':
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['maps'])[self.keep_idx, ...]
        else:
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['csm'])
                self.maps = np.transpose(self.maps, [2, 3, 4, 0, 1])

    def __getitem__(self, index):
        kspace = IFFT2c(self.kspace[index])
        kspace = crop(kspace, 192, 192)
        kspace = FFT2c(kspace)
        kspace = kspace / (1.5 * np.std(kspace))
        maps = crop(torch.from_numpy(self.maps[index]), 192, 192)
        return torch.from_numpy(kspace), maps

    def __len__(self):
        return self.kspace.shape[0]


def get_dataset(config, mode):
    print('Dataset name:', config.data.dataset_name)
    if config.data.dataset_name == 't1rho':
        dataset = T1rhoDataSet_h5(config, mode)
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")

    if mode == 'train':
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        data = DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=False, pin_memory=True)

    print(mode, 'data loaded')
    return data
