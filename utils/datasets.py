import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import FFT2c, IFFT2c, Emat_xyt_complex, crop, get_all_files, normalize_complex


def _require_env_path(env_name, description):
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {env_name} for {description}.")
    return value


class FastMRIKneeDataSet(Dataset):
    _ENV_KEYS = {
        'train': ('FASTMRI_TRAIN_KSPACE_DIR', 'FASTMRI_TRAIN_MAPS_DIR'),
        'test': ('FASTMRI_TEST_KSPACE_DIR', 'FASTMRI_TEST_MAPS_DIR'),
        'sample': ('FASTMRI_SAMPLE_KSPACE_DIR', 'FASTMRI_SAMPLE_MAPS_DIR'),
        'datashift': ('FASTMRI_DATASHIFT_KSPACE_DIR', 'FASTMRI_DATASHIFT_MAPS_DIR'),
    }

    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        if mode not in self._ENV_KEYS:
            raise NotImplementedError

        kspace_key, maps_key = self._ENV_KEYS[mode]
        self.kspace_dir = _require_env_path(kspace_key, f'fastMRI {mode} k-space directory')
        self.maps_dir = _require_env_path(maps_key, f'fastMRI {mode} sensitivity map directory')
        self.mode = mode
        self.file_list = get_all_files(self.kspace_dir)
        self.num_slices = np.zeros((len(self.file_list),), dtype=int)

        for idx, file in enumerate(self.file_list):
            file_path = os.path.join(self.kspace_dir, file)
            with h5py.File(file_path, 'r') as data:
                self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        slice_idx = int(idx) if scan_idx == 0 else int(idx - self.slice_mapper[scan_idx] + self.num_slices[scan_idx] - 1)

        maps_file = os.path.join(self.maps_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            maps_idx = np.asarray(data['s_maps'][slice_idx])
            maps_idx = crop(np.expand_dims(maps_idx, 0), cropx=320, cropy=320)
            maps = np.squeeze(maps_idx, 0)

        raw_file = os.path.join(self.kspace_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            ksp_idx = np.asarray(data['kspace'][slice_idx])
            ksp_idx = crop(IFFT2c(np.expand_dims(ksp_idx, 0)), cropx=320, cropy=320)
            ksp_idx = np.squeeze(FFT2c(ksp_idx), 0)
            if self.config.data.normalize_type == 'minmax':
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
                kspace = np.asarray(Emat_xyt_complex(img_idx, False, maps, 1))
            elif self.config.data.normalize_type == 'std':
                minv = np.std(ksp_idx)
                kspace = np.asarray(ksp_idx / (self.config.data.normalize_coeff * minv))
            else:
                raise ValueError(f"Unsupported normalize_type: {self.config.data.normalize_type}")

        return kspace, maps

    def __len__(self):
        return int(np.sum(self.num_slices))


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


class T1rhoDataSet_h5_5T(Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        if mode == 'train':
            self.kspace_file = _require_env_path('T1RHO_5T_TRAIN_KSPACE_FILE', 'T1rho 5T training k-space file')
            self.maps_file = _require_env_path('T1RHO_5T_TRAIN_MAPS_FILE', 'T1rho 5T training sensitivity map file')
        elif mode == 'sample':
            self.kspace_file = _require_env_path('T1RHO_5T_SAMPLE_KSPACE_FILE', 'T1rho 5T sampling k-space file')
            self.maps_file = _require_env_path('T1RHO_5T_SAMPLE_MAPS_FILE', 'T1rho 5T sampling sensitivity map file')
        else:
            raise NotImplementedError

        self.mode = mode
        if self.mode != 'sample':
            with h5py.File(self.kspace_file, 'r') as data:
                self.kspace = np.array(data['kspace'])
                self.kspace = np.transpose(self.kspace, [2, 3, 4, 0, 1])
        else:
            with h5py.File(self.kspace_file, 'r') as data:
                self.kspace = np.array(data['h_kspce'])
                self.kspace = np.transpose(self.kspace, [2, 3, 4, 0, 1])

        if self.mode != 'sample':
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['maps'])
        else:
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['csm_cor'])
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
    if config.data.dataset_name == 'fastMRI_knee':
        dataset = FastMRIKneeDataSet(config, mode)
    elif config.data.dataset_name == 't1rho':
        dataset = T1rhoDataSet_h5(config, mode)
    elif config.data.dataset_name == 't1rho_5T':
        dataset = T1rhoDataSet_h5_5T(config, mode)
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")

    if mode == 'train':
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        data = DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=False, pin_memory=True)

    print(mode, 'data loaded')
    return data
