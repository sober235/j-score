import os
import sys
import scipy
import torch
import tensorflow as tf
import h5py
import mat73
import scipy.io as scio
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .utils import get_all_files, crop, FFT2c, IFFT2c, Emat_xyt_complex, ifft2c, ifftyc_5, normalize_complex, ifftc_1d, ifft2c_2d, fft2c_2d
from utils.utils import *

class CardiacDataSet(Dataset):
    def __init__(self, dataset_name, mode):
        super(CardiacDataSet, self).__init__()
        self.dataset_name = dataset_name
        if self.dataset_name == 'DYNAMIC_V2':
            # store the raw tensors
            self._k = np.load(os.path.join(
                '/data1/wenqihuang/LplusSNet/data', mode + '_k_newdata_v2.npy')).astype(np.complex64)
            self._label = np.load(os.path.join(
                '/data1/wenqihuang/LplusSNet/data', mode + '_label_newdata_v2.npy')).astype(np.complex64)
        elif self.dataset_name == 'DYNAMIC_V2_MULTICOIL':
            sys.exit("CardiacDataSet: Need to implement DYNAMIC_V2_MULTICOIL")
        else:
            sys.exit("CardiacDataSet: No dataset load")

    def __getitem__(self, index):
        if self.dataset_name == 'DYNAMIC_V2':
            k = self._k[index, :]
            label = self._label[index, :]
            return k, label
        elif self.dataset_name == 'DYNAMIC_V2_MULTICOIL':
            sys.exit("CardiacDataSet: Need to implement DYNAMIC_V2_MULTICOIL")

    def __len__(self):
        return self._label.shape[0]


class FastMRIKneeDataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIKneeDataSet, self).__init__()
        self.config = config
        if mode == 'training':
            self.kspace_dir = '/data/data42/LiuCongcong/score_about/multiscale_score_based-master_cao/fastMRI_knee_sample/T1_data/'
            self.maps_dir = '/data/data42/LiuCongcong/score_about/multiscale_score_based-master_cao/fastMRI_knee_sample/output_maps/'
        elif mode == 'test':
            self.kspace_dir = '/data0/chentao/data/fastMRI_knee_test/T1_data/'
            self.maps_dir = '/data0/chentao/data/fastMRI_knee_test/output_maps/'
        elif mode == 'sample':
            self.kspace_dir = '/data/data42/LiuCongcong/score_about/multiscale_score_based-master_cao/fastMRI_knee_sample/T1_data/'
            self.maps_dir = '/data/data42/LiuCongcong/score_about/multiscale_score_based-master_cao/fastMRI_knee_sample/output_maps/'
        elif mode == 'datashift':
            self.kspace_dir = '/data0/chentao/data/fastMRI_brain/brain_T2/'
            self.maps_dir = '/data0/chentao/data/fastMRI_brain/output_maps/'
        else:
            raise NotImplementedError

        self.mode = mode
        self.file_list = get_all_files(self.kspace_dir) 
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(
                self.kspace_dir, os.path.basename(file)))
            with h5py.File(os.path.join(self.kspace_dir, file), 'r') as data:
                if self.mode != 'sample':
                    # self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 6)
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])
                else:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 被试者编号
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0]) # 返回测试者编号，也就是第几组数据
        # 被试者扫描的帧数编号
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] +
                self.num_slices[scan_idx] - 1) # 寻找在当前组数据的第几张索引

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir,
                                 os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            # 去掉前6帧
            if self.mode != 'sample':
                # slice_idx = slice_idx + 6
                slice_idx = slice_idx 
            maps_idx = data['s_maps'][slice_idx]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = crop(maps_idx, cropx=320, cropy=320)
            maps_idx = np.squeeze(maps_idx, 0)
            maps = np.asarray(maps_idx)

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.kspace_dir,
                                os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            ksp_idx = data['kspace'][slice_idx] # 15x640x368
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = crop(IFFT2c(ksp_idx), cropx=320, cropy=320)
            ksp_idx = FFT2c(ksp_idx)
            ksp_idx = np.squeeze(ksp_idx, 0)
            if self.config.data.normalize_type == 'minmax':
                img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
                img_idx = self.config.data.normalize_coeff * normalize_complex(img_idx)
                ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)
            elif self.config.data.normalize_type == 'std':
                minv = np.std(ksp_idx)
                ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)

            kspace = np.asarray(ksp_idx)

        return kspace, maps

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))


class VascularWallDataSet(Dataset):
    def __init__(self, config, mode):
        super(VascularWallDataSet, self).__init__()
        self.config = config
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class T1rhoDataSet_h5(Dataset):
    def __init__(self, config, mode):
        super(T1rhoDataSet_h5, self).__init__()
        self.config = config
        if mode == 'train':
            self.kspace_file = '/data/data42/LiuCongcong/T1rho/t1rho_raw_cor/Img_in_3dim/cor_raw_full.h5' # 所有.mat
            self.maps_file = '/data/data42/LiuCongcong/T1rho/data/t1rho_map_train_h5_single/t1rho_maps.h5'
        elif mode == 'sample':
            # self.kspace_file = '/data/data42/LiuCongcong/T1rho/t1rho_raw_cor_test/14_cor.h5' # 240x200x30x32x4
            # self.maps_file = '/data/data42/LiuCongcong/T1rho/t1rho_map_cor_test/14_cor_maps.h5' # 4x32x30x200x240

            # 胶质瘤第一次尝试
            # self.kspace_file = '/data/data42/LiuCongcong/T1rho/data/Brain_glioma/raw_new.h5' # 384x192x16x32x4 
            # self.maps_file =  '/data/data42/LiuCongcong/T1rho/data/Brain_glioma/csm_new_sep4chan.h5' #  384x192x16x32x4 
            
            #胶质瘤第二次尝试
            self.kspace_file = '/data/data42/LiuCongcong/T1rho/data/Brain_glioma/glioma2/raw_org2.h5' # 192x192x16x4x32 
            self.maps_file =  '/data/data42/LiuCongcong/T1rho/data/Brain_glioma/glioma2/csm_org2.h5' #  192x192x16x4x32 

        else:
            raise NotImplementedError

        self.mode = mode

        print('Input file:', self.kspace_file)
        if self.mode != 'sample':
            # self.temp_raw = mat73.loadmat(os.path.join(self.kspace_dir, os.path.basename(file))) # 240x200x30x32x4
            with h5py.File(self.kspace_file, 'r') as data:
                self.kspace = np.array(data['kspace']) # 240x200x390x32x4
                self.kspace = np.transpose(self.kspace, [2,3,4,0,1]) # 390x32x4x240x200
        elif self.mode == 'sample':
            with h5py.File(self.kspace_file, 'r') as data:
                # self.kspace = np.array(data['kspace']) # 4x32x30x200x240 
                # self.kspace = np.transpose(self.kspace, [2,1,0,4,3]) # 30x32x4x240x200

                # # 这里是原始测试
                # self.kspace = np.array(data['kspace']) # 240x229x28x32x4  这里补充数
                # self.kspace = np.transpose(self.kspace, [2,1,0,4,3]) # 28x32x4x240x229        
                # # 原始测试结束

                # 病例测试
                self.kspace = np.array(data['raw']) # 192*192*16*4*32 
                self.kspace = np.transpose(self.kspace, [2,4,3,0,1]) # 16x32x4x192x192        
                self.kspace = self.kspace[8:11,:,:,:,:]
                # 病例测试结束    
        else:
            raise NotImplementedError
            # with h5py.File(os.path.join(self.kspace_dir, file), 'r') as data:                
            #     print('Input file:', os.path.join(self.kspace_dir, os.path.basename(file)))
            #     temp_raw = data['kspace'] # 240x200x30x32x4
            #     temp_raw = np.transpose(temp_raw, [2,3,4,0,1]) # 30x32x4x240x200
            #     self.kspace = temp_raw


        print('Input file:', self.maps_file)
        if self.mode != 'sample':
            # self.temp_raw = mat73.loadmat(os.path.join(self.kspace_dir, os.path.basename(file))) # 240x200x30x32x4
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['maps']) # 390x32x4x240x200  
        elif self.mode == 'sample':
            with h5py.File(self.maps_file, 'r') as data:
                # self.maps = np.array(data['maps']) # 240x200x30x32x4     
                # self.maps = np.transpose(self.maps, [2,3,4,0,1])   # 30x32x4x240x200   

                self.maps = np.array(data['csm']) # 192*192*16*4*32
                self.maps = np.transpose(self.maps, [2,4,3,0,1]) # 16x32x4x192*192 
                self.maps = self.maps[8:11,:,:,:,:]
        else:  # if sample
            raise NotImplementedError
            # with h5py.File(os.path.join(self.maps_dir, file), 'r') as data:
            #     temp_raw = data['maps'] # 240x200x30x32x4
            #     temp_raw = np.transpose(temp_raw, [2,3,4,0,1]) # 30x32x4x240x200
            #     self.maps = temp_raw  # 30x32x4x240x200


    def __getitem__(self, index):
        kspace = self.kspace[index] # kspace:32x4x240x200
        kspace = torch.from_numpy(kspace)
        kspace = ifft2c_2d(kspace)
        # kspace = kspace.numpy()
        kspace = crop(kspace, 192, 192) # 32x4x192x192
        # kspace = torch.from_numpy(kspace)
        kspace = fft2c_2d(kspace) # 32x4x192x192
        kspace = kspace.numpy()   
        # kspace = kspace[:,1:2,:,:] # 32x1x192x192 用于单张训练
        
        minv = np.std(kspace)
        self.minv = minv
        kspace = kspace / (1.5 * minv)

        # kspace = kspace[:, 3:4, :, :]

        kspace = torch.from_numpy(kspace) 
        # label = kspace
        # save_mat('.', label, 'label', index, False)
        # kspace = np.expand_dims(kspace, axis=0) # 1 x 32 x 4 x 240 x 200
        maps = self.maps[index] # 32x4x240x192
        # maps = maps[:,1:2,:,:] # slicex32x1x240x200 用于单张训练
        maps = torch.from_numpy(maps)
        maps = crop(maps, 192,192)
        # maps = maps[:, 3:4, :,:]
        return kspace, maps 
    
    def __len__(self):
        slice_num = self.kspace.shape[0] 
        return slice_num

class T1rhoDataSet_h5_5T(Dataset):
    def __init__(self, config, mode):
        super(T1rhoDataSet_h5_5T, self).__init__()
        self.config = config
        if mode == 'train':
            self.kspace_file = '/data/data42/LiuCongcong/T1rho/t1rho_raw_cor/Img_in_3dim/cor_raw_full.h5' # 所有.mat
            self.maps_file = '/data/data42/LiuCongcong/T1rho/data/t1rho_map_train_h5_single/t1rho_maps.h5'
        elif mode == 'sample':
            self.kspace_file = '/data/data42/LiuCongcong/T1rho/data/5T_T1rho/Lili/kData_cor_use.h5'
            self.maps_file = '/data/data42/LiuCongcong/T1rho/data/5T_T1rho/Lili/csm_cor.h5'
        else:
            raise NotImplementedError

        self.mode = mode

        print('Input file:', self.kspace_file)
        if self.mode != 'sample':
            # self.temp_raw = mat73.loadmat(os.path.join(self.kspace_dir, os.path.basename(file))) # 240x200x30x32x4
            with h5py.File(self.kspace_file, 'r') as data:
                self.kspace = np.array(data['kspace']) # 240x200x390x32x4
                self.kspace = np.transpose(self.kspace, [2,3,4,0,1]) # 390x32x4x240x200
        elif self.mode == 'sample':
            with h5py.File(self.kspace_file, 'r') as data:
                self.kspace = np.array(data['h_kspce']) # 240x200x30x48x4
                self.kspace = np.transpose(self.kspace, [2,3,4,0,1]) # 30x48x4x240x200                
        else:
            raise NotImplementedError
            # with h5py.File(os.path.join(self.kspace_dir, file), 'r') as data:                
            #     print('Input file:', os.path.join(self.kspace_dir, os.path.basename(file)))
            #     temp_raw = data['kspace'] # 240x200x30x32x4
            #     temp_raw = np.transpose(temp_raw, [2,3,4,0,1]) # 30x32x4x240x200
            #     self.kspace = temp_raw


        print('Input file:', self.maps_file)
        if self.mode != 'sample':
            # self.temp_raw = mat73.loadmat(os.path.join(self.kspace_dir, os.path.basename(file))) # 240x200x30x32x4
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['maps']) # 390x32x4x240x200  
        elif self.mode == 'sample':
            with h5py.File(self.maps_file, 'r') as data:
                self.maps = np.array(data['csm_cor']) # 240x200x30x48x4   
                self.maps = np.transpose(self.maps, [2,3,4,0,1])   #30x48x4x240x200                 
        else:  # if sample
            raise NotImplementedError
            # with h5py.File(os.path.join(self.maps_dir, file), 'r') as data:
            #     temp_raw = data['maps'] # 240x200x30x32x4
            #     temp_raw = np.transpose(temp_raw, [2,3,4,0,1]) # 30x32x4x240x200
            #     self.maps = temp_raw  # 30x32x4x240x200


    def __getitem__(self, index):
        kspace = self.kspace[index] # kspace:48x4x240x200
        kspace = torch.from_numpy(kspace)
        kspace = ifft2c_2d(kspace)
        # kspace = kspace.numpy()
        kspace = crop(kspace, 192, 192) # 48x4x192x192
        # kspace = torch.from_numpy(kspace)
        kspace = fft2c_2d(kspace) # 32x4x192x192
        kspace = kspace.numpy()   
        # kspace = kspace[:,1:2,:,:] # 32x1x192x192 用于单张训练
        
        minv = np.std(kspace)
        self.minv = minv
        kspace = kspace / (1.5 * minv)

        # kspace = kspace[:,2:3, :, :]

        kspace = torch.from_numpy(kspace) 
        # label = kspace
        # save_mat('.', label, 'label', index, False)
        # kspace = np.expand_dims(kspace, axis=0) # 1 x 32 x 4 x 240 x 200
        maps = self.maps[index] # 32x4x240x192
        # maps = maps[:,1:2,:,:] # slicex32x1x240x200 用于单张训练
        maps = torch.from_numpy(maps)
        maps = crop(maps, 192,192)
        # maps = maps[:, 2:3, :,:]
        return kspace, maps 
    
    def __len__(self):
        slice_num = self.kspace.shape[0] 
        return slice_num

def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == 'fastMRI_knee':
        dataset = FastMRIKneeDataSet(config, mode)
    elif config.data.dataset_name == 'Cardiac':
        dataset = CardiacDataSet(config.data.dataset_name, mode)
    elif config.data.dataset_name == 'VWI':
        dataset = VascularWallDataSet(config.data.dataset_name, mode)
    elif config.data.dataset_name == 't1rho':
        dataset = T1rhoDataSet_h5(config.data.dataset_name, mode)
    elif config.data.dataset_name == 't1rho_5T':
        dataset = T1rhoDataSet_h5_5T(config.data.dataset_name, mode)   

    if mode == 'train':
        data = DataLoader(
            dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    # test: 90多张图，sample：一张图，第十张
    else:  
        data = DataLoader(
            dataset, batch_size=config.sampling.batch_size, shuffle=False, pin_memory=True)

    print(mode, "data loaded")

    return data
