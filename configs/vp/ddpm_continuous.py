# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training DDPM with VP SDE."""

from configs.default_fastMRI_configs import get_default_configs



'''
  training TODO:
  training.sde, vp or ms
  training.mask_type 
  training.acs
  training.mean_equal
  training.acc
  sde的std M_hat
  beta_max, beta_min
  num_scales加速采样后续可能要改
  -------------------
  sampling TODO:
  training.sde, vp or ms
  training.mask_type 
  training.acs
  training.mean_equal
  training.acc
  sde的std M_hat
  beta_max, beta_min
  ---
  sampling.predictor
  sampling.corrector
  sampling.folder
  sampling.ckpt
  sampling.fft
  sampling.mask_type
  sampling.acc
  sampling.acs
  # 加速采样
  x初值, 改noise std
  sde.N
  sde.T
'''
def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'mssde'
  training.continuous = True
  training.reduce_mean = True
  training.mask_type = 'low_frequency' # low_frequency, uniform, center or cartesian 
  training.acc = 'None'
  training.acs = '22'
  training.mean_equal = 'noequal' # equal or noequal

  # sampling
  sampling = config.sampling
  sampling.batch_size = 1
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama' # reverse_diffusion or euler_maruyama
  sampling.corrector = 'langevin' # langevin or none
  # sampling.folder = '2022_07_28T19_44_04_ddpm_vpsde_N_1000'
  # sampling.folder = '2022_07_28T20_05_42_ddpm_mssde_low_frequency_None_18_noequal_N_1000'
  # sampling.folder = '2022_08_02T10_14_28_ddpm_mssde_low_frequency_None_24_noequal_N_1000'
  # sampling.folder = '2022_08_02T10_13_05_ddpm_mssde_low_frequency_None_22_noequal_N_1000'
  # sampling.folder = '2022_08_02T10_11_10_ddpm_mssde_low_frequency_None_20_noequal_N_1000'
  # sampling.folder = '2022_08_08T11_02_12_ddpm_mssde_low_frequency_None_18_noequal_N_100'
  # sampling.folder = '2022_08_08T11_29_11_ddpm_mssde_low_frequency_None_18_noequal_N_200'
  # sampling.folder = '2022_08_08T11_29_50_ddpm_vpsde_N_200'
  sampling.folder = '2022_09_08T11_56_28_ddpm_mssde_low_frequency_None_22_noequal_N_1000'
  # sampling.folder = '2022_09_08T11_58_56_ddpm_mssde_low_frequency_None_18_noequal_N_1000'
  sampling.ckpt = 500
  sampling.mask_type = 'uniform' # uniform, random_uniform or center
  sampling.acc = '10'
  sampling.acs = '18'
  sampling.fft = 'fft' # fft or nofft
  sampling.snr = 0.20 ##### 0.16, 加速采样用0.26会比较好: snr调小可以抑制噪声，容易产生伪影；调大可以抑制伪影，噪声过大
  sampling.mse = 2.5 ##### predictor_mse
  sampling.corrector_mse = 5. ###
  sampling.datashift = 'knee' ### head or knee

  # data
  data = config.data
  data.centered = False # True: Input is in [-1, 1]
  data.dataset_name = 'fastMRI_knee'
  data.image_size = 320
  data.normalize_type = 'std' # minmax or std
  data.normalize_coeff = 1.5 # normalize coefficient

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config
