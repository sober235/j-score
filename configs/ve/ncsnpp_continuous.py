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
"""Training NCSN++ on Church with VE SDE."""
# import default_fastMRI_configs
from configs.default_fastMRI_configs import get_default_configs # 先将get_default_configs内部的变量初始化


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.reduce_mean = True
  training.mask_type = 'low_frequency' # low_frequency, uniform, center or cartesian 
  training.acc = 'None'
  training.acs = '24'
  training.mean_equal = 'noequal' # equal or noequal
  
  # sampling
  sampling = config.sampling
  sampling.extra_chan = 1
  sampling.batch_size = 1
  sampling.shape = 4
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'  
  sampling.corrector = 'langevin'
  sampling.folder = '2022_11_04T23_23_58_ncsnpp_vesde_N_1000'
  # sampling.ckpt = 380
  sampling.ckpt = 1000
  sampling.mask_type = 'uniform_acc2_glioma2'
  sampling.acc = '2'
  sampling.acs = '0'
  sampling.fft = 'nofft' # fft or nofft
  # sampling.snr = 0.458  ##### 0.16, 加速采样用0.26会比较好: snr调小可以抑制噪声，容易产生伪影；调大可以抑制伪影，噪声过大，T1rho这里不能超过1，范围在0.1-0.5
  sampling.snr = 0.6  ##### 0.16, 加速采样用0.26会比较好: snr调小可以抑制噪声，容易产生伪影；调大可以抑制伪影，噪声过大，T1rho这里不能超过1，范围在0.1-0.5
  # sampling.mse = 2.5    ### predictor_mse
  sampling.mse = 2.5
  # sampling.corrector_mse = 5 ##
  sampling.corrector_mse = 5 ##
  sampling.datashift = 'head' ##sh# head or knee
  sampling.gen = False    # 这里控制是加速还是生成

  # data
  data = config.data
  data.centered = False
  data.dataset_name = 't1rho'
  data.image_size = 192
  # data.image_size_x=240
  # data.image_size_y=200
  data.normalize_type = 'std' # minmax or std
  data.normalize_coeff = 1.5 # normalize coefficient

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.dropout = 0.
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 8 # 这里
  model.attn_resolutions = (0,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config