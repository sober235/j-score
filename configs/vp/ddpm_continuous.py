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

def get_config():
  config = get_default_configs()

  training = config.training
  training.sde = 'mssde'
  training.continuous = True
  training.reduce_mean = True
  training.mask_type = 'low_frequency'
  training.acc = 'None'
  training.acs = '22'
  training.mean_equal = 'noequal'

  sampling = config.sampling
  sampling.batch_size = 1
  sampling.method = 'pc'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'langevin'
  sampling.folder = '2022_09_08T11_56_28_ddpm_mssde_low_frequency_None_22_noequal_N_1000'
  sampling.ckpt = 500
  sampling.mask_type = 'uniform'
  sampling.acc = '10'
  sampling.acs = '18'
  sampling.fft = 'fft'
  sampling.snr = 0.20
  sampling.mse = 2.5
  sampling.corrector_mse = 5.
  sampling.datashift = 'knee'

  data = config.data
  data.centered = False
  data.dataset_name = 'fastMRI_knee'
  data.image_size = 320
  data.normalize_type = 'std'
  data.normalize_coeff = 1.5

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
