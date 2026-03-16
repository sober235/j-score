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

"""Training NCSN++ with a VE SDE for T1rho experiments."""

from configs.default_fastMRI_configs import get_default_configs


def get_config():
  config = get_default_configs()

  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.reduce_mean = True
  training.mask_type = 'low_frequency'
  training.acc = 'None'
  training.acs = '10'
  training.mean_equal = 'noequal'

  sampling = config.sampling
  sampling.extra_chan = 1
  sampling.batch_size = 1
  sampling.shape = 4
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.folder = '2022_11_04T23_23_58_ncsnpp_vesde_N_1000'
  sampling.ckpt = 380
  sampling.mask_type = 'released_mask'
  sampling.acc = '10'
  sampling.acs = '10'
  sampling.fft = 'nofft'
  sampling.snr = 0.458
  sampling.mse = 2
  sampling.corrector_mse = 5
  sampling.datashift = 'head'
  sampling.gen = False

  data = config.data
  data.centered = False
  data.dataset_name = 't1rho'
  data.image_size = 192
  data.normalize_type = 'minmax'
  data.normalize_coeff = 1.5

  model = config.model
  model.name = 'ncsnpp'
  model.dropout = 0.0
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 8
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
  model.init_scale = 0.0
  model.fourier_scale = 16
  model.conv_size = 3

  return config
