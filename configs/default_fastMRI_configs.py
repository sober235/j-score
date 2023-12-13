import os
import ml_collections
import torch
import torch.cuda as cuda



def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 1
  training.epochs = 380
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = False
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  # sampling.n_steps_each = 1 # 原始的
  sampling.n_steps_each = 1 # for glioma
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.4 ##### 0.16, 加速采样用0.26会比较好
  sampling.mse = 2.5 ##### predictor_mse
  sampling.corrector_mse = 5. ###

  # data
  config.data = data = ml_collections.ConfigDict()
  data.image_size = 192
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  # data.num_channels = 2 # 单张训练
  data.num_channels = 8   # 联合测试通道

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000 ### 噪声尺度
  model.beta_min = 0.1
  model.beta_max = 20. ###
  model.dropout = 0.1 ###
  model.matrix = True ###
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config