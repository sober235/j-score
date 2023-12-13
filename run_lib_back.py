"""Training and evaluation for score-based generative models. """

from dataclasses import dataclass
import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
# import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ncsnpp, ddpm # 
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
# import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch import nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils.utils import *
import utils.datasets as datasets

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)  # return score name
    ema = ExponentialMovingAverage( score_model.parameters(), decay=config.model.ema_rate )
    optimizer = losses.get_optimizer(config, score_model.parameters()) # 指定优化器
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints") # results/checkpoints
    tf.io.gfile.makedirs(checkpoint_dir)
    # Resume training when intermediate checkpoints are detected
    # state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, 'training')  # 加载训练数据
    num_data = len(train_dl.dataset)

    # Create data scaler and its inverse
    scaler = get_data_scaler(config) # return a funtion
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde': # 
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config) # 返回vesde内部包含的所有对象
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'mssde':
        sde = sde_lib.MSSDE(config)
        sampling_eps = 1e-3  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous   # bool型
    reduce_mean = config.training.reduce_mean # bool型
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(config, sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    # eval_step_fn = losses.get_step_fn(config, sde, train=False, optimize_fn=optimize_fn,
    #                                   reduce_mean=reduce_mean, continuous=continuous,
    #                                   likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        # sampling_shape = (config.training.batch_size, config.data.num_channels,
        #                   config.data.image_size, config.data.image_size)
        # sampling_fn = sampling.get_sampling_fn(
        #     config, sde, sampling_shape, inverse_scaler, sampling_eps)
        pass

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl): # 每一个时刻抽取一张图
            t0 = time.time()
            k0, csm = batch # k0: 1x32x4x240x200, 
            # TODO: mask condition
            label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x4x240x200 合并单通道图像
            label = c2r(label).type(torch.FloatTensor).to(config.device)
            label = scaler(label) # 1x1x4x240x200
            label = torch.permute(label, [2, 0, 1, 3, 4]) # 4x1x1x240x200
            label = torch.squeeze(label, 1) # 4x1x240x200

            # Execute one training step
            loss = train_step_fn(state, label) # 网络的输出和
            loss_sum += loss

            param_num = sum(param.numel()
                            for param in state["model"].parameters())
            if step % 10 == 0:
                print('Epoch', epoch + 1, '/', config.training.epochs, 'Step', step,
                        'loss = ', loss.cpu().data.numpy(),
                        'loss mean =', loss_sum.cpu().data.numpy() / (step + 1),
                        'time', time.time() - t0, 'param_num', param_num)

            if step % config.training.log_freq == 0:
                # logging.info("step: %d, training_loss: %.5e" %
                #              (step, loss.item()))
                # global_step = num_data * epoch + step
                # writer.add_scalar(
                #     "training_loss", scalar_value=loss, global_step=global_step)
                pass

            # Report the loss on an evaluation dataset periodically
            if step % config.training.eval_freq == 0:
                pass

        # Save a checkpoint for every 5 epochs
        # if (epoch + 1) % 5 == 0:
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}.pth'), state)

        # Generate and save samples for every epoch
        if config.training.snapshot_sampling and (epoch + 1) % config.training.snapshot_freq == 0:
            # config.sampling.ckpt = epoch + 1
            # sample_dir = ""
            pass


def sample(config, workdir): 
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.sampling.ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    if FLAGS.config.sampling.datashift == 'head':
        SAMPLING_FOLDER_ID = '_'.join([FLAGS.config.sampling.acc, FLAGS.config.sampling.acs, 
                        FLAGS.config.sampling.mask_type, 'ckpt', str(config.sampling.ckpt),
                        FLAGS.config.sampling.predictor, 
                        FLAGS.config.training.mean_equal,
                        FLAGS.config.sampling.datashift,
                        FLAGS.config.sampling.fft,
                        str(config.sampling.snr),
                        'predictor_mse', str(FLAGS.config.sampling.mse),
                        'corrector_mse', str(FLAGS.config.sampling.corrector_mse),
                        str(FLAGS.config.data.centered)])
        test_dl = datasets.get_dataset(config, 'datashift') # mode=test:90多张图，modex=sample:一张图，第十张
    else:
        SAMPLING_FOLDER_ID = '_'.join([FLAGS.config.sampling.acc, FLAGS.config.sampling.acs, 
                            FLAGS.config.sampling.mask_type, 'ckpt', str(config.sampling.ckpt),
                            FLAGS.config.sampling.predictor, 
                            FLAGS.config.training.mean_equal,
                            FLAGS.config.sampling.fft,
                            str(config.sampling.snr),
                            'predictor_mse', str(FLAGS.config.sampling.mse),
                            'corrector_mse', str(FLAGS.config.sampling.corrector_mse),
                            str(FLAGS.config.data.centered)])
        test_dl = datasets.get_dataset(config, 'sample') # mode=test:90多张图，modex=sample:一张图，第十张
        # test_dl = datasets.get_dataset(config, 'test')
    FLAGS.config.sampling.folder = os.path.join(FLAGS.workdir, 'acc' + SAMPLING_FOLDER_ID)
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)

    # Build data pipeline
    # test_dl = datasets.get_dataset(config, 'sample') # mode=test:90多张图，modex=sample:一张图，第十张

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'mssde':
        sde = sde_lib.MSSDE(config)
        sampling_eps = 1e-3  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")


    atb_mask = get_mask(config, 'sample')
    train_mask = get_mask(config, 'sde')
    # Build the sampling function when sampling is enabled
    sampling_shape = (config.sampling.batch_size, config.data.num_channels,
                                config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, 
                                inverse_scaler, sampling_eps, atb_mask, train_mask)

    for index, point in enumerate(test_dl):
        print('---------------------------------------------')
        print('---------------- point:', index, '------------------')
        print('---------------------------------------------')
        k0, csm = point
        # csm = c2r(csm).type(torch.FloatTensor).to(self.config.device)
        k0 = k0.to(config.device)
        csm = csm.to(config.device)

        # save_mat('.', k0, 'kspace', 0, False)
        # save_mat('.', csm, 'csm', 0, False)

        label = Emat_xyt_complex(k0, True, csm, 1.).to(config.device) # 单张label
        # if FLAGS.config.sampling.datashift == 'head':
        #     save_mat('results/1head', label, 'label', index, normalize=True)
        # else:
        #     save_mat('results/1knee_label', label, 'label', index, normalize=True)

        atb = k0 * atb_mask
        atb_to_image = c2r(Emat_xyt_complex(atb, True, csm, 1)).type(torch.FloatTensor).to(config.device) # 1x2x320x320
        csm = c2r(csm).type(torch.FloatTensor).to(config.device)

        recon, n = sampling_fn(score_model, atb, atb_to_image, csm) #

        recon = r2c(recon)

        save_mat(FLAGS.config.sampling.folder, recon, 'recon', index, normalize=True)
        # hfssde_save_mat(FLAGS.config, recon)