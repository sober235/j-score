"""Training and evaluation for score-based generative models. """

import logging
import os
import time

import numpy as np
import scipy.io as sio
import tensorflow as tf
import torch
from absl import flags
from torch.utils import tensorboard

import likelihood
import losses
import sampling
import sde_lib
import utils.datasets as datasets
from models import ddpm, ncsnpp
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
from utils.utils import *

FLAGS = flags.FLAGS


def _load_sampling_mask():
    mask_path = os.environ.get('T1RHO_MASK_PATH')
    if not mask_path:
        raise RuntimeError('Missing required environment variable T1RHO_MASK_PATH for sampling.')

    atb_mask = sio.loadmat(mask_path)['mask']
    atb_mask = np.squeeze(atb_mask[:, 0:1, :, :, :])
    atb_mask = np.squeeze(atb_mask)
    atb_mask = np.tile(atb_mask, (1, 1, 1, 1))
    print('Sampling mask:', mask_path)
    print('****atb_mask shape*****', atb_mask.shape)
    return torch.from_numpy(atb_mask)


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    sample_dir = os.path.join(workdir, "samples_in_train")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    tf.io.gfile.makedirs(checkpoint_dir)

    start_epoch = 0
    if os.path.isdir(checkpoint_dir):
        existing = [f for f in os.listdir(checkpoint_dir)
                    if f.startswith('checkpoint_') and f.endswith('.pth')]
        if existing:
            latest_epoch = max(int(f.replace('checkpoint_', '').replace('.pth', ''))
                               for f in existing)
            latest_ckpt = os.path.join(checkpoint_dir, f'checkpoint_{latest_epoch}.pth')
            state = restore_checkpoint(latest_ckpt, state, device=config.device)
            start_epoch = latest_epoch
            logging.info(f"Resuming training from {latest_ckpt} (epoch {start_epoch})")

    initial_step = int(state['step'])

    train_dl = datasets.get_dataset(config, 'train')
    num_data = len(train_dl.dataset)

    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

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
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(config, sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)

    if config.training.snapshot_sampling:
        pass

    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(start_epoch, config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            k0, csm = batch
            label = Emat_xyt_complex(k0, True, csm, 1)

            label = torch.squeeze(label, 0)
            label = scaler(label)
            label = c2r(label).type(torch.FloatTensor).to(config.device)
            loss = train_step_fn(state, label)
            loss_sum += loss

            param_num = sum(param.numel()
                            for param in state["model"].parameters())
            if step % 10 == 0:
                print('Epoch', epoch + 1, '/', config.training.epochs, 'Step', step,
                        'loss = ', loss.cpu().data.numpy(),
                        'loss mean =', loss_sum.cpu().data.numpy() / (step + 1),
                        'time', time.time() - t0, 'param_num', param_num)

            if step % config.training.log_freq == 0:
                logging.info("step: %d, training_loss: %.5e" %
                             (step, loss.item()))
                global_step = num_data * epoch + step
                writer.add_scalar(
                    "training_loss", scalar_value=loss, global_step=global_step)

            if step % config.training.eval_freq == 0:
                pass

        if (epoch + 1) % 5 == 0:
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}.pth'), state)

        if config.training.snapshot_sampling and (epoch + 1) % config.training.snapshot_freq == 0:
            pass


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
                    score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.sampling.ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    if FLAGS.config.sampling.datashift == ' z':
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
        test_dl = datasets.get_dataset(config, 'datashift')
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
        test_dl = datasets.get_dataset(config, 'sample')
    FLAGS.config.sampling.folder = os.path.join(FLAGS.workdir, 'acc' + SAMPLING_FOLDER_ID)
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)

    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

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
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    atb_mask = _load_sampling_mask().to(config.device)
    train_mask = atb_mask

    sampling_shape = (
        config.sampling.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, 
                                inverse_scaler, sampling_eps, atb_mask, train_mask)
    recon_fin = []
    for index, point in enumerate(test_dl):
        print('---------------------------------------------')
        print('---------------- point:', index, '------------------')
        print('---------------------------------------------')
        k0, csm = point
        k0 = k0.to(config.device)
        csm = csm.to(config.device)

        label = Emat_xyt_complex(k0, True, csm, 1.).to(config.device)

        label = c2r(label).type(torch.FloatTensor).to(config.device)
        label = torch.squeeze(label,0)
        label = label.permute(1,0,2,3)

        atb = k0 * atb_mask

        atb = torch.squeeze(atb,0)
        csm = torch.squeeze(csm,0)

        atb_to_image = c2r(Emat_xyt_complex_T1rho(atb, True, csm, 1)).type(torch.FloatTensor).to(config.device)
        csm = c2r(csm).type(torch.FloatTensor).to(config.device)

        torch.cuda.synchronize()
        begin = time.time()
        recon, n = sampling_fn(score_model, atb, atb_to_image, csm)
        end = time.time()
        elapsed_time = end - begin
        print(f"elapsed time: {elapsed_time} seconds")
        torch.cuda.synchronize()
        end_time = time.time()
        print(end_time)
        recon = r2c(recon)
        recon = recon.unsqueeze(0)

        recon_fin.append(recon)

    recon_fin = torch.cat(recon_fin,dim=0)
    recon_fin = recon_fin.cpu().data.numpy()
    file = os.path.join(FLAGS.config.sampling.folder, str('recon_fin') + '.mat')
    sio.savemat(file, {'recon_fin': recon_fin})
