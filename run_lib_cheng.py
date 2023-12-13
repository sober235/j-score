"""Training and evaluation for score-based generative models. """

from dataclasses import dataclass
import gc
import io
import os
from os.path import join
import time
import scipy.io as sio
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
from torch import from_numpy, nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils.utils import *
import utils.datasets as datasets
import mat73
import time
import torchvision.transforms as T


FLAGS = flags.FLAGS

# cuda.set_per_process_memory_fraction(0.5)

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
    train_dl = datasets.get_dataset(config, 'train')  # 加载训练数据
    # num_data = len(train_dl.dataset)

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

    for epoch in range(config.training.epochs):  # 1-->1000个时刻
        loss_sum = 0
        for step, batch in enumerate(train_dl): # 在第一个时刻遍历所有的图像，每张图像加噪声求loss
            #****** Joint distribution train**********#
            # t0 = time.time()

            # k0, csm = batch # k0: 1x32x4x192x192, csm: 1x32x4x192x192
            # # TODO: mask condition
            # label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x1x192x192 合并单通道图像
            # # save_mat('.', label, 'label', 0, False)

            # label = torch.squeeze(label, 0) # 1x4x192x192
            # label = scaler(label) # 1x4x192x192
            # label = c2r(label).type(torch.FloatTensor).to(config.device) # 1x8x192x192
            #******End joint distribution train**********#
            
            #********Single image train**********#
            t0 = time.time()
            k0, csm = batch # k0: 1x32x1x192x192, csm: 1x32x1x192x192
            # TODO: mask condition
            label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x1x192x192 合并单通道图像
            save_mat('.', label, 'label', 0, False)

            label = torch.squeeze(label, 0) # 1x4x192x192
            label = scaler(label) # 1x2x192x192
            label = c2r(label).type(torch.FloatTensor).to(config.device) # 1x8x192x192     
            #******End single image train**********#
            # save_mat('.', label, 'label', 0, False)
            # Execute one training step
            # loss = train_step_fn(state, label) # 原来的
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
        if (epoch + 1) % 5 == 0:
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

    if FLAGS.config.sampling.datashift == 'z':
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


    # atb_mask = get_mask(config, 'sample')
    # atb_mask = atb_mask.data.cpu().numpy()
    # atb_mask = crop(atb_mask, 192, 192)
    # atb_mask = torch.from_numpy(atb_mask).to(config.device) # 1x1x192x192

    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/random_acc8_acs16.mat')
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/uniform_192x192x4_acc6_acs16.mat')
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/uniform_192x192x4_acc8_acs17.mat')
    # Lcc vista mask
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/mask_full_acs3_acs3_acs3.mat')
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/vista_mask_8x_acs10.mat')
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/random_mask_1x32x4x192x192_acs10_af6.mat')
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/cartesian_random_6x_1x32x4x192x192_acs10.mat')
    atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/vista_1x32x4x192x192_cui_acs10_af6.mat')
    atb_mask = atb_mask['mask'] # 1x32x4x192x192
    atb_mask = atb_mask[1,:,:,:]

    # atb_mask = np.transpose(atb_mask, [])
    # atb_mask = np.transpose(atb_mask, [2,1,0]) # 4x192x192
    atb_mask = np.tile(atb_mask, [1,1,1,1]) # 1x4x192x192
    # end Lcc vista mask

    # 这里程添加
    zpad = T.CenterCrop((config.data.image_size,config.data.image_size)) # 这里控制着输入数据的维度
    atb_mask = zpad(torch.from_numpy(atb_mask).to(config.device) )
    print('atb_mask shape:', atb_mask.shape)
    # end 程添加

    # atb_mask = atb_mask.data.cpu().numpy()

    # Lcc random mask
    # atb_mask = mat73.loadmat('/data/data42/LiuCongcong/T1rho/code/DiffusionModel_jd_t1rho/mask/random_mask_1x32x4x192x192_acs10_af6.mat')
    # atb_mask = atb_mask['mask'] # 1x32x4x192x192
    # atb_mask = atb_mask[1,:,:,:]
    # # atb_mask = np.transpose(atb_mask, [])
    # # atb_mask = np.transpose(atb_mask, [2,1,0]) # 4x192x192
    # atb_mask = np.tile(atb_mask, [1,1,1,1]) # 1x4x192x192
    # End Lcc random mask

    # atb_mask = atb_mask[:, 3:4, :, :]
    # atb_mask = torch.from_numpy(atb_mask).to(config.device) 
    # save_mat('.', atb_mask, 'atb_mask', 0 , False)


    # train_mask = atb_mask

    # train_mask = get_mask(config, 'sde')
    # train_mask = train_mask.data.cpu().numpy()
    # train_mask = crop(train_mask, 192, 192)
    # train_mask = torch.from_numpy(train_mask).to(config.device) # 1x1x192x192


    # Build the sampling function when sampling is enabled
    sampling_shape = (config.sampling.batch_size, config.data.num_channels,
                                config.data.image_size, config.data.image_size) # 1x8x192x192
    # sampling_shape = (config.sampling.batch_size, config.data.num_channels,
    #                             256, 256) # 1x8x192x192                                 
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, 
                                inverse_scaler, sampling_eps, atb_mask, atb_mask)
    recon_fin = []
    for index, point in enumerate(test_dl): # 每次取一张图
        print('---------------------------------------------')
        print('---------------- point:', index, '------------------')
        print('---------------------------------------------')
        k0, csm = point  # k0: 1x32x4x192x192   csm: 1x32x4x192x192
        k0 = k0.to(config.device)   # 1x32x4x192x192
        csm = csm.to(config.device) # 1x32x4x192x192
        
        img=ifft2c(k0)
        img = zpad(img)
        k0 = fft2c(img)

        csm = zpad(csm)

        label = Emat_xyt_complex(k0, True, csm, 1.).to(config.device) # 1x1x4x192x192
        # save_mat('.', label, 'label', index, False) 

        # Lcc Multi-coil
        # label = c2r(label).type(torch.FloatTensor).to(config.device) # 1x2x4x192x192
        # label = torch.squeeze(label)   # 2x4x192x192 jd测试
        # label = label.permute(1,0,2,3) # 4x2x192x192
        # End Lcc multi-coil

        # Lcc single-coil
        label = c2r(label).type(torch.FloatTensor).to(config.device) # 1x2x4x192x192
        label = torch.squeeze(label,0) # 2x1x192x192
        label = label.permute(1,0,2,3) # 4x2x192x192
        # End Lcc single-coil
        
        # Control Acc or Gen 用于单张加速或者单张纯生成
        if config.sampling.gen == True:   # 如果纯生成k0为高斯噪声
            k0 = torch.randn_like(k0)
            k0 = k0.to(config.device) 
        # save_mat('.', k0, 'k0', 0, True)

        # atb_mask = crop(atb_mask, 192, 192) # 1x4x192x192 
        
        # atb_mask = torch.rot90(atb_mask, -1, dims=[2,3]) # 1x4x192x192
        atb = k0 * atb_mask # k0: 1x32x4x192x192    atb_mask: 1x1x192x192    atb:1x32x4x192x192
        
        # save_mat('.', atb, 'atb', index, True)

        # Lcc multi-coil
        # atb = torch.squeeze(atb) # 32x4x192x192
        # csm = torch.squeeze(csm) # 32x4x192x192
        # End Lcc multi-coil

        # Lcc single-coil
        atb = torch.squeeze(atb,0) # 32x1x192x192
        csm = torch.squeeze(csm,0) # 32x1x192x192
        # End Lcc single-coil
        
        atb_to_image = c2r(Emat_xyt_complex_T1rho(atb, True, csm, 1)).type(torch.FloatTensor).to(config.device)  # 1x8x192x192
        csm = c2r(csm).type(torch.FloatTensor).to(config.device) # 32x8x192x192
        
        torch.cuda.synchronize()
        begin = time.time()
        recon, n = sampling_fn(score_model, atb, atb_to_image, csm) # 

        torch.cuda.synchronize()
        end_time = time.time()
        print(end_time)
        recon = r2c(recon) # 4x192x192
        recon = recon.unsqueeze(3) # 4x192x192x1
        recon_fin.append(recon)
    recon_fin = torch.cat(recon_fin, dim=3)
    save_mat(FLAGS.config.sampling.folder, recon_fin, 'recon_fin', index, normalize=False)
    hfssde_save_mat(FLAGS.config, recon_fin)
