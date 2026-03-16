import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
import tensorflow as tf
import logging



def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_mat(save_dict, variable, file_name, index=0, normalize=True):
    if normalize:
        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) +
                        '_' + str(index + 1) + '.mat')
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def crop(img, cropx, cropy):
    nb, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty:starty + cropy, startx: startx + cropx]


def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img


def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def get_mask(config, caller):
    if caller == 'sde':
        if config.training.mask_type == 'low_frequency':
            mask_file = 'mask/' +  config.training.mask_type + "_acs" + config.training.acs + '.mat'
        elif config.training.mask_type == 'center':
            mask_file = 'mask/' +  config.training.mask_type + '_acc2.mat'
        else:
            mask_file = 'mask/' +  config.training.mask_type + "_acc" + config.training.acc \
                                                + '_acs' + config.training.acs + '.mat'
    elif caller == 'sample':
        mask_file = 'mask/' +  config.sampling.mask_type + "_acc" + config.sampling.acc \
                                                + '_acs' + config.sampling.acs + '.mat'
    elif caller == 'acs':
        mask_file = 'mask/low_frequency_acs18.mat'
    mask = scio.loadmat(mask_file)['mask']
    mask = mask.astype(np.complex128)
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask).to(config.device)

    return mask


def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def fft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))
    return x


def fft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)/np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)/np.math.sqrt(ny)
    return x


def ifft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.ifft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=4), torch.sqrt(ny))
    return x


def ifft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)*np.math.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)*np.math.sqrt(ny)
    return x

def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            if b.ndim == 4:
                b = ifft2c_2d(b)
            else:
                b = ifft2c(b)
            x = c2r(b)
        else:
            b = r2c(b)
            if b.ndim == 4:
                b = fft2c_2d(b) * mask
            else:
                b = fft2c(b) * mask
            x = c2r(b)
    else:
        if inv:
            csm = r2c(csm)
            x = r2c(b) * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)
            x = c2r(x)

        else:
            csm = r2c(csm)
            b = r2c(b)
            b = b*csm   
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b
            x = c2r(x)

    return x


def Emat_xyt_T1rho(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            if b.ndim == 4:
                b = ifft2c_2d(b)
            else:
                b = ifft2c(b)
            x = c2r(b)
        else:
            b = r2c(b)
            if b.ndim == 4:
                b = fft2c_2d(b) * mask
            else:
                b = fft2c(b) * mask
            x = c2r(b)
    else:
        if inv:
            csm = r2c(csm)
            x = r2c(b) * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 0)
            x = torch.unsqueeze(x, 0)
            x = c2r(x)

        else:
            csm = r2c(csm)
            b = r2c(b)
            b = b*csm   
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b
            x = c2r(x)

    return x


def Emat_xyt_complex(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)

        else:
            b = b*csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b

    return x


def Emat_xyt_complex_T1rho(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 0)
            x = torch.unsqueeze(x, 0)

        else:
            b = b*csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b

    return x



def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x


def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x


def Abs(x):
    x = r2c(x)
    return torch.abs(x)


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
