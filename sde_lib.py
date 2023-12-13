"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
from ast import Not
import torch
import numpy as np
from utils.utils import *


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, config):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.config = config
        self.N = config.model.num_scales

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code
        Returns:
          log probability density
        """
        pass

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        config = self.config
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.config = config
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, atb, csm, atb_mask): # 对应宋飏SDE公式 （46上方公式）
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                grad = score_fn(x, t)
                meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(grad)
                meas_grad *= self.config.sampling.mse
                drift = drift - diffusion[:, None, None, None] ** 2 * \
                    (grad - meas_grad) * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t, atb, csm, atb_mask): # 对应宋飏SDE公式 （46）
                """Create discretized iteration rules for the reverse diffusion sampler."""
                grad = score_fn(x, t)
                meas_grad = Emat_xyt_T1rho(x, False, csm, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt_T1rho(meas_grad, True, csm, atb_mask)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(grad)
                meas_grad *= self.config.sampling.mse
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * \
                    (grad - meas_grad) * (0.5 if self.probability_flow else 1.)
                
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VESDE(SDE):
    def __init__(self, config):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(config)
        self.sigma_min = config.model.sigma_min # 
        self.sigma_max = config.model.sigma_max
        self.N = config.model.num_scales
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N)) # log() -> log(348)
        self.config = config

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq.30
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x) 
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t): # Eq.31
        # 注入噪声是geometric sequence, 如下所示（Ye， Eq.19）, 表示
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape): # xT----->逆向初值
        return torch.randn(*shape) * self.sigma_max  

    def prior_logp(self, z): # 
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t): # Eq.45 在class ReverseDiffusionPredictor(Predictor):调用
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), 
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x) # 对应VE的f, 扩散系数
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class VPSDE(SDE):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config)
        self.beta_0 = config.model.beta_min
        self.beta_1 = config.model.beta_max
        self.N = config.model.num_scales
        self.config = config
        self.discrete_betas = torch.linspace(self.beta_0 / self.N, self.beta_1 / self.N, self.N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property   # 修饰方法，表示方法只读
    def T(self):
        return 1

    def sde(self, x, t):
        # Eq.32
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - \
            torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        # 等价无穷小：(1 + x)^alpha - 1 ~ alpha * x ===> -1/2 * beta * x ~ [(1 - beta) ^ 1/2 - 1]x
        # TODO: 为啥这里不直接算，反而要等价无穷小?
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
  def __init__(self, config):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(config)
    self.beta_0 = config.model.beta_min
    self.beta_1 = config.model.beta_max
    self.N = config.model.num_scales

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class MSSDE(SDE):
    def __init__(self, config):
        """Construct a MultiScale SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config)
        self.config = config
        self.beta_0 = config.model.beta_min # 0.1
        self.beta_1 = config.model.beta_max # 20
        self.N = config.model.num_scales    # 200
        self.discrete_betas = torch.linspace(
            self.beta_0 / self.N, self.beta_1 / self.N, self.N)

        self.mask = get_mask(config, 'sde')
        self.identity_mat_comp = torch.ones_like(self.mask, dtype=self.mask.dtype).to(config.device)

        # TODO
        self.alphas = 1. - self.discrete_betas
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        # beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        # drift = -0.5 * beta_t[:, None, None, None] * x
        # diffusion = torch.sqrt(beta_t)
        # return drift, diffusion
        raise NotImplementedError

    def marginal_prob(self, x, t): # 计算扰动核的均值和方差
        max_N = 1000000
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        M_hat_x = c2r(ifft2c_2d((self.identity_mat_comp - self.mask) * fft2c_2d(r2c(x)))
                    ).type(torch.FloatTensor).to(self.config.device)
        
        # TODO: log_mean_coeff->0的时候等价无穷小已经不准确了?
        if self.config.training.mean_equal == 'equal':
            mean = ((1 + log_mean_coeff / max_N) ** max_N - 1) * M_hat_x + x
        # std_coeff = torch.sqrt(1 - (1 + 2 * log_mean_coeff / max_N) ** max_N)
        # print('=====================ceil:', std_coeff)
        else:
            mean = (torch.exp(log_mean_coeff[:, None, None, None]) - 1) * M_hat_x + x
        std_coeff = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        # print('====================floor:', std_coeff)

        return mean, std_coeff

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        # shape = z.shape
        # N = np.prod(shape[1:])
        # logps = -N / 2. * np.log(2 * np.pi) - \
        #     torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        # return logps
        raise NotImplementedError

    def discretize(self, x, t, z):
        # # """MS score-based discretization."""
        # timestep = (t * (self.N - 1) / self.T).long()
        # beta = self.discrete_betas.to(x.device)[timestep]
        # alpha = self.alphas.to(x.device)[timestep]
        # sqrt_beta = torch.sqrt(beta)
        # # TODO:check x is float, rather than complex
        # M_wavy_x = c2r(ifft2c_2d((self.mask) * fft2c_2d(r2c(x)))
        #                  ).type(torch.FloatTensor).to(self.config.device)
        # M_wavy_z = c2r(ifft2c_2d((self.mask) * fft2c_2d(r2c(z)))
        #                  ).type(torch.FloatTensor).to(self.config.device)
        # f_x = M_wavy_x + torch.sqrt(alpha)[:, None, None, None] * M_wavy_x - x
        # G_z = sqrt_beta * M_wavy_z
        
        # return f_x, G_z
        raise NotImplementedError
    

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        mask = self.mask
        discrete_betas = self.discrete_betas
        alphas = self.alphas
        config = self.config
        # sde_fn = self.sde
        # discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
                self.mask = mask

            @property
            def T(self):
                return T

            def sde(self, x, t, atb, csm, atb_mask):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                beta_t = config.model.beta_min + t * (config.model.beta_max - config.model.beta_min)
                M_hat_x = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(x)))
                                ).type(torch.FloatTensor).to(x.device)
                drift = -0.5 * beta_t[:, None, None, None] * M_hat_x
                
                diffusion = torch.sqrt(beta_t)
                
                grad = score_fn(x, t)
                meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                meas_grad /= torch.norm(meas_grad)
                meas_grad *= torch.norm(grad)
                meas_grad *= config.sampling.mse
                # print('mse:', config.sampling.mse)
                if config.model.matrix:
                    G_s = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(grad - meas_grad)))
                                ).type(torch.FloatTensor).to(x.device)
                else:
                    G_s = (grad - meas_grad).type(torch.FloatTensor).to(x.device)

                # score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * \
                    G_s * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion

                dt = -1. / self.N
                z = torch.randn_like(x)

                if config.model.matrix:
                    M_hat_z = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(z)))
                                ).type(torch.FloatTensor).to(x.device)
                else:
                    M_hat_z = z.type(torch.FloatTensor).to(x.device)
                x_mean = x + drift * dt
                x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * M_hat_z
                return x, x_mean

            def discretize(self, x, t, z, atb, csm, atb_mask):
                # timestep = (t * (self.N - 1) / T).long()
                # beta = discrete_betas.to(x.device)[timestep]
                # alpha = alphas.to(x.device)[timestep]
                # sqrt_beta = torch.sqrt(beta)
                # # TODO:check x is float, rather than complex
                # M_hat_x = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(x)))
                #                 ).type(torch.FloatTensor).to(x.device)
                # M_hat_z = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(z)))
                #                 ).type(torch.FloatTensor).to(x.device)
                # f_x = torch.sqrt(alpha)[:, None, None, None] * M_hat_x - M_hat_x
                # G_z = sqrt_beta * M_hat_z


                # grad = score_fn(x, t)
                # meas_grad = Emat_xyt(x, False, csm, atb_mask) - c2r(atb)
                # meas_grad = Emat_xyt(meas_grad, True, csm, atb_mask)
                # meas_grad /= torch.norm(meas_grad)
                # meas_grad *= torch.norm(grad)
                # meas_grad *= config.sampling.mse

                # # TODO: meas_gard前有系数
                # G_s = c2r(ifft2c_2d((1 - self.mask) * fft2c_2d(r2c(grad - meas_grad)))
                #                 ).type(torch.FloatTensor).to(x.device)
                # rev_f = f_x - beta * G_s * (0.5 if self.probability_flow else 1.)

                # # self.probability_flow
                # rev_G = G_z

                # x_mean = x - rev_f
                # x = x_mean + rev_G

                # return x, x_mean
                raise NotImplementedError

        return RSDE()