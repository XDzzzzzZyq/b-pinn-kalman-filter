"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
from inverse.operators import *


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def coefficient(self, t):
        """"Linear SDE Only"""
        pass

    @abc.abstractmethod
    def marginal_coef(self, t):
        """"marginal mu and sigma at t"""
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
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class OBSVSDE(SDE):

    state_sde: SDE
    operator: LinearOperators

    def __init__(self, N, y0, operator):
        super().__init__(N)
        self.y0 = y0
        self.operator = operator
    @abstractmethod
    def observe_sampling(self, z, t):
        pass


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        drift_coef, diffusion_coef = self.coefficient(t)
        drift = drift_coef[:, None, None, None] * x
        diffusion = diffusion_coef
        return drift, diffusion

    def coefficient(self, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift_coef = -0.5 * beta_t
        diffusion_coef = torch.sqrt(beta_t)
        return drift_coef, diffusion_coef

    def marginal_coef(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

        return mean, std

    def marginal_prob(self, x, t):
        mean, std = self.marginal_coef(t)
        return mean[:, None, None, None] * x, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        drift_coef, diffusion_coef = self.coefficient(t)
        drift = drift_coef[:, None, None, None] * x
        diffusion = diffusion_coef
        return drift, diffusion

    def coefficient(self, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift_coef = -0.5 * beta_t
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion_coef = torch.sqrt(beta_t * discount)
        return drift_coef, diffusion_coef

    def marginal_coef(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = 1 - torch.exp(2. * log_mean_coeff)

        return mean, std

    def marginal_prob(self, x, t):
        mean, std = self.marginal_coef(t)
        return mean[:,None,None,None]*x, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        """Construct a Variance Exploding SDE.

        Args:
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          N: number of discretization steps
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        drift_coef, diffusion_coef = self.coefficient(t)
        drift = drift_coef[:, None, None, None]
        diffusion = diffusion_coef
        return drift, diffusion

    def coefficient(self, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift_coef = torch.zeros_like(t)
        diffusion_coef = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                         device=t.device))
        return drift_coef, diffusion_coef

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class LOBSVSDE(OBSVSDE):
    def __init__(self, state_sde: SDE, y0, operator: LinearOperators):
        """Construct a Linear Observation SDE.

        Args:
          state_sde: the SDE that hidden state follows.
          y0:        the observation y_0. ill-posed
          operator:  the linearobervation operator A
        """
        super().__init__(state_sde.N, y0, operator)

        self.state_sde = state_sde
        self.mat = None

    def get_matrix(self, shape):
        if self.mat is None:
            self.mat = self.operator.to_matrix(shape)
        return self.mat

    def marginal_prob(self, z, t):
        alpha, beta = self.state_sde.marginal_coef(t)
        mat = self.get_matrix(z.shape)
        corr = mat & mat
        mean = alpha * self.y0
        std = beta ** 2 * corr

        return mean, std

    def observe_sampling(self, z, t): # shape: (B, C, D)
        alpha, beta = self.state_sde.marginal_coef(t)
        return alpha[:,None,None] * self.y0 + beta[:,None,None] * self.operator(z, False)

    def prior_sampling(self, shape):
        return None

    @property
    def T(self):
        return 1

    def prior_logp(self, z):
        pass

    def sde(self, x, t):
        pass

    def coefficient(self, t):
        pass

    def marginal_coef(self, t):
        pass