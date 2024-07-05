import torch

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from inverse.operators import bcmm, InpaintOperator
import sde_lib
from functools import partial
from utils import Clock

def get_solver(config, ode_func, x0, t1, shape, eps):

    if config.inverse.solver in ['RK45', 'RK23']:
        solution = integrate.solve_ivp(ode_func, (t1, eps), x0,
                                       rtol=1e-3, atol=1e-3,
                                       method=config.inverse.solver, )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(config.device).type(torch.float32)
        print(nfe)

        return x

    elif config.inverse.solver == 'fixed':
        x = x0
        dt = -.00002 # inverse ODE
        for t in torch.linspace(t1, eps, 5000):
            x += ode_func(t, x) * dt
        return torch.tensor(x).reshape(shape).to(config.device).type(torch.float32)





def get_sampler(config, obsv_sde, shape, lambda_schedule=lambda t: (1.0-t)*0.8, eps=1e-3):
    if config.inverse.sampler == 'controlled':
        sampler = get_controlled_sampler(config, obsv_sde, shape, lambda_schedule, eps=eps)
    elif config.inverse.sampler == 'dps':
        sampler = get_dps_sampler(config, obsv_sde, shape, eps=eps)
    else:
        raise NotImplementedError

    return sampler

def get_controlled_sampler(config, obsv_sde:sde_lib.OBSVSDE, shape, lambda_schedule, eps=1e-3):
    """"""

    A, L, T = obsv_sde.operator.decompose(shape)
    M1 = L.transpose(2,3) @ A
    M2 = torch.eye(A.shape[2], device=A.device)-A
    device = config.device
    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(obsv_sde.state_sde, model, train=False, continuous=True)
        rsde = obsv_sde.state_sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def optimize_fn(x, t):
        """Optimize the distance between self and pre-observed state."""
        z = torch.randn_like(x)
        x = x.flatten(2,3)
        yt = obsv_sde.observe_sampling(z, t)
        weight = lambda_schedule(t)

        if isinstance(obsv_sde.operator, InpaintOperator):
            x = weight[:,None,None] * bcmm(M1, yt) + (1.-weight)[:,None,None] * bcmm(A, x) + bcmm(M2, x)

        return x



    def controlled_sampler(model, z=None):
        """The conditional sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = obsv_sde.state_sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x_hat = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=device) * t
                x_hat = optimize_fn(x_hat, vec_t).reshape(shape)
                drift = drift_fn(model, x_hat, vec_t)
                return to_flattened_numpy(drift)

            solver = partial(get_solver, config=config, shape=shape, eps=eps)
            return solver(ode_func=ode_func, x0=to_flattened_numpy(x), t1=obsv_sde.state_sde.T)

    return controlled_sampler


def get_dps_sampler(config, obsv_sde:sde_lib.OBSVSDE, shape, eps=1e-3):
    '''diffusion posterior sampling'''

    device = config.device
    obsv_var = config.inverse.variance
    observation = obsv_sde.y0 + torch.randn_like(obsv_sde.y0, device=device) * obsv_var**.5
    clock = Clock(10)

    def drift_fn(score, score_cond, x, t):
        """Get the drift function of the reverse-time DPS."""

        drift, diffusion = obsv_sde.state_sde.sde(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * (score + score_cond) * 0.5

        return drift

    def x0_hat_fn(model, xt, t):
        """Estimation of x0."""
        score_fn = get_score_fn(obsv_sde.state_sde, model, train=False, continuous=True)
        score = score_fn(xt, t)

        mean, std = obsv_sde.state_sde.marginal_coef(t)

        x0_hat = xt/mean[:,None, None, None] + std[:,None, None, None]**2 * score
        return x0_hat, score

    def cond_grad_fn(xt, x0_hat, scale=True):
        """Gradient of conditional distribution of y on x0_hat."""
        difference = observation - obsv_sde.operator(x0_hat, keep_shape=False)
        norm = torch.linalg.norm(difference)
        logp = - norm**2 / obsv_var
        norm_grad = torch.autograd.grad(outputs=logp, inputs=xt)[0]

        if scale is True:
            norm_grad /= norm

        return norm_grad

    def dps_sampler(model, z=None):
        """The conditional sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """

        # Initial sample
        if z is None:
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = obsv_sde.state_sde.prior_sampling(shape).to(device)
        else:
            x = z

        def ode_func(t, x):
            x_hat = from_flattened_numpy(x, shape).to(device).type(torch.float32).requires_grad_()
            vec_t = torch.ones(shape[0], device=device) * t

            x0_hat, score = x0_hat_fn(model, x_hat, vec_t)
            score_cond = cond_grad_fn(x_hat, x0_hat)
            drift = drift_fn(score, score_cond, x_hat, vec_t)

            clock.tic(f"t = {round(t,5)}")

            return to_flattened_numpy(drift)

        solver = partial(get_solver, config=config, shape=shape, eps=eps)
        return solver(ode_func=ode_func, x0=to_flattened_numpy(x), t1=obsv_sde.state_sde.T)

    return dps_sampler