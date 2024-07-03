import torch

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from inverse.operators import bcmm, InpaintOperator
import sde_lib

def get_sampler(config, obsv_sde, shape, lambda_schedule=lambda t: 1.0-t, eps=1e-3):
    if config.inverse.sampler == 'conditional':
        sampler = get_conditional_sampler(obsv_sde, shape, lambda_schedule, eps=eps, device=config.device)
    else:
        raise NotImplementedError

    return sampler

def get_conditional_sampler(obsv_sde:sde_lib.OBSVSDE, shape, lambda_schedule,
                            rtol=1e-5, atol=1e-5,
                            method='RK45', eps=1e-3, device='cuda'):
    """"""

    A, L, T = obsv_sde.operator.decompose(shape)
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
            I = torch.eye(shape[2]*shape[3]).to(x.device)
            x = weight[:,None,None] * bcmm(L.transpose(2,3) @ A, yt) + (1.-weight)[:,None,None] * bcmm(A, x) + bcmm((I-A), x)

        return x



    def conditional_sampler(model, z=None):
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

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (obsv_sde.state_sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            return x, nfe

    return conditional_sampler
