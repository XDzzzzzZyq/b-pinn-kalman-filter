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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from models import utils as mutils
from sde_lib import VESDE, VPSDE

from bayesian_torch.models.dnn_to_bnn import get_kl_loss


def get_optimizer(config, params, lr_mul=1.0, is_bpinn=False):
    """Returns a flax optimizer object based on `config`."""

    if is_bpinn:
        lr = config.optim.bpinn_lr
        decay = config.optim.bpinn_weight_decay
    else:
        lr = config.optim.lr
        decay = config.optim.weight_decay

    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=lr*lr_mul, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=decay)
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config, is_bpinn=False):
    """Returns an optimize_fn based on `config`."""

    if is_bpinn:
        lr = config.optim.bpinn_lr
    else:
        lr = config.optim.lr

    def optimize_fn(optimizer, params, step, lr=lr, warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
      eps: A `float` number. The smallest time step to sample from.

    Returns:
      A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None, None, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas ** 2)[:, None, None, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + sqrt_1m_alphas_cumprod[
            labels, None, None, None] * noise
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
        https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, continuous=True,
                                  likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn

def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter: {name}')
            return True
    return False

def get_prelim_step_fn(config, train, optimize_fn, is_bpinn=False):

    error_fn = torch.nn.MSELoss()

    def kl_loss_fn(model):
        kl_loss = get_kl_loss(model)
        if kl_loss is not None:
            return kl_loss / config.training.batch_size
        else:
            return 0.0

    def flow_loss_fn(model, operator, batch):

        f1, f2, x, y, t, target = batch
        f1 = operator(f1, keep_shape=True) + torch.randn_like(f1) * config.inverse.variance ** 0.5
        f2 = operator(f2, keep_shape=True) + torch.randn_like(f2) * config.inverse.variance ** 0.5

        veloc_pred = model(f1, f2, x, y, t)
        v_loss = model.multiscale_data_mse(veloc_pred, target, error_fn=error_fn)

        if is_bpinn:
            return v_loss + kl_loss_fn(model) * 0.1
        else:
            return v_loss

    def pres_loss_fn(model, batch):

        f1, f2, x, y, t, target = batch

        cascaded_flow = [target[:,0:2]]
        for i in range(len(config.model.feature_nums)):
            flow = cascaded_flow[-1]
            size = (flow.shape[2]//2, flow.shape[3]//2)
            flow = F.interpolate(input=flow, size=size, mode='bilinear', align_corners=False)
            cascaded_flow.append(flow)

        pres_pred = model(cascaded_flow[::-1], x, y, t)
        p_loss = model.data_mse(pres_pred, target, error_fn=error_fn)

        if is_bpinn:
            return p_loss + kl_loss_fn(model) * 0.01
        else:
            return p_loss

    def step_fn(state, operator, batch):
        model = state['model']
        flownet = model.flownet
        pressurenet = model.pressurenet
        operator.next()

        if train:
            optimizer_flow, optimizer_pres = state['optimizer']
            '''
            
            Flow Net Rraining
            
            '''
            flownet.train()

            optimizer_flow.zero_grad()
            v_loss = flow_loss_fn(flownet, operator, batch)

            v_loss.backward()
            optimize_fn(optimizer_flow, flownet.parameters(), step=state['step'])

            '''
            
            Pressure Net Training
            
            '''

            pressurenet.train()

            optimizer_pres.zero_grad()
            p_loss = pres_loss_fn(pressurenet, batch)

            p_loss.backward()
            optimize_fn(optimizer_pres, pressurenet.parameters(), step=state['step'])

            state['step'] += 1
            state['ema'].update(model.parameters())

        else:
            model.eval()

            ema = state['ema']
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            v_loss = flow_loss_fn(flownet, operator, batch)
            p_loss = pres_loss_fn(pressurenet, batch)
            ema.restore(model.parameters())

        loss = v_loss + p_loss

        return loss, v_loss, p_loss

    return step_fn


def get_pinn_step_fn(config, train, optimize_fn):

    def loss_fn(model, operator, batch):

        f1, f2, x, y, t, target = batch
        f1 = operator(f1, keep_shape=True) + torch.randn_like(f1) * config.inverse.variance ** 0.5
        f2 = operator(f2, keep_shape=True) + torch.randn_like(f2) * config.inverse.variance ** 0.5
        flow_pred, pres_pred = model(f1, f2, x, y, t)

        v_loss = model.flownet.multiscale_data_mse(flow_pred, target)
        p_loss = model.pressurenet.data_mse(pres_pred, target)
        data_loss = v_loss + p_loss

        pinn_loss = model.equation_mse(x, y, t, flow_pred[-1], pres_pred, 10000000.0) * config.training.pinn_loss_weight

        return pinn_loss + data_loss, pinn_loss, data_loss

    def step_fn(state, operator, batch):
        model = state['model']
        operator.next()

        if train:
            optimizer_flow, optimizer_pres = state['optimizer']

            model.train()
            optimizer_flow.zero_grad()
            optimizer_pres.zero_grad()
            loss, pinn_loss, data_loss = loss_fn(model, operator, batch)

            layer = model.pressurenet.end[-1]
            w = layer.weight
            grad = torch.autograd.grad(loss, w, retain_graph=True)[0]
            if torch.isnan(grad).any():
                print(">>> Nan Grad Detected <<<")
                return loss, pinn_loss, data_loss

            loss.backward()
            optimize_fn(optimizer_flow, model.flownet.parameters(), step=state['step'])
            optimize_fn(optimizer_pres, model.pressurenet.parameters(), step=state['step'])

            state['step'] += 1
            state['ema'].update(model.parameters())

        else:
            model.eval()

            ema = state['ema']
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            loss, pinn_loss, data_loss = loss_fn(model, operator, batch)
            ema.restore(model.parameters())

        return loss, pinn_loss, data_loss

    return step_fn

