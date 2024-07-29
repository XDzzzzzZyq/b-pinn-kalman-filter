"""
PINN+incompressible NS equation
2-dimensional unsteady
PINN model +LOSS function
PINN融合不可压缩NS方程
二维非定常流动
PINN模型 + LOSS函数
"""
import os
import numpy as np
import torch
import torch.nn as nn
from models.ddpm import UNet, MLP
from models.flownet import FlowNet, PressureNet, project
from models.liteflownet import LiteFlowNet
import torch.nn.functional as F

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

def get_model(config):
    if config.model.arch == 'flownet':
        return FlowNet(config)
    elif config.model.arch == 'liteflownet':
        return LiteFlowNet(config)
    elif config.model.arch == 'unet':
        return UNet(config)
    elif config.model.arch == 'mlp':
        return MLP(config)
    else:
        raise NotImplementedError

# Define network structure, specified by a list of layers indicating the number of layers and neurons
# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN(nn.Module):

    """
    Input:  field, t  := (x, y, f) : shape=(B, 3, N, N), (B, )
    Output: field_out := (u, v, p) : shape=(B, 3, N, N)
    """
    def __init__(self, config):
        super(PINN, self).__init__()
        self.device = config.device
        self.dt = config.data.dt

        self.flownet = get_model(config).to(self.device)
        self.pressurenet = PressureNet(config).to(self.device)

        self.mask_u, self.mask_v = self.get_mask(config)

    def get_mask(self, config):
        '''for differentiable slicing'''

        N = config.data.image_size
        device = config.device

        zero = torch.zeros(N, N)
        ones = torch.ones(N, N)
        mask1 = torch.stack([ones, zero]).to(device)
        mask2 = torch.stack([zero, ones]).to(device)

        return mask1, mask2

    def forward(self, f1, f2, x, y, t, size=None):
        flow = self.flownet(f1, f2, x, y, t, size=size)
        pressure = self.pressurenet(flow, x, y, t)
        return flow, pressure

    def advection_mse(self, x, y, t, prediction):
        return None

    # derive loss for equation
    def equation_mse(self, x, y, t, flow, pres, Re):

        # 获得预测的输出u,v,p

        u = (self.mask_u * flow).sum(dim=1).unsqueeze(1)
        v = (self.mask_v * flow).sum(dim=1).unsqueeze(1)
        p = pres

        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导

        u_x, u_y, u_t = torch.autograd.grad(u.sum(), (x, y, t), create_graph=True, retain_graph=True)
        v_x, v_y, v_t = torch.autograd.grad(v.sum(), (x, y, t), create_graph=True, retain_graph=True)
        p_x, p_y      = torch.autograd.grad(p.sum(), (x, y),    create_graph=True, retain_graph=True)

        # second-order derivative
        u_xx = torch.autograd.grad(u_x.sum(), x, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, retain_graph=True)[0]

        # reshape
        u_t = u_t[:,None,None,None]
        v_t = v_t[:,None,None,None]

        # residual
        # 计算偏微分方程的残差
        #print(u_t.shape, u.shape, u_x.shape, p_x.shape)
        f_equation_x    = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y    = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        f_equation_mass = u_x + v_y

        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x)
        mse_x    = mse(f_equation_x,    batch_t_zeros)
        mse_y    = mse(f_equation_y,    batch_t_zeros)
        mse_mass = mse(f_equation_mass, batch_t_zeros)

        return mse_x + mse_y + mse_mass

    def step(self, ft, u):
        return project(ft, u, self.dt)

class B_PINN(PINN):
    def __init__(self, config, pretrained_pinn:PINN=None):
        self.using_pretrained = pretrained_pinn is not None

        super(B_PINN, self).__init__(config)
        flow_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 0.1,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",
            "moped_enable": self.using_pretrained,
            "moped_delta": config.model.bpinn_moped_delta, }

        pres_bnn_prior_parameters = {
            "prior_mu": 0.0,
            "prior_sigma": 0.01,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -0.5,
            "type": "Reparameterization",
            "moped_enable": self.using_pretrained,
            "moped_delta": config.model.bpinn_moped_delta,
        }

        if self.using_pretrained:
            self.flownet = pretrained_pinn.flownet
            self.pressurenet = pretrained_pinn.pressurenet

        dnn_to_bnn(self.flownet,     flow_bnn_prior_parameters)
        dnn_to_bnn(self.pressurenet, pres_bnn_prior_parameters)

        self.flownet = self.flownet.to(config.device)
        self.pressurenet = self.pressurenet.to(config.device)
        self.batch = config.training.batch_size

    def sample_uvp(self, f1, f2, x, y, t, n=64):
        flow_pred = []
        pres_pred = []
        for mc_run in range(n):
            flow, pressure = self.forward(f1, f2, x, y, t)
            flow_pred.append(flow[-1])
            pres_pred.append(pressure)

        return flow_pred, pres_pred

    def predict(self, f1, f2, x, y, t, n=64):
        """
        Complete MC Sampling with n samples
        """

        flow_pred, pres_pred = self.sample_uvp(f1, f2, x, y, t, n)

        f_pred = []
        for flow in flow_pred:
            f = self.step(f2, flow)
            f_pred.append(f)

        flow_pred = torch.stack(flow_pred, dim=0)
        pres_pred = torch.stack(pres_pred, dim=0)
        f_pred    = torch.stack(f_pred, dim=0)

        return (flow_pred.mean(dim=0),
                pres_pred.mean(dim=0),
                f_pred.mean(dim=0),
                flow_pred.std(dim=0),
                pres_pred.std(dim=0),
                f_pred.std(dim=0))

    def uncertainty(self, flow, pres):
        return 0

if __name__ == '__main__':
    print(0)