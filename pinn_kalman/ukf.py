import numpy as np
import torch
from torch import nn
from torchfilter.filters import UnscentedKalmanFilter
from pinn_kalman.ukf_utils import NSDynamics, InpaintKFMeasure
from pinn import B_PINN
import types

class UKF(nn.Module):

    def __init__(self, config):
        super(UKF, self).__init__()

        self.dynamic = NSDynamics(config)
        self.measurement = InpaintKFMeasure(config)
        self.ukf = UnscentedKalmanFilter(dynamics_model=self.dynamic, measurement_model=self.measurement)

        self.ukf.initialize_beliefs() #TODO

    def forward(self, obsv):
        return self.ukf(obsv, None)

if __name__ == '__main__':
    from configs.pinn.pinn_pde import get_config
    config = get_config()

    ukf = UKF(config)