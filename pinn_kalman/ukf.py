import numpy as np
import torch
from torch import nn
from torchfilter.filters import UnscentedKalmanFilter, SquareRootUnscentedKalmanFilter, ParticleFilter
from torchfilter.base import ParticleFilterMeasurementModelWrapper
from torchfilter.utils import MerweSigmaPointStrategy
from pinn_kalman.ukf_utils import NSDynamics, InpaintKFMeasure, IdentityKFMeasure, patch, unpatch
from pinn import B_PINN

class UKF(nn.Module):

    def __init__(self, config):
        super(UKF, self).__init__()

        self.dim = config.kf.patch_size
        self.size = config.data.image_size
        self.device = config.device

        self.dynamic = NSDynamics(config)
        self.measurement = IdentityKFMeasure(config)
        self.strategy = MerweSigmaPointStrategy(alpha=1.0, beta=0.0, kappa=0.0)

        if config.kf.type == 'pf':
            self.ukf = ParticleFilter(dynamics_model=self.dynamic,
                                      measurement_model=ParticleFilterMeasurementModelWrapper(self.measurement),
                                      num_particles=100)
        elif config.kf.type == 'ukf':
            self.ukf = UnscentedKalmanFilter(dynamics_model=self.dynamic,
                                             measurement_model=self.measurement,
                                             sigma_point_strategy=self.strategy)
        elif config.kf.type == 'sqrtukf':
            self.ukf = SquareRootUnscentedKalmanFilter(dynamics_model=self.dynamic,
                                                       measurement_model=self.measurement,
                                                       sigma_point_strategy=self.strategy)

    def initialize(self, x0=None, var=0.01):
        # TODO: better initialization
        N = (self.size // self.dim) ** 2 * 4

        if x0 is None:
            mean = torch.ones(N, self.dim ** 2).to(self.device) * 0.1
            covariance = torch.eye(self.dim ** 2).unsqueeze(0).repeat(N, 1, 1).to(self.device) * 0.01
        else:
            mean = x0
            covariance = torch.eye(self.dim ** 2).unsqueeze(0).repeat(N, 1, 1).to(self.device) * var

        self.ukf.initialize_beliefs(mean=mean, covariance=covariance)

    def forward(self, obsv):
        print("obsv", torch.isnan(obsv).any())
        pred = self.ukf(observations=obsv, controls=torch.zeros(obsv.shape[0], device=obsv.device))

        print("pred", torch.isnan(pred).any())
        #print("cov", torch.isnan(self.ukf.belief_covariance).any())
        return pred

class PINN_KF(nn.Module):
    def __init__(self, config):
        super(PINN_KF, self).__init__()
        self.device = config.device
        self.dim = config.kf.patch_size
        self.size = config.data.image_size
        self.sample_num = 2*self.dim**2

        self.ukf = UKF(config).to(self.device)
        self.pinn = B_PINN(config).to(self.device)

    def initialize(self, f, v, p):

        workdir = "../workdir/pde-bpinn/checkpoints-meta/checkpoint.pth"
        self.pinn = utils.load_checkpoint(workdir, self.pinn, self.device)

        for name, param in self.pinn.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in parameter: {name}")
                print(param)

        inital_state = torch.cat([f, v, p], dim=1)
        inital_state = patch(inital_state, config.kf.patch_size)

        self.ukf.initialize(inital_state, 1e-2)
        self.f_prev = f


    def uncertainty(self, samples):
        samples = [patch(s, self.dim) for s in samples]
        samples = torch.stack(samples, dim=1)
        print("samples", samples.shape, torch.isnan(samples).any())

        _mean = samples.mean(dim=1)
        _centered = samples - _mean.unsqueeze(1)
        _covar = torch.matmul(_centered.transpose(1, 2), _centered) / (self.sample_num - 1)
        print("sample_nan", torch.isnan(_mean).any(), torch.isnan(_covar).any())
        return _mean, _covar

    def forward(self, x, y, t, f):
        if self.f_prev is None:
            self.f_prev = torch.ones_like(f) * 0.1

        t = t.to(self.device)

        flow, pres = self.pinn.sample_uvp(self.f_prev, f, x, y, t, n=self.sample_num, size=(self.ukf.size, self.ukf.size))
        flow_mean, flow_uncer = self.uncertainty(flow)
        pres_mean, pres_uncer = self.uncertainty(pres)

        self.f_prev = f
        self.ukf.measurement.update_uncertainty(flow_uncer, pres_uncer)

        obsv = torch.cat([patch(f, self.dim), flow_mean, pres_mean], dim=0)
        pred = self.ukf(obsv)
        result = torch.cat([pred, flow_mean, pres_mean], dim=0)
        print("PINN_KF", pred.shape, result.shape)
        return unpatch(pred, self.dim, self.size, 4)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import utils
    from netCDF4 import Dataset
    from configs.pinn.pinn_pde import get_config
    config = get_config()
    config.data.image_size = 192
    n = 6

    data = Dataset('/data1/DATA_PUBLIC/40000-25-400-200.nc')
    print(data.description)
    data = data['data'][801:803+(n-1)*10]

    pikal = PINN_KF(config)

    def prep(d):
        return torch.from_numpy(d[:, 8:200, 4:-4]).to(config.device).unsqueeze(0)


    f = prep(data[0, 2:3])
    v = prep(data[0, 3:5])
    v = torch.cat([v[:, 1:2], v[:, 0:1]], 1)
    p = prep(data[0, 5:6])
    pikal.initialize(f, v, p)

    pred_list = []
    flow_list = []
    pres_list = []
    gt_list = []
    obsv_list = []
    t = torch.Tensor([802])
    with torch.no_grad():
        for d in data[1:]:
            x = prep(d[0:1])
            y = prep(d[1:2])
            f = prep(d[2:3])
            gt_list.append(f)

            f, _ = pikal.ukf.measurement(f, True)

            pred = pikal(x, y, t, f)
            pred_list.append(pred[:, 0:1])
            flow_list.append(pred[:, 1:3])
            pres_list.append(pred[:, 3:4])
            obsv_list.append(f)

            print(t)
            t += 1

    fig, axe = plt.subplots(nrows=5, ncols=n, figsize=(25, 15))
    for i in range(n):
        axe[0, i].imshow(gt_list[i * 10][0, 0].cpu(), vmin=0.0, vmax=0.6)
        axe[1, i].imshow(pred_list[i * 10][0, 0].cpu(), vmin=0.0, vmax=0.6)
        axe[2, i].imshow(obsv_list[i * 10][0, 0].cpu(), vmin=0.0, vmax=0.6)
        axe[3, i].imshow(flow_list[i * 10][0, 0].cpu(), vmin=-1.5, vmax=1.5)
        axe[4, i].imshow(pres_list[i * 10][0, 0].cpu(), vmin=-4, vmax=0.2)

    print("saved")
    plt.savefig('ukf.png')
    plt.show()