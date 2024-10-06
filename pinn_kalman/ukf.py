import numpy as np
import torch
from torch import nn
from torchfilter.filters import UnscentedKalmanFilter
from pinn_kalman.ukf_utils import NSDynamics, InpaintKFMeasure, IdentityKFMeasure, patch, unpatch
from pinn import B_PINN

class UKF(nn.Module):

    def __init__(self, config):
        super(UKF, self).__init__()

        self.dim = config.kf.patch_size
        self.size = config.data.image_size

        self.dynamic = NSDynamics(config)
        self.measurement = IdentityKFMeasure(config)
        self.ukf = UnscentedKalmanFilter(dynamics_model=self.dynamic, measurement_model=self.measurement)
        # TODO: better initialization
        N = (self.size//self.dim)**2 * 4
        mean = torch.ones(N, config.kf.patch_size**2).to(config.device) * 0.1
        covariance = torch.eye(config.kf.patch_size**2).unsqueeze(0).repeat(N,1,1).to(config.device)
        self.ukf.initialize_beliefs(mean=mean, covariance=covariance)

    def forward(self, obsv):
        obsv = patch(obsv, self.dim)
        pred = self.ukf(observations=obsv, controls=torch.zeros(obsv.shape[0], device=obsv.device))
        pred = unpatch(pred, self.dim, self.size, 4)
        return pred

class PINN_KF(nn.Module):
    def __init__(self, config):
        super(PINN_KF, self).__init__()
        self.device = config.device

        self.ukf = UKF(config).to(self.device)
        self.pinn = B_PINN(config).to(self.device)

        workdir = "../workdir/pde-bpinn/checkpoints-meta/checkpoint.pth"
        self.pinn = utils.load_checkpoint(workdir, self.pinn, config.device)

        self.f_prev = None

    def forward(self, x, y, t, f):
        if self.f_prev is None:
            self.f_prev = torch.randn_like(f)

        t = t.to(self.device)
        flow, pres = self.pinn.sample_uvp(self.f_prev, f, x, y, t, n=8, size=(self.ukf.size, self.ukf.size))
        flow = torch.stack(flow, dim=0).mean(dim=0)
        pres = torch.stack(pres, dim=0).mean(dim=0)

        self.f_prev = f

        obsv = torch.cat([f, flow, pres], dim=1)
        return self.ukf(obsv)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import utils
    from netCDF4 import Dataset
    from configs.pinn.pinn_pde import get_config
    config = get_config()
    config.data.image_size = 192

    data = Dataset('/data1/DATA_PUBLIC/40000-25-400-200.nc')
    print(data.description)
    data = data['data'][802:813]

    pikal = PINN_KF(config)

    def prep(d):
        return torch.from_numpy(d[:, 8:200, 4:-4]).to(config.device).unsqueeze(0)


    pred_list = []
    gt_list = []
    t = torch.Tensor([802])
    with torch.no_grad():
        for d in data:
            x = prep(d[0:1])
            y = prep(d[1:2])
            f = prep(d[2:3])
            t += 1

            pred = pikal(x, y, t, f)
            pred_list.append(pred)
            gt_list.append(f + torch.randn_like(f) * config.inverse.variance ** 0.5)

            print(t)
    print(len(pred_list), len(gt_list))
    fig, axe = plt.subplots(nrows=2, ncols=2, figsize=(100, 40))
    for i in range(2):
        axe[0, i].imshow(gt_list[i*10][0, 0].cpu())
        axe[1, i].imshow(pred_list[i*10][0, 0].cpu())

    plt.show()