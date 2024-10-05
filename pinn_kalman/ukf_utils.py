from torchfilter.base import DynamicsModel, KalmanFilterMeasurementModel
from inverse import operators

import torch
from torchvision.utils import make_grid

def patch(x, p_size):

    # B, (1+2+1), H, W -> (1+2+1), B, H, W -> (1+2+1), S*B, 1
    x = x.transpose(0, 1)
    x = x.unfold(2, p_size, p_size).unfold(3, p_size, p_size)
    x = x.reshape(-1, p_size**2)

    return x

def unpatch(x, p_size, f_size, channel_num=6):
    num = f_size//p_size
    x = x.reshape(-1, num**2, p_size, p_size).transpose(0, 1)
    x = make_grid(x, num, padding=0).reshape(channel_num, -1, f_size, f_size)
    x = x.transpose(0, 1)
    return x

class IdentityKFMeasure(KalmanFilterMeasurementModel):
    def __init__(self, config):
        self.dim = config.kf.patch_size
        super(IdentityKFMeasure, self).__init__(state_dim=self.dim**2, observation_dim=self.dim**2)
        self.var = config.inverse.variance

    def forward(self, states):
        states = states + torch.randn_like(states) * (self.var ** 0.5)
        covar = torch.eye(self.dim**2) * self.var
        covar = covar.unsqueeze(0).repeat(states.shape[0], 1, 1).to(states.device)

        return states, covar

class InpaintKFMeasure(KalmanFilterMeasurementModel):
    def __init__(self, config):
        self.dim = config.kf.patch_size
        super(InpaintKFMeasure, self).__init__(state_dim=self.dim**2, observation_dim=self.dim**2)
        self.ratio = config.inverse.ratio
        self.var = config.inverse.variance
        self.operator = operators.get_operator(config)

    def forward(self, states):
        states = self.operator(states) + torch.randn_like(states) * (self.var ** 0.5)
        covar = torch.eye(self.dim**2) * self.var
        covar = covar.unsqueeze(0).repeat(states.shape[0], 1, 1).to(states.device)

        return states, covar


class NSDynamics(DynamicsModel):
    def __init__(self, config):
        self.dim = config.kf.patch_size
        self.size = config.data.image_size
        assert self.size%self.dim==0
        super(NSDynamics, self).__init__(state_dim=self.dim**2)

    def unpatch(self, x):
        return unpatch(x, self.dim, self.size, 4)

    def forward(self, initial_states, controls):
        from op import ns_step
        # initial_states must be patched
        # unpatch
        unpatched = self.unpatch(initial_states)
        f = unpatched[:, 0:1]
        v = unpatched[:, 1:3]
        p = unpatched[:, 3:4]

        # dynamics

        dt = 0.0005 * 5
        dx = 1 / 200
        v = ns_step.update_velocity(v, p, dt, dx)
        p = ns_step.update_pressure(p, v, dt, dx)
        f = ns_step.update_density(f, v, dt, dx)

        # TODO: using patched covariance
        # TODO: consider batches
        f_std = f.std(dim=0)
        v_std = v.std(dim=0)
        p_std = p.std(dim=0)

        #patch
        state = patch(torch.cat([f, v, p], dim=1), self.dim)
        uncer = patch(torch.cat([f_std, v_std, p_std], dim=0), self.dim)
        uncer = torch.diag_embed(uncer).repeat(2*self.dim**2+1,1,1)
        return state, uncer

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from netCDF4 import Dataset

    data = Dataset('/data1/DATA_PUBLIC/40000-25-400-200.nc')
    print(data.description)
    data = data['data'][:, :, 8:200, 4:-4]

    from configs.pinn.pinn_pde import get_config
    config = get_config()

    img = torch.from_numpy(data[500:503, :].data)
    print(img.shape)
    plt.imshow(img[0, 3])

    p = patch(img, 64)
    B = p.shape[0]//6
    f_p = p[2*B:3*B].reshape(-1, 64, 64)
    print(f_p.shape)

    image_grid = make_grid(f_p[:9].unsqueeze(1), 3, padding=5)
    print(image_grid.shape)
    plt.imshow(image_grid[0])

    rec = unpatch(p, 64, 192)
    print(rec.shape)
    plt.imshow(rec[0, 3])

    plt.show()

