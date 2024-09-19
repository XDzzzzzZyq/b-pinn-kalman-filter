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

class InpaintKFMeasure(KalmanFilterMeasurementModel):
    def __init__(self, config):
        self.dim = config.kf.patch_size
        super(InpaintKFMeasure, self).__init__(state_dim=self.dim**2, observation_dim=self.dim**2)
        self.ratio = config.inverse.ratio
        self.var = config.inverse.variance
        self.operator = operators.get_operator(config)

    def forward(self, states):
        states = self.operator(states) + torch.randn_like(states) * (self.var ** 0.5)
        covar = torch.eye(self.dim**4) * self.var

        return states, covar


class NSDynamics(DynamicsModel):
    def __init__(self, config):
        self.dim = config.kf.patch_size
        self.size = config.data.image_size
        assert self.size%self.dim==0
        super(NSDynamics, self).__init__(state_dim=self.dim**2)

    def unpatch(self, x):
        return unpatch(x, self.dim, self.size)

    def forward(self, initial_states, controls):
        from op import ns_step
        # initial_states must be patched

        # depatch


        # dynamics

        #patch

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

