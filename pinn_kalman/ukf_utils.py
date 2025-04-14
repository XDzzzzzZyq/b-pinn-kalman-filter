from torchfilter.base import DynamicsModel, KalmanFilterMeasurementModel
from torchfilter.utils import SigmaPointStrategy
from inverse import operators

import torch
from torchvision.utils import make_grid

def patch(x, p_size):

    # B, C, H, W -> C, B, H, W -> C*B*N, P*P
    x = x.transpose(0, 1)
    x = x.unfold(2, p_size, p_size).unfold(3, p_size, p_size)
    x = x.reshape(-1, p_size**2)

    return x

def unpatch(x, p_size, f_size, channel_num=6):
    num = f_size//p_size
    x = x.reshape(-1, num**2, p_size, p_size).transpose(0, 1)   # C*B, N, P, P
    x = make_grid(x, num, padding=0).reshape(channel_num, -1, f_size, f_size)
    x = x.transpose(0, 1)
    return x

class IdentityKFMeasure(KalmanFilterMeasurementModel):
    def __init__(self, config):
        self.dim = config.kf.patch_size
        self.size = config.data.image_size
        super(IdentityKFMeasure, self).__init__(state_dim=self.dim**2, observation_dim=self.dim**2)
        self.var = config.inverse.variance

        # uses cholesky decomposition of covariance
        # https://stanford-iprl-lab.github.io/torchfilter/api/torchfilter/base/#torchfilter.base.KalmanFilterMeasurementModel
        self.uncer_flow_L = self.var ** 0.5
        self.uncer_pres_L = self.var ** 0.5

    def update_uncertainty(self, uncer_flow, uncer_pres):
        assert uncer_flow.ndim == uncer_pres.ndim == 3
        assert uncer_flow.shape[1] == uncer_flow.shape[2] == self.dim**2
        self.uncer_flow_L = torch.linalg.cholesky(uncer_flow)
        assert uncer_pres.shape[1] == uncer_pres.shape[2] == self.dim**2
        self.uncer_pres_L = torch.linalg.cholesky(uncer_pres)

        print("cholesky", torch.isnan(uncer_flow).any(), torch.isnan(uncer_pres).any())
        print("cholesky", torch.isnan(self.uncer_flow_L).any(), torch.isnan(self.uncer_pres_L).any())

    def get_noise(self, states):
        assert states.ndim == 2
        assert states.shape[1] == self.dim ** 2    # (1+2+1) * N * D, P*P,
        assert states.shape[0] % 4 == 0            # N: num of patches per image, D: num of particles/sigma points per state
        N = (self.size // self.dim) ** 2
        D = states.shape[0] // 4 // N
        f_noise = torch.randn(N, states.shape[1], D).to(states.device) * self.var ** 0.5
        print("measure_noise", self.uncer_flow_L.shape, self.uncer_pres_L.shape, N, D)
        u_noise = torch.bmm(self.uncer_flow_L, torch.randn(N * 2, states.shape[1], D).to(states.device))
        p_noise = torch.bmm(self.uncer_pres_L, torch.randn(N, states.shape[1], D).to(states.device))
        #  (N, P*P, D)

        noise = torch.cat([f_noise, u_noise, p_noise], dim=0)
        noise = noise.transpose(1, 2).reshape(-1, states.shape[1])
        assert noise.shape == states.shape
        return noise

    def get_lower_uncer(self, states):
        assert states.ndim == 2
        assert states.shape[1] == self.dim ** 2
        assert states.shape[0] % 4 == 0
        N = (self.size // self.dim) ** 2
        D = states.shape[0] // 4 // N
        uncer_f = torch.eye(self.dim ** 2) * self.var ** 0.5
        uncer_f = uncer_f.unsqueeze(0).repeat(states.shape[0] // 4, 1, 1).to(states.device)
        uncer_flow = self.uncer_flow_L.repeat(D, 1, 1)
        uncer_pres = self.uncer_pres_L.repeat(D, 1, 1)
        print("measure_uncer", uncer_f.shape, uncer_flow.shape, uncer_pres.shape)
        lower_uncer = torch.cat([uncer_f, uncer_flow, uncer_pres], dim=0)
        return lower_uncer
    def forward(self, states, f_only=False):

        if f_only:
            states = states + torch.randn_like(states) * (self.var ** 0.5)
            covar = torch.eye(self.dim ** 2) * self.var
            covar = covar.unsqueeze(0).repeat(states.shape[0], 1, 1).to(states.device)

            print("id_covar", torch.isnan(covar).any(), torch.isinf(covar).any())
            print("id_states", torch.isnan(states).any()), torch.isinf(states).any()
            return states, covar

        else:
            states = states + self.get_noise(states)
            lower_uncer = self.get_lower_uncer(states)
            print("id_states_full", torch.isnan(states).any(), torch.isnan(lower_uncer).any())
            return states, lower_uncer


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
        import matplotlib.pyplot as plt
        # initial_states must be patched
        # unpatch
        # TODO: consider batch size
        print("initial_states", torch.isnan(initial_states).any(), torch.isinf(initial_states).any())
        initial_states = initial_states.reshape(4, (self.size//self.dim)**2, -1, self.dim**2)  # C, N, D, P**2
        initial_states = initial_states.transpose(1, 2).contiguous()                # C, D, N, P**2
        unpatched = self.unpatch(initial_states)                                               # D, C, W, H

        dt = 0.0005
        dx = 1 / 200

        f = unpatched[:, 0:1]
        v = unpatched[:, 1:3]
        p = unpatched[:, 3:4]

        df_dx, df_dy = ns_step.diff(f, dx)
        dv_dx, dv_dy = ns_step.diff(v, dx)
        # dynamics
        for _ in range(25):
            v, dv_dx, dv_dy = ns_step.update_velocity(v, dv_dx, dv_dy, p, dt, dx)
            v = ns_step.vorticity_confinement(v, 3.0, dt, dx)
            p = ns_step.update_pressure(p, v, dt, dx)
            f, df_dx, df_dy = ns_step.update_density(f, df_dx, df_dy, v, dt, dx)
        print("f2", torch.isnan(f).any(), torch.isnan(v).any(), torch.isnan(p).any())
        state = torch.cat([f, v, p], dim=1)
        if(torch.isnan(state).any()):
            nan_indices = torch.nonzero(torch.isnan(state), as_tuple=False)
            print(state.shape)
            print(nan_indices)
            for i, j, k, l in nan_indices[:1]:
                plt.imshow(state[i, j].cpu(), cmap='gray')
            plt.show()
            assert False

        #patch
        state = patch(state, self.dim)                                                  # C*D*N, P*P
        state = state.reshape(4, -1, (self.size // self.dim) ** 2, self.dim ** 2)       # C, D, N, P*P
        state = state.transpose(1, 2).contiguous().reshape(-1, self.dim ** 2)           # C, N, D, P*P -> C*N*D, P*P
        uncer = torch.eye(self.dim**2, device=state.device).unsqueeze(0).repeat(state.shape[0],1,1) * 1e-8

        print("uncer", torch.isnan(uncer).any())
        print("state", torch.isnan(state).any())
        return state, uncer

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def show_img(image, ax=plt):
        ax.imshow(image.detach().cpu())
    def compare_imgs(images: list):
        n = len(images)
        fig, axe = plt.subplots(nrows=1, ncols=n, figsize=(10 * n, 10))
        for i, ax in enumerate(axe):
            show_img(images[i], ax=ax)

    from netCDF4 import Dataset

    data = Dataset('/data1/DATA_PUBLIC/40000-25-400-200.nc')
    print(data.description)
    data = data['data'][:, 2:6, 8:200, 4:-4]

    from configs.pinn.pinn_pde import get_config
    from op import ns_step
    config = get_config()

    img = torch.from_numpy(data[500:503, :].data).cuda()
    min_max = (img[0,0].min(), img[0,0].max())
    print(img.shape)

    n = 192//8
    p = patch(img, 8)
    print(p.shape)

    image_grid = make_grid(p[:n*n].reshape(n*n, 1, 8, 8), n, padding=0)
    print(image_grid.shape)

    rec = unpatch(p, 8, 192, channel_num=4)
    print(rec.shape)

    f = rec[:, 0:1]
    u = rec[:, 1:2]
    v = rec[:, 2:3]
    p = rec[:, 3:4]
    v = torch.cat([v, u], dim=1)
    dt = 0.0005 * 5
    dx = 1 / 200

    v = ns_step.update_velocity(v, p, dt, dx)
    p = ns_step.update_pressure(p, v, dt, dx)
    f = ns_step.update_density(f, v, dt, dx)

    next = torch.cat([f, v, p], dim=1)

    compare_imgs([img[0, 0], image_grid[0], rec[0, 0], next[0, 0]])
    #plt.imshow(rec[0, 0])

    plt.show()

