from pinn_kalman.pinn import PINN, B_PINN

import torch

def simulate(model:PINN, begin, t_range=(0, 100), stride=1):

    def prep(data):
        return torch.from_numpy(data[:, 8:200, 4:-4]).to(model.device).unsqueeze(0)

    result = []
    vel = []

    t0, tm = t_range

    f1 = prep(begin[0+t0, 2:3])
    f2 = prep(begin[1+t0, 2:3])
    x  = prep(begin[0+t0, 0:1])
    y  = prep(begin[0+t0, 1:2])

    for t in torch.arange(*t_range, stride):
        t = t.unsqueeze(0).to(model.device)
        flow, pres = model(f1, f2, x, y, t, size=(192, 192))
        f = model.step(f2, flow[-1])
        result.append(f)
        vel.append(flow[-1])

        f1, f2 = f2, f

    return result, vel

def sign(x):
    return -1.0 if x < 0.0 else 1.0

dt = 0.0005
dx = 1/200

def _grad_construct(f):
    dF_dx = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    dF_dy = (f[2:, :] - f[:-2, :]) / (2 * dx)

    # Forward difference for the first point
    dF_dx_forward = (f[:, 1] - f[:, 0]) / dx
    dF_dy_forward = (f[1, :] - f[0, :]) / dx

    # Backward difference for the last point
    dF_dx_backward = (f[:, -1] - f[:, -2]) / dx
    dF_dy_backward = (f[-1, :] - f[-2, :]) / dx

    # Combine results
    dF_dx = torch.cat((dF_dx_forward.unsqueeze(1), dF_dx, dF_dx_backward.unsqueeze(1)), dim=1)
    dF_dy = torch.cat((dF_dy_forward.unsqueeze(0), dF_dy, dF_dy_backward.unsqueeze(0)), dim=0)

    return dF_dx, dF_dy
def _cip_advect(fn, fc, fxc, fyc, v, i, j):
    i_s = int(sign(v[1, i, j]))
    j_s = int(sign(v[0, i, j]))
    i_m = i - i_s
    j_m = j - j_s

    tmp1 = fc[i, j] - fc[i, j_m] - fc[i_m, j] + fc[i_m, j_m]
    tmp2 = fc[i_m, j] - fc[i, j]
    tmp3 = fc[i, j_m] - fc[i, j]

    i_s_denom = i_s * dx ** 3
    j_s_denom = j_s * dx ** 3

    a = (i_s * (fxc[i_m, j] + fxc[i, j]) * dx - 2.0 * (-tmp2)) / i_s_denom
    b = (j_s * (fyc[i, j_m] + fyc[i, j]) * dx - 2.0 * (-tmp3)) / j_s_denom
    c = (-tmp1 - i_s * (fxc[i, j_m] - fxc[i, j]) * dx) / j_s_denom
    d = (-tmp1 - j_s * (fyc[i_m, j] - fyc[i, j]) * dx) / i_s_denom
    e = (3.0 * tmp2 + i_s * (fxc[i_m, j] + 2.0 * fxc[i, j]) * dx) / dx ** 2
    f = (3.0 * tmp3 + j_s * (fyc[i, j_m] + 2.0 * fyc[i, j]) * dx) / dx ** 2
    g = (-(fyc[i_m, j] - fyc[i, j]) + c * dx ** 2) / (i_s * dx)

    X = -v[1, i, j] * dt
    Y = -v[0, i, j] * dt

    # 移流量の更新
    fn[i, j] = (
            ((a * X + c * Y + e) * X + g * Y + fxc[i, j]) * X + ((b * Y + d * X + f) * Y + fyc[i, j]) * Y + fc[i, j])

def advect(f, flow):
    f = f.squeeze(0).squeeze(0)
    grad_x, grad_y = _grad_construct(f)

    fn = torch.zeros_like(f)

    for i in range(f.shape[0]-1):
        for j in range(f.shape[1]-1):
            _cip_advect(fn, f, grad_x, grad_y, flow.squeeze(0), i, j)

    return fn.unsqueeze(0).unsqueeze(0)


def step(model:PINN, begin, t_range=(0, 100), stride=1):

    def prep(data):
        return torch.from_numpy(data[:, 8:200, 4:-4]).to(model.device).unsqueeze(0)

    result = []
    vel = []

    t0, tm = t_range

    f1 = prep(begin[0+t0, 2:3])
    f2 = prep(begin[1+t0, 2:3])
    x  = prep(begin[0+t0, 0:1])
    y  = prep(begin[0+t0, 1:2])

    for t in torch.arange(*t_range, stride):
        flow = prep(begin[t, 3:5])
        f = advect(f2, flow)
        result.append(f)
        vel.append(flow)

        f1, f2 = f2, f

    return result, vel


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import utils
    from pinn_kalman.pinn_lib import unbatch

    from netCDF4 import Dataset

    data = Dataset('/data1/DATA_PUBLIC/40000-25-400-200.nc')
    print(data.description)
    data = data['data']

    from configs.pinn.pinn_pde import get_config
    config = get_config()

    model = PINN(config)

    workdir = "../workdir/pde-pinn/checkpoints-meta/checkpoint_pinn.pth"
    model = utils.load_checkpoint(workdir, model, config.device)

    with torch.no_grad():
        result, vel = step(model, data, t_range=(802, 902))
        result, vel = result[::10], vel[::10]

    fig, axe = plt.subplots(nrows=4, ncols=10, figsize=(100, 30))
    for i in range(10):
        axe[0, i].imshow(vel[i][0, 0].cpu())
        axe[1, i].imshow(data[803 + i * 10, 3, 8:200, 4:-4])
        axe[2, i].imshow(result[i].squeeze().cpu())
        axe[3, i].imshow(data[803 + i * 10, 2, 8:200, 4:-4])

    plt.show()

