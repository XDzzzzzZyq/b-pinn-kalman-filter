from pinn_kalman.pinn import PINN, B_PINN
from op import ns_step

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

dt = 0.0005 * 5
dx = 1/200

def step(model:PINN, begin, t_range=(0, 100), stride=1):

    def prep(data):
        return torch.from_numpy(data[:, 8:200, 4:-4]).to(model.device).unsqueeze(0)

    result = []
    vel = []
    pres = []

    t0, tm = t_range

    f = prep(begin[0+t0, 2:3])
    v = prep(begin[0 + t0, 3:5])
    v = torch.cat([v[:, 1:2], v[:, 0:1]], 1)
    p = prep(begin[0 + t0, 5:6])

    for t in torch.arange(*t_range, stride):
        v = ns_step.update_velocity(v, p, dt, dx)
        p = ns_step.update_pressure(p, v, dt, dx)
        f = ns_step.update_density(f, v, dt, dx)
        result.append(f)
        vel.append(v)
        pres.append(p)

    return result, vel, pres


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
        result, vel, pres = step(model, data, t_range=(802, 902))
        result, vel, pres = result[::10], vel[::10], pres[::10]

    if True:
        fig, axe = plt.subplots(nrows=4, ncols=10, figsize=(100, 40))
        for i in range(10):
            axe[0, i].imshow(result[i].squeeze().cpu())

            axe[1, i].imshow(vel[i][0, 0].cpu())

            axe[2, i].imshow(vel[i][0, 1].cpu())

            axe[3, i].imshow(pres[i].squeeze().cpu())
            print((result[i].squeeze().cpu() - data[803 + i * 10, 2, 8:200, 4:-4].data).square().mean())

    else:
        fig, axe = plt.subplots(nrows=8, ncols=10, figsize=(100, 60))
        for i in range(10):
            axe[0, i].imshow(result[i].squeeze().cpu())
            axe[1, i].imshow(data[803 + i * 10, 2, 8:200, 4:-4])

            axe[2, i].imshow(vel[i][0, 0].cpu())
            axe[3, i].imshow(data[803 + i * 10, 3, 8:200, 4:-4])

            axe[4, i].imshow(vel[i][0, 1].cpu())
            axe[5, i].imshow(data[803 + i * 10, 4, 8:200, 4:-4])

            axe[6, i].imshow(pres[i].squeeze().cpu())
            axe[7, i].imshow(data[803 + i * 10, 5, 8:200, 4:-4])
            print((result[i].squeeze().cpu() - data[803 + i * 10, 2, 8:200, 4:-4].data).square().mean())


    plt.show()

