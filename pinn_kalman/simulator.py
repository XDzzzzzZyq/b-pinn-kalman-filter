from pinn_kalman.pinn import PINN, B_PINN

import torch

def simulate(model:PINN, begin, t_range=(0, 100), stride=1):

    def prep(data):
        return torch.from_numpy(data[8:200, 4:-4]).to(model.device).unsqueeze(0).unsqueeze(0)

    result = []
    vel = []

    f1 = prep(begin[0, 2])
    f2 = prep(begin[1, 2])
    x = prep(begin[0, 0])
    y = prep(begin[0, 1])

    for t in torch.arange(*t_range, stride):
        flow, pres = model(f1, f2, x, y, t.unsqueeze(0).to(model.device), size=(192, 192))
        f = model.step(f2, flow[-1])
        result.append(f)
        vel.append(flow[-1])

        f1, f2 = f2, f

    return result, vel


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import utils
    from pinn_kalman.pinn_lib import unbatch

    from netCDF4 import Dataset

    data = Dataset('/data1/40000-25-400-200.nc')
    print(data.description)
    data = data['data']

    from configs.pinn.pinn_pde import get_config
    config = get_config()

    model = PINN(config)

    workdir = "../workdir/pde-pinn/checkpoints-meta/checkpoint_pinn.pth"
    model = utils.load_checkpoint(workdir, model, config.device)

    with torch.no_grad():
        result, vel = simulate(model, data[800:802], t_range=(802, 902))
        result, vel = result[::10], vel[::10]

    fig, axe = plt.subplots(nrows=4, ncols=10, figsize=(100, 30))
    for i in range(10):
        axe[0, i].imshow(vel[i][0, 0].cpu())
        axe[1, i].imshow(data[803 + i * 10, 3, 8:200, 4:-4])
        axe[2, i].imshow(result[i].squeeze().cpu())
        axe[3, i].imshow(data[803 + i * 10, 2, 8:200, 4:-4])

    plt.show()

