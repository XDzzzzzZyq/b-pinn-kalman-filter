import datasets
from .operators import InpaintOperator
from .conditional_sampling import get_sampler
from models import utils as mutils
from utils import save_checkpoint, load_checkpoint, restore_checkpoint
import os
import numpy as np
from torchvision.utils import make_grid, save_image


def _show_result(result):
    from torchvision.utils import make_grid, save_image
    import matplotlib.pyplot as plt
    import numpy as np

    nrow = int(np.sqrt(result.shape[0]))
    image_grid = make_grid(result, nrow, padding=0)
    print(image_grid.shape, image_grid.min(), image_grid.max())

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    axe.imshow(image_grid[0])
    plt.show()

def get_operator(config):

    if config.inverse.operator in ['inpaint', 'inpaint_rnd']:
        mask_ds = datasets.get_mask_dataset(config)
        mask_iter = iter(mask_ds)
        mask, _ = next(mask_iter)

        operator = InpaintOperator(mask=mask.squeeze(0).to(config.device))

    else:
        raise NotImplementedError

    return operator

def get_obsvsde(config, y0, operator):
    from run_lib import _get_sde
    from sde_lib import LOBSVSDE

    sde, sampling_eps = _get_sde(config)
    if config.inverse.sampler in ['controlled', 'dps']:
        obsvsde = LOBSVSDE(sde, y0, operator)
    else:
        raise NotImplementedError

    return obsvsde, sampling_eps

def _inverse_fn(config, score_model):
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)

    _, test_ds = datasets.get_dataset(config)
    test_iter = iter(test_ds)
    origin, _ = next(test_iter)

    operator = get_operator(config)
    observation_vis = operator(origin.to(config.device), keep_shape=True) # for visualization
    observation = operator(origin.to(config.device), keep_shape=False) # ill-posed observation

    obsvsde, sampling_eps = get_obsvsde(config, observation, operator)
    sampling_fn = get_sampler(config, obsvsde, sampling_shape, eps=sampling_eps)

    sample = sampling_fn(score_model)
    return origin, observation_vis, sample, operator

def inverse(config, ckptdir, workdir, visualize=True):
    score_model = mutils.create_model(config)
    score_model = load_checkpoint(ckptdir, score_model, config.device)

    origin, observation, sample, operator = _inverse_fn(config, score_model)

    workdir = os.path.join(workdir, f"{config.inverse.operator}-{config.inverse.ratio}")
    os.makedirs(workdir, exist_ok=True)
    nrow = int(np.sqrt(sample.shape[0]))
    obsv_grid = make_grid(observation, nrow, padding=2)
    image_grid = make_grid(sample, nrow, padding=2)

    with open(os.path.join(workdir, "inverse.png"), "wb") as fout:
        save_image(image_grid, fout)

    with open(os.path.join(workdir, "observation.png"), "wb") as fout:
        save_image(obsv_grid, fout)

    if visualize:
        import matplotlib.pyplot as plt
        fig, axe = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))
        axe[0].imshow(obsv_grid[0].cpu())
        axe[1].imshow(image_grid[0].cpu())
        plt.show()
        plt.savefig(os.path.join(workdir, "visualize.png"))

def evaluate_inverse(config, origin, inv, operator):
    import torch

    loss = torch.nn.MSELoss()
    device = config.device

    if config.inverse.operator in ['inpaint', 'inpaint_rnd']:

        # mask the masked area and left inverted area
        #origin = operator(origin.to(config.device), keep_shape=False, invert=True)
        #inv = operator(inv.to(config.device), keep_shape=False, invert=True)

        return loss(origin.to(device), inv.to(device)).item()

    else:
        raise NotImplementedError
