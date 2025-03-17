import numpy as np
import torch
import matplotlib.pyplot as plt


def show_hist(image, bins=256, range=(0, 255), ax=plt):
    if image.ndim == 4:
        image = image[0]
    ax.hist(image.flatten().detach().cpu().int(), bins=bins, range=range, color='black', alpha=0.7)


def show_accumulative_hist(image, bins=256, range=(0, 255), ax=plt):
    if image.ndim == 4:
        image = image[0]
    hist_counts, bin_edges = torch.histogram(image.flatten().detach().cpu(), bins=bins, range=range)
    cumulative_counts = torch.cumsum(hist_counts, dim=0)
    width = (bin_edges[1] - bin_edges[0]).item()
    ax.bar(bin_edges[1:], cumulative_counts, width=width, align='edge', color='black', alpha=0.7)


def compare_hist(images: list, bins=256, range=(0, 255), accu=False):
    n = len(images)
    hist = show_accumulative_hist if accu else show_hist
    fig, axe = plt.subplots(nrows=1, ncols=n, figsize=(12 * n, 10))
    for i, ax in enumerate(axe):
        hist(images[i], bins=bins, range=range, ax=ax)


if __name__ == '__main__':
    image = load_raw('../imgs/dku_logo_color.raw', 128, 128, 3)
    show_img(image)
    plt.show()
