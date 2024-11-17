# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import os
import imageio.v2 as imageio


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = imageio.imread(img_path) / 255.0
            images.append(img)
    return images


def trim_images(images, ax, ay, bx, by):
    return np.array([img[ax:bx, ay:by] for img in images])


class Binarize(object):
    def __init__(self, threshold=0.5, invert=False):
        self.threshold = threshold
        self.invert = invert

    def __call__(self, img):
        # Binarize the image tensor
        img = img > self.threshold
        if self.invert:
            img = ~img
        return img.float()


class Repeat:
    def __init__(self, times):
        self.times = times

    def __call__(self, img):
        assert img.ndim <= 3
        return img.repeat(self.times, 1, 1, 1)  # Repeat image 'times' times


class CustomDataset(Dataset):
    def __init__(self, data, split='train', transform=None, land_cut=0, remove_mask=True):
        self.len = len(data)
        self.split = split
        self.data = data
        self.transform = transform
        self.land_cut = land_cut
        self.remove_mask = remove_mask

    def __len__(self):
        return int(self.len * 0.8) if self.split == 'train' else int(self.len * 0.2)

    def __getitem__(self, idx):
        idx = idx if self.split == 'train' else int(self.len * 0.8) + idx
        sample = self.data[idx, 0, self.land_cut:]

        if self.remove_mask:
            sample = sample.data

        if self.transform:
            sample = self.transform(sample)

        return sample, 0


class PDEDataset(Dataset):
    def __init__(self, data, split='train', transform=None, trim=160):
        self.len = len(data)
        self.data = data
        self.split = split
        self.transform = transform
        self.offset = trim

    def __len__(self):
        len = int(self.len * 0.9)-self.offset if self.split == 'train' else int(self.len * 0.1)
        return len - 1

    def __getitem__(self, idx):
        ''' Return a batch of f1, f2, coord, t, target '''

        idx = idx+self.offset if self.split == 'train' else int(self.len * 0.9) + idx
        #t = idx / self.__len__()
        t = idx+1
        sample = self.data[idx:idx+2, :, 5:300,5:-5].data
        sample = torch.from_numpy(sample)
        #sample = sample.reshape(sample.shape[1], sample.shape[2], sample.shape[0])
        #print(sample.shape)
        if self.transform:
            sample = self.transform(sample)
        x_t = sample[1]
        x_p = sample[0]

        return x_p[2:3], x_t[2:3], x_t[0:1], x_t[1:2], t, x_t[3:]
             #    f1,       f2,       x,        y,     t,    target




def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def central_crop(size):
    """Crop the center of an image to the given size."""
    crop_transform = transforms.CenterCrop((size, size))
    return crop_transform


def crop_resize(shape, resolution):
    """Crop and resize an image to the given resolution."""
    h, w = shape[0], shape[1]
    crop = torch.min(h, w)
    crop_transform = central_crop((crop, crop))
    resize_transform = transforms.Resize(
        size=(resolution, resolution),
        antialias=True,
        interpolation=InterpolationMode.BILINEAR)
    return transforms.Compose([crop_transform, resize_transform])


def resize_small(resolution):
    """Shrink an image to the given resolution."""
    resize_transform = transforms.Resize(
        size=(resolution, resolution),
        antialias=True, )
    return resize_transform


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    # prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1
    train_dataset = test_dataset = None

    # Create dataset builders for each dataset.
    if config.data.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([config.data.image_size, config.data.image_size], antialias=True)])

        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    elif config.data.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([config.data.image_size, config.data.image_size], antialias=True)])

        train_dataset = datasets.SVHN(root='./data', split='train',
                                      download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data', split='test',
                                     download=True, transform=transform)


    elif config.data.dataset == 'CELEBA':
        transform = transforms.Compose([
            transforms.ToTensor(),
            central_crop(140),
            transforms.Resize([config.data.image_size, config.data.image_size], antialias=True)])

        train_dataset = datasets.CelebA(root='./data', split='train',
                                        download=True, transform=transform)
        test_dataset = datasets.CelebA(root='./data', split='test',
                                       download=True, transform=transform)

    elif config.data.dataset == 'LSUN':
        if config.data.image_size == 128:
            transform = transforms.Compose([
                transforms.ToTensor(),
                resize_small(config.data.image_size),
                central_crop(config.data.image_size)])

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                central_crop(config.data.image_size)])

        train_dataset = datasets.LSUN(root='./data', classes=[config.data.category], transform=transform)
        test_dataset = datasets.LSUN(root='./data', classes=[config.data.category], transform=transform)

    elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
        raise NotImplementedError("no built-in from pytorch")

    elif config.data.dataset == 'NC':

        from netCDF4 import Dataset

        data = Dataset(
            f'/data1/DATA_PUBLIC/Southern_Ocean/bsose_i122_{config.data.date_range}_{config.data.category}.nc')
        print(data.description)
        data = data[config.data.key]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(config.data.image_size, pad_if_needed=True, padding_mode='constant')])

        train_dataset = CustomDataset(data, split='train', transform=transform, land_cut=config.data.land_cut)
        test_dataset = CustomDataset(data, split='test', transform=transform, land_cut=config.data.land_cut)

    elif config.data.dataset == 'PDE':

        from netCDF4 import Dataset

        data = Dataset('/data1/DATA_PUBLIC/40000-25-400-200.nc')
        print(data.description)
        data = data['data']

        transform = transforms.Compose([
            transforms.RandomCrop(config.data.image_size, pad_if_needed=True, padding_mode='constant')])

        train_dataset = PDEDataset(data, split='train', transform=transform, trim=config.data.time_trim)
        test_dataset = PDEDataset(data, split='test', transform=transform, trim=config.data.time_trim)

    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, test_loader


def get_mask_dataset(config):
    mask_dataset = None

    if config.inverse.operator == 'inpaint':
        transform = transforms.Compose([transforms.Resize(config.data.image_size),
                                        transforms.ToTensor(),
                                        Binarize(config.inverse.ratio, not config.inverse.invert),
                                        Repeat(config.training.batch_size)])

        mask_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    elif config.inverse.operator == 'inpaint_rnd':

        transform = transforms.Compose([transforms.Resize(config.data.image_size),
                                        Binarize(config.inverse.ratio, not config.inverse.invert),
                                        Repeat(config.training.batch_size)])

        rnd_mask = torch.rand(1600, 2, config.data.image_size, config.data.image_size)
        mask_dataset = CustomDataset(rnd_mask, split='train', transform=transform, remove_mask=False)

    mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=True, num_workers=4)
    return mask_loader


if __name__ == '__main__':
    from configs.pinn.pinn_pde import get_config
    config = get_config()

    train_loader, test_loader = get_dataset(config)
    coord, x0, x1, t, target = next(iter(test_loader))

    print(coord.shape, x0.shape, x1.shape, target.shape)
