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
import jax
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode


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
  resize_transform  = transforms.Resize(
    size=(resolution, resolution),
    antialias=True,
    interpolation=InterpolationMode.BILINEAR)
  return transforms.Compose([crop_transform, resize_transform])

def resize_small(resolution):
  """Shrink an image to the given resolution."""
  resize_transform  = transforms.Resize(
    size=(resolution, resolution),
    antialias=True,)
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
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  #prefetch_size = tf.data.experimental.AUTOTUNE
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
    pass
  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  '''
  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)
  '''

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_loader, test_loader
