import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 64
  training.n_iters = 5000
  training.snapshot_freq = 500
  training.log_freq = 1
  training.eval_freq = 50
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 250
  ## produce samples at each snapshot.

  # data
  config.data = data = ml_collections.ConfigDict()
  data.num_channels = 1
  data.dataset = '_'
  data.image_size = 64
  data.random_flip = False
  data.uniform_dequantization = False
  data.centered = False

  # model
  config.model = model = ml_collections.ConfigDict()
  model.arch = 'flownet'
  model.feature_nums = [16, 32, 64, 96] # 4 levels of features

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 0.001
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config