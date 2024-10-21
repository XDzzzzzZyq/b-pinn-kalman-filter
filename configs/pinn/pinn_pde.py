import ml_collections
from configs.pinn.pinn_default_configs import get_default_configs


def get_config():
  config = get_default_configs()
  
  data = config.data
  data.dataset = 'PDE'
  data.dt = 1.7
  data.time_trim = 300

  # inpaint
  inverse = config.inverse = ml_collections.ConfigDict()
  inverse.operator = 'inpaint_rnd'
  inverse.invert = False
  inverse.ratio = 0.9
  inverse.variance = 0.01

  # ukf
  kf = config.kf = ml_collections.ConfigDict()
  kf.patch_size = 8

  return config