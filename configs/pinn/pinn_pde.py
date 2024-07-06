import ml_collections
from configs.pinn.pinn_default_configs import get_default_configs


def get_configs():
  config = get_default_configs()
  
  data = config.data
  data.dataset = 'PDE'
  data.category = 'Theta'
  data.key = 'THETA'
  data.date_range = '2013to2017_1day'
  data.depth = 0
  data.land_cut = 200

  return config