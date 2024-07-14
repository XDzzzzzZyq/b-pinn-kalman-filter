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
import ml_collections
from configs.vp import nc_ddpmpp
# Lint as: python3

def get_config():
  config = nc_ddpmpp.get_config()

  config.training.batch_size = 16

  # inpaint
  inverse = config.inverse = ml_collections.ConfigDict()
  inverse.operator = 'inpaint'
  inverse.invert = False
  inverse.ratio = 0.5
  inverse.sampler = 'controlled'
  inverse.solver = 'fixed' #‘RK45’, ‘RK23’, 'fixed'



  return config
