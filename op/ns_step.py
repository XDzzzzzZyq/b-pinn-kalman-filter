import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
ns_step_forward = load(
    "ns_step_forward",
    sources=[
        os.path.join(module_path, "ns_step.cpp"),
        os.path.join(module_path, "ns_step_kernel.cu"),
    ],
)

def update_density(dens, vel, dt, dx):
    return ns_step_forward.update_density(dens, vel, dt, dx)

def update_velocity(vel, pres, dt, dx):
    return ns_step_forward.update_velocity(vel, pres, dt, dx)

def update_pressure(pres, vel, dt, dx):
    return ns_step_forward.update_pressure(pres, vel, dt, dx)

def vorticity_confinement(vel, weight, dt, dx):
    confinement = ns_step_forward.calc_vort_confinement(vel, dx)
    print(confinement)
    return vel + dt * weight * confinement
