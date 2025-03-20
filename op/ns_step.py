import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
ns_step_forward = load("ns_step_forward",
                sources=[os.path.join(module_path, f"ns_step.cpp"),
                         os.path.join(module_path, f"ns_step_kernel.cu")],
                extra_include_paths=[os.path.join(module_path, "include")],
                verbose=False)

def diff(dens, dx):
    return ns_step_forward.diff(dens, dx)

def update_density(dens, df_dx, df_dy, vel, dt, dx):
    dens_n = ns_step_forward.update_density(dens, df_dx, df_dy, vel, dt, dx)
    return dens_n

def update_velocity(vel, dv_dx, dv_dy, pres, dt, dx):
    vel_n = ns_step_forward.update_velocity(vel, dv_dx, dv_dy, pres, dt, dx)
    return vel_n

def update_pressure(pres, vel, dt, dx, step=2):
    for _ in range(step):
        pres = ns_step_forward.update_pressure(pres, vel, dt, dx)
    return pres

def vorticity_confinement(vel, weight, dt, dx):
    confinement = ns_step_forward.calc_vort_confinement(vel, dx)
    return vel + dt * weight * confinement
