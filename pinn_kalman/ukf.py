import numpy as np
import torch
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from pinn import B_PINN  # 假设B_PINN在pinn.py中定义

class UKF:
    def __init__(self, dim_x, dim_z, b_pinn, dt):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.b_pinn = b_pinn
        self.dt = dt

        
        points = MerweScaledSigmaPoints(dim_x, alpha=1e-3, beta=2., kappa=0)
        self.ukf = UnscentedKalmanFilter(dim_x, dim_z, dt, self.fx, self.hx, points)

        self.ukf.x = np.zeros(dim_x)
        self.ukf.P = np.eye(dim_x)

    def fx(self, x, dt):
        
        f1, f2, x_tensor, y_tensor, t_tensor = self.prepare_inputs(x)
        
        
        flow_mean, pres_mean, f_mean, flow_var, pres_var, f_var = self.b_pinn.predict(f1, f2, x_tensor, y_tensor, t_tensor)

        Q = np.concatenate([
            flow_var.cpu().numpy().flatten(),
            pres_var.cpu().numpy().flatten(),
            f_var.cpu().numpy().flatten()
        ])
        self.ukf.Q = Q 
        self.ukf.R = #这个你加一下
        x_next = np.concatenate([
            flow_mean.cpu().numpy().flatten(),
            pres_mean.cpu().numpy().flatten(),
            f_mean.cpu().numpy().flatten()
        ])

        return x_next

    def hx(self, x):
        #这个你加一下
        pass

    def prepare_inputs(self, x):
    
        flow = torch.from_numpy(x[:self.dim_x//4*2].reshape(2, 64, 64)).to(self.b_pinn.device)
        pressure = torch.from_numpy(x[self.dim_x//4*2:self.dim_x//4*3].reshape(1, 64, 64)).to(self.b_pinn.device)
        f = torch.from_numpy(x[self.dim_x//4*3:].reshape(1, 64, 64)).to(self.b_pinn.device)

        # 生成时间和空间坐标, 这个我忘怎么弄了，你加一下。
        t = torch.tensor([0.0]).to(self.b_pinn.device)  
        x_tensor = torch.linspace(0, 1, 64).to(self.b_pinn.device).repeat(64, 1)
        y_tensor = torch.linspace(0, 1, 64).to(self.b_pinn.device).repeat(64, 1).t()

        return flow, pressure, x_tensor, y_tensor, t

    def step(self, z):
        # 执行一个UKF步骤
        self.ukf.predict()
        self.ukf.update(z)
        return self.ukf.x, self.ukf.P

