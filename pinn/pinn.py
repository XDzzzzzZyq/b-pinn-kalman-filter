"""
PINN+incompressible NS equation
2-dimensional unsteady
PINN model +LOSS function
PINN融合不可压缩NS方程
二维非定常流动
PINN模型 + LOSS函数
"""
import os
import numpy as np
import torch
import torch.nn as nn
from models.ddpm import UNet
import torch.nn.functional as F
import torch.optim as opt 
from pyDOE import lhs


# Define network structure, specified by a list of layers indicating the number of layers and neurons
# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN_Net(nn.Module):

    """
    Input:  X, t  := (x, y, f) : shape=(B, 3, N, N), (B, )
    Output: Y     := (u, v, p) : shape=(B, 3, N, N)
    """
    def __init__(self, config, mean_value, std_value):
        super(PINN_Net, self).__init__()
        self.X_mean = torch.from_numpy(mean_value.astype(np.float32)).to(config.device)
        self.X_std = torch.from_numpy(std_value.astype(np.float32)).to(config.device)
        self.device = config.device

        self.model = torch.nn.DataParallel(UNet(config)).to(self.device)

    # 0-1 norm of input variable
    # 对输入变量进行0-1归一化
    def inner_norm(self, X):
        X_norm = (X - self.X_mean) / self.X_std
        return X_norm

    def forward_inner_norm(self, X, t):
        X_norm = self.inner_norm(X)
        predict = self.model(X_norm, t)
        return predict

    def forward(self, X, t):
        predict = self.model(X, t)
        return predict

    # derive loss for data
    # 类内方法：求数据点的loss
    def data_mse(self, X, t, Y):
        predict_out = self.forward(X, t)
        u, v, p = Y[:,0], Y[:,1], Y[:,2]
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # derive loss for data without pressure
    # 类内方法：求数据点的loss(不含压力数据)
    def data_mse_without_p(self, X, t, Y):
        predict_out = self.forward(X, t)
        u, v = Y[:, 0], Y[:, 1]
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # predict
    # 类内方法：预测
    def predict(self, X, t):
        predict_out = self.forward(X, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        return u_predict, v_predict, p_predict

    # derive loss for equation
    def equation_mse_dimensionless(self, X, t, Re):
        #print("asdasdasda")
        predict_out = self.forward(X, t)
        x, y = X[:,0], X[:,1]
        #print(predict_out.shape, "8908098080")
        # 获得预测的输出u,v,p
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        x.requires_grad_()
        y.requires_grad_()
        t.requires_grad_()
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导

        #print()
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True, allow_unused=True, materialize_grads=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation

    def data_mse_inner_norm(self, X, t, Y):
        predict_out = self.forward_inner_norm(X, t)
        u, v, p = Y[:, 0], Y[:, 1], Y[:, 2]
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v) + mse(p_predict, p)
        return mse_predict

    # derive loss for data without pressure
    # 类内方法：求数据点的loss(不含压力数据)
    def data_mse_without_p_inner_norm(self, X, t, Y):
        predict_out = self.forward_inner_norm(X, t)
        u, v = Y[:, 0], Y[:, 1]
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        mse = torch.nn.MSELoss()
        mse_predict = mse(u_predict, u) + mse(v_predict, v)
        return mse_predict

    # predict
    # 类内方法：预测
    def predict_inner_norm(self, X, t):
        predict_out = self.forward_inner_norm(X, t)
        u_predict = predict_out[:, 0].reshape(-1, 1)
        v_predict = predict_out[:, 1].reshape(-1, 1)
        p_predict = predict_out[:, 2].reshape(-1, 1)
        return u_predict, v_predict, p_predict

    # derive loss for equation
    def equation_mse_dimensionless_inner_norm(self, X, t, Re):
        predict_out = self.forward_inner_norm(X, t)
        x, y = X[:,0], X[:,1]
        # 获得预测的输出u,v,w,p,k,epsilon
        u = predict_out[:, 0].reshape(-1, 1)
        v = predict_out[:, 1].reshape(-1, 1)
        p = predict_out[:, 2].reshape(-1, 1)
        # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
        # first-order derivative
        # 一阶导
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True, allow_unused=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True, allow_unused=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True, allow_unused=True)[0]
        v_x = torch.autograd.grad(v.sum(), x, create_graph=True, allow_unused=True)[0]
        v_y = torch.autograd.grad(v.sum(), y, create_graph=True, allow_unused=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True, allow_unused=True)[0]
        p_x = torch.autograd.grad(p.sum(), x, create_graph=True, allow_unused=True)[0]
        p_y = torch.autograd.grad(p.sum(), y, create_graph=True, allow_unused=True)[0]
        # second-order derivative
        # 二阶导
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, allow_unused=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True, allow_unused=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True, allow_unused=True)[0]
        v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True, allow_unused=True)[0]
        # residual
        # 计算偏微分方程的残差
        f_equation_mass = u_x + v_y
        f_equation_x = u_t + (u * u_x + v * u_y) + p_x - 1.0 / Re * (u_xx + u_yy)
        f_equation_y = v_t + (u * v_x + v * v_y) + p_y - 1.0 / Re * (v_xx + v_yy)
        mse = torch.nn.MSELoss()
        batch_t_zeros = torch.zeros_like(x, dtype=torch.float32, device=self.device)
        mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros) + \
                       mse(f_equation_mass, batch_t_zeros)

        return mse_equation


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from configs.pinn.pinn_pde import get_config

    config = get_config()
    mean_value = np.array([0.5, 0.5, 0.5])  # 需要提供适当的归一化均值和标准差
    std_value = np.array([0.5, 0.5, 0.5])

    Reynolds_number = 100  # 例子中的雷诺兹数

    # 创建设备（允许CPU或GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始模型、优化器和损失函数
    model = PINN_Net(config, mean_value, std_value)
    state = torch.load("pinn_ns_model.pth", map_location=device)
    model.load_state_dict(state)

    # 可视化网格的设置
    num_points = 100  # 可视化时每个维度的数据点数量
    x_vis = np.linspace(0, 1, num_points)
    y_vis = np.linspace(0, 1, num_points)
    t_vis = np.array([1])  # 假设我们评估特定时间点t=1

    # 创建网格数据
    X_vis, Y_vis = np.meshgrid(x_vis, y_vis)
    T_vis = np.full(X_vis.shape, t_vis)

    # 将numpy数组转化为张量
    X_tensor = torch.from_numpy(X_vis.reshape(-1, 1)).float().to(device).requires_grad_()
    Y_tensor = torch.from_numpy(Y_vis.reshape(-1, 1)).float().to(device).requires_grad_()
    T_tensor = torch.from_numpy(T_vis.reshape(-1, 1)).float().to(device).requires_grad_()

    # 对网格上的点进行预测
    with torch.no_grad():
        u_pred, v_pred, p_pred = model.predict(X_tensor, Y_tensor, T_tensor)

    # 将预测结果转换成numpy数组，并且调整形状以匹配图形
    u_pred_np = u_pred.cpu().numpy().reshape(X_vis.shape)
    # 使用matplotlib进行绘图
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X_vis, Y_vis, u_pred_np, 100, cmap=cm.viridis)
    plt.colorbar(contour)
    plt.title('Predicted u velocity field at t=0')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    #u_gt = torch.autograd.grad(outputs=)