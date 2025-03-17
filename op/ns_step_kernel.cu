// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "pixelUtils.cuh"

template <typename scalar_t>
static __global__ void update_gradient_kernel(
    scalar_t* df_dx,
    scalar_t* df_dy,
    const scalar_t* field,
    float dx
) {
    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int loc = batch + y * gridDim.x + x;

    if(blockIdx.z%2==0)
        df_dx[loc] = diff_x(field, x, y, dx);
    else
        df_dy[loc] = diff_y(field, x, y, dx);
}

template <typename scalar_t>
static __global__ void update_gradient2D_kernel(
    scalar_t* df_dx,
    scalar_t* df_dy,
    const scalar_t* field,
    float dx
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    if(blockIdx.z%2==0)
        set_v(df_dx, diff_xv(field, x, y, dx), x, y);
    else
        set_v(df_dy, diff_yv(field, x, y, dx), x, y);
}

template <typename scalar_t>
static __global__ void update_laplacian_kernel(
    scalar_t* lapla,
    const scalar_t* field,
    float dx
) {
    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int loc = batch + y * gridDim.x + x;

    lapla[loc] = diff2_x(field, x, y, dx) + diff2_y(field, x, y, dx);
}

template <typename scalar_t>
static __global__ void update_laplacian2D_kernel(
    scalar_t* lapla,
    const scalar_t* field,
    float dx
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    set_v(lapla, diff2_xv(field, x, y, dx) + diff2_yv(field, x, y, dx), x, y);
}

template <typename scalar_t>
static __global__ void update_vorticity_kernel(
    scalar_t* vort,
    const scalar_t* field,
    float dx
) {
    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int loc = batch + y * gridDim.x + x;

    scalar_t vort_val = diff_xv(field, x, y, dx).y - diff_yv(field, x, y, dx).x;
    vort[loc] = vort_val;
}

template <typename scalar_t>
static __global__ void update_vort_confinement_kernel(
    scalar_t* confinement,
    const scalar_t* vort,
    const scalar_t* dv_dx,
    const scalar_t* dv_dy
) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    scalar_t dx_val = get(dv_dx, x, y);
    scalar_t dy_val = get(dv_dy, x, y);
    scalar_t len = sqrtf(dx_val*dx_val + dy_val*dy_val);

    vector<scalar_t> grad{dy_val, -dx_val};
    grad = grad / len;

    vector<scalar_t> vort_val = get_v(vort, x, y);
    vector<scalar_t> conf = grad * vort_val;

    set_v(confinement, conf.clamp(-1.0, 1.0), x, y);
}

template <typename scalar_t>
static __global__ void cip_advect_kernel(
    scalar_t* dens_n,
    scalar_t* df_dx_n,
    scalar_t* df_dy_n,
    const scalar_t* dens_c,
    const scalar_t* dens_dx,
    const scalar_t* dens_dy,
    const scalar_t* vel_c,
    float dt, float dx
) {

    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;

    int x_s = sign(get_v(vel_c, x, y).x);
    int y_s = sign(get_v(vel_c, x, y).y);
    int x_m = clamp_x(x - x_s);
    int y_m = clamp_y(y - y_s);

    scalar_t tmp1 = get(dens_c, x, y) - get(dens_c, x, y_m) - get(dens_c, x_m, y) + get(dens_c, x_m, y_m);
    scalar_t tmp2 = get(dens_c, x_m, y) - get(dens_c, x, y);
    scalar_t tmp3 = get(dens_c, x, y_m) - get(dens_c, x, y);

    scalar_t x_s_denom = x_s * dx*dx*dx;
    scalar_t y_s_denom = y_s * dx*dx*dx;

    scalar_t a = (x_s * (get(dens_dx, x_m, y) + get(dens_dx, x, y)) * dx - 2.0 * (-tmp2)) / x_s_denom;
    scalar_t b = (y_s * (get(dens_dy, x, y_m) + get(dens_dy, x, y)) * dx - 2.0 * (-tmp3)) / y_s_denom;
    scalar_t c = (-tmp1 - x_s * (get(dens_dx, x, y_m) - get(dens_dx, x, y)) * dx) / y_s_denom;
    scalar_t d = (-tmp1 - y_s * (get(dens_dy, x_m, y) - get(dens_dy, x, y)) * dx) / x_s_denom;
    scalar_t e = (3.0 * tmp2 + x_s * (get(dens_dx, x_m, y) + 2.0 * get(dens_dx, x, y)) * dx) / dx/dx;
    scalar_t f = (3.0 * tmp3 + y_s * (get(dens_dy, x, y_m) + 2.0 * get(dens_dy, x, y)) * dx) / dx/dx;
    scalar_t g = (-(get(dens_dy, x_m, y) - get(dens_dy, x, y)) + c * dx*dx) / (x_s * dx);

    scalar_t X = -get_v(vel_c, x, y).x * dt;
    scalar_t Y = -get_v(vel_c, x, y).y * dt;

    int loc = batch + y * gridDim.x + x;
    dens_n[loc] = ( \
            ((a * X + c * Y + e) * X + g * Y + get(dens_dx, x, y)) * X \
            + ((b * Y + d * X + f) * Y + get(dens_dy, x, y)) * Y \
            + get(dens_c, x, y) \
    );

    scalar_t Fx = (3.0 * a * X + 2.0 * c * Y + 2.0 * e) * X + (d * Y + g) * Y + get(dens_dx, x, y);
    scalar_t Fy = (3.0 * b * Y + 2.0 * d * X + 2.0 * f) * Y + (c * X + g) * X + get(dens_dy, x, y);

    vector<scalar_t> dv_dx = diff_xv(vel_c, x, y, dx);
    vector<scalar_t> dv_dy = diff_yv(vel_c, x, y, dx);

    df_dx_n[loc] = Fx - dt * (Fx * dv_dx.x + Fy * dv_dx.y) / 2.0;
    df_dy_n[loc] = Fy - dt * (Fx * dv_dy.x + Fy * dv_dy.y) / 2.0;
}

template <typename scalar_t>
static __global__ void cip_advect_vec_kernel(
    scalar_t* dens2d_n,
    scalar_t* dens2d_dx_n,
    scalar_t* dens2d_dy_n,
    const scalar_t* dens2d_c,
    const scalar_t* dens2d_dx,
    const scalar_t* dens2d_dy,
    const scalar_t* vel_c,
    float dt, float dx
) {

    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;

    int x_s = sign(get_v(vel_c, x, y).x);
    int y_s = sign(get_v(vel_c, x, y).y);
    int x_m = clamp_x(x - x_s);
    int y_m = clamp_y(y - y_s);

    vector<scalar_t> tmp1 = get_v(dens2d_c, x, y) - get_v(dens2d_c, x, y_m) - get_v(dens2d_c, x_m, y) + get_v(dens2d_c, x_m, y_m);
    vector<scalar_t> tmp2 = get_v(dens2d_c, x_m, y) - get_v(dens2d_c, x, y);
    vector<scalar_t> tmp3 = get_v(dens2d_c, x, y_m) - get_v(dens2d_c, x, y);

    scalar_t x_s_denom = x_s * dx*dx*dx;
    scalar_t y_s_denom = y_s * dx*dx*dx;

    vector<scalar_t> a = (x_s * (get_v(dens2d_dx, x_m, y) + get_v(dens2d_dx, x, y)) * dx - 2.0 * (-1.0*tmp2)) / x_s_denom;
    vector<scalar_t> b = (y_s * (get_v(dens2d_dy, x, y_m) + get_v(dens2d_dy, x, y)) * dx - 2.0 * (-1.0*tmp3)) / y_s_denom;
    vector<scalar_t> c = (-1.0*tmp1 - x_s * (get_v(dens2d_dx, x, y_m) - get_v(dens2d_dx, x, y)) * dx) / y_s_denom;
    vector<scalar_t> d = (-1.0*tmp1 - y_s * (get_v(dens2d_dy, x_m, y) - get_v(dens2d_dy, x, y)) * dx) / x_s_denom;
    vector<scalar_t> e = (3.0 * tmp2 + x_s * (get_v(dens2d_dx, x_m, y) + 2.0 * get_v(dens2d_dx, x, y)) * dx) / dx/dx;
    vector<scalar_t> f = (3.0 * tmp3 + y_s * (get_v(dens2d_dy, x, y_m) + 2.0 * get_v(dens2d_dy, x, y)) * dx) / dx/dx;
    vector<scalar_t> g = (-1.0*(get_v(dens2d_dy, x_m, y) - get_v(dens2d_dy, x, y)) + c * dx*dx) / (x_s * dx);

    scalar_t X = -1.0*get_v(vel_c, x, y).x * dt;
    scalar_t Y = -1.0*get_v(vel_c, x, y).y * dt;

    int loc = batch + y * gridDim.x + x;
    vector<scalar_t> New = ( \
            ((a * X + c * Y + e) * X + g * Y + get_v(dens2d_dx, x, y)) * X \
            + ((b * Y + d * X + f) * Y + get_v(dens2d_dy, x, y)) * Y \
            + get_v(dens2d_c, x, y) \
    );
    set_v(dens2d_n, New, x, y);

    vector<scalar_t> Fx = (3.0 * a * X + 2.0 * c * Y + 2.0 * e) * X + (d * Y + g) * Y + get_v(dens2d_dx, x, y);
    vector<scalar_t> Fy = (3.0 * b * Y + 2.0 * d * X + 2.0 * f) * Y + (c * X + g) * X + get_v(dens2d_dy, x, y);

    vector<scalar_t> dv_dx = diff_xv(vel_c, x, y, dx);
    vector<scalar_t> dv_dy = diff_yv(vel_c, x, y, dx);

    set_v(dens2d_dx_n, Fx - dt * (Fx * dv_dx.x + Fy * dv_dx.y) / 2.0, x, y);
    set_v(dens2d_dy_n, Fy - dt * (Fx * dv_dy.x + Fy * dv_dy.y) / 2.0, x, y);
}

template <typename scalar_t>
static __global__ void advect_kernel(
    scalar_t* dens2d_n,
    const scalar_t* dens2d_c,
    const scalar_t* dens2d_dx,
    const scalar_t* dens2d_dy,
    const scalar_t* vel_c,
    float dt, float dx
) {

    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;

    scalar_t advect = get_v(vel_c, x, y).x * get(dens2d_dx, x, y) + get_v(vel_c, x, y).y * get(dens2d_dy, x, y);

    int loc = batch + y * gridDim.x + x;
    dens2d_n[loc] = get(dens2d_c, x, y) - dt * advect;
}

template <typename scalar_t>
static __global__ void velocity_non_advect_kernel(
    scalar_t* vel_n,
    const scalar_t* vel_c,
    const scalar_t* pres_dx,
    const scalar_t* pres_dy,
    float dt
){
    int batch0 = threadIdx.x * gridDim.x * gridDim.y * 2;
    int batch1 = threadIdx.x * gridDim.x * gridDim.y * 2 + gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;

    vector<scalar_t> vel = get_v(vel_c, x, y);

    if(blockIdx.z%2==0){
        int loc = batch0 + y * gridDim.x + x;
        vel_n[loc] = vel.x - get(pres_dx, x, y)*dt;
    }else{
        int loc = batch1 + y * gridDim.x + x;
        vel_n[loc] = vel.y - get(pres_dy, x, y)*dt;
    }
}

template <typename scalar_t>
static __global__ void pressure_update_kernel(
    scalar_t* pres_n,
    const scalar_t* pres_c,
    const scalar_t* vel_c,
    float dt, float dx
){

    int batch = threadIdx.x * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;

    int x_u = clamp_x(x+1);
    int x_d = clamp_x(x-1);
    int y_u = clamp_y(y+1);
    int y_d = clamp_y(y-1);

    vector<scalar_t> sub_x = diff_xv(vel_c, x, y, dx) * dx * 2;
    vector<scalar_t> sub_y = diff_yv(vel_c, x, y, dx) * dx * 2;

    scalar_t aver_p = 0.25 * (get(pres_c, x_d, y)+get(pres_c, x_u, y)+get(pres_c, x, y_d)+get(pres_c, x, y_u));

    scalar_t pred_p = ( \
        aver_p \
        + (sub_x.x*sub_x.x + sub_y.y*sub_y.y + (sub_y.x * sub_x.y)) / 8.0 \
        - dx * (sub_x.x + sub_y.y) / (8 * dt) \
    );

    int loc = batch + y * gridDim.x + x;
    pres_n[loc] = pred_p;
}

// C++ API

void update_gradient_op(
    torch::Tensor& df_dx,
    torch::Tensor& df_dy,
    const torch::Tensor& field,
    float dx
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = df_dx.size(0);
    int c = df_dx.size(1);
    int h = df_dx.size(2);
    int w = df_dx.size(3);
    dim3 block_size(h, w, 2);

    if(c == 1)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(df_dx.scalar_type(), "update_gradient_kernel", [&] {
            update_gradient_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                df_dx.data_ptr<scalar_t>(),
                df_dy.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                dx
            );
        });
    }else if(c == 2){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(df_dx.scalar_type(), "update_gradient2D_kernel", [&] {
            update_gradient2D_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                df_dx.data_ptr<scalar_t>(),
                df_dy.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                dx
            );
        });
    }
}

void update_laplacian_op(
    torch::Tensor& lapla,
    const torch::Tensor& field,
    float dx
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = field.size(0);
    int c = field.size(1);
    int h = field.size(2);
    int w = field.size(3);
    dim3 block_size(h, w, 1);

    if(c == 1)
    {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(field.scalar_type(), "update_laplacian_kernel", [&] {
            update_laplacian_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                lapla.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                dx
            );
        });
    }else if(c == 2){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(field.scalar_type(), "update_laplacian2D_kernel", [&] {
            update_laplacian2D_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                lapla.data_ptr<scalar_t>(),
                field.data_ptr<scalar_t>(),
                dx
            );
        });
    }
}

void update_vorticity_op(
    torch::Tensor& vort,
    const torch::Tensor& field,
    float dx
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = vort.size(0);
    int h = vort.size(2);
    int w = vort.size(3);
    dim3 block_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vort.scalar_type(), "update_vorticity_kernel", [&] {
        update_vorticity_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            vort.data_ptr<scalar_t>(),
            field.data_ptr<scalar_t>(),
            dx
        );
    });
}

void update_density_op(
    torch::Tensor& dens_n,
    torch::Tensor& dens_dx_n,
    torch::Tensor& dens_dy_n,
    const torch::Tensor& dens_c,
    const torch::Tensor& dens_dx,
    const torch::Tensor& dens_dy,
    const torch::Tensor& vel_c,
    float dt, float dx,
    int method
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = dens_c.size(0);
    int h = dens_c.size(2);
    int w = dens_c.size(3);
    dim3 block_size(h, w, 1);

    switch(method){
    case 0:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dens_c.scalar_type(), "cip_advect_kernel", [&] {
        cip_advect_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                dens_n.data_ptr<scalar_t>(),
                dens_dx_n.data_ptr<scalar_t>(),
                dens_dy_n.data_ptr<scalar_t>(),
                dens_c.data_ptr<scalar_t>(),
                dens_dx.data_ptr<scalar_t>(),
                dens_dy.data_ptr<scalar_t>(),
                vel_c.data_ptr<scalar_t>(),
                dt, dx
            );
        });
        break;
    case 1: //TODO
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dens_c.scalar_type(), "advect_kernel", [&] {
        advect_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                dens_n.data_ptr<scalar_t>(),
                dens_c.data_ptr<scalar_t>(),
                dens_dx.data_ptr<scalar_t>(),
                dens_dy.data_ptr<scalar_t>(),
                vel_c.data_ptr<scalar_t>(),
                dt, dx
            );
        });
        break;
    }
}

void update_velocity_non_advec_op(
    torch::Tensor& vel_n,
    const torch::Tensor& vel_c,
    const torch::Tensor& pres_dx,
    const torch::Tensor& pres_dy,
    float dt
){

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = vel_c.size(0);
    int h = vel_c.size(2);
    int w = vel_c.size(3);
    dim3 block_size(h, w, 2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vel_c.scalar_type(), "velocity_non_advect_kernel", [&] {
    velocity_non_advect_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            vel_n.data_ptr<scalar_t>(),
            vel_c.data_ptr<scalar_t>(),
            pres_dx.data_ptr<scalar_t>(),
            pres_dy.data_ptr<scalar_t>(),
            dt
        );
    });
}

void update_velocity_op(
    torch::Tensor& vel_n,
    torch::Tensor& dv_dx_n,
    torch::Tensor& dv_dy_n,
    const torch::Tensor& vel_c,
    const torch::Tensor& dv_dx_c,
    const torch::Tensor& dv_dy_c,
    float dt, float dx
){

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = vel_c.size(0);
    int h = vel_c.size(2);
    int w = vel_c.size(3);
    dim3 block_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vel_c.scalar_type(), "cip_advect_vec_kernel", [&] {
    cip_advect_vec_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            vel_n.data_ptr<scalar_t>(),
            dv_dx_n.data_ptr<scalar_t>(),
            dv_dy_n.data_ptr<scalar_t>(),

            vel_c.data_ptr<scalar_t>(),
            dv_dx_c.data_ptr<scalar_t>(),
            dv_dy_c.data_ptr<scalar_t>(),

            vel_c.data_ptr<scalar_t>(),
            dt, dx
        );
    });
}

void update_confinement_op(
    torch::Tensor& confinement,
    const torch::Tensor& vort,
    const torch::Tensor& dv_dx,
    const torch::Tensor& dv_dy
){

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = vort.size(0);
    int h = vort.size(2);
    int w = vort.size(3);
    dim3 block_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vort.scalar_type(), "update_vort_confinement_kernel", [&] {
    update_vort_confinement_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            confinement.data_ptr<scalar_t>(),
            vort.data_ptr<scalar_t>(),
            dv_dx.data_ptr<scalar_t>(),
            dv_dy.data_ptr<scalar_t>()
        );
    });
}

void update_pressure_op(
    torch::Tensor& pres_n,
    const torch::Tensor& pres_c,
    const torch::Tensor& vel_c,
    float dt, float dx
) {

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = pres_c.size(0);
    int h = pres_c.size(2);
    int w = pres_c.size(3);
    dim3 block_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pres_c.scalar_type(), "pressure_update_kernel", [&] {
    pressure_update_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            pres_n.data_ptr<scalar_t>(),
            pres_c.data_ptr<scalar_t>(),
            vel_c.data_ptr<scalar_t>(),
            dt, dx
        );
    });

}