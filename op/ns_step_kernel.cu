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

template <typename scalar_t>
struct vector{
    scalar_t x;
    scalar_t y;
};

template <typename scalar_t>
static __device__ scalar_t get(
    const scalar_t* field,
    int x, int y
){
    int batch = blockIdx.z * gridDim.x * gridDim.y;
    return field[batch + y * gridDim.x + x];
}

template <typename scalar_t>
static __device__ vector<scalar_t> get_v(
    const scalar_t* field,
    int x, int y
){
    int batch0 = blockIdx.z * gridDim.x * gridDim.y * 2;
    int batch1 = blockIdx.z * gridDim.x * gridDim.y * 2 + gridDim.x * gridDim.y;
    vector<scalar_t> vec{field[batch0 + y * gridDim.x + x], field[batch1 + y * gridDim.x + x]};
    return vec;
}

template <typename scalar_t>
static __device__ scalar_t diff_x(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (x == 0){
        return (get(field, x+1, y) - get(field, x  , y)) / dx;
    }else if(x == gridDim.x-1){
        return (get(field, x  , y) - get(field, x-1, y)) / dx;
    }else{
        return (get(field, x+1, y) - get(field, x-1, y)) / dx/2;
    }
}

template <typename scalar_t>
static __device__ scalar_t diff_y(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (y == 0){
        return (get(field, x, y+1) - get(field, x, y  )) / dx;
    }else if(y == gridDim.y-1){
        return (get(field, x, y  ) - get(field, x, y-1)) / dx;
    }else{
        return (get(field, x, y+1) - get(field, x, y-1)) / dx/2;
    }
}

static __device__ int clamp_x(int x){
    return x < 0 ? 0 : (x > gridDim.x-1 ? gridDim.x-1 : x);
}

static __device__ int clamp_y(int y){
    return y < 0 ? 0 : (y > gridDim.y-1 ? gridDim.y-1 : y);
}

template <typename scalar_t>
static __device__ int sign(scalar_t x){
    if(x<0.0){
        return -1;
    }else if(x>0.0){
        return 1;
    }else{
        return 0;
    }
}

template <typename scalar_t>
static __global__ void update_gradient_kernel(
    scalar_t* df_dx,
    scalar_t* df_dy,
    const scalar_t* field,
    float dx
) {
    int batch = blockIdx.z * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int loc = batch + y * gridDim.x + x;

    if(threadIdx.x%2==0)
        df_dx[loc] = diff_x(field, x, y, dx);
    else
        df_dy[loc] = diff_y(field, x, y, dx);
}

template <typename scalar_t>
static __global__ void cip_advect_kernel(
    scalar_t* dens_n,
    const scalar_t* dens_c,
    const scalar_t* dens_dx,
    const scalar_t* dens_dy,
    const scalar_t* vel_c,
    float dt, float dx
) {

    int batch = blockIdx.z * gridDim.x * gridDim.y;
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

}

template <typename scalar_t>
static __global__ void advect_kernel(
    scalar_t* dens_n,
    const scalar_t* dens_c,
    const scalar_t* dens_dx,
    const scalar_t* dens_dy,
    const scalar_t* vel_c,
    float dt, float dx
) {

    int batch = blockIdx.z * gridDim.x * gridDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y;

    scalar_t advect = get_v(vel_c, x, y).x * get(dens_dx, x, y) + get_v(vel_c, x, y).y * get(dens_dy, x, y);

    int loc = batch + y * gridDim.x + x;
    dens_n[loc] = get(dens_c, x, y) - dt * advect;
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
    int h = df_dx.size(2);
    int w = df_dx.size(3);
    dim3 block_size(h, w, b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(df_dx.scalar_type(), "update_gradient_kernel", [&] {
        update_gradient_kernel<scalar_t><<<block_size, 2, 0, stream>>>(
            df_dx.data_ptr<scalar_t>(),
            df_dy.data_ptr<scalar_t>(),
            field.data_ptr<scalar_t>(),
            dx
        );
    });
}

void update_density_op(
    torch::Tensor& dens_n,
    const torch::Tensor& dens_c,
    const torch::Tensor& dens_dx,
    const torch::Tensor& dens_dy,
    const torch::Tensor& vel_c,
    float dt, float dx, int method
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = dens_c.size(0);
    int h = dens_c.size(2);
    int w = dens_c.size(3);
    dim3 block_size(h, w, b);

    switch(method){
    case 0:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dens_c.scalar_type(), "cip_advect_kernel", [&] {
        cip_advect_kernel<scalar_t><<<block_size, 1, 0, stream>>>(
                dens_n.data_ptr<scalar_t>(),
                dens_c.data_ptr<scalar_t>(),
                dens_dx.data_ptr<scalar_t>(),
                dens_dy.data_ptr<scalar_t>(),
                vel_c.data_ptr<scalar_t>(),
                dt, dx
            );
        });
        break;
    case 1:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dens_c.scalar_type(), "advect_kernel", [&] {
        advect_kernel<scalar_t><<<block_size, 1, 0, stream>>>(
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


void update_velocity_op(torch::Tensor& vel_n, const torch::Tensor& vel_c, const torch::Tensor& pres_c, float dt, float dx){}
void update_pressure_op(torch::Tensor& pres_n, const torch::Tensor& pres_c, const torch::Tensor& vel_c, float dt, float dx){}