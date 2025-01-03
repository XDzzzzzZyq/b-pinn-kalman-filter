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

    __device__ vector<scalar_t> operator+(const vector<scalar_t>& b) const {
        return vector<scalar_t>{x + b.x, y + b.y};
    }
    __device__ vector<scalar_t> operator-(const vector<scalar_t>& b) const {
        return vector<scalar_t>{x - b.x, y - b.y};
    }
    __device__ vector<scalar_t> operator*(const vector<scalar_t>& b) const {
        return vector<scalar_t>{x * b.x, y * b.y};
    }
    __device__ vector<scalar_t> operator*(const scalar_t& scalar) const {
        return vector<scalar_t>{x * scalar, y * scalar};
    }
    friend __device__ vector<scalar_t> operator*(const scalar_t& scalar, const vector<scalar_t>& v) {
        return vector<scalar_t>{v.x * scalar, v.y * scalar};
    }
    __device__ vector<scalar_t> operator/(const scalar_t& scalar) const {
        return vector<scalar_t>{x / scalar, y / scalar};
    }
    __device__ vector<scalar_t>& operator+=(const vector<scalar_t>& b) {
        x += b.x;
        y += b.y;
        return *this;
    }
    __device__ vector<scalar_t>& operator-=(const vector<scalar_t>& b) {
        x -= b.x;
        y -= b.y;
        return *this;
    }
    __device__ vector<scalar_t>& operator*=(const scalar_t& scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    __device__ vector<scalar_t>& operator/=(const scalar_t& scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }
    __device__ vector<scalar_t> clamp(float _min, float _max){
        scalar_t _x = min(max(x, _min), _max);
        scalar_t _y = min(max(y, _min), _max);
        return vector<scalar_t>{_x, _y};
    }
};

template <typename scalar_t>
static __device__ scalar_t get(
    const scalar_t* field,
    int x, int y
){
    int batch = threadIdx.x * gridDim.x * gridDim.y;
    return field[batch + y * gridDim.x + x];
}

template <typename scalar_t>
static __device__ vector<scalar_t> get_v(
    const scalar_t* field,
    int x, int y
){
    int batch0 = threadIdx.x * gridDim.x * gridDim.y * 2;
    int batch1 = threadIdx.x * gridDim.x * gridDim.y * 2 + gridDim.x * gridDim.y;
    vector<scalar_t> vec{field[batch0 + y * gridDim.x + x], field[batch1 + y * gridDim.x + x]};
    return vec;
}

template <typename scalar_t>
static __device__ vector<scalar_t> set_v(
    scalar_t* field,
    const vector<scalar_t>& vec,
    int x, int y
){
    int batch0 = threadIdx.x * gridDim.x * gridDim.y * 2;
    int batch1 = threadIdx.x * gridDim.x * gridDim.y * 2 + gridDim.x * gridDim.y;
    field[batch0 + y * gridDim.x + x] = vec.x;
    field[batch1 + y * gridDim.x + x] = vec.y;
}

template <typename scalar_t>
static __device__ scalar_t diff_x(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (x == 0){
        return (get(field, x+1, y) - get(field, x  , y)) / dx;
    }else if(x == gridDim.y-1){
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
    }else if(y == gridDim.x-1){
        return (get(field, x, y  ) - get(field, x, y-1)) / dx;
    }else{
        return (get(field, x, y+1) - get(field, x, y-1)) / dx/2;
    }
}

template <typename scalar_t>
static __device__ vector<scalar_t> diff_xv(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (x == 0){
        return (get_v(field, x+1, y) - get_v(field, x  , y)) / dx;
    }else if(x == gridDim.y-1){
        return (get_v(field, x  , y) - get_v(field, x-1, y)) / dx;
    }else{
        return (get_v(field, x+1, y) - get_v(field, x-1, y)) / dx/2;
    }
}

template <typename scalar_t>
static __device__ vector<scalar_t> diff_yv(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (y == 0){
        return (get_v(field, x, y+1) - get_v(field, x, y  )) / dx;
    }else if(y == gridDim.x-1){
        return (get_v(field, x, y  ) - get_v(field, x, y-1)) / dx;
    }else{
        return (get_v(field, x, y+1) - get_v(field, x, y-1)) / dx/2;
    }
}

static __device__ int clamp_x(int x){
    return x < 0 ? -x : (x > gridDim.y-1 ? 2*gridDim.y-2-x : x);
}

static __device__ int clamp_y(int y){
    return y < 0 ? -y : (y > gridDim.x-1 ? 2*gridDim.x-2-y : y);
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

    int x_s = sign(get_v(vel_c, x, y).x);
    int y_s = sign(get_v(vel_c, x, y).y);
    int x_m = clamp_x(x - x_s);
    int y_m = clamp_y(y - y_s);

    scalar_t tmp1 = get(dens2d_c, x, y) - get(dens2d_c, x, y_m) - get(dens2d_c, x_m, y) + get(dens2d_c, x_m, y_m);
    scalar_t tmp2 = get(dens2d_c, x_m, y) - get(dens2d_c, x, y);
    scalar_t tmp3 = get(dens2d_c, x, y_m) - get(dens2d_c, x, y);

    scalar_t x_s_denom = x_s * dx*dx*dx;
    scalar_t y_s_denom = y_s * dx*dx*dx;

    scalar_t a = (x_s * (get(dens2d_dx, x_m, y) + get(dens2d_dx, x, y)) * dx - 2.0 * (-tmp2)) / x_s_denom;
    scalar_t b = (y_s * (get(dens2d_dy, x, y_m) + get(dens2d_dy, x, y)) * dx - 2.0 * (-tmp3)) / y_s_denom;
    scalar_t c = (-tmp1 - x_s * (get(dens2d_dx, x, y_m) - get(dens2d_dx, x, y)) * dx) / y_s_denom;
    scalar_t d = (-tmp1 - y_s * (get(dens2d_dy, x_m, y) - get(dens2d_dy, x, y)) * dx) / x_s_denom;
    scalar_t e = (3.0 * tmp2 + x_s * (get(dens2d_dx, x_m, y) + 2.0 * get(dens2d_dx, x, y)) * dx) / dx/dx;
    scalar_t f = (3.0 * tmp3 + y_s * (get(dens2d_dy, x, y_m) + 2.0 * get(dens2d_dy, x, y)) * dx) / dx/dx;
    scalar_t g = (-(get(dens2d_dy, x_m, y) - get(dens2d_dy, x, y)) + c * dx*dx) / (x_s * dx);

    scalar_t X = -get_v(vel_c, x, y).x * dt;
    scalar_t Y = -get_v(vel_c, x, y).y * dt;

    int loc = batch + y * gridDim.x + x;
    dens2d_n[loc] = ( \
            ((a * X + c * Y + e) * X + g * Y + get(dens2d_dx, x, y)) * X \
            + ((b * Y + d * X + f) * Y + get(dens2d_dy, x, y)) * Y \
            + get(dens2d_c, x, y) \
    );
}

template <typename scalar_t>
static __global__ void cip_advect_vec_kernel(
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

    scalar_t X = -get_v(vel_c, x, y).x * dt;
    scalar_t Y = -get_v(vel_c, x, y).y * dt;

    int loc = batch + y * gridDim.x + x;
    vector<scalar_t> New = ( \
            ((a * X + c * Y + e) * X + g * Y + get_v(dens2d_dx, x, y)) * X \
            + ((b * Y + d * X + f) * Y + get_v(dens2d_dy, x, y)) * Y \
            + get_v(dens2d_c, x, y) \
    );

    set_v(dens2d_n, New, x, y);
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
static __global__ void velocity_update_kernel(
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

    vector<scalar_t> sub_x = get_v(vel_c, x_u, y) - get_v(vel_c, x_d, y);
    vector<scalar_t> sub_y = get_v(vel_c, x, y_u) - get_v(vel_c, x, y_d);

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
    int h = df_dx.size(2);
    int w = df_dx.size(3);
    dim3 block_size(h, w, 2);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(df_dx.scalar_type(), "update_gradient_kernel", [&] {
        update_gradient_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            df_dx.data_ptr<scalar_t>(),
            df_dy.data_ptr<scalar_t>(),
            field.data_ptr<scalar_t>(),
            dx
        );
    });
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
    torch::Tensor& dens2d_n,
    const torch::Tensor& dens2d_c,
    const torch::Tensor& dens2d_dx,
    const torch::Tensor& dens2d_dy,
    const torch::Tensor& vel_c,
    float dt, float dx, int method
) {
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = dens2d_c.size(0);
    int h = dens2d_c.size(2);
    int w = dens2d_c.size(3);
    dim3 block_size(h, w, 1);

    switch(method){
    case 0:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dens2d_c.scalar_type(), "cip_advect_kernel", [&] {
        cip_advect_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                dens2d_n.data_ptr<scalar_t>(),
                dens2d_c.data_ptr<scalar_t>(),
                dens2d_dx.data_ptr<scalar_t>(),
                dens2d_dy.data_ptr<scalar_t>(),
                vel_c.data_ptr<scalar_t>(),
                dt, dx
            );
        });
        break;
    case 1:
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dens2d_c.scalar_type(), "advect_kernel", [&] {
        advect_kernel<scalar_t><<<block_size, b, 0, stream>>>(
                dens2d_n.data_ptr<scalar_t>(),
                dens2d_c.data_ptr<scalar_t>(),
                dens2d_dx.data_ptr<scalar_t>(),
                dens2d_dy.data_ptr<scalar_t>(),
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vel_c.scalar_type(), "velocity_update_kernel", [&] {
    velocity_update_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            vel_n.data_ptr<scalar_t>(),
            vel_c.data_ptr<scalar_t>(),
            pres_dx.data_ptr<scalar_t>(),
            pres_dy.data_ptr<scalar_t>(),
            dt
        );
    });
}

void update_velocity_op(
    torch::Tensor& u_n,
    const torch::Tensor& u_c,
    const torch::Tensor& u_dx,
    const torch::Tensor& u_dy,
    float dt, float dx
){

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    int b = u_c.size(0);
    int h = u_c.size(2);
    int w = u_c.size(3);
    dim3 block_size(h, w, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u_c.scalar_type(), "cip_advect_vec_kernel", [&] {
    cip_advect_vec_kernel<scalar_t><<<block_size, b, 0, stream>>>(
            u_n.data_ptr<scalar_t>(),
            u_c.data_ptr<scalar_t>(),
            u_dx.data_ptr<scalar_t>(),
            u_dy.data_ptr<scalar_t>(),
            u_c.data_ptr<scalar_t>(),
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
){

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