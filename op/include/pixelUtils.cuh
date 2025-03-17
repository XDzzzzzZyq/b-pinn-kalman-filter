#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

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

template <typename scalar_t>
static __device__ scalar_t diff2_x(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (x == 0){
        return (get(field, x+2, y) - 2*get(field, x+1, y) + get(field, x, y)) / dx/dx;
    }else if(x == gridDim.y-1){
        return (get(field, x, y) - 2*get(field, x-1, y) + get(field, x-2, y)) / dx/dx;
    }else{
        return (get(field, x+1, y) - 2*get(field, x, y) + get(field, x-1, y)) / dx/dx;
    }
}

template <typename scalar_t>
static __device__ scalar_t diff2_y(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (y == 0){
        return (get(field, x, y+2) - 2*get(field, x, y+1) + get(field, x, y)) / dx/dx;
    }else if(y == gridDim.x-1){
        return (get(field, x, y) - 2*get(field, x, y-1) + get(field, x, y-2)) / dx/dx;
    }else{
        return (get(field, x, y+1) - 2*get(field, x, y) + get(field, x, y-1)) / dx/dx;
    }
}

template <typename scalar_t>
static __device__ vector<scalar_t> diff2_xv(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (x == 0){
        return (get_v(field, x+2, y) - 2*get_v(field, x+1, y) + get_v(field, x, y)) / dx/dx;
    }else if(x == gridDim.y-1){
        return (get_v(field, x, y) - 2*get_v(field, x-1, y) + get_v(field, x-2, y)) / dx/dx;
    }else{
        return (get_v(field, x+1, y) - 2*get_v(field, x, y) + get_v(field, x-1, y)) / dx/dx;
    }
}

template <typename scalar_t>
static __device__ vector<scalar_t> diff2_yv(
    const scalar_t* field,
    int x, int y, float dx
) {
    if (y == 0){
        return (get_v(field, x, y+2) - 2*get_v(field, x, y+1) + get_v(field, x, y)) / dx/dx;
    }else if(y == gridDim.x-1){
        return (get_v(field, x, y) - 2*get_v(field, x, y-1) + get_v(field, x, y-2)) / dx/dx;
    }else{
        return (get_v(field, x, y+1) - 2*get_v(field, x, y) + get_v(field, x, y-1)) / dx/dx;
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
