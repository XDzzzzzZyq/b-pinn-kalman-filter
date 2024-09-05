#include <torch/extension.h>

void update_gradient_op(
    torch::Tensor& df_dx,
    torch::Tensor& df_dy,
    const torch::Tensor& field,
    float dx
);
void update_density_op(
    torch::Tensor& dens_n,
    const torch::Tensor& dens_c,
    const torch::Tensor& dens_dx,
    const torch::Tensor& dens_dy,
    const torch::Tensor& vel_c,
    float dt,
    float dx,
    int method
);
void update_velocity_op(
    torch::Tensor& vel_n,
    const torch::Tensor& vel_c,
    const torch::Tensor& pres_c,
    float dt, float dx
);
void update_pressure_op(
    torch::Tensor& pres_n,
    const torch::Tensor& pres_c,
    const torch::Tensor& vel_c,
    float dt, float dx
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor update_density(const torch::Tensor& dens_c, const torch::Tensor& vel_c, float dt, float dx) {
    CHECK_CUDA(dens_c);
    CHECK_CUDA(vel_c);

    torch::Tensor df_dx = torch::empty_like(dens_c);
    torch::Tensor df_dy = torch::empty_like(dens_c);
    update_gradient_op(df_dx, df_dy, dens_c, dx);

    torch::Tensor dens_n = torch::empty_like(dens_c);
    update_density_op(dens_n, dens_c, df_dx, df_dy, vel_c, dt, dx, 0);

    return dens_n;
}

torch::Tensor update_velocity(const torch::Tensor& vel_c, const torch::Tensor& pres_c, float dt, float dx) {
    CHECK_CUDA(vel_c);
    CHECK_CUDA(pres_c);

    torch::Tensor vel_n = torch::empty_like(vel_c);
    update_velocity_op(vel_n, vel_c, pres_c, dt, dx);

    return vel_n;
}

torch::Tensor update_pressure(const torch::Tensor& pres_c, const torch::Tensor& vel_c, float dt, float dx) {
    CHECK_CUDA(vel_c);
    CHECK_CUDA(pres_c);

    torch::Tensor pres_n = torch::empty_like(pres_c);
    update_pressure_op(pres_n, pres_c, vel_c, dt, dx);

    return pres_n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update_density", &update_density, "Update density field");
    m.def("update_velocity", &update_velocity, "Update velocity field");
    m.def("update_pressure", &update_pressure, "Update pressure field");
}