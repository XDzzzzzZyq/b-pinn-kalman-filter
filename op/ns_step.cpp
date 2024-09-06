#include <torch/extension.h>
#include <vector>

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
    float dt, float dx,
    int method
);
void update_velocity_op(
    torch::Tensor& u_n,
    const torch::Tensor& u_c,
    const torch::Tensor& u_dx,
    const torch::Tensor& u_dy,
    const torch::Tensor& vel_c,
    const torch::Tensor& pres_dx,
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

    std::vector<torch::Tensor> vel_split = torch::unbind(vel_c, 1);
    torch::Tensor u = torch::unsqueeze(vel_split[0], 1);
    torch::Tensor v = torch::unsqueeze(vel_split[1], 1);

    torch::Tensor dp_dx = torch::empty_like(u);
    torch::Tensor dp_dy = torch::empty_like(u);
    update_gradient_op(dp_dx, dp_dy, pres_c, dx);

    // u update

    torch::Tensor du_dx = torch::empty_like(u);
    torch::Tensor du_dy = torch::empty_like(u);
    update_gradient_op(du_dx, du_dy, u, dx);

    torch::Tensor u_n = torch::empty_like(u);
    update_velocity_op(u_n, u, du_dx, du_dy, vel_c, dp_dx, dt, dx);

    // v update

    torch::Tensor dv_dx = torch::empty_like(v);
    torch::Tensor dv_dy = torch::empty_like(v);
    update_gradient_op(dv_dx, dv_dy, v, dx);

    torch::Tensor v_n = torch::empty_like(v);
    update_velocity_op(v_n, v, dv_dx, dv_dy, vel_c, dp_dy, dt, dx);

    torch::Tensor vel_n = torch::stack(std::vector<torch::Tensor>{u, v}, 1);

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