#include <torch/extension.h>
#include <vector>

void update_gradient_op(
    torch::Tensor& df_dx,
    torch::Tensor& df_dy,
    const torch::Tensor& field,
    float dx
);
void update_vorticity_op(
    torch::Tensor& vort,
    const torch::Tensor& field,
    float dx
);
void update_confinement_op(
    torch::Tensor& confinement,
    const torch::Tensor& vort,
    const torch::Tensor& dv_dx,
    const torch::Tensor& dv_dy
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
void update_velocity_non_advec_op(
    torch::Tensor& vel_n,
    const torch::Tensor& vel_c,
    const torch::Tensor& pres_dx,
    const torch::Tensor& pres_dy,
    float dt
);
void update_velocity_op(
    torch::Tensor& u_n,
    const torch::Tensor& u_c,
    const torch::Tensor& du_dx,
    const torch::Tensor& du_dy,
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

torch::Tensor calc_vort_confinement(const torch::Tensor& vel_c, float dx) {
    CHECK_CUDA(vel_c);

    int B = vel_c.size(0);
    int H = vel_c.size(2);
    int W = vel_c.size(3);
    torch::Tensor vort = torch::empty({B, 1, H, W}).to(vel_c.device());
    update_vorticity_op(vort, vel_c, dx);

    torch::Tensor dv_dx = torch::empty_like(vort);
    torch::Tensor dv_dy = torch::empty_like(vort);
    update_gradient_op(dv_dx, dv_dy, torch::abs(vort), dx);

    torch::Tensor confinement = torch::empty_like(vel_c);
    update_confinement_op(confinement, vort, dv_dx, dv_dy);

    return confinement;
}

torch::Tensor update_velocity(const torch::Tensor& vel_c, const torch::Tensor& pres_c, float dt, float dx) {
    CHECK_CUDA(vel_c);
    CHECK_CUDA(pres_c);

    torch::Tensor dp_dx = torch::empty_like(pres_c);
    torch::Tensor dp_dy = torch::empty_like(pres_c);
    update_gradient_op(dp_dx, dp_dy, pres_c, dx);

    torch::Tensor vel_n = torch::empty_like(vel_c);
    update_velocity_non_advec_op(vel_n, vel_c, dp_dx, dp_dy, dt);

    std::vector<torch::Tensor> vel_split = torch::unbind(vel_n, 1);
    torch::Tensor u = torch::unsqueeze(vel_split[0], 1);
    torch::Tensor v = torch::unsqueeze(vel_split[1], 1);

    torch::Tensor du_dx = torch::empty_like(u);
    torch::Tensor du_dy = torch::empty_like(u);
    update_gradient_op(du_dx, du_dy, u, dx);

    torch::Tensor dv_dx = torch::empty_like(v);
    torch::Tensor dv_dy = torch::empty_like(v);
    update_gradient_op(dv_dx, dv_dy, v, dx);

    torch::Tensor vel_a = torch::empty_like(vel_n);
    update_velocity_op(vel_a,
                       vel_n,
                       torch::cat(std::vector<torch::Tensor>{du_dx, dv_dx}, 1),
                       torch::cat(std::vector<torch::Tensor>{du_dy, dv_dy}, 1),
                       dt, dx);

    return vel_a;
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
    m.def("calc_vort_confinement", &calc_vort_confinement, "Calculate Vorticity Confinement");
}