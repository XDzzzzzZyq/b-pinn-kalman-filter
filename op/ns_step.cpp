#include <torch/extension.h>
#include <vector>

void update_gradient_op(
    torch::Tensor& df_dx,
    torch::Tensor& df_dy,
    const torch::Tensor& field,
    float dx
);
void update_laplacian_op(
    torch::Tensor& lapla,
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
    torch::Tensor& dens_dx_n,
    torch::Tensor& dens_dy_n,
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
    torch::Tensor& vel_n,
    torch::Tensor& dv_dx_n,
    torch::Tensor& dv_dy_n,
    const torch::Tensor& vel_c,
    const torch::Tensor& dv_dx_c,
    const torch::Tensor& dv_dy_c,
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

std::tuple<torch::Tensor, torch::Tensor> diff(const torch::Tensor& dens_c, float dx) {
    CHECK_CUDA(dens_c);

    torch::Tensor df_dx = torch::empty_like(dens_c);
    torch::Tensor df_dy = torch::empty_like(dens_c);
    update_gradient_op(df_dx, df_dy, dens_c, dx);

    return {df_dx, df_dy};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> update_density(
    const torch::Tensor& dens_c,
    const torch::Tensor df_dx,
    const torch::Tensor df_dy,
    const torch::Tensor& vel_c,
    float dt, float dx
) {
    CHECK_CUDA(dens_c);
    CHECK_CUDA(df_dx);
    CHECK_CUDA(df_dy);
    CHECK_CUDA(vel_c);


    // Non advect
    torch::Tensor lapla = torch::empty_like(dens_c);
    update_laplacian_op(lapla, dens_c, dx);
    torch::Tensor dens_n = dens_c + lapla / 10000000.0 * dt;

    // Non advection gradient
    torch::Tensor _df_dx_c = torch::empty_like(dens_c);
    torch::Tensor _df_dy_c = torch::empty_like(dens_c);
    torch::Tensor _df_dx_n = torch::empty_like(dens_n);
    torch::Tensor _df_dy_n = torch::empty_like(dens_n);
    update_gradient_op(_df_dx_c, _df_dy_c, dens_c, dx);
    update_gradient_op(_df_dx_n, _df_dy_n, dens_n, dx);

    torch::Tensor df_dx_n = df_dx + (_df_dx_n - _df_dx_c);
    torch::Tensor df_dy_n = df_dy + (_df_dy_n - _df_dy_c);

    // Advect
    torch::Tensor dens_a = torch::empty_like(dens_n);
    torch::Tensor df_dx_a = torch::empty_like(df_dx);
    torch::Tensor df_dy_a = torch::empty_like(df_dy);
    update_density_op(dens_a, df_dx_a, df_dy_a,
                      dens_n, df_dx_n, df_dy_n,
                      vel_c,
                      dt, dx, 0);

    return {dens_n, df_dx_n, df_dy_n};
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> update_velocity(
    const torch::Tensor& vel_c,
    const torch::Tensor& dv_dx,
    const torch::Tensor& dv_dy,
    const torch::Tensor& pres_c,
    float dt, float dx)
{
    CHECK_CUDA(vel_c);
    CHECK_CUDA(pres_c);

    // Pressure gradient
    torch::Tensor dp_dx = torch::empty_like(pres_c);
    torch::Tensor dp_dy = torch::empty_like(pres_c);
    update_gradient_op(dp_dx, dp_dy, pres_c, dx);

    // Non advection
    torch::Tensor vel_n = torch::empty_like(vel_c);
    update_velocity_non_advec_op(vel_n, vel_c, dp_dx, dp_dy, dt);

    torch::Tensor lapla = torch::empty_like(vel_c);
    update_laplacian_op(lapla, vel_c, dx);
    vel_n = vel_n + lapla / 10000000.0 * dt;

    // Non advection gradient
    torch::Tensor _dv_dx_c = torch::empty_like(vel_c);
    torch::Tensor _dv_dy_c = torch::empty_like(vel_c);
    torch::Tensor _dv_dx_n = torch::empty_like(vel_n);
    torch::Tensor _dv_dy_n = torch::empty_like(vel_n);
    update_gradient_op(_dv_dx_c, _dv_dy_c, vel_c, dx);
    update_gradient_op(_dv_dx_n, _dv_dy_n, vel_n, dx);

    torch::Tensor dv_dx_n = dv_dx + (_dv_dx_n - _dv_dx_c);
    torch::Tensor dv_dy_n = dv_dy + (_dv_dy_n - _dv_dy_c);

    // Advection
    torch::Tensor vel_a = torch::empty_like(vel_n);
    torch::Tensor dv_dx_a = torch::empty_like(vel_n);
    torch::Tensor dv_dy_a = torch::empty_like(vel_n);
    update_velocity_op(vel_a, dv_dx_a, dv_dy_a,
                       vel_n, dv_dx_n, dv_dy_n,
                       dt, dx);

    return {vel_a, dv_dx_a, dv_dy_a};
}

torch::Tensor update_pressure(
    const torch::Tensor& pres_c,
    const torch::Tensor& vel_c,
    float dt, float dx)
{
    CHECK_CUDA(vel_c);
    CHECK_CUDA(pres_c);

    torch::Tensor pres_n = torch::empty_like(pres_c);
    update_pressure_op(pres_n, pres_c, vel_c, dt, dx);

    return pres_n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diff", &diff, "Spatial differentiation");
    m.def("update_density", &update_density, "Update density field");
    m.def("update_velocity", &update_velocity, "Update velocity field");
    m.def("update_pressure", &update_pressure, "Update pressure field");
    m.def("calc_vort_confinement", &calc_vort_confinement, "Calculate Vorticity Confinement");
}