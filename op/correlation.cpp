#include <torch/extension.h>

void correlation_rearrange_op(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int stride
);
void correlation_update_op(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    torch::Tensor& output,
    const int stride
);
void correlation_grad_first_op(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& output,
    torch::Tensor& grad1,
    const int stride
);
void correlation_grad_second_op(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& output,
    torch::Tensor& grad2,
    const int stride
);


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor correlation_rearrange(const torch::Tensor& input, torch::Tensor& output, const int stride) {
    CHECK_CUDA(input);
    CHECK_CUDA(output);

    correlation_rearrange_op(input, output, stride);
    return output;
}

torch::Tensor correlation_update(const torch::Tensor& input1, const torch::Tensor& input2, torch::Tensor& output, const int stride){
    CHECK_CUDA(input1);
    CHECK_CUDA(input2);
    CHECK_CUDA(output);

    correlation_update_op(input1, input2, output, stride);
    return output;
}

torch::Tensor correlation_grad_first(const torch::Tensor& input1, const torch::Tensor& input2, torch::Tensor& output, torch::Tensor& grad1, const int stride){
    CHECK_CUDA(input1);
    CHECK_CUDA(input2);
    CHECK_CUDA(output);
    CHECK_CUDA(grad1);

    correlation_grad_first_op(input1, input2, output, grad1, stride);
    return grad1;
}

torch::Tensor correlation_grad_second(const torch::Tensor& input1, const torch::Tensor& input2, torch::Tensor& output, torch::Tensor& grad2, const int stride){
    CHECK_CUDA(input1);
    CHECK_CUDA(input2);
    CHECK_CUDA(output);
    CHECK_CUDA(grad2);

    correlation_grad_second_op(input1, input2, output, grad2, stride);
    return grad2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("correlation_rearrange", &correlation_rearrange, "correlation rearrangement");
    m.def("correlation_update", &correlation_update, "correlation evaluation");
    m.def("correlation_grad_first", &correlation_grad_first, "correlation backward for first tensor");
    m.def("correlation_grad_second", &correlation_grad_second, "correlation backward for second tensor");
}