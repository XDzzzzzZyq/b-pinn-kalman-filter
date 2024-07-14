import torch
import math
import re
import os

from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
correlation_op = load(
    "correlation",
    sources=[
        os.path.join(module_path, "correlation.cpp"),
        os.path.join(module_path, "correlation_kernel.cu"),
    ],
)

class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second, stride):
        rbot0 = first.new_zeros([
            first.size(0),
            first.size(2) + (6 * stride),
            first.size(3) + (6 * stride),
            first.size(1)
        ])
        rbot1 = first.new_zeros([
            first.size(0),
            first.size(2) + (6 * stride),
            first.size(3) + (6 * stride),
            first.size(1)
        ])

        self.save_for_backward(first, second, rbot0, rbot1)
        self.stride = stride

        assert (first.is_contiguous() == True)
        assert (second.is_contiguous() == True)

        output = first.new_zeros([
            first.size(0), 49,
            int(math.ceil(first.size(2) / stride)),
            int(math.ceil(first.size(3) / stride))
        ])

        if first.is_cuda:
            rbot0 = correlation_op.correlation_rearrange(first, rbot0, stride)
            rbot1 = correlation_op.correlation_rearrange(second, rbot1, stride)

            output = correlation_op.correlation_update(rbot0, rbot1, output, stride)

        else:
            raise NotImplementedError()

        return output

    @staticmethod
    def backward(self, grad):
        first, second, rbot0, rbot1, stride = self.saved_tensors
        stride = self.stride

        assert (grad.is_contiguous() == True)

        gradFirst = first.new_zeros(
            [first.size(0),
             first.size(1),
             first.size(2),
             first.size(3)]) if self.needs_input_grad[0] == True else None
        gradSecond = first.new_zeros(
            [first.size(0),
             first.size(1),
             first.size(2),
             first.size(3)]) if self.needs_input_grad[1] == True else None

        if first.is_cuda:
            if gradFirst is not None:
                gradFirst = correlation_op.correlation_grad_first(rbot0, rbot1, grad, gradFirst, stride)

            if gradSecond is not None:
                gradSecond = correlation_op.correlation_grad_second(rbot0, rbot1, grad, gradSecond, stride)
        else:
            raise NotImplementedError()


        return gradFirst, gradSecond, None

def FunctionCorrelation(tensorFirst, tensorSecond, stride):
    return _FunctionCorrelation.apply(tensorFirst, tensorSecond, stride)


class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tensorFirst, tensorSecond, stride):
        return _FunctionCorrelation.apply(tensorFirst, tensorSecond, stride)
