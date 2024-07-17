#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

struct shape_info{
    int b, c, w, h;
};

shape_info GetShape(const torch::Tensor& a){
    shape_info info;
    info.b = a.size(0);
    info.c = a.size(1);
    info.w = a.size(2);
    info.h = a.size(3);

    return info;
}

template <typename scalar_t>
static __global__ void kernel_Correlation_rearrange(
		const int n,
		const int stride,
		const scalar_t* input,
		scalar_t* output,
		shape_info size_input,
		shape_info size_output
) {
	  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	  if (index >= n) {
	    return;
	  }
	  int sample = blockIdx.z;
	  int num_channel = blockIdx.y;
	  scalar_t dblValue = input[(((sample * size_input.c) + num_channel) * size_input.h * size_input.w) + index];
	  __syncthreads();
	  int padded_y = (index / size_input.w) + 3*stride;
	  int padded_x = (index % size_input.w) + 3*stride;
	  int rearrange = ((size_input.w + 6*stride) * padded_y) + padded_x;
	  output[(((sample * size_output.c * size_output.h) + rearrange) * size_input.c) + num_channel] = dblValue;
	}

template <typename scalar_t>
static __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const int stride,
	  const scalar_t* rbot0,
	  const scalar_t* rbot1,
	  scalar_t* top,
	  shape_info size_rbot0,
	  shape_info size_top
) {
	  extern __shared__ char patch_data_char[];

	  scalar_t *patch_data = (scalar_t *)patch_data_char;

	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = (blockIdx.x + 3) * stride;
	  int y1 = (blockIdx.y + 3) * stride;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;

	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * size_rbot0.w;
	      for (int ch = ch_off; ch < size_rbot0.w; ch += 32) { // CHANNELS
	        int idx1 = ((item * size_rbot0.c + y1+j) * size_rbot0.h + x1+i) * size_rbot0.w + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }

	  __syncthreads();

	  __shared__ scalar_t sum[32];

	  // Compute correlation
	  for (int top_channel = 0; top_channel < size_top.c; top_channel++) {
	    sum[ch_off] = 0;

	    int s2o = (top_channel % 7 - 3) * stride;
	    int s2p = (top_channel / 7 - 3) * stride;

	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * size_rbot0.w;
	        for (int ch = ch_off; ch < size_rbot0.w; ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;

	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * size_rbot0.c + y2+j) * size_rbot0.h + x2+i) * size_rbot0.w + ch;

	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }

	    __syncthreads();

	    if (ch_off == 0) {
	      scalar_t total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = size_rbot0.w;
	      const int index = ((top_channel*size_top.h + blockIdx.y)*size_top.w)+blockIdx.x;
	      top[index + item*size_top.c*size_top.h*size_top.w] = total_sum / (scalar_t)sumelems;
	    }
	  }
	}


#define ROUND_OFF 50000
template <typename scalar_t>
static __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int stride,
	  const int sample,
	  const scalar_t* rbot0,
	  const scalar_t* rbot1,
	  const scalar_t* gradOutput,
	  scalar_t* gradFirst,
	  shape_info size_rbot0,
	  shape_info size_out,
	  shape_info size_first
) {
	for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
	  int n = index % size_first.c; // channels
	  int l = (index / size_first.c) % size_first.w + 3*stride; // w-pos
	  int m = (index / size_first.c / size_first.w) % size_first.h + 3*stride; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = stride * round_off;

	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 3*stride + round_off_s1 - 1) / stride + 1 - round_off; // ceil (l - 3*stride) / stride
	  int ymin = (m - 3*stride + round_off_s1 - 1) / stride + 1 - round_off; // ceil (l - 3*stride) / stride

	  // Same here:
	  int xmax = (l - 3*stride + round_off_s1) / stride - round_off; // floor (l - 3*stride) / stride
	  int ymax = (m - 3*stride + round_off_s1) / stride - round_off; // floor (m - 3*stride) / stride

	  scalar_t sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=size_out.w-1) && (ymin<=size_out.h-1)) {
	    xmin = fmaxf(0,xmin);
	    xmax = fminf(size_out.w-1,xmax);

	    ymin = fmaxf(0,ymin);
	    ymax = fminf(size_out.h-1,ymax);

	    for (int p = -3; p <= 3; p++) {
	      for (int o = -3; o <= 3; o++) {
	        // Get rbot1 data:
	        int s2o = stride * o;
	        int s2p = stride * p;
	        int idxbot1 = ((sample * size_rbot0.c + (m+s2p)) * size_rbot0.h + (l+s2o)) * size_rbot0.w + n;
	        scalar_t bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (sample * size_out.c + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * size_out.h + y) * size_out.w + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = size_first.c;
	  const int bot0index = ((n * size_first.h) + (m-3*stride)) * size_first.w + (l-3*stride);
	  gradFirst[bot0index + sample * size_first.c * size_first.h * size_first.w] = sum / (scalar_t)sumelems;
	} }

#define ROUND_OFF 50000
template <typename scalar_t>
static __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int stride,
	  const int sample,
	  const scalar_t* rbot0,
	  const scalar_t* rbot1,
	  const scalar_t* gradOutput,
	  scalar_t* gradSecond,
	  shape_info size_rbot0,
	  shape_info size_out,
	  shape_info size_second
) {
	  for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
	  int n = index % size_second.c; // channels
	  int l = (index / size_second.c) % size_second.w + 3*stride; // w-pos
	  int m = (index / size_second.c / size_second.w) % size_second.h + 3*stride; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = stride * round_off;

	  scalar_t sum = 0;
	  for (int p = -3; p <= 3; p++) {
	    for (int o = -3; o <= 3; o++) {
	      int s2o = stride * o;
	      int s2p = stride * p;

	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 3*stride - s2o + round_off_s1 - 1) / stride + 1 - round_off; // ceil (l - 3*stride - s2o) / stride
	      int ymin = (m - 3*stride - s2p + round_off_s1 - 1) / stride + 1 - round_off; // ceil (l - 3*stride - s2o) / stride

	      // Same here:
	      int xmax = (l - 3*stride - s2o + round_off_s1) / stride - round_off; // floor (l - 3*stride - s2o) / stride
	      int ymax = (m - 3*stride - s2p + round_off_s1) / stride - round_off; // floor (m - 3*stride - s2p) / stride

	      if (xmax>=0 && ymax>=0 && (xmin<=size_out.w-1) && (ymin<=size_out.h-1)) {
	        xmin = fmaxf(0,xmin);
	        xmax = fminf(size_out.w-1,xmax);

	        ymin = fmaxf(0,ymin);
	        ymax = fminf(size_out.h-1,ymax);

	        // Get rbot0 data:
	        int idxbot0 = ((sample * size_rbot0.c + (m-s2p)) * size_rbot0.h + (l-s2o)) * size_rbot0.w + n;
	        scalar_t bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (sample * size_out.c + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * size_out.h + y) * size_out.w + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = size_second.c;
	  const int bot1index = ((n * size_second.h) + (m-3*stride)) * size_second.w + (l-3*stride);
	  gradSecond[bot1index + sample * size_second.c * size_second.h * size_second.w] = sum / (scalar_t)sumelems;
	} }



// Torch Operators

void correlation_rearrange_op(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int stride
) {

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    shape_info size_input = GetShape(input);
    shape_info size_output = GetShape(output);
    int n = size_input.h * size_input.w;

    dim3 grid_size((n+16-1)/16, size_input.c, size_input.b);
    dim3 block_size(16, 1, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "kernel_Correlation_rearrange", [&] {
        kernel_Correlation_rearrange<scalar_t><<<grid_size, block_size, 0, stream>>>(
            n, stride,
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size_input,
            size_output
        );
    });
}


void correlation_update_op(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    torch::Tensor& output,
    const int stride
){

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    shape_info size_input = GetShape(input1);
    shape_info size_output = GetShape(output);
    int n = size_output.c * size_output.h * size_output.w;

    dim3 grid_size(size_output.w, size_output.h, size_output.b);
    dim3 block_size(32, 1, 1);
    int shared_size = size_input.c * 4;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "kernel_Correlation_updateOutput", [&] {
        kernel_Correlation_updateOutput<scalar_t><<<grid_size, block_size, shared_size, stream>>>(
            n, stride,
            input1.data_ptr<scalar_t>(),
            input2.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size_input,
            size_output
        );
    });
}


void correlation_grad_first_op(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& output,
    torch::Tensor& grad1,
    const int stride
) {

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    shape_info size_input = GetShape(input1);
    shape_info size_output = GetShape(output);
    shape_info size_grad = GetShape(grad1);
    int n = size_input.c * size_input.h * size_input.w;

    dim3 grid_size((n + 512 - 1)/512, 1, 1);
    dim3 block_size(512, 1, 1);

    for(int i = 0; i < size_input.b; i++){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "kernel_Correlation_updateGradFirst", [&] {
            kernel_Correlation_updateGradFirst<scalar_t><<<grid_size, block_size, 0, stream>>>(
                n, stride, i,
                input1.data_ptr<scalar_t>(),
                input2.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                grad1.data_ptr<scalar_t>(),
                size_input,
                size_output,
                size_grad
            );
        });
    }
}


void correlation_grad_second_op(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& output,
    torch::Tensor& grad2,
    const int stride
) {

    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);

    shape_info size_input = GetShape(input2);
    shape_info size_output = GetShape(output);
    shape_info size_grad = GetShape(grad2);
    int n = size_input.c * size_input.h * size_input.w;

    dim3 grid_size((n + 512 - 1)/512, 1, 1);
    dim3 block_size(512, 1, 1);

    for(int i = 0; i < size_input.b; i++){
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input2.scalar_type(), "kernel_Correlation_updateGradSecond", [&] {
            kernel_Correlation_updateGradSecond<scalar_t><<<grid_size, block_size, 0, stream>>>(
                n, stride, i,
                input1.data_ptr<scalar_t>(),
                input2.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                grad2.data_ptr<scalar_t>(),
                size_input,
                size_output,
                size_grad
            );
        });
    }
}

