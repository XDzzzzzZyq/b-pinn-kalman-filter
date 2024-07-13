#include <torch/types.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
static __global__ void kernel_Correlation_rearrange(
		const int n,
		const int stride,
		const scalar_t* input,
		scalar_t* output,
		const long long size_input[4],
		const long long size_output[4]
) {
	  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	  if (index >= n) {
	    return;
	  }
	  int sample = blockIdx.z;
	  int num_channel = blockIdx.y;
	  float dblValue = input[(((sample * size_input[1]) + num_channel) * size_input[2] * size_input[3]) + index];
	  __syncthreads();
	  int padded_y = (index / size_input[3]) + 3*stride;
	  int padded_x = (index % size_input[3]) + 3*stride;
	  int rearrange = ((size_input[3] + 6*stride) * padded_y) + padded_x;
	  output[(((sample * size_output[1] * size_output[2]) + rearrange) * size_input[1]) + num_channel] = dblValue;
	}

template <typename scalar_t>
static __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const int stride,
	  const scalar_t* rbot0,
	  const scalar_t* rbot1,
	  scalar_t* top,
	  const long long size_rbot0[4],
	  const long long size_top[4]
) {
	  extern __shared__ char patch_data_char[];

	  float *patch_data = (float *)patch_data_char;

	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = (blockIdx.x + 3) * stride;
	  int y1 = (blockIdx.y + 3) * stride;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;

	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * size_rbot0[3];
	      for (int ch = ch_off; ch < size_rbot0[3]; ch += 32) { // CHANNELS
	        int idx1 = ((item * size_rbot0[1] + y1+j) * size_rbot0[2] + x1+i) * size_rbot0[3] + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }

	  __syncthreads();

	  __shared__ float sum[32];

	  // Compute correlation
	  for (int top_channel = 0; top_channel < size_top[1]; top_channel++) {
	    sum[ch_off] = 0;

	    int s2o = (top_channel % 7 - 3) * stride;
	    int s2p = (top_channel / 7 - 3) * stride;

	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * size_rbot0[3];
	        for (int ch = ch_off; ch < size_rbot0[3]; ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;

	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * size_rbot0[1] + y2+j) * size_rbot0[2] + x2+i) * size_rbot0[3] + ch;

	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }

	    __syncthreads();

	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = size_rbot0[3];
	      const int index = ((top_channel*size_top[2] + blockIdx.y)*size_top[3])+blockIdx.x;
	      top[index + item*size_top[1]*size_top[2]*size_top[3]] = total_sum / (float)sumelems;
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
	  const long long size_rbot0[4],
	  const long long size_out[4],
	  const long long size_first[4]
) {
	for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
	  int n = index % size_first[1]; // channels
	  int l = (index / size_first[1]) % size_first[3] + 3*stride; // w-pos
	  int m = (index / size_first[1] / size_first[3]) % size_first[2] + 3*stride; // h-pos

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

	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=size_out[3]-1) && (ymin<=size_out[2]-1)) {
	    xmin = fmaxf(0,xmin);
	    xmax = fminf(size_out[3]-1,xmax);

	    ymin = fmaxf(0,ymin);
	    ymax = fminf(size_out[2]-1,ymax);

	    for (int p = -3; p <= 3; p++) {
	      for (int o = -3; o <= 3; o++) {
	        // Get rbot1 data:
	        int s2o = stride * o;
	        int s2p = stride * p;
	        int idxbot1 = ((sample * size_rbot0[1] + (m+s2p)) * size_rbot0[2] + (l+s2o)) * size_rbot0[3] + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (sample * size_out[1] + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * size_out[2] + y) * size_out[3] + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = size_first[1];
	  const int bot0index = ((n * size_first[2]) + (m-3*stride)) * size_first[3] + (l-3*stride);
	  gradFirst[bot0index + sample * size_first[1] * size_first[2] * size_first[3]] = sum / (float)sumelems;
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
	  const long long size_rbot0[4],
	  const long long size_out[4],
	  const long long size_second[4]
) {
	  for (int index = (blockIdx.x * blockDim.x) + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
	  int n = index % size_second[1]; // channels
	  int l = (index / size_second[1]) % size_second[3] + 3*stride; // w-pos
	  int m = (index / size_second[1] / size_second[3]) % size_second[2] + 3*stride; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = stride * round_off;

	  float sum = 0;
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

	      if (xmax>=0 && ymax>=0 && (xmin<=size_out[3]-1) && (ymin<=size_out[2]-1)) {
	        xmin = fmaxf(0,xmin);
	        xmax = fminf(size_out[3]-1,xmax);

	        ymin = fmaxf(0,ymin);
	        ymax = fminf(size_out[2]-1,ymax);

	        // Get rbot0 data:
	        int idxbot0 = ((sample * size_rbot0[1] + (m-s2p)) * size_rbot0[2] + (l-s2o)) * size_rbot0[3] + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+3) * 7 + (o+3); // index[o,p]
	        int idxopoffset = (sample * size_out[1] + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * size_out[2] + y) * size_out[3] + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = size_second[1];
	  const int bot1index = ((n * size_second[2]) + (m-3*stride)) * size_second[3] + (l-3*stride);
	  gradSecond[bot1index + sample * size_second[1] * size_second[2] * size_second[3]] = sum / (float)sumelems;
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

    int n = input.size(2) * input.size(3);
    long long size_input[4] = {input.size(0), input.size(1), input.size(2), input.size(3)};
    long long size_output[4] = {output.size(0), output.size(1), output.size(2), output.size(3)};

    dim3 grid_size((n+16-1)/16, input.size(1), input.size(0));
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

    int n = output.size(1) * output.size(2) * output.size(3);
    long long size_input[4] = {input1.size(0), input1.size(1), input1.size(2), input1.size(3)};
    long long size_output[4] = {output.size(0), output.size(1), output.size(2), output.size(3)};

    dim3 grid_size(output.size(3), output.size(2), output.size(0));
    dim3 block_size(32, 1, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "kernel_Correlation_updateOutput", [&] {
        kernel_Correlation_updateOutput<scalar_t><<<grid_size, block_size, 0, stream>>>(
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

    int n = input1.size(1) * input1.size(2) * input1.size(3);
    long long size_input[4] = {input1.size(0), input1.size(1), input1.size(2), input1.size(3)};
    long long size_output[4] = {output.size(0), output.size(1), output.size(2), output.size(3)};
    long long size_grad[4] = {grad1.size(0), grad1.size(1), grad1.size(2), grad1.size(3)};

    dim3 grid_size((n + 512 - 1)/512, 1, 1);
    dim3 block_size(512, 1, 1);

    for(int i = 0; i < input1.size(0); i++){
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

    int n = input2.size(1) * input2.size(2) * input2.size(3);
    long long size_input[4] = {input2.size(0), input2.size(1), input2.size(2), input2.size(3)};
    long long size_output[4] = {output.size(0), output.size(1), output.size(2), output.size(3)};
    long long size_grad[4] = {grad2.size(0), grad2.size(1), grad2.size(2), grad2.size(3)};

    dim3 grid_size((n + 512 - 1)/512, 1, 1);
    dim3 block_size(512, 1, 1);

    for(int i = 0; i < input2.size(0); i++){
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

