#include "library.cuh"

namespace device {

__global__ void Kernel(const float* x, float* y, int n, float a, float b) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  y[idx] = a * x[idx] + b;
}

cudaError_t LaunchKernel(dim3 grid_size, dim3 block_size, const float* x, float* y, int n, float a, float b) {
  Kernel<<<grid_size, block_size>>>(x, y, n, a, b);
  return cudaGetLastError();
}

}
