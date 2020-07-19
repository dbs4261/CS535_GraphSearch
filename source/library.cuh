#ifndef CS535_GRAPHSEARCH_LIBRARY_CUH
#define CS535_GRAPHSEARCH_LIBRARY_CUH

#include <cuda_runtime_api.h>

namespace device {

__global__ void Kernel(const float* x, float* y, int n, float a, float b);

cudaError_t LaunchKernel(dim3 grid_size, dim3 block_size, const float* x, float* y, int n, float a, float b);

}

#endif //CS535_GRAPHSEARCH_LIBRARY_CUH
