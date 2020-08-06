//
// Created by developer on 7/19/20.
//

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "source/library.cuh"
#include "source/error_checker.h"
#include "source/graphs/graph_utilities.hpp"
#include "source/bfs_sequential.hpp"
#ifdef ENABLE_OPENACC
#include "source/bfs_openacc.hpp"
#endif
#ifdef ENABLE_PARALLEL_STL
#include "source/bfs_parallel_stl.hpp"
#endif

int main(int argc, char** argv) {
  Graph graph = RandomGraphWithDiameter(150, 1.2f, 0.5f);
  std::vector<int> distances = BFS_sequential(graph, 0);
  {
    std::ofstream dot_stream("test.dot");
    graph.ConvertForDot(dot_stream, distances);
    dot_stream.close();
    std::cout << graph << std::endl;
  }

  #ifdef ENABLE_OPENACC
  std::vector<int> openacc_distances = BFS_OpenACC(graph, 0);
  assert(std::equal(distances.begin(), distances.end(), openacc_distances.begin()));
  std::cout << "Finished OpenACC BFS" << std::endl;
  #endif

  #ifdef ENABLE_PARALLEL_STL
  std::vector<int> parallel_stl_distances = BFS_ParallelSTL(graph, 0);
  assert(std::equal(distances.begin(), distances.end(), openacc_distances.begin()));
  std::cout << "Finished ParallelSTL BFS" << std::endl;
  #endif

  int n = 1000;
  std::vector<float> x_host(n);
  std::fill(x_host.begin(), x_host.end(), 1.0f);
  float* x_cu;
  float* y_cu;
  CudaCatchError(cudaMalloc(reinterpret_cast<void**>(&x_cu), sizeof(float) * n));
  CudaCatchError(cudaMemcpy(x_cu, x_host.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
  CudaCatchError(cudaMalloc(reinterpret_cast<void**>(&y_cu), sizeof(float) * n));
  dim3 block_size(512, 1, 1);
  dim3 grid_size(1 + (n - 1) / block_size.x, 1, 1);
  CudaCatchError(device::LaunchKernel(grid_size, block_size, x_cu, y_cu, n, 2.0f, 4.0f));
  std::vector<float> y_host(n);
  CudaCatchError(cudaMemcpy(y_host.data(), y_cu, sizeof(float) * n, cudaMemcpyDeviceToHost));
  CudaCatchError(cudaFree(x_cu));
  CudaCatchError(cudaFree(y_cu));
  if (std::all_of(y_host.begin(), y_host.end(), [](float val){return std::abs(val - 6.0f) < 0.001;})) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}