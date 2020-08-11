//
// Created by developer on 7/19/20.
//

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "source/graphs/graph_utilities.hpp"
#include "source/bfs_sequential.hpp"
#include "source/allocate_for_cuda_bfs.h"
#include "source/cuda_regular_bfs.h"
#include "source/timing.hpp"
#include "source/warp_queue_bfs.h"

#ifdef ENABLE_OPENACC
#include "source/bfs_openacc.hpp"
#endif

CudaTimers TimeDeviceBFS(const Graph& graph, Graph::index_type source, std::vector<int>& distances) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  CudaTimers out{};
  out.upload.Run(AllocateAndCopyFor_device_BFS, graph.NumNodes(), graph.NumEdges() * 2, source,
      graph.matrix.row_indices.data(), graph.matrix.column_indices.data(), &device_edges,
      &device_dests, &device_labels, &device_visited, &current_frontier,
      &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
  out.execution.Run(Launch_device_BFS, graph.NumNodes(), device_edges, device_dests, device_labels, device_visited, current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
  distances.resize(graph.NumNodes());
  out.download.Run(DeallocateFrom_device_BFS, graph.NumNodes(), distances.data(), device_edges, device_dests, device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier, previous_frontier_tail);
  return out;
}

CudaTimers TimeUnifiedBFS(const Graph& graph, Graph::index_type source, std::vector<int>& distances) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  CudaTimers out{};
  out.upload.Run(AllocateAndCopyFor_unified_BFS, graph.NumNodes(), graph.NumEdges() * 2, source,
      graph.matrix.row_indices.data(), graph.matrix.column_indices.data(), &device_edges,
      &device_dests, &device_labels, &device_visited, &current_frontier,
      &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
  out.execution.Run(Launch_unified_BFS, graph.NumNodes(), device_edges, device_dests, device_labels, device_visited, current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
  distances.resize(graph.NumNodes());
  out.download.Run(DeallocateFrom_device_BFS, graph.NumNodes(), distances.data(), device_edges, device_dests, device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier, previous_frontier_tail);
  return out;
}

CudaTimers TimeWarpQueueBFS(const Graph& graph, Graph::index_type source, std::vector<int>& distances) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  CudaTimers out{};
  out.upload.Run(AllocateAndCopyFor_device_BFS, graph.NumNodes(), graph.NumEdges() * 2, source,
      graph.matrix.row_indices.data(), graph.matrix.column_indices.data(), &device_edges,
      &device_dests, &device_labels, &device_visited, &current_frontier,
      &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
  out.execution.Run(LaunchWarpQueueBFS_host, graph.NumNodes(), device_edges, device_dests, device_labels, device_visited, current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
  distances.resize(graph.NumNodes());
  out.download.Run(DeallocateFrom_device_BFS, graph.NumNodes(), distances.data(), device_edges, device_dests, device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier, previous_frontier_tail);
  return out;
}

int main(int argc, char** argv) {
  std::size_t graph_size = 500000;
  float average_diameter = 1.2f;
  float diameter_deviation = 0.5f;
  std::cout << "Generating graph" << std::endl;
  Graph graph = RandomGraphWithDiameter(graph_size, average_diameter, diameter_deviation);
  Graph::index_type source = 0;
  std::cout << "Running Sequential BFS" << std::endl;
  TimingWrapper cpu_timer{};
  std::vector<int> distances = cpu_timer.Run(BFS_sequential, graph, source);
  {
    std::cout << "Writing dot file" << std::endl;
    std::ofstream dot_stream("test.dot");
    graph.ConvertForDot(dot_stream, distances);
    dot_stream.close();
    if (graph_size <= 200) {
      std::cout << graph << std::endl;
    } else {
      std::cout << "Graph is too big to print (size: " << graph_size << ")" << std::endl;
    }
  }

  #ifdef ENABLE_OPENACC
  std::cout << "Running OpenACC BFS" << std::endl;
  TimingWrapper openacc_timer{};
  std::vector<int> openacc_distances = openacc_timer.Run(BFS_OpenACC, graph, source);
  assert(std::equal(distances.begin(), distances.end(), openacc_distances.begin()));
  #endif

  std::cout << "Running Cuda BFS" << std::endl;
  std::vector<int> gpu_bfs_distances(graph.NumNodes());
  CudaTimers regular_bfs_times = TimeDeviceBFS(graph, source, gpu_bfs_distances);
  assert(std::equal(distances.begin(), distances.end(), gpu_bfs_distances.begin()));

  std::cout << "Running Unified Memory BFS" << std::endl;
  std::vector<int> unified_bfs_distances(graph.NumNodes());
  CudaTimers unified_bfs_times = TimeUnifiedBFS(graph, source, unified_bfs_distances);
  assert(std::equal(distances.begin(), distances.end(), unified_bfs_distances.begin()));

  std::vector<int> warpqueue_distances(graph.NumNodes());
  CudaTimers warpqueue_times = TimeDeviceBFS(graph, source, warpqueue_distances);
  assert(std::equal(distances.begin(), distances.end(), warpqueue_distances.begin()));

  std::cout << "Graph of size: " << graph_size << " with average diameter: " << average_diameter << std::endl;
  std::cout << "CPU time: " << cpu_timer << std::endl;
  std::cout << "OpenACC time: " << openacc_timer << std::endl;
  std::cout << "Simple GPU time: " << regular_bfs_times << std::endl;
  std::cout << "Unified GPU time: " << unified_bfs_times << std::endl;
  std::cout << "Unified GPU time: " << warpqueue_times << std::endl;
}