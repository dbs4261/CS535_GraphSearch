//
// Created by developer on 7/19/20.
//

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <thread>
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

CudaTimers TimeDeviceBFS(std::size_t iterations, const Graph& graph, Graph::index_type source, std::vector<int>& distances) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  CudaTimers average{};
  for (std::size_t i = 0; i < iterations; i++) {
    CudaTimers timer{};
    timer.upload.Run(AllocateAndCopyFor_device_BFS, graph.NumNodes(), graph.NumEdges() * 2, source,
        graph.matrix.row_indices.data(), graph.matrix.column_indices.data(), &device_edges,
        &device_dests, &device_labels, &device_visited, &current_frontier,
        &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
    timer.execution.Run(Launch_device_BFS, graph.NumNodes(), device_edges, device_dests, device_labels, device_visited,
        current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
    distances.resize(graph.NumNodes());
    timer.download.Run(DeallocateFrom_device_BFS, graph.NumNodes(), distances.data(), device_edges, device_dests,
        device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier,
        previous_frontier_tail);
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    average += timer;
  }
  average /= static_cast<float>(iterations);
  return average;
}

CudaTimers TimeUnifiedBFS(std::size_t iterations, const Graph& graph, Graph::index_type source, std::vector<int>& distances) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  CudaTimers average{};
  for (std::size_t i = 0; i < iterations; i++) {
    CudaTimers timer{};
    timer.upload.Run(AllocateAndCopyFor_unified_BFS, graph.NumNodes(), graph.NumEdges() * 2, source,
        graph.matrix.row_indices.data(), graph.matrix.column_indices.data(), &device_edges,
        &device_dests, &device_labels, &device_visited, &current_frontier,
        &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
    timer.execution.Run(Launch_unified_BFS, graph.NumNodes(), device_edges, device_dests, device_labels, device_visited,
        current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
    distances.resize(graph.NumNodes());
    timer.download.Run(DeallocateFrom_device_BFS, graph.NumNodes(), distances.data(), device_edges, device_dests,
        device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier,
        previous_frontier_tail);
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    average += timer;
  }
  average /= static_cast<float>(iterations);
  return average;
}

CudaTimers TimeWarpQueueBFS(std::size_t iterations, const Graph& graph, Graph::index_type source, std::vector<int>& distances) {
  int* device_edges = nullptr;
  int* device_dests = nullptr;
  int* device_labels = nullptr;
  int* device_visited = nullptr;
  int* current_frontier = nullptr;
  int* current_frontier_tail = nullptr;
  int* previous_frontier = nullptr;
  int* previous_frontier_tail = nullptr;
  CudaTimers average{};
  for (std::size_t i = 0; i < iterations; i++) {
    CudaTimers timer{};
    timer.upload.Run(AllocateAndCopyFor_device_BFS, graph.NumNodes(), graph.NumEdges() * 2, source,
        graph.matrix.row_indices.data(), graph.matrix.column_indices.data(), &device_edges,
        &device_dests, &device_labels, &device_visited, &current_frontier,
        &current_frontier_tail, &previous_frontier, &previous_frontier_tail);
    timer.execution.Run(LaunchWarpQueueBFS_host, graph.NumNodes(), device_edges, device_dests, device_labels, device_visited,
        current_frontier_tail, current_frontier, previous_frontier_tail, previous_frontier);
    distances.resize(graph.NumNodes());
    timer.download.Run(DeallocateFrom_device_BFS, graph.NumNodes(), distances.data(), device_edges, device_dests,
        device_labels, device_visited, current_frontier, current_frontier_tail, previous_frontier,
        previous_frontier_tail);
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    average += timer;
  }
  average /= static_cast<float>(iterations);
  return average;
}

int main(int argc, char** argv) {
  std::size_t graph_size = 100;
  std::size_t iterations = 25;
  float average_diameter = 1.2f;
  float diameter_deviation = 0.5f;
  std::cout << "Generating graph" << std::endl;
  Graph graph = RandomGraphWithDiameter(graph_size, average_diameter, diameter_deviation);
  Graph::index_type source = 0;

  std::cout << "Running Sequential BFS" << std::endl;
  TimingWrapper average_cpu_timer{};
  std::vector<int> distances;
  for (std::size_t i = 0; i < iterations; i++) {
    TimingWrapper cpu_timer{};
    distances = cpu_timer.Run(BFS_sequential, graph, source);
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    average_cpu_timer += cpu_timer;
  }
  average_cpu_timer /= static_cast<float>(iterations);

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
  TimingWrapper average_openacc_timer{};
  for (std::size_t i = 0; i < iterations; i++) {
    TimingWrapper openacc_timer{};
    std::vector<int> openacc_distances = openacc_timer.Run(BFS_OpenACC, graph, source);
    assert(std::equal(distances.begin(), distances.end(), openacc_distances.begin()));
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    average_openacc_timer += openacc_timer;
  }
  average_openacc_timer /= static_cast<float>(iterations);
  #endif

  std::cout << "Running Cuda BFS" << std::endl;
  std::vector<int> gpu_bfs_distances(graph.NumNodes());
  CudaTimers average_regular_bfs_times = TimeDeviceBFS(iterations, graph, source, gpu_bfs_distances);
  assert(std::equal(distances.begin(), distances.end(), gpu_bfs_distances.begin()));

  std::cout << "Running Unified Memory BFS" << std::endl;
  std::vector<int> unified_bfs_distances(graph.NumNodes());
  CudaTimers average_unified_bfs_times = TimeUnifiedBFS(iterations, graph, source, unified_bfs_distances);
  assert(std::equal(distances.begin(), distances.end(), unified_bfs_distances.begin()));

  std::vector<int> warpqueue_distances(graph.NumNodes());
  CudaTimers average_warpqueue_times = TimeDeviceBFS(iterations, graph, source, warpqueue_distances);
  assert(std::equal(distances.begin(), distances.end(), warpqueue_distances.begin()));

  std::cout << "Graph of size: " << graph_size << " with average diameter: " << average_diameter << std::endl;
  std::cout << "CPU time: " << average_cpu_timer << std::endl;
  std::cout << "OpenACC time: " << average_openacc_timer << std::endl;
  std::cout << "Simple GPU time: " << average_regular_bfs_times << std::endl;
  std::cout << "Unified GPU time: " << average_unified_bfs_times << std::endl;
  std::cout << "Warp Queue GPU time: " << average_warpqueue_times << std::endl;
}