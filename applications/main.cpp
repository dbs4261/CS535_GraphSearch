//
// Created by developer on 7/19/20.
//

#include <algorithm>
#include <cassert>
#include <cmath>
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
#include "source/bfs_parallel_cpu.hpp"

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

struct TimingResults {
  std::size_t graph_size;
  float average_diameter;
  TimingWrapper average_cpu_timer;
  TimingWrapper average_openacc_timer;
  CudaTimers average_regular_bfs_times;
  CudaTimers average_unified_bfs_times;
  CudaTimers average_warpqueue_times;

  static void Header(std::ostream& os) {
    os << "Graph Size, Average Diameter, Average Sequential Cpu Time (ms), "
       << "Average OpenACC Time (ms), Average Cuda Time (ms), "
       << "Average Unified Memory Time (ms), Average Warp Queue Time (ms)\n";
  }

  friend std::ostream& operator<<(std::ostream& os, const TimingResults& results) {
    os << results.graph_size << ", ";
    os << results.average_diameter << ", ";
    os << results.average_cpu_timer << ", ";
    os << results.average_openacc_timer << ", ";
    os << results.average_regular_bfs_times.Total() << ", ";
    os << results.average_unified_bfs_times.Total() << ", ";
    os << results.average_warpqueue_times.Total() << "\n";
    return os;
  }

  friend TimingResults operator+(const TimingResults& a, const TimingResults& b) {
    TimingResults out{};
    out.graph_size = a.graph_size + b.graph_size;
    out.average_diameter = a.average_diameter + b.average_diameter;
    out.average_cpu_timer = a.average_cpu_timer + b.average_cpu_timer;
    out.average_openacc_timer = a.average_openacc_timer + b.average_openacc_timer;
    out.average_regular_bfs_times = a.average_regular_bfs_times + b.average_regular_bfs_times;
    out.average_unified_bfs_times = a.average_unified_bfs_times + b.average_unified_bfs_times;
    out.average_warpqueue_times = a.average_warpqueue_times + b.average_warpqueue_times;
    return out;
  }

  TimingResults& operator+=(const TimingResults& b) {
    this->graph_size += b.graph_size;
    this->average_diameter += b.average_diameter;
    this->average_cpu_timer += b.average_cpu_timer;
    this->average_openacc_timer += b.average_openacc_timer;
    this->average_regular_bfs_times += b.average_regular_bfs_times;
    this->average_unified_bfs_times += b.average_unified_bfs_times;
    this->average_warpqueue_times += b.average_warpqueue_times;
    return *this;
  }

  friend TimingResults operator/(const TimingResults& a, float b) {
    TimingResults out{};
    out.graph_size = a.graph_size / b;
    out.average_diameter = a.average_diameter / b;
    out.average_cpu_timer = a.average_cpu_timer / b;
    out.average_openacc_timer = a.average_openacc_timer / b;
    out.average_regular_bfs_times = a.average_regular_bfs_times / b;
    out.average_unified_bfs_times = a.average_unified_bfs_times / b;
    out.average_warpqueue_times = a.average_warpqueue_times / b;
    return out;
  }

  TimingResults& operator/=(float b) {
    this->graph_size /= b;
    this->average_diameter /= b;
    this->average_cpu_timer /= b;
    this->average_openacc_timer /= b;
    this->average_regular_bfs_times /= b;
    this->average_unified_bfs_times /= b;
    this->average_warpqueue_times /= b;
    return *this;
  }
};

TimingResults RunTimingTest(std::size_t iterations, std::size_t graph_size, float average_diameter, float diameter_deviation, bool write_dot=false) {
  TimingResults results{};
  std::cout << "Generating graph of size: " << graph_size << " with diameter: " << average_diameter << " and deviation: " << diameter_deviation << std::endl;
  Graph graph = RandomGraphWithDiameter(graph_size, average_diameter, diameter_deviation);
  Graph::index_type source = 0;
  results.graph_size = graph.NumNodes();
  results.average_diameter = graph.AverageDiameter();

  std::cout << "Running Sequential BFS" << std::endl;
  std::vector<int> distances;
  for (std::size_t i = 0; i < iterations; i++) {
    TimingWrapper cpu_timer{};
    distances = cpu_timer.Run(BFS_sequential, graph, source);
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    results.average_cpu_timer += cpu_timer;
  }
  results.average_cpu_timer /= static_cast<float>(iterations);

  if (write_dot) {
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
  std::cout << "Running OpenACC BFS" <<std::endl;
  for (std::size_t i = 0; i < iterations; i++) {
    TimingWrapper openacc_timer{};
    std::vector<int> openacc_distances = openacc_timer.Run(BFS_OpenACC, graph, source);
    assert(std::equal(distances.begin(), distances.end(), openacc_distances.begin()));
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    results.average_openacc_timer += openacc_timer;
  }
  results.average_openacc_timer /= static_cast<float>(iterations);
  #endif

  std::cout << "Running Simple CUDA BFS" <<std::endl;
  std::vector<int> gpu_bfs_distances(graph.NumNodes());
  results.average_regular_bfs_times = TimeDeviceBFS(iterations, graph, source, gpu_bfs_distances);
  assert(std::equal(distances.begin(), distances.end(), gpu_bfs_distances.begin()));

  std::cout << "Running Unified Memory BFS" <<std::endl;
  std::vector<int> unified_bfs_distances(graph.NumNodes());
  results.average_unified_bfs_times = TimeUnifiedBFS(iterations, graph, source, unified_bfs_distances);
  assert(std::equal(distances.begin(), distances.end(), unified_bfs_distances.begin()));

  std::cout << "Running Warp Queue BFS" <<std::endl;
  std::vector<int> warpqueue_distances(graph.NumNodes());
  results.average_warpqueue_times = TimeDeviceBFS(iterations, graph, source, warpqueue_distances);
  assert(std::equal(distances.begin(), distances.end(), warpqueue_distances.begin()));

  std::cout << "Test Complete" <<std::endl;
  return results;
}

int main(int argc, char** argv) {
  std::size_t number_of_graphs = 4;
  std::size_t iterations = 25;

  static constexpr std::array<std::pair<float, float>, 4> diameters = {{{1.0f, 0.25f}, {2.0f, 0.5f}, {4.0f, 1.0f}, {8.0f, 2.0f}}};

  std::ofstream csv_file("Timing_results.csv");
  assert(csv_file.is_open());
  TimingResults::Header(csv_file);
  csv_file << std::flush;
  for (std::size_t graph_size = 16; graph_size < 1000000; graph_size *= 4) {
    for (std::size_t d = 0; d < 4; d++) {
      TimingResults average_results{};
      for (std::size_t i = 0; i < number_of_graphs; i++) {
        average_results += RunTimingTest(iterations, graph_size, diameters.at(d).first, diameters.at(d).second);
      }
      average_results /= static_cast<float>(number_of_graphs);
      TimingResults::Header(std::cout);
      std::cout << average_results << std::endl;
      csv_file << average_results << std::flush;
    }
  }
}