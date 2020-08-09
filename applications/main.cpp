//
// Created by developer on 7/19/20.
//

#include <algorithm>
#include <fstream>
#include <vector>

#include <cuda_runtime_api.h>

#include "source/error_checker.h"
#include "source/graphs/graph_utilities.hpp"
#include "source/bfs_sequential.hpp"

int main(int argc, char** argv) {
  Graph graph = RandomGraphWithDiameter(150, 1.2f, 0.5f);
  std::vector<int> distances = BFS_sequential(graph, 0);
  {
    std::ofstream dot_stream("test.dot");
    graph.ConvertForDot(dot_stream, distances);
    dot_stream.close();
    std::cout << graph << std::endl;
  }
}