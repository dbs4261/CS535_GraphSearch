//
// Created by developer on 7/19/20.
//

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "source/error_checker.h"
#include "source/graphs/graph_utilities.hpp"
#include "source/bfs_sequential.hpp"
#ifdef ENABLE_OPENACC
#include "source/bfs_openacc.hpp"
#endif

int main(int argc, char** argv) {
  Graph graph = RandomGraphWithDiameter(20, 1.2f, 0.5f);
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
}